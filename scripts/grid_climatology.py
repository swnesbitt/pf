#!/usr/bin/env python
"""Stage 2: swath-gridded rain climatology — reduce months-of-year → zarr.

Reduces the per-(year,month) sparse grid tables written by
``scripts/grid_month.py`` (``{root}/grid/mission=/year=/month=/{rain,views}.parquet``)
across all years into a **month-of-year (1-12) × UTC-hour (0-23)** climatology on
a 0.05 deg grid, stratified by feature **size** / **20-dBZ echo-top** / per-pixel
**rain type**. Writes:

* a per-mission **zarr store** (marginal breakdowns) to MinIO
  (``s3://$SPACEBORNE_MINIO_BUCKET/pf_grid_{MISSION}.zarr``) on a SHARED ±68° lat
  grid so missions are cell-aligned for comparison/combination, and
* the lossless **sparse joint** Parquet (full size×echotop×raintype crossproduct,
  hour-resolved) at ``{out}/{MISSION}_joint.parquet``.

Each of the 12 months-of-year is reduced independently (sum all ``year=*``), so
they run in a ``spawn`` Pool — one DuckDB connection per month. See :mod:`pf.grid`
(``reduce_grid_rain_month`` / ``write_grid_zarr``).

Usage
-----
    source /data/keeling/a/snesbitt/.spaceborne_minio.env
    python scripts/grid_climatology.py GPM --root /data/scratch/a/snesbitt/pf_db \
        --workers 12
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from pf import grid, grid_swath
from pf.config import PF_ROOT

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mission", help="Mission name, e.g. TRMM or GPM")
    ap.add_argument("--root", default=PF_ROOT, help="Dataset root (holds grid/)")
    ap.add_argument("--out", default=None,
                    help="Local dir for the sparse joint (default {root}/grid)")
    ap.add_argument("--store", default=None,
                    help="Zarr store URL/path. Default s3://$SPACEBORNE_MINIO_BUCKET/"
                         "pf_grid_{MISSION}.zarr (MinIO creds from $SPACEBORNE_MINIO_*)")
    ap.add_argument("--lat-clip", default="-68,68",
                    help="Shared lat band lo,hi in deg for ALL missions (cell-aligned "
                         "grids); 'mission' = per-mission coverage band")
    ap.add_argument("--months", default=None,
                    help="Comma list of months-of-year to run (default 1..12)")
    ap.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", min(12, os.cpu_count() or 12))),
        help="Parallel month workers (Pool size)")
    ap.add_argument("--serial", action="store_true", help="Force single-process run")
    ap.add_argument("--mem", default=os.environ.get("PF_DUCKDB_MEM", "12GB"),
                    help="DuckDB memory_limit per worker")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("PF_DUCKDB_THREADS", "2")),
                    help="DuckDB threads per worker")
    args = ap.parse_args()

    mission = str(args.mission).upper()
    months = ([int(m) for m in args.months.split(",")] if args.months
              else list(range(1, 13)))
    out_dir = Path(args.out) if args.out else Path(args.root) / "grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lat_clip.strip().lower() == "mission":
        lat_clip = None
    else:
        lo_d, hi_d = (float(x) for x in args.lat_clip.split(","))
        lat_clip = (lo_d, hi_d)

    # ONE configured out-of-core DuckDB connection; months reduced serially,
    # just-in-time inside the icechunk writer (peak RAM = one month). DuckDB uses
    # all the node's cores per month and spills to node-local $TMPDIR.
    con = grid.duck_connect(mem=args.mem, threads=args.threads)

    def reduce_month(m):
        rain = grid.reduce_grid_rain_month(con, mission, m, args.root)
        views = grid.reduce_grid_views_month(con, mission, m, args.root)
        metrics = grid.reduce_grid_metrics_month(con, mission, m, args.root)
        return rain, views, metrics

    bucket = os.environ.get("SPACEBORNE_MINIO_BUCKET", "spaceborne-grids")
    prefix = args.store or f"pf_grid_{mission}"   # icechunk repo prefix in the bucket
    storage_options = {
        "key": os.environ["SPACEBORNE_MINIO_ACCESS"],
        "secret": os.environ["SPACEBORNE_MINIO_SECRET"],
        "client_kwargs": {"endpoint_url": os.environ["SPACEBORNE_MINIO_ENDPOINT"]},
    }
    joint_path = str(out_dir / f"{mission}_joint.parquet")
    clip_txt = "per-mission" if lat_clip is None else f"{lat_clip[0]:g}..{lat_clip[1]:g} deg"
    console.print(f"[bold]{mission}[/bold]: streaming marginal icechunk (12×24 slabs, "
                  f"lat {clip_txt}, DuckDB mem={args.mem} threads={args.threads}) "
                  f"-> [cyan]s3://{bucket}/{prefix}[/cyan]")
    store = grid.write_grid_zarr(reduce_month, mission,
                                 bucket=bucket, prefix=prefix,
                                 mode="marginal", lat_clip=lat_clip,
                                 storage_options=storage_options,
                                 joint_out=joint_path,
                                 metric_cols=grid_swath.METRIC_COLS,
                                 log=lambda msg: console.print(f"[dim]{msg}[/dim]"))
    con.close()
    console.print(f"[green]Done. icechunk {store}, joint {joint_path}.[/green]")


if __name__ == "__main__":
    main()
