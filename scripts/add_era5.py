#!/usr/bin/env python
"""ERA-5 co-location driver: batch features by hour, write per-orbit ERA-5 tables.

Reads the existing PF feature table for a mission + date window, groups features
by their nearest ERA-5 hour, co-locates ARCO ERA-5 (centroid value + box stats +
10m->1/3/6 km shear) and writes one ERA-5 Parquet per orbit under ``{root}/era5``
(idempotent overwrite). ERA-5 is opt-in and network-dependent; this driver does
NOT touch the per-orbit radar/imager pipeline.

Usage
-----
    python scripts/add_era5.py TRMM --start 1997-12-30 --end 1997-12-31 \
        --root /data/scratch/a/snesbitt/pf_db --workers 4

A ``spawn`` Pool is used so xarray / zarr / gcsfs initialize cleanly per process.
Each hour-batch is an independent task; results are regrouped by orbit and
written. Use ``--workers 1`` (or ``--serial``) for a single-process run.
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import os

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
from rich.console import Console
from rich.progress import Progress

from pf.config import PF_ROOT
from pf.era5 import era5_for_features, write_era5

console = Console()


def _read_features(root: str, mission: str, start: str, end: str) -> pd.DataFrame:
    """Read feature meta for the orbits whose START falls in ``[start, end)``.

    Reads over a WIDENED window ``[start - 2h, end + 2h)`` so that full orbits
    straddling the ``[start, end)`` boundaries are captured (a radar orbit is
    ~92 min and spans 2-3 ERA-5 hours). Each orbit's START is ``min`` of its
    feature ``time``; only orbits whose START lies in the ORIGINAL
    ``[start, end)`` window are kept, and the FULL feature set (all rows,
    including those past ``end``) of those orbits is returned. This attributes
    every orbit to exactly one day-task -- the one whose ``[start, end)``
    contains the orbit start -- so adjacent tasks never touch the same orbit.
    """
    mission_key = str(mission).upper()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    wide_start = start_ts - pd.Timedelta(hours=2)
    wide_end = end_ts + pd.Timedelta(hours=2)

    dataset = pads.dataset(f"{root}/features", partitioning="hive")
    cols = ["feature_id", "mission", "orbit", "time", "centroid_lat", "centroid_lon"]
    filt = (
        (pc.field("mission") == mission_key)
        & (pc.field("time") >= pa.scalar(wide_start.to_pydatetime(), type=pa.timestamp("us")))
        & (pc.field("time") < pa.scalar(wide_end.to_pydatetime(), type=pa.timestamp("us")))
    )
    df = dataset.to_table(columns=cols, filter=filt).to_pandas()
    if len(df) == 0:
        return df

    df["time"] = pd.to_datetime(df["time"])
    # Orbit START = earliest feature time; keep orbits whose start is in [start, end).
    orbit_start = df.groupby("orbit")["time"].transform("min")
    keep = (orbit_start >= start_ts) & (orbit_start < end_ts)
    return df.loc[keep].reset_index(drop=True)


def _worker(records: list[dict]) -> pd.DataFrame:
    """Co-locate one hour-batch and RETURN its ERA-5 rows (no writing here).

    ``records`` all belong to the same nearest ERA-5 hour (so the hour's global
    field is fetched exactly once) but may span multiple orbits. The returned
    DataFrame is collected in the main process, concatenated across all
    hour-batches, and written per-orbit there -- so each orbit file gets ALL its
    features and no two workers ever race on the same orbit path.
    """
    df = pd.DataFrame(records)
    return era5_for_features(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mission", help="Mission name, e.g. TRMM or GPM")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    ap.add_argument("--root", default=PF_ROOT, help="Dataset root")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, os.cpu_count() or 4))),
        help="Number of parallel hour-batch workers",
    )
    ap.add_argument("--serial", action="store_true", help="Force single-process run")
    args = ap.parse_args()

    mission_key = str(args.mission).upper()
    df = _read_features(args.root, mission_key, args.start, args.end)
    if len(df) == 0:
        console.print(
            f"[yellow]No features for {mission_key} in {args.start}..{args.end}.[/yellow]"
        )
        return

    df["time"] = pd.to_datetime(df["time"])
    df["_hour"] = df["time"].dt.round("h")

    by_hour: dict = collections.defaultdict(list)
    for rec in df.drop(columns=["_hour"]).assign(_hour=df["_hour"]).to_dict(orient="records"):
        by_hour[rec.pop("_hour")].append(rec)

    payloads = list(by_hour.values())
    console.print(
        f"[bold]{len(df)}[/bold] features in [bold]{len(payloads)}[/bold] hour-batches "
        f"({mission_key} {args.start}..{args.end})."
    )

    workers = 1 if args.serial else max(1, args.workers)

    # Parallelize the per-hour ERA-5 COMPUTE/FETCH only (each hour fetched once),
    # collecting the returned rows. Writing happens serially in the main process.
    frames: list[pd.DataFrame] = []
    if workers == 1:
        with Progress(console=console) as progress:
            task = progress.add_task("ERA-5 hour-batches", total=len(payloads))
            for payload in payloads:
                frames.append(_worker(payload))
                progress.advance(task)
    else:
        ctx = mp.get_context("spawn")
        with Progress(console=console) as progress, ctx.Pool(workers) as pool:
            task = progress.add_task("ERA-5 hour-batches", total=len(payloads))
            for frame in pool.imap_unordered(_worker, payloads):
                frames.append(frame)
                progress.advance(task)

    frames = [f for f in frames if f is not None and len(f) > 0]
    if not frames:
        console.print("[yellow]No ERA-5 rows produced.[/yellow]")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Each orbit's features may have come from several hour-batches; group the
    # combined frame so every orbit file is written EXACTLY ONCE with ALL its
    # features. Serial writes in the main process -> no concurrent writers.
    n_written = 0
    n_orbits = 0
    with Progress(console=console) as progress:
        groups = list(combined.groupby("orbit"))
        task = progress.add_task("Writing per-orbit ERA-5", total=len(groups))
        for _orbit, grp in groups:
            write_era5(grp, mission_key, root=args.root)
            n_written += len(grp)
            n_orbits += 1
            progress.advance(task)

    console.print(
        f"[green]Done. {n_written} ERA-5 rows for {n_orbits} orbits written "
        f"under {args.root}/era5.[/green]"
    )


if __name__ == "__main__":
    main()
