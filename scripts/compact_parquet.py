#!/usr/bin/env python
"""Compact the per-orbit PF Parquet tree into per-month files for fast reads.

The build pipeline writes **one Parquet file per orbit per table** (lock-free,
embarrassingly-parallel writes — see ``catalog.py``). That is optimal for
*writing* but pathological for *reading*: ~157k tiny files per table (avg
170 KB features / 500 KB pixels) means every query pays a per-file open +
footer-decode cost across a third of a million files. This is the classic
"write small, compact for read" pattern.

This script reads each ``mission=/year=/month=`` partition's ``orbit=*.parquet``
files and rewrites them as a SINGLE file per month, sorted by ``feature_id`` so
the feature<->pixel join and predicate pushdown get locality. The hive layout
(``mission=/year=/month=``) is preserved exactly, so the compacted tree is a
drop-in replacement at coarser granularity — every query that worked on the
per-orbit tree works unchanged. ``year``/``month`` live only in the path (not as
columns), so keeping those path keys is mandatory; ``mission``/``orbit`` are
real columns, so per-month files lose no orbit identity.

Safety / correctness
--------------------
* **Non-destructive by default.** Writes to a SEPARATE ``--out-root`` (default
  ``{root}_compact``). The per-orbit tree — the source of truth for cheap
  idempotent single-orbit re-runs — is left untouched. Validate, then drop the
  per-orbit tree yourself if you want the space back.
* **Row-count validated.** Every month asserts ``COUNT(*)`` over the per-orbit
  inputs equals ``COUNT(*)`` of the compacted output; any mismatch is reported
  and makes the run exit non-zero. Nothing is deleted.
* **Resumable.** A month whose output already exists with a matching row count
  is skipped, so re-running after an interruption is cheap.
* **Run AFTER the build is finished.** ``add_era5.py`` rewrites the per-orbit
  *feature* files in place; compacting mid-build would freeze stale rows. Run
  only once the radar + ERA-5 arrays and the failed-orbit retry sweep are done.

Parallelism
-----------
Months are independent units of work (mirrors the orbit-level parallelism of
the rest of the pipeline). A ``spawn`` Pool fans out over (table, month)
partitions; each worker uses its own single-threaded, memory-bounded DuckDB
connection so N workers don't oversubscribe cores or RAM.

Usage
-----
    python scripts/compact_parquet.py \
        --root /data/scratch/a/snesbitt/pf_db \
        --out-root /data/scratch/a/snesbitt/pf_db_compact \
        --workers 32
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

DEFAULT_ROOT = os.environ.get("PF_ROOT", "/data/scratch/a/snesbitt/pf_db")

# Per-table read-locality sort key. features/pixels share feature_id (join +
# pushdown locality); the sparse views table has no feature_id, so it sorts by
# its spatial bin keys instead. A table absent here falls back to --order-by.
TABLE_ORDER_BY = {
    "features": "feature_id",
    "pixels": "feature_id",
    "views": "lat_bin, lon_bin",
}


# ---------------------------------------------------------------------------
# Work-list enumeration
# ---------------------------------------------------------------------------
def enumerate_partitions(root: Path, tables: list[str]) -> list[dict]:
    """Find every ``mission=/year=/month=`` partition holding orbit files.

    Returns a list of dicts (picklable → spawn-safe) describing one month of
    one table each.
    """
    parts: list[dict] = []
    for table in tables:
        tdir = root / table
        if not tdir.is_dir():
            continue
        for mdir in sorted(tdir.glob("mission=*")):
            for ydir in sorted(mdir.glob("year=*")):
                for monthdir in sorted(ydir.glob("month=*")):
                    files = list(monthdir.glob("orbit=*.parquet"))
                    if not files:
                        continue
                    parts.append(
                        {
                            "table": table,
                            "mission": mdir.name.split("=", 1)[1],
                            "year": ydir.name.split("=", 1)[1],
                            "month": monthdir.name.split("=", 1)[1],
                            "src_dir": str(monthdir),
                            "n_files": len(files),
                            "rel": f"{table}/{mdir.name}/{ydir.name}/{monthdir.name}",
                        }
                    )
    return parts


# ---------------------------------------------------------------------------
# Worker (top-level → picklable, spawn-safe)
# ---------------------------------------------------------------------------
def _compact_one(job: dict) -> dict:
    """Compact one month of one table into a single sorted Parquet file."""
    import duckdb

    table = job["table"]
    src_glob = f"{job['src_dir']}/orbit=*.parquet"
    out_dir = Path(job["out_root"]) / table / (
        f"mission={job['mission']}"
    ) / f"year={job['year']}" / f"month={job['month']}"
    out_file = out_dir / (
        f"pf_{table}_{job['mission']}_{job['year']}{job['month']}.parquet"
    )

    res = {**job, "out_file": str(out_file), "src_rows": 0, "out_rows": 0,
           "out_bytes": 0, "status": "ok"}

    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute(f"SET memory_limit='{job['mem_limit']}'")

        src_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?, union_by_name=true)", [src_glob]
        ).fetchone()[0]
        res["src_rows"] = src_rows

        # Resumable skip: existing output with matching row count → done.
        if out_file.exists() and not job["overwrite"]:
            try:
                existing = con.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)", [str(out_file)]
                ).fetchone()[0]
                if existing == src_rows:
                    res.update(status="skip", out_rows=existing,
                               out_bytes=out_file.stat().st_size)
                    return res
            except Exception:
                pass  # unreadable/partial → recompact below

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_file.with_suffix(".parquet.tmp")
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet(?, union_by_name=true)
                ORDER BY {job['order_by']}
            ) TO '{tmp}'
            (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {job['row_group_size']})
            """,
            [src_glob],
        )
        out_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(tmp)]
        ).fetchone()[0]
        res["out_rows"] = out_rows

        if out_rows != src_rows:
            res["status"] = "MISMATCH"
            tmp.unlink(missing_ok=True)
            return res

        os.replace(tmp, out_file)  # atomic publish
        res["out_bytes"] = out_file.stat().st_size
        return res
    except Exception as exc:  # noqa: BLE001 — one bad month must not kill the batch
        res["status"] = f"ERROR: {type(exc).__name__}: {exc}"
        return res
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"Per-orbit source tree (default {DEFAULT_ROOT})")
    ap.add_argument("--out-root", default=None,
                    help="Compacted output tree (default {root}_compact)")
    ap.add_argument("--tables", default="features,pixels",
                    help="Comma list of tables to compact (default features,pixels)")
    ap.add_argument(
        "--workers", type=int,
        default=min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))),
        help="Parallel month-compaction workers "
             "(default min(16, SLURM_CPUS_PER_TASK or cpu_count))",
    )
    ap.add_argument("--order-by", default=None,
                    help="ORDER BY column override (default: per-table — "
                         "feature_id for features/pixels, lat_bin,lon_bin for views)")
    ap.add_argument("--row-group-size", type=int, default=1_000_000,
                    help="Parquet row-group size in rows (default 1,000,000)")
    ap.add_argument("--mem-limit", default="4GB",
                    help="Per-worker DuckDB memory limit (default 4GB)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompact even if a matching output already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Enumerate partitions and print the plan; do not write")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else Path(f"{root}_compact")
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    console = Console()

    parts = enumerate_partitions(root, tables)
    if not parts:
        console.print(f"[red]No orbit partitions found under {root} for {tables}[/red]")
        return 2

    tot_files = sum(p["n_files"] for p in parts)
    by_table = {t: sum(1 for p in parts if p["table"] == t) for t in tables}
    console.print(
        f"[bold]Compaction plan[/bold]: {len(parts)} month-partitions "
        f"({', '.join(f'{t}={n}' for t, n in by_table.items())}) "
        f"from {tot_files:,} orbit files\n"
        f"  source : {root}\n"
        f"  output : {out_root}\n"
        f"  sort   : {args.order_by or 'per-table'}   workers: {args.workers}   "
        f"row_group: {args.row_group_size:,}"
    )
    if args.dry_run:
        for p in parts[:3] + parts[-3:]:
            console.print(f"  {p['rel']}  ({p['n_files']} files)")
        return 0

    for p in parts:
        p["out_root"] = str(out_root)
        p["order_by"] = args.order_by or TABLE_ORDER_BY.get(p["table"], "feature_id")
        p["row_group_size"] = args.row_group_size
        p["mem_limit"] = args.mem_limit
        p["overwrite"] = args.overwrite

    t0 = time.time()
    results: list[dict] = []
    cols = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]
    ctx = mp.get_context("spawn")
    with Progress(*cols, console=console) as progress, ctx.Pool(args.workers) as pool:
        task = progress.add_task("compacting months", total=len(parts))
        for res in pool.imap_unordered(_compact_one, parts):
            results.append(res)
            progress.update(task, advance=1)
            if res["status"] not in ("ok", "skip"):
                progress.console.print(
                    f"  [red]{res['status']}[/red]  {res['rel']}  "
                    f"(src={res['src_rows']:,} out={res['out_rows']:,})"
                )

    # ---- summary -----------------------------------------------------------
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skip"]
    bad = [r for r in results if r["status"] not in ("ok", "skip")]
    rows_in = sum(r["src_rows"] for r in results)
    rows_out = sum(r["out_rows"] for r in results if r["status"] in ("ok", "skip"))
    bytes_out = sum(r["out_bytes"] for r in results if r["status"] in ("ok", "skip"))
    files_out = len(ok) + len(skipped)

    console.print(
        f"\n[bold]Done in {time.time()-t0:.0f}s[/bold]: "
        f"{tot_files:,} orbit files -> {files_out:,} month files  "
        f"({tot_files / max(files_out,1):.0f}x fewer)\n"
        f"  compacted {len(ok)}  skipped {len(skipped)}  failed {len(bad)}\n"
        f"  rows: {rows_in:,} in / {rows_out:,} out   "
        f"output size: {bytes_out/1e9:.1f} GB"
    )
    if bad:
        console.print(f"[red]{len(bad)} partition(s) failed/mismatched — "
                      f"source tree untouched, safe to re-run.[/red]")
        return 1
    if rows_in != rows_out:
        console.print(f"[red]Row total mismatch: in {rows_in:,} != out {rows_out:,}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
