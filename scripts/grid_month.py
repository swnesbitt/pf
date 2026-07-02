#!/usr/bin/env python
"""Stage 1: swath-gridded, hour-resolved rain/views accumulator — one month.

Re-downloads the radar swath granules for a date window from NASA Earthdata,
recomputes feature labels + per-feature categories on each swath, grids every
near-surface pixel into 0.05 deg cells with a UTC hour-of-day axis
(:mod:`pf.grid_swath`), accumulates the whole window with bounded groupby-sum,
and writes SPARSE Parquet files per (mission, year, month)::

    {root}/grid/mission={M}/year={YYYY}/month={MM}/views.parquet
    {root}/grid/mission={M}/year={YYYY}/month={MM}/rain.parquet
    {root}/grid/mission={M}/year={YYYY}/month={MM}/metrics.parquet

``metrics.parquet`` carries per-pixel convective echo-tops (20/30/40 dBZ),
heavy-rain occurrence counts (>25/50/75/100 mm/hr), and near-surface DSD params
(epsilon, Nw, Dm) split conv/strat. ``--metrics-only`` writes just that file
(skipping labeling + grid_swath) so it can be added without touching a validated
views/rain grid.

Mirrors ``scripts/run_orbits_parallel.py`` (spawn Pool, /dev/shm cache, cache
monitor, orbit-start-day attribution) but the worker returns sparse frames
instead of writing per-orbit tables, and the gridded rain/views/raining_views
come straight from the swath (not the feature Parquet), preserving the diurnal
cycle and a numerator/denominator on the same pixel universe.

Usage
-----
    python scripts/grid_month.py GPM --start 2018-06-01 --end 2018-06-30 \
        --workers 8 --download-dir /dev/shm --root /data/scratch/a/snesbitt/pf_db
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)

# Reuse the proven driver scaffolding from the radar pipeline.
from run_orbits_parallel import _CacheMonitor, _gb  # type: ignore

console = Console()

# Sparse per-(year,month) schemas.
GRID_VIEWS_SCHEMA = pa.schema([
    pa.field("lat_bin", pa.int16()),
    pa.field("lon_bin", pa.int16()),
    pa.field("hour", pa.int8()),
    pa.field("n_views", pa.int64()),
])
GRID_RAIN_SCHEMA = pa.schema([
    pa.field("lat_bin", pa.int16()),
    pa.field("lon_bin", pa.int16()),
    pa.field("hour", pa.int8()),
    pa.field("size_class", pa.int8()),
    pa.field("echotop_class", pa.int8()),
    pa.field("raintype", pa.int8()),
    pa.field("rain_sum", pa.float64()),
    pa.field("raining_count", pa.int64()),
])

_RAIN_KEYS = ["lat_bin", "lon_bin", "hour", "size_class", "echotop_class", "raintype"]
_RAIN_SUMS = ["rain_sum", "raining_count"]
_VIEWS_KEYS = ["lat_bin", "lon_bin", "hour"]
_VIEWS_SUMS = ["n_views"]

# Metrics table: per-pixel convective echo-tops, heavy-rain counts, epsilon
# (conv/strat), keyed by (lat_bin, lon_bin, hour). Column list comes from
# grid_swath.METRIC_COLS so the schema and the gridder never drift.
from pf import grid_swath as _gs  # noqa: E402
_METRICS_KEYS = ["lat_bin", "lon_bin", "hour"]
_METRICS_SUMS = list(_gs.METRIC_COLS)
GRID_METRICS_SCHEMA = pa.schema(
    [pa.field("lat_bin", pa.int16()), pa.field("lon_bin", pa.int16()),
     pa.field("hour", pa.int8())]
    + [pa.field(c, pa.float64() if c.endswith("_sum") else pa.int64())
       for c in _METRICS_SUMS]
)


# --------------------------------------------------------------------------
# Worker (top-level → picklable, spawn-safe)
# --------------------------------------------------------------------------
def _grid_worker(item: tuple) -> dict:
    """Download + read one orbit, recompute labels, grid the swath.

    item = (mission, orbit, radar_handle, download_dir, root). Returns
    ``{orbit, status, year, month, views, rain[, error]}``; never raises
    (mirrors ``granule.process_orbit`` try/except/finally cleanup).
    """
    import shutil

    mission, orbit, radar, download_dir, root, ym, win, metrics_only = item
    from pf import config, granule, grid_swath
    from pf.label import label_rpf

    if download_dir is not None:
        config.DOWNLOAD_DIR = download_dir
        os.environ["PF_DOWNLOAD_DIR"] = download_dir
    if root is not None:
        config.PF_ROOT = root
        os.environ["PF_ROOT"] = root

    res = {"orbit": int(orbit), "status": "ok", "year": None, "month": None,
           "views": None, "rain": None, "metrics": None}
    tmpdir = Path(config.DOWNLOAD_DIR) / f"pf_grid_{mission}_{orbit}"
    try:
        if radar is None:
            res["status"] = "skipped_no_radar"
            return res
        reader_cls = granule._RADAR_READERS.get(mission)
        if reader_cls is None:
            res["status"] = "failed"
            res["error"] = f"no radar reader for {mission!r}"
            return res

        if not (isinstance(radar, (str, Path)) and Path(radar).exists()):
            import earthaccess
            earthaccess.login(strategy="netrc")
        tmpdir.mkdir(parents=True, exist_ok=True)
        path = granule._download_with_retry(radar, tmpdir)
        swath = reader_cls().read(path)

        # Partition keys = the task's month window (``ym``). grid_swath masks each
        # scan to ``win`` = [month_start, month_end), so every emitted pixel truly
        # belongs to this month — the per-scan analogue of the hour axis. A boundary
        # orbit appears in both adjacent months' work-lists and contributes each
        # scan to exactly one of them (no double-count, no neighbor clobber).
        res["year"], res["month"] = int(ym[0]), int(ym[1])

        # metrics-only (Phase-3 add-on): build_metrics needs only the swath, so
        # skip labeling + grid_swath entirely and leave views/rain untouched.
        if metrics_only:
            res["metrics"] = grid_swath.build_metrics(swath, time_window=win)
            if res["metrics"] is None:
                res["status"] = "empty"
            return res

        # Recompute labels with the SAME thresholds production used so the
        # recomputed size/echo-top classes match the feature DB.
        dbz = config.DBZ_THRESHOLD_BY_MISSION.get(mission, config.DBZ_THRESHOLD)
        labeled, kept = label_rpf(
            swath, dbz_thresh=dbz, min_area_km2=config.MIN_AREA_KM2,
            min_pixels=config.MIN_PIXELS, connectivity=config.CONNECTIVITY)

        views_df, rain_df = grid_swath.grid_swath(swath, labeled, kept, time_window=win)
        res["views"], res["rain"] = views_df, rain_df
        # New per-pixel metrics (echo-tops / heavy-rain / epsilon), same
        # validity + scan-window rule as grid_swath -> aligned to this month.
        res["metrics"] = grid_swath.build_metrics(swath, time_window=win)
        if views_df is None:
            res["status"] = "empty"
        return res
    except Exception as exc:  # noqa: BLE001 — collected, never fatal
        res["status"] = "failed"
        res["error"] = f"{type(exc).__name__}: {exc}"
        return res
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------
# Radar-only work-list (mirror run_orbits_parallel._build_work_items, no imager)
# --------------------------------------------------------------------------
def _month_windows(start, end):
    """Half-open month windows ``(year, month, t0, t1)`` covering [start, end]."""
    ms, me = pd.Timestamp(start), pd.Timestamp(end)
    out, cur = [], pd.Timestamp(ms.year, ms.month, 1)
    while cur <= me:
        nxt = cur + pd.offsets.MonthBegin(1)
        out.append((int(cur.year), int(cur.month), cur, nxt))
        cur = nxt
    return out


def _radar_work_items(mission, year, month, t0, t1, download_dir, root, metrics_only=False):
    """Orbits whose observation span OVERLAPS the half-open month ``[t0, t1)``.

    The search window is widened one day on each side so a boundary orbit from the
    tail of the previous month — crossing UTC midnight into this month — is found;
    membership is then decided precisely from the filename ``(start, end)`` span.
    A boundary orbit therefore appears in BOTH adjacent months' work-lists, and the
    per-scan ``time_window`` mask in the worker gives each of its scans to exactly
    one month (no double-count, no neighbor clobber). Each item carries the month
    key ``(year, month)`` and the scan window ``(t0, t1)``.
    """
    from pf import search as _search

    radar_cls, _imager_cls, radar_short, _imager_short = _search._MISSION_PRODUCTS[mission]
    _search.login()
    radar_reader = radar_cls()
    search_lo = (pd.Timestamp(t0) - pd.Timedelta(days=1)).date().isoformat()
    search_hi = pd.Timestamp(t1).date().isoformat()
    granules = _search.search_granules(radar_short, search_lo, search_hi)
    granules = _search.prefer_version(granules, radar_short, radar_reader)
    by_orbit = _search.group_by_orbit(granules, radar_reader)

    t0_dt, t1_dt = pd.Timestamp(t0).to_pydatetime(), pd.Timestamp(t1).to_pydatetime()
    kept, n_skip, n_unbounded = {}, 0, 0
    for orbit, gran in by_orbit.items():
        span = _search.granule_time_span(gran, radar_reader)
        if span is None:
            kept[orbit] = gran          # can't bound it — let the scan mask decide
            n_unbounded += 1
            continue
        s, e = span
        if s < t1_dt and e > t0_dt:     # [s, e] overlaps [t0, t1)
            kept[orbit] = gran
        else:
            n_skip += 1
    if n_skip or n_unbounded:
        console.print(f"[dim]{mission} {year}-{month:02d}: kept {len(kept)} orbits "
                      f"({n_skip} outside month, {n_unbounded} unbounded)[/dim]")
    ym, win = (year, month), (t0, t1)
    return [(mission, orbit, kept[orbit], download_dir, root, ym, win, metrics_only)
            for orbit in sorted(kept)]


def _grid_one_month(mission, year, month, t0, t1, args, download_dir, root):
    """Resolve, grid (scan-masked to this month), and write ONE month partition.

    Returns ``(written_tuple_or_None, failed_list, totals_dict)``.
    """
    ymk = (year, month)
    metrics_only = getattr(args, "metrics_only", False)
    items = _radar_work_items(mission, year, month, t0, t1, download_dir, root, metrics_only)
    if not items:
        console.print(f"[yellow]{mission} {year}-{month:02d}: no orbits — skip[/yellow]")
        return None, [], {}

    acc_views, acc_rain, buf_views, buf_rain = {}, {}, {}, {}
    acc_metrics, buf_metrics = {}, {}
    failed, totals = [], {"ok": 0, "empty": 0, "skipped_no_radar": 0, "failed": 0}

    def _flush(force=False):
        for d_acc, d_buf, keys, sums in (
                (acc_views, buf_views, _VIEWS_KEYS, _VIEWS_SUMS),
                (acc_rain, buf_rain, _RAIN_KEYS, _RAIN_SUMS),
                (acc_metrics, buf_metrics, _METRICS_KEYS, _METRICS_SUMS)):
            for ym in list(d_buf.keys()):
                if force or len(d_buf[ym]) >= args.flush_every:
                    merged = _reduce(([d_acc[ym]] if ym in d_acc else []) + d_buf[ym], keys, sums)
                    if merged is not None:
                        d_acc[ym] = merged
                    d_buf[ym] = []

    def _account(res):
        totals[res["status"]] = totals.get(res["status"], 0) + 1
        if res["status"] == "failed":
            failed.append(res)
            return
        if (res["year"], res["month"]) != ymk:   # worker emits in-window only; guard
            return
        if res.get("views") is not None:
            buf_views.setdefault(ymk, []).append(res["views"])
        if res.get("rain") is not None:
            buf_rain.setdefault(ymk, []).append(res["rain"])
        if res.get("metrics") is not None:
            buf_metrics.setdefault(ymk, []).append(res["metrics"])

    cols = (SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn())
    if args.serial:
        with Progress(*cols, console=console) as progress:
            task = progress.add_task(f"{mission} {year}-{month:02d}", total=len(items))
            for item in items:
                _account(_grid_worker(item)); _flush(); progress.advance(task)
    else:
        ctx = mp.get_context("spawn")
        with Progress(*cols, console=console) as progress, ctx.Pool(args.workers) as pool:
            task = progress.add_task(f"{mission} {year}-{month:02d}", total=len(items))
            for res in pool.imap_unordered(_grid_worker, items):
                _account(res); _flush(); progress.advance(task)
    _flush(force=True)

    if not metrics_only:
        _write_partition(acc_views.get(ymk), GRID_VIEWS_SCHEMA, root, mission, year, month, "views")
        _write_partition(acc_rain.get(ymk), GRID_RAIN_SCHEMA, root, mission, year, month, "rain")
    _write_partition(acc_metrics.get(ymk), GRID_METRICS_SCHEMA, root, mission, year, month, "metrics")
    nv = 0 if acc_views.get(ymk) is None else len(acc_views[ymk])
    nr = 0 if acc_rain.get(ymk) is None else len(acc_rain[ymk])
    nm = 0 if acc_metrics.get(ymk) is None else len(acc_metrics[ymk])
    tag = "metrics-only " if metrics_only else ""
    console.print(f"  wrote {tag}{mission} {year}-{month:02d}: views={nv} cells, rain={nr} keys, "
                  f"metrics={nm} cells  "
                  + " ".join(f"{k}={v}" for k, v in totals.items()))
    return (year, month, nv, nr), failed, totals


# --------------------------------------------------------------------------
# Bounded accumulation
# --------------------------------------------------------------------------
def _reduce(frames, keys, sums):
    """Concat sparse frames and groupby-sum down to unique keys."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return None
    cat = pd.concat(frames, ignore_index=True)
    return cat.groupby(keys, as_index=False)[sums].sum()


def _write_partition(df, schema, root, mission, year, month, table):
    """Atomic, idempotent write of one sparse grid table."""
    target_dir = (Path(root) / "grid" / f"mission={mission.upper()}"
                  / f"year={year:04d}" / f"month={month:02d}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{table}.parquet"
    if df is None or len(df) == 0:
        df = pd.DataFrame({f.name: pd.Series(dtype=f.type.to_pandas_dtype())
                           for f in schema})
    tbl = pa.Table.from_pandas(df.reindex(columns=[f.name for f in schema]),
                               schema=schema, preserve_index=False)
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    pq.write_table(tbl, tmp, compression="zstd")
    os.replace(tmp, target)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mission", help="GPM or TRMM")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--workers", type=int,
                    default=min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))))
    ap.add_argument("--download-dir", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--flush-every", type=int, default=200,
                    help="Groupby-sum the buffer into the accumulator every N orbits")
    ap.add_argument("--skip-existing", action="store_true",
                    help="No-op if the month partition already has all expected files")
    ap.add_argument("--metrics-only", action="store_true",
                    help="Write ONLY metrics.parquet (echo-tops/heavy-rain/epsilon); "
                         "skip labeling + grid_swath and leave views/rain untouched")
    ap.add_argument("--failed-file", default="failed_grid.txt")
    ap.add_argument("--serial", action="store_true")
    args = ap.parse_args()

    mission = str(args.mission).upper()
    from pf import config, granule

    download_dir = args.download_dir if args.download_dir is not None else config.DOWNLOAD_DIR
    config.DOWNLOAD_DIR = download_dir
    os.environ["PF_DOWNLOAD_DIR"] = download_dir
    root = args.root if args.root is not None else config.PF_ROOT
    config.PF_ROOT = root
    os.environ["PF_ROOT"] = root

    swept = granule.cleanup_stale_downloads(download_dir)
    if swept:
        console.print(f"[dim]Swept {swept} stale pf_* download dir(s)[/dim]")

    months = _month_windows(args.start, args.end)
    console.print(f"Gridding [bold]{mission}[/bold] {args.start} → {args.end}: "
                  f"[bold]{len(months)}[/bold] month(s), {args.workers} workers, cache={download_dir}")

    monitor = _CacheMonitor(download_dir).start()
    t0 = time.time()
    all_written, all_failed = [], []
    grand = {"ok": 0, "empty": 0, "skipped_no_radar": 0, "failed": 0}
    try:
        for year, month, ws, we in months:
            if args.skip_existing:
                part = (Path(root) / "grid" / f"mission={mission}"
                        / f"year={year:04d}" / f"month={month:02d}")
                need = ["metrics.parquet"] if args.metrics_only else \
                    ["rain.parquet", "views.parquet", "metrics.parquet"]
                if all((part / fn).exists() for fn in need):
                    console.print(f"[dim]{mission} {year}-{month:02d}: partition exists — skip[/dim]")
                    continue
            written, failed, totals = _grid_one_month(
                mission, year, month, ws, we, args, download_dir, root)
            if written is not None:
                all_written.append(written)
            all_failed.extend(failed)
            for k, v in totals.items():
                grand[k] = grand.get(k, 0) + v
    finally:
        monitor.stop()

    elapsed = time.time() - t0
    console.print(f"\n[bold]Done.[/bold] months={len(months)} partitions={len(all_written)}  "
                  + "  ".join(f"{k}={v}" for k, v in grand.items()))
    console.print(f"elapsed={elapsed:.1f}s  peak cache={_gb(monitor.peak_bytes):.2f} GB")
    if all_failed:
        Path(args.failed_file).write_text(
            "\n".join(f"{f['orbit']}\t{f.get('error','')}" for f in all_failed) + "\n")
        console.print(f"[yellow]{len(all_failed)} failed orbits → {args.failed_file}[/yellow]")


if __name__ == "__main__":
    main()
