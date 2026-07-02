#!/usr/bin/env python
"""Intra-node parallel driver: process all orbits in a date window concurrently.

Mission-aware (GPM or TRMM), radar **and** imager. Orbits are embarrassingly
parallel — each worker logs in to Earthdata, downloads its radar (and best-effort
imager) granule to a unique ``pf_<mission>_<orbit>`` dir under the download cache,
writes its per-orbit Parquet files, then deletes that dir. There is no shared
state or locking. Failures are collected (not fatal) into ``failed_orbits.txt``.

Cache is **bounded and monitored**: because every worker removes its own
transient dir when done, the live cache is at most ``#workers x per-orbit
granule size``. A lightweight background thread samples the total size of all
``pf_*`` dirs under the download dir every few seconds and tracks the PEAK,
which is reported at the end to prove the bound holds.

Usage
-----
    python scripts/run_orbits_parallel.py GPM --start 2018-07-01 --end 2018-07-02 \
        --workers 8 --download-dir /dev/shm

A ``spawn`` Pool is used (not ``fork``) so h5py / pyresample / HDF5 initialize
cleanly per process.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


# --------------------------------------------------------------------------
# Worker (top-level → picklable, spawn-safe)
# --------------------------------------------------------------------------
def _worker(item: tuple) -> dict:
    """Pool worker: process one (mission, orbit, radar, imager, download_dir, root).

    Each worker re-imports the package fresh (spawn) and pins
    ``config.DOWNLOAD_DIR`` / ``config.PF_ROOT`` from the values resolved by the
    parent so the orchestrator's CLI flags reach every process. Earthdata login
    is handled inside ``process_orbit`` for remote handles.
    """
    mission, orbit, radar, imager, download_dir, root = item
    from pf import config
    from pf.granule import process_orbit

    if download_dir is not None:
        config.DOWNLOAD_DIR = download_dir
        os.environ["PF_DOWNLOAD_DIR"] = download_dir
    if root is not None:
        config.PF_ROOT = root
        os.environ["PF_ROOT"] = root

    return process_orbit(mission, orbit, {"radar": radar, "imager": imager})


# --------------------------------------------------------------------------
# Cache monitoring
# --------------------------------------------------------------------------
def _cache_bytes(download_dir: str) -> int:
    """Total size (bytes) of all ``pf_*`` dirs directly under ``download_dir``."""
    root = Path(download_dir)
    if not root.is_dir():
        return 0
    total = 0
    for entry in root.glob("pf_*"):
        if not entry.is_dir() or entry.is_symlink():
            continue
        for f in entry.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink():
                    total += f.stat().st_size
            except OSError:
                pass
    return total


class _CacheMonitor:
    """Background sampler that tracks the PEAK transient cache size."""

    def __init__(self, download_dir: str, interval: float = 3.0) -> None:
        self.download_dir = download_dir
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, _cache_bytes(self.download_dir))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> "_CacheMonitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)
        self._sample()  # final reading


def _gb(nbytes: int) -> float:
    return nbytes / (1024.0 ** 3)


# --------------------------------------------------------------------------
# Work-list construction
# --------------------------------------------------------------------------
def _build_work_items(mission: str, start: str, end: str,
                      download_dir: str | None, root: str | None) -> list[tuple]:
    """Resolve radar + imager granules per orbit in the window (no download).

    The CMR temporal search returns any granule that *overlaps* the window, so a
    boundary orbit spanning midnight (e.g. starting 23:55 on day N, ending on day
    N+1) is returned by both day N's and day N+1's single-day searches. To avoid
    double-processing across a per-day SLURM array, we keep each orbit only for
    the day on which its radar granule's observation *starts*: orbits are filtered
    to those whose start-date falls within the inclusive [start, end] window.
    """
    import datetime as _dt

    from pf import search as _search

    radar_cls, imager_cls, radar_short, imager_short = _search._MISSION_PRODUCTS[mission]

    _search.login()

    # Radar: one search over the whole window, grouped by orbit. Apply the
    # configured per-product version preference (radars stay V07 explicitly).
    radar_reader = radar_cls()
    radar_granules = _search.search_granules(radar_short, start, end)
    radar_granules = _search.prefer_version(radar_granules, radar_short, radar_reader)
    radar_by_orbit = _search.group_by_orbit(radar_granules, radar_reader)

    # Attribute each orbit to the day it STARTS: keep only orbits whose radar
    # granule's observation start-date is within the inclusive [start, end]
    # window. This makes a single-day task process exactly the orbits that begin
    # on that day, so boundary orbits spanning midnight are handled once (by the
    # day they start), never twice across the array.
    win_start = _dt.date.fromisoformat(start)
    win_end = _dt.date.fromisoformat(end)
    kept: dict = {}
    n_dropped = 0
    n_unparseable = 0
    for orbit, granule in radar_by_orbit.items():
        sdate = _search.granule_start_date(granule, radar_reader)
        if sdate is None:
            n_unparseable += 1
            kept[orbit] = granule  # safe fallback: keep unparseable orbits
        elif win_start <= sdate <= win_end:
            kept[orbit] = granule
        else:
            n_dropped += 1
    radar_by_orbit = kept

    if n_dropped:
        console.print(
            f"[dim]{n_dropped} orbits start outside [{start},{end}] "
            f"(boundary, handled by adjacent day) — skipped[/dim]"
        )
    if n_unparseable:
        console.print(
            f"[yellow]Warning: {n_unparseable} orbit(s) had an unparseable "
            f"start-date and were kept (safe fallback)[/yellow]"
        )

    # Imager (best-effort): same window, grouped by orbit. Apply the same
    # per-product version preference used in search.granules_for_orbit
    # (e.g. V08 imagers when both V07/V08 are present, like TMI/GMI).
    imager_by_orbit: dict = {}
    try:
        imager_reader = imager_cls()
        imager_granules = _search.search_granules(imager_short, start, end)
        imager_granules = _search.prefer_version(
            imager_granules, imager_short, imager_reader
        )
        imager_by_orbit = _search.group_by_orbit(imager_granules, imager_reader)
    except Exception:  # noqa: BLE001 — imager is best-effort
        imager_by_orbit = {}

    items: list[tuple] = []
    for orbit in sorted(radar_by_orbit):
        items.append((
            mission,
            orbit,
            radar_by_orbit[orbit],
            imager_by_orbit.get(orbit),
            download_dir,
            root,
        ))
    return items


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("mission", help="Mission name: GPM or TRMM")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument(
        "--workers",
        type=int,
        default=min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))),
        help="Parallel workers (default min(8, SLURM_CPUS_PER_TASK or cpu_count); "
             "radar work is download-bound so the default is modest)",
    )
    ap.add_argument(
        "--download-dir",
        default=None,
        help="Transient download cache dir (sets PF_DOWNLOAD_DIR / config.DOWNLOAD_DIR "
             "for workers). Default: config.DOWNLOAD_DIR (PF_DOWNLOAD_DIR or /dev/shm)",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="PF output root (sets PF_ROOT / config.PF_ROOT). Default: config.PF_ROOT",
    )
    ap.add_argument(
        "--failed-file",
        default="failed_orbits.txt",
        help="Path to write orbits that failed",
    )
    ap.add_argument(
        "--serial",
        action="store_true",
        help="Run orbits serially in-process (debugging; bypasses the Pool)",
    )
    args = ap.parse_args()

    # Resolve cache/root and pin them in this process too (so the monitor and
    # the optional --serial path see the same values the workers will use).
    from pf import config, granule

    download_dir = args.download_dir if args.download_dir is not None else config.DOWNLOAD_DIR
    config.DOWNLOAD_DIR = download_dir
    os.environ["PF_DOWNLOAD_DIR"] = download_dir
    if args.root is not None:
        config.PF_ROOT = args.root
        os.environ["PF_ROOT"] = args.root

    # Sweep stragglers from hard-killed prior runs before we start.
    swept = granule.cleanup_stale_downloads(download_dir)
    if swept:
        console.print(f"[dim]Swept {swept} stale pf_* download dir(s) from {download_dir}[/dim]")

    console.print(
        f"Resolving [bold]{args.mission}[/bold] orbits "
        f"{args.start} → {args.end} (radar + imager)…"
    )
    items = _build_work_items(args.mission, args.start, args.end, download_dir, args.root)
    n_imager = sum(1 for it in items if it[3] is not None)
    console.print(
        f"[bold]{len(items)}[/bold] orbits to process with "
        f"[bold]{args.workers}[/bold] workers "
        f"([bold]{n_imager}[/bold] with imager, "
        f"[bold]{len(items) - n_imager}[/bold] radar-only). "
        f"cache={download_dir}"
    )

    failed: list[dict] = []
    totals = {"ok": 0, "empty": 0, "skipped_no_radar": 0, "failed": 0}
    n_features = 0
    n_pixels = 0

    monitor = _CacheMonitor(download_dir).start()
    t0 = time.time()

    progress_cols = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )

    def _account(res: dict) -> None:
        nonlocal n_features, n_pixels
        totals[res["status"]] = totals.get(res["status"], 0) + 1
        n_features += res.get("n_features", 0)
        n_pixels += res.get("n_pixels", 0)
        if res["status"] == "failed":
            failed.append(res)

    if not items:
        console.print("[yellow]No orbits found in window — nothing to do.[/yellow]")

    try:
        if args.serial:
            with Progress(*progress_cols, console=console) as progress:
                task = progress.add_task("Processing orbits", total=len(items))
                for item in items:
                    _account(_worker(item))
                    progress.advance(task)
        elif items:
            ctx = mp.get_context("spawn")
            with Progress(*progress_cols, console=console) as progress, \
                    ctx.Pool(args.workers) as pool:
                task = progress.add_task("Processing orbits", total=len(items))
                for res in pool.imap_unordered(_worker, items):
                    _account(res)
                    progress.advance(task)
    finally:
        monitor.stop()

    elapsed = time.time() - t0
    n = len(items)
    throughput = (n / elapsed * 60.0) if elapsed > 0 else 0.0
    peak_gb = _gb(monitor.peak_bytes)

    console.print(
        f"\n[bold]Done.[/bold] orbits={n}  "
        + "  ".join(f"{k}={v}" for k, v in totals.items())
        + f"  features={n_features}  pixels={n_pixels}"
    )
    console.print(
        f"elapsed={elapsed:.1f}s  throughput={throughput:.1f} orbits/min  "
        f"workers={args.workers}"
    )
    console.print(
        f"[bold]peak download cache: {peak_gb:.2f} GB[/bold] "
        f"across {args.workers} workers (cache dir {download_dir})"
    )

    if failed:
        Path(args.failed_file).write_text(
            "\n".join(f"{f['orbit']}\t{f.get('error', '')}" for f in failed) + "\n"
        )
        console.print(
            f"[yellow]{len(failed)} failed orbits written to {args.failed_file}[/yellow]"
        )


if __name__ == "__main__":
    main()
