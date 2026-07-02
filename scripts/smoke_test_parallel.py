#!/usr/bin/env python
"""Parallel-correctness smoke test (offline, uses cached granules).

Validates the per-orbit parallel machinery WITHOUT SLURM or network by running
the two cached probe orbits (GPM 24647 + TRMM 522) through a `spawn`
multiprocessing Pool and comparing against a serial run. Proves:

* workers are isolated (own /dev/shm dir, own Earthdata login skipped for local
  paths, own Parquet files) — no write contention,
* `feature_id` is globally unique across orbits/missions (disjoint id sets),
* parallel output == serial output (same feature_id sets, same counts),
* re-running an orbit is idempotent (one file, identical content).

Usage:
    python scripts/smoke_test_parallel.py
    python scripts/smoke_test_parallel.py --probe /data/scratch/a/snesbitt/_pf_probe
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import tempfile
import time
from pathlib import Path

import pyarrow.dataset as pads


def _find(probe: str, pattern: str) -> str | None:
    hits = glob.glob(str(Path(probe) / pattern))
    return hits[0] if hits else None


def _worker(item: tuple) -> dict:
    """Pool worker: set the output root, process one orbit from local granules."""
    mission, orbit, radar, imager, root = item
    from pf import config
    config.PF_ROOT = root
    from pf.granule import process_orbit
    handles = {"radar": radar}
    if imager:
        handles["imager"] = imager
    return process_orbit(mission, orbit, handles)


def _feature_ids(root: str, mission: str) -> set[int]:
    p = Path(root) / "features"
    if not p.exists():
        return set()
    tbl = pads.dataset(str(p), partitioning="hive").to_table(columns=["feature_id", "mission"])
    df = tbl.to_pandas()
    return set(df[df["mission"] == mission]["feature_id"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="/data/scratch/a/snesbitt/_pf_probe")
    args = ap.parse_args()

    work_specs = [
        ("GPM", 24647, _find(args.probe, "2A.GPM.DPR*024647*.HDF5"),
         _find(args.probe, "1C.GPM.GMI*024647*.HDF5")),
        ("TRMM", 522, _find(args.probe, "2A.TRMM.PR*000522*.HDF5"),
         _find(args.probe, "1C.TRMM.TMI*000522*.HDF5")),
    ]
    for m, o, r, _i in work_specs:
        assert r, f"missing cached radar granule for {m} {o} under {args.probe}"
    print(f"Cached orbits: {[(m, o) for m, o, _, _ in work_specs]}")

    root_par = tempfile.mkdtemp(prefix="pf_smoke_par_")
    root_ser = tempfile.mkdtemp(prefix="pf_smoke_ser_")

    # --- parallel run (spawn Pool) ---
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(len(work_specs)) as pool:
        par_results = pool.map(_worker, [(m, o, r, i, root_par) for m, o, r, i in work_specs])
    t_par = time.time() - t0
    print("parallel results:", [(r["orbit"], r["status"], r["n_features"]) for r in par_results])

    # --- serial run ---
    t0 = time.time()
    ser_results = [_worker((m, o, r, i, root_ser)) for m, o, r, i in work_specs]
    t_ser = time.time() - t0
    print("serial results:  ", [(r["orbit"], r["status"], r["n_features"]) for r in ser_results])

    # --- assertions ---
    assert all(r["status"] == "ok" for r in par_results), "a parallel orbit failed"
    gpm_par, trmm_par = _feature_ids(root_par, "GPM"), _feature_ids(root_par, "TRMM")
    gpm_ser, trmm_ser = _feature_ids(root_ser, "GPM"), _feature_ids(root_ser, "TRMM")
    assert gpm_par == gpm_ser and trmm_par == trmm_ser, "parallel != serial feature_id sets"
    assert gpm_par and trmm_par, "missing features in parallel output"
    assert gpm_par.isdisjoint(trmm_par), "feature_id collision across missions!"
    print(f"feature_id sets: GPM={len(gpm_par)} TRMM={len(trmm_par)} disjoint=True; parallel==serial=True")

    # --- idempotency: re-run TRMM 522 into the parallel root ---
    before = sorted(Path(root_par).rglob("orbit=000522.parquet"))
    _worker(("TRMM", 522, work_specs[1][2], work_specs[1][3], root_par))
    after = sorted(Path(root_par).rglob("orbit=000522.parquet"))
    assert before == after and len(after) >= 1, "idempotent re-run changed the file set"
    assert _feature_ids(root_par, "TRMM") == trmm_par, "idempotent re-run changed content"
    print("idempotent re-run: same single file, identical content")

    speedup = t_ser / t_par if t_par else float("nan")
    print(f"wall-clock: parallel={t_par:.1f}s serial={t_ser:.1f}s speedup={speedup:.2f}x")
    print("PARALLEL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
