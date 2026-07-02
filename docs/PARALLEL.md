# Parallel execution strategy (8 nodes × 20 CPU via SLURM)

The PF build is **two stages with different bottlenecks**. Treat them separately.

| Stage | Unit of work | Bottleneck | Cores you can actually use |
|---|---|---|---|
| 1. Radar+imager → features/pixels | one **orbit** (~1–1.7 GB of Earthdata granules) | **download bandwidth + /dev/shm**, not CPU (labeling an orbit is seconds) | a fraction of 160 — over-subscribing triggers Earthdata throttling and fills tmpfs |
| 2. ERA-5 co-location | one **ERA-5 hour** (~150 MB global from GCS) | **GCS fetch + memory** (light CPU) | all 160 — GCS scales, ~200 MB/worker |

Both stages are **embarrassingly parallel and idempotent**: deterministic `feature_id`, one Parquet file per orbit (Stage 1) / per orbit's hour (Stage 2), atomic writes, no shared state, no locking. Re-running any unit overwrites cleanly. The parallel smoke test (`scripts/smoke_test_parallel.py`) verifies parallel output == serial output, disjoint ids, and idempotent re-runs.

## Stage 1 — radar (download-bound): SLURM job array, capped concurrency
Each orbit worker logs into Earthdata, downloads radar+imager to a unique `/dev/shm/pf_<mission>_<orbit>/`, labels, writes its two Parquet files, and cleans up. **Do not run 20 downloads/node.**

- **Submit:** one array task per month (`scripts/submit_pf.py`, month-array), `%8` to cap to 8 concurrent nodes.
- **Per node:** `run_orbits_parallel.py MISSION --start ... --end ... --workers 8` — a `spawn` Pool of ~**6–10** workers, NOT 20. Why: (a) Earthdata/PPS throttles aggressive concurrent pulls (watch for HTTP 429); (b) 20 × ~1.5 GB = 30 GB of tmpfs/node — check `df -h /dev/shm` and keep `workers × 1.5 GB` under it, or set the download dir to `/data/scratch` instead of `/dev/shm`.
- **Net effect:** ~8 nodes × 8 = ~64 orbits in flight. An orbit is ~1–2 min wall (download-dominated), so ~30–60 orbits/min cluster-wide → a full GPM year (~5,800 orbits) in ~2–3 h; TRMM similar.
- **Tuning knobs:** `--workers` down if you see 429s or tmpfs pressure; the bounded retry+backoff in `granule._download_with_retry` already absorbs transient throttling; failed orbits land in `failed_orbits.txt` for a one-line retry.

```bash
# one month per array task, 8 nodes max in flight
sbatch --array=0-11%8 submit_pf.sh GPM 2018      # internally: run_orbits_parallel.py --workers 8
```

## Stage 2 — ERA-5 (GCS-bound): one fetch per hour, then match-many
ERA-5 reads the **feature table** Stage 1 produced (so it runs *after* Stage 1: `--dependency=afterok:`). `era5_for_features` already groups features by their nearest ERA-5 hour and does **one global GCS fetch per unique hour**, matching all features in that hour in memory (verified: orbit 522's 265 features → **2 fetches**). The only rule for scaling: **partition the unique hours across workers so no hour is fetched twice.**

Two ways, both fine; mirror feng_tracking's choice:

**(a) Dask-jobqueue SLURMCluster (feng_tracking's `era5_claude.py` pattern) — recommended for large runs.** One driver, dynamic load balancing across variable-sized hour batches:
```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
cluster = SLURMCluster(queue="node", cores=20, memory="120GB", walltime="04:00:00",
                       job_extra_directives=["--constraint=..."])   # match your partition
cluster.scale(jobs=8)                                               # 8 nodes
client = Client(cluster)
# hour_batches = list of (records_for_that_hour) — each hour appears once -> fetched once
futures = client.map(process_hour_batch, hour_batches)             # 160 cores, dynamic
```
Use all 20 cores/node (ERA-5 is light: ~200 MB/worker × 20 = 4 GB ≪ 120 GB). `client.map` over per-hour batches means each hour is one task → fetched exactly once; the scheduler balances stragglers.

**(b) SLURM job array (simplest, no scheduler).** Compute the unique-hour list once, split into `array-size` disjoint index-strided chunks; each array task runs `add_era5.py`-style over its hours with an intra-node `spawn` Pool of 20. Robust and idempotent; less dynamic balancing than Dask.

`scripts/add_era5.py` already implements the intra-node half (group by hour → spawn Pool over hour-batches → per-orbit output). For multi-node, either wrap it in a SLURM array over disjoint hour-ranges (b) or lift its `_worker` into the Dask `client.map` (a).

## Why not one giant Dask cluster for both stages?
Stage 1 is download/tmpfs-bound and benefits from *fewer* concurrent workers + per-node tmpfs isolation — a SLURM array with capped Pools models that cleanly. Stage 2 is GCS-bound and benefits from *many* workers + dynamic balancing — Dask shines. Matching the tool to the bottleneck beats one-size-fits-all.

## Quick reference — knobs
- Stage 1 download concurrency: `run_orbits_parallel.py --workers` (start 8; lower on 429/tmpfs).
- Stage 1 download dir: `/dev/shm` (fast, small) vs `/data/scratch` (large) — pick by tmpfs size.
- Stage 2 workers: 20/node (light); ensure each hour assigned to exactly one worker.
- Both: idempotent — safe to re-run failed units; `failed_orbits.txt` (Stage 1) for retry.
- Validate first: `python scripts/smoke_test_parallel.py` (offline, cached orbits) before a cluster run.
