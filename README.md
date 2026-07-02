# pf — TRMM/GPM Precipitation-Feature Database

A Parquet-backed precipitation-feature (PF) database in the spirit of
**Nesbitt et al. (2000, 2006)** and the UU/TAMUCC TRMM-GPM PF databases.

It pulls Level-2 orbital granules directly from NASA Earthdata via
[`earthaccess`](https://earthaccess.readthedocs.io/), identifies **Radar
Precipitation Features (RPFs)** — spatially contiguous near-surface radar
pixels ≥ 20 dBZ, ≥ 75 km² — computes the classic per-feature parameter set
(size, intensity, vertical structure, convective/stratiform partition,
co-located 85/89-GHz PCT, MCS flag), and stores results in a Hive-partitioned
Parquet **feature table + pixel table** joined on a deterministic `feature_id`.

## Status
Phase 1 (GPM-Ku pilot) under construction. See the design plan in
`~/.claude/plans/` and the phased build order there.

## Setup
```bash
conda env create -f environment.yml   # creates env `pf`
conda activate pf
```

## Usage (target)
```bash
pf process-orbit GPM 24531                 # one orbit -> 2 Parquet files
pf process-range GPM 2018-07-01 2018-07-31 # a date range
python scripts/run_orbits_parallel.py ...  # intra-node parallel
python scripts/submit_pf.py ...            # SLURM month-array scale-out
```

Output root: `/data/scratch/a/snesbitt/pf_db` (configurable in `pf/config.py`).
