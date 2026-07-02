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

## Data products & artifacts

The published climatology derived from this code:

- **Hugging Face dataset** (0.05° Zarr + multiscale pyramid + standalone 0.05°/0.25°
  CF-NetCDF): [`snesbitt/pf-grid-tiles`](https://huggingface.co/datasets/snesbitt/pf-grid-tiles)
  — DOI [10.57967/hf/9189](https://doi.org/10.57967/hf/9189)
- **Interactive Atlas viewer** (HF Space):
  [`snesbitt/pf-grid-tiles-app`](https://huggingface.co/spaces/snesbitt/pf-grid-tiles-app)
- **Zenodo archive** (0.05° + 0.25° NetCDF, CC-BY-4.0): _DOI pending publication_

The gridded fields cover TRMM PR (1997–2014) + GPM DPR 2ADPR (2014–present),
81 variables = {GPM, TRMM, COMBINED} × 27 quantities on a ±68°, 0.05° grid.

## License & citation

Code in this repository is released under the **MIT License** (see `LICENSE`).
The **data products** are licensed **CC-BY-4.0**.

If you use the database or its derived climatology, please cite:

> Nesbitt, S. W. (2026). *High Resolution Precipitation Climatologies from NASA
> Precipitation Measurement Missions* [Data set]. University of Illinois
> Urbana-Champaign / Hugging Face. https://doi.org/10.57967/hf/9189

Methodology: Nesbitt, S. W., R. Cifelli, and S. A. Rutledge (2006), *Storm
morphology and rainfall characteristics of TRMM precipitation features*,
Mon. Wea. Rev., 134, 2702–2721, doi:10.1175/MWR3200.1.
