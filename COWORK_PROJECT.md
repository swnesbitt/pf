# Project context — TRMM/GPM Precipitation-Feature (PF) Database & 0.05° Climatology

Use this as the working brief for launching analysis jobs. PI: Stephen W. Nesbitt (UIUC),
NASA PMM/Weather funded. Repo: `/data/keeling/a/snesbitt/python/pf` (package `pf`).

## What the project is
A precipitation-feature database + a very-high-resolution (0.05°) precipitation climatology
built **directly from the orbital radar swaths** of TRMM PR and GPM DPR (Ku). Gridded rain =
the near-surface precipitation rate (`precipRateNearSurface`, a.k.a. `near_sfc_rain`, mm/hr).
Public products: an interactive HF Atlas + a DOI-citable Zarr dataset.

## Data products & where they live (root: `/data/scratch/a/snesbitt/pf_db`)
| product | path | grain |
|---|---|---|
| **features** | `features/mission=/year=/month=/orbit=*.parquet` | one row / precip feature (58 cols) |
| **pixels** | `pixels/…` | one row / feature pixel (2.6 B rows) |
| **views** | `views/…` | per-orbit 0.05° sampling denominator |
| **era5** | `era5/…` | per-feature environment (254 cols) |
| **grid (V07)** | `grid/mission=/year=/month=/{views,rain}.parquet` | 0.05° × UTC-hour × size/echotop/raintype class |
| **superdatabase** | `pf_catalog.duckdb` + `consolidated/{features,pixels,era5}/` | consolidated DuckDB views |
| **shear/CAPE table** | `analysis/shear_cape/` | `features ⨝ era5`, \|lat\|<40, both missions |
| **MinIO climatology** | `s3://spaceborne-grids/pf_grid_{GPM,TRMM}` (icechunk zarr) | month-of-year × hour |
| **annual tiles** | `/data/scratch/a/snesbitt/pf_tiles.zarr` → HF | 18 vars (GPM/TRMM/COMBINED × 6) |
| **HF** | dataset `snesbitt/pf-grid-tiles` (DOI 10.57967/hf/9189), Space `snesbitt/pf-grid-tiles-app` | published Atlas |

## Entry point for analysis — query the catalog (don't open 150k files)
```python
import duckdb
con = duckdb.connect("/data/scratch/a/snesbitt/pf_db/pf_catalog.duckdb", read_only=True)
con.execute("PRAGMA threads=16")
# views: features (158.5M rows), pixels (2.6B), era5 (154M). Push aggregations into SQL.
df = con.execute("""SELECT mission, count(*) FROM features GROUP BY 1""").df()
```
`mission`/`year` are hive columns; derive `month` from the `time` column (pixels has no time).

## Key columns
- **features:** `mission, time, centroid_lat/lon, frac_land/ocean/coast, area_km2, npixels,
  max_ht_{20,30,40}dbz` (m; NaN if no such echo), `min_pct_85_89, min_pct_37` (K),
  `conv_area_km2, strat_area_km2, conv/strat_area_frac, conv/strat_rain_frac,
  volrain_{total,conv,strat}, is_mcs, feature_class, major_axis_km, ...`
- **feature_class** ∈ {`MCS` (radar contiguous area ≥ 2000 km²), `sub_MCS_conv`,
  `stratiform_only`, `weak`}.  Headline result: MCS ≈ 1.7% of features → ~74% of rain.
- **era5 (per feature):** `cape/cin/sst/skt/tpr/shear_{1000,3000,6000}m_centroid` + box stats
  (`{min,max,mean,std,p10..p95}_<var>_{5deg,2p50deg,1p25deg}`). Shear = 10 m→H m bulk
  magnitude (so `shear_6000m` ≈ 0–6 km). Use **ambient box CAPE** (e.g. `p90_cape_2p50deg`),
  **`mean_skt`** (defined over land+ocean, unlike SST) for analysis, not the depleted centroid.

## Conventions / facts (important)
- **Versions: GPM radar = V07, TRMM = V07** (uniform). V08 DPR reprocessing in progress; migrate later.
- **Coverage:** TRMM 1997-12 → **2014-10-07** (PR boost-down; nothing after); GPM 2014-03 → 2026-02.
- **grid `rain` field is Σ instantaneous mm/hr, NOT a depth.** Derived: rate=rain/views,
  freq=raining/views, intensity=rain/raining; annual mm/yr = rate × 8766.
- TRMM is zero poleward of ±38°; COMBINED = GPM+TRMM pooled.
- Sensitivity: TRMM PR ≈17–18 dBZ min vs GPM DPR ≈12 dBZ → light-rain discontinuities.

## Environments (conda)
- **`pf`** (Jupyter kernel "Python (pf)") — pipeline, duckdb, grid/climatology/tiles scripts.
- **`pf_ml`** (kernel "Python (pf_ml)") — ML/XAI: shap, xgboost, lightgbm, catboost, interpret(EBM),
  econml (DML/causal forest), statsmodels, pygam, PyALE.
- **`/data/scratch/a/snesbitt/xpt_venv`** — xpublish-tiles + HF deploy (`hf` CLI).
- Activate: `source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pf`

## Existing analyses (templates to copy)
- `notebooks/pf_overview.ipynb` — PF maps, extremes, populations, by-month inventory (pf).
- `notebooks/nesbitt2006_replication.ipynb` — morphology/rain by class, diurnal, regional, 2-D hist (pf).
- `notebooks/shear_cape/` — shear×CAPE hypotheses: composites (00–05, pf) + regression/EBM/SHAP/ALE/causal-ML
  (06–09, pf_ml); shared engine `_shc.py`; design in `notebooks/shear_cape_hypotheses_PLAN.md`.

## Cluster / compute rules (READ before launching)
- **SLURM; run heavy work on compute nodes, NEVER the head node.** Account `snesbitt-group`,
  constraint `j48` (48-core, 253 GB nodes).
- Partitions: `seseml` (MaxNodes=1), **`sesempi` (MaxNodes=8 — use for multi-node)**,
  `sesebig` (MaxNodes=32), `l40s` (GPU, 96-core — for heavy ML training).
- **Concurrent-job cap ≈ 8** (`AssocMaxJobsLimit`). To use many nodes in one job slot, submit a
  multi-node job that `srun`-packs tasks across its nodes (see `scripts/` + `pack_*.sh` patterns).
- **Earthdata:** keep ≤ ~128 concurrent downloads; a 120 s socket timeout (`PF_DOWNLOAD_TIMEOUT`)
  prevents hung downloads from freezing workers.

## Good first analysis jobs
1. Query the catalog for a regional/seasonal feature subset and characterize intensity (echo-top, PCT).
2. Composite a response (echo-top, MCS probability, stratiform area) in (CAPE × shear) bins — see `_shc.py`.
3. Map any gridded quantity (rate/freq/intensity) from `grid/` or the MinIO zarr.
4. Train an explainable model (`pf_ml`) of convective intensity on the ERA5 environment + SHAP.
