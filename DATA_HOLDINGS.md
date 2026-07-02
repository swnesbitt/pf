# PF Database — Data Holdings

_Snapshot: 2026-06-20. Root: `/data/scratch/a/snesbitt/pf_db` (≈529 GB)._

TRMM/GPM precipitation-feature database. Every product is Hive-partitioned
Parquet under `{PF_ROOT}/<product>/mission={GPM,TRMM}/year=YYYY/month=MM/`,
except the published Zarr climatologies. Missions:

- **GPM** — Ku-band DPR, product `GPM_2ADPR`/`GPM_2AKu` (**V07, version-uniform**),
  `DBZ_THRESHOLD = 12 dBZ`.
- **TRMM** — GPM-reprocessed PR, product `GPM_2APR` (**V07 only**),
  `DBZ_THRESHOLD = 16 dBZ`. Note: the GPM-reprocessed TRMM PR is read with the
  *same* `FS` swath reader as GPM (near-surface rain = `FS/SLV/precipRateNearSurface`).

**Version policy — V07-only (uniform across the whole record).** GPM is pinned to
V07 in `pf.config.PRODUCT_VERSION`; TRMM `GPM_2APR` is V07-only. NASA's V08 DPR
reprocessing covers only part of the GPM archive, so preferring it would mix
versions across the record — it is deliberately **not** used. When V08 (or V10)
covers the full archive, the record will be migrated version-uniform.

Feature thresholds: `MIN_PIXELS = 1`, `MIN_AREA_KM2 = 0` (pixel count binds),
4-connectivity. Grid: 0.05°, origin (−90°, −180°), 3600 × 7200 cells; shared
climatology grid clipped to ±68°.

---

## Products on disk (`/data/scratch/a/snesbitt/pf_db`)

| product | size | partition | grain | what it holds |
|---|---|---|---|---|
| `features/` | 25 GB | mission/year/month, `orbit=NNNNNN.parquet` | one row per feature | 58-col feature table (geometry, echo-tops w/ Hirose-2023 QC, PCT, vol-rain, conv/strat, MCS flag, `feature_class`) |
| `pixels/` | 90 GB | mission/year/month | one row per feature pixel | per-pixel `lat/lon/near_sfc_dbz/near_sfc_rain/pct_85_89/pct_37/rain_type/pixel_area_km2/bb_height` |
| `views/` | 40 GB | mission/year/month | sparse 0.05° cell | per-orbit `n_views` denominator (`lat_bin,lon_bin,n_views,mission,orbit,time`) — **no hour axis** |
| `era5/` | 129 GB | mission/year/month | one row per feature | ERA5 environment co-located to each feature (CAPE/CIN/SST/SKT/shear + 2.5°/5° box stats) |
| `grid/` | 75 GB | mission/year/month, `{views,rain}.parquet` | sparse 0.05° cell × **UTC hour** × class | **swath-gridded** diurnal rain/views (Stage-1 of the diurnal climatology) |
| `consolidated/` | 160 GB | product/mission/year, `data.parquet` | one zstd file per (product, mission, year) | DuckDB superdatabase backing store (see below) |

### `grid/` schema (the diurnal swath-grid product)
- `views.parquet`: `lat_bin(i16), lon_bin(i16), hour(i8), n_views(i64)`
- `rain.parquet`: `lat_bin, lon_bin, hour, size_class(i8), echotop_class(i8), raintype(i8), rain_sum(f64, Σ mm/hr), raining_count(i64)`

A *view* = a pixel with finite `lat/lon/near_sfc_rain` and a valid scan hour.
`raining_count` counts `near_sfc_rain > 0 & raintype ≥ 0`. `rain_sum` is Σ
instantaneous near-surface rate (mm/hr), **not** a depth.

### Superdatabase (`pf_catalog.duckdb` + `consolidated/`)
Query the whole archive without opening ~150k orbit files. `consolidated/<product>/
mission=M/year=Y/data.parquet` holds one zstd-Parquet per (product, mission, year)
— the physical `mission` column is dropped on write so it becomes the hive
partition key on read-back (no duplicate-name collision). `pf_catalog.duckdb`
exposes DuckDB views `features` / `pixels` / `era5` over these.

```python
import duckdb
con = duckdb.connect("/data/scratch/a/snesbitt/pf_db/pf_catalog.duckdb", read_only=True)
con.execute("PRAGMA threads=16")
con.execute("SELECT mission, count(*) FROM features GROUP BY 1").df()
```

> **State note (2026-06-20):** the catalog/`consolidated/` are currently the
> stale 06-16 build (keyed to the old GPM features); they are being **rebuilt to
> V07** automatically once the GPM ERA5 re-colocation (SLURM job 648524) finishes.

---

## Temporal coverage

| mission | first | last | month-partitions | stubs |
|---|---|---|---|---|
| **GPM** | 2014-03 | 2026-02 | **144** | **0** |
| **TRMM** | 1997-12 | 2014-10 | **203** | **0** |

(coverage as measured from the `grid/` product.) The `grid/` product is **fully
built and validated** — every month is a complete partition (GPM 144/144, TRMM
203/203), the `raining_count ≤ n_views` invariant holds everywhere, and the
per-scan UTC hour axis is populated. TRMM is contiguous 1997-12 → 2014-10; there
is nothing after 2014-10 (PR boost-down).

### Grid clobber-bug fix (history)
The earlier "stub" months were caused by a concurrency bug: months were bucketed
by mean scan time, so boundary orbits were written into a neighbor month and
clobbered it under parallel execution. Fixed by **per-scan month-window masking**
(`grid_swath(..., time_window=)` folds a `[t0,t1)` mask into the valid pixels, and
`grid_month.py` assigns each scan to exactly one month). The current `grid/` is
the clean full rebuild under that fix.

### Genuine archive limits (cannot be filled)
- **GPM 2014-03** (launch) and **TRMM 2014-10** (PR shut off ~Oct 7) are real
  *partial* months — they read low by construction, not because of a bug.
- **TRMM after 2014-10** — the PR was powered off for the orbit boost-down; the
  brief low-altitude 2015-02..04 data is not carried in this record.

---

## Grid headline totals (full record)

| | GPM (Ku, ≥12 dBZ) | TRMM (PR, ≥16 dBZ) |
|---|---|---|
| Σ `n_views` | 26,015,587,499 | 42,563,946,567 |
| Σ `rain_sum` (mm/hr) | 2,189,837,552 | 4,320,165,236 |
| Σ `raining_count` | 1,340,075,491 | 1,407,279,176 |

### Diagnostic summary (derived from the totals above)

| metric | GPM (Ku, ≥12 dBZ) | TRMM (PR, ≥16 dBZ) |
|---|---|---|
| raining fraction (Σ raining / Σ views) | 0.0515 | 0.0331 |
| mean (unconditional) rain rate (Σ rain / Σ views) | 0.084 mm/hr | 0.102 mm/hr |
| conditional intensity (Σ rain / Σ raining) | 1.63 mm/hr | 3.07 mm/hr |

The contrast is the expected **sensitivity difference**: GPM Ku detects far more
light precipitation (higher raining fraction, lower conditional intensity); TRMM
PR's ~16–18 dBZ floor misses light rain (lower raining fraction) and so its
*conditional* intensity is biased high.

---

## Feature / environment products

Month-partition counts (on disk now):

| product | GPM months | TRMM months |
|---|---|---|
| features / pixels / views | 144 (2014-03..2026-02) | 198 |
| era5 | 144 (2014-03..2026-02) | 187 (..2013-12) |

- **GPM** features/pixels/views were re-ingested to V07 (2026-06-19); ERA5 is
  being re-colocated to those V07 features now (job 648524).

### ⚠️ Known feature/era5 gaps (the `grid/` product does NOT share these)
- **TRMM feature gap:** `features`/`pixels`/`views` are **missing 2012-12 and
  2013-01..04** — the **`grid/` product has these months**, so this is a
  feature-product hole, not a grid hole.
- **TRMM ERA5 incomplete:** `era5` ends **2013-12** (187 mo) vs TRMM features to
  2014-10 (198 mo) — missing ~2014 plus the 2012-12..2013-04 stretch.
- Filling these is a separate TRMM re-ingest / ERA5 re-colocation task (not yet done).

---

## Published / derived products

| product | location | notes |
|---|---|---|
| `pf_tiles.zarr` | `/data/scratch/a/snesbitt/pf_tiles.zarr` (760 MB) | **annual** 0.05° climatology, 18 vars = {GPM,TRMM,COMBINED} × {rain(mm/yr), rate, freq, intensity, raining_views, views}; served by the HF tile app (**V07**) |
| HF dataset | `snesbitt/pf-grid-tiles` | DOI **10.57967/hf/9189** (CC-BY-4.0); hosts `pf_tiles.zarr` |
| HF Space | `snesbitt/pf-grid-tiles-app` | "PMM High Resolution Precipitation Radar Atlas" — xpublish-tiles map viewer |
| diurnal climatology | `s3://spaceborne-grids/pf_grid_{GPM,TRMM}.zarr` (MinIO icechunk) | Stage-2 output: `views/rain` on (month-of-year, UTC hour, lat, lon) shared ±68° grid (**V07**) |

---

## Provenance notes
- TRMM is zero poleward of ±38°; the shared climatology grid spans ±68°.
- `COMBINED` = GPM + TRMM pooled (annual `pf_tiles.zarr` only).
- Version policy: **V07-only, version-uniform** (GPM pinned V07; TRMM PR V07-only).
  NASA V08 DPR reprocessing is only partial and is not used.
- Grids/rain written via icechunk to MinIO for the swath-grid climatology;
  Parquet products are local on `/data/scratch`.
