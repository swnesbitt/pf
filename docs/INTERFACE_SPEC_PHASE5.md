# PF Phase-5 Interface Specification — ERA-5 environment (no IR)

> Port of feng_tracking's ERA-5 functionality to the PF feature database, co-located at the
> feature **centroid** PLUS feng_tracking's box statistics. IR/VIRS is explicitly dropped.
> ERA-5 is a SEPARATE post-processing step writing a SEPARATE table keyed by `feature_id`
> (mirrors feng_tracking, which writes a separate parquet merged on track ID). The 47-col
> FEATURE_SCHEMA and 13-col PIXEL_SCHEMA stay FROZEN.

## Reference implementation
`/data/keeling/a/snesbitt/python/feng_tracking/era5_claude.py` — replicate its data source,
variables, co-location, box stats, and column naming. Adapt track→feature: feng_tracking uses
(track, base_time, meanlat, meanlon); we use (feature_id, time, centroid_lat, centroid_lon).

## Verified ground truth (pf env, this cluster)
- ARCO ERA-5 on GCS: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`,
  anonymous (`storage_options={'token':'anon'}`), `xr.open_zarr(..., chunks=None)`. Time 1900–2050
  (covers TRMM 1997 + GPM). All needed vars present. Requires network (no offline cache).
- Grid: latitude DESCENDING, longitude 0–360, hourly, pressure `level` 1..1000 hPa.
- Validated: orbit-522 storm (−23.4, −57.6, 1997-12-30 23Z) → CAPE 1550 J/kg, CIN 576 J/kg.
- pf env now has xarray/zarr/gcsfs/dask (add to environment.yml). Use xarray interp for shear
  (feng_tracking's fallback path) — do NOT require xgcm.

## Module: src/pf/era5.py

### Constants (mirror feng_tracking)
```python
ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_VARS_2D = {  # ERA5 name -> short name
    "convective_available_potential_energy": "cape",
    "convective_inhibition": "cin",
    "sea_surface_temperature": "sst",
    "skin_temperature": "skt",
    "total_precipitation": "tpr",
}
WIND10 = ("10m_u_component_of_wind", "10m_v_component_of_wind")  # u10, v10 (for shear)
SHEAR_HEIGHTS_M = [1000, 3000, 6000]            # AGL; -> shear_1000m/3000m/6000m
LEVEL_SLICE = (400, 1000)                        # hPa, for geopotential/u/v
BOX_RADII = [2.5, 1.25, 0.625]                   # degrees
BOX_LABELS = {2.5: "5deg", 1.25: "2p50deg", 0.625: "1p25deg"}
PERCENTILES = [10, 25, 50, 75, 90, 95]
STATS = ["min", "max", "mean", "std", "p10", "p25", "p50", "p75", "p90", "p95"]
# the 8 output variables that get centroid + box stats:
STAT_VARS = ["cape", "cin", "sst", "skt", "tpr", "shear_1000m", "shear_3000m", "shear_6000m"]
ERA5_ROOT_SUBDIR = "era5"
```

### Functions
```python
def open_era5(zarr=ERA5_ZARR) -> xr.Dataset
    # xr.open_zarr(zarr, chunks=None, storage_options={'token':'anon'})

def compute_shear_fields(subset: xr.Dataset) -> dict[int, xr.DataArray]
    # subset has u/v/geopotential on `level` (400-1000 hPa) + u10/v10.
    # ght = geopotential/9.81; for each H in SHEAR_HEIGHTS_M: interp u,v to geometric height H
    #   (vertical interp using ght as the coordinate, xarray .interp / swap_dims, method='linear');
    #   shear_H = sqrt((u_H - u10)**2 + (v_H - v10)**2). Returns {H: DataArray(shear, dims lat,lon)}.

def compute_stats(values: np.ndarray, var: str, box_label: str) -> dict
    # min/max/mean/std + p10..p95 over finite values; keys f"{stat}_{var}_{box_label}". NaN-safe.

def era5_for_features(features: pd.DataFrame, ds: xr.Dataset | None = None) -> pd.DataFrame
    # features needs columns: feature_id, mission, orbit, time, centroid_lat, centroid_lon.
    # Batch by nearest ERA5 hour (group on time rounded/nearest to the hour).
    # Per hour: bounding box over all centroids (+max BOX_RADII margin); .sel(time, method='nearest');
    #   load 2D vars + u10/v10 + level-sliced u/v/geopotential; compute_shear_fields once.
    # Per feature: lon360 = centroid_lon % 360; centroid grid-nearest value for each STAT_VAR
    #   -> f"{var}_centroid"; AND box stats over each BOX_RADII (lat slice [clat+r, clat-r] since
    #   descending; lon cyclic-safe subset) via compute_stats. Returns one row per feature with
    #   feature_id, mission, orbit, time + 8 centroid cols + 8*10*3 box-stat cols.

ERA5_SCHEMA: pa.schema   # feature_id int64, mission string, orbit int32, time timestamp(us),
                         # {var}_centroid float32 (8), {stat}_{var}_{box} float32 (240) = 252 cols

def write_era5(era5_df, mission, root=PF_ROOT) -> Path
    # Hive-partitioned dataset at {root}/era5/mission=/year=/month=/orbit=NNNNNN.parquet,
    # year/month from mean time, zstd, coerce_timestamps='us', atomic write (reuse catalog patterns).
```

### Co-location details (mirror feng_tracking exactly)
- Centroid value: round to 0.25° grid implicitly via `.sel(latitude=clat, longitude=lon360, method='nearest')`.
- Box: `get_cyclic_subset`-style lon wrap (ERA5 0–360); lat slice high→low (descending).
- Time: `.sel(time=feature_time, method='nearest')` (hourly).
- Shear relative to 10 m wind, at 1000/3000/6000 m AGL, via geopotential height interp.
- NaN-safe stats (drop NaN before percentile/mean/std).

## CLI / driver
- `pf era5 MISSION --start YYYY-MM-DD --end YYYY-MM-DD [--root ...]` (new typer command in cli.py):
  read the existing features dataset (pyarrow.dataset over {root}/features, hive filter by mission +
  time window) → `era5_for_features` (batched by hour) → `write_era5`. Print rows written.
- `scripts/add_era5.py`: parallel driver over hour-batches (multiprocessing spawn Pool or dask),
  mirroring feng_tracking's batch-by-time design; one ERA5 parquet per orbit (idempotent overwrite).
- Reads the feature table only (feature_id, mission, orbit, time, centroid_lat, centroid_lon) — does
  NOT touch the per-orbit radar/imager pipeline. ERA-5 is opt-in, network-dependent.

## environment.yml / pyproject
Add `xarray`, `zarr`, `gcsfs`, `dask` to environment.yml and pyproject `[era5]` extra (or main deps).

## Querying
Join on feature_id: `features f JOIN era5 e USING(feature_id)` (duckdb/polars/pyarrow), same hive
partitioning as features/pixels.

## FROZEN guarantees
FEATURE_SCHEMA(47)/PIXEL_SCHEMA(13) UNCHANGED; process_orbit/build_feature_row/build_pixel_rows/
write_orbit UNCHANGED; ERA-5 is a separate table + separate code path; no new columns in feature/pixel.

## Verification
1. `era5_for_features` on a small synthetic features DataFrame (mocked tiny xr.Dataset, offline) →
   correct columns (8 centroid + 240 box-stat + 4 meta), centroid value equals the nearest grid point,
   box stats match hand-computed min/max/mean over the box, NaN-safe.
2. ERA5_SCHEMA has 252 + 4 fields; column names match `{var}_centroid` and `{stat}_{var}_{box}`.
3. write_era5 → era5/mission=/year=/month=/orbit=NNNNNN.parquet, re-read joins to features on feature_id.
4. (Network, orchestrator) real check: orbit-522 features → the 69 K storm feature's `cape_centroid`
   ≈ 1550 J/kg (the validated value); GPM orbit 24647 features also get finite CAPE.
