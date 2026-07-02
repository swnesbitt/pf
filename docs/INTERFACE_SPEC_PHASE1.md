# PF Phase-1 Interface Specification (GPM-Ku pilot)

> Authoritative contract produced by the development/architecture agent. Code-writing
> agents implement against this; the verification agent tests against it. Do not deviate
> without updating this file. Consistent with the approved plan in
> `~/.claude/plans/i-d-like-to-set-velvet-clarke.md`.

Target package root: `/data/keeling/a/snesbitt/python/pf/src/pf/`
Scope: radar-only, GPM 2A-DPR FS swath, end-to-end one orbit → 2 Parquet files. Later-phase
fields are declared but written as NaN/null to keep the 47-column schema frozen.

## Findings from reading `ingest_gpm_dpr.py` (authoritative)

1. **Near-surface reflectivity is a dedicated 2-D field, NOT a bin-index into the 3-D cube.**
   Use `FS/SLV/zFactorFinalNearSurface` directly for `near_sfc_dbz`.
2. Group paths are swath-relative: read `f[swath][relpath]`; Phase-1 `swath="FS"`. Full paths
   like `FS/SLV/precipRateNearSurface`, `FS/SLV/zFactorFinal`, `FS/CSF/typePrecip`, `FS/ScanTime/Year`.
3. FS dims: `cross_track=49` (nray), `range_bin=176` (3-D vertical). `nscan` is the leading axis.
4. Lat/Lon directly under swath group as `Latitude`/`Longitude`, float32, fill `-9999.9`.
5. ScanTime is split int fields (`Year` int16; `Month,DayOfMonth,Hour,Minute,Second` int8;
   `MilliSecond` int16) — compose into datetime64[ns]; no single datetime field exists.
6. `CSF/typePrecip` int32, fill -9999; rain type = `typePrecip // 10_000_000`.
7. **No orbit parsing exists in reusable code** — `orbit_of` is newly specified against the
   standard GPM filename convention. Verify against a real filename in the smoke test.
8. Raw 2A-DPR HDF5 variables are already physical floats; the int16 scale encoding in
   `ingest_gpm_dpr.py` is an Icechunk-store concern only. Reader still honors any HDF5
   `scale_factor`/`add_offset` attrs if present, and masks the fill sentinels to NaN.
9. Download to a unique `/dev/shm/...` dir via `earthaccess.download([h], dir)`, use
   `str(downloaded[0])`, then `shutil.rmtree` the dir. `spawn` start-method for workers.

## 1. `pf/config.py` — module-level constants
```python
DBZ_THRESHOLD: float = 20.0
MIN_AREA_KM2: float = 75.0
CONNECTIVITY: int = 2                 # 8-connectivity
MCS_AREA_KM2: float = 2000.0          # Phase-2 use; defined now
PF_ROOT: str = "/data/scratch/a/snesbitt/pf_db"
MISSION_CODE: dict[str, int] = {"TRMM": 1, "GPM": 2}
MISSION_NAME: dict[int, str] = {1: "TRMM", 2: "GPM"}
SHORT_NAMES: dict[str, str] = {
    "GPM_KU": "GPM_2ADPR", "GPM_GMI": "GPM_1CGMI",
    "TRMM_PR": "TRMM_2A25", "TRMM_2A23": "TRMM_2A23", "TRMM_TMI": "TRMM_2A12",
}
GPM_SWATH: str = "FS"
GPM_RANGE_BIN_SIZE_M: float = 125.0
GPM_N_RANGE_BINS: int = 176
EARTH_RADIUS_KM: float = 6371.0
FILL_SENTINELS: tuple[float, ...] = (-9999.9, -9999.0, -9999, -99.0, -1111.1, -1111)
ID_ORBIT_MULT: int = 100_000          # 1e5
ID_MISSION_MULT: int = 10_000_000_000_000  # 1e13
```
Pure constants module. Functions take thresholds as params defaulting to these.

## 2. `pf/swath.py` — `Swath` dataclass
`@dataclass(slots=True)`. Reference frame `(nscan, nray)`, `nray=49` for FS. 2-D float fields
`np.float32` with fills decoded to NaN; integer fields (`rain_type`,`surface_type`) int8 sentinel -1.

Fields: `mission:str, orbit:int, short_name:str, granule_name:str`;
2-D geoloc/time `lat,lon (float32 nscan,nray)`, `time (datetime64[ns] nscan,)`,
`pixel_area (float32 nscan,nray km^2)`;
2-D radar `near_sfc_dbz, near_sfc_rain (float32)`, `rain_type, surface_type (int8)`;
3-D `dbz_3d (float32 nscan,nray,176)`, `height_3d (float32 m MSL)`;
later-phase NaN placeholders `pct_85_89, bb_height, freezing_level (float32)`.
`@property shape -> (nscan,nray)`; `@classmethod empty(nscan,nray,nbin,*,mission,orbit,short_name,granule_name)`
allocates correct shapes/dtypes, float fields NaN, int fields -1.
Invariants: all 2-D fields share `(nscan,nray)`; `dbz_3d.shape[:2]==lat.shape`; `time.shape==(nscan,)`.

## 3. `pf/readers/hdf5_util.py`
```python
read_var(f, group_path, *, dtype=np.float32) -> np.ndarray      # f[group_path][:]; KeyError if absent
decode_fill(arr, *, sentinels=FILL_SENTINELS, atol=0.05) -> np.ndarray   # sentinel→NaN, float32 copy
read_float(f, group_path, *, sentinels=FILL_SENTINELS) -> np.ndarray     # read+decode; apply scale_factor/add_offset attrs
read_int(f, group_path, dtype=np.int32) -> np.ndarray           # integer field unchanged
has_path(f, group_path) -> bool
```
Stateless; never mutate inputs; group paths root-relative incl. swath prefix.

## 4. `pf/readers/base.py` — `SwathReader` ABC
Class attrs `short_name:str`, `mission:str`. Abstract `read(path)->Swath` (pure read, no network/temp),
`orbit_of(granule_or_filename)->int` (deterministic). Side-effect-free.

## 5. `pf/readers/gpm_ku.py` — `GpmKuReader(SwathReader)`
`short_name="GPM_2ADPR"`, `mission="GPM"`, `swath="FS"`.

| Swath field | HDF5 path | dtype | decode |
|---|---|---|---|
| lat | `FS/Latitude` | float32 | fill→NaN |
| lon | `FS/Longitude` | float32 | fill→NaN |
| near_sfc_dbz | `FS/SLV/zFactorFinalNearSurface` | float32 | fill→NaN (authoritative) |
| near_sfc_rain | `FS/SLV/precipRateNearSurface` | float32 | fill→NaN (mm/hr) |
| dbz_3d | `FS/SLV/zFactorFinal` | float32 | fill→NaN; if trailing `frequency` axis, select Ku index 0 |
| rain_type | `FS/CSF/typePrecip` int32 → `//10_000_000` | int8 | 1=strat,2=conv,3=other,0=none |
| surface_type | `FS/PRE/landSurfaceType` if `has_path` else -1 | int8 | raw code |
| time | `FS/ScanTime/{Year,Month,DayOfMonth,Hour,Minute,Second,MilliSecond}` | →datetime64[ns] | compose per scan |

`height_3d` (m MSL): `height_3d[s,r,b] = (GPM_N_RANGE_BINS-1-b)*GPM_RANGE_BIN_SIZE_M`
(constant 125 m spacing; bottom bin≈0 m, top≈21875 m). Document the approximation; ellipsoid/zenith
refinement is later-phase. Used for `max_ht_{20,30,40}dbz`.

`orbit_of`: GPM filename `2A.GPM.DPR.V<...>.YYYYMMDD-S<...>-E<...>.<ORBIT>.V<NN>.HDF5`; split basename
on `.`, take the 5–6-digit field immediately before the trailing `V\d+`; `int(orbit)`. For an
`earthaccess.DataGranule` derive filename via data link / `GranuleUR` first. `ValueError` if not found.

## 6. `pf/geometry.py`
```python
footprint_area_km2(lat2d, lon2d) -> area2d(float32, km^2)   # per-pixel; great-circle neighbor diffs; NaN→NaN. Across-track varying. Replaces cell_detector._estimate_pixel_area (invalid for swath).
area_weighted_centroid(lat2d, lon2d, weights2d) -> (lat, lon)   # unit-vector lon mean for antimeridian
pca_axes(member_lat, member_lon) -> (major_km, minor_km, orientation_deg, aspect_ratio)
    # PCA on member centers in local equal-area km plane (not index space).
    # major/minor = 4*sqrt(eigenvalues); orientation CCW from East in [-90,90]; aspect=major/minor>=1.
    # >=2 points required; 1 point -> (0,0,0,1).
```
Pure functions. `footprint_area_km2` is the single source of truth for per-pixel area.

## 7. `pf/label.py`
```python
label_rpf(swath, dbz_thresh=DBZ_THRESHOLD, min_area_km2=MIN_AREA_KM2, connectivity=CONNECTIVITY)
    -> (labeled_int32_2d, kept: list[tuple[local_label:int, area_km2:float]])
# mask = isfinite(near_sfc_dbz) & (near_sfc_dbz>=thresh)
# skimage.measure.label(mask, connectivity, return_num=True)
# area per label = scipy.ndimage.sum_labels(swath.pixel_area, labeled, L); keep if >= min_area_km2
# NO 40-dBZ core gate. Array adjacency == swath contiguity.
# labeled returns ONLY retained labels non-zero (others zeroed); kept sorted by local_label.
touches_edge(labeled, local_label) -> bool   # member on row 0/-1 or col 0/-1
```
`local_label` stable (skimage row-major) → deterministic feature_id.

## 8. `pf/feature_id.py`
```python
encode(mission, orbit, local_label) -> int   # mission_code*1e13 + orbit*1e5 + local_label
# asserts 0<local_label<1e5; 0<=orbit<1e5; mission_code in MISSION_CODE.values()
decode(fid) -> (mission_name, orbit, local_label)
```
`decode(encode(...))` round-trips. Fits int64. Sole feature↔pixel join key.

## 9. `pf/features.py` — `build_feature_row(swath, labeled, local_label, area_km2, edge) -> dict`
Phase-1 computes columns 1–34; sets 35–47 to NaN/None. The module exposes
`FEATURE_SCHEMA: pa.schema([...])` (authoritative, passed to `catalog.write_orbit`).
`npixels=int((labeled==local_label).sum())`; `max_ht_Xdbz = nanmax(height_3d where dbz_3d>=X within member)`
(NaN if none); `volrain_total = Σ(near_sfc_rain*pixel_area)` over member.

Frozen 47-column schema (order, dtype, phase):
1 feature_id int64 (P1) · 2 mission string [partition] (P1) · 3 orbit int32 [partition] (P1) ·
4 local_label int32 (P1) · 5 time timestamp(us) (P1) · 6 npixels int32 (P1) · 7 area_km2 float32 (P1) ·
8 centroid_lat float32 · 9 centroid_lon float32 · 10-13 bbox_scan_min/scan_max/ray_min/ray_max int32 ·
14-17 bbox_lat_min/lat_max/lon_min/lon_max float32 · 18 frac_land · 19 frac_ocean · 20 frac_coast float32 ·
21 surface_flag int8 · 22 max_near_sfc_dbz · 23 max_near_sfc_rain · 24 mean_near_sfc_rain float32 ·
25 max_ht_20dbz · 26 max_ht_30dbz · 27 max_ht_40dbz float32 (m MSL) · 28 volrain_total float32 ·
29 major_axis_km · 30 minor_axis_km · 31 orientation_deg · 32 aspect_ratio float32 ·
33 eccentricity float32 (regionprops) · 34 edge bool ·
[placeholders→NaN/null] 35 min_pct_85_89 (P3) · 36 conv_area_km2 · 37 strat_area_km2 ·
38 conv_area_frac · 39 strat_area_frac · 40 conv_rain_frac · 41 strat_rain_frac ·
42 volrain_conv · 43 volrain_strat · 44 mean_bb_height · 45 mean_freezing_level (all P2) ·
46 is_mcs bool-nullable (P2) · 47 feature_class string (P2).

## 10. `pf/catalog.py` — `write_orbit(features_df, pixels_df, mission, root=PF_ROOT) -> (features_path, pixels_path)`
Layout: `{root}/features/mission={M}/year={YYYY}/month={MM}/orbit={NNNNNN}.parquet` and same under
`pixels/`. year/month from MEAN of `features_df['time']`; orbit zero-padded 6 digits; mission upper-cased.
pyarrow write with frozen schema, `compression='zstd'`, `coerce_timestamps='us'`,
`allow_truncated_timestamps=True`. Atomic write (`.tmp` → `os.replace`); `mkdir(parents=True, exist_ok=True)`.
`pixels_df` None/empty (Phase-1 may skip) → still write features. No appends/locking; per-orbit isolated.
Also define `PIXEL_SCHEMA` (Phase-2) but Phase-1 pixels optional.

## 11. `pf/granule.py` — `process_orbit(mission, orbit, granule_handles: dict, cfg) -> dict`
Returns `{"orbit","n_features","n_pixels","status"[, "error"]}`; status ∈
{ok, skipped_no_radar, empty, failed}. Order: login(netrc) → unique `/dev/shm/pf_{mission}_{orbit}` →
download radar (retry/backoff) → `GpmKuReader().read` → `label_rpf` (empty→status empty) →
`build_feature_row` rows + `feature_id.encode` → `write_orbit` → `finally shutil.rmtree`.
Never raises to the Pool — exceptions caught → `{status:'failed','error':repr(e),...}`. Idempotent.
Phase-1 `pixels_df=None`.

## 12. `pf/cli.py` — typer `app`
`process-orbit MISSION ORBIT [--root --overwrite/--skip-existing]`: resolve granule via
`search.granules_for_orbit`, call `process_orbit`, print result, non-zero exit on failure.
`search MISSION --start --end [--short-name]`: login, `search_data`, group by `orbit_of`, print
rich table `orbit | granule_name | products_present`. No downloads. `main()`/`app()` console entry.

## Cross-cutting flags
- Near-surface dBZ = `FS/SLV/zFactorFinalNearSurface` (NOT 3-D index). Confirmed.
- `orbit_of` is new (not in reusable code) — verify against a real filename in smoke test.
- `cell_detector._estimate_pixel_area` NOT reused (regular-grid assumption invalid); reuse its
  skimage label/regionprops *usage pattern* only.
- Schema frozen at 47 columns now so Phases 2–4 only fill placeholders, never migrate Parquet.
