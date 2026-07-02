# PF Phase-3 Interface Specification (imager co-location + PCT fill)

> Authoritative contract produced by the development/architecture agent. Code-writing
> agents implement strictly against this; the verification agent tests against it.
> Phase 3 **must not break** the frozen Phase-1/Phase-2 contracts
> (`docs/INTERFACE_SPEC_PHASE1.md`, `docs/INTERFACE_SPEC_PHASE2.md`). Consistent with
> the approved plan `~/.claude/plans/i-d-like-to-set-velvet-clarke.md` (Phase 3 = imager).

Target package root: `/data/keeling/a/snesbitt/python/pf/src/pf/`

## Scope and frozen-contract guarantees

Phase 3 **only**:
1. Co-locates GPM GMI 89 GHz polarization-corrected temperature (PCT) onto the radar
   Ku FS swath, with the Utah/Liu-et-al. PF-database parallax correction.
2. Populates the two currently-NaN PCT columns:
   - FEATURE_SCHEMA col 35 `min_pct_85_89` (currently hard-NaN in `features.py`).
   - PIXEL_SCHEMA col 10 `pct_85_89` (currently NaN placeholder in `pixels.py`).
3. Adds two **in-memory-only** radar-native gating fields to the `Swath` dataclass
   (`storm_top`, `pia`) — these are NOT new Parquet columns.
4. Adds a new imager reader `src/pf/readers/gpm_gmi.py` and a new module
   `src/pf/colocate.py`, and wires imager handling through `granule.process_orbit`
   and `search.granules_for_orbit`.
5. Fixes one config bug: `SHORT_NAMES["GPM_GMI"]` must be `"GPM_1CGPMGMI"` (V07), not
   `"GPM_1CGMI"`.

The following remain **FROZEN and UNCHANGED**:
- `FEATURE_SCHEMA` (47 columns) and `PIXEL_SCHEMA` (13 columns). **No column
  added/removed/reordered/retyped.** Phase 3 only fills the two existing placeholder
  columns (feature 35, pixel 10).
- `build_feature_row`, `build_pixel_rows`, `write_orbit`, `process_orbit` **signatures**
  and return-dict key sets.
- `Swath` field list as of Phase 2 — Phase 3 **appends** two fields (`storm_top`, `pia`);
  `pct_85_89` already exists and keeps its name/shape/dtype.
- The partition layout in `catalog.py`. No `catalog.py` change.

## Ground truth (cached real granules in /data/scratch/a/snesbitt/_pf_probe/ — do NOT re-download)
- GMI: `1C.GPM.GMI.XCAL2016-C.20180630-S230252-E003525.024647.V07A.HDF5`; short_name `GPM_1CGPMGMI` (V07).
- DPR: `2A.GPM.DPR.V9-20211125.20180630-S230252-E003525.024647.V07A.HDF5`.
- GMI `S1/Latitude`,`S1/Longitude` = (2963,221); `S1/Tc` = (2963,221,9); `S1/incidenceAngle`=(2963,221,1); `S1/ScanTime/{Year..MilliSecond}`.
- Channel order confirmed by background-Tb (idx7 89V≈274.1 K > idx8 89H≈257.2 K): **PCT89 = 1.818·Tc[...,7] − 0.818·Tc[...,8]**. Tc `_FillValue=-9999.9` (in FILL_SENTINELS); still apply explicit `Tc<0→NaN`.
- DPR `FS/PRE/heightStormTop`=(7935,49) m (valid ≈760–17024); `FS/SLV/piaFinal`=(7935,49,2), Ku index 0 (use `_select_ku`), Ku range 0–28.7 dBZ.
- Existing `_ORBIT_RE = re.compile(r"\.(\d{5,6})\.V\d+", re.IGNORECASE)` parses `024647` from the GMI filename (no new regex).
- Repo `SHORT_NAMES["GPM_GMI"]` is wrong (`GPM_1CGMI`) → fix to `GPM_1CGPMGMI`.

## 1. config.py
```python
SHORT_NAMES["GPM_GMI"] = "GPM_1CGPMGMI"   # FIX (was GPM_1CGMI)
PCT_A: float = 1.818
PCT_B: float = 0.818
PCT89_V_IDX: int = 7
PCT89_H_IDX: int = 8
COLOCATE_RADIUS_M: float = 15000.0
PARALLAX_STORMTOP_M: float = 5000.0
PARALLAX_PIA_DBZ: float = 0.4
```

## 2. swath.py — two new in-memory gating fields (NOT Parquet columns)
Add to `Swath` dataclass and allocate NaN in `Swath.empty()`:
```python
    storm_top: np.ndarray   # float32 (nscan,nray) m; NaN fill
    pia: np.ndarray         # float32 (nscan,nray) Ku piaFinal dBZ; NaN fill
```
`Swath.empty()` supplies `storm_top=f2d(), pia=f2d()`. `pct_85_89` unchanged.

## 3. gpm_ku.py — populate storm_top & pia
Inside the open-file block (with Phase-2 bb_height/freezing_level), guarded by `has_path`:
```python
swath.storm_top = hdf5_util.read_float(f, s + "PRE/heightStormTop")        # (nscan,nray) m
swath.pia = self._select_ku(hdf5_util.read_float(f, s + "SLV/piaFinal"))   # (nscan,nray) dBZ
```

## 4. NEW reader gpm_gmi.py
`@dataclass(slots=True) Imager`: mission, orbit, short_name, granule_name, lat (float32 2-D), lon (float32 2-D), pct (float32 2-D 89 GHz PCT K, NaN where either chan missing), time (datetime64[ns] (nscan_gmi,)), **sc_lat (float32 (nscan_gmi,)), sc_lon (float32 (nscan_gmi,))** — per-scan spacecraft sub-satellite point, for the geometry-driven parallax direction.
Read sc_lat/sc_lon from `S1/SCstatus/SClatitude`, `S1/SCstatus/SClongitude` (float32 (nscan,)). (`S1/SCstatus/SCorientation` int16 — 0 or 180 yaw flag — may be read for documentation but the direction is derived geometrically from the sub-track, which captures orientation automatically.)
`GpmGmiReader` (NOT subclass of SwathReader — read returns Imager not Swath): `short_name="GPM_1CGPMGMI"`, `mission="GPM"`, `swath="S1"`, methods `read(path)->Imager`, `orbit_of(g)->int`.
read paths: `S1/Latitude`,`S1/Longitude` (read_float); `S1/Tc` (read_var float32, then `Tc[Tc<0]=NaN`); time from `S1/ScanTime/{Year,Month,DayOfMonth,Hour,Minute,Second,MilliSecond}` (reuse gpm_ku scan-time composition). PCT: `tc_v=Tc[...,PCT89_V_IDX]; tc_h=Tc[...,PCT89_H_IDX]; pct=(PCT_A*tc_v-PCT_B*tc_h).astype(float32)`. Self-check (raise ValueError if `nanmean(tc_v)<=nanmean(tc_h)`). `orbit_of` reuses GpmKuReader regex/approach.

## 5. NEW module colocate.py  (geometry-driven parallax direction)
```python
parallax_shift_geoloc(lat2d, lon2d, sc_lat, sc_lon) -> (lat_shift, lon_shift)
colocate_pct(swath, imager, cfg=pf.config) -> np.ndarray  # (nscan,nray) float32, NaN outside COLOCATE_RADIUS_M
```
**Parallax: move the GMI/TMI data one scan TOWARD the sub-satellite point** (user-directed; the elevated ice-scattering pixel is geolocated to the ellipsoid displaced away from nadir, so the surface column it overlies sits one parallax step toward the sub-satellite track). Applied only where the radar gate holds (storm_top>5 km AND PIA>0.4 dBZ). The direction MUST be derived from spacecraft geometry, NOT hardcoded, so it flips correctly across GPM yaw maneuvers (SCorientation 0↔180).

`parallax_shift_geoloc(lat2d, lon2d, sc_lat, sc_lon)`:
- For the orbit, determine which along-track neighbor is CLOSER to the spacecraft sub-point: compare mean great-circle distance from `sc_lat/sc_lon[s]` to mid-scan `pixel[s-1]` vs `pixel[s+1]` over interior scans. The "toward" neighbor = the closer one.
- If `pixel[s+1]` is closer (toward = later scan), shift = -1 (data at s gets coords of s+1; `np.roll(shift=-1,axis=0)`, replicate last row). If `pixel[s-1]` is closer, shift = +1 (replicate row 0). One uniform per-orbit roll in the geometry-determined toward direction.
- float32 copies; inputs not mutated; returns (lat_shift, lon_shift) same shape.
- For orbit 24647 (SCorientation=180, ascending): pixel[s+1] ~470 km vs pixel[s-1] ~496 km from SC → toward = s+1 → shift -1. (Empirically this maximizes the cold-PCT↔Ku-storm-top anti-correlation, r −0.75 vs −0.60 unshifted — consistent with the toward-sub-point physics.)

`colocate_pct`: pyresample `geometry.SwathDefinition` + `kd_tree.resample_nearest` (NEAREST; preserve cold-PCT min; reuse Precip_features/process_matching.py pattern).
1. target_def = SwathDefinition(swath.lon, swath.lat).
2. src_nopar = SwathDefinition(imager.lon, imager.lat); (lat_p,lon_p)=parallax_shift_geoloc(imager.lat, imager.lon, imager.sc_lat, imager.sc_lon); src_par = SwathDefinition(lon_p, lat_p).
3. pct_nopar / pct_par = resample_nearest(src_*, imager.pct, target_def, radius_of_influence=COLOCATE_RADIUS_M, fill_value=np.nan).
4. gate = isfinite(storm_top)&(storm_top>PARALLAX_STORMTOP_M)&isfinite(pia)&(pia>PARALLAX_PIA_DBZ).
5. return np.where(gate, pct_par, pct_nopar).astype(float32).

## 6. granule.py — optional imager wiring
After `swath = reader_cls().read(path)` and BEFORE `label_rpf`: if `granule_handles.get("imager")` present, login(netrc) if remote, `_download_with_retry` into same tmpdir, `GpmGmiReader().read`, `swath.pct_85_89 = colocate.colocate_pct(swath, imager, cfg)`. Wrap in try/except that swallows imager errors (best-effort; PCT stays NaN, radar PFs still produced). Signature & return keys UNCHANGED. granule_handles may now carry "imager".

## 7. features.py — populate col 35
Replace `"min_pct_85_89": float("nan")` with member-min:
```python
member_pct = swath.pct_85_89[member]
min_pct_85_89 = float(np.nanmin(member_pct)) if np.isfinite(member_pct).any() else float("nan")
```
Signature unchanged; dict still 47 keys.

## 8. pixels.py — populate col 10
`pct_85_89 = swath.pct_85_89[scan_idx, ray_idx]`; per-pixel `"pct_85_89": float(pct_85_89[k])`. 13 keys unchanged.

## 9. search.py — resolve imager
`granules_for_orbit(...)` gains `*, imager_short_name=None`; defaults to `SHORT_NAMES["GPM_GMI"]`; search same temporal window; group_by_orbit with GpmGmiReader; return `{"radar":..., "imager":...}` (imager None if no match / search fails — wrap in try/except). Radar behavior unchanged.

## Verification hooks
- FEATURE_SCHEMA / PIXEL_SCHEMA byte-identical (no columns added; storm_top/pia in-memory only).
- col 35 populated on orbit 024647 (with imager); finite values in ~70–300 K; NaN where no GMI sample.
- pixel col 10 finite for co-located member pixels; per-feature min(pixel pct) ≈ feature min_pct_85_89.
- Graceful no-imager: process_orbit with {"radar":...} only reproduces Phase-2 output (PCT all NaN), counts/status unchanged.
- Nearest preserves minima (co-located min equals an actual GMI sample).
- Parallax gating: PCT differs gated vs ungated only where storm_top>5000 & pia>0.4.
- Parallax sign calibrated so min-PCT moves toward max-dBZ core.
- config: SHORT_NAMES["GPM_GMI"]=="GPM_1CGPMGMI".
- GpmGmiReader.read raises ValueError if nanmean(89V)<=nanmean(89H).
