# PF Phase-4 Interface Specification — TRMM (PR + TMI) registration

> Authoritative contract. Code agents implement strictly against this; the verification
> agent tests against it. Phases 1–3 (GPM) are frozen. Phase 4 registers mission "TRMM"
> by thin subclassing/parameterization of the existing readers — no schema/algorithm changes.

## Key insight
GPM-reprocessed TRMM products (PR `GPM_2APR` V07, TMI `GPM_1CTRMMTMI` V07) are HDF5 in the
SAME structure as their GPM counterparts. Phase 4 = parameterization + thin subclassing of
`GpmKuReader`/`GpmGmiReader`. No new decoding, no schema changes, no algorithm changes.

## Ground truth (cached, orbit 522, 1997-12-30 — Zipser-2006 N-Argentina storm; in /data/scratch/a/snesbitt/_pf_probe/)
- PR `2A.TRMM.PR...000522.V07A.HDF5` — swath `FS`, identical paths to GPM 2A-DPR, 176 bins @ **125 m (confirmed)**. Reflectivity 2-D `(9139,49)` / 3-D `(9139,49,176)` with NO freq axis; `piaFinal` 2-D `(9139,49)`.
- TMI `1C.TRMM.TMI...000522.V07A.HDF5` — 85.5 GHz V/H in group **S3**: `S3/Tc (2885,208,2)` V=0/H=1, `S3/Latitude`,`S3/Longitude`,`S3/ScanTime/*`,`S3/SCstatus/{SClatitude,SClongitude,SCorientation}`. PCT=1.818·V−0.818·H. Validated: PCT min **69.1 K** at (−23.41,−57.57), co-located PR **59.7 dBZ** near-surface, **17688 m** storm top.
- GPM_1CTRMMTMI has V07 (HDF5) and V08 (.nc); prefer V07. h5py reads both.

## 1. NEW src/pf/readers/trmm_pr.py
```python
from pf.readers.gpm_ku import GpmKuReader
class TrmmPrReader(GpmKuReader):
    short_name = "GPM_2APR"
    mission = "TRMM"
```
Override ONLY `short_name`, `mission`. Inherit everything else: `swath="FS"`, `read()` (all FS/* paths exist identically), `_select_ku` (verified safe no-op: TRMM arrays are 2-D or 3-D-with-shape[-1]=176, guard never fires), `_read_scan_time`, `_height_3d` (176 bins @125 m identical), `orbit_of`/`_ORBIT_RE` (parses 522 from `000522`). Returns Swath(mission="TRMM", short_name="GPM_2APR", orbit=522). NO edit to gpm_ku.py needed.

## 2a. EDIT src/pf/readers/gpm_gmi.py (parameterize — GPM behavior byte-identical)
Add class attrs to `GpmGmiReader`: `pct_swath="S1"`, `pct_v_idx=7`, `pct_h_idx=8`. In `read()`:
- `s = f"{self.pct_swath}/"` (was `self.swath`)
- `tc_v = tc[..., self.pct_v_idx]`, `tc_h = tc[..., self.pct_h_idx]` (was config.PCT89_*_IDX)
- self-check ValueError message uses `self.pct_v_idx`/`self.pct_h_idx`
- PCT formula `config.PCT_A*tc_v - config.PCT_B*tc_h` UNCHANGED.
All else unchanged (reads Latitude/Longitude/Tc/ScanTime/SCstatus from `s`, V>H self-check, tc<0→nan, Imager construction, orbit_of). For GPM pct_swath=="S1" → identical. Imager dataclass UNCHANGED.

## 2b. NEW src/pf/readers/trmm_tmi.py
```python
from pf.readers.gpm_gmi import GpmGmiReader
class TrmmTmiReader(GpmGmiReader):
    short_name = "GPM_1CTRMMTMI"
    mission = "TRMM"
    swath = "S3"
    pct_swath = "S3"
    pct_v_idx = 0   # 85.5V
    pct_h_idx = 1   # 85.5H
```
Inherits read() (now reads S3/*), V>H self-check, PCT_A/PCT_B, orbit_of, Imager. Validated PCT min ≈69 K.

## 3. EDIT src/pf/granule.py
```python
from pf.readers.trmm_pr import TrmmPrReader
from pf.readers.trmm_tmi import TrmmTmiReader
_RADAR_READERS  = {"GPM": GpmKuReader,  "TRMM": TrmmPrReader}
_IMAGER_READERS = {"GPM": GpmGmiReader, "TRMM": TrmmTmiReader}
```
In process_orbit imager block, select `imager_cls = _IMAGER_READERS.get(mission)` instead of hardcoded GpmGmiReader; if None skip silently. Keep the best-effort try/except. process_orbit signature, granule_handles shape, colocate_pct call all UNCHANGED (S3 supplies sc_lat/sc_lon → parallax reused verbatim).

## 4. EDIT src/pf/search.py (mission-aware)
```python
from pf.readers.trmm_pr import TrmmPrReader
from pf.readers.trmm_tmi import TrmmTmiReader
_MISSION_PRODUCTS = {
  "GPM":  (GpmKuReader,  GpmGmiReader,  SHORT_NAMES["GPM_KU"],  SHORT_NAMES["GPM_GMI"]),
  "TRMM": (TrmmPrReader, TrmmTmiReader, SHORT_NAMES["TRMM_PR"], SHORT_NAMES["TRMM_TMI"]),
}
```
In `granules_for_orbit`: `radar_cls, imager_cls, radar_default, imager_default = _MISSION_PRODUCTS[mission]`; default short_names from these; group radar with `radar_cls()` and imager with `imager_cls()` (was hardcoded GpmKuReader/GpmGmiReader). Signature & return shape UNCHANGED. TMI V07 preference: filter imager granules whose basename contains `.V07` before group_by_orbit, fall back to all if empty (search-layer only; no reader change).

## 5. EDIT src/pf/config.py
```python
SHORT_NAMES = {
  "GPM_KU": "GPM_2ADPR", "GPM_GMI": "GPM_1CGPMGMI",
  "TRMM_PR": "GPM_2APR",        # was TRMM_2A25
  "TRMM_TMI": "GPM_1CTRMMTMI",  # was TRMM_2A12
}
```
Drop unused "TRMM_2A23". Keep PCT_A/PCT_B. Keep PCT89_V_IDX/H_IDX in config for back-compat (no longer read by gpm_gmi.py, which uses self.pct_*_idx). MISSION_CODE unchanged.

## 6. FROZEN
FEATURE_SCHEMA(47)/PIXEL_SCHEMA(13) unchanged; build_feature_row/build_pixel_rows/write_orbit/process_orbit signatures unchanged; colocate_pct/parallax_shift_geoloc reused as-is; Imager fields unchanged; feature_id unchanged (TRMM=1, orbit 522 in range). No new columns/dtype changes.

## 7. Verification (cached granules; pass local paths as handles — no downloads)
1. process_orbit("TRMM", 522, {"radar": <2A.TRMM.PR...522>, "imager": <1C.TRMM.TMI...522>}) → status ok, n_features>0.
2. Feature containing (−23.41,−57.57): min_pct_85_89 ≈ 69 K, max_near_sfc_dbz ≈ 59, high echo top (~17.7 km).
3. feature_id.decode(fid) → ("TRMM", 522, label).
4. TrmmPrReader().read → Swath(TRMM, GPM_2APR, 522), dbz_3d (9139,49,176). TrmmTmiReader().read → Imager(TRMM), pct from S3, min ≈69 K.
5. GPM regression: process_orbit("GPM", 24647, {radar,imager}) identical to Phase 3 (parameterization is a no-op for GPM: S1, idx 7/8).
6. granules_for_orbit("TRMM",522,...) resolves GPM_2APR/GPM_1CTRMMTMI; ("GPM",24647) still resolves GPM_2ADPR/GPM_1CGPMGMI.
