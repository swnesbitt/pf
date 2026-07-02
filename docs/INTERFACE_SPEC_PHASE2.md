# PF Phase-2 Interface Specification (classify + pixel table + radar schema fill)

> Authoritative contract produced by the development/architecture agent. Code-writing
> agents implement strictly against this; the verification agent tests against it.
> Phase 2 **must not break** the frozen Phase-1 contract
> (`docs/INTERFACE_SPEC_PHASE1.md`). Consistent with the approved plan
> `~/.claude/plans/i-d-like-to-set-velvet-clarke.md`.

Target package root: `/data/keeling/a/snesbitt/python/pf/src/pf/`

## Scope and frozen-contract guarantees

Phase 2 **only**:
1. Populates `swath.bb_height` and `swath.freezing_level` in the GPM-Ku reader
   (Phase-1 NaN placeholders).
2. Adds `src/pf/classify.py` (new) and wires it into `build_feature_row` to fill
   FEATURE_SCHEMA columns **36–47**.
3. Adds `src/pf/pixels.py` (new) producing PIXEL_SCHEMA rows.
4. Wires the pixel table through `granule.process_orbit` → `catalog.write_orbit`.

The following remain **FROZEN and UNCHANGED**:
- `FEATURE_SCHEMA` (47 columns; order/dtypes) and `PIXEL_SCHEMA` (13 columns) in
  `features.py`. **No column is added, removed, reordered, or retyped.**
- Column 35 `min_pct_85_89` stays `NaN` (Phase 3).
- The `Swath` dataclass field list (it already declares `bb_height`,
  `freezing_level`; Phase 2 merely fills them with real data instead of NaN).
- `build_feature_row` **signature** is unchanged.
- `write_orbit` **signature** and partition layout are unchanged.
- `process_orbit` **signature** and return-dict keys are unchanged.

---

## Findings from reading the actual source / granule (flag where assumptions changed)

1. **`bb_height` / `freezing_level` fills are already covered by
   `FILL_SENTINELS`.** `config.FILL_SENTINELS = (-9999.9, -9999.0, -9999, -99.0,
   -1111.1, -1111)`. `heightBB` fill `-1111.1` and `heightZeroDeg` fill `-9999.9`
   are both present, so `hdf5_util.read_float` (which calls `decode_fill` with
   `atol=0.05`) masks them to `NaN` with **no special handling needed**.
   Verified against the cached granule:
   - `FS/CSF/heightBB`  shape `(7935, 49)` float32, min `-1111.1`, max `5730.4`.
   - `FS/VER/heightZeroDeg` shape `(7935, 49)` float32, min `-9999.9`, max `5134.6`.
   Valid maxima are far from every sentinel, so `atol=0.05` causes no false masking.

2. **Authoritative member-pixel rain-type codes (verified in `gpm_ku.py`
   lines 117–123).** Reader computes `rain_type = typePrecip // 10_000_000`, then
   `np.where(rain_type < 0, -1, rain_type).astype(np.int8)`. Therefore the
   **only** values that ever appear in `swath.rain_type` are:
   - `-1` = none/missing (fill `-1111 // 1e7 == -1`, and any negative),
   - `1`  = stratiform,
   - `2`  = convective,
   - `3`  = other.
   `0` is theoretically reachable (`typePrecip` in `0..9_999_999`) but does not
   occur for precipitating pixels; classify must treat only `==2` as convective
   and `==1` as stratiform (everything else — `3`, `-1`, `0` — is neither).
   The Phase-1 spec's "0=none" note is superseded: in practice "none" is `-1`.

3. **`Swath.empty` already allocates `bb_height`/`freezing_level` as `f2d()` =
   `(nscan, nray)` float32 NaN.** The reader just needs two assignments. No
   dataclass change.

4. **`catalog.write_orbit` ALREADY fully implements the pixel write**
   (lines 167–175): it builds `pixels_path`, and when `pixels_df` is non-empty it
   casts via `pa.Table.from_pandas(pixels_df, schema=PIXEL_SCHEMA, ...)` and writes
   atomically under `pixels/mission=/year=/month=/orbit=NNNNNN.parquet`, using the
   **same** `year/month` derived from the FEATURE table's mean time. `_coerce_time_us`
   is already applied to `pixels_df` and is a **no-op** (pixel table has no `time`
   column). **No change to `catalog.py` is required** — see §6. This changed the
   assumption that the pixel write might be missing.

5. **`PIXEL_SCHEMA` dtype note:** `ray` is `pa.int16()` and `scan` is `pa.int32()`
   (features.py lines 116–132). Pixel builder must emit `ray` as int16-range and
   `scan` as int32. `rain_type` is `pa.int8()`. There is **no `time`** column in
   PIXEL_SCHEMA.

---

## 1. Reader extension — `src/pf/readers/gpm_ku.py`

Inside `GpmKuReader.read`, within the open-file block (after the surface-type
block, ~line 131, still inside `with h5py.File(...)`), add **two assignments**:

```python
# --- bright-band height (Phase 2) --------------------------------
bb_path = s + "CSF/heightBB"
if hdf5_util.has_path(f, bb_path):
    swath.bb_height = hdf5_util.read_float(f, bb_path)

# --- freezing level / 0 deg height (Phase 2) ---------------------
fz_path = s + "VER/heightZeroDeg"
if hdf5_util.has_path(f, fz_path):
    swath.freezing_level = hdf5_util.read_float(f, fz_path)
```

Contract:
- `swath.bb_height` ← `FS/CSF/heightBB` via `hdf5_util.read_float` →
  `float32`, shape `(nscan, nray)`, fill `-1111.1` → `NaN` (automatic).
- `swath.freezing_level` ← `FS/VER/heightZeroDeg` via `hdf5_util.read_float` →
  `float32`, shape `(nscan, nray)`, fill `-9999.9` → `NaN` (automatic).
- `has_path` guard: if a field is absent the Phase-1 `NaN` allocation from
  `Swath.empty` remains (so the reader never raises on an older granule).
- **No `_select_ku`** — these are 2-D `(nscan, nray)` already (no frequency axis).
- Affects `Swath` fields `bb_height`, `freezing_level` (previously NaN).
  Downstream: FEATURE cols 44/45 and PIXEL col `bb_height`.

---

## 2. New module — `src/pf/classify.py`

Public function (the ONLY public symbol required):

```python
def classify_feature(
    swath: "pf.swath.Swath",
    labeled: np.ndarray,        # int32, (nscan, nray), from label_rpf
    local_label: int,
    area_km2: float,            # feature area (km^2) from label_rpf 'kept'
    volrain_total: float,       # from build_feature_row (Sigma rain*area over member)
    cfg: ModuleType = pf.config,
) -> dict:
```

Returns a dict with **EXACTLY** these 12 keys (FEATURE_SCHEMA cols 36–47),
all native Python `float`/`bool`/`str` (never numpy scalars):

| Key (col) | Type | Definition |
|---|---|---|
| `conv_area_km2` (36) | float | `sum(pixel_area[conv])` |
| `strat_area_km2` (37) | float | `sum(pixel_area[strat])` |
| `conv_area_frac` (38) | float | `conv_area_km2/area_km2` if `area_km2>0` else `NaN` |
| `strat_area_frac` (39) | float | `strat_area_km2/area_km2` if `area_km2>0` else `NaN` |
| `conv_rain_frac` (40) | float | `volrain_conv/volrain_total` if `volrain_total>0` else `NaN` |
| `strat_rain_frac` (41) | float | `volrain_strat/volrain_total` if `volrain_total>0` else `NaN` |
| `volrain_conv` (42) | float | `nansum(near_sfc_rain[conv]*pixel_area[conv])` |
| `volrain_strat` (43) | float | `nansum(near_sfc_rain[strat]*pixel_area[strat])` |
| `mean_bb_height` (44) | float | `nanmean(bb_height[member])`; `NaN` if all-NaN/empty |
| `mean_freezing_level` (45) | float | `nanmean(freezing_level[member])`; `NaN` if all-NaN/empty |
| `is_mcs` (46) | bool | `bool(area_km2 >= cfg.MCS_AREA_KM2)` (radar-only, 2000 km²) |
| `feature_class` (47) | str | enum, see rules below |

### Masks (exact)
```
member = (labeled == local_label)                      # (nscan, nray) bool
rt     = swath.rain_type                                # int8, values in {-1,1,2,3}
conv   = member & (rt == 2)
strat  = member & (rt == 1)
```

### Field-by-field NaN / division-guard contract
- `conv_area_km2 = float(np.nansum(swath.pixel_area[conv]))`. If no conv pixels
  the masked array is empty → `np.nansum([]) == 0.0`, so `conv_area_km2 = 0.0`
  (NOT NaN). Same for `strat_area_km2`.
- `conv_area_frac`: `float(conv_area_km2/area_km2)` only when `area_km2 > 0`,
  else `float("nan")`. (area_km2 from label_rpf is always > 0 in practice, but
  guard anyway.) `strat_area_frac` analogous.
- `volrain_conv = float(np.nansum(swath.near_sfc_rain[conv] * swath.pixel_area[conv]))`.
  Empty mask → `0.0`. NaN rain/area entries are dropped by `nansum`
  (treated as 0). `volrain_strat` analogous.
- `conv_rain_frac`: `float(volrain_conv/volrain_total)` when
  `volrain_total > 0` (and finite), else `float("nan")`. NOTE: `volrain_total`
  is passed in; if it is `NaN` (member had no finite rain) the guard
  `volrain_total > 0` is `False` → `NaN`. `strat_rain_frac` analogous.
- `mean_bb_height`: take `vals = swath.bb_height[member]`; if
  `np.isfinite(vals).any()` then `float(np.nanmean(vals))` else `float("nan")`.
  (Use the finite-any guard so an all-NaN slice does not emit a numpy
  RuntimeWarning.) `mean_freezing_level` analogous on `swath.freezing_level`.
- `is_mcs`: `bool(float(area_km2) >= float(cfg.MCS_AREA_KM2))`. Always a real
  Python bool (never None) — col 46 is nullable in the schema but Phase 2
  always writes a concrete bool.

### `feature_class` enum (exact, evaluated top-to-bottom)
```
if is_mcs:                       feature_class = "MCS"
elif conv_area_km2 > 0:          feature_class = "sub_MCS_conv"
elif strat_area_km2 > 0:         feature_class = "stratiform_only"
else:                            feature_class = "weak"
```
Returns one of the literal strings `{"MCS","sub_MCS_conv","stratiform_only","weak"}`.
Always a non-null `str`. (A feature that is MCS by area but has no conv pixels is
still `"MCS"` — area dominates.)

### Numerical guarantees
- All 10 float keys are finite-or-NaN Python floats; the two area/volrain
  *quantities* default to `0.0` (not NaN) on empty masks; the four *fractions*
  are NaN when their guard denominator is `<= 0` or non-finite.
- No exception is raised for empty `conv`/`strat`/`member`.

---

## 3. `features.py` change — wire classify into `build_feature_row`

**Signature UNCHANGED:** `build_feature_row(swath, labeled, local_label, area_km2, edge) -> dict`.

Integration point: after `volrain_total` is computed (current line ~352) and the
return dict is assembled, replace the **12 hard-coded placeholders for cols
36–47** (currently lines 412–423: `conv_area_km2 ... feature_class`) with a merge
of `classify.classify_feature(...)`:

```python
from pf import classify   # add to imports

# ... after volrain_total is computed:
class_cols = classify.classify_feature(
    swath, labeled, local_label, float(area_km2), float(volrain_total),
)
```

Then in the returned dict, keep `"min_pct_85_89": float("nan")` (col 35, Phase 3)
exactly as-is, and replace the 12 placeholder entries (36–47) with the values from
`class_cols`. Two equivalent implementations are acceptable; the contract is the
**returned row**:
- Either spread `**class_cols` in place of the 12 literals, OR
- assemble `row = {...cols 1–35...}; row.update(class_cols); return row`.

Hard requirements:
- The returned dict still has **exactly the 47 FEATURE_SCHEMA keys**, same order
  is not required (DataFrame is built with explicit `columns=`), but all 12 keys
  from `class_cols` MUST be present and overwrite the old NaN/None.
- `min_pct_85_89` MUST remain `float("nan")`.
- `is_mcs` becomes a real bool; `feature_class` becomes a real str
  (previously `None`/`None`).
- `volrain_total` passed to classify is the **same** value written to col 28
  (single source of truth — do not recompute inside classify).

Affected FEATURE_SCHEMA columns: 36–47 (now populated); 35 unchanged.

---

## 4. New module — `src/pf/pixels.py`

Public function (only required symbol):

```python
def build_pixel_rows(
    swath: "pf.swath.Swath",
    labeled: np.ndarray,        # int32 (nscan, nray)
    local_label: int,
    mission: str,
    orbit: int,
) -> list[dict]:
```

Returns **one dict per member pixel** (`member = labeled == local_label`),
each with **EXACTLY** the 13 PIXEL_SCHEMA keys, in the schema's dtype domain:

| Key | dtype (schema) | Source |
|---|---|---|
| `feature_id` | int64 | `pf.feature_id.encode(mission, orbit, local_label)` (one value, repeated) |
| `mission` | string | `str(mission)` |
| `orbit` | int32 | `int(orbit)` |
| `scan` | int32 | member scan index `s` (row), from `np.nonzero(member)[0]` |
| `ray` | int16 | member ray index `r` (col), from `np.nonzero(member)[1]` |
| `lat` | float32 | `swath.lat[s, r]` |
| `lon` | float32 | `swath.lon[s, r]` |
| `near_sfc_dbz` | float32 | `swath.near_sfc_dbz[s, r]` |
| `near_sfc_rain` | float32 | `swath.near_sfc_rain[s, r]` |
| `pct_85_89` | float32 | `float("nan")` — Phase-3 placeholder |
| `rain_type` | int8 | `int(swath.rain_type[s, r])` (values in {-1,1,2,3}) |
| `pixel_area_km2` | float32 | `swath.pixel_area[s, r]` |
| `bb_height` | float32 | `swath.bb_height[s, r]` (NaN where fill, from §1) |

Contract:
- Iterate member pixels in **row-major (scan-major) order** as produced by
  `np.nonzero(member)` (deterministic, matches `npixels = member.sum()`).
- `feature_id` is computed **once** via `feature_id.encode` and identical on
  every row → the feature↔pixel join recovers exactly `npixels` rows per feature
  (round-trip invariant from the plan's verification section).
- Each value cast to a native Python scalar (`int`/`float`) so the downstream
  `pd.DataFrame` + `pa.Table.from_pandas(..., schema=PIXEL_SCHEMA)` cast is
  lossless. `NaN` allowed for the float fields.
- Empty member (should not happen for a kept label) → returns `[]`.
- Vectorized construction is permitted as long as the returned object is a
  `list[dict]` with the 13 keys.

Affects: PIXEL_SCHEMA (all 13 columns). `pct_85_89` is the only Phase-3
placeholder (NaN).

---

## 5. `granule.py` change — build and pass `pixels_df`

In `process_orbit`, replace the Phase-1 block (current lines 120–135):

```python
rows = [...]                       # unchanged: build_feature_row per kept feature
features_df = pd.DataFrame(rows, columns=[f.name for f in _features.FEATURE_SCHEMA])
pixels_df = None                   # <-- REMOVE
write_orbit(features_df, pixels_df, mission, cfg.PF_ROOT)
result["n_pixels"] = 0             # <-- REMOVE
```

with:

```python
from pf import pixels as _pixels   # add import

pixel_rows: list[dict] = []
for (local_label, _area_km2) in kept:
    pixel_rows.extend(
        _pixels.build_pixel_rows(swath, labeled, local_label, mission, int(orbit))
    )

if pixel_rows:
    pixels_df = pd.DataFrame(
        pixel_rows, columns=[f.name for f in _features.PIXEL_SCHEMA]
    )
else:
    pixels_df = None

write_orbit(features_df, pixels_df, mission, cfg.PF_ROOT)

result["n_features"] = len(rows)
result["n_pixels"] = 0 if pixels_df is None else len(pixels_df)
result["status"] = "ok"
return result
```

Contract:
- `pixels_df` columns ordered to PIXEL_SCHEMA via explicit `columns=[f.name ...]`
  (mirrors the existing features_df construction).
- Concatenation order: features iterate over `kept` (sorted by `local_label`
  ascending from `label_rpf`); pixel rows follow the same feature order. Within a
  feature, pixels are in `np.nonzero` row-major order.
- `result["n_pixels"] = len(pixels_df)` (0 when no pixels — only possible if
  `kept` is empty, but that path already returns `status="empty"` earlier, so in
  the `ok` path `n_pixels` will be `>= 1`).
- `process_orbit` **signature and return-dict key set are UNCHANGED**
  (`orbit`, `n_features`, `n_pixels`, `status`[, `error`]).
- All other ordering (login → download → read → label → finally rmtree) and the
  exception-to-`failed` contract are unchanged.

---

## 6. `catalog.py` — CHECK (no change required)

Verified against the current source:
- `write_orbit` already casts the pixel frame:
  `pa.Table.from_pandas(pixels_df, schema=PIXEL_SCHEMA, preserve_index=False)`
  (line 172) and writes atomically via `_write_atomic`.
- Pixel path:
  `{root}/pixels/mission={M}/year={YYYY}/month={MM}/orbit={NNNNNN}.parquet`
  with `year`/`month` derived from the **FEATURE** table's mean `time`
  (lines 145–151, 168–169) — the pixel table inherits the same partition keys,
  which is correct (pixels have no `time` column).
- `_coerce_time_us(pixels_df)` (line 171) is a documented **no-op** for pixels
  (no `time` column) — safe.
- Non-empty `pixels_df` triggers the write; `None`/empty still returns the
  target `pixels_path`. With Phase 2 passing a real frame, the file is now
  written.

**Action: none.** If the verification agent finds the pixel file is not written
for a non-empty frame, the bug is in `granule.py` wiring (§5) or the
`build_pixel_rows` output dtypes (§4), not in `catalog.py`.

`PIXEL_SCHEMA` is imported in `catalog.py` (line 27) from `pf.features` — the
single authoritative definition. Do not redefine it.

---

## Cross-cutting / verification hooks (for the testing agent)

- **Schema invariance:** assert `FEATURE_SCHEMA` and `PIXEL_SCHEMA` are byte-for-byte
  identical to Phase 1 (column names, order, types). Phase 2 adds no fields.
- **Join round-trip:** for each feature, `count(pixels WHERE feature_id == fid)`
  == that feature's `npixels` (col 6).
- **Class/area consistency:** every `is_mcs == True` feature has
  `area_km2 >= 2000` and `feature_class == "MCS"`.
- **Fraction bounds:** where finite, `0 <= conv_area_frac <= 1`,
  `0 <= strat_area_frac <= 1`, and `conv_area_frac + strat_area_frac <= 1`
  (the remainder is `rain_type in {-1,0,3}` pixels).
- **Rain-frac guard:** features with `volrain_total <= 0` (or NaN) have
  `conv_rain_frac` and `strat_rain_frac` == NaN.
- **bb/freezing fill:** `mean_bb_height` / `mean_freezing_level` are NaN for
  features whose members are entirely over no-bright-band / fill regions;
  finite and within `~0..6000 m` otherwise.
- **Col 35 unchanged:** `min_pct_85_89` is NaN for every row.
- **Pixel dtypes:** `ray` fits int16, `scan` fits int32, `rain_type` in
  `{-1,1,2,3}`, `pct_85_89` all NaN.
