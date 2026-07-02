# Plan — New Gridded Convective / Intensity / DSD Metrics

_Draft 2026-06-20. Adds a new Stage-1 grid table (`metrics.parquet`) carrying
convective echo-top heights, convective-pixel frequency, heavy-rain-rate
frequencies, and mean DSD epsilon — then threads them through Stage-2
climatology → Stage-3 tiles → HF Atlas._

## Goal

Add to the 0.05° × UTC-hour grid product, **per cell**:

1. **Convective echo-top height** at **20, 30, 40 dBZ** (mean over convective pixels).
2. **Frequency of convective pixels** (convective count ÷ views).
3. **Frequency of pixels > 25, 50, 75, 100 mm/hr** near-surface rain rate (4 fields).
4. **Mean epsilon** (the 2A DPR/PR DSD adjustment parameter), **separately for
   convective and stratiform pixels**.
5. **Convective & stratiform rain totals and raining-pixel counts** (→ conv/strat
   rain accumulation, frequency, conditional intensity).

All eventually published to the HF Atlas + Zarr dataset.

**Important — items (2) and (5) are nearly free.** Convective/stratiform rain
totals and raining-pixel counts (and thus conv/strat freq, rate, intensity, and
the convective-pixel frequency in (2)) are **already in the validated
`rain.parquet`** — it's stratified by `raintype` (cross-cut by size/echotop
class). So they come from a **Stage-2 re-aggregation of existing data, with NO
re-grid**, and can ship before the heavy pass. Only the *new physics* — per-pixel
convective echo-tops, heavy-rain thresholds, and epsilon — needs the new
`metrics.parquet` and the full orbit re-read.

## Data-availability findings (from code exploration)

| quantity | source | status |
|---|---|---|
| convective flag | `swath.rain_type == 2` (`FS/CSF/typePrecip // 1e7`) | already loaded |
| rain-rate thresholds | `swath.near_sfc_rain` (mm/hr, `FS/SLV/precipRateNearSurface`) | already loaded |
| echo-top 20/30/40 | `swath.dbz_3d` + `swath.height_3d` (176 bins, 125 m, slant-corrected) | loaded, but only a **per-feature** scalar is computed today (`echotop_qc.feature_echo_tops`); need a **per-pixel** version |
| epsilon | `FS/SLV/epsilon` — confirmed present **both missions** (Phase 0) | **reader change required** |

**Phase 0 result (epsilon verified 2026-06-20 on real granules):** `FS/SLV/epsilon`
exists in **both** GPM 2AKu V07 and TRMM 2APR V07, identical layout —
**`(nscan, nray, 176)` float32, PER-BIN**, `_FillValue=-9999.9`. Values cluster
near 1.0 (GPM 0.24–1.97; TRMM 0.34–3.79) — a multiplicative DSD adjustment.
Because it's per-bin, the reader must reduce the profile to one value per pixel:
**use the near-surface clutter-free gate** (the same bin index already used for
`near_sfc_dbz`) → `swath.epsilon` `(nscan, nray)`. (`FS/SLV/paramDSD`
`(…,176,2)` is also present if Dm/Nw ever wanted.)

Key facts:
- Existing grid keys: `views.parquet`=(lat_bin,lon_bin,hour,n_views);
  `rain.parquet`=(…,size_class,echotop_class,raintype,rain_sum,raining_count).
- `raintype` already separates convective (`==1` in grid space) — but the new
  metrics need per-pixel echo-top/epsilon/thresholds that don't fit that table's
  keys, so they go in a **new table**.
- Echo-top geometry: per-pixel top = highest bin where `dbz_3d ≥ thr`, value =
  `height_3d` there. QC gates already loaded (`bin_mirror_image`,
  `bin_clutter_bottom`, outer-ray sidelobe) — apply the same Hirose-2023 gates
  per pixel; restrict to feature-member convective pixels for context.

## Design — a new `metrics.parquet`, keyed (lat_bin, lon_bin, hour)

One new sparse table per month dir, alongside `views`/`rain` (untouched). Same
hour axis → diurnal cycle preserved. **All means are stored as sum + count** so
Stage-2/3 can pool correctly (esp. COMBINED = (ΣGPM+ΣTRMM)/(nGPM+nTRMM), never a
mean-of-means).

Columns (all keyed by lat_bin int16, lon_bin int16, hour int8):

| column | dtype | meaning |
|---|---|---|
| `et20_sum`,`et20_n` | f64,int64 | Σ & count of conv-pixel 20 dBZ echo-top (m) |
| `et30_sum`,`et30_n` | f64,int64 | … 30 dBZ (only pixels reaching 30 dBZ) |
| `et40_sum`,`et40_n` | f64,int64 | … 40 dBZ (only pixels reaching 40 dBZ) |
| `cnt_gt25`,`cnt_gt50`,`cnt_gt75`,`cnt_gt100` | int64 | # pixels with near_sfc_rain ≥ 25/50/75/100 mm/hr |
| `eps_conv_sum`,`eps_conv_n` | f64,int64 | Σ & count of epsilon over **convective** pixels |
| `eps_strat_sum`,`eps_strat_n` | f64,int64 | Σ & count of epsilon over **stratiform** pixels |
| `nw_conv_sum`,`nw_conv_n` | f64,int64 | Σ & count of Nw (dBNw) over **convective** pixels |
| `nw_strat_sum`,`nw_strat_n` | f64,int64 | Σ & count of Nw over **stratiform** pixels |
| `dm_conv_sum`,`dm_conv_n` | f64,int64 | Σ & count of Dm (mm) over **convective** pixels |
| `dm_strat_sum`,`dm_strat_n` | f64,int64 | Σ & count of Dm over **stratiform** pixels |

(22 metric columns total. Nw=paramDSD[...,0] dBNw, Dm=paramDSD[...,1] mm, both
near-surface gate, conv/strat split like epsilon.)

Denominator for all frequencies = `n_views` from the existing `views.parquet`
(same cell+hour key) → frequencies are consistent with the published `views`.

`conv_count`/`strat_count` are **not** duplicated here — convective & stratiform
raining-pixel counts and rain totals are read from the existing `rain.parquet`
(`raining_count`/`rain_sum` summed over size_class+echotop_class for raintype
conv vs strat) in Stage-2. This avoids double-storing validated data and lets
those tiles ship without the re-grid.

Open design choices (defaults chosen; flag to confirm):
- **Echo-top population**: convective pixels only (matches "convective echo top"). ✔ default.
- **Rain-rate thresholds**: all precipitating pixels (heavy-rain frequency), not conv-only. ✔ default (could add conv-only later).
- **Epsilon**: split convective vs stratiform (per request). ✔.
- **Conv/strat rain & pixels**: from existing `rain.parquet` (no re-grid). ✔.
- **Thresholds**: `≥` (inclusive). ✔.
- Keep keyed by (lat,lon,hour) only — **not** stratified by size/echotop/raintype
  (keeps the table small; conv/strat split comes from `rain.parquet`).

## Implementation steps

### Phase 0 — verify epsilon on a real granule (read-only) ✅ DONE
**Verdict:** `FS/SLV/epsilon`, `(nscan,nray,176)` per-bin float32, fill −9999.9,
both missions. Reduction decision: **near-surface clutter-free gate** →
`swath.epsilon (nscan,nray)`. Probe script: `/data/scratch/a/snesbitt/eps_probe.py`.

### Phase 1 — reader + per-pixel compute (`src/pf/`) ✅ DONE + VALIDATED
- `swath.py`: added `epsilon` 2-D field (+ `empty()`).
- `readers/gpm_ku.py`: read `FS/SLV/epsilon` (per-bin), `_select_ku`, reduce to the
  near-surface clutter-free gate via `_near_surface_gate(cube, bin_clutter_bottom)`
  → `swath.epsilon (nscan,nray)`, fills→NaN. TRMM inherits (subclass). Absent → NaN.
- `echotop_qc.py`: added `pixel_echo_tops(swath, mask, thresholds=(20,30,40))` →
  dict `{20,30,40: (nscan,nray) m, NaN outside mask / below thr}`, applying the
  geometric Hirose gates (floor / outer-ray sidelobe / above-mirror). Vectorized.
- **Validated** (2026-06-20, probe granules): epsilon ~1.0 (GPM 0.934/TRMM 0.954),
  echo-tops monotonic per pixel (0 violations), both **0 finite outside valid views**.

### Interference handling (per user note)
The new metrics MUST be interference-safe. Confirmed: epsilon & echo-tops are NaN
wherever the retrieval was dropped (source field fill→NaN) — so gating on the same
`valid` view mask as the gridder already excludes interference-dropped profiles
(0 px finite outside views in validation). **Phase 2 build_metrics will (a) gate all
accumulators on the existing `valid` mask, and (b) additionally consult
`FS/FLG/qualityFlag` (read into the swath) to drop interference-flagged-but-present
pixels from echo-top/epsilon means.** Flags available: `FS/FLG/qualityFlag`,
`qualityData`, `FS/SLV/qualitySLV`, `FS/scanStatus/dataQuality`.

### Phase 2 — gridder + writer ✅ DONE + VALIDATED
- `grid_swath.py`: added `build_metrics(swath, *, time_window, ...)` — sibling of
  `grid_swath` returning the sparse `metrics` frame keyed (lat,lon,hour); bincount
  per accumulator over the shared valid-pixel key universe; echo-tops via
  `echotop_qc.pixel_echo_tops` on convective valid pixels; all-zero rows dropped
  (compact like rain.parquet). `METRIC_COLS` (22) is the single source of truth.
- `scripts/grid_month.py`: `GRID_METRICS_SCHEMA` auto-built from `METRIC_COLS`;
  worker returns `metrics`; accumulate/flush/reduce + `metrics.parquet` write reuse
  the views path; **`--metrics-only`** flag (skip labeling+grid_swath, write only
  metrics.parquet — for Phase 3 against the validated grid).
- **Validated** (probe granules, both missions): 25 cols, metric keys ⊆ views keys
  (interference-safe), echo-top & heavy-rain counts monotonic, eps_conv>eps_strat,
  nw_conv>nw_strat, Dm ~1.2–1.3 mm, schema roundtrip OK, ~30k rows/orbit. An
  end-to-end `--metrics-only` integration on a 2-day window confirms the
  download→worker→write path (views/rain untouched).

### Phase 3 — full metrics pass (compute)
Re-read every orbit to emit `metrics.parquet` (the 3D cube + epsilon force a fresh
read — same cost profile as the original grid build). Use the existing multi-node
packers (`pack_grid.sh`-style). **Sequence after the running era5 job (648639) +
catalog rebuild finish**, to avoid stacking heavy jobs. GPM 144 mo + TRMM 203 mo.
Leaves `views`/`rain` untouched (no risk to the validated grid).

### Phase 4 — Stage-2 climatology (`src/pf/grid.py`, `scripts/grid_climatology.py`)
- **Conv/strat (no re-grid):** extend the existing rain reducer to also emit
  `raining_count`/`rain_sum` summed over size+echotop **split by raintype**
  (conv=1, strat=0) → carries conv/strat rain + raining-pixel counts to the zarr.
- **Metrics:** add `reduce_grid_metrics_month()` (DuckDB sum over years, keyed
  lat,lon,hour); extend `build_month_hour_dataset()`/`write_grid_zarr()` to carry
  the metric sums+counts; extend `grid_climatology.py reduce_month()` tuple.

### Phase 5 — Stage-3 tiles (`scripts/_precompute_tiles.py`, `_tile_ranges.py`)
Derive annual tile quantities (per member, then COMBINED via **pooled
sums/counts**, never mean-of-means):
- conv/strat (from rain.parquet): `conv_rain`,`strat_rain` (mm/yr),
  `conv_freq`,`strat_freq`, `conv_intensity`,`strat_intensity` — 6
- echo-tops: `conv_echotop20/30/40 = et*_sum/et*_n` (km) — 3
- heavy rain: `freq_gt25/50/75/100 = cnt_gtX/views` — 4
- epsilon: `epsilon_conv = eps_conv_sum/eps_conv_n`, `epsilon_strat` — 2

**15 new quantities × 3 members = 45 new tile vars** (total 18 → **63**). Add
formulas + long_name/units; bump the `_tile_ranges.py` count assertion; regenerate
`tile_ranges.json`. (Trim the conv/strat set if 63 is too many for the UI.)

### Phase 6 — HF (`hf_tiles_space/app.py`, `DATASET_README.md`)
Add the 9 new quantities to `QUANTS`/`QLABEL`/`QDISPLAY`/`LAYMAN` (colorbars
auto-scale from `tile_ranges.json`); add table rows + new variable count to the
README; deploy via `deploy_hf.sh`.

## Files touched (summary)
- `src/pf/readers/gpm_ku.py`, `src/pf/swath.py`, `src/pf/echotop_qc.py`,
  `src/pf/grid_swath.py`, `src/pf/grid.py`
- `scripts/grid_month.py`, `scripts/grid_climatology.py`,
  `scripts/_precompute_tiles.py`, `scripts/_tile_ranges.py`
- `hf_tiles_space/app.py`, `hf_tiles_space/DATASET_README.md`, `deploy_hf.sh`
- New on-disk: `grid/mission=/year=/month=/metrics.parquet` (~similar size to
  `views/`, est. +60–90 GB); +27 vars in `pf_tiles.zarr`.

## Verification
- Stage-1: invariants (`conv_count ≤ n_views`, `et*_n ≤ conv_count`), spot-map a
  month, compare conv-echotop vs feature `max_ht_*dbz` distribution.
- Stage-3: sanity maps (ITCZ/warm-pool high conv-freq & 40 dBZ tops; mid-lat low).
- HF: each new var renders + colorbar reasonable.

## Status (2026-06-20)
- ✅ **Quick win SHIPPED** — conv/strat rain/freq/intensity added as 6 new
  quantities (×3 members → **18→36 HF tile vars**), purely from the existing
  Stage-2 `rain_sum_by_raintype`/`raining_count_by_raintype` (no re-grid). Edits:
  `_precompute_tiles.py`, `_tile_ranges.py`, `app.py`, `DATASET_README.md`.
  Physics validated: conv+strat≈total; conv_int>strat_int; TRMM 56% conv (tropics,
  matches Nesbitt 2006), GPM 41% (to ±68°). Deployed to HF.
- ✅ **Phase 0 (epsilon)** — verified (see above).
- ⏳ **Phases 1–3** (per-pixel echo-tops, heavy-rain thresholds, epsilon split
  conv/strat in `metrics.parquet`) — pending; the full re-read waits for era5 (648639).

## Sequencing note
Three independent tracks:
- ✅ **Quick win (no re-grid):** conv/strat rain/freq/intensity — DONE, shipped to HF.
- **Dev now:** Phases 0–2 (epsilon verify, reader, per-pixel echo-top, gridder,
  validate on a few months) — dev work, no cluster job.
- **Heavy pass (Phase 3):** the full orbit re-read for echo-tops + epsilon +
  heavy-rain thresholds — waits for era5 (648639) + catalog rebuild to finish so we
  don't stack heavy multi-node jobs.
