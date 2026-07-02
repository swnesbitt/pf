---
license: cc-by-4.0
pretty_name: High Resolution Precipitation Climatologies from NASA Precipitation Measurement Missions
tags:
- climate
- precipitation
- remote-sensing
- earth-observation
- TRMM
- GPM
- weather-radar
- climatology
---

# High Resolution Precipitation Climatologies from NASA Precipitation Measurement Missions

[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9189-blue)](https://doi.org/10.57967/hf/9189)

**Author:** Stephen W. Nesbitt, Department of Climate, Meteorology &amp; Atmospheric
Sciences, University of Illinois Urbana-Champaign.

Annual, **0.05°** precipitation climatology from the NASA Precipitation Measurement
Missions spaceborne radars: the **TRMM** Precipitation Radar (Ku-band, 1997–2014)
and the **GPM** Dual-frequency Precipitation Radar (DPR, Ku band, 2014–present).

**Interactive map viewer:** https://huggingface.co/spaces/snesbitt/pf-grid-tiles-app

## Contents

`pf_tiles.zarr` — a Zarr v3 store (Zstd-compressed) holding **81 variables** =
{GPM, TRMM, COMBINED} × 27 quantities, on a shared ±68° latitude, 0.05° grid
(lat 2721 × lon 7200), CF `lat`/`lon` coordinates.

`pf_tiles_ms.zarr` — the same 81 variables as a **GeoZarr multiscale pyramid**
with levels at **0.05° / 0.1° / 0.25° / 0.5° / 1°** (native + 2×/5×/10×/20×
nanmean-coarsened, `spatial:transform` per level). This is what the map app
serves: xpublish-tiles selects the resolution level per zoom so low-zoom views
read pre-averaged coarse levels (smooth) while zoomed-in views read native
resolution.

`pf_grid_0p05deg.nc` and `pf_grid_0p25deg.nc` — standalone, self-describing
**CF-1.8 NetCDF4** files holding the same 81 fields at 0.05° (2721 × 7200) and
0.25° (544 × 1440) respectively. These are the download-and-go companions to the
Zarr stores for users who want a single file; each carries CF `long_name`,
`units`, and `valid_min`/`valid_max` per variable.

```python
import xarray as xr
ds = xr.open_dataset("pf_grid_0p25deg.nc")   # or pf_grid_0p05deg.nc
```

Quantities per member (`{GPM,TRMM,COMBINED}_<quantity>`):

| variable | definition | units |
|---|---|---|
| `*_rain` | annual precipitation accumulation = mean unconditional rate × 8766 h/yr | mm/year |
| `*_rate` | unconditional mean rate = Σrain / Nviews | mm/hr |
| `*_freq` | precipitation frequency = Nraining / Nviews | fraction |
| `*_intensity` | conditional mean rate = Σrain / Nraining | mm/hr |
| `*_raining_views` | count of pixel-views with precipitation | count |
| `*_views` | count of radar pixel-views (sampling denominator) | count |
| `*_conv_rain` | **convective** annual accumulation = (Σrain_conv / Nviews) × 8766 | mm/year |
| `*_strat_rain` | **stratiform** annual accumulation = (Σrain_strat / Nviews) × 8766 | mm/year |
| `*_conv_freq` | convective frequency = Nraining_conv / Nviews | fraction |
| `*_strat_freq` | stratiform frequency = Nraining_strat / Nviews | fraction |
| `*_conv_intensity` | convective conditional rate = Σrain_conv / Nraining_conv | mm/hr |
| `*_strat_intensity` | stratiform conditional rate = Σrain_strat / Nraining_strat | mm/hr |
| `*_conv_rain_frac` | convective rainfall fraction = Σrain_conv / Σrain_total (accumulation) | fraction |
| `*_conv_pixel_frac` | convective area fraction = Nraining_conv / Nraining (occurrence) | fraction |
| `*_echotop20` | **convective** 20 dBZ echo-top height (mean over convective pixels) | m |
| `*_echotop30` | **convective** 30 dBZ echo-top height (mean) | m |
| `*_echotop40` | **convective** 40 dBZ echo-top height (mean) | m |
| `*_freq_gt25` | frequency of near-surface rain ≥ 25 mm/hr = N(rain≥25) / Nviews | fraction |
| `*_freq_gt50` | frequency of near-surface rain ≥ 50 mm/hr | fraction |
| `*_freq_gt75` | frequency of near-surface rain ≥ 75 mm/hr | fraction |
| `*_freq_gt100` | frequency of near-surface rain ≥ 100 mm/hr | fraction |
| `*_eps_conv` / `*_eps_strat` | mean near-surface DSD ε (epsilon) over convective / stratiform pixels | — |
| `*_nw_conv` / `*_nw_strat` | mean near-surface log₁₀(Nw) (normalized intercept) over convective / stratiform | log₁₀(mm⁻¹ m⁻³) |
| `*_dm_conv` / `*_dm_strat` | mean near-surface Dm (mass-weighted mean diameter) over convective / stratiform | mm |

Convective vs stratiform follows the radar 2A `typePrecip` classification. Echo-tops
and DSD parameters (ε, Nw, Dm) are reduced to the near-surface clutter-free gate;
echo-tops use the Hirose-2023-style geometric QC and are computed for convective
pixels only. Each mean is stored as Σ/N so `COMBINED` pools correctly.
`COMBINED` = GPM + TRMM pooled (TRMM is zero poleward of ±38°).

## Method

Every observed radar pixel is gridded directly from the orbital swaths, so `views`
counts all sampling and the precipitation fields are consistent with that
denominator. Reading the store:

```python
import xarray as xr
ds = xr.open_zarr("pf_tiles.zarr", consolidated=True)
```

## Source data & product versions

The gridded quantity is the **near-surface precipitation rate** — the
`precipRateNearSurface` field (`FS/SLV/precipRateNearSurface`, the radar
algorithm's near-surface rain estimate, mm hr⁻¹). The rain quantities are derived
from it and the per-pixel sampling; the echo-top and DSD quantities come from the
3-D reflectivity and the `FS/SLV` DSD retrieval (epsilon, `paramDSD` Nw/Dm).

| mission | instrument | product (short name) | version |
|---|---|---|---|
| GPM | Dual-frequency Precipitation Radar (DPR), Ku band | `GPM_2ADPR` (FS swath, Ku) | **V07** |
| TRMM | Precipitation Radar (PR), GPM-reprocessed | `GPM_2APR` | **V07** |

GPM is pinned to **V07** for a version-uniform record: the V08/V10 DPR
reprocessing is in progress and only partially covers the archive, so preferring
it would mix versions across the record. TRMM PR is **V07-only**. (When V08 DPR
covers the full archive, the record will be migrated to V08-uniform.)

## Sensitivity note

TRMM's Precipitation Radar has a minimum detectable reflectivity of ≈17–18 dBZ,
whereas GPM's Dual-frequency Precipitation Radar (DPR) is more sensitive (≈12 dBZ) and detects substantially
more light precipitation. This can introduce discontinuities between the TRMM and
GPM records — and within the COMBINED member — most pronounced where light
precipitation is prevalent.

## Key references

- Nesbitt, S. W., and A. M. Anders (2009), *Very high resolution precipitation climatologies from the Tropical Rainfall Measuring Mission precipitation radar*, Geophys. Res. Lett., 36, L15815, doi:10.1029/2009GL038026
- Hirose, M., and K. Nakamura (2005), *Spatial and diurnal variation of precipitation systems over Asia observed by the TRMM Precipitation Radar*, J. Geophys. Res., 110, D05106, doi:10.1029/2004JD004815
- Bookhagen, B., and D. W. Burbank (2006), *Topography, relief, and TRMM-derived rainfall variations along the Himalaya*, Geophys. Res. Lett., 33, L08405, doi:10.1029/2006GL026037
- Kidd, C., J. Kwiatkowski, and S. W. Nesbitt (2010), *Investigations into high resolution mapping of precipitation features utilizing the TRMM precipitation radar*, IGARSS 2010 (IEEE Int. Geosci. Remote Sens. Symp.), 2337–2340, doi:10.1109/IGARSS.2010.5649629
- Biasutti, M., S. E. Yuter, C. D. Burleyson, and A. H. Sobel (2012), *Very high resolution rainfall patterns measured by TRMM precipitation radar: seasonal and diurnal cycles*, Clim. Dyn., 39, 239–258, doi:10.1007/s00382-011-1146-6
- Anders, A. M., and S. W. Nesbitt (2015), *Altitudinal precipitation gradients in the tropics from Tropical Rainfall Measuring Mission (TRMM) precipitation radar*, J. Hydrometeorol., 16, 441–448, doi:10.1175/JHM-D-14-0178.1

## Cite this dataset

> Nesbitt, S. W. (2026). *High Resolution Precipitation Climatologies from NASA
> Precipitation Measurement Missions* [Data set]. Department of Climate, Meteorology
> &amp; Atmospheric Sciences, University of Illinois Urbana-Champaign / Hugging Face.
> https://doi.org/10.57967/hf/9189

DOI: **[10.57967/hf/9189](https://doi.org/10.57967/hf/9189)**

## Contact & acknowledgment

© University of Illinois Board of Trustees. Contact:
[Steve Nesbitt](https://swnesbitt.github.io). This work was supported by projects
from the NASA Precipitation Measurement Missions and Weather programs to the
University of Illinois.
