#!/usr/bin/env python
"""Precompute the tileable dataset for the xpublish-tiles map server.

Reads the GPM & TRMM Icechunk grid stores, collapses month/hour/subcategory to
ANNUAL totals, builds the COMBINED (GPM+TRMM pooled) member, derives the
quantities, and writes a single small Zarr with 81 flat (lat, lon) variables
named ``{member}_{quantity}`` + CF lat/lon coords. Run on a COMPUTE node.

  rain, raining_views, views, rate(=rain/views), freq(=raining/views),
  intensity(=rain/raining),
  conv_rain, strat_rain, conv_freq, strat_freq, conv_intensity, strat_intensity,
  conv_rain_frac, conv_pixel_frac,
  echotop20/30/40 (convective echo-top height, m),
  freq_gt25/50/75/100 (heavy near-surface rain occurrence frequency),
  eps_conv/strat, nw_conv/strat, dm_conv/strat (near-surface DSD parameter means)
    x   GPM, TRMM, COMBINED   = 27 x 3 = 81
"""
import os
import numpy as np
import xarray as xr
import icechunk

OUT = "/data/scratch/a/snesbitt/pf_tiles.zarr"
MIN_VIEWS = 10
MIN_SAMPLES = 10  # min per-cell sample count (_n) for a metric mean tile (echo-top, DSD)
HOURS_PER_YEAR = 24 * 365.25  # 8766; mean rate (mm/hr) -> annual accumulation (mm/yr)
# Bit-rounding: keep this many of float32's 23 mantissa bits, zero the rest, so
# the (lossy but controlled) trailing zeros compress hugely. 12 bits ~= 3.6
# significant digits -- far beyond the real information content of a precip
# climatology, and the store stays plain float32 (transparent to readers).
KEEPBITS = 12


def bitround(a, keepbits=KEEPBITS):
    """Zero the low mantissa bits of a float32 array (round-to-nearest), keeping
    ``keepbits`` of 23. NaN/inf are passed through untouched."""
    a = np.asarray(a, dtype=np.float32)
    if keepbits >= 23:
        return a
    drop = np.uint32(23 - keepbits)
    mask = np.uint32(0xFFFFFFFF) << drop
    half = np.uint32(1) << (drop - np.uint32(1))
    bits = a.view(np.uint32)
    rounded = ((bits + half) & mask).view(np.float32)
    return np.where(np.isfinite(a), rounded, a).astype(np.float32)
QUANTS = ["rain", "raining_views", "views", "rate", "freq", "intensity",
          "conv_rain", "strat_rain", "conv_freq", "strat_freq",
          "conv_intensity", "strat_intensity", "conv_rain_frac", "conv_pixel_frac",
          # new per-pixel metrics (Phase 5): convective echo-top heights, heavy-rain
          # occurrence frequencies, and conv/strat DSD parameter means
          "echotop20", "echotop30", "echotop40",
          "freq_gt25", "freq_gt50", "freq_gt75", "freq_gt100",
          "eps_conv", "eps_strat", "nw_conv", "nw_strat", "dm_conv", "dm_strat"]

# metric total vars carried from the Stage-2 store (additive -> pool then derive)
METRIC_TOTALS = ["et20_sum", "et20_n", "et30_sum", "et30_n", "et40_sum", "et40_n",
                 "cnt_gt25", "cnt_gt50", "cnt_gt75", "cnt_gt100",
                 "eps_conv_sum", "eps_conv_n", "eps_strat_sum", "eps_strat_n",
                 "nw_conv_sum", "nw_conv_n", "nw_strat_sum", "nw_strat_n",
                 "dm_conv_sum", "dm_conv_n", "dm_strat_sum", "dm_strat_n"]


def open_grid(m):
    st = icechunk.s3_storage(bucket="spaceborne-grids", prefix=f"pf_grid_{m}",
        endpoint_url=os.environ["SPACEBORNE_MINIO_ENDPOINT"], allow_http=True,
        force_path_style=True, region="us-east-1",
        access_key_id=os.environ["SPACEBORNE_MINIO_ACCESS"],
        secret_access_key=os.environ["SPACEBORNE_MINIO_SECRET"])
    return xr.open_zarr(icechunk.Repository.open(st).readonly_session("main").store,
                        consolidated=False)


# raintype_class index meaning (from grid.raintype_class): 0=stratiform, 1=convective, 2=other
RT_STRAT, RT_CONV = 0, 1


def annual_totals(m):
    """Annual Σrain, Σraining, Nviews (totals over all months/hours/subcats),
    plus convective/stratiform splits (Σrain & Σraining for raintype_class
    conv=1 / strat=0). All are additive totals so the COMBINED member pools
    correctly (sum of GPM+TRMM totals, never a mean-of-means)."""
    ds = open_grid(m)
    print(f"{m}: reducing native store -> annual totals ...", flush=True)
    # Collapse month/hour ONCE per by-raintype var (keep raintype_class) so the
    # big (raintype,month,hour,lat,lon) array is streamed once, not per-slice.
    # Result is small: (raintype_class=3, lat, lon). Then derive total + splits.
    rt_rain = ds["rain_sum_by_raintype"].sum(("month", "hour")).compute()
    rt_rning = ds["raining_count_by_raintype"].sum(("month", "hour")).compute()
    views = ds["views"].sum(("month", "hour")).compute()

    def _slice(da, idx):
        return (da.sel(raintype_class=idx)
                  .drop_vars(["raintype_class", "raintype_label"], errors="ignore"))

    base = dict(
        rain=rt_rain.sum("raintype_class"), raining=rt_rning.sum("raintype_class"),
        views=views,
        conv_rain=_slice(rt_rain, RT_CONV), strat_rain=_slice(rt_rain, RT_STRAT),
        conv_raining=_slice(rt_rning, RT_CONV), strat_raining=_slice(rt_rning, RT_STRAT),
    )
    # new per-pixel metric totals (summed over month/hour -> (lat,lon)); additive,
    # so the COMBINED member pools them then derives means/freqs after the sum.
    metric_tot = {c: ds[c].sum(("month", "hour")).compute()
                  for c in METRIC_TOTALS if c in ds.data_vars}
    if metric_tot:
        print(f"  {m}: + {len(metric_tot)} metric totals", flush=True)
    out = xr.Dataset({**base, **metric_tot})
    print(f"  {m}: done {dict(out.sizes)}", flush=True)
    return out


raw = {"GPM": annual_totals("GPM"), "TRMM": annual_totals("TRMM")}
raw["COMBINED"] = (raw["GPM"] + raw["TRMM"]).assign_coords(raw["GPM"].coords)

data_vars = {}
for member, d in raw.items():
    R, RN, V = d["rain"], d["raining"], d["views"]
    Rc, RNc = d["conv_rain"], d["conv_raining"]
    Rs, RNs = d["strat_rain"], d["strat_raining"]
    with np.errstate(divide="ignore", invalid="ignore"):
        fields = {
            # annual rainfall accumulation = mean unconditional rate x hours/year
            "rain": ((R / V) * HOURS_PER_YEAR).where(V >= MIN_VIEWS),
            "raining_views": RN.where(V >= MIN_VIEWS),
            "views": V,
            "rate": (R / V).where(V >= MIN_VIEWS),
            "freq": (RN / V).where(V >= MIN_VIEWS),
            "intensity": (R / RN).where(RN >= 1),
            # convective / stratiform splits (from rain_*_by_raintype; no re-grid)
            "conv_rain": ((Rc / V) * HOURS_PER_YEAR).where(V >= MIN_VIEWS),
            "strat_rain": ((Rs / V) * HOURS_PER_YEAR).where(V >= MIN_VIEWS),
            "conv_freq": (RNc / V).where(V >= MIN_VIEWS),
            "strat_freq": (RNs / V).where(V >= MIN_VIEWS),
            "conv_intensity": (Rc / RNc).where(RNc >= 1),
            "strat_intensity": (Rs / RNs).where(RNs >= 1),
            # convective RAINFALL fraction: convective accumulation / total
            # accumulation (= Σrain_conv / Σrain_total)
            "conv_rain_frac": (Rc / R).where((V >= MIN_VIEWS) & (R > 0)),
            # convective OCCURRENCE/AREA fraction: convective raining pixels /
            # all raining pixels
            "conv_pixel_frac": (RNc / RN).where(RN >= 1),
        }
        # --- new per-pixel metric quantities (Phase 5) ----------------------
        # means = Σ_sum/Σ_n (gate on per-cell sample count); heavy-rain = cnt/views.
        if "et20_sum" in d.data_vars:
            def _mean(sumv, nv):
                return (d[sumv] / d[nv]).where(d[nv] >= MIN_SAMPLES)
            fields.update({
                "echotop20": _mean("et20_sum", "et20_n"),
                "echotop30": _mean("et30_sum", "et30_n"),
                "echotop40": _mean("et40_sum", "et40_n"),
                "freq_gt25": (d["cnt_gt25"] / V).where(V >= MIN_VIEWS),
                "freq_gt50": (d["cnt_gt50"] / V).where(V >= MIN_VIEWS),
                "freq_gt75": (d["cnt_gt75"] / V).where(V >= MIN_VIEWS),
                "freq_gt100": (d["cnt_gt100"] / V).where(V >= MIN_VIEWS),
                "eps_conv": _mean("eps_conv_sum", "eps_conv_n"),
                "eps_strat": _mean("eps_strat_sum", "eps_strat_n"),
                # Nw is stored as dBNw (=10*log10 Nw); /10 -> log10(Nw), Nw in mm^-1 m^-3
                "nw_conv": _mean("nw_conv_sum", "nw_conv_n") / 10.0,
                "nw_strat": _mean("nw_strat_sum", "nw_strat_n") / 10.0,
                "dm_conv": _mean("dm_conv_sum", "dm_conv_n"),
                "dm_strat": _mean("dm_strat_sum", "dm_strat_n"),
            })
    for q, da in fields.items():
        da = da.astype("float32")
        da = da.copy(data=bitround(da.values))   # lossy bit-round for compression
        da.attrs["long_name"] = {
            "rain": "annual rainfall accumulation (unconditional rate x 8766 h/yr)",
            "raining_views": "annual count of raining pixel-views",
            "views": "annual count of radar pixel-views (sampling denominator)",
            "rate": "unconditional mean rain rate (mm/hr)",
            "freq": "rain frequency (raining/views)",
            "intensity": "conditional mean rain rate (mm/hr)",
            "conv_rain": "convective annual rainfall accumulation (mm/yr)",
            "strat_rain": "stratiform annual rainfall accumulation (mm/yr)",
            "conv_freq": "convective rain frequency (conv raining/views)",
            "strat_freq": "stratiform rain frequency (strat raining/views)",
            "conv_intensity": "convective conditional mean rate (mm/hr)",
            "strat_intensity": "stratiform conditional mean rate (mm/hr)",
            "conv_rain_frac": "convective rainfall fraction (conv accumulation / total accumulation)",
            "conv_pixel_frac": "convective fraction of precipitating pixels (conv raining / all raining)",
            "echotop20": "convective 20 dBZ echo-top height (mean, m)",
            "echotop30": "convective 30 dBZ echo-top height (mean, m)",
            "echotop40": "convective 40 dBZ echo-top height (mean, m)",
            "freq_gt25": "frequency of near-surface rain >= 25 mm/hr (occurrence/views)",
            "freq_gt50": "frequency of near-surface rain >= 50 mm/hr (occurrence/views)",
            "freq_gt75": "frequency of near-surface rain >= 75 mm/hr (occurrence/views)",
            "freq_gt100": "frequency of near-surface rain >= 100 mm/hr (occurrence/views)",
            "eps_conv": "convective near-surface DSD epsilon (mean)",
            "eps_strat": "stratiform near-surface DSD epsilon (mean)",
            "nw_conv": "convective near-surface log10(Nw) (mean; Nw in mm^-1 m^-3)",
            "nw_strat": "stratiform near-surface log10(Nw) (mean; Nw in mm^-1 m^-3)",
            "dm_conv": "convective near-surface mass-weighted mean diameter Dm (mean, mm)",
            "dm_strat": "stratiform near-surface mass-weighted mean diameter Dm (mean, mm)",
        }[q]
        da.attrs["units"] = {"rate": "mm/hr", "intensity": "mm/hr", "freq": "1",
                             "rain": "mm/year", "raining_views": "count", "views": "count",
                             "conv_rain": "mm/year", "strat_rain": "mm/year",
                             "conv_freq": "1", "strat_freq": "1",
                             "conv_intensity": "mm/hr", "strat_intensity": "mm/hr",
                             "conv_rain_frac": "1", "conv_pixel_frac": "1",
                             "echotop20": "m", "echotop30": "m", "echotop40": "m",
                             "freq_gt25": "1", "freq_gt50": "1", "freq_gt75": "1",
                             "freq_gt100": "1",
                             "eps_conv": "1", "eps_strat": "1",
                             "nw_conv": "log10(mm-1 m-3)", "nw_strat": "log10(mm-1 m-3)",
                             "dm_conv": "mm", "dm_strat": "mm"}[q]
        # valid_min/valid_max so xpublish-tiles auto-scales colors (no client
        # colorscalerange needed). Robust 99th-pct top, 0 floor.
        vals = da.values
        vmax = float(np.nanpercentile(vals, 99)) if np.isfinite(vals).any() else 1.0
        da.attrs["valid_min"] = 0.0
        da.attrs["valid_max"] = vmax if vmax > 0 else 1.0
        data_vars[f"{member}_{q}"] = da

ds = xr.Dataset(data_vars)
# CF geospatial coords so xpublish-tiles can georeference / reproject to WebMercator
ds["lat"].attrs.update(standard_name="latitude", units="degrees_north", axis="Y")
ds["lon"].attrs.update(standard_name="longitude", units="degrees_east", axis="X")
ds.attrs.update(
    title="PF swath-grid climatology — annual tile fields (GPM/TRMM/COMBINED)",
    institution="University of Illinois", Conventions="CF-1.8",
    note="27 quantities x 3 members, annual (all months & UTC hours), 0.05 deg shared grid",
)

import zarr
from zarr.codecs import BloscCodec, BloscShuffle
# Blosc(zstd, clevel=9) + BITSHUFFLE on the bit-rounded float32 fields: bitshuffle
# groups the zeroed low-mantissa bits into long runs zstd crushes -> typically
# 2-4x smaller than the previous Zstd-5. Blosc is a standard zarr-v3 codec, so
# the HF app (same zarr stack) reads it transparently.
comp = BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.bitshuffle, typesize=4)
enc = {v: {"compressors": [comp], "chunks": (340, 720)} for v in ds.data_vars}
print(f"writing {OUT} ({len(ds.data_vars)} vars, {ds['lat'].size}x{ds['lon'].size}) ...", flush=True)
if os.path.exists(OUT):
    import shutil
    shutil.rmtree(OUT)
ds.to_zarr(OUT, mode="w", consolidated=True, encoding=enc, zarr_format=3)
print("DONE", OUT, flush=True)
print("vars:", list(ds.data_vars), flush=True)
