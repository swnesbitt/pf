#!/usr/bin/env python
"""Compute per-variable color ranges for the tile server.

Default color range = the [0.001, 0.999] quantiles (0.1st–99.9th percentile) of
each ``{member}_{quantity}`` field. Also emits a slider domain [dlo, dhi] so the
front-end can interactively rescale the colormap (dlo=0 since all fields are
non-negative; dhi = a robust 99.99th-percentile upper bound with headroom above
the default max). Writes a tiny JSON sidecar. Run on a COMPUTE node.
"""
import json
import numpy as np
import xarray as xr

ZARR = "/data/scratch/a/snesbitt/pf_tiles.zarr"
OUT = "/data/keeling/a/snesbitt/python/pf/hf_tiles_space/tile_ranges.json"

ds = xr.open_zarr(ZARR, consolidated=True)
ranges = {}
print(f"{'variable':26s} {'n_finite':>11s}  {'min':>10s} {'q001':>10s} "
      f"{'q999':>10s} {'p9999':>10s} {'max':>10s}", flush=True)
for v in sorted(ds.data_vars):
    da = ds[v]
    vals = da.values.ravel()
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n:
        vmin_data = float(vals.min()); vmax_data = float(vals.max())
        q001, q999, p9999 = (float(x) for x in np.quantile(vals, [0.001, 0.999, 0.9999]))
    else:
        vmin_data = vmax_data = q001 = q999 = p9999 = 0.0

    # RIGOR CHECKS -----------------------------------------------------------
    assert n > 0, f"{v}: no finite values!"
    assert vmin_data >= -1e-6, f"{v}: negative data min {vmin_data} (unexpected)"
    assert q001 <= q999, f"{v}: q001 {q001} > q999 {q999}!"
    assert vmin_data <= q001 <= q999 <= vmax_data + 1e-6, (
        f"{v}: quantiles not bracketed by data: "
        f"min={vmin_data} q001={q001} q999={q999} max={vmax_data}")

    # defaults + slider domain
    lo, hi = q001, q999
    if not (hi > lo):                      # degenerate field -> sane fallback
        lo, hi = 0.0, (hi if hi > 0 else 1.0)
    dlo = 0.0                              # all fields non-negative
    dhi = max(p9999, hi * 1.05)            # headroom above the default max
    if not (dhi > dlo):
        dhi = hi if hi > 0 else 1.0

    ranges[v] = {
        "vmin": round(lo, 6), "vmax": round(hi, 6),
        "dlo": round(dlo, 6), "dhi": round(dhi, 6),
        "units": da.attrs.get("units", ""),
        "long_name": da.attrs.get("long_name", v),
    }
    print(f"{v:26s} {n:11d}  {vmin_data:10.4g} {q001:10.4g} "
          f"{q999:10.4g} {p9999:10.4g} {vmax_data:10.4g}", flush=True)

# SHARED color scales -------------------------------------------------------
# Intrinsic DSD parameters must use ONE scale across all members (GPM/TRMM/COMBINED)
# AND across conv/strat, so the maps are directly comparable. Override the per-var
# quantile ranges with a single shared range for each group.
def _apply_shared(suffixes, vmin=None, vmax=None, dlo=None, dhi=None):
    vs = [v for v in ranges if any(v.endswith(s) for s in suffixes)]
    if vmin is None:                      # pool the data across the group
        allv = np.concatenate([ds[v].values.ravel() for v in vs])
        allv = allv[np.isfinite(allv)]
        vmin, vmax = (float(x) for x in np.quantile(allv, [0.01, 0.99]))
    dlo = 0.0 if dlo is None else dlo
    dhi = vmax * 1.05 if dhi is None else dhi
    for v in vs:
        ranges[v].update(vmin=round(vmin, 6), vmax=round(vmax, 6),
                         dlo=round(dlo, 6), dhi=round(dhi, 6))
    print(f"shared scale for *{suffixes}: vmin={vmin:.4g} vmax={vmax:.4g} ({len(vs)} vars)", flush=True)

# Nw: fixed log10(Nw) scale 2.5-4.5 (data is ~3.0-3.8, max ~4.7); the rest pooled.
_apply_shared(["nw_conv", "nw_strat"], vmin=2.5, vmax=4.5, dlo=2.0, dhi=5.0)
_apply_shared(["dm_conv", "dm_strat"], dlo=0.0)               # Dm (mm), shared
_apply_shared(["eps_conv", "eps_strat"], dlo=0.0)            # epsilon, shared
# echo-tops: ONE scale across thresholds (20/30/40) AND members, so storm-depth
# maps are comparable (40 dBZ tops read lower than 20 dBZ tops on the same scale).
_apply_shared(["echotop20", "echotop30", "echotop40"], dlo=0.0)

# final monotonicity / structure check across all vars
for v, r in ranges.items():
    assert r["dlo"] <= r["vmin"] <= r["vmax"] <= r["dhi"] + 1e-6, f"domain check failed: {v} {r}"
assert len(ranges) == 81, f"expected 81 vars, got {len(ranges)}"

with open(OUT, "w") as f:
    json.dump(ranges, f, indent=2)
print(f"\nALL {len(ranges)} vars checked OK; wrote {OUT}", flush=True)
