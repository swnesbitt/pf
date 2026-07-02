#!/usr/bin/env python
"""Build notebooks/pf_grid_preview.ipynb — interactive browser for the gridded
rain-contribution climatology (pf.grid marginal-mode NetCDFs)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md("""# Gridded Rain-Contribution Climatology — Interactive Browser

Browse the 0.05° **rain-contribution** grids (`pf.grid`, *marginal* mode) for TRMM
and GPM. Each cell stores, per month-of-year and storm category, three additive
quantities so any rate is derivable:

| stored | meaning |
|---|---|
| `rain_sum` | Σ near-surface rain rate (mm/hr) — *sum of instantaneous rates, not a depth* |
| `raining_count` | # of raining pixels (rain > 0) |
| `views` | # of radar near-surface observations (the **sampling denominator**) |

Derived metrics:

- **Unconditional rate** = `rain_sum / views` (mm/hr) — climatological mean rain rate
- **Rain frequency** = `raining_count / views` (%) — how often it rains
- **Conditional intensity** = `rain_sum / raining_count` (mm/hr) — how hard, when it rains

Stratifiers: **size** (`major_axis_km`), **20-dBZ echo-top**, **rain type**
(stratiform / convective / other). Pick a breakdown + class, or *All*.

> Widgets need a live kernel — run all cells, then drive the dropdowns at the bottom.""")

co("""import glob, os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ipywidgets as W
from IPython.display import display

# Where the *_clim.nc grids live. Repoint at the final product dir once the
# full climatology is built (e.g. /data/scratch/a/snesbitt/pf_db/grid).
GRID_DIR = os.environ.get("PF_GRID_DIR", "/tmp/pf_grid_prev")

paths = sorted(glob.glob(f"{GRID_DIR}/*_clim.nc"))
DS = {os.path.basename(p).split("_clim")[0]: xr.open_dataset(p) for p in paths}
print("loaded:", {k: dict(v.sizes) for k, v in DS.items()})
assert DS, f"no *_clim.nc under {GRID_DIR}"
""")

md("""## Field assembly

`get_field` slices the requested mission / month / breakdown / class, **block-sums**
the raw accumulators to the display resolution (correct for rates — we coarsen the
*sums* then divide, never average a ratio), and forms the chosen metric. Cells with
fewer than `min_views` observations are masked.""")

co("""BREAKDOWNS = {
    \"All storms\":   (\"size\",     None),   # sum over size classes == all rain
    \"By size\":      (\"size\",     \"size_label\"),
    \"By echo-top\":  (\"echotop\",  \"echotop_label\"),
    \"By rain type\": (\"raintype\", \"raintype_label\"),
}
METRICS = [\"Unconditional rate (mm/hr)\", \"Rain frequency (%)\",
           \"Conditional intensity (mm/hr)\", \"Rain sum (Σ mm/hr)\", \"Views (count)\"]
COARSEN = {\"0.05° (full)\": 1, \"0.1°\": 2, \"0.25°\": 5, \"0.5°\": 10}


def _block_sum(a, f):
    if f == 1:
        return a
    nlat = (a.shape[0] // f) * f
    nlon = (a.shape[1] // f) * f
    a = a[:nlat, :nlon]
    return a.reshape(nlat // f, f, nlon // f, f).sum(axis=(1, 3))


def _coords_coarsen(lat, lon, f):
    if f == 1:
        return lat, lon
    nlat = (lat.size // f) * f
    nlon = (lon.size // f) * f
    return lat[:nlat].reshape(-1, f).mean(1), lon[:nlon].reshape(-1, f).mean(1)


def get_field(mission, month, breakdown, klass, metric, f, min_views):
    ds = DS[mission]
    axis, labelvar = BREAKDOWNS[breakdown]
    rs = ds[f\"rain_sum_by_{axis}\"].sel(month=month).values        # (class, lat, lon)
    rc = ds[f\"raining_count_by_{axis}\"].sel(month=month).values
    views = ds[\"views\"].sel(month=month).values.astype(np.float64)
    if labelvar is None or klass == \"All\":
        rs2, rc2 = rs.sum(0), rc.sum(0)
    else:
        labels = list(ds[labelvar].values.astype(str))
        ci = labels.index(klass)
        rs2, rc2 = rs[ci], rc[ci]
    rs2 = _block_sum(rs2.astype(np.float64), f)
    rc2 = _block_sum(rc2.astype(np.float64), f)
    vv = _block_sum(views, f)
    lat, lon = _coords_coarsen(ds[\"lat\"].values, ds[\"lon\"].values, f)
    m = vv < min_views
    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):
        if metric.startswith(\"Unconditional\"):
            field, cmap = np.where(m, np.nan, rs2 / vv), \"viridis\"
        elif metric.startswith(\"Rain frequency\"):
            field, cmap = np.where(m, np.nan, 100 * rc2 / vv), \"YlGnBu\"
        elif metric.startswith(\"Conditional\"):
            field, cmap = np.where(m | (rc2 < 1), np.nan, rs2 / np.maximum(rc2, 1)), \"magma\"
        elif metric.startswith(\"Rain sum\"):
            field, cmap = np.where(m, np.nan, rs2), \"turbo\"
        else:
            field, cmap = np.where(vv < 1, np.nan, vv), \"cividis\"
    return lat, lon, field, cmap
""")

md("""## Browser

Mission · month · breakdown · class · metric · display resolution · min-views ·
upper-percentile colour clip. The class list updates with the breakdown.""")

co("""missions = list(DS)
months = sorted(int(x) for x in DS[missions[0]][\"month\"].values)
MONTH_NAMES = [\"\", \"Jan\", \"Feb\", \"Mar\", \"Apr\", \"May\", \"Jun\",
               \"Jul\", \"Aug\", \"Sep\", \"Oct\", \"Nov\", \"Dec\"]

w_mission = W.Dropdown(options=missions, description=\"Mission\")
w_month = W.Dropdown(options=[(MONTH_NAMES[m], m) for m in months], description=\"Month\")
w_break = W.Dropdown(options=list(BREAKDOWNS), description=\"Breakdown\")
w_class = W.Dropdown(options=[\"All\"], description=\"Class\")
w_metric = W.Dropdown(options=METRICS, description=\"Metric\")
w_coarse = W.Dropdown(options=list(COARSEN), value=\"0.25°\", description=\"Res\")
w_minv = W.IntSlider(value=10, min=1, max=200, step=1, description=\"min views\")
w_clip = W.FloatSlider(value=99.0, min=90.0, max=100.0, step=0.5, description=\"clip pctl\")
out = W.Output()


def _class_options(breakdown):
    axis, labelvar = BREAKDOWNS[breakdown]
    if labelvar is None:
        return [\"All\"]
    labels = list(DS[w_mission.value][labelvar].values.astype(str))
    return [\"All\"] + labels


def _on_break(change):
    w_class.options = _class_options(w_break.value)
    w_class.value = \"All\"


def redraw(*_):
    with out:
        out.clear_output(wait=True)
        lat, lon, field, cmap = get_field(
            w_mission.value, w_month.value, w_break.value, w_class.value,
            w_metric.value, COARSEN[w_coarse.value], w_minv.value)
        finite = field[np.isfinite(field)]
        vmax = np.percentile(finite, w_clip.value) if finite.size else 1.0
        vmax = float(vmax) if vmax > 0 else 1.0
        fig = plt.figure(figsize=(13, 5.2))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=0, vmax=vmax,
                           shading=\"auto\", transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4, color=\"0.25\")
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor=\"0.6\")
        ax.set_extent([-180, 180, float(lat.min()), float(lat.max())],
                      crs=ccrs.PlateCarree())
        cls = \"\" if w_class.value == \"All\" else f\" — {w_class.value}\"
        ax.set_title(f\"{w_mission.value} {MONTH_NAMES[w_month.value]} · \"
                     f\"{w_metric.value} · {w_break.value}{cls}\", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        plt.show()


w_break.observe(_on_break, names=\"value\")
for w in (w_mission, w_month, w_break, w_class, w_metric, w_coarse, w_minv, w_clip):
    w.observe(redraw, names=\"value\")

controls = W.VBox([
    W.HBox([w_mission, w_month, w_metric]),
    W.HBox([w_break, w_class, w_coarse]),
    W.HBox([w_minv, w_clip]),
])
display(controls, out)
redraw()
""")

md("""### Notes

- **Coarsening is statistically exact for rates**: we block-*sum* `rain_sum`,
  `raining_count`, and `views` to the display resolution, *then* take the ratio —
  so a coarsened mean rate equals the true area-aggregated mean, not an average of
  per-cell rates. Drop to `0.05° (full)` for publication detail (slower redraw).
- `min views` masks under-sampled cells (swath-edge slivers, mission band edges).
- Preview grids cover only the **months built so far**; repoint `PF_GRID_DIR` at the
  final product directory once the full climatology run completes.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out_path = "notebooks/pf_grid_preview.ipynb"
with open(out_path, "w") as fh:
    nbf.write(nb, fh)
print("wrote", out_path, "with", len(cells), "cells")
