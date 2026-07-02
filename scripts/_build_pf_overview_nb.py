#!/usr/bin/env python
"""Build notebooks/pf_overview.ipynb (PF climatology overview) via nbformat."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md("""# Precipitation-Feature (PF) Database — Overview

TRMM + GPM Radar Precipitation Features built into `/data/scratch/a/snesbitt/pf_db`
(Nesbitt-style RPFs, current permissive definition: 1-pixel minimum, per-instrument
noise-floor threshold). This notebook reads **every per-orbit feature Parquet
processed so far** and produces:

1. PF frequency on a **0.5° grid** (TRMM and GPM).
2. Maps of the **top-10 max 40-dBZ heights** and **10 lowest 85/89-GHz PCTs** per mission.
3. A **land / ocean / coastline** summary table.
4. **PDFs** of max 40-dBZ height and 85/89-GHz PCT for each surface population, per sensor.

> Re-run any time: it snapshots whatever orbits are on disk, so it tracks the build as it grows.""")

co("""import glob, warnings
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False
print("cartopy:", HAVE_CARTOPY)

PF_ROOT = "/data/scratch/a/snesbitt/pf_db"
GRID_DEG = 0.5
# Columns we need (read directly from each file so the hive 'mission' partition
# never conflicts with the physical 'mission' column).
COLS = ["mission", "centroid_lat", "centroid_lon",
        "max_ht_40dbz", "min_pct_85_89",
        "frac_land", "frac_ocean", "frac_coast"]""")

md("## Load all processed feature files")

co("""files = sorted(glob.glob(f"{PF_ROOT}/features/**/*.parquet", recursive=True))
print(f"{len(files):,} orbit files on disk")

def _read(f):
    try:
        return pq.ParquetFile(f).read(columns=COLS)
    except Exception:
        return None

with ThreadPoolExecutor(max_workers=16) as ex:
    tabs = [t for t in ex.map(_read, files) if t is not None]
df = pa.concat_tables(tabs).to_pandas()
del tabs

# Surface population: ocean if ocean-fraction dominates, land if land-fraction
# dominates, else coastline (mixed margin).
fo, fl = df["frac_ocean"].to_numpy(), df["frac_land"].to_numpy()
df["population"] = np.where(fo >= 0.5, "ocean", np.where(fl >= 0.5, "land", "coast"))
df["ht40_km"] = df["max_ht_40dbz"].to_numpy() / 1000.0   # m -> km

print(f"{len(df):,} features  |  by mission:")
print(df["mission"].value_counts().to_string())
print("\\nby population:")
print(df["population"].value_counts().to_string())""")

md("""## 1 — PF frequency on a 0.5° grid

Counts of PF centroids per 0.5°×0.5° box, log-scaled (each mission on its own swath-limited latitude band).""")

co("""LON_EDGES = np.arange(-180, 180 + GRID_DEG, GRID_DEG)
LAT_EDGES = np.arange(-90, 90 + GRID_DEG, GRID_DEG)

def grid_counts(sub):
    H, _, _ = np.histogram2d(sub["centroid_lon"], sub["centroid_lat"],
                             bins=[LON_EDGES, LAT_EDGES])
    return H.T  # (lat, lon)

proj = dict(projection=ccrs.PlateCarree()) if HAVE_CARTOPY else {}
fig, axes = plt.subplots(2, 1, figsize=(14, 11), subplot_kw=proj,
                         constrained_layout=True)
for ax, mission in zip(axes, ["TRMM", "GPM"]):
    sub = df[df["mission"] == mission]
    H = grid_counts(sub)
    Hm = np.ma.masked_equal(H, 0)
    latband = (sub["centroid_lat"].min(), sub["centroid_lat"].max())
    pcm = ax.pcolormesh(LON_EDGES, LAT_EDGES, np.log10(Hm),
                        cmap="turbo", shading="auto",
                        transform=ccrs.PlateCarree() if HAVE_CARTOPY else None)
    if HAVE_CARTOPY:
        ax.coastlines(linewidth=0.4, color="k")
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="grey")
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
        gl.top_labels = gl.right_labels = False
        ax.set_ylim(max(-67, latband[0] - 2), min(67, latband[1] + 2))
    else:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    cb = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("log$_{10}$(PF count per 0.5° box)")
    ax.set_title(f"{mission}: {len(sub):,} PFs  ({GRID_DEG}° grid)")
plt.show()""")

md("""## 1b — Longitude profile of PF counts near 100°W

A 1°-binned, **linear-count** slice from −105° to −95°E. The 0.5° map above is
`log10`-scaled, which visually exaggerates the apparent "dip" near 100°W. In
linear counts it is a broad, shallow (~25–30 %) minimum — the eastern-Pacific
subsidence zone offshore and the 100th-meridian humid/arid divide over North
America — **not** a binning or PF-algorithm artifact (it varies smoothly with
latitude and appears over both ocean and land).""")

co("""lon_edges = np.arange(-105, -95 + 1, 1.0)            # 1° bins, -105 .. -95
centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
lon = df["centroid_lon"].to_numpy()
pop = df["population"].to_numpy()

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
tot, _ = np.histogram(lon, bins=lon_edges)
ax.bar(centers, tot, width=0.9, color="0.82", edgecolor="0.4",
       label=f"all (n={tot.sum():,})", zorder=1)
for p, c in [("ocean", "tab:blue"), ("land", "tab:green"), ("coast", "tab:orange")]:
    h, _ = np.histogram(lon[pop == p], bins=lon_edges)
    ax.plot(centers, h, "-o", ms=4, lw=1.6, color=c,
            label=f"{p} (n={h.sum():,})", zorder=2)
ax.set_xlabel("longitude [°E]"); ax.set_ylabel("PF count per 1° bin")
ax.set_xticks(np.arange(-105, -94, 1))
ax.set_title("PF centroid counts, 1° longitude bins (−105° to −95°)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.show()""")

md("""## 2 — Extremes: top-10 deepest 40-dBZ echoes & 10 lowest PCTs

Most intense convection by two independent metrics: the **highest 40-dBZ echo top**
(radar intensity / updraft proxy) and the **lowest 85/89-GHz PCT** (large-ice scattering).""")

co("""def topn(sub, col, n=10, largest=True):
    s = sub[np.isfinite(sub[col])]
    return s.nlargest(n, col) if largest else s.nsmallest(n, col)

fig, axes = plt.subplots(2, 2, figsize=(15, 10), subplot_kw=proj,
                         constrained_layout=True)
specs = [("TRMM", "ht40_km", True,  "Top-10 max 40-dBZ height [km]", "Reds"),
         ("GPM",  "ht40_km", True,  "Top-10 max 40-dBZ height [km]", "Reds"),
         ("TRMM", "min_pct_85_89", False, "10 lowest 85-GHz PCT [K]", "Blues_r"),
         ("GPM",  "min_pct_85_89", False, "10 lowest 89-GHz PCT [K]", "Blues_r")]
for ax, (mission, col, largest, label, cmap) in zip(axes.ravel(), specs):
    sub = df[df["mission"] == mission]
    ext = topn(sub, col, 10, largest)
    if HAVE_CARTOPY:
        ax.set_global(); ax.coastlines(linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor="0.92")
        ax.set_ylim(-60, 60)
    sc = ax.scatter(ext["centroid_lon"], ext["centroid_lat"], c=ext[col],
                    cmap=cmap, s=140, edgecolor="k", zorder=5,
                    transform=ccrs.PlateCarree() if HAVE_CARTOPY else None)
    for _, r in ext.iterrows():
        ax.annotate(f"{r[col]:.1f}", (r["centroid_lon"], r["centroid_lat"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points",
                    transform=ccrs.PlateCarree() if HAVE_CARTOPY else ax.transData)
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label=label.split("[")[-1].rstrip("]"))
    ax.set_title(f"{mission} — {label}")
plt.show()

# Printed tables
for mission in ["TRMM", "GPM"]:
    sub = df[df["mission"] == mission]
    print(f"\\n=== {mission}: top-10 deepest 40-dBZ ===")
    print(topn(sub, "ht40_km", 10, True)[["centroid_lat","centroid_lon","ht40_km","min_pct_85_89","population"]].to_string(index=False))
    print(f"\\n=== {mission}: 10 lowest PCT ===")
    print(topn(sub, "min_pct_85_89", 10, False)[["centroid_lat","centroid_lon","min_pct_85_89","ht40_km","population"]].to_string(index=False))""")

md("## 3 — PF counts by surface population")

co("""tab = (df.groupby(["mission", "population"]).size()
         .rename("n_PFs").reset_index()
         .pivot(index="mission", columns="population", values="n_PFs")
         .reindex(columns=["land", "ocean", "coast"]))
tab["total"] = tab.sum(axis=1)
pct = tab[["land","ocean","coast"]].div(tab["total"], axis=0) * 100
summary = tab.copy()
for c in ["land","ocean","coast"]:
    summary[c] = tab[c].map("{:,.0f}".format) + " (" + pct[c].map("{:.1f}%".format) + ")"
summary["total"] = tab["total"].map("{:,.0f}".format)
print("PFs by surface population (count and % of mission):\\n")
print(summary.to_string())
summary""")

md("""## 4 — Distributions by population and sensor

PDFs (density-normalized histograms) of **max 40-dBZ height** and **85/89-GHz PCT**,
split by surface population, for each sensor.""")

co("""pops = ["land", "ocean", "coast"]
colors = {"land": "tab:green", "ocean": "tab:blue", "coast": "tab:orange"}
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
panels = [("TRMM", "ht40_km", "max 40-dBZ height [km]", (0, 20)),
          ("GPM",  "ht40_km", "max 40-dBZ height [km]", (0, 20)),
          ("TRMM", "min_pct_85_89", "85-GHz PCT [K]", (50, 300)),
          ("GPM",  "min_pct_85_89", "89-GHz PCT [K]", (50, 300))]
for ax, (mission, col, xlabel, xr) in zip(axes.ravel(), panels):
    sub = df[df["mission"] == mission]
    for p in pops:
        v = sub.loc[sub["population"] == p, col]
        v = v[np.isfinite(v) & (v > 0 if col == "ht40_km" else True)]
        if len(v) > 10:
            ax.hist(v, bins=60, range=xr, density=True, histtype="step",
                    lw=1.8, color=colors[p], label=f"{p} (n={len(v):,})")
    ax.set_xlabel(xlabel); ax.set_ylabel("PDF"); ax.set_xlim(*xr)
    ax.set_title(f"{mission} — {xlabel.split('[')[0].strip()}")
    ax.legend(fontsize=8)
plt.show()""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                 "language": "python"},
                  "language_info": {"name": "python"}}
out = "/data/keeling/a/snesbitt/python/pf/notebooks/pf_overview.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
