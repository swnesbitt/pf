#!/usr/bin/env python
"""Generate notebooks/pf_grid_viz.ipynb — a gallery of visualizations of the
swath-grid climatology Icechunk stores (GPM & TRMM) on MinIO.

Pattern: read the heavy 0.05° stores ONCE, coarsen-by-sum to 1° into a small
local cache, then do every plot on the cache (fast). Visualizations: global
maps, seasonal cycle, diurnal cycle (local solar time), diurnal peak-hour phase
map, zonal means, class-contribution, GPM-vs-TRMM comparison, Hovmoller.

Run:  python scripts/_build_grid_viz_nb.py
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# PF swath-grid climatology — visualization gallery (GPM & TRMM)

A spread of views of the diurnal/seasonal precipitation-feature climatology stored
in **Icechunk** on MinIO (`pf_grid_{GPM,TRMM}`), dims `month`(1–12) × `hour`(0–23 UTC)
× `lat` × `lon`, with `rain_sum` / `raining_count` (by **size**, **echo-top**,
**rain-type**) and the shared `views` denominator.

**Strategy:** the 0.05° arrays are large, so we read each store **once**, coarsen
by **sum** to 1° (totals & ratios are preserved) into a small local cache, then
every plot below runs on the cache. Run the *Build cache* cell once per mission
(on a compute node); after that everything is fast.

Derived: **rate**=rain/views (mm/hr) · **freq**=raining/views · **intensity**=rain/raining.""")

# ---- setup / open ----
co("""import os, numpy as np, xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _load_minio_env():
    p = os.path.expanduser("~/.spaceborne_minio.env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                k, v = line[7:].split("=", 1)
                os.environ.setdefault(k, v.strip().strip("'\\""))
_load_minio_env()

import icechunk
def open_grid(mission):
    st = icechunk.s3_storage(bucket=os.environ.get("SPACEBORNE_MINIO_BUCKET", "spaceborne-grids"),
        prefix=f"pf_grid_{mission}", endpoint_url=os.environ["SPACEBORNE_MINIO_ENDPOINT"],
        allow_http=True, force_path_style=True, region="us-east-1",
        access_key_id=os.environ["SPACEBORNE_MINIO_ACCESS"],
        secret_access_key=os.environ["SPACEBORNE_MINIO_SECRET"])
    return xr.open_zarr(icechunk.Repository.open(st).readonly_session("main").store, consolidated=False)

MISSIONS = ["GPM", "TRMM"]
BREAKDOWNS = ["size", "echotop", "raintype"]
CACHE = "/data/scratch/a/snesbitt/pf_grid_coarse_{m}.nc"
print("helpers ready")""")

md("""## 1. Build the 1° cache  ⏳ (heavy — run once per mission, on a compute node)
Coarsens every `rain_sum_by_*`, `raining_count_by_*` and `views` from 0.05° to 1°
by **sum** (so rates `rain/views` stay correct), and saves a ~1–2 GB NetCDF. Skip
if the cache already exists.""")

co("""COARSEN = 20   # 0.05 deg * 20 = 1 deg

def build_cache(mission, factor=COARSEN):
    out = CACHE.format(m=mission)
    if os.path.exists(out):
        print("cache exists:", out); return out
    ds = open_grid(mission)
    keep = [v for v in ds.data_vars]  # views + 6 by-breakdown vars
    dsc = ds[keep].coarsen(lat=factor, lon=factor, boundary="trim").sum()
    # carry the class-label coords through
    for c in ds.coords:
        if c.endswith("_label"):
            dsc = dsc.assign_coords({c: ds[c]})
    enc = {v: {"zlib": True, "complevel": 4} for v in dsc.data_vars}
    print(f"{mission}: computing+writing {out} ...", flush=True)
    dsc.to_netcdf(out, engine="h5netcdf", encoding=enc)
    print("wrote", out); return out

# for m in MISSIONS: build_cache(m)   # <-- uncomment & run on a compute node
print("run build_cache('GPM') / build_cache('TRMM') on a compute node if caches are missing")""")

co("""# Load the small 1-degree caches for all plots below
C = {}
for m in MISSIONS:
    p = CACHE.format(m=m)
    if os.path.exists(p):
        C[m] = xr.open_dataset(p)
        print(f"{m}: cache {dict(C[m].sizes)}")
    else:
        print(f"{m}: cache missing -> run build_cache('{m}')")

def labels(ds, bd):
    lab = f"{bd}_label"
    return [str(x) for x in ds[lab].values] if lab in ds.coords else \
           [str(i) for i in ds[f"{bd}_class"].values]

def wlat(ds):   # cosine-latitude area weights
    return np.cos(np.deg2rad(ds["lat"]))

def total_rain(ds):   # sum over a breakdown's classes = total rain field
    return ds["rain_sum_by_raintype"].sum("raintype_class")
def total_raining(ds):
    return ds["raining_count_by_raintype"].sum("raintype_class")""")

md("""## 2. Global maps — unconditional rain rate by subcategory
Annual (all months), all UTC hours. Rows: rate / frequency; columns: classes.""")
co("""def maps_panel(mission, bd, quantity="rate", months=None):
    ds = C[mission]; labs = labels(ds, bd); n = len(labs); cdim = f"{bd}_class"
    sel = dict(month=months) if months else {}
    RS = ds[f"rain_sum_by_{bd}"].sel(**sel).sum(("month", "hour"))
    RC = ds[f"raining_count_by_{bd}"].sel(**sel).sum(("month", "hour"))
    V = ds["views"].sel(**sel).sum(("month", "hour"))
    fig, ax = plt.subplots(1, n, figsize=(3.2*n, 2.6))
    for c in range(n):
        with np.errstate(divide="ignore", invalid="ignore"):
            num = RS.isel({cdim: c}) if quantity == "rate" else RC.isel({cdim: c})
            fld = (num / V).where(V >= 5)
        fld.plot(ax=ax[c], x="lon", y="lat", robust=True, cmap="turbo",
                 add_colorbar=True, cbar_kwargs=dict(shrink=0.7))
        ax[c].set_title(f"{bd}={labs[c]}", fontsize=9); ax[c].set_xlabel(""); ax[c].set_ylabel("")
    fig.suptitle(f"{mission} {quantity} (annual) by {bd}", y=1.05); fig.tight_layout()

if "GPM" in C: maps_panel("GPM", "raintype"); plt.show()""")

md("""## 3. Seasonal cycle — tropical-mean rate & frequency vs month, by rain type""")
co("""def seasonal(mission, bd="raintype", latband=(-30, 30)):
    ds = C[mission].sel(lat=slice(*latband)); labs = labels(ds, bd); cdim = f"{bd}_class"
    w = wlat(ds)
    RS = ds[f"rain_sum_by_{bd}"].sum("hour"); RC = ds[f"raining_count_by_{bd}"].sum("hour")
    V = ds["views"].sum("hour")
    def areamean(x):
        return x.weighted(w).mean(("lat", "lon"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for c in range(len(labs)):
        rate = areamean(RS.isel({cdim: c})) / areamean(V)
        freq = areamean(RC.isel({cdim: c})) / areamean(V)
        ax[0].plot(ds["month"], rate, marker="o", label=labs[c])
        ax[1].plot(ds["month"], freq, marker="o", label=labs[c])
    ax[0].set(title=f"{mission} seasonal rate (mm/hr) [{latband} lat]", xlabel="month", ylabel="rain/views")
    ax[1].set(title=f"{mission} seasonal rain frequency", xlabel="month", ylabel="raining/views")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8); fig.tight_layout()

if "GPM" in C: seasonal("GPM"); plt.show()""")

md("""## 4. Diurnal cycle in **local solar time** — by rain type
Each cell's UTC-hour cycle is shifted to local solar time (`LST = UTC + lon/15`)
before averaging over the region, recovering the true land-afternoon convective peak.""")
co("""def diurnal_lst(mission, bd="raintype", latband=(-30, 30)):
    ds = C[mission].sel(lat=slice(*latband)); labs = labels(ds, bd); cdim = f"{bd}_class"
    w = wlat(ds); lon = ds["lon"].values
    shift = np.round(lon / 15.0).astype(int)   # hours to roll per lon -> LST
    V = ds["views"].sum("month")               # (hour,lat,lon)
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in range(len(labs)):
        RS = ds[f"rain_sum_by_{bd}"].isel({cdim: c}).sum("month")
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = (RS / V).where(V >= 5)                      # (hour,lat,lon), mm/hr
        arr = rate.transpose("hour", "lat", "lon").values
        # roll each lon column so index = local solar hour
        for j in range(arr.shape[2]):
            arr[:, :, j] = np.roll(arr[:, :, j], shift[j], axis=0)
        lst = np.nansum(arr * w.values[None, :, None], axis=(1, 2)) / \
              np.nansum(np.isfinite(arr) * w.values[None, :, None], axis=(1, 2))
        ax.plot(np.arange(24), lst, marker="o", label=labs[c])
    ax.set(title=f"{mission} diurnal cycle of rain rate (local solar time) [{latband} lat]",
           xlabel="local solar hour", ylabel="rain rate (mm/hr)", xticks=range(0, 24, 3))
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()

if "GPM" in C: diurnal_lst("GPM"); plt.show()""")

md("""## 5. Diurnal **peak-hour phase map** (local solar time)
Map of the local solar hour of maximum total rain rate — the classic
afternoon-land / nocturnal-ocean diurnal-phase signature. Cyclic colormap.""")
co("""def phase_map(mission, min_rain=0.02):
    ds = C[mission]; lon = ds["lon"].values
    V = ds["views"].sum("month")
    RS = total_rain(ds).sum("month")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (RS / V).where(V >= 5)            # (hour,lat,lon)
    rate = rate.transpose("hour", "lat", "lon")
    peak_utc = rate.fillna(-1).argmax("hour")     # index of max hour
    meanrate = rate.mean("hour")
    peak_lst = (peak_utc + xr.DataArray(np.round(lon/15).astype(int),
                dims="lon", coords={"lon": ds["lon"]})) % 24
    peak_lst = peak_lst.where(meanrate > min_rain)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    pm = ax.pcolormesh(ds["lon"], ds["lat"], peak_lst, cmap="twilight", vmin=0, vmax=24, shading="auto")
    ax.set_title(f"{mission} — local solar hour of peak rain rate"); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(pm, ax=ax, label="local solar hour", shrink=0.8); fig.tight_layout()

if "GPM" in C: phase_map("GPM"); plt.show()""")

md("""## 6. Zonal mean — rain rate vs latitude, by subcategory (both missions)""")
co("""def zonal(bd="size"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for k, mission in enumerate([m for m in MISSIONS if m in C]):
        ds = C[mission]; labs = labels(ds, bd); cdim = f"{bd}_class"
        V = ds["views"].sum(("month", "hour", "lon"))
        RS = ds[f"rain_sum_by_{bd}"].sum(("month", "hour", "lon"))
        for c in range(len(labs)):
            with np.errstate(divide="ignore", invalid="ignore"):
                ax[k].plot((RS.isel({cdim: c})/V), ds["lat"], label=labs[c])
        ax[k].set(title=f"{mission} zonal rate by {bd}", xlabel="rain/views (mm/hr)")
        ax[k].legend(fontsize=8); ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("latitude"); fig.tight_layout()

if C: zonal("size"); plt.show()""")

md("""## 7. Class contribution — fraction of total rain by class vs latitude (stacked)""")
co("""def contribution(mission, bd="size"):
    ds = C[mission]; labs = labels(ds, bd); cdim = f"{bd}_class"
    RS = ds[f"rain_sum_by_{bd}"].sum(("month", "hour", "lon"))   # (class, lat)
    frac = RS / RS.sum(cdim)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(ds["lat"], *[frac.isel({cdim: c}) for c in range(len(labs))], labels=labs, alpha=0.85)
    ax.set(title=f"{mission} — fraction of total rain by {bd}", xlabel="latitude",
           ylabel="rain fraction", ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()

if "GPM" in C: contribution("GPM", "size"); plt.show()""")

md("""## 8. GPM vs TRMM — overlap comparison (±38° band)
Both on the same grid, so map the annual rate difference and scatter the cells.""")
co("""def gpm_vs_trmm(latband=(-37, 37)):
    if not all(m in C for m in MISSIONS):
        print("need both caches"); return
    def annual_rate(m):
        ds = C[m].sel(lat=slice(*latband))
        return (total_rain(ds).sum(("month", "hour")) /
                ds["views"].sum(("month", "hour"))).where(ds["views"].sum(("month","hour")) >= 5)
    g, t = annual_rate("GPM"), annual_rate("TRMM")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    (g - t).plot(ax=ax[0], x="lon", y="lat", robust=True, cmap="RdBu_r",
                 cbar_kwargs=dict(shrink=0.7, label="GPM - TRMM (mm/hr)"))
    ax[0].set_title("annual rate difference (±38°)")
    gv, tv = g.values.ravel(), t.values.ravel()
    ok = np.isfinite(gv) & np.isfinite(tv)
    ax[1].hexbin(tv[ok], gv[ok], gridsize=50, bins="log", cmap="viridis")
    lim = np.nanpercentile(np.r_[gv[ok], tv[ok]], 99)
    ax[1].plot([0, lim], [0, lim], "r--", lw=1)
    ax[1].set(title="cell-by-cell rate", xlabel="TRMM (mm/hr)", ylabel="GPM (mm/hr)",
              xlim=(0, lim), ylim=(0, lim)); fig.tight_layout()

gpm_vs_trmm(); plt.show()""")

md("""## 9. Hovmöller — latitude × month rain rate (seasonal migration of the ITCZ etc.)""")
co("""def hovmoller(mission):
    ds = C[mission]
    rate = (total_rain(ds).sum("hour") / ds["views"].sum("hour")).where(ds["views"].sum("hour") >= 5)
    zm = rate.mean("lon")    # (month, lat)
    fig, ax = plt.subplots(figsize=(7, 5))
    pm = ax.pcolormesh(ds["month"], ds["lat"], zm.T, cmap="turbo", shading="auto")
    ax.set(title=f"{mission} — zonal-mean rain rate (lat × month)", xlabel="month", ylabel="latitude",
           xticks=range(1, 13))
    plt.colorbar(pm, ax=ax, label="rain/views (mm/hr)", shrink=0.85); fig.tight_layout()

if "GPM" in C: hovmoller("GPM"); plt.show()""")

md("""## 10. Interactive explorer (single map, any selection)""")
co("""import ipywidgets as W
def _go(mission, bd, cls, quantity, month, hour):
    ds = C[mission]; cdim = f"{bd}_class"; labs = labels(ds, bd); cls = min(cls, len(labs)-1)
    msel = {} if month == "all" else dict(month=int(month))
    hsel = {} if hour == "all" else dict(hour=int(hour))
    RS = ds[f"rain_sum_by_{bd}"].isel({cdim: cls}).sel(**msel, **hsel)
    RC = ds[f"raining_count_by_{bd}"].isel({cdim: cls}).sel(**msel, **hsel)
    V = ds["views"].sel(**msel, **hsel)
    for d in ("month", "hour"):
        if d in RS.dims: RS = RS.sum(d); RC = RC.sum(d); V = V.sum(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        fld = {"rate": RS/V, "freq": RC/V, "intensity": RS/RC}[quantity].where(V >= 5)
    fld.plot(figsize=(11, 4), x="lon", y="lat", robust=True, cmap="turbo")
    plt.title(f"{mission} {quantity} {bd}={labs[cls]} month={month} hour={hour}"); plt.show()

if C:
    W.interact(_go, mission=W.Dropdown(options=list(C)),
               bd=W.Dropdown(options=BREAKDOWNS, value="size"),
               cls=W.IntSlider(min=0, max=4, value=1),
               quantity=W.Dropdown(options=["rate", "freq", "intensity"]),
               month=W.Dropdown(options=["all"]+[str(i) for i in range(1,13)]),
               hour=W.Dropdown(options=["all"]+[str(i) for i in range(24)]))""")

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "pf_grid_viz.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out, "—", len(cells), "cells")
