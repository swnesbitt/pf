#!/usr/bin/env python
"""Generate notebooks/pf_grid_icechunk_maps.ipynb — maps of the swath-grid
climatology (rain / raining-views / views and their derived rates) for every
storm subcategory, for TRMM and GPM, read straight from the Icechunk stores on
MinIO.

Run:  python scripts/_build_grid_icechunk_nb.py
Then: source ~/.spaceborne_minio.env && jupyter lab notebooks/pf_grid_icechunk_maps.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# PF swath-grid climatology — maps by storm subcategory (TRMM & GPM)

Reads the **Icechunk** stores `s3://spaceborne-grids/pf_grid_{GPM,TRMM}` from MinIO
and maps, on the shared 0.05° grid, the three gridded quantities and their derived
rates — for every **size**, **echo-top**, and **rain-type** subcategory.

Per 0.05° cell (dims `month`=1–12 of year, `hour`=0–23 UTC):
- **rain_sum** `rain_sum_by_{size,echotop,raintype}` — Σ near-surface rate (mm/hr, a sum of instantaneous rates)
- **raining_count** `raining_count_by_{...}` — # raining pixel-views
- **views** — # radar pixel-views (the shared, un-stratified sampling denominator)

Derived: **rate** = rain/views (unconditional mm/hr) · **freq** = raining/views · **intensity** = rain/raining (conditional mm/hr).

> The stores are written with one commit per month-of-year, so this notebook works
> even **mid-build** — uncommitted months read as zeros and fill in as the Stage-2
> jobs finish. Re-run the cells to refresh.""")

co("""import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- MinIO creds: from env (source ~/.spaceborne_minio.env) or the creds file ---
def _load_minio_env():
    need = ("SPACEBORNE_MINIO_ENDPOINT", "SPACEBORNE_MINIO_ACCESS",
            "SPACEBORNE_MINIO_SECRET")
    if all(k in os.environ for k in need):
        return
    p = os.path.expanduser("~/.spaceborne_minio.env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                k, v = line[len("export "):].split("=", 1)
                os.environ.setdefault(k, v.strip().strip("'\\""))
_load_minio_env()
ENDPOINT = os.environ["SPACEBORNE_MINIO_ENDPOINT"]
ACCESS = os.environ["SPACEBORNE_MINIO_ACCESS"]
SECRET = os.environ["SPACEBORNE_MINIO_SECRET"]
BUCKET = os.environ.get("SPACEBORNE_MINIO_BUCKET", "spaceborne-grids")
print("MinIO:", ENDPOINT, "bucket", BUCKET)""")

co("""import icechunk

def open_mission(mission):
    \"\"\"Open a mission's Icechunk store on MinIO as an xarray Dataset (lazy).\"\"\"
    storage = icechunk.s3_storage(
        bucket=BUCKET, prefix=f"pf_grid_{mission.upper()}",
        endpoint_url=ENDPOINT, allow_http=ENDPOINT.startswith("http://"),
        force_path_style=True, region="us-east-1",
        access_key_id=ACCESS, secret_access_key=SECRET)
    repo = icechunk.Repository.open(storage)
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    return ds

DS = {}
for m in ("GPM", "TRMM"):
    try:
        DS[m] = open_mission(m)
        print(f"{m}: opened {dict(DS[m].sizes)}")
    except Exception as e:
        print(f"{m}: not available yet ({type(e).__name__}: {str(e)[:80]})")
""")

co("""# Subcategory labels (fall back to indices if label coords are absent)
BREAKDOWNS = ["size", "echotop", "raintype"]
UNITS = {"rain": "mm/hr (Σ rate)", "raining_views": "count", "views": "count",
         "rate": "mm/hr", "freq": "fraction", "intensity": "mm/hr"}

def labels(ds, breakdown):
    cdim = f"{breakdown}_class"
    lab = f"{breakdown}_label"
    if lab in ds.coords:
        return [str(x) for x in ds[lab].values]
    return [str(i) for i in ds[cdim].values]

def field(ds, breakdown, cls, quantity, months=None, hours=None):
    \"\"\"Climatological map (sum over selected months & hours) of one quantity for
    one subcategory class. quantity in {rain, raining_views, views, rate, freq,
    intensity}. months/hours: None = all, else a list/slice.\"\"\"
    cdim = f"{breakdown}_class"
    rs = ds[f"rain_sum_by_{breakdown}"].isel({cdim: cls})
    rc = ds[f"raining_count_by_{breakdown}"].isel({cdim: cls})
    vv = ds["views"]
    if months is not None:
        rs = rs.sel(month=months); rc = rc.sel(month=months); vv = vv.sel(month=months)
    if hours is not None:
        rs = rs.sel(hour=hours); rc = rc.sel(hour=hours); vv = vv.sel(hour=hours)
    RS = rs.sum(("month", "hour")); RC = rc.sum(("month", "hour")); V = vv.sum(("month", "hour"))
    with np.errstate(divide="ignore", invalid="ignore"):
        if quantity == "rain":          out = RS
        elif quantity == "raining_views": out = RC
        elif quantity == "views":        out = V
        elif quantity == "rate":         out = RS / V
        elif quantity == "freq":         out = RC / V
        elif quantity == "intensity":    out = RS / RC
        else: raise ValueError(quantity)
    return out.where(np.isfinite(out))
""")

co("""def plot_map(da, title="", ax=None, cmap="turbo", log=False, vmax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.2))
    arr = da
    norm = mcolors.LogNorm() if log else None
    pm = ax.pcolormesh(da["lon"], da["lat"], arr, cmap=cmap, norm=norm,
                       vmax=(None if log else vmax), shading="auto", rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-180, 180); ax.set_ylim(float(da["lat"].min()), float(da["lat"].max()))
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(pm, ax=ax, shrink=0.85, pad=0.01)
    return ax

# quick smoke: one map (annual, all hours) — unconditional rate, convective rain type
m = "GPM" if "GPM" in DS else next(iter(DS), None)
if m:
    da = field(DS[m], "raintype", 1, "rate")   # raintype idx 1 = convective
    plot_map(da, f"{m}  unconditional rain rate  (raintype=convective, annual, all hours)")
    plt.show()
""")

md("""## Subcategory panels — the three quantities × every class

For a mission and a breakdown, a grid of maps: **rows = the derived trio**
(unconditional rate, rain frequency, conditional intensity), **columns = each
subcategory class**. Annual (all months), all UTC hours. Swap `quantities` to
`["rain","raining_views","views"]` for the raw stored arrays instead.""")

co("""def panel(mission, breakdown, quantities=("rate", "freq", "intensity"),
          months=None, hours=None, cmaps=None, logs=None):
    ds = DS[mission]
    labs = labels(ds, breakdown)
    n = len(labs)
    cmaps = cmaps or {"rate": "turbo", "freq": "viridis", "intensity": "magma",
                      "rain": "turbo", "raining_views": "cividis", "views": "bone"}
    logs = logs or {"views": True, "rain": True, "raining_views": True}
    fig, axes = plt.subplots(len(quantities), n, figsize=(3.2 * n, 2.6 * len(quantities)),
                             squeeze=False)
    for r, q in enumerate(quantities):
        for c in range(n):
            da = field(ds, breakdown, c, q, months=months, hours=hours)
            plot_map(da, f"{q} | {breakdown}={labs[c]}", ax=axes[r][c],
                     cmap=cmaps.get(q, "turbo"), log=logs.get(q, False))
    fig.suptitle(f"{mission} — {breakdown} subcategories  "
                 f"(months={'all' if months is None else months}, "
                 f"hours={'all' if hours is None else hours})", y=1.01)
    fig.tight_layout()
    return fig

# Example: GPM size breakdown
if "GPM" in DS:
    panel("GPM", "size"); plt.show()
""")

md("### All breakdowns, both missions\nHeavy (many panels) — run when the stores are fully built.")

co("""for mission in [x for x in ("GPM", "TRMM") if x in DS]:
    for bd in BREAKDOWNS:
        panel(mission, bd)
        plt.show()
""")

md("""## Interactive explorer
Pick mission · quantity · breakdown · class · month(s) · hour(s) and draw a single map.""")

co("""import ipywidgets as W
from IPython.display import display

def _explore(mission, quantity, breakdown, cls, month, hour):
    ds = DS[mission]
    months = None if month == "all" else [int(month)]
    hours = None if hour == "all" else [int(hour)]
    labs = labels(ds, breakdown)
    cls = min(cls, len(labs) - 1)
    da = field(ds, breakdown, cls, quantity, months=months, hours=hours)
    plot_map(da, f"{mission} · {quantity} · {breakdown}={labs[cls]} · "
                 f"month={month} · hour={hour}  [{UNITS[quantity]}]",
             log=(quantity in ("rain", "raining_views", "views")))
    plt.show()

if DS:
    miss = list(DS)
    W.interact(
        _explore,
        mission=W.Dropdown(options=miss, value=miss[0]),
        quantity=W.Dropdown(options=["rate", "freq", "intensity",
                                     "rain", "raining_views", "views"], value="rate"),
        breakdown=W.Dropdown(options=BREAKDOWNS, value="size"),
        cls=W.IntSlider(min=0, max=4, value=1, description="class idx"),
        month=W.Dropdown(options=["all"] + [str(i) for i in range(1, 13)], value="all"),
        hour=W.Dropdown(options=["all"] + [str(i) for i in range(24)], value="all"),
    )
else:
    print("No stores open yet — re-run the open cell once the Stage-2 builds land.")
""")

md("""## Diurnal bonus
Because the grid keeps a UTC-hour axis, you can also map a single hour, or the
**diurnal amplitude** of any quantity. Example: convective rain frequency at 18Z
minus 06Z (a crude land-afternoon vs. night contrast).""")

co("""if DS:
    m = "GPM" if "GPM" in DS else list(DS)[0]
    a = field(DS[m], "raintype", 1, "freq", hours=[18])
    b = field(DS[m], "raintype", 1, "freq", hours=[6])
    plot_map((a - b), f"{m}  convective rain-freq  18Z - 06Z  (diurnal contrast)",
             cmap="RdBu_r", vmax=None)
    plt.show()
""")

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "pf_grid_icechunk_maps.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)
