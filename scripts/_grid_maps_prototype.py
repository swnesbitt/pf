#!/usr/bin/env python
"""Prototype maps from the Icechunk grid stores — rate/freq/intensity for every
subcategory, GPM & TRMM. Headless (Agg), saves PNGs. Run on a COMPUTE node.

Efficient: reads each (summed) array ONCE into memory, then derives all
class x quantity panels with numpy (no per-panel re-reads).

  python scripts/_grid_maps_prototype.py [MONTHS]   # e.g. "1" or "1,2"; default "1"
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import icechunk

ENDPOINT = os.environ["SPACEBORNE_MINIO_ENDPOINT"]
ACCESS = os.environ["SPACEBORNE_MINIO_ACCESS"]
SECRET = os.environ["SPACEBORNE_MINIO_SECRET"]
BUCKET = os.environ.get("SPACEBORNE_MINIO_BUCKET", "spaceborne-grids")
MONTHS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1"])]
NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
         8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
MLABEL = "+".join(NAMES[x] for x in MONTHS)
OUT = "/data/scratch/a/snesbitt/pf_grid_maps"
os.makedirs(OUT, exist_ok=True)
BREAKDOWNS = ["size", "echotop", "raintype"]
MIN_VIEWS = 5
CMAP = {"rate": "turbo", "freq": "viridis", "intensity": "magma"}


def open_mission(m):
    st = icechunk.s3_storage(bucket=BUCKET, prefix=f"pf_grid_{m}", endpoint_url=ENDPOINT,
                             allow_http=ENDPOINT.startswith("http://"), force_path_style=True,
                             region="us-east-1", access_key_id=ACCESS, secret_access_key=SECRET)
    return xr.open_zarr(icechunk.Repository.open(st).readonly_session("main").store,
                        consolidated=False)


def labels(ds, bd):
    lab = f"{bd}_label"
    return ([str(x) for x in ds[lab].values] if lab in ds.coords
            else [str(i) for i in ds[f"{bd}_class"].values])


def plot_map(da_lon, da_lat, arr, title, ax, cmap):
    finite = np.isfinite(arr)
    vmin = float(np.nanpercentile(arr, 2)) if finite.any() else 0.0
    vmax = float(np.nanpercentile(arr, 98)) if finite.any() else 1.0
    pm = ax.pcolormesh(da_lon, da_lat, arr, cmap=cmap, shading="auto",
                       rasterized=True, vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-180, 180); ax.set_ylim(float(da_lat.min()), float(da_lat.max()))
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(pm, ax=ax, shrink=0.8, pad=0.01)


for m in ("GPM", "TRMM"):
    try:
        ds = open_mission(m)
    except Exception as e:
        print(f"{m}: skip ({e})", flush=True); continue
    lon = ds["lon"].values; lat = ds["lat"].values
    t0 = time.time()
    V = ds["views"].sel(month=MONTHS).sum(("month", "hour")).load().values.astype("f8")
    print(f"{m}: loaded views {V.shape} in {time.time()-t0:.0f}s", flush=True)
    for bd in BREAKDOWNS:
        t0 = time.time()
        RS = ds[f"rain_sum_by_{bd}"].sel(month=MONTHS).sum(("month", "hour")).load().values.astype("f8")
        RC = ds[f"raining_count_by_{bd}"].sel(month=MONTHS).sum(("month", "hour")).load().values.astype("f8")
        labs = labels(ds, bd); n = len(labs)
        print(f"{m} {bd}: loaded {RS.shape} in {time.time()-t0:.0f}s", flush=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(V >= MIN_VIEWS, RS / V, np.nan)        # (class,lat,lon)
            freq = np.where(V >= MIN_VIEWS, RC / V, np.nan)
            inten = np.where(RC >= 1, RS / RC, np.nan)
        rows = [("unconditional rate (rain/views)", rate),
                ("rain frequency (raining/views)", freq),
                ("conditional intensity (rain/raining)", inten)]
        fig, axes = plt.subplots(3, n, figsize=(3.1 * n, 7.6), squeeze=False)
        for r, (qname, cube) in enumerate(rows):
            qkey = ["rate", "freq", "intensity"][r]
            for c in range(n):
                plot_map(lon, lat, cube[c], f"{qkey} | {bd}={labs[c]}", axes[r][c], CMAP[qkey])
        fig.suptitle(f"{m} — {bd} subcategories — {MLABEL} climatology (all years, all UTC hours)\n"
                     f"rows: unconditional rate · rain frequency · conditional intensity",
                     y=1.01, fontsize=10)
        fig.tight_layout()
        p = f"{OUT}/{m}_{bd}_{MLABEL}.png"
        fig.savefig(p, dpi=95, bbox_inches="tight"); plt.close(fig)
        print("wrote", p, flush=True)
print("DONE", flush=True)
