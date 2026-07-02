#!/usr/bin/env python
"""Generate notebooks/nesbitt_anders_2009.ipynb — stage a COMBINED GPM+TRMM
dataset on the shared grid and recreate the key analysis of

  Nesbitt, S. W., and E. A. Anders (2009), Very high resolution precipitation
  climatologies from the Tropical Rainfall Measuring Mission precipitation
  radar, Geophys. Res. Lett., 36, L15815, doi:10.1029/2009GL038026.

i.e. very-high-resolution (0.05 deg) tropical maps of unconditional rain rate,
rain frequency, and conditional rain rate from the radar — and the central
result that **conditional rain rate and rain frequency depend strongly on grid
box size** (coarsening smears peaks and fills dry areas) while the area-mean
**accumulation is scale-conserved**.

Run:  python scripts/_build_nesbitt_anders_nb.py
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Recreating Nesbitt & Anders (2009) — very-high-resolution tropical radar precip climatology

> Nesbitt, S. W., and E. A. Anders (2009), *Very high resolution precipitation
> climatologies from the TRMM precipitation radar*, **GRL** 36, L15815,
> doi:10.1029/2009GL038026.

This notebook (a) **stages a combined GPM+TRMM dataset** on the shared 0.05° grid,
and (b) reproduces the paper's three core climatologies and its headline
**resolution-dependence** result, using our swath-grid Icechunk stores.

Per 0.05° cell (from the radar pixel-views):
- **unconditional rain rate** = Σrain / Nviews  (mm/hr) — area-mean rate / "accumulation"
- **rain frequency** = Nraining / Nviews  — fraction of overpasses that were raining
- **conditional rain rate** = Σrain / Nraining  (mm/hr) — intensity *when raining*

The paper's point: at fine resolution the radar resolves sharp coastal/orographic
gradients and a broad rain-rate distribution; **coarsening systematically raises
apparent rain frequency and lowers conditional intensity**, while the unconditional
mean is conserved. TRMM PR (their instrument) is here joined by GPM Ku.""")

co("""import os, numpy as np, xarray as xr
import matplotlib.pyplot as plt

def _load_minio_env():
    p = os.path.expanduser("~/.spaceborne_minio.env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                k, v = line[7:].split("=", 1); os.environ.setdefault(k, v.strip().strip("'\\""))
_load_minio_env()
import icechunk
def open_grid(mission):
    st = icechunk.s3_storage(bucket=os.environ.get("SPACEBORNE_MINIO_BUCKET","spaceborne-grids"),
        prefix=f"pf_grid_{mission}", endpoint_url=os.environ["SPACEBORNE_MINIO_ENDPOINT"],
        allow_http=True, force_path_style=True, region="us-east-1",
        access_key_id=os.environ["SPACEBORNE_MINIO_ACCESS"], secret_access_key=os.environ["SPACEBORNE_MINIO_SECRET"])
    return xr.open_zarr(icechunk.Repository.open(st).readonly_session("main").store, consolidated=False)

TROPICS = (-37, 37)          # TRMM PR coverage band (Nesbitt & Anders domain)
STAGE = "/data/scratch/a/snesbitt/na2009_stage.nc"
print("ready")""")

md("""## 1. Stage the combined climatology  ⏳ (heavy — run ONCE on a compute node)
Reads each native 0.05° store over the tropics, collapses month & hour & class to
three summed fields — **Σrain**, **Σraining**, **Nviews** — per mission, then
**pools GPM+TRMM** (sum) into a `COMBINED` member. Saves a small NetCDF (~0.5 GB).
Total rain = Σ over a breakdown's classes (rain in any class = total rain).""")

co("""def stage_combined(out=STAGE, lat=TROPICS):
    if os.path.exists(out):
        print("stage exists:", out); return out
    fields = {}
    for m in ("GPM", "TRMM"):
        ds = open_grid(m).sel(lat=slice(*lat))
        rain = ds["rain_sum_by_raintype"].sum(("raintype_class", "month", "hour"))
        rning = ds["raining_count_by_raintype"].sum(("raintype_class", "month", "hour"))
        views = ds["views"].sum(("month", "hour"))
        print(f"{m}: reducing native tropics ...", flush=True)
        fields[m] = xr.Dataset(dict(rain=rain, raining=rning, views=views)).compute()
    comb = sum(fields.values())                        # pool GPM+TRMM by summing
    out_ds = xr.concat([fields["GPM"], fields["TRMM"], comb],
                       dim=xr.Variable("member", ["GPM", "TRMM", "COMBINED"]))
    out_ds.to_netcdf(out, engine="h5netcdf",
                     encoding={v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars})
    print("wrote", out); return out

# stage_combined()    # <-- uncomment and run on a compute node (srun)
print("run stage_combined() on a compute node if", STAGE, "is missing")""")

co("""S = xr.open_dataset(STAGE) if os.path.exists(STAGE) else None
if S is not None:
    print("staged:", dict(S.sizes), "| members:", list(S["member"].values))

def stats(ds, member, min_views=10):
    \"\"\"unconditional rate, frequency, conditional rate at native 0.05 deg.\"\"\"
    d = ds.sel(member=member)
    V, R, RN = d["views"], d["rain"], d["raining"]
    with np.errstate(divide="ignore", invalid="ignore"):
        uncond = (R / V).where(V >= min_views)          # mm/hr
        freq = (RN / V).where(V >= min_views)           # fraction
        cond = (R / RN).where(RN >= 1)                  # mm/hr
    return uncond, freq, cond""")

md("""## 2. Figure 1 — the three high-resolution climatologies (combined product)
Unconditional rain rate, rain frequency, and conditional rain rate at 0.05°
across the tropics (cf. Nesbitt & Anders Fig. 1).""")
co("""def fig_three(member="COMBINED"):
    u, f, c = stats(S, member)
    fig, ax = plt.subplots(3, 1, figsize=(13, 9))
    u.plot(ax=ax[0], x="lon", y="lat", robust=True, cmap="turbo",
           cbar_kwargs=dict(label="mm/hr", shrink=0.8))
    ax[0].set_title(f"{member}: unconditional rain rate (Σrain/Nviews)")
    (100*f).plot(ax=ax[1], x="lon", y="lat", robust=True, cmap="viridis",
                 cbar_kwargs=dict(label="%", shrink=0.8))
    ax[1].set_title(f"{member}: rain frequency (Nraining/Nviews)")
    c.plot(ax=ax[2], x="lon", y="lat", robust=True, cmap="magma",
           cbar_kwargs=dict(label="mm/hr", shrink=0.8))
    ax[2].set_title(f"{member}: conditional rain rate (Σrain/Nraining)")
    for a in ax: a.set_xlabel(""); a.set_ylabel("lat")
    fig.tight_layout()

if S is not None: fig_three(); plt.show()""")

md("""## 3. Figure 2 — fine-scale structure (regional zooms)
The motivation for very-high resolution: sharp coastal/orographic gradients the
radar resolves that coarse grids smear. Pick a window (defaults: Maritime
Continent).""")
co("""def zoom(member="COMBINED", lon=(90, 150), lat=(-12, 12), what="uncond"):
    u, f, c = stats(S, member)
    da = {"uncond": u, "freq": 100*f, "cond": c}[what].sel(lon=slice(*lon), lat=slice(*lat))
    da.plot(figsize=(11, 5), x="lon", y="lat", robust=True,
            cmap={"uncond": "turbo", "freq": "viridis", "cond": "magma"}[what])
    plt.title(f"{member} {what} — 0.05° detail  lon{lon} lat{lat}")
    plt.tight_layout()

if S is not None:
    zoom(what="uncond"); plt.show()       # Maritime Continent
    zoom(lon=(-95,-75), lat=(-5,20), what="uncond"); plt.show()   # Central America / E. Pacific ITCZ""")

md("""## 4. Figure 3 — **resolution dependence** (the paper's headline)
Coarsen the 0.05° **box-mean** rain rate to a range of grid sizes; over the tropical
domain track how the three statistics change with box size. Accumulation (mean
unconditional rate) is conserved; **conditional rate falls and rain frequency rises**
as boxes grow (peaks smeared, dry cells wetted).""")
co("""def resolution_curve(member="COMBINED", rates=(1, 2, 5, 10, 20, 50), eps=0.005):
    d = S.sel(member=member)
    base_deg = 0.05
    res_deg, mean_rate, cond_rate, freq_exc = [], [], [], []
    for f in rates:
        # block-sum -> box totals; box-mean unconditional rate = Σrain/Σviews
        rb = d["rain"].coarsen(lat=f, lon=f, boundary="trim").sum()
        vb = d["views"].coarsen(lat=f, lon=f, boundary="trim").sum()
        rnb = d["raining"].coarsen(lat=f, lon=f, boundary="trim").sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = (rb / vb).where(vb >= 10)            # box-mean rate (mm/hr)
            condb = (rb / rnb).where(rnb >= 1)          # box conditional rate
        wet = rate > eps
        res_deg.append(base_deg * f)
        mean_rate.append(float(rate.mean()))            # domain mean (≈ conserved)
        cond_rate.append(float(condb.where(wet).mean()))
        freq_exc.append(float(wet.mean()))              # fraction of boxes "raining"
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(res_deg, mean_rate, "o-"); ax[0].set(title="mean unconditional rate (conserved)",
              xlabel="grid box (deg)", ylabel="mm/hr", xscale="log")
    ax[1].plot(res_deg, cond_rate, "o-", color="firebrick"); ax[1].set(title="conditional rain rate",
              xlabel="grid box (deg)", ylabel="mm/hr", xscale="log")
    ax[2].plot(res_deg, freq_exc, "o-", color="seagreen"); ax[2].set(title="rain frequency (boxes wet)",
              xlabel="grid box (deg)", ylabel="fraction", xscale="log")
    for a in ax: a.grid(alpha=0.3)
    fig.suptitle(f"{member} — resolution dependence over the tropics", y=1.04); fig.tight_layout()

if S is not None: resolution_curve(); plt.show()""")

md("""## 5. Figure 4 — rain-rate PDF narrows with averaging
The mechanism behind Fig. 3: the distribution of box-mean rain rate collapses
toward the mean as resolution coarsens (extreme rates lost, zeros filled).""")
co("""def rate_pdf(member="COMBINED", rates=(1, 4, 20), bins=np.logspace(-3, 1, 60)):
    d = S.sel(member=member)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for f in rates:
        rb = d["rain"].coarsen(lat=f, lon=f, boundary="trim").sum()
        vb = d["views"].coarsen(lat=f, lon=f, boundary="trim").sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = (rb / vb).where(vb >= 10).values.ravel()
        rate = rate[np.isfinite(rate) & (rate > 0)]
        ax.hist(rate, bins=bins, histtype="step", density=True, lw=2, label=f"{0.05*f:.2f}°")
    ax.set(xscale="log", xlabel="box-mean rain rate (mm/hr)", ylabel="density",
           title=f"{member} — rain-rate PDF vs grid box size")
    ax.legend(title="resolution"); ax.grid(alpha=0.3); fig.tight_layout()

if S is not None: rate_pdf(); plt.show()""")

md("""## 6. GPM vs TRMM vs COMBINED — consistency of the high-res climatology
Zonal-mean unconditional rate for each member: TRMM (paper's instrument), GPM,
and the pooled product (more samples → smoother).""")
co("""def member_compare():
    fig, ax = plt.subplots(figsize=(7, 5))
    for member in ["TRMM", "GPM", "COMBINED"]:
        d = S.sel(member=member)
        with np.errstate(divide="ignore", invalid="ignore"):
            zm = (d["rain"].sum("lon") / d["views"].sum("lon")).where(d["views"].sum("lon") >= 50)
        ax.plot(zm, S["lat"], label=member, lw=2)
    ax.set(xlabel="unconditional rain rate (mm/hr)", ylabel="latitude",
           title="zonal-mean rate — TRMM / GPM / combined"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()

if S is not None: member_compare(); plt.show()""")

md("""---
### Notes
- The **combined** member simply pools pixel-views (Σrain, Σraining, Nviews) from
  both radars — legitimate as a sampling union on the identical grid, though GPM Ku
  and TRMM PR differ in sensitivity/era (caveat for quantitative use).
- Faithful Nesbitt & Anders (2009) reproduction uses the **TRMM** member; GPM
  extends it past 2014 and to ±68°.
- All three statistics here come straight from the radar pixel-views in the
  Icechunk stores — no external rain product.""")

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "nesbitt_anders_2009.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out, "—", len(cells), "cells")
