#!/usr/bin/env python
"""Headless: stage the combined GPM+TRMM tropical climatology and render the
Nesbitt & Anders (2009) recreation figures. Run on a COMPUTE node."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import icechunk

TROPICS = (-37, 37)
STAGE = "/data/scratch/a/snesbitt/na2009_stage.nc"
OUT = "/data/scratch/a/snesbitt/pf_grid_maps"
os.makedirs(OUT, exist_ok=True)


def open_grid(m):
    st = icechunk.s3_storage(bucket="spaceborne-grids", prefix=f"pf_grid_{m}",
        endpoint_url=os.environ["SPACEBORNE_MINIO_ENDPOINT"], allow_http=True,
        force_path_style=True, region="us-east-1",
        access_key_id=os.environ["SPACEBORNE_MINIO_ACCESS"],
        secret_access_key=os.environ["SPACEBORNE_MINIO_SECRET"])
    return xr.open_zarr(icechunk.Repository.open(st).readonly_session("main").store, consolidated=False)


def stage():
    if os.path.exists(STAGE):
        print("stage exists", flush=True); return
    fields = {}
    for m in ("GPM", "TRMM"):
        ds = open_grid(m).sel(lat=slice(*TROPICS))
        print(f"staging {m} ...", flush=True)
        fields[m] = xr.Dataset(dict(
            rain=ds["rain_sum_by_raintype"].sum(("raintype_class", "month", "hour")),
            raining=ds["raining_count_by_raintype"].sum(("raintype_class", "month", "hour")),
            views=ds["views"].sum(("month", "hour")))).compute()
        print(f"  {m} done", flush=True)
    comb = fields["GPM"] + fields["TRMM"]
    out = xr.concat([fields["GPM"], fields["TRMM"], comb],
                    dim=xr.Variable("member", ["GPM", "TRMM", "COMBINED"]))
    out.to_netcdf(STAGE, engine="h5netcdf",
                  encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
    print("wrote", STAGE, flush=True)


def stats(S, member, mv=10):
    d = S.sel(member=member)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (d.rain / d.views).where(d.views >= mv)
        f = (d.raining / d.views).where(d.views >= mv)
        c = (d.rain / d.raining).where(d.raining >= 1)
    return u, f, c


stage()
S = xr.open_dataset(STAGE)
print("staged", dict(S.sizes), flush=True)

# Fig 1: three climatologies (combined)
u, f, c = stats(S, "COMBINED")
fig, ax = plt.subplots(3, 1, figsize=(13, 9))
u.plot(ax=ax[0], x="lon", y="lat", robust=True, cmap="turbo", cbar_kwargs=dict(label="mm/hr", shrink=0.8))
ax[0].set_title("COMBINED GPM+TRMM: unconditional rain rate (Σrain/Nviews)")
(100 * f).plot(ax=ax[1], x="lon", y="lat", robust=True, cmap="viridis", cbar_kwargs=dict(label="%", shrink=0.8))
ax[1].set_title("rain frequency (Nraining/Nviews)")
c.plot(ax=ax[2], x="lon", y="lat", robust=True, cmap="magma", cbar_kwargs=dict(label="mm/hr", shrink=0.8))
ax[2].set_title("conditional rain rate (Σrain/Nraining)")
for a in ax: a.set_xlabel(""); a.set_ylabel("lat")
fig.suptitle("Nesbitt & Anders (2009) recreation — high-res tropical radar precip climatology", y=1.005)
fig.tight_layout(); fig.savefig(f"{OUT}/na2009_climatologies.png", dpi=95, bbox_inches="tight"); plt.close(fig)
print("wrote na2009_climatologies.png", flush=True)

# Fig 2: resolution dependence
d = S.sel(member="COMBINED"); rates = (1, 2, 5, 10, 20, 50); eps = 0.005
res, mr, cr, fe = [], [], [], []
for ff in rates:
    rb = d.rain.coarsen(lat=ff, lon=ff, boundary="trim").sum()
    vb = d.views.coarsen(lat=ff, lon=ff, boundary="trim").sum()
    rnb = d.raining.coarsen(lat=ff, lon=ff, boundary="trim").sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (rb / vb).where(vb >= 10); condb = (rb / rnb).where(rnb >= 1)
    wet = rate > eps
    res.append(0.05 * ff); mr.append(float(rate.mean()))
    cr.append(float(condb.where(wet).mean())); fe.append(float(wet.mean()))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(res, mr, "o-"); ax[0].set(title="mean unconditional rate (conserved)", xlabel="grid box (deg)", ylabel="mm/hr", xscale="log")
ax[1].plot(res, cr, "o-", color="firebrick"); ax[1].set(title="conditional rain rate", xlabel="grid box (deg)", ylabel="mm/hr", xscale="log")
ax[2].plot(res, fe, "o-", color="seagreen"); ax[2].set(title="rain frequency (boxes wet)", xlabel="grid box (deg)", ylabel="fraction", xscale="log")
for a in ax: a.grid(alpha=0.3)
fig.suptitle("Resolution dependence over the tropics (combined) — the paper's headline", y=1.04)
fig.tight_layout(); fig.savefig(f"{OUT}/na2009_resolution.png", dpi=95, bbox_inches="tight"); plt.close(fig)
print("wrote na2009_resolution.png", flush=True)

# Fig 3: rate PDF vs resolution
fig, ax = plt.subplots(figsize=(8, 4.2)); bins = np.logspace(-3, 1, 60)
for ff in (1, 4, 20):
    rb = d.rain.coarsen(lat=ff, lon=ff, boundary="trim").sum()
    vb = d.views.coarsen(lat=ff, lon=ff, boundary="trim").sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (rb / vb).where(vb >= 10).values.ravel()
    rate = rate[np.isfinite(rate) & (rate > 0)]
    ax.hist(rate, bins=bins, histtype="step", density=True, lw=2, label=f"{0.05*ff:.2f}deg")
ax.set(xscale="log", xlabel="box-mean rain rate (mm/hr)", ylabel="density", title="rain-rate PDF narrows with averaging")
ax.legend(title="resolution"); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{OUT}/na2009_pdf.png", dpi=95, bbox_inches="tight"); plt.close(fig)
print("wrote na2009_pdf.png", flush=True)
print("DONE", flush=True)
