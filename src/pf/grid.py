"""Part B: gridded rain-contribution climatology by storm morphology.

Grids the pixel-table near-surface rain into 0.05 deg lat/lon bins, stratified by
feature **size** (``major_axis_km``), **20-dBZ echo-top** (``max_ht_20dbz``), and
per-pixel **rain type**, by **month-of-year**, per mission. Combined with the
un-stratified **views** denominator (:mod:`pf.views`) every cell yields:

    unconditional rate = rain_sum / views
    rain frequency     = raining_count / views
    conditional rate   = rain_sum / raining_count

The heavy join+aggregate is pushed into DuckDB (out-of-core, memory-bounded). The
ground-truth product is a **sparse joint** table
``(lat_bin, lon_bin, month, size_class, echotop_class, raintype, rain_sum,
raining_count)``; NetCDF renderings (marginal breakdowns or the full
cross-product) are built from it.

Classes (edges from :mod:`pf.config`): size -> ``[<20, 20-50, 50-100, >100]`` +
``undefined``; echo-top -> ``[<5, 5-7.5, 7.5-12, >12]`` + ``undefined``; raintype
-> ``[stratiform, convective, other]``. NaN size/echo-top go to ``undefined``
(never the top physical bin); non-1/2/3 rain types are dropped from the rain
accumulators (they still count as views).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pf import config

# --- class cardinalities --------------------------------------------------
SIZE_N = len(config.SIZE_EDGES_KM) + 1          # 4 physical
SIZE_UNDEF = SIZE_N                              # index 4
SIZE_SLOTS = SIZE_N + 1                          # incl. undefined
ECHOTOP_N = len(config.ECHOTOP_EDGES_KM) + 1     # 4 physical
ECHOTOP_UNDEF = ECHOTOP_N                        # index 4
ECHOTOP_SLOTS = ECHOTOP_N + 1
RAINTYPE_N = len(config.RAINTYPE_CLASSES)        # 3

# Reduce-time sanity ceiling: a single (cell, hour) summed over all years holds at
# most ~thousands of pixel-views/raining-pixels; anything >=1e8 is corrupt
# (e.g. a bad parquet row) and is dropped so it can't overflow the int64 SUM.
_SANE_COUNT = 100_000_000

SIZE_LABELS = ["<20", "20-50", "50-100", ">100", "undefined"]
ECHOTOP_LABELS = ["<5", "5-7.5", "7.5-12", ">12", "undefined"]
RAINTYPE_LABELS = ["stratiform", "convective", "other"]

# Shared lat band for cross-mission comparison/combination. Both missions are
# gridded on the SAME (lat, lon) coordinates (full-global lon, this lat band) so
# the per-mission zarr stores are cell-for-cell aligned; TRMM cells poleward of
# its ~+/-38 deg coverage are simply all-zero (views=0). GPM's +/-68 deg band is
# the wider envelope, so it is the shared default.
SHARED_LAT_CLIP = (-68.0, 68.0)


# --- pure class assignment (vectorised, NaN-safe) -------------------------
def size_class(major_axis_km) -> np.ndarray:
    """Size class index from feature major axis (km); NaN -> undefined."""
    x = np.asarray(major_axis_km, dtype=np.float64)
    cls = np.digitize(x, config.SIZE_EDGES_KM, right=False).astype(np.int16)
    cls[~np.isfinite(x)] = SIZE_UNDEF
    return cls


def echotop_class(max_ht_20dbz_m) -> np.ndarray:
    """Echo-top class index from max_ht_20dbz (metres -> km); NaN -> undefined."""
    km = np.asarray(max_ht_20dbz_m, dtype=np.float64) / 1000.0
    cls = np.digitize(km, config.ECHOTOP_EDGES_KM, right=False).astype(np.int16)
    cls[~np.isfinite(km)] = ECHOTOP_UNDEF
    return cls


def raintype_class(rain_type) -> np.ndarray:
    """Per-pixel: 1->0 strat, 2->1 conv, 3->2 other, else -1 (drop from rain)."""
    rt = np.asarray(rain_type)
    out = np.full(rt.shape, -1, dtype=np.int16)
    out[rt == 1] = 0
    out[rt == 2] = 1
    out[rt == 3] = 2
    return out


def latlon_to_bin(
    lat, lon, *,
    grid_deg=config.GRID_DEG, lat_min=config.GRID_LAT_MIN, lon_min=config.GRID_LON_MIN,
    n_lat=config.GRID_N_LAT, n_lon=config.GRID_N_LON,
):
    """Floor-bin lat/lon to grid indices (lon wrapped to [-180,180), clipped)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = ((np.asarray(lon, dtype=np.float64) - lon_min) % 360.0) + lon_min
    lat_bin = np.clip(np.floor((lat - lat_min) / grid_deg).astype(np.int64), 0, n_lat - 1)
    lon_bin = np.clip(np.floor((lon - lon_min) / grid_deg).astype(np.int64), 0, n_lon - 1)
    return lat_bin, lon_bin


# --- DuckDB SQL fragments (shared edges -> CASE expressions) --------------
def _size_case(col: str) -> str:
    e = config.SIZE_EDGES_KM
    return (f"CASE WHEN {col} IS NULL OR isnan({col}) THEN {SIZE_UNDEF} "
            f"WHEN {col} < {e[0]} THEN 0 WHEN {col} < {e[1]} THEN 1 "
            f"WHEN {col} < {e[2]} THEN 2 ELSE 3 END")


def _echotop_case(col: str) -> str:
    e = config.ECHOTOP_EDGES_KM
    km = f"({col}/1000.0)"
    return (f"CASE WHEN {col} IS NULL OR isnan({col}) THEN {ECHOTOP_UNDEF} "
            f"WHEN {km} < {e[0]} THEN 0 WHEN {km} < {e[1]} THEN 1 "
            f"WHEN {km} < {e[2]} THEN 2 ELSE 3 END")


def _raintype_case(col: str) -> str:
    return (f"CASE {col} WHEN 1 THEN 0 WHEN 2 THEN 1 WHEN 3 THEN 2 ELSE NULL END")


def _bin_sql(latcol: str, loncol: str) -> tuple[str, str]:
    g, la0, lo0 = config.GRID_DEG, config.GRID_LAT_MIN, config.GRID_LON_MIN
    lat_b = f"CAST(floor(({latcol} - ({la0}))/{g}) AS INTEGER)"
    lon_w = f"((({loncol} - ({lo0})) - floor(({loncol} - ({lo0}))/360.0)*360.0))"
    lon_b = f"CAST(floor({lon_w}/{g}) AS INTEGER)"
    # clip
    lat_b = f"greatest(0, least({config.GRID_N_LAT - 1}, {lat_b}))"
    lon_b = f"greatest(0, least({config.GRID_N_LON - 1}, {lon_b}))"
    return lat_b, lon_b


def _hive_glob(root: str, table: str, mission: str, month: int) -> str:
    """All year partitions for a given month-of-year (climatological)."""
    return (f"{root}/{table}/mission={mission.upper()}/year=*/"
            f"month={month:02d}/*.parquet")


def accumulate_month(con, mission: str, month: int, root: str = config.PF_ROOT) -> pd.DataFrame:
    """Sparse joint rain accumulator for one month-of-year (all years).

    Returns columns ``lat_bin, lon_bin, size_class, echotop_class, raintype,
    rain_sum, raining_count`` (rain>0 pixels only; non-1/2/3 raintype dropped).
    Empty DataFrame if the month has no data.
    """
    pix = _hive_glob(root, "pixels", mission, month)
    feat = _hive_glob(root, "features", mission, month)
    lat_b, lon_b = _bin_sql("p.lat", "p.lon")
    sql = f"""
        SELECT {lat_b} AS lat_bin, {lon_b} AS lon_bin,
               {_size_case('f.major_axis_km')} AS size_class,
               {_echotop_case('f.max_ht_20dbz')} AS echotop_class,
               {_raintype_case('p.rain_type')} AS raintype,
               SUM(p.near_sfc_rain) AS rain_sum,
               COUNT(*) AS raining_count
        FROM read_parquet('{pix}', union_by_name=true) p
        JOIN read_parquet('{feat}', union_by_name=true) f USING (feature_id)
        WHERE p.near_sfc_rain > 0 AND {_raintype_case('p.rain_type')} IS NOT NULL
        GROUP BY 1,2,3,4,5
    """
    try:
        return con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001 — empty/missing partition
        if "No files found" in str(exc) or "IO Error" in str(exc):
            return pd.DataFrame(columns=["lat_bin", "lon_bin", "size_class",
                                         "echotop_class", "raintype",
                                         "rain_sum", "raining_count"])
        raise


def reduce_views_month(con, mission: str, month: int, root: str = config.PF_ROOT) -> pd.DataFrame:
    """Sum the sparse per-orbit views for one month-of-year (all years).

    Returns ``lat_bin, lon_bin, n_views``.
    """
    glob = _hive_glob(root, config.VIEWS_ROOT_SUBDIR, mission, month)
    sql = f"""
        SELECT lat_bin, lon_bin, CAST(SUM(n_views) AS BIGINT) AS n_views
        FROM read_parquet('{glob}', union_by_name=true)
        GROUP BY 1,2
    """
    try:
        return con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        if "No files found" in str(exc) or "IO Error" in str(exc):
            return pd.DataFrame(columns=["lat_bin", "lon_bin", "n_views"])
        raise


def lat_clip_bins(mission: str, lat_clip: tuple[float, float] | None = None) -> tuple[int, int]:
    """Inclusive [lo, hi] lat_bin range for the grid.

    With ``lat_clip=(lo_deg, hi_deg)`` the band is explicit (mission-independent)
    so multiple missions share an identical grid — see :data:`SHARED_LAT_CLIP`.
    Otherwise it falls back to the per-mission coverage band in
    ``config.GRID_LAT_CLIP``.
    """
    if lat_clip is not None:
        lo_deg, hi_deg = lat_clip
    else:
        lo_deg, hi_deg = config.GRID_LAT_CLIP.get(mission.upper(), (-90.0, 90.0))
    lo = int(np.floor((lo_deg - config.GRID_LAT_MIN) / config.GRID_DEG))
    hi = int(np.floor((hi_deg - config.GRID_LAT_MIN) / config.GRID_DEG))
    return max(0, lo), min(config.GRID_N_LAT - 1, hi)


def _cell_centers(lo_bin: int, hi_bin: int, origin: float) -> np.ndarray:
    return origin + (np.arange(lo_bin, hi_bin + 1) + 0.5) * config.GRID_DEG


def build_dataset(
    rain_by_month: dict[int, pd.DataFrame],
    views_by_month: dict[int, pd.DataFrame],
    mission: str,
    *,
    mode: str = "marginal",
    lat_clip: tuple[float, float] | None = None,
) -> "xr.Dataset":
    """Densify the sparse accumulators into a CF NetCDF dataset (lat-clipped).

    ``mode='marginal'`` (default): three independent breakdowns
    (``rain_sum_by_size`` etc.) + ``views`` — tractable, the recommended view.
    ``mode='crossproduct'``: the full 6-D joint (large; densified per month).
    The complete joint is always available losslessly via :func:`write_sparse_joint`.
    ``lat_clip`` forces a shared lat band (see :func:`lat_clip_bins`).
    """
    import xarray as xr

    months = list(range(1, 13))
    lo, hi = lat_clip_bins(mission, lat_clip)
    nlat = hi - lo + 1
    nlon = config.GRID_N_LON
    lats = _cell_centers(lo, hi, config.GRID_LAT_MIN)
    lons = _cell_centers(0, nlon - 1, config.GRID_LON_MIN)

    views = np.zeros((12, nlat, nlon), np.int64)
    for mi, m in enumerate(months):
        vdf = views_by_month.get(m)
        if vdf is not None and len(vdf):
            inb = (vdf.lat_bin >= lo) & (vdf.lat_bin <= hi)
            v = vdf[inb]
            np.add.at(views[mi], (v.lat_bin.to_numpy() - lo, v.lon_bin.to_numpy()),
                      v.n_views.to_numpy())

    data_vars: dict = {}
    coords = {
        "month": ("month", months),
        "lat": ("lat", lats.astype(np.float32)),
        "lon": ("lon", lons.astype(np.float32)),
    }

    if mode == "marginal":
        specs = [("size", "size_class", SIZE_SLOTS, SIZE_LABELS),
                 ("echotop", "echotop_class", ECHOTOP_SLOTS, ECHOTOP_LABELS),
                 ("raintype", "raintype", RAINTYPE_N, RAINTYPE_LABELS)]
        for name, keycol, nslot, labels in specs:
            rs = np.zeros((12, nslot, nlat, nlon), np.float32)
            rc = np.zeros((12, nslot, nlat, nlon), np.int32)
            for mi, m in enumerate(months):
                df = rain_by_month.get(m)
                if df is None or not len(df):
                    continue
                g = (df.groupby(["lat_bin", "lon_bin", keycol], as_index=False)
                       [["rain_sum", "raining_count"]].sum())
                inb = (g.lat_bin >= lo) & (g.lat_bin <= hi)
                g = g[inb]
                ix = (g.lat_bin.to_numpy() - lo, g.lon_bin.to_numpy())
                k = g[keycol].to_numpy().astype(int)
                np.add.at(rs[mi], (k, *ix), g.rain_sum.to_numpy().astype(np.float32))
                np.add.at(rc[mi], (k, *ix), g.raining_count.to_numpy().astype(np.int32))
            cdim = f"{name}_class"
            coords[cdim] = (cdim, np.arange(nslot))
            coords[f"{name}_label"] = (cdim, labels)
            data_vars[f"rain_sum_by_{name}"] = (("month", cdim, "lat", "lon"), rs)
            data_vars[f"raining_count_by_{name}"] = (("month", cdim, "lat", "lon"), rc)
    elif mode == "crossproduct":
        rs = np.zeros((12, SIZE_SLOTS, ECHOTOP_SLOTS, RAINTYPE_N, nlat, nlon), np.float32)
        rc = np.zeros((12, SIZE_SLOTS, ECHOTOP_SLOTS, RAINTYPE_N, nlat, nlon), np.int32)
        for mi, m in enumerate(months):
            df = rain_by_month.get(m)
            if df is None or not len(df):
                continue
            inb = (df.lat_bin >= lo) & (df.lat_bin <= hi)
            d = df[inb]
            ix = (d.size_class.to_numpy().astype(int), d.echotop_class.to_numpy().astype(int),
                  d.raintype.to_numpy().astype(int), d.lat_bin.to_numpy() - lo, d.lon_bin.to_numpy())
            np.add.at(rs[mi], ix, d.rain_sum.to_numpy().astype(np.float32))
            np.add.at(rc[mi], ix, d.raining_count.to_numpy().astype(np.int32))
        for cdim, n, labels in [("size_class", SIZE_SLOTS, SIZE_LABELS),
                                ("echotop_class", ECHOTOP_SLOTS, ECHOTOP_LABELS),
                                ("raintype", RAINTYPE_N, RAINTYPE_LABELS)]:
            coords[cdim] = (cdim, np.arange(n))
            coords[f"{cdim}_label"] = (cdim, labels)
        dims = ("month", "size_class", "echotop_class", "raintype", "lat", "lon")
        data_vars["rain_sum"] = (dims, rs)
        data_vars["raining_count"] = (dims, rc)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    data_vars["views"] = (("month", "lat", "lon"), views.astype(np.int64))

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update({
        "Conventions": "CF-1.8",
        "title": f"PF rain-contribution climatology ({mission.upper()})",
        "mission": mission.upper(),
        "grid_deg": config.GRID_DEG,
        "month_axis": "month-of-year (1-12), all years summed",
        "rain_sum_units": "mm/hr (sum of instantaneous near-surface rates, NOT a depth)",
        "derived": "rate=rain_sum/views; freq=raining_count/views; intensity=rain_sum/raining_count",
        "size_edges_km": list(config.SIZE_EDGES_KM),
        "echotop_edges_km": list(config.ECHOTOP_EDGES_KM),
    })
    ds["lat"].attrs.update(units="degrees_north", long_name="latitude")
    ds["lon"].attrs.update(units="degrees_east", long_name="longitude")
    ds["views"].attrs.update(long_name="radar near-surface pixel views (sampling denominator)")
    return ds


def build_month_dataset(
    rain_df, views_df, *, month: int, mission: str, lo: int, hi: int,
    mode: str = "marginal",
) -> "xr.Dataset":
    """Densify a SINGLE month-of-year slab into an xarray Dataset (month dim len 1).

    Identical class/lat/lon layout to :func:`build_dataset`, but one month at a
    time so peak RAM is a single slab — used by :func:`write_zarr` to stream the
    store month-by-month (``append_dim='month'``). ``lo``/``hi`` are the inclusive
    ``lat_bin`` bounds of the shared grid (from :func:`lat_clip_bins`)."""
    import xarray as xr

    nlat = hi - lo + 1
    nlon = config.GRID_N_LON
    lats = _cell_centers(lo, hi, config.GRID_LAT_MIN)
    lons = _cell_centers(0, nlon - 1, config.GRID_LON_MIN)

    # views denominator (un-stratified)
    views = np.zeros((1, nlat, nlon), np.int64)
    if views_df is not None and len(views_df):
        inb = (views_df.lat_bin >= lo) & (views_df.lat_bin <= hi)
        v = views_df[inb]
        np.add.at(views[0], (v.lat_bin.to_numpy() - lo, v.lon_bin.to_numpy()),
                  v.n_views.to_numpy())

    coords = {
        "month": ("month", [month]),
        "lat": ("lat", lats.astype(np.float32)),
        "lon": ("lon", lons.astype(np.float32)),
    }
    data_vars: dict = {}

    if mode == "marginal":
        specs = [("size", "size_class", SIZE_SLOTS, SIZE_LABELS),
                 ("echotop", "echotop_class", ECHOTOP_SLOTS, ECHOTOP_LABELS),
                 ("raintype", "raintype", RAINTYPE_N, RAINTYPE_LABELS)]
        for name, keycol, nslot, labels in specs:
            rs = np.zeros((1, nslot, nlat, nlon), np.float32)
            rc = np.zeros((1, nslot, nlat, nlon), np.int32)
            if rain_df is not None and len(rain_df):
                g = (rain_df.groupby(["lat_bin", "lon_bin", keycol], as_index=False)
                            [["rain_sum", "raining_count"]].sum())
                inb = (g.lat_bin >= lo) & (g.lat_bin <= hi)
                g = g[inb]
                ix = (g.lat_bin.to_numpy() - lo, g.lon_bin.to_numpy())
                k = g[keycol].to_numpy().astype(int)
                np.add.at(rs[0], (k, *ix), g.rain_sum.to_numpy().astype(np.float32))
                np.add.at(rc[0], (k, *ix), g.raining_count.to_numpy().astype(np.int32))
            cdim = f"{name}_class"
            coords[cdim] = (cdim, np.arange(nslot))
            coords[f"{name}_label"] = (cdim, labels)
            data_vars[f"rain_sum_by_{name}"] = (("month", cdim, "lat", "lon"), rs)
            data_vars[f"raining_count_by_{name}"] = (("month", cdim, "lat", "lon"), rc)
    elif mode == "crossproduct":
        rs = np.zeros((1, SIZE_SLOTS, ECHOTOP_SLOTS, RAINTYPE_N, nlat, nlon), np.float32)
        rc = np.zeros((1, SIZE_SLOTS, ECHOTOP_SLOTS, RAINTYPE_N, nlat, nlon), np.int32)
        if rain_df is not None and len(rain_df):
            inb = (rain_df.lat_bin >= lo) & (rain_df.lat_bin <= hi)
            d = rain_df[inb]
            ix = (d.size_class.to_numpy().astype(int), d.echotop_class.to_numpy().astype(int),
                  d.raintype.to_numpy().astype(int), d.lat_bin.to_numpy() - lo, d.lon_bin.to_numpy())
            np.add.at(rs[0], ix, d.rain_sum.to_numpy().astype(np.float32))
            np.add.at(rc[0], ix, d.raining_count.to_numpy().astype(np.int32))
        for cdim, n, labels in [("size_class", SIZE_SLOTS, SIZE_LABELS),
                                ("echotop_class", ECHOTOP_SLOTS, ECHOTOP_LABELS),
                                ("raintype", RAINTYPE_N, RAINTYPE_LABELS)]:
            coords[cdim] = (cdim, np.arange(n))
            coords[f"{cdim}_label"] = (cdim, labels)
        dims = ("month", "size_class", "echotop_class", "raintype", "lat", "lon")
        data_vars["rain_sum"] = (dims, rs)
        data_vars["raining_count"] = (dims, rc)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    data_vars["views"] = (("month", "lat", "lon"), views)
    ds = xr.Dataset(data_vars, coords=coords)
    ds["lat"].attrs.update(units="degrees_north", long_name="latitude")
    ds["lon"].attrs.update(units="degrees_east", long_name="longitude")
    ds["views"].attrs.update(long_name="radar near-surface pixel views (sampling denominator)")
    return ds


def _zarr_encoding(ds, complevel: int = 5, lat_chunk: int = 340, lon_chunk: int = 720) -> dict:
    """Per-variable zarr v3 encoding: Zstd compressor + cloud-friendly chunks
    (month=1 so each month is a clean append/region boundary)."""
    from zarr.codecs import ZstdCodec

    enc = {}
    for name, da in ds.data_vars.items():
        chunks = []
        for d, n in zip(da.dims, da.shape):
            if d in ("month", "hour"):
                chunks.append(1)
            elif d == "lat":
                chunks.append(min(lat_chunk, n))
            elif d == "lon":
                chunks.append(min(lon_chunk, n))
            else:                       # class dims: keep whole (tiny)
                chunks.append(n)
        enc[name] = {"compressors": [ZstdCodec(level=complevel)], "chunks": tuple(chunks)}
    return enc


def write_zarr(
    rain_by_month: dict[int, pd.DataFrame],
    views_by_month: dict[int, pd.DataFrame],
    mission: str,
    store: str,
    *,
    mode: str = "marginal",
    lat_clip: tuple[float, float] | None = SHARED_LAT_CLIP,
    storage_options: dict | None = None,
    complevel: int = 5,
    log=lambda *a, **k: None,
) -> str:
    """Stream the climatology to a zarr store, one month-of-year slab at a time.

    Writes month 1 with ``mode='w'`` then appends months 2..12 along ``month``,
    so peak RAM is a single densified slab (not all 12). ``store`` may be a local
    path or an ``s3://bucket/key.zarr`` URL (``storage_options`` -> fsspec/s3fs,
    e.g. MinIO endpoint+creds). All missions written with the same ``lat_clip``
    share an identical (lat, lon) grid for trivial cross-mission combination.
    Returns ``store``."""
    lo, hi = lat_clip_bins(mission, lat_clip)
    so = storage_options or None
    months = sorted(rain_by_month.keys() | views_by_month.keys()) or list(range(1, 13))

    attrs = {
        "Conventions": "CF-1.8",
        "title": f"PF rain-contribution climatology ({mission.upper()})",
        "mission": mission.upper(),
        "mode": mode,
        "grid_deg": config.GRID_DEG,
        "lat_clip_deg": list(lat_clip) if lat_clip else "per-mission",
        "shared_grid": "all missions on identical (lat,lon); TRMM zero poleward of its coverage",
        "month_axis": "month-of-year (1-12), all years summed",
        "rain_sum_units": "mm/hr (sum of instantaneous near-surface rates, NOT a depth)",
        "derived": "rate=rain_sum/views; freq=raining_count/views; intensity=rain_sum/raining_count",
        "size_edges_km": list(config.SIZE_EDGES_KM),
        "echotop_edges_km": list(config.ECHOTOP_EDGES_KM),
    }

    for i, m in enumerate(months):
        ds = build_month_dataset(
            rain_by_month.get(m), views_by_month.get(m),
            month=m, mission=mission, lo=lo, hi=hi, mode=mode)
        if i == 0:
            ds.attrs.update(attrs)
            ds.to_zarr(store, mode="w", storage_options=so, consolidated=True,
                       encoding=_zarr_encoding(ds, complevel))
        else:
            ds.to_zarr(store, mode="a", append_dim="month", storage_options=so,
                       consolidated=True)
        log(f"    {mission} month {m:02d} -> {store} ({'init' if i == 0 else 'append'})")
    return store


# ======================================================================
# Swath-gridded, HOUR-RESOLVED climatology (Stage 2): reduce the per-(year,
# month) sparse grid tables (written by scripts/grid_month.py) across years
# into a month-of-year x hour climatology, densify per (month, hour) slab, and
# stream a marginal zarr store. Distinct from the legacy DuckDB-over-pixels path
# above (which has no hour axis); both live here so callers can pick.
# ======================================================================

def _grid_hive_glob(root: str, mission: str, month: int, table: str) -> str:
    """All year partitions of a swath-grid table for one month-of-year."""
    return (f"{root}/grid/mission={mission.upper()}/year=*/"
            f"month={month:02d}/{table}.parquet")


def duck_connect(mem: str = "32GB", threads: int = 16):
    """A DuckDB connection configured for an OUT-OF-CORE grid reduce.

    The grid tables are small on disk (compressed small ints) but ENORMOUS in
    row count — a month-of-year of views, summed over all years, is hundreds of
    millions of cell-hour rows, far too many to materialize in pandas. DuckDB
    streams + spills to disk, so it handles this in bounded memory — **provided
    its spill directory works**. (The earlier ``OUT_OF_RANGE``/garbage-sum
    corruption was a FAILED spill to a missing temp dir; pointing ``temp_directory``
    at node-local ``$TMPDIR`` fixes it.)"""
    import os

    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'")
    con.execute(f"SET threads={threads}")
    td = os.path.join(os.environ.get("TMPDIR") or "/tmp", f"pf_duck_{os.getpid()}")
    os.makedirs(td, exist_ok=True)
    con.execute(f"SET temp_directory='{td}'")
    con.execute("SET preserve_insertion_order=false")  # lower memory for big GROUP BY
    return con


def reduce_grid_rain_month(con, mission: str, month: int,
                           root: str = config.PF_ROOT) -> pd.DataFrame:
    """Sum the per-(year,month) sparse rain tables for one month-of-year (DuckDB,
    out-of-core). Pass a :func:`duck_connect` ``con`` (or None to make one)."""
    own = con is None
    if own:
        con = duck_connect()
    glob = _grid_hive_glob(root, mission, month, "rain")
    sql = f"""
        SELECT lat_bin, lon_bin, hour, size_class, echotop_class, raintype,
               CAST(SUM(rain_sum) AS DOUBLE) AS rain_sum,
               CAST(SUM(raining_count) AS BIGINT) AS raining_count
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE raining_count > 0 AND raining_count < {_SANE_COUNT}
          AND isfinite(rain_sum) AND rain_sum >= 0 AND rain_sum < 1e15
          AND lat_bin BETWEEN 0 AND {config.GRID_N_LAT - 1}
          AND lon_bin BETWEEN 0 AND {config.GRID_N_LON - 1}
          AND hour BETWEEN 0 AND 23
          AND size_class BETWEEN 0 AND {SIZE_SLOTS - 1}
          AND echotop_class BETWEEN 0 AND {ECHOTOP_SLOTS - 1}
          AND raintype BETWEEN 0 AND {RAINTYPE_N - 1}
        GROUP BY 1,2,3,4,5,6
    """
    try:
        return con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        if "No files found" in str(exc) or "IO Error" in str(exc):
            return pd.DataFrame(columns=["lat_bin", "lon_bin", "hour", "size_class",
                                         "echotop_class", "raintype", "rain_sum",
                                         "raining_count"])
        raise
    finally:
        if own:
            con.close()


def reduce_grid_views_month(con, mission: str, month: int,
                            root: str = config.PF_ROOT) -> pd.DataFrame:
    """Sum the per-(year,month) sparse views tables for one month-of-year (DuckDB,
    out-of-core). Pass a :func:`duck_connect` ``con`` (or None to make one)."""
    own = con is None
    if own:
        con = duck_connect()
    glob = _grid_hive_glob(root, mission, month, "views")
    sql = f"""
        SELECT lat_bin, lon_bin, hour, CAST(SUM(n_views) AS BIGINT) AS n_views
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE n_views > 0 AND n_views < {_SANE_COUNT}
          AND lat_bin BETWEEN 0 AND {config.GRID_N_LAT - 1}
          AND lon_bin BETWEEN 0 AND {config.GRID_N_LON - 1}
          AND hour BETWEEN 0 AND 23
        GROUP BY 1,2,3
    """
    try:
        return con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        if "No files found" in str(exc) or "IO Error" in str(exc):
            return pd.DataFrame(columns=["lat_bin", "lon_bin", "hour", "n_views"])
        raise
    finally:
        if own:
            con.close()


def reduce_grid_metrics_month(con, mission: str, month: int,
                              root: str = config.PF_ROOT) -> pd.DataFrame:
    """Sum the per-(year,month) sparse ``metrics`` tables for one month-of-year
    (DuckDB, out-of-core), keyed ``(lat_bin, lon_bin, hour)``.

    Each of the :data:`pf.grid_swath.METRIC_COLS` is a Σ (``*_sum``) or a count
    (``*_n`` / ``cnt_gt*``) that pools by simple addition across years — so a mean
    downstream is ``Σ_sum / Σ_n`` (COMBINED missions via summed totals too). Returns
    an empty frame (right columns) if the month has no metrics partitions yet."""
    from .grid_swath import METRIC_COLS
    own = con is None
    if own:
        con = duck_connect()
    glob = _grid_hive_glob(root, mission, month, "metrics")
    aggs = ",\n               ".join(
        (f"CAST(SUM({c}) AS DOUBLE) AS {c}" if c.endswith("_sum")
         else f"CAST(SUM({c}) AS BIGINT) AS {c}")
        for c in METRIC_COLS)
    sql = f"""
        SELECT lat_bin, lon_bin, hour,
               {aggs}
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE lat_bin BETWEEN 0 AND {config.GRID_N_LAT - 1}
          AND lon_bin BETWEEN 0 AND {config.GRID_N_LON - 1}
          AND hour BETWEEN 0 AND 23
        GROUP BY 1,2,3
    """
    try:
        return con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        if "No files found" in str(exc) or "IO Error" in str(exc):
            return pd.DataFrame(columns=["lat_bin", "lon_bin", "hour", *METRIC_COLS])
        raise
    finally:
        if own:
            con.close()


def build_month_hour_dataset(
    rain_df, views_df, *, month: int, hour: int, mission: str, lo: int, hi: int,
    mode: str = "marginal", metrics_df=None,
) -> "xr.Dataset":
    """Densify ONE (month-of-year, hour) marginal slab (both dims length 1).

    ``rain_df``/``views_df`` are the month's sparse accumulators (all hours);
    this filters to ``hour`` and scatters into the shared lat-clipped grid.
    Vars: ``views(month,hour,lat,lon)`` and ``{rain_sum,raining_count}_by_
    {size,echotop,raintype}(month,hour,class,lat,lon)``."""
    import xarray as xr

    if mode != "marginal":
        raise ValueError("hour-resolved zarr supports mode='marginal' only "
                         "(the crossproduct is kept as sparse Parquet)")
    nlat = hi - lo + 1
    nlon = config.GRID_N_LON
    lats = _cell_centers(lo, hi, config.GRID_LAT_MIN)
    lons = _cell_centers(0, nlon - 1, config.GRID_LON_MIN)

    rh = rain_df[rain_df.hour == hour] if rain_df is not None and len(rain_df) else None
    vh = views_df[views_df.hour == hour] if views_df is not None and len(views_df) else None

    views = np.zeros((1, 1, nlat, nlon), np.int32)
    if vh is not None and len(vh):
        inb = (vh.lat_bin >= lo) & (vh.lat_bin <= hi)
        v = vh[inb]
        np.add.at(views[0, 0], (v.lat_bin.to_numpy() - lo, v.lon_bin.to_numpy()),
                  v.n_views.to_numpy())

    coords = {
        "month": ("month", [month]),
        "hour": ("hour", [hour]),
        "lat": ("lat", lats.astype(np.float32)),
        "lon": ("lon", lons.astype(np.float32)),
    }
    data_vars: dict = {}
    specs = [("size", "size_class", SIZE_SLOTS, SIZE_LABELS),
             ("echotop", "echotop_class", ECHOTOP_SLOTS, ECHOTOP_LABELS),
             ("raintype", "raintype", RAINTYPE_N, RAINTYPE_LABELS)]
    for name, keycol, nslot, labels in specs:
        rs = np.zeros((1, 1, nslot, nlat, nlon), np.float32)
        rc = np.zeros((1, 1, nslot, nlat, nlon), np.int32)
        if rh is not None and len(rh):
            g = (rh.groupby(["lat_bin", "lon_bin", keycol], as_index=False)
                   [["rain_sum", "raining_count"]].sum())
            inb = (g.lat_bin >= lo) & (g.lat_bin <= hi)
            g = g[inb]
            ix = (g.lat_bin.to_numpy() - lo, g.lon_bin.to_numpy())
            k = g[keycol].to_numpy().astype(int)
            np.add.at(rs[0, 0], (k, *ix), g.rain_sum.to_numpy().astype(np.float32))
            np.add.at(rc[0, 0], (k, *ix), g.raining_count.to_numpy().astype(np.int32))
        cdim = f"{name}_class"
        coords[cdim] = (cdim, np.arange(nslot))
        coords[f"{name}_label"] = (cdim, labels)
        data_vars[f"rain_sum_by_{name}"] = (("month", "hour", cdim, "lat", "lon"), rs)
        data_vars[f"raining_count_by_{name}"] = (("month", "hour", cdim, "lat", "lon"), rc)

    data_vars["views"] = (("month", "hour", "lat", "lon"), views)

    # --- new per-pixel metrics (Σ/ count grids, keyed lat,lon,hour) ----------
    # Each METRIC_COL is its own (month,hour,lat,lon) var: f4 for sums, i4 for
    # counts. A month with no metrics rows emits none -> stays skeleton zeros.
    if metrics_df is not None and len(metrics_df):
        from .grid_swath import METRIC_COLS
        mh = metrics_df[metrics_df.hour == hour]
        inb = (mh.lat_bin >= lo) & (mh.lat_bin <= hi)
        mh = mh[inb]
        li = mh.lat_bin.to_numpy() - lo
        lj = mh.lon_bin.to_numpy()
        for c in METRIC_COLS:
            is_sum = c.endswith("_sum")
            arr = np.zeros((1, 1, nlat, nlon), np.float32 if is_sum else np.int32)
            if len(mh):
                np.add.at(arr[0, 0], (li, lj), mh[c].to_numpy().astype(arr.dtype))
            data_vars[c] = (("month", "hour", "lat", "lon"), arr)

    ds = xr.Dataset(data_vars, coords=coords)
    ds["lat"].attrs.update(units="degrees_north", long_name="latitude")
    ds["lon"].attrs.update(units="degrees_east", long_name="longitude")
    ds["hour"].attrs.update(long_name="UTC hour of day", units="hour")
    ds["views"].attrs.update(long_name="radar near-surface pixel views (sampling denominator)")
    return ds


def icechunk_repo(bucket: str, prefix: str, storage_options: dict):
    """Open-or-create an Icechunk repository on an S3/MinIO bucket+prefix.

    ``storage_options`` follows the s3fs convention used elsewhere:
    ``{"key", "secret", "client_kwargs": {"endpoint_url"}}`` (+ optional
    ``region``). For a MinIO ``http://`` endpoint, path-style addressing and
    plaintext HTTP are enabled automatically."""
    import icechunk

    so = storage_options or {}
    ep = (so.get("client_kwargs") or {}).get("endpoint_url", "") or ""
    storage = icechunk.s3_storage(
        bucket=bucket, prefix=prefix,
        endpoint_url=ep or None,
        allow_http=ep.startswith("http://"),
        force_path_style=bool(ep),
        region=so.get("region", "us-east-1"),
        access_key_id=so.get("key"),
        secret_access_key=so.get("secret"),
    )
    return icechunk.Repository.open_or_create(storage)


def write_grid_zarr(
    reduce_month,
    mission: str,
    *,
    bucket: str,
    prefix: str,
    storage_options: dict,
    mode: str = "marginal",
    lat_clip: tuple[float, float] | None = SHARED_LAT_CLIP,
    complevel: int = 5,
    joint_out: str | None = None,
    metric_cols: list[str] | None = None,
    log=lambda *a, **k: None,
) -> str:
    """Stream the hour-resolved marginal climatology into an **Icechunk** repo.

    ``reduce_month`` is a callable ``m -> (rain_df, views_df)`` (e.g. wrapping
    :func:`reduce_grid_rain_month`/:func:`reduce_grid_views_month`). It is called
    ONE month at a time inside the write loop — the month is densified into its 24
    hour slabs, committed to Icechunk, then freed before the next — so peak RAM is
    a single month's reduced frames, never all 12 (views alone is ~10⁸ rows/month).

    Writes a transactional Icechunk store at ``s3://{bucket}/{prefix}`` (MinIO via
    ``storage_options``): a full ``(month=12, hour=24, [class], lat, lon)`` skeleton
    (metadata-only) is committed first, then each (month, hour) slab is written into
    its ``region=`` and **committed once per month** (resumable). If ``joint_out``
    is given, the lossless sparse rain joint is streamed there incrementally.
    Returns the ``s3://`` location string."""
    import dask.array as dskar
    import numpy as _np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xarray as xr

    lo, hi = lat_clip_bins(mission, lat_clip)
    nlat = hi - lo + 1
    nlon = config.GRID_N_LON
    months = list(range(1, 13))
    hours = list(range(24))
    lats = _cell_centers(lo, hi, config.GRID_LAT_MIN).astype(_np.float32)
    lons = _cell_centers(0, nlon - 1, config.GRID_LON_MIN).astype(_np.float32)

    # --- skeleton: full shape, lazy zeros, metadata only -------------------
    def _z(shape, dtype, chunks):
        return dskar.zeros(shape, dtype=dtype, chunks=chunks)

    cls_specs = [("size", SIZE_SLOTS, SIZE_LABELS),
                 ("echotop", ECHOTOP_SLOTS, ECHOTOP_LABELS),
                 ("raintype", RAINTYPE_N, RAINTYPE_LABELS)]
    lat_c, lon_c = min(340, nlat), min(720, nlon)
    coords = {
        "month": ("month", months), "hour": ("hour", hours),
        "lat": ("lat", lats), "lon": ("lon", lons),
    }
    data_vars = {"views": (("month", "hour", "lat", "lon"),
                           _z((12, 24, nlat, nlon), "i4", (1, 1, lat_c, lon_c)))}
    for name, nslot, labels in cls_specs:
        cdim = f"{name}_class"
        coords[cdim] = (cdim, _np.arange(nslot))
        coords[f"{name}_label"] = (cdim, labels)
        data_vars[f"rain_sum_by_{name}"] = (
            ("month", "hour", cdim, "lat", "lon"),
            _z((12, 24, nslot, nlat, nlon), "f4", (1, 1, nslot, lat_c, lon_c)))
        data_vars[f"raining_count_by_{name}"] = (
            ("month", "hour", cdim, "lat", "lon"),
            _z((12, 24, nslot, nlat, nlon), "i4", (1, 1, nslot, lat_c, lon_c)))
    for c in (metric_cols or []):
        dt = "f4" if c.endswith("_sum") else "i4"
        data_vars[c] = (("month", "hour", "lat", "lon"),
                        _z((12, 24, nlat, nlon), dt, (1, 1, lat_c, lon_c)))
    skeleton = xr.Dataset(data_vars, coords=coords)
    skeleton.attrs.update({
        "Conventions": "CF-1.8",
        "title": f"PF swath-gridded rain climatology ({mission.upper()})",
        "mission": mission.upper(),
        "mode": mode,
        "grid_deg": config.GRID_DEG,
        "lat_clip_deg": list(lat_clip) if lat_clip else "per-mission",
        "shared_grid": "all missions on identical (lat,lon); TRMM zero poleward of its coverage",
        "month_axis": "month-of-year (1-12), all years summed",
        "hour_axis": "UTC hour of day (0-23)",
        "rain_sum_units": "mm/hr (sum of instantaneous near-surface rates, NOT a depth)",
        "derived": "rate=rain_sum/views; freq=raining_count/views; intensity=rain_sum/raining_count",
        "size_edges_km": list(config.SIZE_EDGES_KM),
        "echotop_edges_km": list(config.ECHOTOP_EDGES_KM),
    })
    if metric_cols:
        skeleton.attrs["metrics"] = (
            "per-pixel metric grids keyed (month,hour,lat,lon): et{20,30,40} "
            "convective echo-top height (m, mean=Σ_sum/Σ_n); cnt_gt{25,50,75,100} "
            "near-surface rain>=thr pixel counts; {eps,nw,dm}_{conv,strat} near-"
            "surface DSD params (mean=Σ_sum/Σ_n). counts pool with views as the "
            "denominator for occurrence frequency.")
    # --- Icechunk repo + skeleton (metadata only), committed --------------
    repo = icechunk_repo(bucket, prefix, storage_options)
    session = repo.writable_session("main")
    skeleton.to_zarr(session.store, mode="w", compute=False, consolidated=False,
                     zarr_format=3, encoding=_zarr_encoding(skeleton, complevel))
    session.commit(f"skeleton: {mission.upper()} swath-grid climatology")
    store = f"s3://{bucket}/{prefix}"
    log(f"    {mission}: icechunk skeleton committed -> {store} (12x24 slabs, lat {nlat}x lon {nlon})")

    # --- per month: reduce -> densify 24 hours -> commit -> free ----------
    # Drop coords with no region dim (class indices + labels) — they already
    # live in the skeleton; region writes carry only the sliced data vars.
    drop = ["size_label", "echotop_label", "raintype_label",
            "size_class", "echotop_class", "raintype_class"]
    jw = None
    jtmp = Path(joint_out).with_suffix(Path(joint_out).suffix + f".{os.getpid()}.tmp") \
        if joint_out else None
    for m in months:
        res = reduce_month(m)
        rdf, vdf, mdf = res if len(res) == 3 else (res[0], res[1], None)
        log(f"    {mission} month {m:02d}: reduced "
            f"({0 if rdf is None else len(rdf)} rain rows, "
            f"{0 if vdf is None else len(vdf)} view cells, "
            f"{0 if mdf is None else len(mdf)} metric cells)")
        if jtmp is not None and rdf is not None and len(rdf):
            d = rdf.copy()
            d.insert(0, "month", np.int16(m))
            d.insert(0, "mission", mission.upper())
            t = pa.Table.from_pandas(d, preserve_index=False)
            if jw is None:
                jw = pq.ParquetWriter(jtmp, t.schema, compression="zstd")
            jw.write_table(t)
            del d, t
        session = repo.writable_session("main")
        for h in hours:
            slab = build_month_hour_dataset(rdf, vdf, month=m, hour=h,
                                            mission=mission, lo=lo, hi=hi, mode=mode,
                                            metrics_df=mdf)
            slab = slab.drop_vars([d for d in drop if d in slab.coords])
            slab.to_zarr(session.store, region={"month": slice(m - 1, m),
                                                "hour": slice(h, h + 1),
                                                "lat": slice(0, nlat), "lon": slice(0, nlon)},
                         consolidated=False)
        session.commit(f"{mission.upper()} month {m:02d} (24 hour slabs)")
        log(f"    {mission} month {m:02d}: 24 hour-slabs committed")
        del rdf, vdf, mdf
    if jw is not None:
        jw.close()
        os.replace(jtmp, joint_out)
        log(f"    {mission}: sparse joint -> {joint_out}")
    return store


def write_grid_sparse_joint(rain_by_month: dict[int, pd.DataFrame], mission: str,
                            out: str) -> Path:
    """Lossless full joint (incl. hour + crossproduct) to one Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["mission", "month", "lat_bin", "lon_bin", "hour", "size_class",
            "echotop_class", "raintype", "rain_sum", "raining_count"]
    frames = []
    for m, df in rain_by_month.items():
        if df is not None and len(df):
            d = df.copy()
            d.insert(0, "month", np.int16(m))
            d.insert(0, "mission", mission.upper())
            frames.append(d)
    full = (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=cols))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pandas(full, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, out_path)
    return out_path


def write_sparse_joint(rain_by_month: dict[int, pd.DataFrame], mission: str, out: str) -> Path:
    """Write the complete sparse joint accumulator to one Parquet (lossless)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    frames = []
    for m, df in rain_by_month.items():
        if df is not None and len(df):
            d = df.copy()
            d.insert(0, "month", np.int16(m))
            d.insert(0, "mission", mission.upper())
            frames.append(d)
    full = (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["mission", "month", "lat_bin", "lon_bin",
                                       "size_class", "echotop_class", "raintype",
                                       "rain_sum", "raining_count"]))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pandas(full, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, out_path)
    return out_path


def write_netcdf(ds, out: str) -> Path:
    """Atomically write the dataset to NetCDF (CF, zlib level 5)."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 5} for v in ds.data_vars}
    tmp = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    # h5netcdf (pure-python on the already-present h5py) is the NetCDF4 backend.
    ds.to_netcdf(tmp, format="NETCDF4", engine="h5netcdf", encoding=enc)
    os.replace(tmp, out_path)
    return out_path


__all__ = [
    "SIZE_SLOTS", "ECHOTOP_SLOTS", "RAINTYPE_N", "SHARED_LAT_CLIP",
    "SIZE_LABELS", "ECHOTOP_LABELS", "RAINTYPE_LABELS",
    "size_class", "echotop_class", "raintype_class", "latlon_to_bin",
    "accumulate_month", "reduce_views_month", "lat_clip_bins",
    "build_dataset", "build_month_dataset", "write_zarr",
    "write_sparse_joint", "write_netcdf",
    "reduce_grid_rain_month", "reduce_grid_views_month",
    "build_month_hour_dataset", "write_grid_zarr", "write_grid_sparse_joint",
    "icechunk_repo",
]
