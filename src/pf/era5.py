"""ERA-5 environmental co-location for PF features (Phase 5).

A SEPARATE post-processing step that co-locates ARCO ERA-5 environmental
variables at each PF feature **centroid** plus surrounding box statistics,
mirroring ``feng_tracking/era5_claude.py`` (track -> feature). It writes a
SEPARATE hive-partitioned Parquet table keyed by ``feature_id``; the frozen
FEATURE_SCHEMA / PIXEL_SCHEMA and the per-orbit radar/imager pipeline are
untouched.

Data source: the public ARCO ERA-5 Zarr on GCS (anonymous access; requires
network). For each feature the centroid value is the grid-nearest ERA-5 cell;
box stats are computed over 5deg / 2.5deg / 1.25deg boxes (radii 2.5/1.25/0.625
degrees). Vertical wind shear at 1000/3000/6000 m AGL is computed relative to
the 10 m wind via geopotential-height interpolation (xarray ``interp``; no
xgcm dependency).

Output table (252 value columns + 4 meta) at::

    {root}/era5/mission={M}/year={YYYY}/month={MM}/orbit={NNNNNN}.parquet

Join to features on ``feature_id``.
"""

from __future__ import annotations

import collections
import gc
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pf.catalog import _coerce_time_us
from pf.config import PF_ROOT

# --- Constants (mirror feng_tracking) ------------------------------------
ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ERA5_VARS_2D = {  # ERA5 name -> short name
    "convective_available_potential_energy": "cape",
    "convective_inhibition": "cin",
    "sea_surface_temperature": "sst",
    "skin_temperature": "skt",
    "total_precipitation": "tpr",
}

WIND10 = ("10m_u_component_of_wind", "10m_v_component_of_wind")  # u10, v10

# 3-D vars needed for shear (sliced to LEVEL_SLICE hPa)
ERA5_VARS_3D = ("geopotential", "u_component_of_wind", "v_component_of_wind")

SHEAR_HEIGHTS_M = [1000, 3000, 6000]   # AGL -> shear_1000m / shear_3000m / shear_6000m
LEVEL_SLICE = (400, 1000)              # hPa, for geopotential / u / v
G = 9.81                               # m/s^2, geopotential -> geometric height

BOX_RADII = [2.5, 1.25, 0.625]         # degrees
BOX_LABELS = {2.5: "5deg", 1.25: "2p50deg", 0.625: "1p25deg"}
BOX_RADIUS_MAX = max(BOX_RADII)

PERCENTILES = [10, 25, 50, 75, 90, 95]
STATS = ["min", "max", "mean", "std", "p10", "p25", "p50", "p75", "p90", "p95"]

# the 8 output variables that get a centroid value + box stats:
STAT_VARS = ["cape", "cin", "sst", "skt", "tpr",
             "shear_1000m", "shear_3000m", "shear_6000m"]

ERA5_ROOT_SUBDIR = "era5"

_GRID = 0.25  # ERA-5 horizontal resolution (deg)

logger = logging.getLogger(__name__)

# Module-level counter of how many UNIQUE-hour global ERA-5 fields have been
# fetched (one .compute() per call). Lets callers/tests prove "one download per
# hour": after era5_for_features it equals the number of unique hours, NOT the
# number of features. Reset at the start of each era5_for_features call.
FETCH_COUNT = 0


# --- Zarr access ----------------------------------------------------------
def open_era5(zarr: str = ERA5_ZARR):
    """Open the ARCO ERA-5 Zarr store (anonymous, lazy).

    Parameters
    ----------
    zarr : str
        Zarr store URL (default :data:`ERA5_ZARR`).

    Returns
    -------
    xarray.Dataset
        Lazily-opened dataset; latitude descending, longitude 0-360, hourly,
        pressure ``level`` 1..1000 hPa.
    """
    import xarray as xr

    return xr.open_zarr(zarr, chunks=None, storage_options={"token": "anon"})


# --- Longitude-cyclic subset (mirror feng_tracking.get_cyclic_subset) -----
def get_cyclic_subset(data, lat_sel, lon_min_req, lon_max_req):
    """Select a lat slice + lon range, wrapping across the 0/360 seam.

    ``data`` may be an ``xarray.Dataset`` or ``DataArray`` with ERA-5
    coordinates (latitude descending, longitude 0-360). ``lat_sel`` is a
    ``slice(high, low)``. Longitude requests outside ``[0, 360)`` are wrapped
    and concatenated so a box straddling Greenwich still returns the full ring.
    """
    import xarray as xr

    try:
        ds_lon_min = float(data.longitude.values.min())
        ds_lon_max = float(data.longitude.values.max())

        if lon_min_req < ds_lon_min:
            # Overlap left seam -> grab wrapped right side + left side
            wrapped_req = lon_min_req + 360
            left_part = data.sel(
                latitude=lat_sel, longitude=slice(wrapped_req, ds_lon_max)
            )
            right_part = data.sel(
                latitude=lat_sel, longitude=slice(ds_lon_min, lon_max_req)
            )
            return xr.concat([left_part, right_part], dim="longitude")

        if lon_max_req > ds_lon_max:
            # Overlap right seam -> grab right side + wrapped left side
            wrapped_req = lon_max_req - 360
            left_part = data.sel(
                latitude=lat_sel, longitude=slice(lon_min_req, ds_lon_max)
            )
            right_part = data.sel(
                latitude=lat_sel, longitude=slice(ds_lon_min, wrapped_req)
            )
            return xr.concat([left_part, right_part], dim="longitude")

        return data.sel(latitude=lat_sel, longitude=slice(lon_min_req, lon_max_req))
    except Exception:  # noqa: BLE001
        return data.sel(latitude=lat_sel, longitude=slice(lon_min_req, lon_max_req))


# --- Shear (xarray interp path; no xgcm) ---------------------------------
def compute_shear_fields(subset):
    """Compute 10 m -> H m bulk wind shear fields for each H in SHEAR_HEIGHTS_M.

    Parameters
    ----------
    subset : xarray.Dataset
        Must contain ``geopotential``, ``u_component_of_wind``,
        ``v_component_of_wind`` on pressure ``level`` (sliced to
        :data:`LEVEL_SLICE`) plus the 10 m wind components in :data:`WIND10`.
        Should be already loaded into memory.

    Returns
    -------
    dict[int, xarray.DataArray]
        ``{H: shear_magnitude(lat, lon)}``. A value is ``None`` if the
        interpolation failed for that height.

    Notes
    -----
    Geometric height is ``geopotential / 9.81``. Because that height varies per
    grid column (it is a 3-D field, not a 1-D coordinate), ``u``/``v`` are
    interpolated to each constant geometric height ``H`` *per column* via
    :func:`numpy.interp` along the ``level`` axis (``xarray.apply_ufunc``,
    method='linear'). The shear magnitude is then taken relative to the 10 m
    wind. Height is monotonically decreasing with increasing pressure level, so
    each column's height profile is sorted ascending before interpolation.
    """
    import xarray as xr

    ght = subset["geopotential"] / G  # geometric height (m), dims (level, lat, lon)
    u = subset["u_component_of_wind"]
    v = subset["v_component_of_wind"]
    u10 = subset[WIND10[0]]
    v10 = subset[WIND10[1]]

    def _interp_col(wind_col, height_col, target):
        # wind_col, height_col: 1-D along level. np.interp needs ascending x.
        order = np.argsort(height_col)
        h = height_col[order]
        w = wind_col[order]
        good = np.isfinite(h) & np.isfinite(w)
        if good.sum() < 2:
            return np.nan
        return np.interp(target, h[good], w[good], left=np.nan, right=np.nan)

    def _interp_to_height(wind, target):
        return xr.apply_ufunc(
            _interp_col,
            wind,
            ght,
            kwargs={"target": target},
            input_core_dims=[["level"], ["level"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    fields: dict[int, object] = {}
    for height_m in SHEAR_HEIGHTS_M:
        try:
            u_h = _interp_to_height(u, float(height_m))
            v_h = _interp_to_height(v, float(height_m))
            shear = np.sqrt((u_h - u10) ** 2 + (v_h - v10) ** 2)
            fields[height_m] = shear
        except Exception:  # noqa: BLE001
            fields[height_m] = None
    return fields


# --- Box statistics (NaN-safe; mirror feng_tracking.compute_stats) -------
def compute_stats(values, var: str, box_label: str) -> dict:
    """NaN-safe min/max/mean/std/p10..p95 over ``values``.

    Parameters
    ----------
    values : numpy.ndarray or xarray.DataArray
        Values inside one box (any shape; flattened).
    var : str
        Short variable name (one of :data:`STAT_VARS`).
    box_label : str
        Box label (one of :data:`BOX_LABELS` values).

    Returns
    -------
    dict
        Keys ``f"{stat}_{var}_{box_label}"`` for every stat in :data:`STATS`.
        All-NaN / empty boxes yield ``np.nan`` for every key.
    """
    arr = np.asarray(getattr(values, "values", values)).ravel()
    valid = arr[np.isfinite(arr)]

    result: dict[str, float] = {}
    if valid.size == 0:
        result[f"min_{var}_{box_label}"] = np.nan
        result[f"max_{var}_{box_label}"] = np.nan
        result[f"mean_{var}_{box_label}"] = np.nan
        result[f"std_{var}_{box_label}"] = np.nan
        for p in PERCENTILES:
            result[f"p{p}_{var}_{box_label}"] = np.nan
        return result

    result[f"min_{var}_{box_label}"] = float(np.min(valid))
    result[f"max_{var}_{box_label}"] = float(np.max(valid))
    result[f"mean_{var}_{box_label}"] = float(np.mean(valid))
    result[f"std_{var}_{box_label}"] = float(np.std(valid))
    for p in PERCENTILES:
        result[f"p{p}_{var}_{box_label}"] = float(np.percentile(valid, p))
    return result


# --- Deterministic column ordering ---------------------------------------
def _centroid_columns() -> list[str]:
    """Centroid value columns, in STAT_VARS order: ``f"{var}_centroid"``."""
    return [f"{var}_centroid" for var in STAT_VARS]


def _box_stat_columns() -> list[str]:
    """Box-stat columns in stable order: var-outer, box-outer, stat-inner.

    Order is ``for var in STAT_VARS: for radius in BOX_RADII: for stat in
    STATS`` -> ``f"{stat}_{var}_{box_label}"`` (8 x 3 x 10 = 240 columns).
    """
    cols: list[str] = []
    for var in STAT_VARS:
        for radius in BOX_RADII:
            box_label = BOX_LABELS[radius]
            for stat in STATS:
                cols.append(f"{stat}_{var}_{box_label}")
    return cols


# Stable, documented column order: 4 meta + 8 centroid + 240 box-stat = 252.
META_COLUMNS = ["feature_id", "mission", "orbit", "time"]
CENTROID_COLUMNS = _centroid_columns()
BOX_STAT_COLUMNS = _box_stat_columns()
VALUE_COLUMNS = CENTROID_COLUMNS + BOX_STAT_COLUMNS
ALL_COLUMNS = META_COLUMNS + VALUE_COLUMNS


def _build_schema() -> pa.Schema:
    """Build ERA5_SCHEMA in the stable ALL_COLUMNS order."""
    fields = [
        pa.field("feature_id", pa.int64()),
        pa.field("mission", pa.string()),
        pa.field("orbit", pa.int32()),
        pa.field("time", pa.timestamp("us")),
    ]
    for col in VALUE_COLUMNS:
        fields.append(pa.field(col, pa.float32()))
    return pa.schema(fields)


ERA5_SCHEMA: pa.Schema = _build_schema()


# --- Per-box extraction helpers ------------------------------------------
def _box_slices(clat: float, clon360: float, radius: float):
    """Grid-aligned (lat_sel, lon_min, lon_max) for a centroid + radius.

    Latitude slice is high->low (ERA-5 descending). Longitude bounds may fall
    outside ``[0, 360)``; :func:`get_cyclic_subset` handles the wrap.
    """
    clat_grid = np.round(clat / _GRID) * _GRID
    clon_grid = np.round(clon360 / _GRID) * _GRID

    lat_max = np.ceil((clat_grid + radius) / _GRID) * _GRID
    lat_min = np.floor((clat_grid - radius) / _GRID) * _GRID
    lat_sel = slice(lat_max, lat_min)

    lon_min = np.floor((clon_grid - radius) / _GRID) * _GRID
    lon_max = np.ceil((clon_grid + radius) / _GRID) * _GRID
    return lat_sel, lon_min, lon_max


# --- Single global per-hour load -----------------------------------------
def _load_hour(ds, hour):
    """Fetch ONE global ERA-5 field for a single hour (the only download).

    Selects the needed variables (the 5 2-D vars + 10 m wind + level-sliced
    3-D geopotential/u/v over :data:`LEVEL_SLICE` hPa) for the nearest hour and
    ``.compute()`` s the FULL GLOBAL field into memory (no lat/lon bounding box,
    no cross-zero concat). Shear fields are computed once on the loaded array.

    Parameters
    ----------
    ds : xarray.Dataset
        Open ERA-5 dataset.
    hour : pandas.Timestamp
        Nearest ERA-5 hour to load.

    Returns
    -------
    tuple[xarray.Dataset, dict]
        ``(subset, shear_fields)`` where ``subset`` is the numpy-backed global
        field for ``hour`` and ``shear_fields`` is :func:`compute_shear_fields`
        output. Increments the module-level :data:`FETCH_COUNT` by one.
    """
    global FETCH_COUNT

    all_vars = (
        list(ERA5_VARS_2D.keys())
        + list(WIND10)
        + list(ERA5_VARS_3D)
    )

    ds_lazy = (
        ds[all_vars]
        .sel(time=hour, method="nearest")
        .sel(level=slice(LEVEL_SLICE[0], LEVEL_SLICE[1]))
    )
    subset = ds_lazy.compute()  # the single global download for this hour
    FETCH_COUNT += 1
    shear_fields = compute_shear_fields(subset)
    return subset, shear_fields


# --- Main co-location -----------------------------------------------------
def era5_for_features(features: pd.DataFrame, ds=None) -> pd.DataFrame:
    """Co-locate ERA-5 to each feature centroid + box stats.

    Parameters
    ----------
    features : pandas.DataFrame
        Must contain columns ``feature_id``, ``mission``, ``orbit``, ``time``,
        ``centroid_lat``, ``centroid_lon`` (longitude in -180..180 or 0..360).
    ds : xarray.Dataset, optional
        Open ERA-5 dataset. Opened via :func:`open_era5` if ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per feature with columns :data:`ALL_COLUMNS` (4 meta + 8
        centroid + 240 box-stat = 252). Features whose ERA-5 fetch fails are
        emitted with NaN value columns.

    Notes
    -----
    Features are batched by their nearest ERA-5 hour. Each UNIQUE hour is
    downloaded EXACTLY ONCE: :func:`_load_hour` selects the needed variables
    (5 2-D vars + 10 m wind + level-sliced 3-D vars) and ``.compute()`` s the
    FULL GLOBAL field into memory, computing shear fields once. ALL features in
    that hour are then matched against that single in-memory (numpy-backed)
    global field -- centroid value is the grid-nearest cell, box stats use a
    cyclic-safe lat/lon subset (the only cyclic handling left, for boxes near
    the 0/360 seam). No further I/O per feature. The in-memory field is freed
    (``del`` + ``gc.collect``) after each hour to bound memory. The module-level
    :data:`FETCH_COUNT` records how many unique hours were fetched.
    """
    import xarray as xr  # noqa: F401  (ensure xarray import error surfaces early)

    global FETCH_COUNT
    FETCH_COUNT = 0

    if features is None or len(features) == 0:
        return pd.DataFrame(columns=ALL_COLUMNS)

    if ds is None:
        ds = open_era5()

    feats = features.copy()
    feats["time"] = pd.to_datetime(feats["time"])
    # Nearest ERA-5 hour for batching.
    feats["_hour"] = feats["time"].dt.round("h")

    rows: list[dict] = []

    # Carry the original input position so the per-hour batching does not
    # reorder the returned rows: features are emitted in input order.
    by_hour = collections.defaultdict(list)
    for pos, rec in enumerate(feats.to_dict(orient="records")):
        rec["_order"] = pos
        by_hour[rec["_hour"]].append(rec)

    n_features = sum(len(recs) for recs in by_hour.values())
    logger.info(
        "ERA-5 co-location: %d features over %d unique hours "
        "(one global download per hour)",
        n_features,
        len(by_hour),
    )

    for hour, recs in by_hour.items():
        try:
            # The single global download for this hour, matched in memory.
            subset, shear_fields = _load_hour(ds, hour)
        except Exception:  # noqa: BLE001
            logger.warning("ERA-5 fetch failed for hour %s; emitting NaN rows", hour)
            for r in recs:
                rows.append(_nan_row(r))
            continue

        for r in recs:
            rows.append(_feature_row(r, subset, shear_fields))

        # Bound memory: drop this hour's global field before the next fetch.
        del subset, shear_fields
        gc.collect()

    logger.info(
        "ERA-5 co-location complete: %d features matched, %d hours fetched",
        n_features,
        FETCH_COUNT,
    )

    out = pd.DataFrame(rows)
    # Restore input feature order (per-hour batching emits rows out of order),
    # then enforce the stable column order (drops the internal _order key).
    if "_order" in out.columns:
        out = out.sort_values("_order", kind="stable").reset_index(drop=True)
    out = out.reindex(columns=ALL_COLUMNS)
    return out


def _nan_row(rec: dict) -> dict:
    """A feature row with NaN value columns (used when the fetch failed)."""
    row = {
        "feature_id": int(rec["feature_id"]),
        "mission": str(rec["mission"]),
        "orbit": int(rec["orbit"]),
        "time": pd.to_datetime(rec["time"]),
    }
    for col in VALUE_COLUMNS:
        row[col] = np.nan
    if "_order" in rec:
        row["_order"] = rec["_order"]
    return row


def _feature_row(rec: dict, subset, shear_fields: dict) -> dict:
    """Build one feature's row: centroid values + box stats from ``subset``.

    ``subset`` is the in-memory global field for the feature's hour (loaded once
    by :func:`_load_hour`); all centroid/box extraction below is pure in-memory
    xarray/numpy indexing -- no further I/O. The only cyclic handling is in the
    per-feature box subset via :func:`get_cyclic_subset` for boxes that straddle
    the 0/360 longitude seam.
    """
    row = {
        "feature_id": int(rec["feature_id"]),
        "mission": str(rec["mission"]),
        "orbit": int(rec["orbit"]),
        "time": pd.to_datetime(rec["time"]),
    }
    if "_order" in rec:
        row["_order"] = rec["_order"]

    clat = float(rec["centroid_lat"])
    clon360 = float(rec["centroid_lon"]) % 360

    # --- centroid values (grid-nearest) ---------------------------------
    for era5_var, short in ERA5_VARS_2D.items():
        try:
            val = subset[era5_var].sel(
                latitude=clat, longitude=clon360, method="nearest"
            ).values
            row[f"{short}_centroid"] = float(val)
        except Exception:  # noqa: BLE001
            row[f"{short}_centroid"] = np.nan

    for height_m in SHEAR_HEIGHTS_M:
        name = f"shear_{height_m}m"
        field = shear_fields.get(height_m)
        if field is None:
            row[f"{name}_centroid"] = np.nan
            continue
        try:
            val = field.sel(
                latitude=clat, longitude=clon360, method="nearest"
            ).values
            row[f"{name}_centroid"] = float(val)
        except Exception:  # noqa: BLE001
            row[f"{name}_centroid"] = np.nan

    # --- box stats ------------------------------------------------------
    for radius in BOX_RADII:
        box_label = BOX_LABELS[radius]
        lat_sel, lon_min, lon_max = _box_slices(clat, clon360, radius)

        # 2-D vars
        try:
            box_ds = get_cyclic_subset(subset, lat_sel, lon_min, lon_max)
        except Exception:  # noqa: BLE001
            box_ds = None

        for era5_var, short in ERA5_VARS_2D.items():
            if box_ds is None:
                row.update(_nan_stats(short, box_label))
            else:
                try:
                    row.update(compute_stats(box_ds[era5_var], short, box_label))
                except Exception:  # noqa: BLE001
                    row.update(_nan_stats(short, box_label))

        # shear vars
        for height_m in SHEAR_HEIGHTS_M:
            name = f"shear_{height_m}m"
            field = shear_fields.get(height_m)
            if field is None:
                row.update(_nan_stats(name, box_label))
                continue
            try:
                shear_box = get_cyclic_subset(field, lat_sel, lon_min, lon_max)
                row.update(compute_stats(shear_box, name, box_label))
            except Exception:  # noqa: BLE001
                row.update(_nan_stats(name, box_label))

    return row


def _nan_stats(var: str, box_label: str) -> dict:
    """All-NaN stat dict for one (var, box)."""
    return {f"{stat}_{var}_{box_label}": np.nan for stat in STATS}


# --- Writer (mirror catalog.py patterns) ---------------------------------
def write_era5(era5_df: pd.DataFrame, mission: str, root: str = PF_ROOT) -> Path:
    """Write one orbit's ERA-5 table to the hive-partitioned dataset.

    Layout::

        {root}/era5/mission={M}/year={YYYY}/month={MM}/orbit={NNNNNN}.parquet

    ``year``/``month`` come from the mean of ``era5_df['time']``; ``orbit`` is
    zero-padded to 6 digits; ``mission`` is upper-cased. The table is cast to
    :data:`ERA5_SCHEMA`, written zstd-compressed with ``coerce_timestamps='us'``
    via an atomic ``.tmp`` -> :func:`os.replace`.

    Parameters
    ----------
    era5_df : pandas.DataFrame
        Output of :func:`era5_for_features` (must contain ``time`` and
        ``orbit`` columns).
    mission : str
        Mission name; upper-cased for the partition key.
    root : str, optional
        Dataset root (default :data:`pf.config.PF_ROOT`).

    Returns
    -------
    pathlib.Path
        The written Parquet file path.

    Raises
    ------
    ValueError
        If ``era5_df`` is empty or lacks ``time``/``orbit``.
    """
    if era5_df is None or len(era5_df) == 0:
        raise ValueError("era5_df must be a non-empty DataFrame")
    if "time" not in era5_df.columns:
        raise ValueError("era5_df must contain a 'time' column")
    if "orbit" not in era5_df.columns:
        raise ValueError("era5_df must contain an 'orbit' column")

    mission_key = str(mission).upper()

    times = pd.to_datetime(era5_df["time"])
    mean_time = times.mean()
    if pd.isna(mean_time):
        raise ValueError("era5_df['time'] has no valid timestamps")
    year = int(mean_time.year)
    month = int(mean_time.month)

    orbit = int(era5_df["orbit"].iloc[0])
    orbit_str = f"{orbit:06d}"

    df = era5_df.reindex(columns=ALL_COLUMNS)
    df = _coerce_time_us(df)
    table = pa.Table.from_pandas(df, schema=ERA5_SCHEMA, preserve_index=False)

    target_dir = (
        Path(root)
        / ERA5_ROOT_SUBDIR
        / f"mission={mission_key}"
        / f"year={year:04d}"
        / f"month={month:02d}"
    )
    target = target_dir / f"orbit={orbit_str}.parquet"

    target_dir.mkdir(parents=True, exist_ok=True)
    # Process-unique temp name so a stale/partial .tmp from another process can
    # never break a later read or a concurrent writer. Remove any pre-existing
    # temp before writing, then atomically replace the final file.
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    pq.write_table(
        table,
        tmp,
        compression="zstd",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    os.replace(tmp, target)

    return target
