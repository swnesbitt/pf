"""Swath-gridded, hour-resolved rain/views accumulation (pure, no I/O).

Where :mod:`pf.views` grids only the *sampling denominator* (``n_views``) per
orbit, this module grids the full near-surface rain field of one swath into
0.05 deg cells with a **UTC hour-of-day axis (24 bins)** and stratifies the rain
by storm **size** / **20-dBZ echo-top** / per-pixel **rain type**. It is the core
of the swath-gridded climatology (Stage 1): the gridded ``rain``/``views``/
``raining_views`` come straight from the swath (not the feature Parquet), so the
numerator and the ``n_views`` denominator share the same pixel universe, and the
diurnal cycle is preserved.

Categories are **recomputed from the swath** (not joined from Parquet) using the
exact calls :func:`pf.features.build_feature_row` makes — ``geometry.pca_axes``
for the major axis and ``echotop_qc.feature_echo_tops`` for the 20-dBZ top — so
the recomputed classes match the stored feature DB (given the same label
thresholds). Raining pixels that belong to no retained feature fall in the
existing "undefined" size/echo-top slot; their rain type still comes per-pixel
from the radar product.

Two sparse DataFrames are returned (one row per touched key):

* ``views_df``: ``lat_bin, lon_bin, hour, n_views`` — every valid observed pixel.
* ``rain_df`` : ``lat_bin, lon_bin, hour, size_class, echotop_class, raintype,
  rain_sum, raining_count`` — pixels with ``rain > 0`` and a 1/2/3 rain type.

:func:`build_metrics` adds a third sparse table on the same ``(lat_bin, lon_bin,
hour)`` key: per-pixel convective echo-tops (20/30/40 dBZ), heavy near-surface
rain-rate occurrence counts (>25/50/75/100 mm/hr), and the near-surface DSD
parameters (epsilon, Nw, Dm) split convective/stratiform — each a paired
``_sum``/``_n`` so the climatology pools correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf import config, echotop_qc, geometry, grid

# Class cardinalities (reused from pf.grid — do not redefine).
SIZE_SLOTS = grid.SIZE_SLOTS
ECHOTOP_SLOTS = grid.ECHOTOP_SLOTS
RAINTYPE_N = grid.RAINTYPE_N
N_HOURS = 24

# Guard: the composite flat key must fit in signed int64.
_MAX_KEY = (((((config.GRID_N_LAT - 1) * config.GRID_N_LON + (config.GRID_N_LON - 1))
              * N_HOURS + (N_HOURS - 1)) * SIZE_SLOTS + (SIZE_SLOTS - 1))
            * ECHOTOP_SLOTS + (ECHOTOP_SLOTS - 1)) * RAINTYPE_N + (RAINTYPE_N - 1)
assert _MAX_KEY < 2 ** 63, "composite grid key overflows int64"


def _hour_of_day_2d(swath) -> np.ndarray:
    """Per-pixel UTC hour-of-day (0..23), or -1 where the scan time is NaT.

    ``swath.time`` is per-scan ``(nscan,)``; broadcast it across rays so every
    pixel inherits its scan's hour (the load-bearing time-vs-space broadcast).
    """
    t = np.asarray(swath.time)
    if t.dtype.kind != "M":
        t = t.astype("datetime64[ns]")
    hours = t.astype("datetime64[h]").astype("int64") % N_HOURS
    hour_scan = np.where(np.isnat(t), -1, hours).astype(np.int64)  # (nscan,)
    return np.broadcast_to(hour_scan[:, None], swath.lat.shape)


def build_class_maps(swath, labeled, kept, *, cfg=config):
    """Per-pixel 2-D size/echo-top class maps recomputed from the swath.

    Returns ``(size_map, echotop_map)``, both int16 ``(nscan, nray)`` initialised
    to the undefined slot. For each retained feature in ``kept`` the major-axis
    and 20-dBZ echo-top are computed once (same calls as
    ``features.build_feature_row``) and scattered onto its member pixels.
    """
    size_map = np.full(swath.lat.shape, grid.SIZE_UNDEF, dtype=np.int16)
    echotop_map = np.full(swath.lat.shape, grid.ECHOTOP_UNDEF, dtype=np.int16)
    for local_label, _area_km2 in kept:
        member = labeled == local_label
        if not member.any():
            continue
        major_km = geometry.pca_axes(swath.lat[member], swath.lon[member])[0]
        max_ht20 = echotop_qc.feature_echo_tops(swath, member, cfg=cfg)["max_ht_20dbz"]
        sc = int(grid.size_class(np.array([major_km], dtype=np.float64))[0])
        ec = int(grid.echotop_class(np.array([max_ht20], dtype=np.float64))[0])
        size_map[member] = sc
        echotop_map[member] = ec
    return size_map, echotop_map


def _bin_latlon(lat, lon, *, grid_deg, lat_min, lon_min, n_lat, n_lon):
    """Floor-bin lat/lon to grid indices (lon wrapped, clipped). 1-D in/out."""
    lonw = ((lon - lon_min) % 360.0) + lon_min
    lat_idx = np.floor((lat - lat_min) / grid_deg).astype(np.int64)
    lon_idx = np.floor((lonw - lon_min) / grid_deg).astype(np.int64)
    np.clip(lat_idx, 0, n_lat - 1, out=lat_idx)
    np.clip(lon_idx, 0, n_lon - 1, out=lon_idx)
    return lat_idx, lon_idx


def grid_swath(
    swath,
    labeled,
    kept,
    *,
    grid_deg: float = config.GRID_DEG,
    lat_min: float = config.GRID_LAT_MIN,
    lon_min: float = config.GRID_LON_MIN,
    n_lat: int = config.GRID_N_LAT,
    n_lon: int = config.GRID_N_LON,
    time_window=None,
    cfg=config,
):
    """Grid one swath into hour-resolved sparse ``(views_df, rain_df)``.

    A *view* is a pixel with finite ``lat``/``lon``/``near_sfc_rain`` and a valid
    scan time (``rain == 0`` counts; NaN rain / NaT time excluded). The rain frame
    additionally requires ``rain > 0`` and a 1/2/3 rain type. Either component is
    ``None`` when empty.

    ``time_window=(t0, t1)`` (half-open, UTC) restricts gridding to scans whose
    time lies in ``[t0, t1)`` — the per-scan analogue of month binning. A boundary
    orbit crossing a month edge thus contributes each scan to exactly one month
    (mirroring the per-scan hour axis), with no double-count across adjacent
    months. ``None`` grids every valid scan.
    """
    import pandas as pd

    lat = np.asarray(swath.lat, dtype=np.float64)
    lon = np.asarray(swath.lon, dtype=np.float64)
    rain = np.asarray(swath.near_sfc_rain, dtype=np.float64)

    hour_2d = _hour_of_day_2d(swath)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(rain) & (hour_2d >= 0)
    if time_window is not None:
        t = np.asarray(swath.time)
        t0 = np.datetime64(pd.Timestamp(time_window[0]))
        t1 = np.datetime64(pd.Timestamp(time_window[1]))
        in_win_scan = (~np.isnat(t)) & (t >= t0) & (t < t1)        # (nscan,)
        valid = valid & np.broadcast_to(in_win_scan[:, None], swath.lat.shape)
    if not valid.any():
        return None, None

    bin_kw = dict(grid_deg=grid_deg, lat_min=lat_min, lon_min=lon_min,
                  n_lat=n_lat, n_lon=n_lon)

    # --- views: every valid observed pixel, by (cell, hour) ----------------
    vlat_idx, vlon_idx = _bin_latlon(lat[valid], lon[valid], **bin_kw)
    vhour = hour_2d[valid]
    vflat = (vlat_idx * n_lon + vlon_idx) * N_HOURS + vhour
    vuniq, vcounts = np.unique(vflat, return_counts=True)
    v_hour = vuniq % N_HOURS
    v_cell = vuniq // N_HOURS
    v_lat, v_lon = np.divmod(v_cell, n_lon)
    views_df = pd.DataFrame(
        {
            "lat_bin": v_lat.astype(np.int16),
            "lon_bin": v_lon.astype(np.int16),
            "hour": v_hour.astype(np.int8),
            "n_views": vcounts.astype(np.int64),
        }
    )

    # --- rain: raining pixels with 1/2/3 rain type, by (cell, hour, class) --
    raintype = grid.raintype_class(swath.rain_type)            # (nscan,nray), -1 = drop
    rain_mask = valid & (rain > 0.0) & (raintype >= 0)
    if not rain_mask.any():
        return views_df, None

    size_map, echotop_map = build_class_maps(swath, labeled, kept, cfg=cfg)
    rlat_idx, rlon_idx = _bin_latlon(lat[rain_mask], lon[rain_mask], **bin_kw)
    rhour = hour_2d[rain_mask]
    rsize = size_map[rain_mask].astype(np.int64)
    recho = echotop_map[rain_mask].astype(np.int64)
    rrt = raintype[rain_mask].astype(np.int64)
    rvals = rain[rain_mask]

    key = ((((((rlat_idx * n_lon + rlon_idx) * N_HOURS + rhour)
              * SIZE_SLOTS + rsize) * ECHOTOP_SLOTS + recho) * RAINTYPE_N + rrt))
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    rain_sum = np.bincount(inv, weights=rvals)
    raining_count = np.bincount(inv)

    u = uniq
    rt_d = u % RAINTYPE_N;        u //= RAINTYPE_N
    ec_d = u % ECHOTOP_SLOTS;     u //= ECHOTOP_SLOTS
    sz_d = u % SIZE_SLOTS;        u //= SIZE_SLOTS
    hr_d = u % N_HOURS;           u //= N_HOURS
    lon_d = u % n_lon
    lat_d = u // n_lon
    rain_df = pd.DataFrame(
        {
            "lat_bin": lat_d.astype(np.int16),
            "lon_bin": lon_d.astype(np.int16),
            "hour": hr_d.astype(np.int8),
            "size_class": sz_d.astype(np.int8),
            "echotop_class": ec_d.astype(np.int8),
            "raintype": rt_d.astype(np.int8),
            "rain_sum": rain_sum.astype(np.float64),
            "raining_count": raining_count.astype(np.int64),
        }
    )
    return views_df, rain_df


# Heavy-rain near-surface thresholds (mm/hr) gridded as pixel-occurrence counts.
RAIN_THRESHOLDS = (25.0, 50.0, 75.0, 100.0)

# metrics_df columns (besides the lat_bin/lon_bin/hour key), in write order.
METRIC_COLS = [
    "et20_sum", "et20_n", "et30_sum", "et30_n", "et40_sum", "et40_n",
    "cnt_gt25", "cnt_gt50", "cnt_gt75", "cnt_gt100",
    "eps_conv_sum", "eps_conv_n", "eps_strat_sum", "eps_strat_n",
    "nw_conv_sum", "nw_conv_n", "nw_strat_sum", "nw_strat_n",
    "dm_conv_sum", "dm_conv_n", "dm_strat_sum", "dm_strat_n",
]


def build_metrics(
    swath,
    *,
    grid_deg: float = config.GRID_DEG,
    lat_min: float = config.GRID_LAT_MIN,
    lon_min: float = config.GRID_LON_MIN,
    n_lat: int = config.GRID_N_LAT,
    n_lon: int = config.GRID_N_LON,
    time_window=None,
    cfg=config,
):
    """Grid the new per-pixel metrics into sparse ``(lat_bin, lon_bin, hour)`` rows.

    Same pixel universe / validity / time-window rule as :func:`grid_swath` (a
    *view* = finite ``lat``/``lon``/``near_sfc_rain`` + valid scan hour), so these
    are interference-safe: profiles dropped to fill are NaN and never counted.
    Each mean is stored as a paired ``_sum`` + ``_n`` so the climatology pools
    correctly downstream (mean = Σ/ n, COMBINED via summed totals).

    Columns (see :data:`METRIC_COLS`):

    * ``et{20,30,40}_sum``/``_n`` — Σ & count of per-pixel CONVECTIVE echo-top
      height (m) at 20/30/40 dBZ (``echotop_qc.pixel_echo_tops`` on the
      convective, valid pixels; only finite tops counted).
    * ``cnt_gt{25,50,75,100}`` — count of valid pixels with
      ``near_sfc_rain >= threshold`` (mm/hr), any rain type.
    * ``{eps,nw,dm}_{conv,strat}_sum``/``_n`` — Σ & count of the near-surface
      DSD parameters (epsilon, Nw in dBNw, Dm in mm) over convective /
      stratiform valid pixels (finite values only).

    Returns a sparse ``metrics_df`` (one row per touched cell+hour) or ``None``.
    """
    import pandas as pd

    lat = np.asarray(swath.lat, dtype=np.float64)
    lon = np.asarray(swath.lon, dtype=np.float64)
    rain = np.asarray(swath.near_sfc_rain, dtype=np.float64)

    hour_2d = _hour_of_day_2d(swath)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(rain) & (hour_2d >= 0)
    if time_window is not None:
        t = np.asarray(swath.time)
        t0 = np.datetime64(pd.Timestamp(time_window[0]))
        t1 = np.datetime64(pd.Timestamp(time_window[1]))
        in_win_scan = (~np.isnat(t)) & (t >= t0) & (t < t1)
        valid = valid & np.broadcast_to(in_win_scan[:, None], swath.lat.shape)
    if not valid.any():
        return None

    bin_kw = dict(grid_deg=grid_deg, lat_min=lat_min, lon_min=lon_min,
                  n_lat=n_lat, n_lon=n_lon)

    # One (cell, hour) key universe over all valid pixels — every metric aligns to it.
    vlat_idx, vlon_idx = _bin_latlon(lat[valid], lon[valid], **bin_kw)
    vhour = hour_2d[valid]
    vkey = (vlat_idx * n_lon + vlon_idx) * N_HOURS + vhour
    uniq, inv = np.unique(vkey, return_inverse=True)
    inv = inv.ravel()
    nrows = uniq.size

    rt_v = np.asarray(swath.rain_type)[valid]
    rain_v = rain[valid]
    eps_v = np.asarray(swath.epsilon, dtype=np.float64)[valid]
    conv_v = rt_v == 2
    strat_v = rt_v == 1

    def _sum(weights, mask):
        return np.bincount(inv, weights=np.where(mask, weights, 0.0), minlength=nrows)

    def _count(mask):
        return np.bincount(inv, weights=mask.astype(np.float64), minlength=nrows).astype(np.int64)

    cols: dict = {}

    # --- convective echo-tops (m) at 20/30/40 dBZ, finite tops only ---------
    conv_full = np.zeros(swath.lat.shape, dtype=bool)
    conv_full[valid] = conv_v
    et = echotop_qc.pixel_echo_tops(swath, conv_full, cfg=cfg)
    for thr in (20, 30, 40):
        a = np.asarray(et[thr], dtype=np.float64)[valid]
        fin = np.isfinite(a)
        cols[f"et{thr}_sum"] = _sum(a, fin)
        cols[f"et{thr}_n"] = _count(fin)

    # --- heavy near-surface rain occurrence counts --------------------------
    for thr in RAIN_THRESHOLDS:
        cols[f"cnt_gt{int(thr)}"] = _count(rain_v >= thr)

    # --- DSD params (epsilon, Nw, Dm) split convective/stratiform (finite) ---
    for name, src in (("eps", eps_v),
                      ("nw", np.asarray(swath.nw, dtype=np.float64)[valid]),
                      ("dm", np.asarray(swath.dm, dtype=np.float64)[valid])):
        fin = np.isfinite(src)
        cols[f"{name}_conv_sum"] = _sum(src, conv_v & fin)
        cols[f"{name}_conv_n"] = _count(conv_v & fin)
        cols[f"{name}_strat_sum"] = _sum(src, strat_v & fin)
        cols[f"{name}_strat_n"] = _count(strat_v & fin)

    # Keep only cells with at least one nonzero metric (an echo-top, a heavy-rain
    # pixel, or an epsilon sample). All-zero rows carry no information and Stage-2
    # treats a missing key as zero, so dropping them keeps metrics.parquet compact
    # (like rain.parquet) instead of as dense as views.parquet.
    keep = np.zeros(nrows, dtype=bool)
    for c in METRIC_COLS:
        if not c.endswith("_sum"):          # the count columns (et*_n, cnt_gt*, eps_*_n)
            keep |= cols[c] > 0
    if not keep.any():
        return None

    sel_uniq = uniq[keep]
    u_hour = sel_uniq % N_HOURS
    u_cell = sel_uniq // N_HOURS
    u_lat, u_lon = np.divmod(u_cell, n_lon)
    out = {"lat_bin": u_lat.astype(np.int16), "lon_bin": u_lon.astype(np.int16),
           "hour": u_hour.astype(np.int8)}
    for c in METRIC_COLS:
        v = cols[c][keep]
        out[c] = v.astype(np.float64) if c.endswith("_sum") else v.astype(np.int64)
    return pd.DataFrame(out)


__all__ = ["build_class_maps", "grid_swath", "build_metrics", "N_HOURS",
           "RAIN_THRESHOLDS", "METRIC_COLS",
           "SIZE_SLOTS", "ECHOTOP_SLOTS", "RAINTYPE_N"]
