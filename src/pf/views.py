"""Per-orbit radar *sampling* (pixel-view) gridding — Part A of the climatology.

The gridded rain climatology (:mod:`pf.grid`) needs a denominator: how many times
each 0.05 deg grid cell was *observed* by the radar, raining or not. That count
cannot be recovered from the feature/pixel Parquet tables (which hold only
feature pixels), so it is captured here as a byproduct of the radar pipeline,
which already reads every swath.

For each orbit this module grids every valid near-surface radar observation into
0.05 deg cells and writes a SPARSE per-orbit table (one row per touched cell)
alongside ``features/`` and ``pixels/``::

    {root}/views/mission={M}/year={YYYY}/month={MM}/orbit={NNNNNN}.parquet

A *view* is a pixel with finite ``lat`` AND finite ``lon`` AND finite
``near_sfc_rain`` (``rain == 0`` counts; a NaN rain is a non-observation / fill
and does not). Writes are atomic and idempotent, mirroring
:func:`pf.era5.write_era5`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pf import config
from pf.catalog import _coerce_time_us

#: Sparse per-orbit views schema. ``lat_bin`` in [0, GRID_N_LAT), ``lon_bin`` in
#: [0, GRID_N_LON); both fit int16 (max 7199 < 32767). ``mission``/``orbit`` are
#: kept as columns (also partition keys); ``time`` drives year/month partitions.
VIEWS_SCHEMA = pa.schema(
    [
        pa.field("lat_bin", pa.int16()),
        pa.field("lon_bin", pa.int16()),
        pa.field("n_views", pa.int32()),
        pa.field("mission", pa.string()),
        pa.field("orbit", pa.int32()),
        pa.field("time", pa.timestamp("us")),
    ]
)


def grid_orbit_views(
    swath,
    *,
    grid_deg: float = config.GRID_DEG,
    lat_min: float = config.GRID_LAT_MIN,
    lon_min: float = config.GRID_LON_MIN,
    n_lat: int = config.GRID_N_LAT,
    n_lon: int = config.GRID_N_LON,
) -> pd.DataFrame | None:
    """Grid one orbit's near-surface views into sparse 0.05 deg cell counts.

    Parameters
    ----------
    swath : pf.swath.Swath
        Orbit swath; uses ``lat``/``lon`` (2-D, deg, fills already NaN),
        ``near_sfc_rain`` (NaN = non-observation), and per-scan ``time``.
    grid_deg, lat_min, lon_min, n_lat, n_lon
        Grid definition (defaults from :mod:`pf.config`).

    Returns
    -------
    pandas.DataFrame or None
        Columns ``lat_bin, lon_bin, n_views, mission, orbit, time`` — one row per
        touched cell. ``None`` if the swath has no valid view.
    """
    lat = np.asarray(swath.lat, dtype=np.float64)
    lon = np.asarray(swath.lon, dtype=np.float64)
    rain = np.asarray(swath.near_sfc_rain, dtype=np.float64)

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(rain)
    if not valid.any():
        return None

    latv = lat[valid]
    # Wrap longitude into [-180, 180) so antimeridian pixels bin correctly.
    lonv = ((lon[valid] - lon_min) % 360.0) + lon_min

    lat_idx = np.floor((latv - lat_min) / grid_deg).astype(np.int64)
    lon_idx = np.floor((lonv - lon_min) / grid_deg).astype(np.int64)
    np.clip(lat_idx, 0, n_lat - 1, out=lat_idx)
    np.clip(lon_idx, 0, n_lon - 1, out=lon_idx)

    # Sparse count over touched cells only (most of the 26M-cell grid is empty).
    flat = lat_idx * n_lon + lon_idx
    uniq, counts = np.unique(flat, return_counts=True)
    cell_lat, cell_lon = np.divmod(uniq, n_lon)

    mean_time = _mean_scan_time(swath)

    return pd.DataFrame(
        {
            "lat_bin": cell_lat.astype(np.int16),
            "lon_bin": cell_lon.astype(np.int16),
            "n_views": counts.astype(np.int32),
            "mission": str(getattr(swath, "mission", "")).upper(),
            "orbit": np.int32(int(getattr(swath, "orbit", 0))),
            "time": mean_time,
        }
    )


def _mean_scan_time(swath) -> np.datetime64:
    """Mean of the swath's per-scan times (NaT if none finite). Mirrors features.py."""
    t = np.asarray(getattr(swath, "time", np.array([], dtype="datetime64[ns]")))
    if t.size == 0:
        return np.datetime64("NaT", "ns")
    finite = t[~np.isnat(t)]
    if finite.size == 0:
        return np.datetime64("NaT", "ns")
    mean_ns = finite.astype("datetime64[ns]").astype(np.int64).mean()
    return np.datetime64(int(round(mean_ns)), "ns")


def write_orbit_views(
    views_df: pd.DataFrame | None,
    mission: str,
    root: str = config.PF_ROOT,
) -> Path | None:
    """Atomically write one orbit's sparse views table; idempotent.

    Layout ``{root}/views/mission={M}/year={YYYY}/month={MM}/orbit={NNNNNN}.parquet``;
    ``year``/``month`` from the mean ``time``, ``orbit`` zero-padded to 6 digits.
    Returns the path, or ``None`` if ``views_df`` is empty/None.
    """
    if views_df is None or len(views_df) == 0:
        return None

    mission_key = str(mission).upper()
    mean_time = pd.to_datetime(views_df["time"]).mean()
    if pd.isna(mean_time):
        return None
    year = int(mean_time.year)
    month = int(mean_time.month)
    orbit = int(views_df["orbit"].iloc[0])
    orbit_str = f"{orbit:06d}"

    # Truncate ns->us so the strict schema cast in from_pandas does not raise
    # ArrowInvalid on mean scan times that carry sub-microsecond remainder
    # (same convention as features/pixels/era5).
    df = _coerce_time_us(views_df.reindex(columns=[f.name for f in VIEWS_SCHEMA]))
    table = pa.Table.from_pandas(df, schema=VIEWS_SCHEMA, preserve_index=False)

    target_dir = (
        Path(root)
        / config.VIEWS_ROOT_SUBDIR
        / f"mission={mission_key}"
        / f"year={year:04d}"
        / f"month={month:02d}"
    )
    target = target_dir / f"orbit={orbit_str}.parquet"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Process-unique temp -> atomic replace (same convention as pf.era5).
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    pq.write_table(table, tmp, compression="zstd",
                   coerce_timestamps="us", allow_truncated_timestamps=True)
    os.replace(tmp, target)
    return target


__all__ = ["VIEWS_SCHEMA", "grid_orbit_views", "write_orbit_views"]
