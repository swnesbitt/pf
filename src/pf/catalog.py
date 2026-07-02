"""Per-orbit Parquet catalog writer.

Writes one orbit's feature table (and, from Phase 2, its pixel table) to a
hive-partitioned Parquet layout::

    {root}/features/mission={M}/year={YYYY}/month={MM}/orbit={NNNNNN}.parquet
    {root}/pixels/  mission={M}/year={YYYY}/month={MM}/orbit={NNNNNN}.parquet

``year``/``month`` are derived from the **mean** of ``features_df['time']``;
``orbit`` is zero-padded to 6 digits; ``mission`` is upper-cased. Writes are
atomic (``.tmp`` then :func:`os.replace`) and per-orbit isolated — there is no
appending or locking. The feature table is cast to the frozen
:data:`pf.features.FEATURE_SCHEMA` before writing; ``orbit`` is kept as a data
column even though it is also a partition key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pf.config import PF_ROOT
from pf.features import FEATURE_SCHEMA, PIXEL_SCHEMA

_COMPRESSION = "zstd"


def _coerce_time_us(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with any datetime ``time`` column truncated to us.

    The frozen schemas declare ``time`` as ``timestamp('us')``. The strict
    schema cast in :func:`pyarrow.Table.from_pandas` raises ``ArrowInvalid`` for
    ``datetime64[ns]`` values that are not exact multiples of one microsecond
    (e.g. the mean of member scan times). Truncating ns->us up front via
    ``astype`` makes that cast lossless; sub-microsecond precision is
    meaningless for satellite scan times. No-op if there is no datetime ``time``
    column (so it is safe for the pixel path, which has none in Phase 1).
    """
    if "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"]):
        df = df.copy()
        df["time"] = df["time"].astype("datetime64[us]")
    return df


def _partition_dir(
    root: str, table: str, mission: str, year: int, month: int
) -> Path:
    """Build the hive partition directory for ``table`` of one orbit.

    Parameters
    ----------
    root : str
        Catalog root directory.
    table : str
        Either ``"features"`` or ``"pixels"``.
    mission : str
        Upper-cased mission name.
    year, month : int
        Partition year and month.

    Returns
    -------
    pathlib.Path
        The partition directory (not yet created).
    """
    return (
        Path(root)
        / table
        / f"mission={mission}"
        / f"year={year:04d}"
        / f"month={month:02d}"
    )


def _write_atomic(table: pa.Table, target: Path) -> None:
    """Write ``table`` to ``target`` atomically via a sibling ``.tmp`` file.

    Parameters
    ----------
    table : pyarrow.Table
        Table to write.
    target : pathlib.Path
        Final destination path. Its parent directory is created if needed.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    pq.write_table(
        table,
        tmp,
        compression=_COMPRESSION,
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    os.replace(tmp, target)


def write_orbit(
    features_df: pd.DataFrame,
    pixels_df: pd.DataFrame | None,
    mission: str,
    root: str = PF_ROOT,
) -> tuple[str, str]:
    """Write one orbit's feature (and optional pixel) tables to Parquet.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Feature rows matching :data:`pf.features.FEATURE_SCHEMA`. Must contain
        a ``time`` column and an ``orbit`` column.
    pixels_df : pandas.DataFrame or None
        Per-pixel rows matching :data:`pf.features.PIXEL_SCHEMA`. ``None`` or an
        empty DataFrame (Phase-1) skips the pixel write but still returns the
        target pixel path.
    mission : str
        Mission name; upper-cased for the partition key.
    root : str, optional
        Catalog root directory (default :data:`pf.config.PF_ROOT`).

    Returns
    -------
    tuple of (str, str)
        ``(features_path, pixels_path)``. ``pixels_path`` is the target path
        regardless of whether a pixel table was written.

    Raises
    ------
    ValueError
        If ``features_df`` is empty or lacks the required ``time``/``orbit``
        columns.
    """
    if features_df is None or len(features_df) == 0:
        raise ValueError("features_df must be a non-empty DataFrame")
    if "time" not in features_df.columns:
        raise ValueError("features_df must contain a 'time' column")
    if "orbit" not in features_df.columns:
        raise ValueError("features_df must contain an 'orbit' column")

    mission_key = str(mission).upper()

    # Partition year/month from the MEAN feature time.
    times = pd.to_datetime(features_df["time"])
    mean_time = times.mean()
    if pd.isna(mean_time):
        raise ValueError("features_df['time'] has no valid timestamps")
    year = int(mean_time.year)
    month = int(mean_time.month)

    # Orbit comes from the data (kept as a column); zero-padded for the path.
    orbit = int(features_df["orbit"].iloc[0])
    orbit_str = f"{orbit:06d}"

    # --- features (cast to the frozen schema; never drop columns) --------
    # Truncate sub-microsecond ns time to us so the strict schema cast to
    # timestamp('us') is lossless (mean scan time is not us-aligned).
    features_df = _coerce_time_us(features_df)
    features_table = pa.Table.from_pandas(
        features_df, schema=FEATURE_SCHEMA, preserve_index=False
    )
    features_dir = _partition_dir(root, "features", mission_key, year, month)
    features_path = features_dir / f"orbit={orbit_str}.parquet"
    _write_atomic(features_table, features_path)

    # --- pixels (target path always returned; written only if non-empty) -
    pixels_dir = _partition_dir(root, "pixels", mission_key, year, month)
    pixels_path = pixels_dir / f"orbit={orbit_str}.parquet"
    if pixels_df is not None and len(pixels_df) > 0:
        pixels_df = _coerce_time_us(pixels_df)
        pixels_table = pa.Table.from_pandas(
            pixels_df, schema=PIXEL_SCHEMA, preserve_index=False
        )
        _write_atomic(pixels_table, pixels_path)

    return (str(features_path), str(pixels_path))
