"""Tests for pf.views — per-orbit sampling (pixel-view) gridding (Part A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pf import config, views


def _bin(coord: float, origin: float) -> int:
    return int(np.floor((coord - origin) / config.GRID_DEG))


def test_known_pixels_to_known_cells(synthetic_swath):
    """A handful of placed pixels land in the expected sparse cells."""
    lat = np.array([[10.02, 10.02, 25.77]], dtype=np.float32)   # row 0
    lon = np.array([[100.03, 100.03, -60.40]], dtype=np.float32)
    rain = np.array([[1.5, 2.5, 0.0]], dtype=np.float32)        # all finite -> all views
    sw = synthetic_swath(nscan=1, nray=3, lat=lat, lon=lon, near_sfc_rain=rain,
                          mission="GPM", orbit=4242)

    df = views.grid_orbit_views(sw)
    assert df is not None
    # two pixels share one cell -> n_views=2; third is its own cell.
    cell_a = (_bin(10.02, config.GRID_LAT_MIN), _bin(100.03, config.GRID_LON_MIN))
    cell_b = (_bin(25.77, config.GRID_LAT_MIN), _bin(-60.40, config.GRID_LON_MIN))
    got = {(int(r.lat_bin), int(r.lon_bin)): int(r.n_views) for r in df.itertuples()}
    assert got == {cell_a: 2, cell_b: 1}
    assert (df["mission"] == "GPM").all()
    assert (df["orbit"] == 4242).all()


def test_view_definition_edge_cases(synthetic_swath):
    """rain==0 counts; NaN rain excluded; NaN lat/lon excluded."""
    lat = np.array([[10.0, 10.1, 10.2, 10.3]], dtype=np.float32)
    lon = np.array([[100.0, 100.1, 100.2, 100.3]], dtype=np.float32)
    rain = np.array([[0.0, np.nan, 5.0, 5.0]], dtype=np.float32)
    lat[0, 3] = np.nan  # 4th pixel has NaN lat -> excluded despite finite rain
    sw = synthetic_swath(nscan=1, nray=4, lat=lat, lon=lon, near_sfc_rain=rain)

    df = views.grid_orbit_views(sw)
    # valid views: pixel0 (rain 0) and pixel2 (rain 5). pixel1 NaN rain, pixel3 NaN lat.
    assert int(df["n_views"].sum()) == 2
    assert len(df) == 2


def test_conservation_total_views(synthetic_swath):
    """Sum of n_views equals the count of valid (finite lat/lon/rain) pixels."""
    rng = np.random.default_rng(0)
    rain = rng.uniform(0, 10, size=(8, 49)).astype(np.float32)
    rain[rain < 1.0] = np.nan  # some non-observations
    sw = synthetic_swath(nscan=8, nray=49, near_sfc_rain=rain)
    valid = (np.isfinite(sw.lat) & np.isfinite(sw.lon) & np.isfinite(rain)).sum()
    df = views.grid_orbit_views(sw)
    assert int(df["n_views"].sum()) == int(valid)


def test_longitude_wrap(synthetic_swath):
    """A pixel just past +180 wraps to the western edge, not out of range."""
    lat = np.array([[0.0]], dtype=np.float32)
    lon = np.array([[180.02]], dtype=np.float32)  # == -179.98 after wrap
    rain = np.array([[3.0]], dtype=np.float32)
    sw = synthetic_swath(nscan=1, nray=1, lat=lat, lon=lon, near_sfc_rain=rain)
    df = views.grid_orbit_views(sw)
    assert 0 <= int(df["lon_bin"].iloc[0]) < config.GRID_N_LON
    assert int(df["lon_bin"].iloc[0]) == _bin(-179.98, config.GRID_LON_MIN)


def test_all_invalid_returns_none(synthetic_swath):
    rain = np.full((4, 4), np.nan, dtype=np.float32)
    sw = synthetic_swath(nscan=4, nray=4, near_sfc_rain=rain)
    assert views.grid_orbit_views(sw) is None


def test_write_is_atomic_idempotent_and_hive_pathed(synthetic_swath, tmp_path):
    lat = np.array([[10.02, 10.07]], dtype=np.float32)
    lon = np.array([[100.03, 100.08]], dtype=np.float32)
    rain = np.array([[1.0, 2.0]], dtype=np.float32)
    sw = synthetic_swath(nscan=1, nray=2, lat=lat, lon=lon, near_sfc_rain=rain,
                         mission="GPM", orbit=4242)
    df = views.grid_orbit_views(sw)

    p1 = views.write_orbit_views(df, "GPM", str(tmp_path))
    assert p1 is not None and p1.exists()
    # hive layout: views/mission=GPM/year=2020/month=06/orbit=004242.parquet
    parts = p1.relative_to(tmp_path).parts
    assert parts[0] == config.VIEWS_ROOT_SUBDIR
    assert parts[1] == "mission=GPM"
    assert parts[2] == "year=2020" and parts[3] == "month=06"
    assert parts[4] == "orbit=004242.parquet"

    # idempotent re-write: same single file, no stray .tmp
    p2 = views.write_orbit_views(df, "GPM", str(tmp_path))
    assert p2 == p1
    assert not list(p1.parent.glob("*.tmp"))
    assert len(list(p1.parent.glob("*.parquet"))) == 1

    # schema round-trips (read the single file directly; reading via the hive
    # tree would re-infer the partition 'mission' as a dictionary column)
    back = pq.ParquetFile(str(p1)).read()
    assert back.schema.equals(views.VIEWS_SCHEMA)
    assert back.num_rows == len(df)


def test_write_with_subus_scan_times(synthetic_swath, tmp_path):
    """Real scan-time means carry ns remainder; the us cast must not raise
    (regression: an unhandled ArrowInvalid silently dropped all views files)."""
    # per-scan times with nanosecond remainder
    t0 = np.datetime64("2019-06-01T00:00:00", "ns")
    times = t0 + np.array([123, 456789], dtype="timedelta64[ns]")
    lat = np.array([[10.02], [10.07]], dtype=np.float32)
    lon = np.array([[100.03], [100.08]], dtype=np.float32)
    rain = np.array([[1.0], [2.0]], dtype=np.float32)
    sw = synthetic_swath(nscan=2, nray=1, lat=lat, lon=lon, near_sfc_rain=rain,
                         time=times, mission="GPM", orbit=7777)
    df = views.grid_orbit_views(sw)
    p = views.write_orbit_views(df, "GPM", str(tmp_path))   # must not raise
    assert p is not None and p.exists()
    back = pq.ParquetFile(str(p)).read()
    assert back.schema.equals(views.VIEWS_SCHEMA)


def test_empty_write_returns_none(tmp_path):
    assert views.write_orbit_views(None, "GPM", str(tmp_path)) is None
    assert views.write_orbit_views(pd.DataFrame(), "GPM", str(tmp_path)) is None
