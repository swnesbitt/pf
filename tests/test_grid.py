"""Tests for pf.grid — Part B gridded rain-contribution climatology."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pf import config, grid

duckdb = pytest.importorskip("duckdb")
xr = pytest.importorskip("xarray")


# --- pure class assignment ------------------------------------------------
def test_size_class_edges_and_nan():
    x = np.array([5.0, 20.0, 49.9, 50.0, 100.0, 200.0, np.nan])
    assert list(grid.size_class(x)) == [0, 1, 1, 2, 3, 3, grid.SIZE_UNDEF]


def test_echotop_class_edges_and_nan():
    m = np.array([4000.0, 5000.0, 7400.0, 7500.0, 12000.0, 15000.0, np.nan])  # metres
    assert list(grid.echotop_class(m)) == [0, 1, 1, 2, 3, 3, grid.ECHOTOP_UNDEF]


def test_raintype_class():
    assert list(grid.raintype_class([1, 2, 3, 0, -1])) == [0, 1, 2, -1, -1]


def test_latlon_to_bin_and_wrap():
    lat_bin, lon_bin = grid.latlon_to_bin([0.02, 0.0], [0.03, 180.02])
    assert lat_bin[0] == int(np.floor((0.02 + 90) / 0.05))
    assert lon_bin[0] == int(np.floor((0.03 + 180) / 0.05))
    # +180.02 wraps to -179.98
    assert lon_bin[1] == int(np.floor((-179.98 + 180) / 0.05))
    assert 0 <= lon_bin[1] < config.GRID_N_LON


# --- DuckDB accumulate + views + combine ----------------------------------
def _write_parquet(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


@pytest.fixture
def synthetic_db(tmp_path):
    """One GPM month=03 partition: 1 feature, 3 conv raining pixels in one cell,
    plus a matching views table."""
    root = tmp_path
    mm = "month=03"
    yy = "year=2014"
    fdir = root / "features" / "mission=GPM" / yy / mm / "orbit=000001.parquet"
    pdir = root / "pixels" / "mission=GPM" / yy / mm / "orbit=000001.parquet"
    vdir = root / config.VIEWS_ROOT_SUBDIR / "mission=GPM" / yy / mm / "orbit=000001.parquet"

    # feature: major_axis 30 km -> size class 1; echo-top 6000 m -> echotop class 1
    _write_parquet(fdir, pd.DataFrame({
        "feature_id": np.int64([100]),
        "major_axis_km": np.float32([30.0]),
        "max_ht_20dbz": np.float32([6000.0]),
    }))
    # 3 convective (rain_type 2 -> class 1) raining pixels at lat 0.02 lon 0.03
    _write_parquet(pdir, pd.DataFrame({
        "feature_id": np.int64([100, 100, 100]),
        "lat": np.float32([0.02, 0.02, 0.02]),
        "lon": np.float32([0.03, 0.03, 0.03]),
        "near_sfc_rain": np.float32([2.0, 4.0, 6.0]),
        "rain_type": np.int16([2, 2, 2]),
    }))
    # views: 10 observations of the same cell
    lat_bin = int(np.floor((0.02 + 90) / 0.05))
    lon_bin = int(np.floor((0.03 + 180) / 0.05))
    _write_parquet(vdir, pd.DataFrame({
        "lat_bin": np.int16([lat_bin]),
        "lon_bin": np.int16([lon_bin]),
        "n_views": np.int32([10]),
    }))
    return root, lat_bin, lon_bin


def test_accumulate_month(synthetic_db):
    root, lat_bin, lon_bin = synthetic_db
    con = duckdb.connect()
    df = grid.accumulate_month(con, "GPM", 3, str(root))
    assert len(df) == 1
    row = df.iloc[0]
    assert int(row.lat_bin) == lat_bin and int(row.lon_bin) == lon_bin
    assert int(row.size_class) == 1 and int(row.echotop_class) == 1 and int(row.raintype) == 1
    assert row.rain_sum == pytest.approx(12.0)  # 2+4+6
    assert int(row.raining_count) == 3


def test_views_reduce_and_combine(synthetic_db):
    root, lat_bin, lon_bin = synthetic_db
    con = duckdb.connect()
    rain = grid.accumulate_month(con, "GPM", 3, str(root))
    views = grid.reduce_views_month(con, "GPM", 3, str(root))
    assert int(views.iloc[0].n_views) == 10

    ds = grid.build_dataset({3: rain}, {3: views}, "GPM", mode="marginal")
    lo, _ = grid.lat_clip_bins("GPM")
    li, mi = lat_bin - lo, 2  # month index for March
    # rain_sum_by_raintype at convective(1): 12 over 10 views -> rate 1.2 mm/hr
    rs = ds["rain_sum_by_raintype"].values[mi, 1, li, lon_bin]
    v = ds["views"].values[mi, li, lon_bin]
    assert rs == pytest.approx(12.0)
    assert int(v) == 10
    assert rs / v == pytest.approx(1.2)
    # raining_count (3) <= views (10)
    assert ds["raining_count_by_raintype"].values[mi, 1, li, lon_bin] <= v


def test_build_dataset_cf_and_roundtrip(synthetic_db, tmp_path):
    root, lat_bin, lon_bin = synthetic_db
    con = duckdb.connect()
    rain = grid.accumulate_month(con, "GPM", 3, str(root))
    views = grid.reduce_views_month(con, "GPM", 3, str(root))
    ds = grid.build_dataset({3: rain}, {3: views}, "GPM", mode="marginal")
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert list(ds["size_label"].values[:4]) == ["<20", "20-50", "50-100", ">100"]
    out = grid.write_netcdf(ds, str(tmp_path / "clim.nc"))
    assert out.exists()
    back = xr.open_dataset(out)
    assert back["views"].values.sum() == 10
    back.close()


def test_crossproduct_mode(synthetic_db):
    root, lat_bin, lon_bin = synthetic_db
    con = duckdb.connect()
    rain = grid.accumulate_month(con, "GPM", 3, str(root))
    views = grid.reduce_views_month(con, "GPM", 3, str(root))
    ds = grid.build_dataset({3: rain}, {3: views}, "GPM", mode="crossproduct")
    lo, _ = grid.lat_clip_bins("GPM")
    val = ds["rain_sum"].values[2, 1, 1, 1, lat_bin - lo, lon_bin]  # size1,echotop1,conv1
    assert val == pytest.approx(12.0)


def test_missing_partition_is_empty():
    con = duckdb.connect()
    df = grid.accumulate_month(con, "GPM", 7, "/nonexistent/root")
    assert len(df) == 0
