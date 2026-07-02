"""Catalog write / hive-layout / atomic-overwrite tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pf import catalog
from pf.features import FEATURE_SCHEMA


def _make_features_df(n=3, orbit=12345, base_label=1, max_dbz_start=30.0):
    """Build a small DataFrame consistent with FEATURE_SCHEMA."""
    t0 = np.datetime64("2021-07-15T03:00:00")
    rows = []
    for i in range(n):
        row = {f.name: None for f in FEATURE_SCHEMA}
        row.update(
            feature_id=2 * 10_000_000_000_000 + orbit * 100_000 + (base_label + i),
            mission="GPM",
            orbit=orbit,
            local_label=base_label + i,
            time=t0 + np.timedelta64(i, "s"),
            npixels=10 + i,
            area_km2=100.0 + i,
            centroid_lat=10.0 + i,
            centroid_lon=100.0 + i,
            bbox_scan_min=i,
            bbox_scan_max=i + 2,
            bbox_ray_min=i,
            bbox_ray_max=i + 3,
            bbox_lat_min=10.0,
            bbox_lat_max=11.0,
            bbox_lon_min=100.0,
            bbox_lon_max=101.0,
            frac_land=0.0,
            frac_ocean=1.0,
            frac_coast=0.0,
            surface_flag=0,
            max_near_sfc_dbz=max_dbz_start + i,
            max_near_sfc_rain=5.0,
            mean_near_sfc_rain=2.0,
            max_ht_20dbz=5000.0,
            max_ht_30dbz=4000.0,
            max_ht_40dbz=3000.0,
            volrain_total=200.0,
            major_axis_km=10.0,
            minor_axis_km=5.0,
            orientation_deg=12.0,
            aspect_ratio=2.0,
            eccentricity=0.8,
            edge=False,
            # 35-45 NaN placeholders
            min_pct_85_89=np.nan,
            conv_area_km2=np.nan,
            strat_area_km2=np.nan,
            conv_area_frac=np.nan,
            strat_area_frac=np.nan,
            conv_rain_frac=np.nan,
            strat_rain_frac=np.nan,
            volrain_conv=np.nan,
            volrain_strat=np.nan,
            mean_bb_height=np.nan,
            mean_freezing_level=np.nan,
            is_mcs=None,
            feature_class=None,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_write_orbit_hive_path_and_roundtrip(tmp_path):
    df = _make_features_df(n=3, orbit=12345)
    fpath, ppath = catalog.write_orbit(df, None, "GPM", root=str(tmp_path))

    # hive path: features/mission=GPM/year=2021/month=07/orbit=012345.parquet
    expected = (
        tmp_path
        / "features"
        / "mission=GPM"
        / "year=2021"
        / "month=07"
        / "orbit=012345.parquet"
    )
    assert expected.exists()
    assert str(expected) == fpath

    # re-read via dataset with hive partitioning
    dataset = ds.dataset(
        str(tmp_path / "features"), format="parquet", partitioning="hive"
    )
    table = dataset.to_table()
    assert table.num_rows == 3

    # key dtypes preserved
    schema = table.schema
    assert schema.field("feature_id").type == pa.int64()
    assert schema.field("area_km2").type == pa.float32()
    # time microsecond resolution
    tfield = schema.field("time").type
    assert pa.types.is_timestamp(tfield)
    assert tfield.unit == "us"

    # feature_id values survive round trip
    got_ids = set(table.column("feature_id").to_pylist())
    assert got_ids == set(df["feature_id"].tolist())


def test_pixels_path_returned_when_none(tmp_path):
    df = _make_features_df(n=2, orbit=222)
    fpath, ppath = catalog.write_orbit(df, None, "gpm", root=str(tmp_path))
    # pixels path returned but file not written (Phase 1 skips pixels)
    assert "pixels" in ppath
    assert "mission=GPM" in ppath  # upper-cased
    assert not (tmp_path / "pixels").exists() or not list(
        (tmp_path / "pixels").rglob("*.parquet")
    )


def test_atomic_overwrite_idempotent(tmp_path):
    df1 = _make_features_df(n=3, orbit=999, max_dbz_start=30.0)
    catalog.write_orbit(df1, None, "GPM", root=str(tmp_path))

    df2 = _make_features_df(n=3, orbit=999, max_dbz_start=55.0)  # new content
    fpath2, _ = catalog.write_orbit(df2, None, "GPM", root=str(tmp_path))

    part_dir = tmp_path / "features" / "mission=GPM" / "year=2021" / "month=07"
    parquet_files = list(part_dir.glob("*.parquet"))
    tmp_files = list(part_dir.glob("*.tmp"))
    # exactly one file, no stray .tmp
    assert len(parquet_files) == 1
    assert tmp_files == []

    # content is the latest write. Read the single file directly via
    # ParquetFile (the filename contains '=', which would otherwise trip
    # pyarrow's hive-partition inference on the leaf path).
    table = pq.ParquetFile(fpath2).read()
    got = sorted(table.column("max_near_sfc_dbz").to_pylist())
    assert got == [55.0, 56.0, 57.0]


def test_empty_features_raises(tmp_path):
    import pytest

    empty = _make_features_df(n=0)
    with pytest.raises(ValueError):
        catalog.write_orbit(empty, None, "GPM", root=str(tmp_path))
