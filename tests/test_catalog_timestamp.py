"""Regression test for ns->us timestamp truncation in catalog.write_orbit.

DEFECT: ``build_feature_row`` produces ``time`` as ``datetime64[ns]`` with a
sub-microsecond remainder (e.g. the mean of member scan times). The frozen
``FEATURE_SCHEMA`` declares ``time`` as ``timestamp('us')``, and the strict
schema cast in ``pa.Table.from_pandas(schema=FEATURE_SCHEMA)`` raises
``ArrowInvalid`` ("would lose data") for ns values not divisible by 1000.
``write_orbit`` must truncate ns->us up front so the cast is lossless.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from pf import catalog
from pf.features import FEATURE_SCHEMA

# ns timestamp with a sub-microsecond remainder: 1530399785739333376 ns is not
# divisible by 1000 (it ends in ...376), so a strict ns->us cast would lose the
# trailing 376 ns and raise ArrowInvalid unless write_orbit truncates first.
_NS_VALUE = 1530399785739333376


def _make_one_row_df(*, orbit=12345, time):
    """Build a one-row DataFrame conforming to FEATURE_SCHEMA column names."""
    row = {f.name: None for f in FEATURE_SCHEMA}
    row.update(
        feature_id=2 * 10_000_000_000_000 + orbit * 100_000 + 1,
        mission="GPM",
        orbit=orbit,
        local_label=1,
        time=time,
        npixels=10,
        area_km2=100.0,
        centroid_lat=10.0,
        centroid_lon=100.0,
        bbox_scan_min=0,
        bbox_scan_max=2,
        bbox_ray_min=0,
        bbox_ray_max=3,
        bbox_lat_min=10.0,
        bbox_lat_max=11.0,
        bbox_lon_min=100.0,
        bbox_lon_max=101.0,
        frac_land=0.0,
        frac_ocean=1.0,
        frac_coast=0.0,
        surface_flag=0,
        max_near_sfc_dbz=30.0,
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
    df = pd.DataFrame([row])
    # Ensure the time column is genuinely datetime64[ns] with the sub-us value.
    df["time"] = pd.Series([time], dtype="datetime64[ns]")
    return df


def test_write_orbit_truncates_subus_ns_time(tmp_path):
    ts = pd.Timestamp(_NS_VALUE, unit="ns")
    assert ts.value % 1000 != 0  # sub-microsecond remainder present

    df = _make_one_row_df(orbit=12345, time=ts)
    assert df["time"].dtype == np.dtype("datetime64[ns]")

    # Must NOT raise ArrowInvalid.
    fpath, _ = catalog.write_orbit(df, None, "GPM", root=str(tmp_path))

    assert os.path.exists(fpath)

    # Re-read and confirm us-resolution timestamp equal to the ns value
    # truncated to microseconds.
    dataset = ds.dataset(
        str(tmp_path / "features"), format="parquet", partitioning="hive"
    )
    table = dataset.to_table()
    assert table.num_rows == 1

    tfield = table.schema.field("time").type
    import pyarrow as pa

    assert pa.types.is_timestamp(tfield)
    assert tfield.unit == "us"

    got = table.column("time").to_pylist()[0]
    expected_us = ts.value // 1000  # ns -> us truncation
    got_us = pd.Timestamp(got).value // 1000  # got is us-resolution datetime
    assert got_us == expected_us
