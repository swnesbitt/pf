"""Tests for pf.grid_swath — swath-gridded, hour-resolved rain/views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pf import echotop_qc, geometry, grid, grid_swath
from pf.config import GPM_N_RANGE_BINS, GPM_RANGE_BIN_SIZE_M


def test_empty_returns_none(synthetic_swath):
    sw = synthetic_swath(nscan=4, nray=6,
                         near_sfc_rain=np.full((4, 6), np.nan, np.float32))
    views, rain = grid_swath.grid_swath(sw, np.zeros((4, 6), np.int32), [])
    assert views is None and rain is None


def test_hour_binning_and_wrap(synthetic_swath):
    # Two scans straddling midnight UTC: 23:30 and 00:30 -> hours 23 and 0.
    nscan, nray = 2, 5
    t = np.array(["2020-06-01T23:30:00", "2020-06-02T00:30:00"], dtype="datetime64[ns]")
    sw = synthetic_swath(nscan=nscan, nray=nray, time=t,
                         near_sfc_rain=np.ones((nscan, nray), np.float32))
    views, rain = grid_swath.grid_swath(sw, np.zeros((nscan, nray), np.int32), [])
    assert rain is None  # default rain_type 0 -> dropped
    assert set(views["hour"].tolist()) == {0, 23}
    # every valid pixel counted exactly once
    assert int(views["n_views"].sum()) == nscan * nray


def test_time_window_masks_scans_per_month(synthetic_swath):
    # An orbit straddling a MONTH boundary: scan 0 at 2020-05-31T23:30, scan 1 at
    # 2020-06-01T00:30. The May window must keep ONLY scan 0, the June window ONLY
    # scan 1 — each scan to exactly one month (per-scan month binning, no
    # double-count, no loss), mirroring the hour axis.
    nscan, nray = 2, 5
    t = np.array(["2020-05-31T23:30:00", "2020-06-01T00:30:00"], dtype="datetime64[ns]")
    sw = synthetic_swath(nscan=nscan, nray=nray, time=t,
                         near_sfc_rain=np.ones((nscan, nray), np.float32))
    labeled = np.zeros((nscan, nray), np.int32)
    may = (pd.Timestamp("2020-05-01"), pd.Timestamp("2020-06-01"))
    jun = (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-07-01"))
    v_may, _ = grid_swath.grid_swath(sw, labeled, [], time_window=may)
    v_jun, _ = grid_swath.grid_swath(sw, labeled, [], time_window=jun)
    assert set(v_may["hour"].tolist()) == {23}
    assert set(v_jun["hour"].tolist()) == {0}
    # union over the two adjacent months == every pixel exactly once
    assert int(v_may["n_views"].sum()) == nray
    assert int(v_jun["n_views"].sum()) == nray
    assert int(v_may["n_views"].sum()) + int(v_jun["n_views"].sum()) == nscan * nray
    # a window touching neither scan grids nothing
    jul = (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-08-01"))
    assert grid_swath.grid_swath(sw, labeled, [], time_window=jul) == (None, None)


def test_granule_time_span_midnight_wrap():
    from pf import search

    class _R:                          # minimal reader exposing _filename_of
        def _filename_of(self, g):
            return g

    wrap = "2A.TRMM.PR.V9-20220125.19971230-S225350-E002507.061218.V07A.HDF5"
    s, e = search.granule_time_span(wrap, _R())
    assert (s.year, s.month, s.day, s.hour, s.minute, s.second) == (1997, 12, 30, 22, 53, 50)
    assert (e.year, e.month, e.day, e.hour, e.minute, e.second) == (1997, 12, 31, 0, 25, 7)
    assert e > s                       # rolled to next day across UTC midnight

    same = "2A.GPM.Ku.V9-20220125.20180615-S010000-E013000.012345.V07A.HDF5"
    s2, e2 = search.granule_time_span(same, _R())
    assert s2.day == e2.day == 15 and e2.hour == 1 and e2.minute == 30


def test_undefined_slot_for_nonfeature(synthetic_swath):
    # A convective raining pixel that belongs to NO feature -> undefined slots.
    nscan, nray = 3, 4
    rt = np.zeros((nscan, nray), np.int8)
    rt[1, 2] = 2  # convective
    sw = synthetic_swath(nscan=nscan, nray=nray, rain_type=rt,
                         near_sfc_rain=np.ones((nscan, nray), np.float32))
    labeled = np.zeros((nscan, nray), np.int32)  # nothing labeled
    views, rain = grid_swath.grid_swath(sw, labeled, [])
    assert rain is not None and len(rain) == 1
    row = rain.iloc[0]
    assert int(row.size_class) == grid.SIZE_UNDEF
    assert int(row.echotop_class) == grid.ECHOTOP_UNDEF
    assert int(row.raintype) == 1  # rain_type 2 -> convective index 1
    assert row.rain_sum == pytest.approx(1.0)
    assert int(row.raining_count) == 1


def test_raintype_passthrough(synthetic_swath):
    nscan, nray = 1, 5
    rt = np.array([[1, 2, 3, 0, -1]], np.int8)
    sw = synthetic_swath(nscan=nscan, nray=nray, rain_type=rt,
                         near_sfc_rain=np.ones((nscan, nray), np.float32))
    views, rain = grid_swath.grid_swath(sw, np.zeros((nscan, nray), np.int32), [])
    # 0 and -1 dropped from rain; 1/2/3 -> 0/1/2
    assert sorted(rain["raintype"].tolist()) == [0, 1, 2]
    # all 5 pixels are still views
    assert int(views["n_views"].sum()) == 5


def test_rain_sum_groupsum(synthetic_swath):
    # Two raining conv pixels forced into the SAME cell -> one summed row.
    nscan, nray = 1, 2
    lat = np.array([[0.011, 0.012]], np.float32)   # both in lat cell floor(0.01/0.05)
    lon = np.array([[0.013, 0.014]], np.float32)   # both in same lon cell
    sw = synthetic_swath(nscan=nscan, nray=nray, lat=lat, lon=lon,
                         rain_type=np.full((1, 2), 2, np.int8),
                         near_sfc_rain=np.array([[2.0, 5.0]], np.float32))
    views, rain = grid_swath.grid_swath(sw, np.zeros((1, 2), np.int32), [])
    assert len(rain) == 1
    assert rain.iloc[0].rain_sum == pytest.approx(7.0)
    assert int(rain.iloc[0].raining_count) == 2
    assert int(views.iloc[0].n_views) == 2


def test_conservation_raining_le_views(synthetic_swath):
    # Mixed rain types so raining pixels are a strict subset of views.
    nscan, nray = 6, 8
    rng = np.arange(nscan * nray).reshape(nscan, nray)
    rt = np.where(rng % 2 == 0, 2, 0).astype(np.int8)  # half convective, half none
    sw = synthetic_swath(nscan=nscan, nray=nray, rain_type=rt,
                         near_sfc_rain=np.ones((nscan, nray), np.float32))
    views, rain = grid_swath.grid_swath(sw, np.zeros((nscan, nray), np.int32), [])
    agg = (rain.groupby(["lat_bin", "lon_bin", "hour"], as_index=False)
               ["raining_count"].sum())
    merged = agg.merge(views, on=["lat_bin", "lon_bin", "hour"], how="left")
    assert (merged["raining_count"] <= merged["n_views"]).all()


def test_category_parity_with_direct_calls(synthetic_swath):
    # A labeled feature: assert build_class_maps == direct geometry/echotop calls.
    nscan, nray = 6, 6
    nbin = GPM_N_RANGE_BINS
    # 20-dBZ echo at a known height (5000 m) inside the feature block.
    dbz_3d = np.full((nscan, nray, nbin), np.nan, np.float32)
    hi_bin = nbin - 1 - int(round(5000.0 / GPM_RANGE_BIN_SIZE_M))  # height ~5000 m
    member_block = (slice(2, 5), slice(2, 5))
    dbz_3d[member_block[0], member_block[1], hi_bin] = 25.0
    sw = synthetic_swath(nscan=nscan, nray=nray, dbz_3d=dbz_3d,
                         rain_type=np.full((nscan, nray), 2, np.int8),
                         near_sfc_rain=np.ones((nscan, nray), np.float32))
    labeled = np.zeros((nscan, nray), np.int32)
    labeled[member_block] = 1
    kept = [(1, float(np.count_nonzero(labeled == 1)) * 20.0)]

    size_map, echo_map = grid_swath.build_class_maps(sw, labeled, kept)
    member = labeled == 1
    major = geometry.pca_axes(sw.lat[member], sw.lon[member])[0]
    ht20 = echotop_qc.feature_echo_tops(sw, member)["max_ht_20dbz"]
    exp_size = int(grid.size_class(np.array([major]))[0])
    exp_echo = int(grid.echotop_class(np.array([ht20]))[0])

    assert set(np.unique(size_map[member]).tolist()) == {exp_size}
    assert set(np.unique(echo_map[member]).tolist()) == {exp_echo}
    # non-member pixels stay undefined
    assert (size_map[~member] == grid.SIZE_UNDEF).all()
    assert (echo_map[~member] == grid.ECHOTOP_UNDEF).all()
    # and those classes surface in rain_df for the member cells
    _, rain = grid_swath.grid_swath(sw, labeled, kept)
    feat_rows = rain[(rain.size_class == exp_size) & (rain.echotop_class == exp_echo)]
    assert len(feat_rows) >= 1
