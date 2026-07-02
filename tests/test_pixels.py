"""Tests for :func:`pf.pixels.build_pixel_rows` (PIXEL_SCHEMA, 13 cols)."""

from __future__ import annotations

import numpy as np

from pf import feature_id
from pf.features import PIXEL_SCHEMA
from pf.pixels import build_pixel_rows

_PIXEL_KEYS = [f.name for f in PIXEL_SCHEMA]


def _stamp(synthetic_swath, member_pixels, *, nscan=10, nray=12,
           local_label=1, mission="GPM", orbit=12345):
    """Build a swath + labeled with a known member set, distinct field values."""
    rain_type = np.full((nscan, nray), -1, dtype=np.int8)
    pixel_area = np.full((nscan, nray), 20.0, dtype=np.float32)
    dbz = np.full((nscan, nray), np.nan, dtype=np.float32)
    rain = np.full((nscan, nray), np.nan, dtype=np.float32)
    bb = np.full((nscan, nray), np.nan, dtype=np.float32)

    # distinct, hand-checkable values per member pixel
    for i, (s, r) in enumerate(member_pixels):
        rain_type[s, r] = (i % 3) + 1  # cycles 1,2,3
        dbz[s, r] = 20.0 + i
        rain[s, r] = 1.0 + i
        bb[s, r] = 4000.0 + 100.0 * i
        pixel_area[s, r] = 15.0 + i

    sw = synthetic_swath(
        nscan=nscan, nray=nray, mission=mission, orbit=orbit,
        rain_type=rain_type, pixel_area=pixel_area,
        near_sfc_dbz=dbz, near_sfc_rain=rain,
    )
    sw.bb_height = bb

    labeled = np.zeros((nscan, nray), dtype=np.int32)
    for s, r in member_pixels:
        labeled[s, r] = local_label
    return sw, labeled


def test_npixels_and_keys(synthetic_swath):
    member = [(4, 4), (4, 5), (5, 4), (6, 7)]
    sw, labeled = _stamp(synthetic_swath, member)
    rows = build_pixel_rows(sw, labeled, 1, "GPM", 12345)

    assert len(rows) == len(member)
    for row in rows:
        # exactly the 13 keys, in PIXEL_SCHEMA order
        assert list(row.keys()) == _PIXEL_KEYS


def test_feature_id_constant_and_correct(synthetic_swath):
    member = [(4, 4), (5, 5), (6, 6)]
    sw, labeled = _stamp(synthetic_swath, member, mission="GPM", orbit=777)
    rows = build_pixel_rows(sw, labeled, 1, "GPM", 777)
    expected = feature_id.encode("GPM", 777, 1)
    for row in rows:
        assert row["feature_id"] == expected
    # single distinct value
    assert len({row["feature_id"] for row in rows}) == 1


def test_indices_and_values_match_swath(synthetic_swath):
    member = [(4, 4), (4, 5), (5, 4), (6, 7), (7, 2)]
    sw, labeled = _stamp(synthetic_swath, member)
    rows = build_pixel_rows(sw, labeled, 1, "GPM", 12345)

    # row-major order from np.nonzero(member)
    exp_scan, exp_ray = np.nonzero(labeled == 1)
    got_scan = [row["scan"] for row in rows]
    got_ray = [row["ray"] for row in rows]
    assert got_scan == exp_scan.tolist()
    assert got_ray == exp_ray.tolist()

    for row, s, r in zip(rows, exp_scan, exp_ray):
        assert np.isclose(row["lat"], sw.lat[s, r])
        assert np.isclose(row["lon"], sw.lon[s, r])
        assert np.isclose(row["near_sfc_dbz"], sw.near_sfc_dbz[s, r])
        assert np.isclose(row["near_sfc_rain"], sw.near_sfc_rain[s, r])
        assert row["rain_type"] == int(sw.rain_type[s, r])
        assert np.isclose(row["pixel_area_km2"], sw.pixel_area[s, r])
        assert np.isclose(row["bb_height"], sw.bb_height[s, r])


def test_pct_85_89_is_nan(synthetic_swath):
    member = [(4, 4), (5, 5)]
    sw, labeled = _stamp(synthetic_swath, member)
    rows = build_pixel_rows(sw, labeled, 1, "GPM", 12345)
    for row in rows:
        assert np.isnan(row["pct_85_89"])


def test_empty_member_returns_empty_list(synthetic_swath):
    member = [(4, 4), (5, 5)]
    sw, labeled = _stamp(synthetic_swath, member, local_label=1)
    # label 99 is not present
    rows = build_pixel_rows(sw, labeled, 99, "GPM", 12345)
    assert rows == []


def test_scan_ray_dtype_domain(synthetic_swath):
    member = [(4, 4), (5, 5), (6, 6)]
    sw, labeled = _stamp(synthetic_swath, member)
    rows = build_pixel_rows(sw, labeled, 1, "GPM", 12345)
    for row in rows:
        assert isinstance(row["scan"], int)
        assert isinstance(row["ray"], int)
        assert -(2**15) <= row["ray"] < 2**15      # int16 range
        assert -(2**31) <= row["scan"] < 2**31     # int32 range
        assert row["rain_type"] in (-1, 1, 2, 3)
        assert isinstance(row["mission"], str)
