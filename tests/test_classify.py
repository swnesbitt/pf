"""Tests for :func:`pf.classify.classify_feature` (FEATURE cols 36-47).

All synthetic and offline. A known member region is stamped into a ``labeled``
array, and ``rain_type`` / ``near_sfc_rain`` / ``pixel_area`` / ``bb_height`` /
``freezing_level`` are set on the swath so every output can be hand-checked.
"""

from __future__ import annotations

import numpy as np

from pf import classify
from pf.config import MCS_AREA_KM2


def _stamp(synthetic_swath, *, nscan=10, nray=12,
           conv_pixels=(), strat_pixels=(), other_pixels=(),
           pixel_area_val=20.0, rain_val=2.0,
           bb_val=4800.0, fz_val=4900.0):
    """Build a swath + labeled with a single feature (label 1) of known pixels.

    Returns ``(swath, labeled, local_label, member_pixels)`` where the member is
    the union of conv/strat/other pixels. rain_type: conv->2, strat->1,
    other->3. ``bb_height``/``freezing_level`` set to scalars over the member.
    """
    conv_pixels = list(conv_pixels)
    strat_pixels = list(strat_pixels)
    other_pixels = list(other_pixels)
    member_pixels = conv_pixels + strat_pixels + other_pixels

    rain_type = np.full((nscan, nray), -1, dtype=np.int8)
    for s, r in conv_pixels:
        rain_type[s, r] = 2
    for s, r in strat_pixels:
        rain_type[s, r] = 1
    for s, r in other_pixels:
        rain_type[s, r] = 3

    pixel_area = np.full((nscan, nray), pixel_area_val, dtype=np.float32)
    rain = np.full((nscan, nray), np.nan, dtype=np.float32)
    for s, r in member_pixels:
        rain[s, r] = rain_val

    sw = synthetic_swath(
        nscan=nscan, nray=nray,
        rain_type=rain_type, pixel_area=pixel_area, near_sfc_rain=rain,
    )

    # bb_height / freezing_level default to NaN from Swath.empty; set on member.
    sw.bb_height = np.full((nscan, nray), np.nan, dtype=np.float32)
    sw.freezing_level = np.full((nscan, nray), np.nan, dtype=np.float32)
    for s, r in member_pixels:
        sw.bb_height[s, r] = bb_val
        sw.freezing_level[s, r] = fz_val

    labeled = np.zeros((nscan, nray), dtype=np.int32)
    for s, r in member_pixels:
        labeled[s, r] = 1
    return sw, labeled, 1, member_pixels


def test_conv_strat_area_split(synthetic_swath):
    conv = [(4, 4), (4, 5)]          # 2 * 20 = 40
    strat = [(5, 4), (5, 5), (6, 4)]  # 3 * 20 = 60
    sw, labeled, ll, member = _stamp(
        synthetic_swath, conv_pixels=conv, strat_pixels=strat,
        pixel_area_val=20.0,
    )
    area_km2 = 20.0 * len(member)  # 100.0
    vol_total = 2.0 * 20.0 * len(member)  # rain=2, area=20 each
    out = classify.classify_feature(sw, labeled, ll, area_km2, vol_total)

    assert np.isclose(out["conv_area_km2"], 40.0)
    assert np.isclose(out["strat_area_km2"], 60.0)
    assert np.isclose(out["conv_area_frac"], 40.0 / area_km2)
    assert np.isclose(out["strat_area_frac"], 60.0 / area_km2)
    # consistent with area_km2
    assert np.isclose(out["conv_area_frac"] + out["strat_area_frac"], 1.0)


def test_area_frac_nan_when_area_zero(synthetic_swath):
    conv = [(4, 4)]
    sw, labeled, ll, member = _stamp(synthetic_swath, conv_pixels=conv)
    out = classify.classify_feature(sw, labeled, ll, 0.0, 100.0)
    assert np.isnan(out["conv_area_frac"])
    assert np.isnan(out["strat_area_frac"])
    # quantities default to 0 / hand-summed, NOT NaN
    assert np.isclose(out["conv_area_km2"], 20.0)
    assert np.isclose(out["strat_area_km2"], 0.0)


def test_volrain_split_and_fracs(synthetic_swath):
    conv = [(4, 4), (4, 5)]           # rain 2 * area 20 = 40 each -> 80
    strat = [(5, 4), (5, 5), (6, 4)]  # -> 120
    sw, labeled, ll, member = _stamp(
        synthetic_swath, conv_pixels=conv, strat_pixels=strat,
        pixel_area_val=20.0, rain_val=2.0,
    )
    vol_conv = 2.0 * 20.0 * len(conv)   # 80
    vol_strat = 2.0 * 20.0 * len(strat)  # 120
    vol_total = vol_conv + vol_strat     # 200
    out = classify.classify_feature(sw, labeled, ll, 100.0, vol_total)

    assert np.isclose(out["volrain_conv"], vol_conv)
    assert np.isclose(out["volrain_strat"], vol_strat)
    assert np.isclose(out["conv_rain_frac"], vol_conv / vol_total)
    assert np.isclose(out["strat_rain_frac"], vol_strat / vol_total)
    assert np.isclose(
        out["conv_rain_frac"] + out["strat_rain_frac"], 1.0
    )


def test_rain_frac_nan_when_volrain_total_zero(synthetic_swath):
    conv = [(4, 4)]
    strat = [(5, 5)]
    sw, labeled, ll, member = _stamp(
        synthetic_swath, conv_pixels=conv, strat_pixels=strat,
    )
    out = classify.classify_feature(sw, labeled, ll, 100.0, 0.0)
    assert np.isnan(out["conv_rain_frac"])
    assert np.isnan(out["strat_rain_frac"])


def test_rain_frac_nan_when_volrain_total_nan(synthetic_swath):
    conv = [(4, 4)]
    sw, labeled, ll, member = _stamp(synthetic_swath, conv_pixels=conv)
    out = classify.classify_feature(sw, labeled, ll, 100.0, float("nan"))
    assert np.isnan(out["conv_rain_frac"])
    assert np.isnan(out["strat_rain_frac"])


def test_mean_bb_and_freezing(synthetic_swath):
    conv = [(4, 4), (4, 5)]
    strat = [(5, 4)]
    sw, labeled, ll, member = _stamp(
        synthetic_swath, conv_pixels=conv, strat_pixels=strat,
        bb_val=4800.0, fz_val=4900.0,
    )
    # set distinct per-pixel values so nanmean is non-trivial
    bb_vals = {(4, 4): 4000.0, (4, 5): 5000.0, (5, 4): 6000.0}
    fz_vals = {(4, 4): 4100.0, (4, 5): 5100.0, (5, 4): 6100.0}
    for (s, r), v in bb_vals.items():
        sw.bb_height[s, r] = v
    for (s, r), v in fz_vals.items():
        sw.freezing_level[s, r] = v

    out = classify.classify_feature(sw, labeled, ll, 100.0, 100.0)
    member_mask = labeled == ll
    assert np.isclose(
        out["mean_bb_height"], np.nanmean(sw.bb_height[member_mask])
    )
    assert np.isclose(
        out["mean_freezing_level"],
        np.nanmean(sw.freezing_level[member_mask]),
    )
    assert np.isclose(out["mean_bb_height"], np.mean([4000.0, 5000.0, 6000.0]))


def test_mean_bb_all_nan_member_is_nan(synthetic_swath):
    conv = [(4, 4), (5, 5)]
    sw, labeled, ll, member = _stamp(synthetic_swath, conv_pixels=conv)
    member_mask = labeled == ll
    sw.bb_height[member_mask] = np.nan
    sw.freezing_level[member_mask] = np.nan
    out = classify.classify_feature(sw, labeled, ll, 100.0, 100.0)
    assert np.isnan(out["mean_bb_height"])
    assert np.isnan(out["mean_freezing_level"])


def test_is_mcs_true_at_threshold(synthetic_swath):
    conv = [(4, 4)]
    sw, labeled, ll, member = _stamp(synthetic_swath, conv_pixels=conv)
    out = classify.classify_feature(sw, labeled, ll, MCS_AREA_KM2, 100.0)
    assert out["is_mcs"] is True
    out_below = classify.classify_feature(
        sw, labeled, ll, MCS_AREA_KM2 - 1.0, 100.0
    )
    assert out_below["is_mcs"] is False


def test_feature_class_mcs(synthetic_swath):
    # area >= 2000 -> MCS, even with conv/strat present
    conv = [(4, 4)]
    strat = [(5, 5)]
    sw, labeled, ll, member = _stamp(
        synthetic_swath, conv_pixels=conv, strat_pixels=strat,
    )
    out = classify.classify_feature(sw, labeled, ll, MCS_AREA_KM2, 100.0)
    assert out["feature_class"] == "MCS"
    assert out["is_mcs"] is True


def test_feature_class_sub_mcs_conv(synthetic_swath):
    # below MCS area, has conv -> sub_MCS_conv
    conv = [(4, 4)]
    strat = [(5, 5)]
    sw, labeled, ll, member = _stamp(
        synthetic_swath, conv_pixels=conv, strat_pixels=strat,
    )
    out = classify.classify_feature(sw, labeled, ll, 100.0, 100.0)
    assert out["feature_class"] == "sub_MCS_conv"
    assert out["is_mcs"] is False


def test_feature_class_stratiform_only(synthetic_swath):
    # below MCS area, no conv, has strat -> stratiform_only
    strat = [(5, 5), (5, 6)]
    sw, labeled, ll, member = _stamp(synthetic_swath, strat_pixels=strat)
    out = classify.classify_feature(sw, labeled, ll, 100.0, 100.0)
    assert out["feature_class"] == "stratiform_only"
    assert np.isclose(out["conv_area_km2"], 0.0)


def test_feature_class_weak(synthetic_swath):
    # below MCS area, no conv, no strat (only 'other' rain_type==3) -> weak
    other = [(5, 5), (5, 6)]
    sw, labeled, ll, member = _stamp(synthetic_swath, other_pixels=other)
    out = classify.classify_feature(sw, labeled, ll, 100.0, 100.0)
    assert out["feature_class"] == "weak"
    assert np.isclose(out["conv_area_km2"], 0.0)
    assert np.isclose(out["strat_area_km2"], 0.0)


def test_returns_native_scalars(synthetic_swath):
    conv = [(4, 4)]
    strat = [(5, 5)]
    sw, labeled, ll, member = _stamp(
        synthetic_swath, conv_pixels=conv, strat_pixels=strat,
    )
    out = classify.classify_feature(sw, labeled, ll, 100.0, 100.0)
    expected_keys = {
        "conv_area_km2", "strat_area_km2", "conv_area_frac", "strat_area_frac",
        "conv_rain_frac", "strat_rain_frac", "volrain_conv", "volrain_strat",
        "mean_bb_height", "mean_freezing_level", "is_mcs", "feature_class",
    }
    assert set(out.keys()) == expected_keys
    for k in expected_keys - {"is_mcs", "feature_class"}:
        assert isinstance(out[k], float), f"{k} not a native float: {type(out[k])}"
    assert isinstance(out["is_mcs"], bool)
    assert isinstance(out["feature_class"], str)
