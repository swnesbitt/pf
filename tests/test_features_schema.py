"""Schema-shape and ``build_feature_row`` correctness tests."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from pf.config import GPM_N_RANGE_BINS, GPM_RANGE_BIN_SIZE_M
from pf.features import FEATURE_SCHEMA, PIXEL_SCHEMA, build_feature_row
from pf.label import label_rpf, touches_edge

# Expected 48-column order + dtype straight from the spec (section 9).
_EXPECTED_FIELDS = [
    ("feature_id", pa.int64()),
    ("mission", pa.string()),
    ("orbit", pa.int32()),
    ("local_label", pa.int32()),
    ("time", pa.timestamp("us")),
    ("npixels", pa.int32()),
    ("area_km2", pa.float32()),
    ("centroid_lat", pa.float32()),
    ("centroid_lon", pa.float32()),
    ("bbox_scan_min", pa.int32()),
    ("bbox_scan_max", pa.int32()),
    ("bbox_ray_min", pa.int32()),
    ("bbox_ray_max", pa.int32()),
    ("bbox_lat_min", pa.float32()),
    ("bbox_lat_max", pa.float32()),
    ("bbox_lon_min", pa.float32()),
    ("bbox_lon_max", pa.float32()),
    ("frac_land", pa.float32()),
    ("frac_ocean", pa.float32()),
    ("frac_coast", pa.float32()),
    ("surface_flag", pa.int8()),
    ("max_near_sfc_dbz", pa.float32()),
    ("max_near_sfc_rain", pa.float32()),
    ("mean_near_sfc_rain", pa.float32()),
    ("max_ht_20dbz", pa.float32()),
    ("max_ht_30dbz", pa.float32()),
    ("max_ht_40dbz", pa.float32()),
    ("max_ht_20dbz_scan", pa.int32()),
    ("max_ht_20dbz_ray", pa.int32()),
    ("max_ht_30dbz_scan", pa.int32()),
    ("max_ht_30dbz_ray", pa.int32()),
    ("max_ht_40dbz_scan", pa.int32()),
    ("max_ht_40dbz_ray", pa.int32()),
    ("echotop_qc_flags", pa.int16()),
    ("max_ht_20dbz_censored", pa.bool_()),
    ("ray_obs_ceiling_m", pa.float32()),
    ("volrain_total", pa.float32()),
    ("major_axis_km", pa.float32()),
    ("minor_axis_km", pa.float32()),
    ("orientation_deg", pa.float32()),
    ("aspect_ratio", pa.float32()),
    ("eccentricity", pa.float32()),
    ("is_thin", pa.bool_()),
    ("edge", pa.bool_()),
    ("min_pct_85_89", pa.float32()),
    ("min_pct_37", pa.float32()),
    ("conv_area_km2", pa.float32()),
    ("strat_area_km2", pa.float32()),
    ("conv_area_frac", pa.float32()),
    ("strat_area_frac", pa.float32()),
    ("conv_rain_frac", pa.float32()),
    ("strat_rain_frac", pa.float32()),
    ("volrain_conv", pa.float32()),
    ("volrain_strat", pa.float32()),
    ("mean_bb_height", pa.float32()),
    ("mean_freezing_level", pa.float32()),
    ("is_mcs", pa.bool_()),
    ("feature_class", pa.string()),
]

# After Phase 2 only col 35 stays a NaN placeholder (Phase 3); cols 36-47 are
# populated by classify.
_PLACEHOLDER_FLOATS = [
    "min_pct_85_89",
    "min_pct_37",
]

# Phase-2-populated numeric columns: must be present and finite-or-NaN (never
# None) — NaN is allowed by the per-field division/empty-mask rules.
_PHASE2_NUMERIC = [
    "conv_area_km2", "strat_area_km2", "conv_area_frac", "strat_area_frac",
    "conv_rain_frac", "strat_rain_frac", "volrain_conv", "volrain_strat",
    "mean_bb_height", "mean_freezing_level",
]

# Populated (non-placeholder) prefix: identity..edge, including is_thin and the
# 6 max_ht_*_scan/ray provenance cols (which pushed edge to index 40).
_names = [name for name, _ in _EXPECTED_FIELDS]
_COLS_1_35 = _names[: _names.index("edge") + 1]
assert _COLS_1_35[-1] == "edge"
assert "is_thin" in _COLS_1_35
assert "max_ht_40dbz_ray" in _COLS_1_35


def test_feature_schema_exact_48_order_and_dtype():
    assert len(FEATURE_SCHEMA) == 58   # +3 echo-top QC cols (flags/censored/ceiling)
    actual = [(f.name, f.type) for f in FEATURE_SCHEMA]
    assert actual == _EXPECTED_FIELDS
    # is_thin sits between eccentricity and edge, with a bool dtype.
    names = [f.name for f in FEATURE_SCHEMA]
    assert names.index("is_thin") == names.index("eccentricity") + 1
    assert names.index("is_thin") + 1 == names.index("edge")
    assert names.index("is_thin") == 42   # +6 ray-of-max cols, +3 echo-top QC cols
    assert FEATURE_SCHEMA.field("is_thin").type == pa.bool_()


def test_pixel_schema_13_fields():
    assert len(PIXEL_SCHEMA) == 14


def _make_labeled_swath(synthetic_swath):
    """Build a swath with one retained feature and return (swath, labeled, kept)."""
    nscan, nray = 10, 12
    dbz = np.full((nscan, nray), np.nan, dtype=np.float32)
    member_pixels = [(4, 4), (4, 5), (5, 4), (5, 5), (6, 5)]
    dbz_vals = [22.0, 24.0, 41.0, 33.0, 28.0]
    for (s, r), v in zip(member_pixels, dbz_vals):
        dbz[s, r] = v
    synthetic_max_dbz = max(dbz_vals)

    pixel_area = np.full((nscan, nray), 20.0, dtype=np.float32)  # 5*20 = 100

    rain = np.full((nscan, nray), np.nan, dtype=np.float32)
    for s, r in member_pixels:
        rain[s, r] = 2.0

    # 3-D cube: put a 40-dBZ echo at a KNOWN high bin only at pixel (5,4).
    nbin = GPM_N_RANGE_BINS
    dbz_3d = np.full((nscan, nray, nbin), np.nan, dtype=np.float32)
    # height convention: height = (nbin-1-b)*125; pick b such that height is known
    hi_bin = nbin - 1 - 40  # height = 40 * 125 = 5000 m
    dbz_3d[5, 4, hi_bin] = 45.0       # >= 40 dBZ at 5000 m
    dbz_3d[4, 4, nbin - 1 - 8] = 22.0  # a low echo, height 1000 m

    sw = synthetic_swath(
        nscan=nscan,
        nray=nray,
        near_sfc_dbz=dbz,
        near_sfc_rain=rain,
        pixel_area=pixel_area,
        dbz_3d=dbz_3d,
        surface_type=np.zeros((nscan, nray), dtype=np.int8),  # ocean
    )
    labeled, kept = label_rpf(sw)
    return sw, labeled, kept, member_pixels, synthetic_max_dbz


def test_build_feature_row_keys_and_order(synthetic_swath):
    sw, labeled, kept, member_pixels, max_dbz = _make_labeled_swath(synthetic_swath)
    assert len(kept) == 1
    local_label, area_km2 = kept[0]
    edge = touches_edge(labeled, local_label)
    row = build_feature_row(sw, labeled, local_label, area_km2, edge)

    # keys == schema field names, in order
    assert list(row.keys()) == [f.name for f in FEATURE_SCHEMA]


def test_build_feature_row_values(synthetic_swath):
    sw, labeled, kept, member_pixels, max_dbz = _make_labeled_swath(synthetic_swath)
    local_label, area_km2 = kept[0]
    edge = touches_edge(labeled, local_label)
    row = build_feature_row(sw, labeled, local_label, area_km2, edge)

    # npixels == member count
    assert row["npixels"] == len(member_pixels)
    # area matches the area-weighted sum
    assert np.isclose(row["area_km2"], 100.0)
    # max near-surface dbz matches the synthetic max
    assert np.isclose(row["max_near_sfc_dbz"], max_dbz)
    # max_ht_40dbz uses height_3d: 40-dBZ echo placed at 5000 m
    assert np.isclose(row["max_ht_40dbz"], 5000.0)
    # 20-dBZ top should be at least as high (also 5000 m here)
    assert row["max_ht_20dbz"] >= row["max_ht_40dbz"] - 1e-6
    # volrain_total = sum(rain * area) = 5 * (2.0 * 20.0) = 200
    assert np.isclose(row["volrain_total"], 200.0)
    # surface flag ocean
    assert row["surface_flag"] == 0
    assert np.isclose(row["frac_ocean"], 1.0)


def test_build_feature_row_cols_1_35_finite(synthetic_swath):
    sw, labeled, kept, member_pixels, max_dbz = _make_labeled_swath(synthetic_swath)
    local_label, area_km2 = kept[0]
    edge = touches_edge(labeled, local_label)
    row = build_feature_row(sw, labeled, local_label, area_km2, edge)

    # columns 1-35 (identity..edge, incl. is_thin) populated (not None);
    # numeric ones finite.
    for name in _COLS_1_35:
        val = row[name]
        assert val is not None, f"{name} is None but should be populated"
        if name in ("mission",):
            assert isinstance(val, str)
        elif name == "time":
            assert not np.isnat(np.datetime64(val))
        elif name in ("edge", "is_thin"):
            assert isinstance(val, (bool, np.bool_))
        elif isinstance(val, (int, float, np.integer, np.floating)):
            assert np.isfinite(val), f"{name} not finite: {val}"


def test_build_feature_row_placeholders_null(synthetic_swath):
    sw, labeled, kept, member_pixels, max_dbz = _make_labeled_swath(synthetic_swath)
    local_label, area_km2 = kept[0]
    edge = touches_edge(labeled, local_label)
    row = build_feature_row(sw, labeled, local_label, area_km2, edge)

    # col 35 is the only remaining NaN placeholder (Phase 3).
    for name in _PLACEHOLDER_FLOATS:
        assert np.isnan(row[name]), f"{name} should be NaN placeholder"

    # cols 36-45 are now populated by classify: present, not None, and either
    # finite or NaN-by-rule (never None).
    for name in _PHASE2_NUMERIC:
        val = row[name]
        assert val is not None, f"{name} should be populated (not None)"
        assert isinstance(val, (int, float, np.integer, np.floating)), (
            f"{name} should be a numeric scalar, got {type(val)}"
        )
        # finite or NaN-by-rule, but never +/-inf
        assert np.isfinite(val) or np.isnan(val), f"{name} must be finite or NaN"

    # is_mcs is now a real bool; feature_class a real str (no longer None).
    assert row["is_mcs"] is not None
    assert isinstance(row["is_mcs"], (bool, np.bool_)), type(row["is_mcs"])
    assert row["feature_class"] is not None
    assert isinstance(row["feature_class"], str)
    assert row["feature_class"] in {
        "MCS", "sub_MCS_conv", "stratiform_only", "weak"
    }
