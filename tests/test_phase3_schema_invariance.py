"""Phase-3 must NOT touch the frozen Parquet schemas.

``storm_top`` and ``pia`` are in-memory ``Swath`` gating fields only; they are
NOT new Parquet columns. ``FEATURE_SCHEMA`` is 48 fields (the derived
``is_thin`` shape flag was inserted after ``eccentricity``) and
``PIXEL_SCHEMA`` stays at 13 fields, with the same names/order/types, and the
PCT placeholders (feature col 36 / pixel col 10) keep their position.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa

from pf.features import FEATURE_SCHEMA, PIXEL_SCHEMA
from pf.swath import Swath

# Frozen field names/types captured from the Phase-1/2 contract (authoritative).
_EXPECTED_FEATURE_FIELDS = [
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
    ("is_thin", pa.bool_()),           # col index 33 (0-based)
    ("edge", pa.bool_()),
    ("min_pct_85_89", pa.float32()),   # col index 35 (0-based)
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

_EXPECTED_PIXEL_FIELDS = [
    ("feature_id", pa.int64()),
    ("mission", pa.string()),
    ("orbit", pa.int32()),
    ("scan", pa.int32()),
    ("ray", pa.int16()),
    ("lat", pa.float32()),
    ("lon", pa.float32()),
    ("near_sfc_dbz", pa.float32()),
    ("near_sfc_rain", pa.float32()),
    ("pct_85_89", pa.float32()),       # col index 9 (0-based)
    ("pct_37", pa.float32()),
    ("rain_type", pa.int8()),
    ("pixel_area_km2", pa.float32()),
    ("bb_height", pa.float32()),
]


def test_feature_schema_48_fields_unchanged():
    assert len(FEATURE_SCHEMA) == 58   # +3 echo-top QC cols
    got = [(f.name, f.type) for f in FEATURE_SCHEMA]
    assert got == _EXPECTED_FEATURE_FIELDS
    # is_thin is the only Phase-3 addition: a bool flag wedged between the
    # eccentricity and edge columns.
    assert FEATURE_SCHEMA.field("is_thin").type == pa.bool_()


def test_pixel_schema_13_fields_unchanged():
    assert len(PIXEL_SCHEMA) == 14
    got = [(f.name, f.type) for f in PIXEL_SCHEMA]
    assert got == _EXPECTED_PIXEL_FIELDS


def test_min_pct_85_89_position():
    names = [f.name for f in FEATURE_SCHEMA]
    assert "min_pct_85_89" in names
    # is_thin inserted after eccentricity shifts everything after it by +1,
    # so min_pct_85_89 moves from 0-based index 34 -> 35 (= col 36). Derive
    # the expected index relative to edge to keep the check robust.
    assert names.index("min_pct_85_89") == names.index("edge") + 1
    assert names.index("min_pct_85_89") == 44   # +6 ray-of-max cols, +3 echo-top QC cols


def test_pct_85_89_in_pixel_schema():
    names = [f.name for f in PIXEL_SCHEMA]
    assert "pct_85_89" in names
    assert names.index("pct_85_89") == 9   # 0-based col index 9 = col 10


def test_swath_has_storm_top_and_pia_fields():
    field_names = {f.name for f in dataclasses.fields(Swath)}
    assert "storm_top" in field_names
    assert "pia" in field_names


def test_swath_empty_allocates_storm_top_pia_nan_float32():
    nscan, nray = 4, 7
    sw = Swath.empty(
        nscan, nray,
        mission="GPM", orbit=1, short_name="GPM_2ADPR",
        granule_name="x.HDF5",
    )
    for name in ("storm_top", "pia"):
        arr = getattr(sw, name)
        assert arr.shape == (nscan, nray)
        assert arr.dtype == np.float32
        assert np.isnan(arr).all()
