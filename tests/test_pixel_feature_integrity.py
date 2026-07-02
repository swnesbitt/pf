"""Offline end-to-end feature<->pixel join integrity at the DataFrame level.

Mirrors the real-granule integrity check (plan verification section) but fully
synthetic: build a swath, label it, build the features and pixels DataFrames
exactly as :func:`pf.granule.process_orbit` does, and assert the join invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.catalog import _coerce_time_us
from pf.features import FEATURE_SCHEMA, PIXEL_SCHEMA, build_feature_row
from pf.label import label_rpf, touches_edge
from pf.pixels import build_pixel_rows


def _two_feature_swath(synthetic_swath):
    """Swath with two well-separated retained features (each > MIN_AREA_KM2)."""
    nscan, nray = 16, 16
    dbz = np.full((nscan, nray), np.nan, dtype=np.float32)
    rain_type = np.full((nscan, nray), -1, dtype=np.int8)
    rain = np.full((nscan, nray), np.nan, dtype=np.float32)

    # Feature A: interior, mixed conv/strat, 5 pixels * 20 = 100 km^2
    feat_a = [(3, 3), (3, 4), (4, 3), (4, 4), (5, 3)]
    # Feature B: separated, 6 pixels * 20 = 120 km^2
    feat_b = [(10, 10), (10, 11), (11, 10), (11, 11), (12, 10), (12, 11)]

    for i, (s, r) in enumerate(feat_a):
        dbz[s, r] = 30.0
        rain[s, r] = 2.0
        rain_type[s, r] = 2 if i < 2 else 1   # some conv, some strat
    for i, (s, r) in enumerate(feat_b):
        dbz[s, r] = 35.0
        rain[s, r] = 3.0
        rain_type[s, r] = 1                    # all stratiform

    pixel_area = np.full((nscan, nray), 20.0, dtype=np.float32)
    sw = synthetic_swath(
        nscan=nscan, nray=nray, near_sfc_dbz=dbz, near_sfc_rain=rain,
        rain_type=rain_type, pixel_area=pixel_area,
    )
    sw.bb_height = np.full((nscan, nray), 4800.0, dtype=np.float32)
    sw.freezing_level = np.full((nscan, nray), 4900.0, dtype=np.float32)
    return sw


def _build_frames(sw):
    """Replicate process_orbit's feature/pixel frame construction (offline)."""
    labeled, kept = label_rpf(sw)
    rows = []
    pixel_rows = []
    for local_label, area_km2 in kept:
        edge = touches_edge(labeled, local_label)
        rows.append(build_feature_row(sw, labeled, local_label, area_km2, edge))
        pixel_rows.extend(
            build_pixel_rows(sw, labeled, local_label, sw.mission, int(sw.orbit))
        )
    features_df = pd.DataFrame(
        rows, columns=[f.name for f in FEATURE_SCHEMA]
    )
    pixels_df = pd.DataFrame(
        pixel_rows, columns=[f.name for f in PIXEL_SCHEMA]
    )
    return labeled, kept, features_df, pixels_df


def test_two_features_kept(synthetic_swath):
    sw = _two_feature_swath(synthetic_swath)
    _, kept, features_df, _ = _build_frames(sw)
    assert len(kept) == 2
    assert len(features_df) == 2


def test_every_pixel_feature_id_in_features(synthetic_swath):
    sw = _two_feature_swath(synthetic_swath)
    _, _, features_df, pixels_df = _build_frames(sw)
    feature_ids = set(features_df["feature_id"].tolist())
    pixel_ids = set(pixels_df["feature_id"].tolist())
    assert pixel_ids.issubset(feature_ids)
    # both features actually have pixels
    assert pixel_ids == feature_ids


def test_per_feature_pixel_count_equals_npixels(synthetic_swath):
    sw = _two_feature_swath(synthetic_swath)
    _, _, features_df, pixels_df = _build_frames(sw)
    counts = pixels_df.groupby("feature_id").size()
    for _, frow in features_df.iterrows():
        fid = frow["feature_id"]
        assert int(counts.loc[fid]) == int(frow["npixels"]), (
            f"pixel count mismatch for feature {fid}"
        )


def test_total_pixels_equals_sum_npixels(synthetic_swath):
    sw = _two_feature_swath(synthetic_swath)
    _, _, features_df, pixels_df = _build_frames(sw)
    assert len(pixels_df) == int(features_df["npixels"].sum())


def test_frames_cast_to_schema(synthetic_swath):
    """Round-trip both frames through their pyarrow schemas (lossless cast)."""
    import pyarrow as pa

    sw = _two_feature_swath(synthetic_swath)
    _, _, features_df, pixels_df = _build_frames(sw)
    # should not raise — exercises the same cast catalog.write_orbit performs,
    # including the ns->us time coercion write_orbit applies before casting.
    ftbl = pa.Table.from_pandas(
        _coerce_time_us(features_df), schema=FEATURE_SCHEMA, preserve_index=False
    )
    ptbl = pa.Table.from_pandas(
        pixels_df, schema=PIXEL_SCHEMA, preserve_index=False
    )
    assert ftbl.num_rows == len(features_df)
    assert ptbl.num_rows == len(pixels_df)
