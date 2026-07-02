"""Tests for :mod:`pf.label` connected-component labeling and edge logic."""

from __future__ import annotations

import numpy as np

from pf.config import (
    DBZ_THRESHOLD,
    DBZ_THRESHOLD_BY_MISSION,
    MIN_AREA_KM2,
    MIN_PIXELS,
)
from pf.label import label_rpf, touches_edge


def _blank_dbz(nscan, nray):
    return np.full((nscan, nray), np.nan, dtype=np.float32)


def test_big_blob_kept_small_blob_rejected(synthetic_swath):
    # NEW RULE: the binding minimum is MIN_PIXELS (default 1), not a 75 km^2
    # area floor. A single-pixel (and any >=1-pixel) >=threshold blob is now
    # RETAINED. We additionally verify the min_pixels parameter behaves as a
    # working floor when set explicitly.
    nscan, nray = 12, 12
    dbz = _blank_dbz(nscan, nray)
    # Big interior blob: 5 contiguous pixels.
    big_pixels = [(4, 4), (4, 5), (5, 4), (5, 5), (6, 4)]
    for s, r in big_pixels:
        dbz[s, r] = 30.0
    # Tiny far-corner blob: a single, isolated pixel.
    small_pixels = [(0, 11)]
    for s, r in small_pixels:
        dbz[s, r] = 25.0

    pixel_area = np.full((nscan, nray), 20.0, dtype=np.float32)
    sw = synthetic_swath(
        nscan=nscan, nray=nray, near_sfc_dbz=dbz, pixel_area=pixel_area
    )

    # --- default (min_pixels=1): BOTH blobs are retained -------------------
    labeled, kept = label_rpf(sw)
    assert len(kept) == 2

    # every >=threshold pixel ends up labeled non-zero (single-pixel kept too)
    for s, r in big_pixels + small_pixels:
        assert labeled[s, r] != 0

    # the single-pixel feature is present with the correct area-weighted area
    kept_areas = sorted(area for _lid, area in kept)
    assert kept_areas == [20.0, 100.0]  # 1*20 and 5*20

    # labeled non-zero labels are exactly the kept labels
    nonzero_labels = set(np.unique(labeled[labeled != 0]).tolist())
    assert nonzero_labels == {lid for lid, _ in kept}

    # --- explicit min_pixels=3: small (1-pixel) blob rejected, big kept ----
    labeled3, kept3 = label_rpf(sw, min_pixels=3)
    assert len(kept3) == 1
    kept_label, kept_area = kept3[0]
    assert kept_area == 100.0  # the 5-pixel blob, area-weighted 5*20
    member = labeled3 == kept_label
    assert int(member.sum()) == 5
    # the single-pixel blob is zeroed out under the 3-pixel floor
    for s, r in small_pixels:
        assert labeled3[s, r] == 0
    nonzero3 = set(np.unique(labeled3[labeled3 != 0]).tolist())
    assert nonzero3 == {kept_label}


def test_area_weighted_not_pixel_count(synthetic_swath):
    # NEW RULE: with defaults (min_pixels=1, min_area_km2=0) every contiguous
    # >=threshold blob is kept regardless of area, but the returned area_km2 is
    # still the area-weighted sum of pixel_area (value still computed
    # correctly even though it is no longer used as a retention floor).
    nscan, nray = 10, 10
    dbz = _blank_dbz(nscan, nray)
    blob = [(3, 3), (3, 4), (4, 3), (4, 4), (5, 3)]
    for s, r in blob:
        dbz[s, r] = 40.0
    pixel_area = np.full((nscan, nray), 1.0, dtype=np.float32)  # 5 km^2 total
    sw = synthetic_swath(
        nscan=nscan, nray=nray, near_sfc_dbz=dbz, pixel_area=pixel_area
    )

    # defaults: kept despite tiny area; area is area-weighted sum (5 * 1.0)
    labeled, kept = label_rpf(sw)
    assert len(kept) == 1
    kept_label, kept_area = kept[0]
    assert kept_area == 5.0  # NOT pixel count (5) coincidence: area-weighted
    member = labeled == kept_label
    assert int(member.sum()) == 5
    assert np.isclose(pixel_area[member].sum(), kept_area)

    # min_area_km2 still works as a floor when explicitly passed.
    labeled2, kept2 = label_rpf(sw, min_area_km2=1e6)
    assert kept2 == []
    assert not labeled2.any()

    # and a non-default-but-modest area floor between the two values works too:
    # 5 km^2 area passes a floor of 5.0 but fails a floor of 5.0 + epsilon.
    _l3, kept3 = label_rpf(sw, min_area_km2=5.0)
    assert len(kept3) == 1
    _l4, kept4 = label_rpf(sw, min_area_km2=5.0001)
    assert kept4 == []


def test_threshold_gate(synthetic_swath):
    # pixels just below threshold must not form a feature
    nscan, nray = 8, 8
    dbz = _blank_dbz(nscan, nray)
    for s, r in [(3, 3), (3, 4), (4, 3), (4, 4)]:
        dbz[s, r] = DBZ_THRESHOLD - 0.5
    pixel_area = np.full((nscan, nray), 50.0, dtype=np.float32)
    sw = synthetic_swath(
        nscan=nscan, nray=nray, near_sfc_dbz=dbz, pixel_area=pixel_area
    )
    labeled, kept = label_rpf(sw)
    assert kept == []
    assert not labeled.any()


def test_8connectivity_joins_diagonals(synthetic_swath):
    # two diagonally-adjacent pixels must be ONE feature under 8-connectivity
    nscan, nray = 8, 8
    dbz = _blank_dbz(nscan, nray)
    diag = [(3, 3), (4, 4), (5, 5), (6, 6)]
    for s, r in diag:
        dbz[s, r] = 30.0
    pixel_area = np.full((nscan, nray), 25.0, dtype=np.float32)  # 4*25 = 100
    sw = synthetic_swath(
        nscan=nscan, nray=nray, near_sfc_dbz=dbz, pixel_area=pixel_area
    )
    labeled, kept = label_rpf(sw)  # connectivity=2 default
    assert len(kept) == 1
    member = labeled == kept[0][0]
    assert int(member.sum()) == 4  # all four diagonal pixels joined


def test_across_track_contiguous_single_feature(synthetic_swath):
    # a blob spanning multiple rays in one scan is a single feature
    nscan, nray = 6, 10
    dbz = _blank_dbz(nscan, nray)
    for r in range(2, 8):  # 6 across-track contiguous pixels in scan 3
        dbz[3, r] = 35.0
    pixel_area = np.full((nscan, nray), 20.0, dtype=np.float32)  # 6*20=120
    sw = synthetic_swath(
        nscan=nscan, nray=nray, near_sfc_dbz=dbz, pixel_area=pixel_area
    )
    labeled, kept = label_rpf(sw)
    assert len(kept) == 1
    assert int((labeled == kept[0][0]).sum()) == 6


def test_touches_edge_true_and_false(synthetic_swath):
    nscan, nray = 10, 10
    dbz = _blank_dbz(nscan, nray)
    # edge-touching feature (column 0)
    edge_pixels = [(4, 0), (5, 0), (4, 1), (5, 1)]
    for s, r in edge_pixels:
        dbz[s, r] = 30.0
    # interior feature
    interior_pixels = [(3, 5), (3, 6), (4, 5), (4, 6)]
    for s, r in interior_pixels:
        dbz[s, r] = 30.0
    pixel_area = np.full((nscan, nray), 25.0, dtype=np.float32)  # 4*25=100 each
    sw = synthetic_swath(
        nscan=nscan, nray=nray, near_sfc_dbz=dbz, pixel_area=pixel_area
    )
    labeled, kept = label_rpf(sw)
    assert len(kept) == 2

    # identify which kept label is the edge one vs interior one
    edge_label = labeled[4, 0]
    interior_label = labeled[3, 5]
    assert edge_label != 0 and interior_label != 0
    assert edge_label != interior_label

    assert touches_edge(labeled, edge_label) is True
    assert touches_edge(labeled, interior_label) is False


def test_min_area_default_constant():
    # guard: NEW RPF definition -- 1-pixel minimum, no area floor, per-mission
    # near-surface noise-floor thresholds.
    assert MIN_AREA_KM2 == 0.0
    assert MIN_PIXELS == 1
    assert DBZ_THRESHOLD == 12.0
    assert DBZ_THRESHOLD_BY_MISSION == {"GPM": 12.0, "TRMM": 16.0}
