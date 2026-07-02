"""Phase-3 PCT-fill integration: features.col35 and pixels.col10.

Fully synthetic (no co-location): set ``swath.pct_85_89`` to known values over a
labeled member region, then assert :func:`pf.features.build_feature_row`
populates ``min_pct_85_89`` with the member nanmin and
:func:`pf.pixels.build_pixel_rows` carries the per-pixel swath PCT value. An
all-NaN-PCT member must yield ``min_pct_85_89 == NaN`` (graceful no-imager).
"""

from __future__ import annotations

import numpy as np

from pf.features import build_feature_row
from pf.label import label_rpf, touches_edge
from pf.pixels import build_pixel_rows


def _single_feature_swath(synthetic_swath):
    """One retained interior feature of 6 pixels (> MIN_AREA_KM2 at 20 km^2)."""
    nscan, nray = 12, 12
    dbz = np.full((nscan, nray), np.nan, dtype=np.float32)
    rain = np.full((nscan, nray), np.nan, dtype=np.float32)
    rain_type = np.full((nscan, nray), -1, dtype=np.int8)

    members = [(4, 4), (4, 5), (5, 4), (5, 5), (6, 4), (6, 5)]
    for s, r in members:
        dbz[s, r] = 35.0
        rain[s, r] = 3.0
        rain_type[s, r] = 1

    pixel_area = np.full((nscan, nray), 20.0, dtype=np.float32)
    sw = synthetic_swath(
        nscan=nscan, nray=nray, near_sfc_dbz=dbz, near_sfc_rain=rain,
        rain_type=rain_type, pixel_area=pixel_area,
    )
    return sw, members


def _first_label(sw):
    labeled, kept = label_rpf(sw)
    assert len(kept) == 1
    local_label, area_km2 = kept[0]
    return labeled, local_label, area_km2


def test_feature_min_pct_is_member_nanmin(synthetic_swath):
    sw, members = _single_feature_swath(synthetic_swath)
    labeled, local_label, area_km2 = _first_label(sw)

    # Assign distinct PCT values to member pixels; the rest stay NaN.
    pct = np.full(sw.shape, np.nan, dtype=np.float32)
    values = [260.0, 255.0, 200.0, 240.0, 250.0, 245.0]  # min 200.0
    for (s, r), v in zip(members, values):
        pct[s, r] = v
    sw.pct_85_89 = pct

    edge = touches_edge(labeled, local_label)
    row = build_feature_row(sw, labeled, local_label, area_km2, edge)

    member_mask = labeled == local_label
    expected = float(np.nanmin(sw.pct_85_89[member_mask]))
    assert expected == 200.0
    assert row["min_pct_85_89"] == expected


def test_feature_min_pct_ignores_one_nan_member(synthetic_swath):
    """A NaN at one member pixel is ignored by the nanmin."""
    sw, members = _single_feature_swath(synthetic_swath)
    labeled, local_label, area_km2 = _first_label(sw)

    pct = np.full(sw.shape, np.nan, dtype=np.float32)
    values = [260.0, np.nan, 210.0, 240.0, 250.0, 245.0]
    for (s, r), v in zip(members, values):
        pct[s, r] = v
    sw.pct_85_89 = pct

    edge = touches_edge(labeled, local_label)
    row = build_feature_row(sw, labeled, local_label, area_km2, edge)
    assert row["min_pct_85_89"] == 210.0


def test_pixel_pct_matches_swath_value(synthetic_swath):
    sw, members = _single_feature_swath(synthetic_swath)
    labeled, local_label, _ = _first_label(sw)

    pct = np.full(sw.shape, np.nan, dtype=np.float32)
    values = [260.0, 255.0, 200.0, 240.0, 250.0, 245.0]
    for (s, r), v in zip(members, values):
        pct[s, r] = v
    sw.pct_85_89 = pct

    rows = build_pixel_rows(sw, labeled, local_label, sw.mission, int(sw.orbit))
    assert len(rows) == len(members)
    for row in rows:
        s, r = row["scan"], row["ray"]
        expected = float(sw.pct_85_89[s, r])
        assert row["pct_85_89"] == expected


def test_all_nan_member_yields_nan_min_pct(synthetic_swath):
    """Graceful no-imager: an all-NaN PCT member gives min_pct_85_89 == NaN."""
    sw, _ = _single_feature_swath(synthetic_swath)
    labeled, local_label, area_km2 = _first_label(sw)

    # pct_85_89 left as all-NaN (Swath.empty default).
    assert np.isnan(sw.pct_85_89).all()

    edge = touches_edge(labeled, local_label)
    row = build_feature_row(sw, labeled, local_label, area_km2, edge)
    assert np.isnan(row["min_pct_85_89"])
