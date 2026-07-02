"""Per-pixel (member) table construction for one labeled feature.

Each labeled precipitation feature expands into one row per member pixel,
matching the frozen :data:`pf.features.PIXEL_SCHEMA` (14 columns). The pixel
table shares the single int64 ``feature_id`` join key with the feature catalog,
so ``count(pixels WHERE feature_id == fid)`` recovers exactly that feature's
``npixels``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pf import feature_id

#: Field order is authoritative — matches :data:`pf.features.PIXEL_SCHEMA`.
_PIXEL_KEYS = (
    "feature_id",
    "mission",
    "orbit",
    "scan",
    "ray",
    "lat",
    "lon",
    "near_sfc_dbz",
    "near_sfc_rain",
    "pct_85_89",
    "pct_37",
    "rain_type",
    "pixel_area_km2",
    "bb_height",
)


def build_pixel_rows(
    swath: Any,
    labeled: np.ndarray,
    local_label: int,
    mission: str,
    orbit: int,
) -> list[dict]:
    """Build one PIXEL_SCHEMA row per member pixel of ``local_label``.

    Parameters
    ----------
    swath : pf.swath.Swath
        The decoded orbit swath the feature was labeled from.
    labeled : ndarray, int32, shape (nscan, nray)
        Connected-component label image from :func:`pf.label.label_rpf`.
    local_label : int
        The label identifying this feature within ``labeled``.
    mission : str
        Mission name, e.g. ``"GPM"``.
    orbit : int
        Orbit number; combined with ``mission``/``local_label`` into the
        shared ``feature_id``.

    Returns
    -------
    list of dict
        One dict per member pixel, in ``np.nonzero`` row-major (scan-major)
        order, each keyed by the 13 :data:`pf.features.PIXEL_SCHEMA` field
        names with native Python scalars. An empty member mask returns ``[]``.
    """
    member = labeled == local_label
    scan_idx, ray_idx = np.nonzero(member)
    if scan_idx.size == 0:
        return []

    fid = int(feature_id.encode(mission, orbit, int(local_label)))
    mission_str = str(mission)
    orbit_int = int(orbit)

    # Vectorized field extraction at the member positions (row-major order).
    lat = swath.lat[scan_idx, ray_idx]
    lon = swath.lon[scan_idx, ray_idx]
    near_sfc_dbz = swath.near_sfc_dbz[scan_idx, ray_idx]
    near_sfc_rain = swath.near_sfc_rain[scan_idx, ray_idx]
    rain_type = swath.rain_type[scan_idx, ray_idx]
    pixel_area = swath.pixel_area[scan_idx, ray_idx]
    bb_height = swath.bb_height[scan_idx, ray_idx]
    pct_85_89 = swath.pct_85_89[scan_idx, ray_idx]
    pct_37 = swath.pct_37[scan_idx, ray_idx]

    rows: list[dict] = [
        {
            "feature_id": fid,
            "mission": mission_str,
            "orbit": orbit_int,
            "scan": int(scan_idx[k]),
            "ray": int(ray_idx[k]),
            "lat": float(lat[k]),
            "lon": float(lon[k]),
            "near_sfc_dbz": float(near_sfc_dbz[k]),
            "near_sfc_rain": float(near_sfc_rain[k]),
            "pct_85_89": float(pct_85_89[k]),
            "pct_37": float(pct_37[k]),
            "rain_type": int(rain_type[k]),
            "pixel_area_km2": float(pixel_area[k]),
            "bb_height": float(bb_height[k]),
        }
        for k in range(scan_idx.size)
    ]
    return rows


__all__ = ["build_pixel_rows"]
