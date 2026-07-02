"""Shared fixtures for the offline synthetic PF test suite.

Everything here is synthetic — no network, no real granules — so the suite
runs fast and anywhere. The ``synthetic_swath`` factory builds a
:class:`pf.swath.Swath` whose field names/dtypes match ``src/pf/swath.py``
exactly (verified against the dataclass definition).
"""

from __future__ import annotations

import numpy as np
import pytest

from pf.config import GPM_N_RANGE_BINS, GPM_RANGE_BIN_SIZE_M
from pf.swath import Swath


def _default_height_3d(nscan: int, nray: int, nbin: int) -> np.ndarray:
    """Bin-center heights following the reader convention.

    ``height_3d[s, r, b] = (nbin - 1 - b) * GPM_RANGE_BIN_SIZE_M`` so the bottom
    bin is ~0 m and the top is ~21875 m, broadcast over all pixels.
    """
    b = np.arange(nbin, dtype=np.float32)
    col = (nbin - 1 - b) * GPM_RANGE_BIN_SIZE_M
    return np.broadcast_to(col, (nscan, nray, nbin)).astype(np.float32).copy()


@pytest.fixture
def synthetic_swath():
    """Factory building a controllable synthetic :class:`Swath`.

    Returns a callable. All array arguments are optional; anything left as
    ``None`` is allocated with a sensible synthetic default of the correct
    dtype/shape. The factory keeps the Swath invariants (all 2-D fields share
    ``(nscan, nray)``; ``dbz_3d``/``height_3d`` add the trailing bin axis;
    ``time`` is per-scan).
    """

    def _make(
        nscan: int = 8,
        nray: int = 49,
        nbin: int = GPM_N_RANGE_BINS,
        *,
        mission: str = "GPM",
        orbit: int = 12345,
        short_name: str = "GPM_2ADPR",
        granule_name: str = "synthetic.HDF5",
        near_sfc_dbz=None,
        near_sfc_rain=None,
        dbz_3d=None,
        height_3d=None,
        pixel_area=None,
        rain_type=None,
        surface_type=None,
        lat=None,
        lon=None,
        time=None,
    ) -> Swath:
        sw = Swath.empty(
            nscan,
            nray,
            nbin,
            mission=mission,
            orbit=orbit,
            short_name=short_name,
            granule_name=granule_name,
        )

        # --- geolocation: regular ~4.4 km grid unless overridden -----------
        if lat is None or lon is None:
            dlat = 4.4 / 111.195  # ~4.4 km in degrees
            base_lat = (np.arange(nscan, dtype=np.float32)[:, None]
                        * dlat) + 10.0
            base_lon = (np.arange(nray, dtype=np.float32)[None, :]
                        * dlat) + 100.0
            grid_lat = np.broadcast_to(base_lat, (nscan, nray)).astype(np.float32)
            grid_lon = np.broadcast_to(base_lon, (nscan, nray)).astype(np.float32)
        if lat is not None:
            sw.lat = np.asarray(lat, dtype=np.float32)
        else:
            sw.lat = grid_lat.copy()
        if lon is not None:
            sw.lon = np.asarray(lon, dtype=np.float32)
        else:
            sw.lon = grid_lon.copy()

        # --- time: per-scan, 1 second per scan -----------------------------
        if time is not None:
            sw.time = np.asarray(time, dtype="datetime64[ns]")
        else:
            t0 = np.datetime64("2020-06-01T00:00:00", "ns")
            sw.time = (t0 + np.arange(nscan) * np.timedelta64(1, "s")).astype(
                "datetime64[ns]"
            )

        # --- pixel area ----------------------------------------------------
        if pixel_area is not None:
            sw.pixel_area = np.asarray(pixel_area, dtype=np.float32)
        else:
            sw.pixel_area = np.full((nscan, nray), 20.0, dtype=np.float32)

        # --- radar 2-D -----------------------------------------------------
        if near_sfc_dbz is not None:
            sw.near_sfc_dbz = np.asarray(near_sfc_dbz, dtype=np.float32)
        # else leave the NaN field from empty()

        if near_sfc_rain is not None:
            sw.near_sfc_rain = np.asarray(near_sfc_rain, dtype=np.float32)
        else:
            sw.near_sfc_rain = np.full((nscan, nray), 1.0, dtype=np.float32)

        if rain_type is not None:
            sw.rain_type = np.asarray(rain_type, dtype=np.int8)
        else:
            sw.rain_type = np.zeros((nscan, nray), dtype=np.int8)

        if surface_type is not None:
            sw.surface_type = np.asarray(surface_type, dtype=np.int8)
        else:
            sw.surface_type = np.zeros((nscan, nray), dtype=np.int8)  # 0 -> ocean

        # --- radar 3-D -----------------------------------------------------
        if dbz_3d is not None:
            sw.dbz_3d = np.asarray(dbz_3d, dtype=np.float32)
        # else leave NaN cube from empty()

        if height_3d is not None:
            sw.height_3d = np.asarray(height_3d, dtype=np.float32)
        else:
            sw.height_3d = _default_height_3d(nscan, nray, nbin)

        return sw

    return _make
