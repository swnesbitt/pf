"""Reader for the GPM 1C-GMI (V07) S1 imager swath.

Reads a local 1C-GMI HDF5 granule into an :class:`Imager` carrying the 89 GHz
polarization-corrected temperature (PCT) used to fill the radar swath's
``pct_85_89`` column during co-location (see :mod:`pf.colocate`).

Authoritative HDF5 paths (all swath-relative under ``S1``):

================================  ==========================================
Imager field                      HDF5 path
================================  ==========================================
``lat``                           ``S1/Latitude``
``lon``                           ``S1/Longitude``
``Tc`` (89V at idx 7, 89H idx 8)  ``S1/Tc`` ``(nscan, nray, 9)``
``time``                          ``S1/ScanTime/{Year,Month,DayOfMonth,Hour,
                                  Minute,Second,MilliSecond}``
================================  ==========================================

The :class:`GpmGmiReader` is intentionally **not** a
:class:`~pf.readers.base.SwathReader`: :meth:`read` returns an :class:`Imager`
(geolocation + PCT), not a radar :class:`~pf.swath.Swath`. It reuses the orbit
regex and the split-integer ``ScanTime`` composition from
:class:`~pf.readers.gpm_ku.GpmKuReader`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from pf import config
from pf.readers import hdf5_util
from pf.readers.gpm_ku import GpmKuReader, _ORBIT_RE  # noqa: F401  (regex reused)


@dataclass(slots=True)
class Imager:
    """One granule of GPM GMI 89 GHz PCT geolocated on the S1 swath.

    Attributes
    ----------
    mission : str
        Mission name, e.g. ``"GPM"``.
    orbit : int
        Orbit number parsed from the granule filename.
    short_name : str
        NASA product short name (``"GPM_1CGPMGMI"``).
    granule_name : str
        Source granule filename.
    lat, lon : ndarray, float32, shape (nscan, nray)
        S1 pixel-center geolocation in degrees; fills decoded to NaN.
    pct : ndarray, float32, shape (nscan, nray)
        89 GHz polarization-corrected temperature (K); NaN where either the
        89V or 89H channel is missing.
    time : ndarray, datetime64[ns], shape (nscan,)
        Per-scan acquisition time.
    sc_lat, sc_lon : ndarray, float32, shape (nscan,)
        Per-scan spacecraft sub-satellite point (nadir) latitude/longitude in
        degrees, from ``S1/SCstatus/SClatitude`` / ``SClongitude``. Used by
        :func:`pf.colocate.parallax_shift_geoloc` to derive the toward-the-
        sub-point parallax shift direction geometrically (so it flips correctly
        across GPM yaw maneuvers).
    pct37 : ndarray, float32, shape (nscan37, nray37)
        37 GHz (GMI 36.5 / TMI 37) polarization-corrected temperature (K); NaN
        where a channel is missing.
    lat37, lon37 : ndarray, float32, shape (nscan37, nray37)
        Geolocation of the 37 GHz swath. For GMI this is the S1 swath (identical
        to ``lat``/``lon``); for TMI it is the SEPARATE S2 swath, which has its
        own pixel grid and scan count, hence stored independently.
    sc_lat37, sc_lon37 : ndarray, float32, shape (nscan37,)
        Per-scan spacecraft sub-satellite point for the 37 GHz swath (drives its
        own parallax shift direction).
    """

    mission: str
    orbit: int
    short_name: str
    granule_name: str

    lat: np.ndarray
    lon: np.ndarray
    pct: np.ndarray
    time: np.ndarray
    sc_lat: np.ndarray
    sc_lon: np.ndarray

    pct37: np.ndarray
    lat37: np.ndarray
    lon37: np.ndarray
    sc_lat37: np.ndarray
    sc_lon37: np.ndarray


class GpmGmiReader:
    """Reader for the GPM 1C-GMI (V07) S1 swath (89 GHz PCT)."""

    short_name: str = "GPM_1CGPMGMI"
    mission: str = "GPM"
    #: Swath group read from the granule.
    swath: str = "S1"
    #: Swath group the PCT channels are read from (data reads route through this).
    pct_swath: str = "S1"
    #: Last-axis index of the V-pol PCT channel in ``<pct_swath>/Tc``.
    pct_v_idx: int = 7
    #: Last-axis index of the H-pol PCT channel in ``<pct_swath>/Tc``.
    pct_h_idx: int = 8

    #: Swath group the 37 GHz channels are read from. For GMI the 36.5 GHz
    #: channels share the S1 swath; for TMI 37 GHz lives in the separate S2.
    pct37_swath: str = "S1"
    #: Last-axis index of the 37 GHz V-pol channel in ``<pct37_swath>/Tc``.
    pct37_v_idx: int = 5
    #: Last-axis index of the 37 GHz H-pol channel in ``<pct37_swath>/Tc``.
    pct37_h_idx: int = 6

    #: Filename-derivation helper (reused from GpmKuReader). Exposed as an
    #: instance/class attribute so search-side imager resolution can call
    #: ``imager_reader._filename_of(granule)``; inherited by TrmmTmiReader.
    _filename_of = staticmethod(GpmKuReader._filename_of)

    def read(self, path: str) -> Imager:
        """Read a local 1C-GMI HDF5 granule into an :class:`Imager`.

        Parameters
        ----------
        path : str
            Absolute path to a local 1C-GMI ``.HDF5`` granule.

        Returns
        -------
        Imager
            Geolocation, per-scan time, and 89 GHz PCT.

        Raises
        ------
        ValueError
            If the channel-order self-check fails, i.e. the granule-mean 89V
            brightness temperature does not exceed the 89H mean (89V should be
            warmer; a failure indicates a swapped/incorrect channel index).
        """
        granule_name = os.path.basename(path)
        orbit = self.orbit_of(granule_name)

        with h5py.File(path, "r") as f:
            # --- main PCT swath (89 GHz / 85.5 GHz) --------------------------
            s = f"{self.pct_swath}/"
            lat, lon, sc_lat, sc_lon, tc = self._read_swath_geoloc_tc(f, s)

            # Per-scan time via the shared split-integer ScanTime composition.
            time = GpmKuReader._read_scan_time(f, s)

            pct = self._pct_from_tc(
                tc, self.pct_v_idx, self.pct_h_idx,
                config.PCT_A, config.PCT_B,
            )

            # --- 37 GHz swath -----------------------------------------------
            # GMI: same S1 swath -> reuse the already-read geoloc + Tc.
            # TMI: separate S2 swath -> read its own geoloc + Tc.
            if self.pct37_swath == self.pct_swath:
                lat37, lon37, sc_lat37, sc_lon37, tc37 = lat, lon, sc_lat, sc_lon, tc
            else:
                s37 = f"{self.pct37_swath}/"
                lat37, lon37, sc_lat37, sc_lon37, tc37 = (
                    self._read_swath_geoloc_tc(f, s37)
                )

            pct37 = self._pct_from_tc(
                tc37, self.pct37_v_idx, self.pct37_h_idx,
                config.PCT37_A, config.PCT37_B,
            )

        return Imager(
            mission=self.mission,
            orbit=orbit,
            short_name=self.short_name,
            granule_name=granule_name,
            lat=lat,
            lon=lon,
            pct=pct,
            time=time,
            sc_lat=sc_lat,
            sc_lon=sc_lon,
            pct37=pct37,
            lat37=lat37,
            lon37=lon37,
            sc_lat37=sc_lat37,
            sc_lon37=sc_lon37,
        )

    @staticmethod
    def _read_swath_geoloc_tc(
        f: Any, s: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read ``Latitude``/``Longitude``/``SCstatus``/``Tc`` from swath ``s``.

        Returns ``(lat, lon, sc_lat, sc_lon, tc)`` with the Tc cube's negative
        fills (incl. ``-9999.9``) decoded to NaN. ``s`` is the swath prefix
        including the trailing slash, e.g. ``"S1/"``.
        """
        lat = hdf5_util.read_float(f, s + "Latitude",
                                   sentinels=config.GEO_FILL_SENTINELS)
        lon = hdf5_util.read_float(f, s + "Longitude",
                                   sentinels=config.GEO_FILL_SENTINELS)

        # Tc: (nscan, nray, nchan); negative values (incl. -9999.9 fill) -> NaN.
        tc = hdf5_util.read_var(f, s + "Tc", dtype=np.float32)
        tc[tc < 0] = np.nan

        # Per-scan spacecraft sub-satellite point (for the geometry-driven
        # parallax direction in pf.colocate.parallax_shift_geoloc).
        sc_lat = hdf5_util.read_float(f, s + "SCstatus/SClatitude",
                                      sentinels=config.GEO_FILL_SENTINELS).astype(np.float32)
        sc_lon = hdf5_util.read_float(f, s + "SCstatus/SClongitude",
                                      sentinels=config.GEO_FILL_SENTINELS).astype(np.float32)
        return lat, lon, sc_lat, sc_lon, tc

    @staticmethod
    def _pct_from_tc(
        tc: np.ndarray, v_idx: int, h_idx: int, a: float, b: float
    ) -> np.ndarray:
        """Compute ``a*V - b*H`` PCT (K) from a Tc cube, with a V>H self-check.

        Raises
        ------
        ValueError
            If the granule-mean V brightness temperature does not exceed the H
            mean (V should be warmer; a failure indicates a swapped/incorrect
            channel index).
        """
        tc_v = tc[..., v_idx]
        tc_h = tc[..., h_idx]
        if not (np.nanmean(tc_v) > np.nanmean(tc_h)):
            raise ValueError(
                "imager channel-order self-check failed: "
                f"nanmean(V@{v_idx})={np.nanmean(tc_v):.2f} K <= "
                f"nanmean(H@{h_idx})={np.nanmean(tc_h):.2f} K"
            )
        return (a * tc_v - b * tc_h).astype(np.float32)

    def orbit_of(self, granule_or_filename: Any) -> int:
        """Parse the orbit number from a GPM granule or filename.

        Reuses :class:`~pf.readers.gpm_ku.GpmKuReader`'s regex and
        filename-derivation logic (the 5-6 digit field before the trailing
        ``V<NN>`` version token).

        Parameters
        ----------
        granule_or_filename : str or object
            A filename/path string, or a granule object from which a filename
            is derived.

        Returns
        -------
        int
            The orbit number.

        Raises
        ------
        ValueError
            If no orbit number can be parsed.
        """
        name = self._filename_of(granule_or_filename)
        match = _ORBIT_RE.search(os.path.basename(name))
        if match is None:
            raise ValueError(f"Could not parse GPM orbit number from {name!r}")
        return int(match.group(1))
