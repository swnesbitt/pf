"""Reader for the GPM 2A-Ku (Ku-only; FS swath).

Reads a local 2A-Ku HDF5 granule into a Phase-1 :class:`~pf.swath.Swath`. Only
the FS swath is used. The HDF5 group paths, fill handling, and split-integer
``ScanTime`` composition mirror the reusable ingest in ``ingest_gpm_dpr.py``.

Authoritative HDF5 paths (all swath-relative under ``FS``):

================================  ==========================================
Swath field                       HDF5 path
================================  ==========================================
``lat``                           ``FS/Latitude``
``lon``                           ``FS/Longitude``
``near_sfc_dbz``                  ``FS/SLV/zFactorFinalNearSurface`` (Ku index 0)
``near_sfc_rain``                 ``FS/SLV/precipRateNearSurface``
``dbz_3d``                        ``FS/SLV/zFactorFinal`` (Ku index 0 if 4-D)
``rain_type``                     ``FS/CSF/typePrecip`` ``// 10_000_000``
``surface_type``                  ``FS/PRE/landSurfaceType`` (if present)
``time``                          ``FS/ScanTime/{Year,Month,DayOfMonth,Hour,
                                  Minute,Second,MilliSecond}``
================================  ==========================================

.. note::
   2A-DPR FS reflectivity fields carry a trailing dual-frequency axis
   ``[Ku, Ka]`` (``zFactorFinalNearSurface`` is ``(nscan, nray, 2)`` and
   ``zFactorFinal`` is ``(nscan, nray, nbin, 2)``). This reader selects the
   Ku channel (index 0) on the last axis so the stored fields are exactly
   ``(nscan, nray)`` and ``(nscan, nray, nbin)`` respectively.
"""

from __future__ import annotations

import os
import re
from typing import Any

import h5py
import numpy as np

from pf import config
from pf.geometry import footprint_area_km2
from pf.readers import hdf5_util
from pf.readers.base import SwathReader
from pf.swath import Swath

# GPM 2A-DPR granule basename convention, e.g.:
#   2A.GPM.DPR.V9-20211125.20240101-S000000-E001226.057999.V07A.HDF5
# The orbit number is the 5-6 digit field immediately preceding the trailing
# ``V<NN>`` version token.
_ORBIT_RE = re.compile(r"\.(\d{5,6})\.V\d+", re.IGNORECASE)
# Product version token following the orbit field, e.g. ".024647.V07A.HDF5".
_VERSION_RE = re.compile(r"\.\d{5,6}\.(V\d+[A-Z]?)", re.IGNORECASE)


class GpmKuReader(SwathReader):
    """Reader for the GPM 2A-Ku (Ku-only; FS swath) (Phase-1)."""

    short_name: str = "GPM_2ADPR"
    mission: str = "GPM"
    #: Swath group read from the granule.
    swath: str = config.GPM_SWATH

    def read(self, path: str) -> Swath:
        """Read a local 2A-DPR HDF5 granule into a Phase-1 :class:`Swath`.

        Parameters
        ----------
        path : str
            Absolute path to a local 2A-DPR ``.HDF5`` granule.

        Returns
        -------
        pf.swath.Swath
            Swath with all Phase-1 fields populated; later-phase placeholders
            (``pct_85_89``, ``bb_height``, ``freezing_level``) are left as their
            NaN allocation from :meth:`Swath.empty`.
        """
        s = f"{self.swath}/"
        granule_name = os.path.basename(path)
        orbit = self.orbit_of(granule_name)

        with h5py.File(path, "r") as f:
            lat = hdf5_util.read_float(f, s + "Latitude",
                                       sentinels=config.GEO_FILL_SENTINELS)
            lon = hdf5_util.read_float(f, s + "Longitude",
                                       sentinels=config.GEO_FILL_SENTINELS)
            nscan, nray = lat.shape

            swath = Swath.empty(
                nscan,
                nray,
                config.GPM_N_RANGE_BINS,
                mission=self.mission,
                orbit=orbit,
                short_name=self.short_name,
                granule_name=granule_name,
            )

            # --- geolocation ---------------------------------------------
            swath.lat = lat
            swath.lon = lon

            # --- per-scan time -------------------------------------------
            swath.time = self._read_scan_time(f, s)

            # --- near-surface radar fields -------------------------------
            # FS carries a trailing [Ku, Ka] axis on the reflectivity fields;
            # select Ku (index 0) so near_sfc_dbz is exactly (nscan, nray).
            swath.near_sfc_dbz = self._select_ku(
                hdf5_util.read_float(f, s + "SLV/zFactorFinalNearSurface")
            )
            swath.near_sfc_rain = hdf5_util.read_float(
                f, s + "SLV/precipRateNearSurface"
            )

            # --- 3-D reflectivity cube -----------------------------------
            # Trailing frequency axis (nscan, nray, nbin, nfreq); Ku is index 0.
            dbz_3d = self._select_ku(hdf5_util.read_float(f, s + "SLV/zFactorFinal"))
            swath.dbz_3d = np.ascontiguousarray(dbz_3d, dtype=np.float32)

            # --- precipitation type --------------------------------------
            type_precip = hdf5_util.read_int(f, s + "CSF/typePrecip", dtype=np.int32)
            rain_type = type_precip // 10_000_000
            # Negative fill (e.g. -9999 // 1e7 -> -1) collapses to the int8 -1
            # "none/missing" sentinel; valid codes are 0..3.
            rain_type = np.where(rain_type < 0, -1, rain_type)
            swath.rain_type = rain_type.astype(np.int8)

            # --- surface type --------------------------------------------
            surf_path = s + "PRE/landSurfaceType"
            if hdf5_util.has_path(f, surf_path):
                swath.surface_type = hdf5_util.read_int(
                    f, surf_path, dtype=np.int32
                ).astype(np.int8)
            # else: leave Swath.empty()'s -1 fill in place.

            # --- bright-band height (Phase 2) ----------------------------
            # 2-D (nscan, nray); fill -1111.1 -> NaN via FILL_SENTINELS.
            bb_path = s + "CSF/heightBB"
            if hdf5_util.has_path(f, bb_path):
                swath.bb_height = hdf5_util.read_float(f, bb_path)
            # else: leave Swath.empty()'s NaN fill in place.

            # --- freezing level / 0 deg height (Phase 2) -----------------
            # 2-D (nscan, nray); fill -9999.9 -> NaN via FILL_SENTINELS.
            fz_path = s + "VER/heightZeroDeg"
            if hdf5_util.has_path(f, fz_path):
                swath.freezing_level = hdf5_util.read_float(f, fz_path)
            # else: leave Swath.empty()'s NaN fill in place.

            # --- storm top / path-integrated attenuation (Phase 3) -------
            # In-memory gating fields for the imager parallax shift; NOT
            # Parquet columns. heightStormTop is 2-D (nscan, nray) m;
            # piaFinal carries a trailing [Ku, Ka] axis (Ku index 0).
            st_path = s + "PRE/heightStormTop"
            if hdf5_util.has_path(f, st_path):
                swath.storm_top = hdf5_util.read_float(f, st_path)
            # else: leave Swath.empty()'s NaN fill in place.

            pia_path = s + "SLV/piaFinal"
            if hdf5_util.has_path(f, pia_path):
                swath.pia = self._select_ku(hdf5_util.read_float(f, pia_path))
            # else: leave Swath.empty()'s NaN fill in place.

            # --- per-ray off-nadir viewing angle (slant correction) ------
            # 2-D (nscan, nray) deg from nadir (~0 at swath centre, ~18 deg at
            # the edge). Used to slant-correct the 3-D bin heights below.
            lza_path = s + "PRE/localZenithAngle"
            if hdf5_util.has_path(f, lza_path):
                swath.local_zenith = self._select_ku(
                    hdf5_util.read_float(f, lza_path)
                )
            # else: leave Swath.empty()'s 0 (nadir) fill -> no correction.

            # --- echo-top QC inputs (pf.echotop_qc, Hirose et al. 2023) --
            # All optional: a granule lacking a field keeps Swath.empty()'s
            # benign default so the QC degrades gracefully. precipRate has no
            # frequency axis in the Ku-only product; fills (-9999.9) decode to
            # NaN. flagEcho / bin* are raw integer codes (no fill decode).
            rr_path = s + "SLV/precipRate"
            if hdf5_util.has_path(f, rr_path):
                swath.rain_rate_3d = np.ascontiguousarray(
                    self._select_ku(hdf5_util.read_float(f, rr_path)),
                    dtype=np.float32,
                )
            fe_path = s + "FLG/flagEcho"
            if hdf5_util.has_path(f, fe_path):
                swath.flag_echo = np.ascontiguousarray(
                    hdf5_util.read_int(f, fe_path, dtype=np.int8)
                )
            mir_path = s + "PRE/binMirrorImageL2"
            if hdf5_util.has_path(f, mir_path):
                swath.bin_mirror_image = hdf5_util.read_int(
                    f, mir_path, dtype=np.int16
                )
            cfb_path = s + "PRE/binClutterFreeBottom"
            if hdf5_util.has_path(f, cfb_path):
                swath.bin_clutter_bottom = hdf5_util.read_int(
                    f, cfb_path, dtype=np.int16
                )

            # --- DSD epsilon (near-surface gate) -------------------------
            # FS/SLV/epsilon is per-bin (nscan, nray, nbin) [+ trailing Ku/Ka
            # axis on 2A-DPR]; reduce to the near-surface clutter-free gate so
            # it matches the near_sfc_rain convention. Fills (-9999.9) -> NaN.
            eps_path = s + "SLV/epsilon"
            if hdf5_util.has_path(f, eps_path):
                eps_3d = self._select_ku(hdf5_util.read_float(f, eps_path))
                swath.epsilon = self._near_surface_gate(
                    eps_3d, swath.bin_clutter_bottom
                )
            # else: leave Swath.empty()'s NaN fill in place.

            # --- DSD parameters Nw, Dm (near-surface gate) ---------------
            # FS/SLV/paramDSD is (nscan, nray, nbin, nDSD=2): index 0 = Nw
            # (dBNw = 10*log10 Nw), index 1 = Dm (mm). The trailing axis is the
            # DSD-parameter axis (NOT Ku/Ka) -- do NOT _select_ku it. Fills NaN.
            dsd_path = s + "SLV/paramDSD"
            if hdf5_util.has_path(f, dsd_path):
                dsd = hdf5_util.read_float(f, dsd_path)   # (nscan, nray, nbin, 2)
                if dsd.ndim == 4 and dsd.shape[-1] == 2:
                    swath.nw = self._near_surface_gate(
                        np.ascontiguousarray(dsd[..., 0]), swath.bin_clutter_bottom)
                    swath.dm = self._near_surface_gate(
                        np.ascontiguousarray(dsd[..., 1]), swath.bin_clutter_bottom)
            # else: leave Swath.empty()'s NaN fill in place.

        # --- product version (drives the QC branch) ----------------------
        swath.product_version = self._version_of(granule_name)

        # --- per-pixel footprint area (single source of truth) -----------
        swath.pixel_area = footprint_area_km2(swath.lat, swath.lon)

        # --- 3-D height (m MSL), slant-corrected for off-nadir viewing ---
        swath.height_3d = self._height_3d(nscan, nray, swath.local_zenith)

        return swath

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _version_of(granule_name: str) -> str:
        """Parse the product version token (e.g. ``"V07A"``) from a filename.

        Returns the upper-cased token, or ``""`` if absent. Drives the echo-top
        QC branch (GPM Ku V07/V08 populate the JAXA mirror flag; TRMM PR V07A
        leaves it fill, so QC falls back to the geometric/isolation tests).
        """
        m = _VERSION_RE.search(str(granule_name))
        return m.group(1).upper() if m else ""

    @staticmethod
    def _select_ku(arr: np.ndarray) -> np.ndarray:
        """Strip a trailing dual-frequency axis, selecting Ku (index 0).

        2A-DPR FS reflectivity fields carry a trailing ``[Ku, Ka]`` axis of
        size 2 as the *last* dimension (after any range-bin axis). When present
        (``ndim >= 3`` and ``shape[-1] == 2``) this returns ``arr[..., 0]`` so
        the field collapses to its non-frequency shape; otherwise ``arr`` is
        returned unchanged.
        """
        if arr.ndim >= 3 and arr.shape[-1] == 2:
            return arr[..., 0]
        return arr

    @staticmethod
    def _near_surface_gate(cube: np.ndarray, bin_idx: np.ndarray) -> np.ndarray:
        """Gather a 3-D cube's value at the per-pixel near-surface gate.

        ``cube`` is ``(nscan, nray, nbin)`` and ``bin_idx`` is the per-ray
        ``binClutterFreeBottom`` ``(nscan, nray)`` (the lowest reliable bin).
        Returns ``(nscan, nray)`` float32 with the cube value at that bin, or
        NaN where the bin index is out of range. Cube fills are already NaN.
        """
        nscan, nray, nbin = cube.shape
        bi = np.asarray(bin_idx).astype(np.int64)
        valid = (bi >= 0) & (bi < nbin)
        bi_c = np.clip(bi, 0, nbin - 1)
        rows = np.arange(nscan)[:, None]
        cols = np.arange(nray)[None, :]
        out = cube[rows, cols, bi_c].astype(np.float32)
        return np.where(valid, out, np.nan).astype(np.float32)

    @staticmethod
    def _read_scan_time(f: h5py.File, swath_prefix: str) -> np.ndarray:
        """Compose a ``(nscan,)`` ``datetime64[ns]`` array from split int fields.

        The 2A-DPR ``ScanTime`` group stores year/month/day/hour/minute/second
        and millisecond as separate integer arrays (no single datetime field).
        """
        st = swath_prefix + "ScanTime/"
        year = hdf5_util.read_int(f, st + "Year", dtype=np.int64)
        month = hdf5_util.read_int(f, st + "Month", dtype=np.int64)
        day = hdf5_util.read_int(f, st + "DayOfMonth", dtype=np.int64)
        hour = hdf5_util.read_int(f, st + "Hour", dtype=np.int64)
        minute = hdf5_util.read_int(f, st + "Minute", dtype=np.int64)
        second = hdf5_util.read_int(f, st + "Second", dtype=np.int64)
        msec = hdf5_util.read_int(f, st + "MilliSecond", dtype=np.int64)

        nscan = year.shape[0]
        out = np.empty(nscan, dtype="datetime64[ns]")
        nat = np.datetime64("NaT", "ns")

        # Fill sentinels appear as negative values across these fields.
        valid = (year > 0) & (month > 0) & (day > 0)
        for s in range(nscan):
            if not valid[s]:
                out[s] = nat
                continue
            base = np.datetime64(
                f"{int(year[s]):04d}-{int(month[s]):02d}-{int(day[s]):02d}", "ns"
            )
            ms = max(int(msec[s]), 0)
            out[s] = (
                base
                + np.timedelta64(int(hour[s]), "h")
                + np.timedelta64(int(minute[s]), "m")
                + np.timedelta64(int(second[s]), "s")
                + np.timedelta64(ms, "ms")
            )
        return out

    @staticmethod
    def _height_3d(
        nscan: int, nray: int, local_zenith: np.ndarray | None = None
    ) -> np.ndarray:
        """Slant-corrected 3-D bin height (m above ellipsoid), ``(nscan,nray,nbin)``.

        The range gates are sampled every ``BIN_SIZE_M`` (125 m) *along the
        slant beam*, so the bin's **vertical** spacing is
        ``BIN_SIZE_M * cos(local_zenith)``. With the surface referenced at the
        bottom bin (index ``N_BINS - 1``) the height of bin ``b`` for a ray at
        off-nadir angle ``theta`` is::

            height[s, r, b] = (N_BINS - 1 - b) * BIN_SIZE_M * cos(theta[s, r])

        At nadir (``theta = 0``) this reduces to the old uniform 125 m column;
        at the ~18 deg swath edge ``cos(theta) ~ 0.951`` lowers a 18 km echo top
        by ~0.9 km, removing the off-nadir high bias. ``local_zenith`` is in
        degrees; non-finite or negative-fill entries fall back to nadir
        (``cos = 1``). Earth-ellipsoid curvature over the gate column is still
        neglected (sub-pixel at these heights).
        """
        nbin = config.GPM_N_RANGE_BINS
        bins = np.arange(nbin, dtype=np.float32)
        col = (nbin - 1 - bins) * np.float32(config.GPM_RANGE_BIN_SIZE_M)
        if local_zenith is None:
            return np.broadcast_to(col, (nscan, nray, nbin)).astype(
                np.float32, copy=True
            )
        theta = np.asarray(local_zenith, dtype=np.float32)
        cosz = np.cos(np.deg2rad(np.clip(theta, 0.0, 89.0)))
        # Non-finite / fill (e.g. -9999) -> nadir (no correction).
        cosz = np.where(np.isfinite(cosz) & (theta >= 0.0), cosz, 1.0).astype(
            np.float32
        )
        return (cosz[:, :, None] * col[None, None, :]).astype(np.float32)

    def orbit_of(self, granule_or_filename: Any) -> int:
        """Parse the orbit number from a GPM granule or filename.

        Parameters
        ----------
        granule_or_filename : str or object
            A filename/path string, or a granule object (e.g. an
            :class:`earthaccess.DataGranule`) from which a filename is derived
            via its data link or ``GranuleUR``.

        Returns
        -------
        int
            The orbit number (the 5-6 digit field before the trailing
            ``V<NN>`` token in the granule basename).

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

    @staticmethod
    def _filename_of(granule_or_filename: Any) -> str:
        """Derive a filename string from a granule handle or path."""
        if isinstance(granule_or_filename, (str, os.PathLike)):
            return os.fspath(granule_or_filename)

        g = granule_or_filename

        # earthaccess.DataGranule: prefer a concrete data link.
        try:
            links = g.data_links()  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            links = None
        if links:
            return os.path.basename(str(links[0]))

        # Fall back to the granule's UR (GranuleUR), often the bare filename.
        for getter in ("umm",):
            umm = getattr(g, getter, None)
            if isinstance(umm, dict):
                ur = umm.get("GranuleUR")
                if ur:
                    return os.path.basename(str(ur))

        ur = getattr(g, "GranuleUR", None)
        if ur:
            return os.path.basename(str(ur))

        raise ValueError(
            f"Could not derive a filename from granule object {granule_or_filename!r}"
        )
