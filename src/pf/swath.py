"""The :class:`Swath` container — one orbit of decoded GPM-Ku FS data.

Reference frame is ``(nscan, nray)`` with ``nray == 49`` for the FS swath.
Floating-point 2-D fields are ``float32`` with fill sentinels already decoded
to ``NaN``; integer category fields use ``int8`` with sentinel ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pf.config import GPM_N_RANGE_BINS


@dataclass(slots=True)
class Swath:
    """Decoded single-orbit swath of GPM-Ku (2A-DPR FS) radar data.

    All 2-D fields share the ``(nscan, nray)`` shape; the 3-D fields add a
    trailing ``range_bin`` axis of length :data:`pf.config.GPM_N_RANGE_BINS`.

    Attributes
    ----------
    mission : str
        Mission name, e.g. ``"GPM"``.
    orbit : int
        Orbit number.
    short_name : str
        NASA product short name, e.g. ``"GPM_2ADPR"``.
    granule_name : str
        Source granule filename.
    lat, lon : ndarray, float32, shape (nscan, nray)
        Pixel-center geolocation in degrees; fills decoded to NaN.
    time : ndarray, datetime64[ns], shape (nscan,)
        Per-scan acquisition time.
    pixel_area : ndarray, float32, shape (nscan, nray)
        Per-pixel footprint area in km^2 (great-circle neighbour diffs).
    near_sfc_dbz : ndarray, float32, shape (nscan, nray)
        Near-surface final reflectivity (dBZ); fills NaN.
    near_sfc_rain : ndarray, float32, shape (nscan, nray)
        Near-surface precipitation rate (mm/hr); fills NaN.
    rain_type : ndarray, int8, shape (nscan, nray)
        Rain type (0 none, 1 strat, 2 conv, 3 other); sentinel -1.
    surface_type : ndarray, int8, shape (nscan, nray)
        Land-surface type code; sentinel -1.
    dbz_3d : ndarray, float32, shape (nscan, nray, nbin)
        Final reflectivity cube (dBZ); fills NaN.
    height_3d : ndarray, float32, shape (nscan, nray, nbin)
        Bin-center height above MSL (m).
    pct_85_89, bb_height, freezing_level : ndarray, float32, (nscan, nray)
        Later-phase placeholders, all NaN in Phase-1.
    pct_37 : ndarray, float32, (nscan, nray)
        Co-located 37 GHz (GMI 36.5 / TMI 37) polarization-corrected
        temperature; NaN placeholder until the imager phase fills it.
    storm_top, pia : ndarray, float32, (nscan, nray)
        Phase-3 in-memory gating fields (radar-native ``heightStormTop`` m and
        Ku ``piaFinal`` dBZ). These are **NOT** Parquet columns; they only feed
        the imager parallax gate in :func:`pf.colocate.colocate_pct`.
    """

    mission: str
    orbit: int
    short_name: str
    granule_name: str

    # Geolocation / time / geometry
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray
    pixel_area: np.ndarray

    # Radar 2-D
    near_sfc_dbz: np.ndarray
    near_sfc_rain: np.ndarray
    rain_type: np.ndarray
    surface_type: np.ndarray
    # DSD parameters at the near-surface clutter-free gate (FS/SLV, fills NaN),
    # all (nscan, nray): epsilon (FS/SLV/epsilon, ~1.0 dimensionless),
    # nw (paramDSD[...,0], dBNw = 10log10 Nw), dm (paramDSD[...,1], mm).
    epsilon: np.ndarray
    nw: np.ndarray
    dm: np.ndarray

    # Radar 3-D
    dbz_3d: np.ndarray
    height_3d: np.ndarray

    # Later-phase NaN placeholders
    pct_85_89: np.ndarray
    pct_37: np.ndarray
    bb_height: np.ndarray
    freezing_level: np.ndarray

    # Phase-3 in-memory gating fields (NOT Parquet columns)
    storm_top: np.ndarray
    pia: np.ndarray

    # Per-ray off-nadir viewing angle (deg from nadir), used to slant-correct
    # the 3-D bin heights. In-memory only (NOT a Parquet column). Default 0
    # (nadir) so a granule lacking the field falls back to no correction.
    local_zenith: np.ndarray

    # --- Echo-top QC inputs (in-memory only, NOT Parquet columns) ----------
    # Read for pf.echotop_qc (Hirose et al. 2023). All optional: a granule
    # lacking a field keeps the benign default so QC degrades gracefully.
    # rain_rate_3d: per-bin precip rate (mm/hr), fills NaN -> DD-from-top test.
    # flag_echo: per-bin echo classification (int8, raw codes; 0 = clear).
    # bin_mirror_image: per-ray JAXA mirror-image bin index (int16); <0 or
    #   >=nbin = no flag (GPM Ku populates it; TRMM PR leaves it fill).
    # bin_clutter_bottom: per-ray lowest clutter-free bin index (int16).
    # product_version: granule ProductVersion (e.g. "V07A"); drives QC branch.
    rain_rate_3d: np.ndarray
    flag_echo: np.ndarray
    bin_mirror_image: np.ndarray
    bin_clutter_bottom: np.ndarray
    product_version: str

    @property
    def shape(self) -> tuple[int, int]:
        """Return the ``(nscan, nray)`` shape of the 2-D fields."""
        return self.lat.shape  # type: ignore[return-value]

    @classmethod
    def empty(
        cls,
        nscan: int,
        nray: int,
        nbin: int = GPM_N_RANGE_BINS,
        *,
        mission: str,
        orbit: int,
        short_name: str,
        granule_name: str,
    ) -> "Swath":
        """Allocate a zero-filled :class:`Swath` of the requested shape.

        Float fields are filled with ``NaN`` and integer category fields with
        the sentinel ``-1``; ``time`` is filled with ``NaT``.

        Parameters
        ----------
        nscan, nray : int
            Along-track and cross-track dimensions.
        nbin : int, optional
            Vertical bins for the 3-D fields (default
            :data:`pf.config.GPM_N_RANGE_BINS`).
        mission, orbit, short_name, granule_name
            Metadata copied verbatim onto the new instance.

        Returns
        -------
        Swath
        """
        def f2d() -> np.ndarray:
            return np.full((nscan, nray), np.nan, dtype=np.float32)

        def i2d() -> np.ndarray:
            return np.full((nscan, nray), -1, dtype=np.int8)

        def f3d() -> np.ndarray:
            return np.full((nscan, nray, nbin), np.nan, dtype=np.float32)

        def i2d_fill(val: int) -> np.ndarray:
            return np.full((nscan, nray), val, dtype=np.int16)

        return cls(
            mission=mission,
            orbit=orbit,
            short_name=short_name,
            granule_name=granule_name,
            lat=f2d(),
            lon=f2d(),
            time=np.full((nscan,), np.datetime64("NaT"), dtype="datetime64[ns]"),
            pixel_area=f2d(),
            near_sfc_dbz=f2d(),
            near_sfc_rain=f2d(),
            rain_type=i2d(),
            surface_type=i2d(),
            epsilon=f2d(),
            nw=f2d(),
            dm=f2d(),
            dbz_3d=f3d(),
            height_3d=f3d(),
            pct_85_89=f2d(),
            pct_37=f2d(),
            bb_height=f2d(),
            freezing_level=f2d(),
            storm_top=f2d(),
            pia=f2d(),
            local_zenith=np.zeros((nscan, nray), dtype=np.float32),
            # Echo-top QC inputs: benign defaults (no rate, clear echo, no
            # mirror flag, clutter bottom = surface bin, unknown version).
            rain_rate_3d=f3d(),
            flag_echo=np.zeros((nscan, nray, nbin), dtype=np.int8),
            bin_mirror_image=i2d_fill(-9999),
            bin_clutter_bottom=i2d_fill(nbin - 1),
            product_version="",
        )
