"""Co-locate GPM GMI 89 GHz PCT onto the radar Ku FS swath.

Nearest-neighbour resampling (pyresample :func:`kd_tree.resample_nearest`)
maps each radar pixel to the closest GMI 89 GHz PCT sample within
:data:`pf.config.COLOCATE_RADIUS_M`. NEAREST is used deliberately so the
co-located field preserves true GMI cold-PCT minima (no smoothing).

Parallax correction (move the microwave data one scan TOWARD the sub-track)
---------------------------------------------------------------------------
The 89 GHz scattering depression originates near deep-convective cloud tops,
not at the surface the radar geolocates. The elevated ice-scattering pixel is
geolocated to the ellipsoid displaced *away* from nadir, so the surface column
it actually overlies sits one parallax step *toward* the sub-satellite (ground)
track. We approximate that displacement by shifting the GMI geolocation one
scan along-track, in the TOWARD-the-sub-satellite-point direction, before
resampling.

The along-track direction is **derived from spacecraft geometry**, not
hardcoded: for each interior scan we compare the great-circle distance from the
per-scan sub-satellite point ``(sc_lat[s], sc_lon[s])`` to the mid-scan pixel of
the previous scan ``s-1`` versus the next scan ``s+1``; the TOWARD neighbour is
the closer one. Deriving the sign geometrically lets it flip correctly across
GPM yaw maneuvers (``SCorientation`` 0 <-> 180), which reverse the along-track
sense of the scan index, without any per-orbit hardcoding.

The shifted PCT is only substituted for the unshifted PCT where the radar
indicates deep convection (``storm_top`` > :data:`pf.config.PARALLAX_STORMTOP_M`
AND ``pia`` > :data:`pf.config.PARALLAX_PIA_DBZ`); elsewhere the nadir-geolocated
(unshifted) PCT is kept.

Empirically (orbit 24647, ascending, ``SCorientation`` = 180) the geometry
resolves the toward direction to a ``shift = -1`` roll (data at scan ``s`` takes
the coordinates of scan ``s+1``), which maximizes the cold-PCT / Ku-storm-top
anti-correlation: r(PCT, storm_top) = -0.75 (toward) vs -0.60 (unshifted).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from pyresample import geometry, kd_tree

from pf import config


def _haversine_m(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (m) between point arrays, degrees in.

    Small internal helper for the parallax-direction geometry; uses a spherical
    Earth (radius 6371 km), adequate for comparing along-track neighbour
    distances.
    """
    r = 6_371_000.0
    phi1 = np.radians(np.asarray(lat1, dtype=np.float64))
    phi2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dphi = phi2 - phi1
    dlmb = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return r * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def parallax_shift_geoloc(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    sc_lat: np.ndarray,
    sc_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift a 2-D geolocation one scan TOWARD the sub-satellite point.

    The along-track shift direction is **derived from spacecraft geometry**,
    not hardcoded, so it flips correctly across GPM yaw maneuvers
    (``SCorientation`` 0 <-> 180). Over interior scans, the mean great-circle
    distance from the per-scan sub-satellite point ``(sc_lat[s], sc_lon[s])`` to
    the mid-scan pixel of scan ``s-1`` is compared against that to scan ``s+1``;
    the TOWARD neighbour is the closer one. The whole swath is then rolled one
    scan in that direction:

    - If ``pixel[s+1]`` is closer (toward = later scan), ``shift = -1``: data at
      scan ``s`` takes the coordinates of scan ``s+1``
      (``np.roll(shift=-1, axis=0)``), with the LAST row replicated (not
      wrapped).
    - If ``pixel[s-1]`` is closer (toward = earlier scan), ``shift = +1``: data
      at scan ``s`` takes the coordinates of scan ``s-1``
      (``np.roll(shift=+1, axis=0)``), with row 0 replicated.

    Returns ``float32`` copies; the inputs are never mutated.

    Empirically, for orbit 24647 (ascending, ``SCorientation`` = 180) the
    geometry resolves to ``shift = -1`` (pixel[s+1] ~470 km vs pixel[s-1]
    ~496 km from the sub-point), which maximizes the cold-PCT / Ku-storm-top
    anti-correlation: r(PCT, storm_top) = -0.75 (toward) vs -0.60 (unshifted).

    Parameters
    ----------
    lat2d, lon2d : ndarray, shape (nscan, nray)
        Pixel-center latitude/longitude in degrees.
    sc_lat, sc_lon : ndarray, shape (nscan,)
        Per-scan spacecraft sub-satellite point latitude/longitude in degrees.

    Returns
    -------
    lat_shift, lon_shift : ndarray, float32, shape (nscan, nray)
        One-scan-toward-sub-point-shifted geolocation copies.
    """
    lat = np.asarray(lat2d, dtype=np.float32)
    lon = np.asarray(lon2d, dtype=np.float32)
    slat = np.asarray(sc_lat, dtype=np.float64)
    slon = np.asarray(sc_lon, dtype=np.float64)

    nscan = lat.shape[0]

    # Default direction (used when too few scans to decide geometrically).
    toward_shift = -1

    if nscan >= 3:
        mid = lat.shape[1] // 2
        # Interior scans s = 1 .. nscan-2; compare sub-point[s] distance to the
        # mid-scan pixel of the previous (s-1) vs next (s+1) scan.
        s = np.arange(1, nscan - 1)
        plat = lat[:, mid].astype(np.float64)
        plon = lon[:, mid].astype(np.float64)

        d_prev = _haversine_m(slat[s], slon[s], plat[s - 1], plon[s - 1])
        d_next = _haversine_m(slat[s], slon[s], plat[s + 1], plon[s + 1])

        mean_prev = np.nanmean(d_prev)
        mean_next = np.nanmean(d_next)

        if np.isfinite(mean_prev) and np.isfinite(mean_next):
            # Closer neighbour = TOWARD the sub-satellite point.
            # next closer -> toward later scan -> shift -1; else shift +1.
            toward_shift = -1 if mean_next < mean_prev else 1

    lat_shift = np.roll(lat.copy(), shift=toward_shift, axis=0)
    lon_shift = np.roll(lon.copy(), shift=toward_shift, axis=0)

    # np.roll wraps; we want edge replication of the row that has no neighbour.
    if toward_shift == -1:
        lat_shift[-1] = lat[-1]
        lon_shift[-1] = lon[-1]
    else:  # toward_shift == +1
        lat_shift[0] = lat[0]
        lon_shift[0] = lon[0]

    return lat_shift, lon_shift


def _resample_pct_with_parallax(
    swath: Any,
    lat: np.ndarray,
    lon: np.ndarray,
    sc_lat: np.ndarray,
    sc_lon: np.ndarray,
    pct: np.ndarray,
    cfg: Any,
) -> np.ndarray:
    """Resample one imager PCT field onto the radar grid with parallax gating.

    Shared core of :func:`colocate_pct` (89/85.5 GHz) and
    :func:`colocate_pct37` (37 GHz). The two channels can carry **different**
    geolocation (GMI 36.5 GHz shares the 89 GHz S1 grid, but TMI 37 GHz lives in
    a separate S2 swath), so the geolocation is passed in explicitly rather than
    read off a fixed imager attribute. The deep-convection parallax gate is a
    property of the radar swath and is identical for both channels.

    Parameters
    ----------
    swath : pf.swath.Swath
        Radar swath; reads ``lat``, ``lon``, ``storm_top`` and ``pia``.
    lat, lon : ndarray, shape (nscan_i, nray_i)
        Imager pixel-center geolocation for this channel's swath.
    sc_lat, sc_lon : ndarray, shape (nscan_i,)
        Per-scan spacecraft sub-satellite point for this channel's swath.
    pct : ndarray, shape (nscan_i, nray_i)
        Imager PCT samples to resample.
    cfg : module
        Configuration namespace providing ``COLOCATE_RADIUS_M``,
        ``PARALLAX_STORMTOP_M`` and ``PARALLAX_PIA_DBZ``.

    Returns
    -------
    ndarray, float32, shape (nscan, nray)
        Co-located PCT on the radar grid; ``NaN`` where no imager sample falls
        within ``COLOCATE_RADIUS_M``.
    """
    # 1. target swath definition (radar grid).
    target_def = geometry.SwathDefinition(lons=swath.lon, lats=swath.lat)

    # 2. source swath definitions: nadir-geolocated and parallax-shifted.
    src_nopar = geometry.SwathDefinition(lons=lon, lats=lat)
    lat_p, lon_p = parallax_shift_geoloc(lat, lon, sc_lat, sc_lon)
    src_par = geometry.SwathDefinition(lons=lon_p, lats=lat_p)

    # 3. nearest-neighbour resample PCT through both source geometries.
    #    pyresample emits expected RuntimeWarnings for empty neighbourhoods.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pct_nopar = kd_tree.resample_nearest(
            src_nopar,
            pct,
            target_def,
            radius_of_influence=cfg.COLOCATE_RADIUS_M,
            fill_value=np.nan,
        )
        pct_par = kd_tree.resample_nearest(
            src_par,
            pct,
            target_def,
            radius_of_influence=cfg.COLOCATE_RADIUS_M,
            fill_value=np.nan,
        )

    # 4. parallax gate: deep-convective radar pixels only.
    storm_top = swath.storm_top
    pia = swath.pia
    gate = (
        np.isfinite(storm_top)
        & (storm_top > cfg.PARALLAX_STORMTOP_M)
        & np.isfinite(pia)
        & (pia > cfg.PARALLAX_PIA_DBZ)
    )

    # 5. substitute shifted PCT where gated; keep nadir PCT elsewhere.
    return np.where(gate, pct_par, pct_nopar).astype(np.float32)


def colocate_pct(swath: Any, imager: Any, cfg: Any = None) -> np.ndarray:
    """Co-locate GMI 89 GHz PCT onto the radar swath, with parallax gating.

    Parameters
    ----------
    swath : pf.swath.Swath
        Radar swath; reads ``lat``, ``lon``, ``storm_top`` and ``pia``.
    imager : pf.readers.gpm_gmi.Imager
        GMI imager; reads ``lat``, ``lon``, ``pct``, ``sc_lat`` and ``sc_lon``
        (the last two drive the geometry-derived parallax direction).
    cfg : module, optional
        Configuration namespace (default :mod:`pf.config`) providing
        ``COLOCATE_RADIUS_M``, ``PARALLAX_STORMTOP_M`` and ``PARALLAX_PIA_DBZ``.

    Returns
    -------
    ndarray, float32, shape (nscan, nray)
        Co-located 89 GHz PCT on the radar grid; ``NaN`` where no GMI sample
        falls within ``COLOCATE_RADIUS_M``.
    """
    if cfg is None:
        cfg = config
    return _resample_pct_with_parallax(
        swath, imager.lat, imager.lon, imager.sc_lat, imager.sc_lon,
        imager.pct, cfg,
    )


def colocate_pct37(swath: Any, imager: Any, cfg: Any = None) -> np.ndarray:
    """Co-locate the 37 GHz PCT onto the radar swath, with parallax gating.

    Identical machinery to :func:`colocate_pct` but using the imager's 37 GHz
    field and **its own** geolocation (``lat37``/``lon37``/``sc_lat37``/
    ``sc_lon37``) — which differs from the 89/85.5 GHz swath for TMI (S2 vs S3).

    Parameters
    ----------
    swath : pf.swath.Swath
        Radar swath; reads ``lat``, ``lon``, ``storm_top`` and ``pia``.
    imager : pf.readers.gpm_gmi.Imager
        Imager; reads ``lat37``, ``lon37``, ``pct37``, ``sc_lat37`` and
        ``sc_lon37``.
    cfg : module, optional
        Configuration namespace (default :mod:`pf.config`).

    Returns
    -------
    ndarray, float32, shape (nscan, nray)
        Co-located 37 GHz PCT on the radar grid; ``NaN`` where no imager sample
        falls within ``COLOCATE_RADIUS_M``.
    """
    if cfg is None:
        cfg = config
    return _resample_pct_with_parallax(
        swath, imager.lat37, imager.lon37, imager.sc_lat37, imager.sc_lon37,
        imager.pct37, cfg,
    )
