"""Swath geometry helpers: per-pixel area, centroid, and PCA shape axes.

These are pure functions. Unlike a regular-grid approximation (see
``cell_detector._estimate_pixel_area``, which is *invalid* for a conically
scanning swath), :func:`footprint_area_km2` measures each pixel from the
great-circle spacing to its in-scan and cross-scan neighbours, so the
cross-track area variation of the GPM-Ku swath is captured exactly.
"""

from __future__ import annotations

import numpy as np

from pf.config import EARTH_RADIUS_KM

# km per degree of latitude (mean Earth radius), used for the local equal-area
# plane in pca_axes. 2 * pi * R / 360.
_KM_PER_DEG: float = np.pi * EARTH_RADIUS_KM / 180.0  # ~111.195 km/deg


def _great_circle_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance (km) between two arrays of lat/lon points.

    Uses the haversine formula. NaN inputs propagate to NaN outputs.
    """
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def footprint_area_km2(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """Per-pixel footprint area (km^2) from great-circle neighbour spacing.

    For each pixel the along-scan (axis 0) and cross-scan (axis 1) spacings
    are estimated with central differences over the great-circle distance to
    the adjacent pixels (one-sided at the swath edges). The area is the
    product of the two orthogonalised spacings. Pixels whose geolocation —
    or whose required neighbours — are NaN yield NaN area.

    Parameters
    ----------
    lat2d, lon2d : ndarray, shape (nscan, nray)
        Pixel-center geolocation in degrees.

    Returns
    -------
    ndarray, float32, shape (nscan, nray)
        Footprint area per pixel in km^2.
    """
    lat = np.asarray(lat2d, dtype=np.float64)
    lon = np.asarray(lon2d, dtype=np.float64)
    nscan, nray = lat.shape

    # Along-scan spacing (axis 0): central where possible, one-sided at edges.
    dy = np.full((nscan, nray), np.nan, dtype=np.float64)
    if nscan >= 2:
        # forward distance i -> i+1 for rows [0, nscan-2]
        d_fwd = _great_circle_km(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
        # interior rows: half of (back + fwd) == central difference
        dy[1:-1, :] = 0.5 * (d_fwd[:-1, :] + d_fwd[1:, :])
        dy[0, :] = d_fwd[0, :]      # one-sided at first row
        dy[-1, :] = d_fwd[-1, :]    # one-sided at last row
    else:
        dy[:] = np.nan

    # Cross-scan spacing (axis 1).
    dx = np.full((nscan, nray), np.nan, dtype=np.float64)
    if nray >= 2:
        d_fwd = _great_circle_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        dx[:, 1:-1] = 0.5 * (d_fwd[:, :-1] + d_fwd[:, 1:])
        dx[:, 0] = d_fwd[:, 0]
        dx[:, -1] = d_fwd[:, -1]
    else:
        dx[:] = np.nan

    area = dx * dy

    # Any pixel with NaN geolocation must be NaN regardless of neighbour math.
    bad = ~np.isfinite(lat) | ~np.isfinite(lon)
    area[bad] = np.nan
    return area.astype(np.float32)


def area_weighted_centroid(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    weights2d: np.ndarray,
) -> tuple[float, float]:
    """Weighted geographic centroid, antimeridian-safe in longitude.

    Latitude is a simple weighted mean. Longitude is computed from the
    weighted mean of unit vectors ``(cos lon, sin lon)`` so that features
    straddling the +/-180 deg seam are handled correctly.

    Parameters
    ----------
    lat2d, lon2d : ndarray
        Geolocation in degrees (matching shape).
    weights2d : ndarray
        Non-negative weights (e.g. pixel area); same shape.

    Returns
    -------
    (lat, lon) : tuple of float
        Centroid latitude and longitude in degrees, ``lon`` in [-180, 180].
        Returns ``(nan, nan)`` if the total valid weight is zero.
    """
    lat = np.asarray(lat2d, dtype=np.float64)
    lon = np.asarray(lon2d, dtype=np.float64)
    w = np.asarray(weights2d, dtype=np.float64)

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(w) & (w > 0)
    wsum = w[valid].sum()
    if wsum <= 0:
        return (float("nan"), float("nan"))

    lat_c = float((lat[valid] * w[valid]).sum() / wsum)

    lon_r = np.radians(lon[valid])
    x = float((np.cos(lon_r) * w[valid]).sum())
    y = float((np.sin(lon_r) * w[valid]).sum())
    if x == 0.0 and y == 0.0:
        lon_c = float(lon[valid].mean())  # degenerate; fall back to raw mean
    else:
        lon_c = float(np.degrees(np.arctan2(y, x)))
    return (lat_c, lon_c)


def pca_axes(
    member_lat: np.ndarray,
    member_lon: np.ndarray,
) -> tuple[float, float, float, float]:
    """Principal-axis shape descriptors for a feature's member pixels.

    Member pixel centers are projected onto a *local equal-area* km plane
    about their centroid (small-angle approximation: ``x`` eastward,
    ``y`` northward in km), the covariance is eigen-decomposed, and the
    axis lengths are reported as ``4 * sqrt(eigenvalue)`` (four standard
    deviations, matching skimage regionprops conventions).

    Parameters
    ----------
    member_lat, member_lon : array_like
        1-D arrays of member pixel latitudes / longitudes (degrees).

    Returns
    -------
    (major_km, minor_km, orientation_deg, aspect_ratio) : tuple of float
        ``major_km >= minor_km``; ``orientation_deg`` is the major-axis angle
        CCW from East in [-90, 90]; ``aspect_ratio = major / minor >= 1``.
        ``aspect_ratio`` is clamped to a finite maximum of ``1e4`` so that
        collinear / single-pixel features yield a large-but-finite value
        rather than ``inf``. A single point returns ``(0.0, 0.0, 0.0, 1.0)``.
    """
    lat = np.asarray(member_lat, dtype=np.float64).ravel()
    lon = np.asarray(member_lon, dtype=np.float64).ravel()
    good = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[good]
    lon = lon[good]

    n = lat.size
    if n < 2:
        return (0.0, 0.0, 0.0, 1.0)

    lat0 = lat.mean()
    # Unwrap member longitudes onto a continuous frame about a reference so
    # that features straddling the +/-180 deg seam are not split. Without this
    # a tiny seam-crossing blob (e.g. lons -179.95 and +179.9) would span ~360
    # deg and yield a major axis larger than Earth's circumference.
    ref = lon[0]
    dlon = ((lon - ref + 180.0) % 360.0) - 180.0
    lon_unwrapped = ref + dlon
    lon0 = lon_unwrapped.mean()
    # Local equal-area km plane (small-angle projection about the centroid).
    x = (lon_unwrapped - lon0) * np.cos(np.radians(lat0)) * _KM_PER_DEG
    y = (lat - lat0) * _KM_PER_DEG

    cov = np.cov(np.vstack((x, y)), bias=True)
    # Symmetric 2x2 -> eigh gives ascending real eigenvalues + orthonormal vecs.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)

    minor_eig, major_eig = eigvals[0], eigvals[1]
    major_vec = eigvecs[:, 1]  # eigenvector of the larger eigenvalue

    major_km = float(4.0 * np.sqrt(major_eig))
    minor_km = float(4.0 * np.sqrt(minor_eig))

    # Orientation CCW from East of the major axis, folded into [-90, 90].
    orientation = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    if orientation > 90.0:
        orientation -= 180.0
    elif orientation <= -90.0:
        orientation += 180.0

    # Harden aspect ratio: collinear / very-thin tiny features have a near-zero
    # minor axis, which would give inf. Floor the minor axis at a small epsilon
    # and clamp the ratio to a finite maximum so the result is always finite.
    _EPS_KM = 1e-3
    _ASPECT_MAX = 1e4
    if major_km <= 0.0:
        aspect = 1.0
    else:
        aspect = min(major_km / max(minor_km, _EPS_KM), _ASPECT_MAX)

    return (major_km, minor_km, orientation, aspect)
