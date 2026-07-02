"""Tests for :mod:`pf.geometry` pure functions."""

from __future__ import annotations

import numpy as np
import pytest

from pf.geometry import area_weighted_centroid, footprint_area_km2, pca_axes

_KM_PER_DEG = np.pi * 6371.0 / 180.0  # ~111.195


def _regular_grid(n=6, spacing_km=4.4, lat0=10.0, lon0=100.0):
    dlat = spacing_km / _KM_PER_DEG
    lats = lat0 + np.arange(n) * dlat
    lons = lon0 + np.arange(n) * dlat
    LON, LAT = np.meshgrid(lons, lats)
    return LAT.astype(np.float32), LON.astype(np.float32)


def test_footprint_area_regular_grid():
    LAT, LON = _regular_grid(n=6, spacing_km=4.4)
    area = footprint_area_km2(LAT, LON)
    # interior pixels should be ~4.4*4.4 ~= 19.36 km^2
    interior = area[1:-1, 1:-1]
    assert np.all(np.isfinite(interior))
    assert np.allclose(interior, 4.4 * 4.4, atol=0.5)
    assert 19.0 <= np.nanmean(interior) <= 20.0
    assert area.dtype == np.float32


def test_footprint_area_nan_geoloc_propagates():
    LAT, LON = _regular_grid(n=6)
    LAT = LAT.copy()
    LAT[2, 3] = np.nan
    area = footprint_area_km2(LAT, LON)
    assert np.isnan(area[2, 3])
    # a clearly distant interior pixel remains finite
    assert np.isfinite(area[0, 0]) or np.isfinite(area[4, 4])


def test_centroid_antimeridian():
    lat = np.array([[0.0, 0.0]], dtype=np.float64)
    lon = np.array([[179.0, -179.0]], dtype=np.float64)
    w = np.ones((1, 2))
    clat, clon = area_weighted_centroid(lat, lon, w)
    assert abs(clat) < 1e-6
    # centroid should be near +/-180, NOT near 0
    assert abs(abs(clon) - 180.0) < 1.0
    assert abs(clon) > 170.0


def test_centroid_zero_weight_returns_nan():
    lat = np.array([[1.0, 2.0]])
    lon = np.array([[1.0, 2.0]])
    w = np.zeros((1, 2))
    clat, clon = area_weighted_centroid(lat, lon, w)
    assert np.isnan(clat) and np.isnan(clon)


def test_pca_ew_elongated():
    # spread E-W (longitude), no spread N-S
    lons = np.linspace(-2.0, 2.0, 41)
    lats = np.zeros_like(lons)
    major, minor, orient, aspect = pca_axes(lats, lons)
    assert major > minor
    assert abs(orient) < 1.0          # ~0 deg (East-aligned)
    assert aspect > 1.0


def test_pca_isotropic_blob():
    rng = np.random.default_rng(42)
    th = rng.uniform(0, 2 * np.pi, 2000)
    r = np.sqrt(rng.uniform(0, 1, 2000))
    lat = r * np.sin(th)
    lon = r * np.cos(th)
    major, minor, orient, aspect = pca_axes(lat, lon)
    assert aspect == pytest.approx(1.0, abs=0.1)


def test_pca_single_point():
    assert pca_axes([5.0], [10.0]) == (0.0, 0.0, 0.0, 1.0)
