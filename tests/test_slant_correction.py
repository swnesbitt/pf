"""Slant-angle correction of 3-D bin heights + ray-of-max provenance.

The range gates are sampled along the slant beam, so a bin's *vertical* height
is ``(N_BINS-1-b) * BIN_SIZE_M * cos(localZenithAngle)``. These tests lock in:
  * nadir (zenith 0) reproduces the old uniform-125 m column,
  * an off-nadir ray is lowered by exactly ``cos(theta)``,
  * fill/non-finite zenith falls back to nadir,
  * ``_max_ht_at_threshold`` returns the (scan, ray) of the contributing pixel.
"""
import numpy as np

from pf import config
from pf.readers.gpm_ku import GpmKuReader
from pf.features import _max_ht_at_threshold


def test_nadir_matches_uniform_column():
    nscan, nray = 3, 5
    nadir = GpmKuReader._height_3d(nscan, nray, np.zeros((nscan, nray), np.float32))
    legacy = GpmKuReader._height_3d(nscan, nray, None)
    assert np.allclose(nadir, legacy)
    # bottom bin ~0 m, top bin (N-1)*125 m
    assert np.isclose(legacy[0, 0, -1], 0.0)
    assert np.isclose(legacy[0, 0, 0],
                      (config.GPM_N_RANGE_BINS - 1) * config.GPM_RANGE_BIN_SIZE_M)


def test_offnadir_scaled_by_cos():
    nscan, nray = 1, 3
    theta = np.array([[0.0, 18.0, 18.0]], np.float32)
    h = GpmKuReader._height_3d(nscan, nray, theta)
    base = GpmKuReader._height_3d(nscan, nray, None)
    assert np.allclose(h[0, 0], base[0, 0])                     # nadir unchanged
    assert np.allclose(h[0, 1], base[0, 1] * np.cos(np.radians(18.0)), rtol=1e-5)
    # an 18 km echo at 18 deg is lowered by ~0.9 km
    top_uncorr = base[0, 1, 0]
    assert (top_uncorr - h[0, 1, 0]) > 800.0


def test_fill_zenith_falls_back_to_nadir():
    h = GpmKuReader._height_3d(1, 2, np.array([[-9999.0, np.nan]], np.float32))
    base = GpmKuReader._height_3d(1, 2, None)
    assert np.allclose(h, base)


def test_max_ht_returns_ray_of_max():
    nscan, nray, nbin = 2, 4, config.GPM_N_RANGE_BINS
    dbz = np.full((nscan, nray, nbin), np.nan, np.float32)
    height = GpmKuReader._height_3d(nscan, nray, None)
    member = np.zeros((nscan, nray), bool)
    member[0, 1] = member[1, 3] = True
    # Put a 45-dBZ echo higher (smaller bin index) at (1,3) than at (0,1).
    dbz[0, 1, 120] = 45.0
    dbz[1, 3, 60] = 45.0   # higher altitude -> should win
    val, scan, ray = _max_ht_at_threshold(member, dbz, height, 40.0)
    assert (scan, ray) == (1, 3)
    assert np.isclose(val, height[1, 3, 60])
    # threshold unmet -> sentinel
    v2, s2, r2 = _max_ht_at_threshold(member, dbz, height, 50.0)
    assert np.isnan(v2) and (s2, r2) == (-1, -1)
