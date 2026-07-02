"""Regression test for the 2A-DPR dual-frequency axis being stripped.

DEFECT: FS reflectivity fields carry a trailing ``[Ku, Ka]`` size-2 axis:
``FS/SLV/zFactorFinalNearSurface`` is ``(nscan, nray, 2)`` and
``FS/SLV/zFactorFinal`` is ``(nscan, nray, nbin, 2)``. The reader must select
the Ku channel (index 0) on that trailing axis so the stored fields are exactly
``(nscan, nray)`` and ``(nscan, nray, nbin)``. The fix added the
``GpmKuReader._select_ku`` helper; this test exercises it directly.
"""

from __future__ import annotations

import numpy as np

from pf.readers.gpm_ku import GpmKuReader


def _ku_ka(base_shape):
    """Build a (*base_shape, 2) array with distinct Ku (0) vs Ka (1) values.

    Ku channel (index 0) holds ``base``; Ka channel (index 1) holds
    ``base + 1000`` so a wrong selection is unmistakable.
    """
    base = np.arange(int(np.prod(base_shape)), dtype=np.float32).reshape(base_shape)
    out = np.empty((*base_shape, 2), dtype=np.float32)
    out[..., 0] = base
    out[..., 1] = base + 1000.0
    return out, base


def test_select_ku_near_surface_2d_freq_axis():
    """(4, 5, 2) -> (4, 5), equal to arr[..., 0] (Ku, not Ka)."""
    arr, base = _ku_ka((4, 5))
    out = GpmKuReader._select_ku(arr)
    assert out.shape == (4, 5)
    np.testing.assert_array_equal(out, arr[..., 0])
    # Prove Ku (index 0), not Ka (index 1), was selected.
    np.testing.assert_array_equal(out, base)
    assert not np.array_equal(out, arr[..., 1])


def test_select_ku_3d_cube_freq_axis():
    """(4, 5, 176, 2) -> (4, 5, 176), equal to arr[..., 0]."""
    arr, base = _ku_ka((4, 5, 176))
    out = GpmKuReader._select_ku(arr)
    assert out.shape == (4, 5, 176)
    np.testing.assert_array_equal(out, arr[..., 0])
    np.testing.assert_array_equal(out, base)
    assert not np.array_equal(out, arr[..., 1])


def test_select_ku_already_2d_unchanged():
    """A plain (4, 5) array (no frequency axis) is returned unchanged."""
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    out = GpmKuReader._select_ku(arr)
    assert out.shape == (4, 5)
    np.testing.assert_array_equal(out, arr)


def test_select_ku_3d_non_freq_unchanged():
    """A 3-D (4, 5, 176) array whose trailing dim != 2 is returned unchanged."""
    arr = np.arange(4 * 5 * 176, dtype=np.float32).reshape(4, 5, 176)
    out = GpmKuReader._select_ku(arr)
    assert out.shape == (4, 5, 176)
    np.testing.assert_array_equal(out, arr)
