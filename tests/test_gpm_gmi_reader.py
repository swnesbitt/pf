"""Offline tests for :class:`pf.readers.gpm_gmi.GpmGmiReader`.

A tiny synthetic 1C-GMI-like HDF5 granule is written into ``tmp_path`` with
h5py (no real granule, no network). It carries an ``S1`` group with
``Latitude``/``Longitude`` ``(n, m)``, ``Tc`` ``(n, m, 9)`` (89V at index 7
warmer than 89H at index 8), the split-integer ``ScanTime`` fields, and
``SCstatus/SClatitude``/``SClongitude`` ``(n,)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pf import config
from pf.readers.gpm_gmi import GpmGmiReader

# The orbit regex needs ``.<digits>.V<NN>`` in the basename.
_GRANULE = "1C.GPM.GMI.XCAL2016-C.20180630-S230252-E003525.024647.V07A.HDF5"


def _write_synthetic_gmi(path, *, n=5, m=4, v_warmer=True):
    """Write a minimal 1C-GMI-like HDF5 granule and return the path string."""
    h5py = pytest.importorskip("h5py")

    rng = np.random.default_rng(0)
    lat = (np.arange(n, dtype=np.float32)[:, None] * 0.05 + 10.0)
    lat = np.broadcast_to(lat, (n, m)).astype(np.float32).copy()
    lon = (np.arange(m, dtype=np.float32)[None, :] * 0.05 + 100.0)
    lon = np.broadcast_to(lon, (n, m)).astype(np.float32).copy()

    tc = rng.uniform(250.0, 260.0, size=(n, m, 9)).astype(np.float32)
    # 36.5V (idx 5) / 36.5H (idx 6): keep V deterministically warmer than H so
    # the 37 GHz channel-order self-check passes (these share the S1 swath).
    tc[..., 5] = 268.0
    tc[..., 6] = 252.0
    # 89V (idx 7) and 89H (idx 8): control their relative warmth.
    if v_warmer:
        tc[..., 7] = 274.1
        tc[..., 8] = 257.2
    else:
        tc[..., 7] = 250.0   # 89V colder than 89H -> should trip self-check
        tc[..., 8] = 270.0

    sc_lat = (np.arange(n, dtype=np.float32) * 0.05 + 9.5)
    sc_lon = np.full(n, 100.0, dtype=np.float32)

    with h5py.File(path, "w") as f:
        s1 = f.create_group("S1")
        s1.create_dataset("Latitude", data=lat)
        s1.create_dataset("Longitude", data=lon)
        s1.create_dataset("Tc", data=tc)

        st = s1.create_group("ScanTime")
        st.create_dataset("Year", data=np.full(n, 2018, dtype=np.int16))
        st.create_dataset("Month", data=np.full(n, 6, dtype=np.int16))
        st.create_dataset("DayOfMonth", data=np.full(n, 30, dtype=np.int16))
        st.create_dataset("Hour", data=np.full(n, 23, dtype=np.int16))
        st.create_dataset("Minute", data=np.arange(n, dtype=np.int16))
        st.create_dataset("Second", data=np.zeros(n, dtype=np.int16))
        st.create_dataset("MilliSecond", data=np.zeros(n, dtype=np.int16))

        scs = s1.create_group("SCstatus")
        scs.create_dataset("SClatitude", data=sc_lat)
        scs.create_dataset("SClongitude", data=sc_lon)
        scs.create_dataset("SCorientation", data=np.full(n, 180, dtype=np.int16))

    return str(path)


def test_reader_pct_and_geoloc(tmp_path):
    path = _write_synthetic_gmi(tmp_path / _GRANULE, n=5, m=4)
    im = GpmGmiReader().read(path)

    assert im.pct.shape == (5, 4)
    assert im.pct.dtype == np.float32

    # PCT == PCT_A*Tc[...,7] - PCT_B*Tc[...,8] where finite.
    expected = config.PCT_A * 274.1 - config.PCT_B * 257.2
    finite = im.pct[np.isfinite(im.pct)]
    assert finite.size == 5 * 4
    assert np.allclose(finite, expected, atol=1e-2)

    # sc_lat / sc_lon populated, shape (n,).
    assert im.sc_lat.shape == (5,)
    assert im.sc_lon.shape == (5,)
    assert np.isfinite(im.sc_lat).all()
    assert np.isfinite(im.sc_lon).all()

    # time dtype datetime64[ns].
    assert im.time.dtype == np.dtype("datetime64[ns]")
    assert im.time.shape == (5,)

    # 37 GHz (36.5 GHz) shares the S1 swath -> idx 5/6, same geolocation.
    expected37 = config.PCT37_A * 268.0 - config.PCT37_B * 252.0
    finite37 = im.pct37[np.isfinite(im.pct37)]
    assert finite37.size == 5 * 4
    assert np.allclose(finite37, expected37, atol=1e-2)
    # Same-swath: lat37/lon37 are the S1 grid (identical arrays).
    assert np.array_equal(im.lat37, im.lat)
    assert np.array_equal(im.lon37, im.lon)

    # metadata.
    assert im.short_name == "GPM_1CGPMGMI"
    assert im.mission == "GPM"
    assert im.orbit == 24647


def test_reader_pct_fill_to_nan(tmp_path):
    """Tc < 0 (fill) propagates to NaN PCT."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / _GRANULE
    _write_synthetic_gmi(path, n=5, m=4)
    # Poke a fill value into one 89V sample.
    with h5py.File(path, "r+") as f:
        tc = f["S1/Tc"][:]
        tc[0, 0, 7] = -9999.9
        del f["S1/Tc"]
        f["S1"].create_dataset("Tc", data=tc)

    im = GpmGmiReader().read(str(path))
    assert np.isnan(im.pct[0, 0])
    # Other pixels remain finite.
    assert np.isfinite(im.pct[1, 1])


def test_reader_channel_order_self_check_raises(tmp_path):
    """89V colder than 89H trips the ValueError self-check."""
    path = _write_synthetic_gmi(tmp_path / _GRANULE, n=5, m=4, v_warmer=False)
    with pytest.raises(ValueError):
        GpmGmiReader().read(path)
