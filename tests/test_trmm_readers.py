"""Offline tests for the Phase-4 TRMM readers (PR + TMI).

Phase 4 registers mission ``"TRMM"`` by thin subclassing of the GPM readers:

* :class:`pf.readers.trmm_pr.TrmmPrReader` subclasses
  :class:`pf.readers.gpm_ku.GpmKuReader` (only ``short_name``/``mission``
  overridden), reading the same ``FS`` swath.
* :class:`pf.readers.trmm_tmi.TrmmTmiReader` subclasses
  :class:`pf.readers.gpm_gmi.GpmGmiReader` and reads the 85.5 GHz V/H PCT from
  the ``S3`` group (V at last-axis index 0, H at index 1).

Everything here is synthetic — tiny HDF5 granules written with h5py, no real
700 MB granules and no network. The synthetic-HDF5 helpers adapt
``tests/test_gpm_gmi_reader.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pf import config
from pf.readers.gpm_gmi import GpmGmiReader
from pf.readers.gpm_ku import GpmKuReader
from pf.readers.trmm_pr import TrmmPrReader
from pf.readers.trmm_tmi import TrmmTmiReader

# Orbit 522, 1997-12-30 — the Zipser-2006 N-Argentina storm granule names.
_PR_GRANULE = "2A.TRMM.PR.V9-x.19971230-S225350-E002507.000522.V07A.HDF5"
_TMI_GRANULE = "1C.TRMM.TMI.XCAL.19971230-S225350-E002507.000522.V07A.HDF5"


# ----------------------------------------------------------------------
# Class-attribute / inheritance tests (no granule needed)
# ----------------------------------------------------------------------
def test_trmm_pr_is_gpm_ku_subclass():
    assert issubclass(TrmmPrReader, GpmKuReader)


def test_trmm_pr_class_attrs():
    assert TrmmPrReader.short_name == "GPM_2APR"
    assert TrmmPrReader.mission == "TRMM"
    # swath inherited from GpmKuReader (config.GPM_SWATH == "FS").
    assert TrmmPrReader.swath == "FS"
    assert TrmmPrReader().swath == "FS"


def test_trmm_tmi_is_gpm_gmi_subclass():
    assert issubclass(TrmmTmiReader, GpmGmiReader)


def test_trmm_tmi_class_attrs():
    assert TrmmTmiReader.short_name == "GPM_1CTRMMTMI"
    assert TrmmTmiReader.mission == "TRMM"
    assert TrmmTmiReader.swath == "S3"
    assert TrmmTmiReader.pct_swath == "S3"
    assert TrmmTmiReader.pct_v_idx == 0
    assert TrmmTmiReader.pct_h_idx == 1
    # 37 GHz lives in the separate S2 swath at idx 3/4.
    assert TrmmTmiReader.pct37_swath == "S2"
    assert TrmmTmiReader.pct37_v_idx == 3
    assert TrmmTmiReader.pct37_h_idx == 4


def test_gpm_gmi_defaults_unchanged():
    """Parameterization must be a no-op for GPM: S1, idx 7/8."""
    assert GpmGmiReader.pct_swath == "S1"
    assert GpmGmiReader.pct_v_idx == 7
    assert GpmGmiReader.pct_h_idx == 8
    assert GpmGmiReader.swath == "S1"
    # GMI 36.5 GHz shares the S1 swath at idx 5/6.
    assert GpmGmiReader.pct37_swath == "S1"
    assert GpmGmiReader.pct37_v_idx == 5
    assert GpmGmiReader.pct37_h_idx == 6


def test_orbit_of_inherited_pr():
    assert TrmmPrReader().orbit_of(_PR_GRANULE) == 522


def test_orbit_of_inherited_tmi():
    assert TrmmTmiReader().orbit_of(_TMI_GRANULE) == 522


# ----------------------------------------------------------------------
# Synthetic TRMM TMI S3 reader test
# ----------------------------------------------------------------------
def _write_synthetic_tmi(path, *, n=6, m=8, v_warmer=True):
    """Write a minimal 1C-TRMM-TMI-like HDF5 granule.

    Adapts ``tests/test_gpm_gmi_reader.py``: GMI used S1 with 9 channels at
    idx 7/8; here we use S3 with 2 channels at idx 0 (85.5V) / 1 (85.5H) for the
    85.5 GHz PCT, PLUS a separate S2 group with 5 channels carrying the 37 GHz
    V/H at idx 3 / 4 (TMI's 37 GHz lives in its own swath, not S3).
    """
    h5py = pytest.importorskip("h5py")

    lat = (np.arange(n, dtype=np.float32)[:, None] * 0.05 - 23.0)
    lat = np.broadcast_to(lat, (n, m)).astype(np.float32).copy()
    lon = (np.arange(m, dtype=np.float32)[None, :] * 0.05 - 57.5)
    lon = np.broadcast_to(lon, (n, m)).astype(np.float32).copy()

    tc = np.empty((n, m, 2), dtype=np.float32)
    if v_warmer:
        tc[..., 0] = 274.1   # 85.5V (warmer)
        tc[..., 1] = 257.2   # 85.5H
    else:
        tc[..., 0] = 250.0   # 85.5V colder than H -> should trip self-check
        tc[..., 1] = 270.0

    # S2 37 GHz cube: 5 channels (19V,19H,21V,37V@3,37H@4); 37V warmer than 37H.
    tc_s2 = np.empty((n, m, 5), dtype=np.float32)
    tc_s2[..., 0] = 260.0   # 19V
    tc_s2[..., 1] = 240.0   # 19H
    tc_s2[..., 2] = 265.0   # 21V
    tc_s2[..., 3] = 270.5   # 37V (warmer)
    tc_s2[..., 4] = 255.3   # 37H
    # S2 geolocation slightly offset from S3 (a genuinely separate swath grid).
    lat_s2 = (lat + 0.01).astype(np.float32)
    lon_s2 = (lon + 0.01).astype(np.float32)

    sc_lat = (np.arange(n, dtype=np.float32) * 0.05 - 23.5)
    sc_lon = np.full(n, -57.5, dtype=np.float32)

    with h5py.File(path, "w") as f:
        s3 = f.create_group("S3")
        s3.create_dataset("Latitude", data=lat)
        s3.create_dataset("Longitude", data=lon)
        s3.create_dataset("Tc", data=tc)

        st = s3.create_group("ScanTime")
        st.create_dataset("Year", data=np.full(n, 1997, dtype=np.int16))
        st.create_dataset("Month", data=np.full(n, 12, dtype=np.int16))
        st.create_dataset("DayOfMonth", data=np.full(n, 30, dtype=np.int16))
        st.create_dataset("Hour", data=np.full(n, 23, dtype=np.int16))
        st.create_dataset("Minute", data=np.arange(n, dtype=np.int16))
        st.create_dataset("Second", data=np.zeros(n, dtype=np.int16))
        st.create_dataset("MilliSecond", data=np.zeros(n, dtype=np.int16))

        scs = s3.create_group("SCstatus")
        scs.create_dataset("SClatitude", data=sc_lat)
        scs.create_dataset("SClongitude", data=sc_lon)
        scs.create_dataset("SCorientation", data=np.full(n, 0, dtype=np.int16))

        # S2 swath (37 GHz): geoloc + Tc + SCstatus (no ScanTime needed — the
        # reader takes per-scan time only from the main pct_swath, S3).
        s2 = f.create_group("S2")
        s2.create_dataset("Latitude", data=lat_s2)
        s2.create_dataset("Longitude", data=lon_s2)
        s2.create_dataset("Tc", data=tc_s2)
        scs2 = s2.create_group("SCstatus")
        scs2.create_dataset("SClatitude", data=sc_lat)
        scs2.create_dataset("SClongitude", data=sc_lon)
        scs2.create_dataset("SCorientation", data=np.full(n, 0, dtype=np.int16))

    return str(path)


def test_trmm_tmi_reader_pct_and_geoloc(tmp_path):
    n, m = 6, 8
    path = _write_synthetic_tmi(tmp_path / _TMI_GRANULE, n=n, m=m)
    im = TrmmTmiReader().read(path)

    # Imager metadata: TRMM mission, TMI short-name, orbit 522.
    assert im.mission == "TRMM"
    assert im.short_name == "GPM_1CTRMMTMI"
    assert im.orbit == 522

    # PCT shape (n, m) and value == 1.818*V - 0.818*H where finite.
    assert im.pct.shape == (n, m)
    assert im.pct.dtype == np.float32
    expected = config.PCT_A * 274.1 - config.PCT_B * 257.2
    finite = im.pct[np.isfinite(im.pct)]
    assert finite.size == n * m
    assert np.allclose(finite, expected, atol=1e-2)
    # And explicitly the spec formula 1.818*V - 0.818*H.
    assert np.allclose(finite, 1.818 * 274.1 - 0.818 * 257.2, atol=1e-2)

    # sc_lat / sc_lon populated, shape (n,).
    assert im.sc_lat.shape == (n,)
    assert im.sc_lon.shape == (n,)
    assert np.isfinite(im.sc_lat).all()
    assert np.isfinite(im.sc_lon).all()

    # 37 GHz read from the SEPARATE S2 swath (idx 3/4), with its own geoloc.
    assert im.pct37.shape == (n, m)
    expected37 = config.PCT37_A * 270.5 - config.PCT37_B * 255.3
    finite37 = im.pct37[np.isfinite(im.pct37)]
    assert finite37.size == n * m
    assert np.allclose(finite37, expected37, atol=1e-2)
    # S2 geoloc is offset +0.01 from S3 -> proves it is NOT the S3 grid.
    assert np.allclose(im.lat37, im.lat + 0.01, atol=1e-4)
    assert np.allclose(im.lon37, im.lon + 0.01, atol=1e-4)

    # time dtype datetime64[ns], per-scan.
    assert im.time.dtype == np.dtype("datetime64[ns]")
    assert im.time.shape == (n,)


def test_trmm_tmi_reader_fill_to_nan(tmp_path):
    """Tc < 0 (fill) on the V channel propagates to NaN PCT in S3."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / _TMI_GRANULE
    _write_synthetic_tmi(path, n=6, m=8)
    with h5py.File(path, "r+") as f:
        tc = f["S3/Tc"][:]
        tc[0, 0, 0] = -9999.9   # poke a fill into 85.5V at idx 0
        del f["S3/Tc"]
        f["S3"].create_dataset("Tc", data=tc)

    im = TrmmTmiReader().read(str(path))
    assert np.isnan(im.pct[0, 0])
    assert np.isfinite(im.pct[1, 1])


def test_trmm_tmi_channel_order_self_check_raises(tmp_path):
    """85.5V colder than 85.5H (idx 0 < idx 1) trips the ValueError."""
    path = _write_synthetic_tmi(tmp_path / _TMI_GRANULE, n=6, m=8, v_warmer=False)
    with pytest.raises(ValueError):
        TrmmTmiReader().read(path)


# ----------------------------------------------------------------------
# Synthetic TRMM PR FS reader test
# ----------------------------------------------------------------------
def _write_synthetic_pr(path, *, n=5, m=49, nbin=config.GPM_N_RANGE_BINS):
    """Write a minimal 2A-TRMM-PR-like HDF5 granule (FS group).

    Writes exactly the FS/* fields that ``GpmKuReader.read`` consumes. TRMM PR
    reflectivity fields are 2-D ``(n, m)`` / 3-D ``(n, m, nbin)`` with NO
    trailing dual-frequency axis (unlike GPM 2A-DPR), so ``_select_ku`` is a
    safe no-op on them — this test verifies it leaves the 2-D fields untouched.
    """
    h5py = pytest.importorskip("h5py")

    lat = (np.arange(n, dtype=np.float32)[:, None] * 0.05 - 23.0)
    lat = np.broadcast_to(lat, (n, m)).astype(np.float32).copy()
    lon = (np.arange(m, dtype=np.float32)[None, :] * 0.05 - 57.5)
    lon = np.broadcast_to(lon, (n, m)).astype(np.float32).copy()

    near_sfc_dbz = np.full((n, m), 40.0, dtype=np.float32)
    near_sfc_dbz[2, 3] = 59.7                       # marker value
    precip_rate = np.full((n, m), 2.0, dtype=np.float32)
    dbz_3d = np.full((n, m, nbin), 30.0, dtype=np.float32)
    type_precip = np.full((n, m), 10_000_000, dtype=np.int32)   # // 1e7 -> 1
    height_storm_top = np.full((n, m), 17688.0, dtype=np.float32)
    pia_final = np.full((n, m), 0.5, dtype=np.float32)
    pia_final[2, 3] = 1.23                          # marker value
    height_zero_deg = np.full((n, m), 4500.0, dtype=np.float32)
    height_bb = np.full((n, m), 4400.0, dtype=np.float32)
    land_surface_type = np.zeros((n, m), dtype=np.int32)

    with h5py.File(path, "w") as f:
        fs = f.create_group("FS")
        fs.create_dataset("Latitude", data=lat)
        fs.create_dataset("Longitude", data=lon)

        slv = fs.create_group("SLV")
        slv.create_dataset("zFactorFinalNearSurface", data=near_sfc_dbz)
        slv.create_dataset("precipRateNearSurface", data=precip_rate)
        slv.create_dataset("zFactorFinal", data=dbz_3d)
        slv.create_dataset("piaFinal", data=pia_final)

        csf = fs.create_group("CSF")
        csf.create_dataset("typePrecip", data=type_precip)
        csf.create_dataset("heightBB", data=height_bb)

        pre = fs.create_group("PRE")
        pre.create_dataset("heightStormTop", data=height_storm_top)
        pre.create_dataset("landSurfaceType", data=land_surface_type)

        ver = fs.create_group("VER")
        ver.create_dataset("heightZeroDeg", data=height_zero_deg)

        st = fs.create_group("ScanTime")
        st.create_dataset("Year", data=np.full(n, 1997, dtype=np.int16))
        st.create_dataset("Month", data=np.full(n, 12, dtype=np.int16))
        st.create_dataset("DayOfMonth", data=np.full(n, 30, dtype=np.int16))
        st.create_dataset("Hour", data=np.full(n, 23, dtype=np.int16))
        st.create_dataset("Minute", data=np.arange(n, dtype=np.int16))
        st.create_dataset("Second", data=np.zeros(n, dtype=np.int16))
        st.create_dataset("MilliSecond", data=np.zeros(n, dtype=np.int16))

    return str(path)


def test_trmm_pr_reader_fs_swath(tmp_path):
    n, m, nbin = 5, 49, config.GPM_N_RANGE_BINS
    path = _write_synthetic_pr(tmp_path / _PR_GRANULE, n=n, m=m, nbin=nbin)
    sw = TrmmPrReader().read(path)

    # Swath metadata: TRMM mission, PR short-name, orbit 522.
    assert sw.mission == "TRMM"
    assert sw.short_name == "GPM_2APR"
    assert sw.orbit == 522

    # near_sfc_dbz is 2-D (n, m); _select_ku left the 2-D field untouched.
    assert sw.near_sfc_dbz.shape == (n, m)
    assert np.isclose(sw.near_sfc_dbz[2, 3], 59.7, atol=1e-2)

    # dbz_3d is (n, m, nbin).
    assert sw.dbz_3d.shape == (n, m, nbin)

    # piaFinal 2-D (n, m) left untouched by _select_ku (no freq axis).
    assert sw.pia.shape == (n, m)
    assert np.isclose(sw.pia[2, 3], 1.23, atol=1e-2)

    # Phase-2/3 gating fields read through unchanged.
    assert sw.storm_top.shape == (n, m)
    assert np.isclose(sw.storm_top[0, 0], 17688.0, atol=1.0)
