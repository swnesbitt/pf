"""Tests for pf.echotop_qc — Hirose-style echo-top quality control."""

from __future__ import annotations

import numpy as np

from pf import config, echotop_qc
from pf.swath import Swath

NBIN = 80
BINSIZE = 250.0  # m per synthetic bin (bin 0 = top)


def _swath(nscan=7, nray=49, mission="GPM"):
    """Synthetic swath with a known linear height grid (bin 0 highest)."""
    sw = Swath.empty(nscan, nray, NBIN, mission=mission, orbit=1,
                     short_name="GPM_2AKu", granule_name="2A.x.000001.V07A.HDF5")
    heights = (NBIN - 1 - np.arange(NBIN)) * BINSIZE          # (nbin,)
    sw.height_3d[:] = heights[None, None, :].astype(np.float32)
    sw.lat[:] = 0.0
    sw.lon[:] = 0.0
    return sw


def _bin_of(alt_m: float) -> int:
    return NBIN - 1 - int(round(alt_m / BINSIZE))


def _put_echo(sw, scan, ray, top_m, dbz=30.0, base_m=2000.0, rate=None):
    """Fill a column with Z=dbz from base_m up to top_m (inclusive)."""
    b_top, b_base = _bin_of(top_m), _bin_of(base_m)
    sw.dbz_3d[scan, ray, b_top:b_base + 1] = dbz
    if rate is not None:
        sw.rain_rate_3d[scan, ray, b_top:b_base + 1] = rate


def _member(sw, pixels):
    m = np.zeros(sw.shape, dtype=bool)
    for s, r in pixels:
        m[s, r] = True
    return m


def test_clean_tower_below_floor_unflagged():
    """A deep-but-sub-15 km tower is taken at face value, no flags."""
    sw = _swath()
    px = [(3, 24), (3, 25), (4, 24)]
    for s, r in px:
        _put_echo(sw, s, r, top_m=10000.0)
    out = echotop_qc.feature_echo_tops(sw, _member(sw, px))
    assert out["echotop_qc_flags"] == 0
    assert not out["max_ht_20dbz_censored"]
    assert abs(out["max_ht_20dbz"] - 10000.0) <= BINSIZE


def test_mirror_echo_above_flag_removed():
    """A 20 km echo above the binMirrorImageL2 altitude is dropped."""
    sw = _swath()
    px = [(3, 24), (3, 25)]
    for s, r in px:
        _put_echo(sw, s, r, top_m=12000.0)
    # mirror artifact (single high bin) above the flag; flag mirror bin ~16 km.
    _put_echo(sw, 3, 24, top_m=18000.0, base_m=18000.0)
    sw.bin_mirror_image[3, 24] = _bin_of(16000.0)
    out = echotop_qc.feature_echo_tops(sw, _member(sw, px))
    assert out["echotop_qc_flags"] & config.ETH_FLAG_MIRROR_REMOVED
    assert out["max_ht_20dbz"] <= 12000.0 + BINSIZE   # mirror gone, real top kept


def test_sidelobe_outer_ray_removed():
    """A high echo on an outer ray (sidelobe zone) is gated above the floor."""
    sw = _swath()
    inner = [(3, 24), (3, 25)]
    for s, r in inner:
        _put_echo(sw, s, r, top_m=12000.0)
    outer_ray = config.ETH_INNER_RAY_LO - 2
    _put_echo(sw, 3, outer_ray, top_m=18000.0, base_m=18000.0)   # single high bin
    px = inner + [(3, outer_ray)]
    out = echotop_qc.feature_echo_tops(sw, _member(sw, px))
    assert out["echotop_qc_flags"] & config.ETH_FLAG_SIDELOBE_REMOVED
    assert out["max_ht_20dbz"] <= 12000.0 + BINSIZE


def test_outer_ray_below_floor_kept():
    """An outer-ray echo BELOW the 15 km floor is NOT removed."""
    sw = _swath()
    outer_ray = config.ETH_INNER_RAY_HI + 2
    px = [(3, outer_ray)]
    _put_echo(sw, 3, outer_ray, top_m=13000.0)
    out = echotop_qc.feature_echo_tops(sw, _member(sw, px))
    assert not (out["echotop_qc_flags"] & config.ETH_FLAG_SIDELOBE_REMOVED)
    assert abs(out["max_ht_20dbz"] - 13000.0) <= BINSIZE


def test_isolated_noise_peeled():
    """An isolated high spike (no neighbour storm top > 15 km) is peeled."""
    sw = _swath()
    sw.storm_top[:] = 5000.0                    # all neighbours shallow -> isolated
    px = [(3, 24), (3, 25)]
    for s, r in px:
        _put_echo(sw, s, r, top_m=12000.0)
    _put_echo(sw, 3, 24, top_m=17000.0, base_m=17000.0)   # isolated noise at 17 km
    out = echotop_qc.feature_echo_tops(sw, _member(sw, px))
    assert out["echotop_qc_flags"] & config.ETH_FLAG_NOISE_PEELED
    assert out["max_ht_20dbz"] <= 12000.0 + BINSIZE


def test_overshoot_real_kept_and_censored():
    """A deep top reaching the ceiling, NOT isolated, is kept and censored."""
    sw = _swath()
    sw.storm_top[:] = 16000.0                   # neighbours deep -> NOT isolated
    top = 18900.0                               # >= 19000 - 1*125 -> truncated (GPM)
    px = [(3, 24), (3, 25), (4, 24), (4, 25)]
    for s, r in px:
        _put_echo(sw, s, r, top_m=top, rate=2.0)
        sw.rain_rate_3d[s, r, _bin_of(top)] = 30.0     # rate max at the top -> DD
    out = echotop_qc.feature_echo_tops(sw, _member(sw, px))
    f = out["echotop_qc_flags"]
    assert f & config.ETH_FLAG_TOP_TRUNCATED
    assert f & config.ETH_FLAG_CENSORED
    assert f & config.ETH_FLAG_OVERSHOOT_REAL
    assert not (f & config.ETH_FLAG_ISOLATED)
    assert out["max_ht_20dbz_censored"]
    assert out["max_ht_20dbz"] >= 18000.0       # the deep top is KEPT


def test_empty_member_returns_empty():
    sw = _swath()
    out = echotop_qc.feature_echo_tops(sw, np.zeros(sw.shape, dtype=bool))
    assert np.isnan(out["max_ht_20dbz"])
    assert out["max_ht_20dbz_scan"] == -1
    assert out["echotop_qc_flags"] == 0


def test_qc_never_exceeds_raw_on_real_granule():
    """On a real granule the QC top must never be ABOVE the raw nanmax top."""
    import os
    import pytest
    from pf.readers.gpm_ku import GpmKuReader
    from pf.label import label_rpf
    from pf import features

    probe = "/data/scratch/a/snesbitt/_pf_probe/2A.GPM.Ku.V9-20211125.20180630-S230252-E003525.024647.V07A.HDF5"
    if not os.path.exists(probe):
        pytest.skip("probe granule not present")
    sw = GpmKuReader().read(probe)
    lab, kept = label_rpf(sw, dbz_thresh=config.DBZ_THRESHOLD_BY_MISSION["GPM"],
                          min_area_km2=0, min_pixels=1, connectivity=2)
    for ll, _a in kept[:300]:
        m = lab == ll
        raw, _, _ = features._max_ht_at_threshold(m, sw.dbz_3d, sw.height_3d, 20.0)
        qc = echotop_qc.feature_echo_tops(sw, m)["max_ht_20dbz"]
        if np.isfinite(raw) and np.isfinite(qc):
            assert qc <= raw + 1.0          # QC only ever removes, never adds height
