"""Offline synthetic tests for :mod:`pf.colocate`.

All grids are tiny and synthetic — no real granules, no network. The two
concerns under test are:

1. ``parallax_shift_geoloc`` derives the along-track shift direction from the
   spacecraft sub-track geometry (NOT a hardcoded sign), so it flips correctly
   when the sub-point is on the other along-track side (a yaw maneuver).
2. ``colocate_pct`` nearest-neighbour resamples GMI 89 GHz PCT onto the radar
   grid, substitutes the parallax-shifted resample only where the radar gate
   holds, preserves true imager values (no averaging), and yields NaN outside
   ``COLOCATE_RADIUS_M``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pf import config
from pf.colocate import colocate_pct, colocate_pct37, parallax_shift_geoloc
from pf.readers.gpm_gmi import Imager


# ---------------------------------------------------------------------------
# parallax_shift_geoloc — geometry-driven direction
# ---------------------------------------------------------------------------
def _synthetic_gmi_grid(nscan: int = 7, npix: int = 5):
    """A regular along-track-increasing-lat GMI lat/lon grid.

    Latitude increases with scan index (each scan ~0.04 deg further north);
    longitude increases across-track. Mid-scan column is npix//2.
    """
    dlat = 0.04
    base_lat = (np.arange(nscan, dtype=np.float32)[:, None] * dlat) + 10.0
    base_lon = (np.arange(npix, dtype=np.float32)[None, :] * dlat) + 100.0
    lat = np.broadcast_to(base_lat, (nscan, npix)).astype(np.float32).copy()
    lon = np.broadcast_to(base_lon, (nscan, npix)).astype(np.float32).copy()
    return lat, lon


def test_parallax_direction_toward_later_scan():
    """Sub-point north of the swath -> later scan (s+1) closer -> shift -1.

    With latitude increasing with scan index, placing the sub-satellite point
    well to the NORTH of every scan makes pixel[s+1] (the higher-lat, later
    scan) unambiguously closer than pixel[s-1]. The "toward" neighbour is s+1,
    so data at scan s should take the coordinates of s+1: ``lat_shift[:-1] ==
    lat[1:]`` with the last row replicated.
    """
    lat, lon = _synthetic_gmi_grid()
    nscan = lat.shape[0]
    mid = lon.shape[1] // 2
    # Sub-point far north (higher lat than any scan), same lon as mid column.
    sc_lat = np.full(nscan, lat[-1, mid] + 5.0, dtype=np.float32)
    sc_lon = np.full(nscan, lon[0, mid], dtype=np.float32)

    lat_in = lat.copy()
    lon_in = lon.copy()
    lat_s, lon_s = parallax_shift_geoloc(lat, lon, sc_lat, sc_lon)

    # toward = s+1 => shift -1: data at s gets coords of s+1.
    assert np.array_equal(lat_s[:-1], lat[1:])
    assert np.array_equal(lon_s[:-1], lon[1:])
    # Last row has no s+1 neighbour -> replicated (not wrapped).
    assert np.array_equal(lat_s[-1], lat[-1])
    assert np.array_equal(lon_s[-1], lon[-1])

    assert lat_s.dtype == np.float32
    assert lon_s.dtype == np.float32
    # Inputs unmutated.
    assert np.array_equal(lat, lat_in)
    assert np.array_equal(lon, lon_in)


def test_parallax_direction_flips_toward_earlier_scan():
    """Mirrored: sub-point south of the swath -> earlier scan (s-1) closer.

    Same grid, but the sub-point is placed far SOUTH (lower lat than any scan).
    Now pixel[s-1] is the closer along-track neighbour, so the toward direction
    flips to shift +1: data at scan s takes the coordinates of s-1
    (``lat_shift[1:] == lat[:-1]``) with row 0 replicated. This proves the sign
    is geometry-derived, not hardcoded, and would flip across a yaw maneuver.
    """
    lat, lon = _synthetic_gmi_grid()
    nscan = lat.shape[0]
    mid = lon.shape[1] // 2
    # Sub-point far south (lower lat than any scan).
    sc_lat = np.full(nscan, lat[0, mid] - 5.0, dtype=np.float32)
    sc_lon = np.full(nscan, lon[0, mid], dtype=np.float32)

    lat_s, lon_s = parallax_shift_geoloc(lat, lon, sc_lat, sc_lon)

    # toward = s-1 => shift +1: data at s gets coords of s-1.
    assert np.array_equal(lat_s[1:], lat[:-1])
    assert np.array_equal(lon_s[1:], lon[:-1])
    # Row 0 has no s-1 neighbour -> replicated.
    assert np.array_equal(lat_s[0], lat[0])
    assert np.array_equal(lon_s[0], lon[0])

    assert lat_s.dtype == np.float32 and lon_s.dtype == np.float32


# ---------------------------------------------------------------------------
# colocate_pct — resample, gating, nearest-neighbour, radius
# ---------------------------------------------------------------------------
def _tiny_imager(
    lat, lon, pct, sc_lat, sc_lon,
    pct37=None, lat37=None, lon37=None, sc_lat37=None, sc_lon37=None,
) -> Imager:
    """Build a synthetic Imager.

    The 37 GHz fields default to the 89 GHz geolocation (the GMI same-S1-swath
    case); pass ``lat37``/``lon37``/``pct37``/``sc_lat37``/``sc_lon37`` to
    exercise the separate-swath (TMI S2) path.
    """
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    pct = np.asarray(pct, dtype=np.float32)
    sc_lat = np.asarray(sc_lat, dtype=np.float32)
    sc_lon = np.asarray(sc_lon, dtype=np.float32)
    return Imager(
        mission="GPM",
        orbit=24647,
        short_name="GPM_1CGPMGMI",
        granule_name="synthetic.HDF5",
        lat=lat,
        lon=lon,
        pct=pct,
        time=np.array(
            ["2018-06-30T23:00:00"] * lat.shape[0], dtype="datetime64[ns]"
        ),
        sc_lat=sc_lat,
        sc_lon=sc_lon,
        pct37=pct.copy() if pct37 is None else np.asarray(pct37, dtype=np.float32),
        lat37=lat if lat37 is None else np.asarray(lat37, dtype=np.float32),
        lon37=lon if lon37 is None else np.asarray(lon37, dtype=np.float32),
        sc_lat37=sc_lat if sc_lat37 is None
        else np.asarray(sc_lat37, dtype=np.float32),
        sc_lon37=sc_lon if sc_lon37 is None
        else np.asarray(sc_lon37, dtype=np.float32),
    )


def test_colocate_shape_and_dtype(synthetic_swath):
    """Returned array matches the radar swath shape and is float32."""
    nscan, nray = 6, 5
    sw = synthetic_swath(nscan=nscan, nray=nray)
    # Imager covering the same region so every radar pixel has a sample.
    imlat = sw.lat.copy()
    imlon = sw.lon.copy()
    pct = np.full((nscan, nray), 250.0, dtype=np.float32)
    sc_lat = np.full(nscan, float(sw.lat.max()) + 5.0, dtype=np.float32)
    sc_lon = np.full(nscan, float(sw.lon.mean()), dtype=np.float32)
    im = _tiny_imager(imlat, imlon, pct, sc_lat, sc_lon)

    out = colocate_pct(sw, im)
    assert out.shape == sw.shape == (nscan, nray)
    assert out.dtype == np.float32


def test_colocate_gate_selects_parallax_vs_unshifted(synthetic_swath):
    """Gated pixels use the parallax (shifted) resample; ungated use unshifted.

    Construct the imager PCT so that a single radar pixel resolves to DIFFERENT
    imager values under the shifted vs unshifted geolocation. The shift here is
    toward s+1 (sub-point to the north), i.e. shift -1: parallax src at radar
    pixel (s, r) samples the imager value originally at (s+1, r).

    The radar gate (storm_top > 5000 m AND pia > 0.4 dBZ) is set True at exactly
    one pixel and False elsewhere, so we can assert np.where behaviour directly.
    """
    nscan, nray = 6, 5
    sw = synthetic_swath(nscan=nscan, nray=nray)

    # Imager exactly coincident with the radar grid so nearest-neighbour is
    # an identity within each (shifted/unshifted) geometry.
    imlat = sw.lat.copy()
    imlon = sw.lon.copy()

    # PCT increases with scan index so shifted (s -> s+1) != unshifted (s).
    pct = np.tile(
        (np.arange(nscan, dtype=np.float32) * 10.0 + 200.0)[:, None], (1, nray)
    )
    im = _tiny_imager(
        imlat,
        imlon,
        pct,
        sc_lat=np.full(nscan, float(sw.lat.max()) + 5.0, dtype=np.float32),
        sc_lon=np.full(nscan, float(sw.lon.mean()), dtype=np.float32),
    )

    # Radar gate: True at exactly (gs, gr); enforce storm_top/pia accordingly.
    sw.storm_top = np.zeros((nscan, nray), dtype=np.float32)  # below threshold
    sw.pia = np.zeros((nscan, nray), dtype=np.float32)
    gs, gr = 2, 2
    sw.storm_top[gs, gr] = 8000.0   # > PARALLAX_STORMTOP_M (5000)
    sw.pia[gs, gr] = 1.0            # > PARALLAX_PIA_DBZ (0.4)

    out = colocate_pct(sw, im)

    # Reference resamples: independently reproduce the unshifted and the
    # parallax-shifted nearest-neighbour fields and assert np.where selects the
    # shifted one ONLY at the gated pixel and the unshifted one elsewhere. This
    # asserts the gating dispatch without hardcoding the roll algebra.
    from pyresample import geometry, kd_tree

    target = geometry.SwathDefinition(lons=sw.lon, lats=sw.lat)
    src_nopar = geometry.SwathDefinition(lons=im.lon, lats=im.lat)
    lat_p, lon_p = parallax_shift_geoloc(im.lat, im.lon, im.sc_lat, im.sc_lon)
    src_par = geometry.SwathDefinition(lons=lon_p, lats=lat_p)
    pct_nopar = kd_tree.resample_nearest(
        src_nopar, im.pct, target,
        radius_of_influence=config.COLOCATE_RADIUS_M, fill_value=np.nan,
    )
    pct_par = kd_tree.resample_nearest(
        src_par, im.pct, target,
        radius_of_influence=config.COLOCATE_RADIUS_M, fill_value=np.nan,
    )

    # The shifted and unshifted resamples must actually differ at the gated
    # pixel, otherwise the test would not distinguish the two branches.
    assert pct_par[gs, gr] != pct_nopar[gs, gr]

    # Gated pixel -> parallax-shifted resample.
    assert out[gs, gr] == pytest.approx(pct_par[gs, gr])
    # An ungated pixel -> unshifted resample.
    us, ur = 1, 1
    assert out[us, ur] == pytest.approx(pct_nopar[us, ur])
    # And for an ungated pixel the unshifted value is its own coincident scan.
    assert out[us, ur] == pytest.approx(pct[us, ur])


def test_colocate_nearest_preserves_imager_value(synthetic_swath):
    """Nearest-neighbour preserves an actual imager PCT sample (no averaging).

    With a cold-PCT spot in the imager and an ungated radar grid, the
    co-located value at the radar pixel nearest that spot equals the imager
    sample exactly — there is no smoothing/averaging.
    """
    nscan, nray = 6, 4
    sw = synthetic_swath(nscan=nscan, nray=nray)
    # Ungated everywhere so we always use the unshifted (identity) resample.
    sw.storm_top = np.zeros((nscan, nray), dtype=np.float32)
    sw.pia = np.zeros((nscan, nray), dtype=np.float32)

    imlat = sw.lat.copy()
    imlon = sw.lon.copy()
    pct = np.full((nscan, nray), 270.0, dtype=np.float32)
    cold_value = 123.5
    cs, cr = 3, 1
    pct[cs, cr] = cold_value
    im = _tiny_imager(
        imlat,
        imlon,
        pct,
        sc_lat=np.full(nscan, float(sw.lat.max()) + 5.0, dtype=np.float32),
        sc_lon=np.full(nscan, float(sw.lon.mean()), dtype=np.float32),
    )

    out = colocate_pct(sw, im)
    # The cold value survives verbatim at the coincident radar pixel.
    assert out[cs, cr] == pytest.approx(cold_value)
    # And it appears as an actual imager sample value somewhere in the output.
    assert np.any(np.isclose(out, cold_value))


def test_colocate_nan_outside_radius(synthetic_swath):
    """Radar pixels with no imager sample within COLOCATE_RADIUS_M are NaN."""
    nscan, nray = 6, 5
    sw = synthetic_swath(nscan=nscan, nray=nray)
    sw.storm_top = np.zeros((nscan, nray), dtype=np.float32)
    sw.pia = np.zeros((nscan, nray), dtype=np.float32)

    # Place the imager grid far away (tens of degrees) so nothing is within
    # COLOCATE_RADIUS_M (15 km) of any radar pixel.
    imlat = sw.lat.copy() + 40.0
    imlon = sw.lon.copy() + 40.0
    pct = np.full((nscan, nray), 250.0, dtype=np.float32)
    im = _tiny_imager(
        imlat,
        imlon,
        pct,
        sc_lat=np.full(nscan, float(imlat.max()) + 5.0, dtype=np.float32),
        sc_lon=np.full(nscan, float(imlon.mean()), dtype=np.float32),
    )

    out = colocate_pct(sw, im)
    assert np.isnan(out).all()


def test_pct_formula_via_colocate(synthetic_swath):
    """PCT = PCT_A*V - PCT_B*H propagates unchanged through co-location.

    Build the imager PCT from a hand-computed PCT_A*V - PCT_B*H over a
    coincident, ungated grid; the nearest-neighbour co-located value at each
    radar pixel must equal that hand-computed PCT.
    """
    nscan, nray = 6, 4
    sw = synthetic_swath(nscan=nscan, nray=nray)
    sw.storm_top = np.zeros((nscan, nray), dtype=np.float32)
    sw.pia = np.zeros((nscan, nray), dtype=np.float32)

    tc_v = np.full((nscan, nray), 274.1, dtype=np.float32)
    tc_h = np.full((nscan, nray), 257.2, dtype=np.float32)
    pct = (config.PCT_A * tc_v - config.PCT_B * tc_h).astype(np.float32)
    expected = 1.818 * 274.1 - 0.818 * 257.2

    im = _tiny_imager(
        sw.lat.copy(),
        sw.lon.copy(),
        pct,
        sc_lat=np.full(nscan, float(sw.lat.max()) + 5.0, dtype=np.float32),
        sc_lon=np.full(nscan, float(sw.lon.mean()), dtype=np.float32),
    )
    out = colocate_pct(sw, im)
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert np.allclose(finite, expected, atol=1e-2)


def test_colocate_pct37_uses_its_own_geolocation(synthetic_swath):
    """colocate_pct37 resamples the 37 GHz field via its OWN (lat37/lon37) grid.

    Mimics the TMI case where 37 GHz lives in a separate swath: the 89 GHz grid
    is placed far away (so colocate_pct would be all-NaN), while the 37 GHz grid
    is coincident with the radar. colocate_pct37 must use lat37/lon37 and return
    the 37 GHz samples, proving it does not fall back to the 89 GHz geolocation.
    """
    nscan, nray = 6, 4
    sw = synthetic_swath(nscan=nscan, nray=nray)
    sw.storm_top = np.zeros((nscan, nray), dtype=np.float32)
    sw.pia = np.zeros((nscan, nray), dtype=np.float32)

    # 89 GHz grid far from the radar; 37 GHz grid coincident with it.
    far_lat = sw.lat.copy() + 40.0
    far_lon = sw.lon.copy() + 40.0
    pct89 = np.full((nscan, nray), 250.0, dtype=np.float32)

    pct37 = np.full((nscan, nray), 280.0, dtype=np.float32)
    pct37[2, 1] = 199.0  # a distinctive cold 37 GHz value
    im = _tiny_imager(
        far_lat, far_lon, pct89,
        sc_lat=np.full(nscan, float(far_lat.max()) + 5.0, dtype=np.float32),
        sc_lon=np.full(nscan, float(far_lon.mean()), dtype=np.float32),
        pct37=pct37,
        lat37=sw.lat.copy(),
        lon37=sw.lon.copy(),
        sc_lat37=np.full(nscan, float(sw.lat.max()) + 5.0, dtype=np.float32),
        sc_lon37=np.full(nscan, float(sw.lon.mean()), dtype=np.float32),
    )

    # 89 GHz: nothing within radius -> all NaN.
    assert np.isnan(colocate_pct(sw, im)).all()
    # 37 GHz: coincident grid -> samples preserved at their radar pixels.
    out37 = colocate_pct37(sw, im)
    assert out37.dtype == np.float32
    assert out37[2, 1] == pytest.approx(199.0)
    assert np.any(np.isclose(out37, 280.0))
