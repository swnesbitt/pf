"""Quality-controlled echo-top heights (Hirose, Okada, Kawaguchi & Takahashi
2023, *J. Atmos. Oceanic Technol.*, 40, 969-985, doi:10.1175/JTECH-D-22-0114.1).

The naive echo top -- ``max height where Z>=thresh over the feature volume`` --
is contaminated by three high-altitude artifacts the spaceborne radars suffer:

* **mirror / second-trip echoes** -- reflections of ~15 km solid precip that
  appear at ~18-22 km because the pulse-sampling window is short; GPM Ku/TRMM PR
  V07 tag the suspect bin in ``binMirrorImageL2``;
* **sidelobe surface clutter** -- surface backscatter through antenna sidelobes,
  worst at large scan angle and high altitude (outer cross-track rays);
* **isolated noise** -- a lone weak high-altitude bin with no adjacent precip.

A *real* overshooting top is embedded in a deep system: it reaches the ray's
observation ceiling (right-censored) but is NOT spatially isolated, so it is
kept (and flagged censored), never dropped.

This module removes the artifacts BEFORE taking the echo top and annotates each
feature with a flag bitfield and a right-censor indicator. The decision logic,
in order, on the achieving pixel/column:

1. **Floor** -- below :data:`config.ETH_FILTER_FLOOR_M` (15 km) nothing is
   removed (the filter is known to clip some real >30 dBZ tops at 15-17 km).
2. **Inner-swath gate** -- above the floor, high echoes on outer rays
   (outside ``[ETH_INNER_RAY_LO, ETH_INNER_RAY_HI]``) are dropped (sidelobe).
3. **Mirror truncation** -- above the floor, bins higher than the
   ``binMirrorImageL2`` altitude are dropped (GPM; TRMM leaves the flag fill).
4. **Isolated-noise peel** -- if the surviving top is above the floor and
   spatially isolated (no 8-neighbour storm top > 15 km), its above-floor bins
   are masked and the top recomputed, up to :data:`config.ETH_MAX_PEEL` times.
5. **Right-censor** -- a surviving top within :data:`config.ETH_CENSOR_BIN_TOL`
   bins of the column's highest observed bin is a lower bound, not a point.

Bins are masked geometrically (by slant-corrected height), so the same cleaning
applies consistently to the 20/30/40 dBZ tops; the flag bitfield describes the
primary 20 dBZ top used for echo-top stratification.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pf import config

THRESHOLDS: tuple[float, ...] = (20.0, 30.0, 40.0)


def _empty() -> dict:
    """Result when a feature has no member pixel reaching any threshold."""
    out: dict[str, Any] = {}
    for thr in THRESHOLDS:
        t = int(thr)
        out[f"max_ht_{t}dbz"] = float("nan")
        out[f"max_ht_{t}dbz_scan"] = -1
        out[f"max_ht_{t}dbz_ray"] = -1
    out["echotop_qc_flags"] = 0
    out["max_ht_20dbz_censored"] = False
    out["ray_obs_ceiling_m"] = float("nan")
    return out


def _is_isolated(swath, scan: int, ray: int, neighbor_m: float) -> bool:
    """True if no 8-neighbour (scan +/-1, ray +/-1) has storm top > ``neighbor_m``.

    Uses ``swath.storm_top`` (heightStormTop, m). If every neighbour is NaN the
    test cannot be assessed -> returns ``False`` (do not drop on missing data).
    """
    st = swath.storm_top
    nscan, nray = st.shape
    s0, s1 = max(0, scan - 1), min(nscan, scan + 2)
    r0, r1 = max(0, ray - 1), min(nray, ray + 2)
    win = st[s0:s1, r0:r1].copy()
    # Exclude the centre pixel itself.
    win[scan - s0, ray - r0] = np.nan
    finite = win[np.isfinite(win)]
    if finite.size == 0:
        return False
    return bool(finite.max() <= neighbor_m)


def _dd_from_top(rate_col: np.ndarray, top_bin: int, above_floor_col: np.ndarray) -> bool:
    """True if the retrieved rate is maximal at the top and decreases downward.

    The mirror-image signature: a (spurious) rate peak at the highest bin with
    smaller rates below it. Compares the top bin's rate to the finite rates in
    the rest of the above-floor column.
    """
    if not np.isfinite(rate_col[top_bin]):
        return False
    below = above_floor_col.copy()
    below[top_bin] = False
    others = rate_col[below]
    others = others[np.isfinite(others)]
    if others.size == 0:
        return False
    return bool(rate_col[top_bin] >= others.max())


def feature_echo_tops(
    swath,
    member: np.ndarray,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    cfg=config,
) -> dict:
    """QC'd echo-top heights + flags for one feature's member mask.

    Parameters
    ----------
    swath : pf.swath.Swath
        Orbit swath; uses ``dbz_3d``, ``height_3d`` (slant-corrected m MSL),
        ``rain_rate_3d``, ``bin_mirror_image``, ``storm_top``.
    member : ndarray of bool, shape (nscan, nray)
        Feature member mask.

    Returns
    -------
    dict
        ``max_ht_{20,30,40}dbz`` (m, NaN if unmet/removed) and their
        ``_scan``/``_ray`` provenance (-1 if unmet), ``echotop_qc_flags``
        (int bitfield), ``max_ht_20dbz_censored`` (bool), ``ray_obs_ceiling_m``.
    """
    scan_idx, ray_idx = np.nonzero(member)
    n = scan_idx.size
    if n == 0:
        return _empty()

    nbin = swath.dbz_3d.shape[2]
    sub_dbz = swath.dbz_3d[member]              # (n, nbin)
    sub_ht = swath.height_3d[member]            # (n, nbin)
    sub_rate = swath.rain_rate_3d[member]       # (n, nbin)

    above_floor = sub_ht > cfg.ETH_FILTER_FLOOR_M           # (n, nbin)
    outer_ray = (ray_idx < cfg.ETH_INNER_RAY_LO) | (ray_idx > cfg.ETH_INNER_RAY_HI)

    # Mirror truncation height per pixel: the altitude of the binMirrorImageL2
    # bin; bins ABOVE it are mirror-suspect. inf (no truncation) where the flag
    # is fill/out-of-range (TRMM PR, or rays without a flag).
    mir_bin = np.asarray(swath.bin_mirror_image)[scan_idx, ray_idx].astype(np.int64)
    valid_mir = (mir_bin >= 0) & (mir_bin < nbin)
    mir_height = np.full(n, np.inf, dtype=np.float64)
    if valid_mir.any():
        rows = np.arange(n)
        h_at_mir = sub_ht[rows, np.clip(mir_bin, 0, nbin - 1)]
        mir_height = np.where(valid_mir & np.isfinite(h_at_mir), h_at_mir, np.inf)
    above_mirror = sub_ht > mir_height[:, None]             # (n, nbin); inf -> all False

    at20 = np.isfinite(sub_dbz) & (sub_dbz >= 20.0)
    flags = 0
    if (at20 & above_floor & outer_ray[:, None]).any():
        flags |= cfg.ETH_FLAG_SIDELOBE_REMOVED
    if (at20 & above_floor & above_mirror).any():
        flags |= cfg.ETH_FLAG_MIRROR_REMOVED

    # Geometric keep-mask shared by all thresholds: drop above-floor bins that
    # are outer-ray (sidelobe) or above the mirror flag.
    reject = above_floor & (outer_ray[:, None] | above_mirror)
    keep = ~reject

    # --- locate the 20 dBZ top, peeling isolated-noise spikes ---------------
    top_i = top_bin = -1
    top_ht = float("nan")
    for _ in range(cfg.ETH_MAX_PEEL + 1):
        valid = at20 & keep
        if not valid.any():
            break
        masked_ht = np.where(valid, sub_ht, np.nan)
        flat = int(np.nanargmax(masked_ht))
        i, b = divmod(flat, nbin)
        ht = float(masked_ht.flat[flat])
        if ht > cfg.ETH_FILTER_FLOOR_M and _is_isolated(
            swath, int(scan_idx[i]), int(ray_idx[i]), cfg.ETH_OVERSHOOT_NEIGHBOR_M
        ):
            # Isolated high echo with no deep neighbour -> noise/residual
            # artifact. Peel this pixel's above-floor bins and retry.
            keep[i, above_floor[i]] = False
            flags |= cfg.ETH_FLAG_NOISE_PEELED
            continue
        top_i, top_bin, top_ht = i, b, ht
        break

    out = _empty()
    out["echotop_qc_flags"] = flags
    if top_i < 0:
        return out  # no surviving 20 dBZ top

    # Per-mission instrument observation ceiling (sensitivity limit), NOT the
    # column's highest echo -- so a shallow storm that simply ends is never
    # falsely flagged truncated.
    ceiling_m = cfg.ETH_OBS_CEILING_M.get(
        str(swath.mission).upper(), cfg.ETH_OBS_CEILING_DEFAULT_M
    )
    out["ray_obs_ceiling_m"] = float(ceiling_m)

    # The DD-from-top / isolation / truncation / censor flags are only
    # meaningful for HIGH tops (above the floor); a clean low feature carries no
    # high-altitude flags.
    if top_ht > cfg.ETH_FILTER_FLOOR_M:
        isolated = _is_isolated(swath, int(scan_idx[top_i]), int(ray_idx[top_i]),
                                cfg.ETH_OVERSHOOT_NEIGHBOR_M)
        dd = _dd_from_top(sub_rate[top_i], top_bin, above_floor[top_i])
        # Right-censored if the top reaches within tol of the sensitivity
        # ceiling (the radar could not have seen a higher echo).
        bin_m = cfg.GPM_RANGE_BIN_SIZE_M
        truncated = top_ht >= ceiling_m - cfg.ETH_CENSOR_BIN_TOL * bin_m

        if dd:
            flags |= cfg.ETH_FLAG_DD_FROM_TOP
        if isolated:
            flags |= cfg.ETH_FLAG_ISOLATED
        if truncated:
            flags |= cfg.ETH_FLAG_TOP_TRUNCATED
            flags |= cfg.ETH_FLAG_CENSORED
            if dd and not isolated:
                flags |= cfg.ETH_FLAG_OVERSHOOT_REAL
        out["max_ht_20dbz_censored"] = bool(truncated)

    out["echotop_qc_flags"] = flags

    # --- all thresholds through the SAME cleaned keep-mask ------------------
    for thr in thresholds:
        t = int(thr)
        valid = np.isfinite(sub_dbz) & (sub_dbz >= thr) & keep
        if not valid.any():
            continue
        masked_ht = np.where(valid, sub_ht, np.nan)
        flat = int(np.nanargmax(masked_ht))
        i, _b = divmod(flat, nbin)
        out[f"max_ht_{t}dbz"] = float(masked_ht.flat[flat])
        out[f"max_ht_{t}dbz_scan"] = int(scan_idx[i])
        out[f"max_ht_{t}dbz_ray"] = int(ray_idx[i])
    return out


def pixel_echo_tops(
    swath,
    mask: np.ndarray,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    cfg=config,
) -> dict:
    """Per-pixel QC'd echo-top heights (m MSL) for the pixels in ``mask``.

    Unlike :func:`feature_echo_tops` (one scalar per feature volume), this
    returns a height for EACH masked pixel column independently — the highest
    bin in that column where Z >= threshold, after the *geometric* Hirose-2023
    gates (floor / outer-ray sidelobe / above-mirror truncation). The
    feature-context tests (isolated-noise peel, DD-from-top, censor) are omitted
    because a single pixel has no feature neighbourhood; the geometric gates
    remove the dominant high-altitude artifacts and match the ``keep`` mask used
    by :func:`feature_echo_tops`.

    Parameters
    ----------
    swath : pf.swath.Swath
        Uses ``dbz_3d``, ``height_3d`` (slant-corrected m MSL), ``bin_mirror_image``.
    mask : ndarray of bool, shape (nscan, nray)
        Pixels to compute (e.g. convective pixels). Others are left NaN.

    Returns
    -------
    dict[int, ndarray]
        ``{20: arr, 30: arr, 40: arr}`` each ``(nscan, nray)`` float32, with the
        echo-top height (m) where the pixel is in ``mask`` and a bin reaches the
        threshold, else NaN.
    """
    nscan, nray = swath.lat.shape
    nbin = swath.dbz_3d.shape[2]
    out = {int(t): np.full((nscan, nray), np.nan, dtype=np.float32) for t in thresholds}

    scan_idx, ray_idx = np.nonzero(mask)
    n = scan_idx.size
    if n == 0:
        return out

    sub_dbz = swath.dbz_3d[mask]          # (n, nbin)
    sub_ht = swath.height_3d[mask]        # (n, nbin)
    above_floor = sub_ht > cfg.ETH_FILTER_FLOOR_M
    outer_ray = (ray_idx < cfg.ETH_INNER_RAY_LO) | (ray_idx > cfg.ETH_INNER_RAY_HI)

    # Mirror truncation height per pixel (altitude of binMirrorImageL2); bins
    # above it are mirror-suspect. inf where the flag is fill/out-of-range.
    mir_bin = np.asarray(swath.bin_mirror_image)[scan_idx, ray_idx].astype(np.int64)
    valid_mir = (mir_bin >= 0) & (mir_bin < nbin)
    mir_height = np.full(n, np.inf, dtype=np.float64)
    if valid_mir.any():
        rows = np.arange(n)
        h_at_mir = sub_ht[rows, np.clip(mir_bin, 0, nbin - 1)]
        mir_height = np.where(valid_mir & np.isfinite(h_at_mir), h_at_mir, np.inf)
    above_mirror = sub_ht > mir_height[:, None]

    # Same geometric keep-mask as feature_echo_tops: drop above-floor bins that
    # are outer-ray (sidelobe) or above the mirror flag.
    keep = ~(above_floor & (outer_ray[:, None] | above_mirror))

    for thr in thresholds:
        valid = np.isfinite(sub_dbz) & (sub_dbz >= thr) & keep   # (n, nbin)
        masked_ht = np.where(valid, sub_ht, np.nan)
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
                tops = np.nanmax(masked_ht, axis=1)              # (n,), NaN if none
        out[int(thr)][scan_idx, ray_idx] = tops.astype(np.float32)
    return out


__all__ = ["feature_echo_tops", "pixel_echo_tops", "THRESHOLDS"]
