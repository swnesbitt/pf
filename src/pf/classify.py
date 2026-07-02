"""Convective/stratiform classification and MCS labelling for a feature.

This module fills FEATURE_SCHEMA columns 36-47 for a single labeled
precipitation feature. It partitions the feature's member pixels into
convective (``rain_type == 2``) and stratiform (``rain_type == 1``) sets,
accumulates area and volumetric-rain shares for each, summarises bright-band
and freezing-level heights, and applies the radar-only MCS area test.

Rain-type codes (from :mod:`pf.readers.gpm_ku`) are ``-1`` none/missing,
``1`` stratiform, ``2`` convective, ``3`` other; only ``2`` and ``1`` are
treated as convective/stratiform here. Everything else contributes to neither
share.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from pf import config


def _mean_finite(vals: np.ndarray) -> float:
    """Return ``nanmean(vals)`` as a float, or ``NaN`` if no finite values.

    The finite-any guard avoids the numpy ``RuntimeWarning`` (and ``NaN``
    result) raised when ``nanmean`` is applied to an all-NaN/empty slice.

    Parameters
    ----------
    vals : ndarray
        Candidate values (may contain ``NaN`` or be empty).

    Returns
    -------
    float
        Mean of the finite values, else ``float("nan")``.
    """
    if vals.size and np.isfinite(vals).any():
        return float(np.nanmean(vals))
    return float("nan")


def classify_feature(
    swath: Any,
    labeled: np.ndarray,
    local_label: int,
    area_km2: float,
    volrain_total: float,
    cfg: ModuleType | None = None,
) -> dict:
    """Classify one feature into convective/stratiform shares and an MCS class.

    Parameters
    ----------
    swath : pf.swath.Swath
        The decoded orbit swath the feature was labeled from. Uses
        ``rain_type``, ``pixel_area``, ``near_sfc_rain``, ``bb_height`` and
        ``freezing_level``.
    labeled : ndarray, int32, shape (nscan, nray)
        Connected-component label image from :func:`pf.label.label_rpf`.
    local_label : int
        The label identifying this feature within ``labeled``.
    area_km2 : float
        Feature area (km^2) from :func:`pf.label.label_rpf`.
    volrain_total : float
        Total volumetric rain (mm/hr * km^2) for the member, as computed in
        :func:`pf.features.build_feature_row` (the single source of truth;
        not recomputed here). May be ``NaN``.
    cfg : module, optional
        Configuration module supplying ``MCS_AREA_KM2``; defaults to
        :mod:`pf.config`.

    Returns
    -------
    dict
        FEATURE_SCHEMA columns 36-47 as native Python scalars:
        ``conv_area_km2``, ``strat_area_km2``, ``conv_area_frac``,
        ``strat_area_frac``, ``conv_rain_frac``, ``strat_rain_frac``,
        ``volrain_conv``, ``volrain_strat``, ``mean_bb_height``,
        ``mean_freezing_level``, ``is_mcs`` and ``feature_class``.
    """
    if cfg is None:
        cfg = config

    member = labeled == local_label
    rt = swath.rain_type
    conv = member & (rt == 2)
    strat = member & (rt == 1)

    pixel_area = swath.pixel_area
    near_sfc_rain = swath.near_sfc_rain

    # --- area shares (empty mask -> nansum([]) == 0.0) -------------------
    conv_area_km2 = float(np.nansum(pixel_area[conv]))
    strat_area_km2 = float(np.nansum(pixel_area[strat]))

    if area_km2 > 0:
        conv_area_frac = float(conv_area_km2 / area_km2)
        strat_area_frac = float(strat_area_km2 / area_km2)
    else:
        conv_area_frac = float("nan")
        strat_area_frac = float("nan")

    # --- volumetric-rain shares (empty mask -> 0.0; NaN entries dropped) -
    volrain_conv = float(np.nansum(near_sfc_rain[conv] * pixel_area[conv]))
    volrain_strat = float(np.nansum(near_sfc_rain[strat] * pixel_area[strat]))

    if np.isfinite(volrain_total) and volrain_total > 0:
        conv_rain_frac = float(volrain_conv / volrain_total)
        strat_rain_frac = float(volrain_strat / volrain_total)
    else:
        conv_rain_frac = float("nan")
        strat_rain_frac = float("nan")

    # --- bright-band / freezing-level means over the member --------------
    mean_bb_height = _mean_finite(swath.bb_height[member])
    mean_freezing_level = _mean_finite(swath.freezing_level[member])

    # --- MCS test and class enum -----------------------------------------
    is_mcs = bool(float(area_km2) >= float(cfg.MCS_AREA_KM2))

    if is_mcs:
        feature_class = "MCS"
    elif conv_area_km2 > 0:
        feature_class = "sub_MCS_conv"
    elif strat_area_km2 > 0:
        feature_class = "stratiform_only"
    else:
        feature_class = "weak"

    return {
        "conv_area_km2": conv_area_km2,
        "strat_area_km2": strat_area_km2,
        "conv_area_frac": conv_area_frac,
        "strat_area_frac": strat_area_frac,
        "conv_rain_frac": conv_rain_frac,
        "strat_rain_frac": strat_rain_frac,
        "volrain_conv": volrain_conv,
        "volrain_strat": volrain_strat,
        "mean_bb_height": mean_bb_height,
        "mean_freezing_level": mean_freezing_level,
        "is_mcs": is_mcs,
        "feature_class": feature_class,
    }
