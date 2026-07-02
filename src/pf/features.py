"""Per-feature attribute computation and the frozen Parquet schemas.

This module is the authoritative source of the precipitation-feature (PF)
table layout. :data:`FEATURE_SCHEMA` declares the **frozen 47-column** schema
(see the Phase-1 interface spec, section 9). Phases 2-4 only *populate*
placeholder columns 35-47 — the column set, order, and dtypes never change, so
the on-disk Parquet never has to be migrated.

:func:`build_feature_row` computes the Phase-1 columns (1-34) for a single
labeled feature and returns the later-phase columns (35-47) as ``NaN``/``None``.

Surface-type mapping
--------------------
GPM ``FS/PRE/landSurfaceType`` is a small integer code. Following the GPM
convention the hundreds digit encodes the broad surface class::

    code in   0.. 99  -> ocean
    code in 100..199  -> land
    code in 200..299  -> coast / inland water
    code in 300..399  -> (treated as coast for fraction purposes)

For a feature the land/ocean/coast *fractions* are the share of member pixels
falling in each class (ignoring sentinel ``-1`` pixels). ``surface_flag`` is the
**dominant** class for the feature: ``0`` ocean, ``1`` land, ``2`` coast. If the
feature's ``surface_type`` is entirely sentinel (``-1``, e.g. the field was
absent in the granule) the three fractions are ``NaN`` and ``surface_flag`` is
``-1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from skimage.measure import regionprops

from pf import classify, echotop_qc, feature_id, geometry

# ---------------------------------------------------------------------------
# Frozen 47-column feature schema (order + dtype are authoritative)
# ---------------------------------------------------------------------------
#: Authoritative pyarrow schema for the feature catalog. EXACTLY 55 columns:
#: the 47 spec columns, the derived ``is_thin`` shape flag, the 6
#: ``max_ht_*_scan/ray`` provenance indices for the slant-corrected echo tops,
#: and ``min_pct_37`` (co-located 37 GHz PCT minimum).
#: Imported by
#: :mod:`pf.catalog` and any consumer that needs to validate / cast feature
#: DataFrames.
FEATURE_SCHEMA: pa.Schema = pa.schema(
    [
        # --- 1-7 identity / size (Phase 1) ---------------------------------
        pa.field("feature_id", pa.int64()),
        pa.field("mission", pa.string()),          # partition key
        pa.field("orbit", pa.int32()),             # partition key
        pa.field("local_label", pa.int32()),
        pa.field("time", pa.timestamp("us")),
        pa.field("npixels", pa.int32()),
        pa.field("area_km2", pa.float32()),
        # --- 8-9 centroid --------------------------------------------------
        pa.field("centroid_lat", pa.float32()),
        pa.field("centroid_lon", pa.float32()),
        # --- 10-13 bbox in index space ------------------------------------
        pa.field("bbox_scan_min", pa.int32()),
        pa.field("bbox_scan_max", pa.int32()),
        pa.field("bbox_ray_min", pa.int32()),
        pa.field("bbox_ray_max", pa.int32()),
        # --- 14-17 bbox in lat/lon ----------------------------------------
        pa.field("bbox_lat_min", pa.float32()),
        pa.field("bbox_lat_max", pa.float32()),
        pa.field("bbox_lon_min", pa.float32()),
        pa.field("bbox_lon_max", pa.float32()),
        # --- 18-21 surface ------------------------------------------------
        pa.field("frac_land", pa.float32()),
        pa.field("frac_ocean", pa.float32()),
        pa.field("frac_coast", pa.float32()),
        pa.field("surface_flag", pa.int8()),
        # --- 22-24 near-surface radar -------------------------------------
        pa.field("max_near_sfc_dbz", pa.float32()),
        pa.field("max_near_sfc_rain", pa.float32()),
        pa.field("mean_near_sfc_rain", pa.float32()),
        # --- 25-27 echo-top heights (m MSL, slant-corrected) --------------
        pa.field("max_ht_20dbz", pa.float32()),
        pa.field("max_ht_30dbz", pa.float32()),
        pa.field("max_ht_40dbz", pa.float32()),
        # scan/ray index of the pixel achieving each echo top (provenance:
        # the cross-track ray pins the off-nadir zenith used to slant-correct
        # the height). -1 when the threshold is unmet.
        pa.field("max_ht_20dbz_scan", pa.int32()),
        pa.field("max_ht_20dbz_ray", pa.int32()),
        pa.field("max_ht_30dbz_scan", pa.int32()),
        pa.field("max_ht_30dbz_ray", pa.int32()),
        pa.field("max_ht_40dbz_scan", pa.int32()),
        pa.field("max_ht_40dbz_ray", pa.int32()),
        # echo-top quality control (pf.echotop_qc, Hirose et al. 2023): the
        # max_ht_* values above are ALREADY QC'd (mirror/sidelobe/noise removed
        # before the max). These annotate the primary 20 dBZ top:
        #   echotop_qc_flags   -- int16 bitfield (config.ETH_FLAG_*)
        #   max_ht_20dbz_censored -- top reached the instrument ceiling -> the
        #       value is a right-censored LOWER BOUND, not a point value
        #   ray_obs_ceiling_m  -- per-mission observation ceiling (m MSL)
        pa.field("echotop_qc_flags", pa.int16()),
        pa.field("max_ht_20dbz_censored", pa.bool_()),
        pa.field("ray_obs_ceiling_m", pa.float32()),
        # --- 28 volumetric rain -------------------------------------------
        pa.field("volrain_total", pa.float32()),
        # --- 29-33 shape ---------------------------------------------------
        pa.field("major_axis_km", pa.float32()),
        pa.field("minor_axis_km", pa.float32()),
        pa.field("orientation_deg", pa.float32()),
        pa.field("aspect_ratio", pa.float32()),
        pa.field("eccentricity", pa.float32()),
        # is_thin: grouped with the shape/morphology columns (after the axis
        # descriptors). True for degenerate/thin features (collinear,
        # single-line, or single-pixel) with no meaningful 2-D shape; their
        # aspect_ratio is the clamp sentinel.
        pa.field("is_thin", pa.bool_()),
        # --- 34 edge flag --------------------------------------------------
        pa.field("edge", pa.bool_()),
        # --- 35-47 later-phase placeholders (written NaN/null in Phase 1) --
        pa.field("min_pct_85_89", pa.float32()),     # Phase 3
        pa.field("min_pct_37", pa.float32()),        # Phase 3 (37 GHz)
        pa.field("conv_area_km2", pa.float32()),     # Phase 2
        pa.field("strat_area_km2", pa.float32()),
        pa.field("conv_area_frac", pa.float32()),
        pa.field("strat_area_frac", pa.float32()),
        pa.field("conv_rain_frac", pa.float32()),
        pa.field("strat_rain_frac", pa.float32()),
        pa.field("volrain_conv", pa.float32()),
        pa.field("volrain_strat", pa.float32()),
        pa.field("mean_bb_height", pa.float32()),
        pa.field("mean_freezing_level", pa.float32()),
        pa.field("is_mcs", pa.bool_()),              # Phase 2 (nullable)
        pa.field("feature_class", pa.string()),      # Phase 2
    ]
)

# ---------------------------------------------------------------------------
# Pixel (member) schema — declared now, populated from Phase 2 onward.
# ---------------------------------------------------------------------------
#: Schema for the per-pixel (member) table. Phase-1 may leave the pixel table
#: empty/unwritten, but the schema must exist so that consumers and the
#: catalog writer have a stable target.
PIXEL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("feature_id", pa.int64()),
        pa.field("mission", pa.string()),
        pa.field("orbit", pa.int32()),
        pa.field("scan", pa.int32()),
        pa.field("ray", pa.int16()),
        pa.field("lat", pa.float32()),
        pa.field("lon", pa.float32()),
        pa.field("near_sfc_dbz", pa.float32()),
        pa.field("near_sfc_rain", pa.float32()),
        pa.field("pct_85_89", pa.float32()),
        pa.field("pct_37", pa.float32()),
        pa.field("rain_type", pa.int8()),
        pa.field("pixel_area_km2", pa.float32()),
        pa.field("bb_height", pa.float32()),
    ]
)


# ---------------------------------------------------------------------------
# Surface-type classification
# ---------------------------------------------------------------------------
# Broad-class codes used for surface_flag and the frac_* columns.
_SURFACE_OCEAN = 0
_SURFACE_LAND = 1
_SURFACE_COAST = 2


def _surface_class(codes: np.ndarray) -> np.ndarray:
    """Map raw GPM ``landSurfaceType`` codes to broad classes.

    Parameters
    ----------
    codes : ndarray of int
        Raw ``FS/PRE/landSurfaceType`` values for the member pixels. Sentinel
        ``-1`` marks pixels with no surface information.

    Returns
    -------
    ndarray of int
        Per-pixel broad class: ``0`` ocean, ``1`` land, ``2`` coast, ``-1``
        unknown (sentinel passes through).
    """
    codes = np.asarray(codes)
    out = np.full(codes.shape, -1, dtype=np.int64)
    hundreds = codes // 100
    out[codes == 0] = _SURFACE_OCEAN
    out[(codes > 0) & (hundreds == 0)] = _SURFACE_OCEAN  # 1..99 ocean
    out[hundreds == 1] = _SURFACE_LAND                   # 100..199 land
    out[hundreds >= 2] = _SURFACE_COAST                  # 200+ coast/inland
    out[codes < 0] = -1                                  # sentinels stay -1
    return out


def _surface_fractions(
    surface_type: np.ndarray, member: np.ndarray
) -> tuple[float, float, float, int]:
    """Compute land/ocean/coast fractions and the dominant surface flag.

    Parameters
    ----------
    surface_type : ndarray, int, shape (nscan, nray)
        Raw surface-type codes for the whole swath.
    member : ndarray of bool, shape (nscan, nray)
        Member mask for the feature.

    Returns
    -------
    tuple of (float, float, float, int)
        ``(frac_land, frac_ocean, frac_coast, surface_flag)``. If every member
        pixel is sentinel ``-1`` the fractions are ``NaN`` and the flag is
        ``-1``.
    """
    codes = np.asarray(surface_type)[member]
    classes = _surface_class(codes)
    valid = classes >= 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return (float("nan"), float("nan"), float("nan"), -1)

    n_ocean = int((classes == _SURFACE_OCEAN).sum())
    n_land = int((classes == _SURFACE_LAND).sum())
    n_coast = int((classes == _SURFACE_COAST).sum())

    frac_ocean = n_ocean / n_valid
    frac_land = n_land / n_valid
    frac_coast = n_coast / n_valid

    counts = (n_ocean, n_land, n_coast)
    flag = int(np.argmax(counts))  # 0 ocean, 1 land, 2 coast
    return (frac_land, frac_ocean, frac_coast, flag)


def _max_ht_at_threshold(
    member: np.ndarray, dbz_3d: np.ndarray, height_3d: np.ndarray, thresh: float
) -> tuple[float, int, int]:
    """Maximum bin height (m MSL) where reflectivity meets ``thresh``, and where.

    Parameters
    ----------
    member : ndarray of bool, shape (nscan, nray)
        Member mask for the feature.
    dbz_3d : ndarray, float32, shape (nscan, nray, nbin)
        Reflectivity cube.
    height_3d : ndarray, float32, shape (nscan, nray, nbin)
        Bin-center height above MSL (m), already slant-corrected.
    thresh : float
        Reflectivity threshold (dBZ).

    Returns
    -------
    (height, scan, ray) : (float, int, int)
        ``nanmax`` of ``height_3d`` over the member volume where
        ``dbz_3d >= thresh``, plus the along-track ``scan`` and cross-track
        ``ray`` index of the pixel achieving it (the ``ray`` pins the off-nadir
        zenith used for the slant correction, so the height is auditable). If no
        member bin reaches the threshold, returns ``(NaN, -1, -1)``.
    """
    # Boolean indexing preserves C order, so row i of sub_* maps to the i-th
    # member pixel given by np.nonzero(member) — used to recover scan/ray.
    scan_idx, ray_idx = np.nonzero(member)        # each (n_member,)
    sub_dbz = dbz_3d[member]                       # (n_member, nbin)
    sub_ht = height_3d[member]                     # (n_member, nbin)
    at = np.isfinite(sub_dbz) & (sub_dbz >= thresh)
    if not at.any():
        return float("nan"), -1, -1
    masked = np.where(at, sub_ht, np.nan)          # (n_member, nbin)
    flat = int(np.nanargmax(masked))               # ignores NaNs
    i, _b = divmod(flat, masked.shape[1])
    return float(masked.flat[flat]), int(scan_idx[i]), int(ray_idx[i])


def build_feature_row(
    swath: Any,
    labeled: np.ndarray,
    local_label: int,
    area_km2: float,
    edge: bool,
) -> dict:
    """Compute the feature-table row for one labeled precipitation feature.

    Phase-1 computes columns 1-34; columns 35-47 are returned as
    ``float('nan')`` / ``None`` so the frozen :data:`FEATURE_SCHEMA` is always
    fully populated.

    Parameters
    ----------
    swath : pf.swath.Swath
        The decoded orbit swath the feature was labeled from.
    labeled : ndarray, int32, shape (nscan, nray)
        Connected-component label image from :func:`pf.label.label_rpf`.
    local_label : int
        The label identifying this feature within ``labeled``.
    area_km2 : float
        Feature area (km^2) as returned alongside ``local_label`` by
        :func:`pf.label.label_rpf`.
    edge : bool
        Whether the feature touches a swath edge
        (:func:`pf.label.touches_edge`).

    Returns
    -------
    dict
        One row keyed by the field names of :data:`FEATURE_SCHEMA`.
    """
    member = labeled == local_label
    npixels = int(member.sum())

    scan_idx, ray_idx = np.nonzero(member)

    # --- geolocation of member pixels ------------------------------------
    lat = swath.lat
    lon = swath.lon
    member_lat = lat[member]
    member_lon = lon[member]

    # --- time: mean scan time of member pixels ---------------------------
    # swath.time is per-scan (nscan,); average over the scans the member spans.
    scan_times = swath.time[scan_idx]
    finite_t = scan_times[~np.isnat(scan_times)] if scan_times.size else scan_times
    if finite_t.size:
        # Mean of datetime64 via int64 view (ns since epoch), back to datetime.
        mean_ns = finite_t.view("i8").mean()
        feature_time = np.datetime64(int(round(mean_ns)), "ns")
    else:
        feature_time = np.datetime64("NaT", "ns")

    # --- centroid (area-weighted) ----------------------------------------
    weights = swath.pixel_area
    centroid_lat, centroid_lon = geometry.area_weighted_centroid(
        np.where(member, lat, np.nan),
        np.where(member, lon, np.nan),
        np.where(member, weights, 0.0),
    )

    # --- bounding boxes ---------------------------------------------------
    bbox_scan_min = int(scan_idx.min())
    bbox_scan_max = int(scan_idx.max())
    bbox_ray_min = int(ray_idx.min())
    bbox_ray_max = int(ray_idx.max())

    finite_lat = member_lat[np.isfinite(member_lat)]
    finite_lon = member_lon[np.isfinite(member_lon)]
    bbox_lat_min = float(finite_lat.min()) if finite_lat.size else float("nan")
    bbox_lat_max = float(finite_lat.max()) if finite_lat.size else float("nan")
    # Seam-aware longitude extent: unwrap member lons about the (antimeridian-
    # safe) centroid so a feature straddling +/-180 deg is not reported as ~360
    # deg wide, then wrap the edges back into [-180, 180]. NOTE: for a feature
    # that crosses the +/-180 deg seam this yields bbox_lon_min > bbox_lon_max,
    # the standard antimeridian bounding-box convention (min>max => seam cross).
    if finite_lon.size and np.isfinite(centroid_lon):
        dlon = ((finite_lon - centroid_lon + 180.0) % 360.0) - 180.0
        lon_lo = centroid_lon + dlon.min()
        lon_hi = centroid_lon + dlon.max()
        bbox_lon_min = float(((lon_lo + 180.0) % 360.0) - 180.0)
        bbox_lon_max = float(((lon_hi + 180.0) % 360.0) - 180.0)
    else:
        bbox_lon_min = float("nan")
        bbox_lon_max = float("nan")

    # --- surface fractions / flag ----------------------------------------
    frac_land, frac_ocean, frac_coast, surface_flag = _surface_fractions(
        swath.surface_type, member
    )

    # --- near-surface radar statistics -----------------------------------
    member_dbz = swath.near_sfc_dbz[member]
    member_rain = swath.near_sfc_rain[member]

    with np.errstate(invalid="ignore"):
        max_near_sfc_dbz = (
            float(np.nanmax(member_dbz))
            if np.isfinite(member_dbz).any()
            else float("nan")
        )
        max_near_sfc_rain = (
            float(np.nanmax(member_rain))
            if np.isfinite(member_rain).any()
            else float("nan")
        )
        mean_near_sfc_rain = (
            float(np.nanmean(member_rain))
            if np.isfinite(member_rain).any()
            else float("nan")
        )

    # --- echo-top heights: QC'd (mirror/sidelobe/noise removed) + flags ----
    # pf.echotop_qc replaces the raw nanmax: it masks mirror/second-trip echoes
    # (binMirrorImageL2), sidelobe clutter (outer rays), and isolated noise
    # above a 15 km floor BEFORE taking the max, and right-censors tops that
    # reach the instrument ceiling. See Hirose et al. (2023).
    eth = echotop_qc.feature_echo_tops(swath, member)
    max_ht_20 = eth["max_ht_20dbz"]; ht20_scan = eth["max_ht_20dbz_scan"]; ht20_ray = eth["max_ht_20dbz_ray"]
    max_ht_30 = eth["max_ht_30dbz"]; ht30_scan = eth["max_ht_30dbz_scan"]; ht30_ray = eth["max_ht_30dbz_ray"]
    max_ht_40 = eth["max_ht_40dbz"]; ht40_scan = eth["max_ht_40dbz_scan"]; ht40_ray = eth["max_ht_40dbz_ray"]

    # --- volumetric rain (rain * pixel area, summed) ---------------------
    member_area = swath.pixel_area[member]
    rain_vol = member_rain * member_area
    volrain_total = (
        float(np.nansum(rain_vol)) if np.isfinite(rain_vol).any() else float("nan")
    )

    # --- shape: PCA on member lat/lon ------------------------------------
    major_km, minor_km, orientation_deg, aspect_ratio = geometry.pca_axes(
        member_lat, member_lon
    )

    # --- eccentricity from regionprops on the binary member mask ---------
    eccentricity = _member_eccentricity(member)

    # --- thin / degenerate shape flag ------------------------------------
    # True for degenerate/thin features (collinear, single-line, or
    # single-pixel) with no meaningful 2-D shape; their aspect_ratio is the
    # clamp sentinel. The thresholds mirror geometry.pca_axes' internal
    # _EPS_KM (minor-axis floor) and _ASPECT_MAX (aspect clamp); kept in sync
    # by value here since geometry defines them locally in pca_axes.
    _MINOR_FLOOR = 1e-3   # geometry.pca_axes _EPS_KM (km)
    _ASPECT_MAX = 1e4     # geometry.pca_axes _ASPECT_MAX
    is_thin = bool(
        minor_km <= _MINOR_FLOOR * (1.0 + 1e-9)  # at/below the minor-axis floor
        or aspect_ratio >= _ASPECT_MAX           # aspect at the clamp sentinel
        or npixels <= 1                          # single pixel -> no 2-D extent
    )

    # --- minimum co-located 89 GHz PCT (col 35, Phase 3) -----------------
    # NaN everywhere if no imager was co-located (graceful no-imager path).
    member_pct = swath.pct_85_89[member]
    min_pct_85_89 = (
        float(np.nanmin(member_pct))
        if np.isfinite(member_pct).any()
        else float("nan")
    )

    # --- minimum co-located 37 GHz PCT (Phase 3) -------------------------
    member_pct37 = swath.pct_37[member]
    min_pct_37 = (
        float(np.nanmin(member_pct37))
        if np.isfinite(member_pct37).any()
        else float("nan")
    )

    # --- feature id -------------------------------------------------------
    fid = feature_id.encode(swath.mission, swath.orbit, local_label)

    # --- convective/stratiform classification (cols 36-47) ---------------
    # Pass the already-computed volrain_total (single source of truth) so the
    # rain fractions are consistent with col 28.
    class_cols = classify.classify_feature(
        swath, labeled, local_label, float(area_km2), float(volrain_total)
    )

    row = {
        # 1-7
        "feature_id": int(fid),
        "mission": str(swath.mission),
        "orbit": int(swath.orbit),
        "local_label": int(local_label),
        "time": feature_time,
        "npixels": npixels,
        "area_km2": float(area_km2),
        # 8-9
        "centroid_lat": float(centroid_lat),
        "centroid_lon": float(centroid_lon),
        # 10-13
        "bbox_scan_min": bbox_scan_min,
        "bbox_scan_max": bbox_scan_max,
        "bbox_ray_min": bbox_ray_min,
        "bbox_ray_max": bbox_ray_max,
        # 14-17
        "bbox_lat_min": bbox_lat_min,
        "bbox_lat_max": bbox_lat_max,
        "bbox_lon_min": bbox_lon_min,
        "bbox_lon_max": bbox_lon_max,
        # 18-21
        "frac_land": frac_land,
        "frac_ocean": frac_ocean,
        "frac_coast": frac_coast,
        "surface_flag": surface_flag,
        # 22-24
        "max_near_sfc_dbz": max_near_sfc_dbz,
        "max_near_sfc_rain": max_near_sfc_rain,
        "mean_near_sfc_rain": mean_near_sfc_rain,
        # 25-27
        "max_ht_20dbz": max_ht_20,
        "max_ht_30dbz": max_ht_30,
        "max_ht_40dbz": max_ht_40,
        "max_ht_20dbz_scan": ht20_scan,
        "max_ht_20dbz_ray": ht20_ray,
        "max_ht_30dbz_scan": ht30_scan,
        "max_ht_30dbz_ray": ht30_ray,
        "max_ht_40dbz_scan": ht40_scan,
        "max_ht_40dbz_ray": ht40_ray,
        # echo-top QC annotations
        "echotop_qc_flags": int(eth["echotop_qc_flags"]),
        "max_ht_20dbz_censored": bool(eth["max_ht_20dbz_censored"]),
        "ray_obs_ceiling_m": float(eth["ray_obs_ceiling_m"]),
        # 28
        "volrain_total": volrain_total,
        # 29-33
        "major_axis_km": float(major_km),
        "minor_axis_km": float(minor_km),
        "orientation_deg": float(orientation_deg),
        "aspect_ratio": float(aspect_ratio),
        "eccentricity": float(eccentricity),
        "is_thin": is_thin,
        # 34
        "edge": bool(edge),
        # 35 minimum co-located 89 GHz PCT (Phase 3)
        "min_pct_85_89": min_pct_85_89,
        # minimum co-located 37 GHz PCT (Phase 3)
        "min_pct_37": min_pct_37,
        # 36-47 convective/stratiform classification (Phase 2)
        "conv_area_km2": class_cols["conv_area_km2"],
        "strat_area_km2": class_cols["strat_area_km2"],
        "conv_area_frac": class_cols["conv_area_frac"],
        "strat_area_frac": class_cols["strat_area_frac"],
        "conv_rain_frac": class_cols["conv_rain_frac"],
        "strat_rain_frac": class_cols["strat_rain_frac"],
        "volrain_conv": class_cols["volrain_conv"],
        "volrain_strat": class_cols["volrain_strat"],
        "mean_bb_height": class_cols["mean_bb_height"],
        "mean_freezing_level": class_cols["mean_freezing_level"],
        "is_mcs": class_cols["is_mcs"],
        "feature_class": class_cols["feature_class"],
    }

    return row


def _member_eccentricity(member: np.ndarray) -> float:
    """Eccentricity of the binary member mask via ``skimage.regionprops``.

    Parameters
    ----------
    member : ndarray of bool, shape (nscan, nray)
        Member mask for the feature.

    Returns
    -------
    float
        Region eccentricity in ``[0, 1]``; ``0.0`` for single-pixel /
        degenerate regions where regionprops cannot define an ellipse.
    """
    mask = member.astype(np.int32)
    props = regionprops(mask)
    if not props:
        return 0.0
    ecc = props[0].eccentricity
    if ecc is None or not np.isfinite(ecc):
        return 0.0
    return float(ecc)
