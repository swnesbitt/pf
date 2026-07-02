"""Project-wide constants for the precipitation-feature (PF) pipeline.

Pure constants module — no runtime logic, stdlib typing only. Functions
throughout the package take thresholds as parameters defaulting to these
values so callers can override them without editing this file.
"""

from __future__ import annotations

import os

# --- Feature detection thresholds ----------------------------------------
DBZ_THRESHOLD: float = 12.0          # default/fallback near-surface reflectivity
                                     # gate (dBZ) = lowest per-mission noise floor;
                                     # see DBZ_THRESHOLD_BY_MISSION for the actual
                                     # per-mission floors used in processing.
# Per-mission near-surface reflectivity floor = each radar's min detectable Z
# (above noise). GPM Ku ~12.6 dBZ, TRMM PR ~16.3 dBZ observed in V07 granules.
DBZ_THRESHOLD_BY_MISSION: dict[str, float] = {"GPM": 12.0, "TRMM": 16.0}
MIN_AREA_KM2: float = 0.0            # no area floor; the binding minimum is now
                                     # MIN_PIXELS (label still references this).
MIN_PIXELS: int = 1                  # minimum pixels per retained feature
CONNECTIVITY: int = 2                # 8-connectivity for skimage.measure.label
MCS_AREA_KM2: float = 2000.0         # Phase-2 MCS area threshold; defined now

# --- Storage --------------------------------------------------------------
PF_ROOT: str = "/data/scratch/a/snesbitt/pf_db"

# Transient per-orbit download cache. Each worker downloads its granules into a
# unique ``pf_<mission>_<orbit>`` subdir here and removes it when done, so the
# total cache stays bounded by (#workers x per-orbit granule size). Defaults to
# tmpfs (``/dev/shm``); override via ``PF_DOWNLOAD_DIR`` for disk-backed scratch.
DOWNLOAD_DIR: str = os.environ.get("PF_DOWNLOAD_DIR", "/dev/shm")

# --- Mission codes --------------------------------------------------------
MISSION_CODE: dict[str, int] = {"TRMM": 1, "GPM": 2}
MISSION_NAME: dict[int, str] = {1: "TRMM", 2: "GPM"}

# --- NASA short-name mapping ----------------------------------------------
SHORT_NAMES: dict[str, str] = {
    "GPM_KU": "GPM_2ADPR",          # dual-frequency DPR (Ku channel for reflectivity,
                                    # dual-freq DSD/epsilon retrieval); was GPM_2AKu
    "GPM_GMI": "GPM_1CGPMGMI",
    "TRMM_PR": "GPM_2APR",
    "TRMM_TMI": "GPM_1CTRMMTMI",
}

# --- Preferred data version per product -----------------------------------
# Preferred data version per product (granule filename has .V07A/.V08A). Prefer the
# newest reprocessing where available; search.prefer_version falls back gracefully
# when the preferred version has no granules for a given time (e.g. V08 DPR pre-2014-rollout).
PRODUCT_VERSION: dict[str, str] = {
    # GPM radar pinned to V07 for a UNIFORM climatology: V07 is complete across the
    # whole record (2014-03..2026-02); V08 reprocessing only covers 2014-03..2015-07
    # and 2026-03+, so preferring V08 under-samples the partial 2015-08..2018-09 band.
    # prefer_version falls back per-orbit to V08 where V07 is absent (the 2026-03+ tail).
    "GPM_2AKu": "07",
    "GPM_2ADPR": "07",
    "GPM_2APR": "07",        # TRMM PR: V07 only
    "GPM_1CGPMGMI": "08",
    "GPM_1CTRMMTMI": "08",
}
DEFAULT_PRODUCT_VERSION: str = "07"

# --- GPM 2A-DPR FS swath geometry ----------------------------------------
GPM_SWATH: str = "FS"
GPM_RANGE_BIN_SIZE_M: float = 125.0  # constant range-bin spacing (m)
GPM_N_RANGE_BINS: int = 176          # vertical bins in the FS 3-D cube

# --- Physical constants ---------------------------------------------------
EARTH_RADIUS_KM: float = 6371.0

# --- GMI 89 GHz polarization-corrected temperature (PCT) ------------------
# PCT89 = PCT_A * Tc[89V] - PCT_B * Tc[89H]; S1/Tc channel order places 89V at
# index 7 and 89H at index 8 (confirmed by background-Tb 89V > 89H).
PCT_A: float = 1.818
PCT_B: float = 0.818
PCT89_V_IDX: int = 7
PCT89_H_IDX: int = 8

# --- 37 GHz polarization-corrected temperature (PCT37) --------------------
# PCT37 = PCT37_A * Tc[37V] - PCT37_B * Tc[37H]  (Cecil 2009 coefficients).
# GMI 36.5 GHz V/H share the S1 swath at idx 5/6 (same geolocation as 89 GHz);
# TMI 37 GHz V/H live in the SEPARATE S2 swath at idx 3/4 (own geolocation).
PCT37_A: float = 2.15
PCT37_B: float = 1.15

# --- Imager co-location / parallax correction -----------------------------
COLOCATE_RADIUS_M: float = 15000.0    # nearest-neighbour search radius (m)
PARALLAX_STORMTOP_M: float = 5000.0   # apply parallax shift above this storm top (m)
PARALLAX_PIA_DBZ: float = 0.4         # ...and above this Ku piaFinal (dBZ)

# --- Fill sentinels present in raw HDF5 / encoded stores ------------------
FILL_SENTINELS: tuple[float, ...] = (-9999.9, -9999.0, -9999, -99.0, -1111.1, -1111)

# Geolocation (lat/lon, spacecraft nadir) must NOT mask coordinate-valued
# sentinels. -99.0 is a VALID longitude (central Mexico / US Great Plains), so
# masking it (decode_fill uses atol=0.05) silently deleted every pixel in the
# band lon=-99.0+/-0.05, carving a spurious empty meridian at 99 deg W. Only the
# extreme true-fill values belong on geolocation arrays.
GEO_FILL_SENTINELS: tuple[float, ...] = (-9999.9, -9999.0, -9999)

# --- feature_id packing factors ------------------------------------------
ID_ORBIT_MULT: int = 100_000              # 1e5
ID_MISSION_MULT: int = 10_000_000_000_000  # 1e13

# --- Gridded climatology (0.05 deg) --------------------------------------
# Shared by the per-orbit VIEWS byproduct (pf.views) and the post-hoc gridding
# tool (pf.grid). The grid is a global 0.05 deg cell-edge lattice; bins are the
# floor of (coord - origin)/GRID_DEG.
GRID_DEG: float = 0.05
GRID_LAT_MIN: float = -90.0               # southern cell edge (origin)
GRID_LON_MIN: float = -180.0              # western cell edge (origin)
GRID_N_LAT: int = 3600                    # 180 / 0.05
GRID_N_LON: int = 7200                    # 360 / 0.05
# Densify only the covered latitude band per mission (rows outside are all-zero).
GRID_LAT_CLIP: dict[str, tuple[float, float]] = {
    "GPM": (-68.0, 68.0),                 # GPM ~+/-67 deg inclination
    "TRMM": (-38.0, 38.0),                # TRMM ~+/-36 deg
}

# Stratification class edges (km). np.digitize -> len(edges)+1 physical classes;
# NaN inputs are mapped to a separate "undefined" class by pf.grid (never the
# top bin). Size from feature major_axis_km; echo-top from max_ht_20dbz (m->km).
SIZE_EDGES_KM: tuple[float, ...] = (20.0, 50.0, 100.0)   # -> <20, 20-50, 50-100, >100
ECHOTOP_EDGES_KM: tuple[float, ...] = (5.0, 7.5, 12.0)   # -> <5, 5-7.5, 7.5-12, >12
# Per-pixel rain_type classes (1=stratiform, 2=convective, 3=other).
RAINTYPE_CLASSES: tuple[int, ...] = (1, 2, 3)

VIEWS_ROOT_SUBDIR: str = "views"          # {PF_ROOT}/views/mission=.../...

# --- Echo-top-height quality control (Hirose et al. 2023, JTECH) ----------
# The raw "max height where Z>=thresh over the feature volume" is contaminated
# by mirror/second-trip echoes, sidelobe surface clutter, and isolated noise,
# all of which spike the high tail (their study: ~80-90% of >19 km echoes are
# artifacts). pf.echotop_qc removes these BEFORE taking the echo top. The raw
# 3-D precipRate / flagEcho / binMirrorImageL2 needed are read from the granule
# (present in both GPM Ku V07/V08 and TRMM PR V07A) and never persisted.

# Inner cross-track ray window (0-based, inclusive) trusted for HIGH-altitude
# echoes; outer rays suffer sidelobe surface clutter at large scan angle. Per
# Hirose this removes ~96% of spurious 19-20 km echoes. Applied only ABOVE the
# floor below -- near-surface data on outer rays is untouched.
ETH_INNER_RAY_LO: int = 8
ETH_INNER_RAY_HI: int = 42

# Altitude floor (m MSL) below which NO high-altitude artifact removal is
# applied: the filter is known to clip some real >30 dBZ echoes at 15-17 km, so
# tops at or below this height are always taken at face value.
ETH_FILTER_FLOOR_M: float = 15000.0

# Isolation test: a high top is artifact-suspect if NO neighbour in the 8-pixel
# (scan x ray) window has a storm top above this height. A real overshoot is
# embedded in a deep system and fails the test (kept, right-censored).
ETH_OVERSHOOT_NEIGHBOR_M: float = 15000.0

# Per-mission radar observation ceiling (m MSL): the maximum altitude at which a
# 20 dBZ echo could still be detected. A cleaned top within ETH_CENSOR_BIN_TOL
# bins of this is RIGHT-CENSORED (a lower bound, not a point). This is an
# instrument sensitivity limit -- NOT "the highest bin that happens to have
# echo" -- so a shallow storm that simply ends is not falsely flagged truncated.
# Hirose et al.: TRMM PR 15.4-19.75 km (degraded to ~13-15 km Jul/Aug 2004 and
# post-Jul 2014); GPM quality guaranteed to 19 km, data to ~22 km.
ETH_OBS_CEILING_M: dict[str, float] = {"GPM": 19000.0, "TRMM": 18000.0}
ETH_OBS_CEILING_DEFAULT_M: float = 18000.0

# Max isolated-noise "peel" iterations: if the current top pixel is judged an
# isolated artifact, its above-floor bins are masked and the top recomputed,
# up to this many times (bounds a pathological noisy column).
ETH_MAX_PEEL: int = 3

# Right-censor when the cleaned top is within this many bins of the highest
# observed bin in its column (the per-ray observation ceiling): the true top is
# unknown, so the value is a lower bound, never a point value.
ETH_CENSOR_BIN_TOL: int = 1

# Per-feature echo-top QC flag bits (packed into int16 echotop_qc_flags).
ETH_FLAG_TOP_TRUNCATED: int = 1 << 0   # top at the ray observation ceiling
ETH_FLAG_DD_FROM_TOP: int = 1 << 1     # precip rate decreases downward from top
ETH_FLAG_ISOLATED: int = 1 << 2        # no neighbour storm top > 15 km
ETH_FLAG_MIRROR_REMOVED: int = 1 << 3  # bins above JAXA mirror flag dropped
ETH_FLAG_SIDELOBE_REMOVED: int = 1 << 4  # outer-ray high bins dropped
ETH_FLAG_NOISE_PEELED: int = 1 << 5    # >=1 isolated-noise peel applied
ETH_FLAG_OVERSHOOT_REAL: int = 1 << 6  # DD & truncated & NOT isolated -> kept
ETH_FLAG_CENSORED: int = 1 << 7        # echo top is a right-censored lower bound
