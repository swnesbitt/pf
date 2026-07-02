"""Phase-4 registration wiring: reader registries, product map, short-names.

Asserts that mission ``"TRMM"`` is wired into the orchestrator
(``granule._RADAR_READERS`` / ``_IMAGER_READERS``), the search layer
(``search._MISSION_PRODUCTS``), and ``config.SHORT_NAMES`` — all by class
identity / exact value — while GPM is left byte-identical. No granules, no
network.
"""

from __future__ import annotations

from pf import config, feature_id, granule, search
from pf.features import FEATURE_SCHEMA, PIXEL_SCHEMA
from pf.readers.gpm_gmi import GpmGmiReader
from pf.readers.gpm_ku import GpmKuReader
from pf.readers.trmm_pr import TrmmPrReader
from pf.readers.trmm_tmi import TrmmTmiReader


# ----------------------------------------------------------------------
# granule.py reader registries (by class identity)
# ----------------------------------------------------------------------
def test_radar_readers_registry():
    assert granule._RADAR_READERS == {"GPM": GpmKuReader, "TRMM": TrmmPrReader}
    assert granule._RADAR_READERS["GPM"] is GpmKuReader
    assert granule._RADAR_READERS["TRMM"] is TrmmPrReader


def test_imager_readers_registry():
    assert granule._IMAGER_READERS == {"GPM": GpmGmiReader, "TRMM": TrmmTmiReader}
    assert granule._IMAGER_READERS["GPM"] is GpmGmiReader
    assert granule._IMAGER_READERS["TRMM"] is TrmmTmiReader


# ----------------------------------------------------------------------
# search.py mission -> product map
# ----------------------------------------------------------------------
def test_mission_products_trmm():
    radar_cls, imager_cls, radar_sn, imager_sn = search._MISSION_PRODUCTS["TRMM"]
    assert radar_cls is TrmmPrReader
    assert imager_cls is TrmmTmiReader
    assert radar_sn == "GPM_2APR"
    assert imager_sn == "GPM_1CTRMMTMI"


def test_mission_products_gpm():
    radar_cls, imager_cls, radar_sn, imager_sn = search._MISSION_PRODUCTS["GPM"]
    assert radar_cls is GpmKuReader
    assert imager_cls is GpmGmiReader
    assert radar_sn == "GPM_2AKu"
    assert imager_sn == "GPM_1CGPMGMI"


# ----------------------------------------------------------------------
# config.SHORT_NAMES
# ----------------------------------------------------------------------
def test_short_names_values():
    assert config.SHORT_NAMES["TRMM_PR"] == "GPM_2APR"
    assert config.SHORT_NAMES["TRMM_TMI"] == "GPM_1CTRMMTMI"
    assert config.SHORT_NAMES["GPM_KU"] == "GPM_2AKu"
    assert config.SHORT_NAMES["GPM_GMI"] == "GPM_1CGPMGMI"


def test_short_names_no_stale_trmm_2a_values():
    """No stale TRMM_2A25 / TRMM_2A12 short-names remain anywhere."""
    vals = set(config.SHORT_NAMES.values())
    assert "TRMM_2A25" not in vals
    assert "TRMM_2A12" not in vals
    # Old key spellings must also be gone.
    assert "TRMM_2A25" not in config.SHORT_NAMES
    assert "TRMM_2A12" not in config.SHORT_NAMES


# ----------------------------------------------------------------------
# feature_id round-trip for TRMM
# ----------------------------------------------------------------------
def test_feature_id_trmm_round_trip():
    fid = feature_id.encode("TRMM", 522, 18)
    assert feature_id.decode(fid) == ("TRMM", 522, 18)


def test_mission_code_trmm_is_1():
    assert config.MISSION_CODE["TRMM"] == 1
    assert config.MISSION_NAME[1] == "TRMM"


# ----------------------------------------------------------------------
# Schema invariance: Phase 4 must not touch the frozen Parquet schemas.
# (Full field-by-field check lives in test_phase3_schema_invariance.py.)
# ----------------------------------------------------------------------
def test_phase4_schema_invariance():
    # 58 = 55 + 3 echo-top QC cols (flags/censored/ceiling); pixel 14 (+pct_37).
    assert len(FEATURE_SCHEMA) == 58
    assert len(PIXEL_SCHEMA) == 14
