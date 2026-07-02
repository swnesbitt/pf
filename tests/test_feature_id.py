"""Round-trip and range-guard tests for :mod:`pf.feature_id`."""

from __future__ import annotations

import itertools

import pytest

from pf.config import MISSION_CODE, MISSION_NAME
from pf.feature_id import decode, encode


@pytest.mark.parametrize("mission", ["GPM", "TRMM"])
@pytest.mark.parametrize("orbit", [0, 1, 12345, 99999])
@pytest.mark.parametrize("local_label", [1, 7, 42, 99999])
def test_round_trip_by_name(mission, orbit, local_label):
    fid = encode(mission, orbit, local_label)
    name, dorbit, dlabel = decode(fid)
    assert name == mission
    assert dorbit == orbit
    assert dlabel == local_label


@pytest.mark.parametrize("code", [1, 2])
@pytest.mark.parametrize("orbit", [0, 1, 50000, 99999])
@pytest.mark.parametrize("local_label", [1, 99999])
def test_round_trip_by_code(code, orbit, local_label):
    fid = encode(code, orbit, local_label)
    name, dorbit, dlabel = decode(fid)
    assert name == MISSION_NAME[code]
    assert dorbit == orbit
    assert dlabel == local_label


def test_ids_fit_int64_and_unique():
    seen: dict[int, tuple] = {}
    missions = list(MISSION_CODE) + list(MISSION_CODE.values())
    orbits = [0, 1, 12345, 99999]
    labels = [1, 99999]
    int64_max = 2**63 - 1
    for mission, orbit, label in itertools.product(missions, orbits, labels):
        fid = encode(mission, orbit, label)
        assert 0 <= fid <= int64_max, f"id {fid} out of int64 range"
        # normalize mission to name so name/code duplicates collapse correctly
        norm = decode(fid)
        if norm in seen:
            # same decoded tuple must produce the same id
            assert seen[norm] == fid
        else:
            seen[norm] = fid
    # every distinct decoded tuple maps to a distinct id
    assert len(set(seen.values())) == len(seen)


def test_local_label_too_large_raises():
    with pytest.raises(AssertionError):
        encode("GPM", 100, 100_000)  # local_label >= 1e5


def test_local_label_zero_raises():
    with pytest.raises(AssertionError):
        encode("GPM", 100, 0)  # local_label must be > 0


def test_orbit_too_large_raises():
    with pytest.raises(AssertionError):
        encode("GPM", 100_000, 1)  # orbit >= 1e5


def test_unknown_mission_name_raises():
    with pytest.raises(AssertionError):
        encode("AQUA", 100, 1)


def test_unknown_mission_code_raises():
    with pytest.raises(AssertionError):
        encode(9, 100, 1)


def test_decode_unknown_mission_code_raises():
    # craft an id with mission_code 9 directly
    bogus = 9 * 10_000_000_000_000 + 1 * 100_000 + 1
    with pytest.raises(AssertionError):
        decode(bogus)
