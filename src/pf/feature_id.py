"""Reversible packing of ``(mission, orbit, local_label)`` into an int64.

The feature id is the sole join key between the feature catalog and the
per-pixel store::

    feature_id = mission_code * 1e13 + orbit * 1e5 + local_label

with ``0 < local_label < 1e5`` and ``0 <= orbit < 1e5``. The maximum value
(mission_code 9, orbit 99999, label 99999) comfortably fits in int64.
"""

from __future__ import annotations

from pf.config import ID_MISSION_MULT, ID_ORBIT_MULT, MISSION_CODE, MISSION_NAME


def encode(mission: str | int, orbit: int, local_label: int) -> int:
    """Pack mission, orbit and local label into a single int64 feature id.

    Parameters
    ----------
    mission : str or int
        Mission name (e.g. ``"GPM"``) or its numeric code (e.g. ``2``).
    orbit : int
        Orbit number, ``0 <= orbit < 100000``.
    local_label : int
        Per-orbit feature label, ``0 < local_label < 100000``.

    Returns
    -------
    int
        The packed feature id.

    Raises
    ------
    AssertionError
        If any component is out of range or the mission is unknown.
    """
    if isinstance(mission, str):
        assert mission in MISSION_CODE, f"unknown mission name: {mission!r}"
        mission_code = MISSION_CODE[mission]
    else:
        mission_code = int(mission)
    assert mission_code in MISSION_CODE.values(), f"unknown mission code: {mission_code}"
    assert 0 <= orbit < ID_ORBIT_MULT, f"orbit out of range: {orbit}"
    assert 0 < local_label < ID_ORBIT_MULT, f"local_label out of range: {local_label}"

    return mission_code * ID_MISSION_MULT + orbit * ID_ORBIT_MULT + local_label


def decode(fid: int) -> tuple[str, int, int]:
    """Unpack a feature id into ``(mission_name, orbit, local_label)``.

    Parameters
    ----------
    fid : int
        A feature id produced by :func:`encode`.

    Returns
    -------
    tuple of (str, int, int)
        Mission name, orbit number and local label.

    Raises
    ------
    AssertionError
        If the encoded mission code is unknown.
    """
    fid = int(fid)
    mission_code = fid // ID_MISSION_MULT
    remainder = fid % ID_MISSION_MULT
    orbit = remainder // ID_ORBIT_MULT
    local_label = remainder % ID_ORBIT_MULT

    assert mission_code in MISSION_NAME, f"unknown mission code: {mission_code}"
    return MISSION_NAME[mission_code], int(orbit), int(local_label)
