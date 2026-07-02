"""NASA Earthdata granule discovery for the PF pipeline.

Thin wrapper over :mod:`earthaccess` that the CLI (:mod:`pf.cli`) and the
orchestrator (:mod:`pf.granule`) use to find granules by date range or by orbit
number. Phase-1 is radar-only (GPM 2A-DPR); the helpers are structured so that
imager (GMI) and TRMM products slot in later by adding entries to
``granules_for_orbit``'s product map.
"""

from __future__ import annotations

import datetime as _dt
import re as _re
from typing import Any

import earthaccess

from pf.config import DEFAULT_PRODUCT_VERSION, PRODUCT_VERSION, SHORT_NAMES
from pf.readers.gpm_gmi import GpmGmiReader
from pf.readers.gpm_ku import GpmKuReader
from pf.readers.trmm_pr import TrmmPrReader
from pf.readers.trmm_tmi import TrmmTmiReader

# Module-level guard so repeated calls don't re-authenticate.
_LOGGED_IN = False

# Mission -> (radar reader, imager reader, radar short-name, imager short-name).
_MISSION_PRODUCTS = {
    "GPM": (GpmKuReader, GpmGmiReader,
            SHORT_NAMES["GPM_KU"], SHORT_NAMES["GPM_GMI"]),
    "TRMM": (TrmmPrReader, TrmmTmiReader,
             SHORT_NAMES["TRMM_PR"], SHORT_NAMES["TRMM_TMI"]),
}

# Search window (hours) on either side of an orbit's nominal time when we have
# only an orbit number and need to find the granule by parsing filenames.
_ORBIT_WINDOW_HOURS = 6


def login(strategy: str = "netrc") -> Any:
    """Authenticate to NASA Earthdata, idempotently.

    Parameters
    ----------
    strategy : str, optional
        ``earthaccess.login`` strategy (default ``"netrc"``).

    Returns
    -------
    object
        The :class:`earthaccess.Auth` object returned by
        :func:`earthaccess.login`.
    """
    global _LOGGED_IN
    auth = earthaccess.login(strategy=strategy)
    _LOGGED_IN = True
    return auth


def search_granules(
    short_name: str,
    start: str | _dt.datetime | _dt.date,
    end: str | _dt.datetime | _dt.date,
) -> list:
    """Search Earthdata for granules of ``short_name`` in a time range.

    Parameters
    ----------
    short_name : str
        NASA product short-name (e.g. ``"GPM_2ADPR"``).
    start, end : str or datetime or date
        Inclusive temporal bounds passed to
        :func:`earthaccess.search_data` as ``temporal=(start, end)``.

    Returns
    -------
    list
        A list of :class:`earthaccess.DataGranule` objects (possibly empty).
    """
    if not _LOGGED_IN:
        login()
    return earthaccess.search_data(
        short_name=short_name,
        temporal=(start, end),
    )


def prefer_version(granules: list, short_name: str, reader: Any) -> list:
    """Filter ``granules`` to the preferred data version for ``short_name``.

    The preferred version comes from :data:`pf.config.PRODUCT_VERSION` (falling
    back to :data:`pf.config.DEFAULT_PRODUCT_VERSION`) and is matched against the
    granule filename, which embeds the version as ``.V07A``/``.V08A``. If
    filtering would leave nothing (e.g. only the other version is available),
    the original ``granules`` list is returned unchanged as a graceful fallback.

    Parameters
    ----------
    granules : list
        Granules accepted by ``reader._filename_of``.
    short_name : str
        NASA product short-name (e.g. ``"GPM_1CGPMGMI"``).
    reader : pf.readers.base.SwathReader
        Reader exposing ``_filename_of`` for these granules.

    Returns
    -------
    list
        The version-filtered granules, or the original list if filtering is empty.
    """
    version = PRODUCT_VERSION.get(short_name, DEFAULT_PRODUCT_VERSION)
    token = f".V{version}"
    # Keep only the configured version (GPM pinned to V07 — V07 is complete across
    # 2014-03..2026-02, so this is full, uniform, and never mixes in V08). The
    # empty-fallback only matters outside the processed date range.
    filtered = [g for g in granules if token in reader._filename_of(g)]
    return filtered if filtered else granules


def group_by_orbit(granules: list, reader: Any) -> dict[int, Any]:
    """Group granules by orbit number using ``reader.orbit_of``.

    Parameters
    ----------
    granules : list
        Granules (or filenames) accepted by ``reader.orbit_of``.
    reader : pf.readers.base.SwathReader
        Reader whose ``orbit_of`` parses orbit numbers for these granules.

    Returns
    -------
    dict[int, object]
        Mapping of orbit number to a single granule. If multiple granules map
        to the same orbit the last one wins (Phase-1 expects one per orbit).
    """
    out: dict[int, Any] = {}
    for granule in granules:
        try:
            orbit = reader.orbit_of(granule)
        except ValueError:
            continue
        out[orbit] = granule
    return out


# Observation start in product filenames: ``.<YYYYMMDD>-S<HHMMSS>``. The
# ``-S<HHMMSS>`` anchor distinguishes the observation date from the earlier
# processing date that follows ``V9-``/``V10-``.
_OBS_START_RE = _re.compile(r"\.(\d{8})-S(\d{6})")
#: Observation end ``-E<HHMMSS>`` (same anchor block as ``-S``); the date is the
#: start date, rolled forward one day when the orbit crosses UTC midnight.
_OBS_END_RE = _re.compile(r"-E(\d{6})")


def granule_time_span(
    granule: Any, reader: Any
) -> tuple[_dt.datetime, _dt.datetime] | None:
    """Parse a granule's observation ``(start, end)`` UTC datetimes from its name.

    Filenames embed ``.<YYYYMMDD>-S<HHMMSS>-E<HHMMSS>`` (e.g.
    ``…19971230-S225350-E002507…`` → start 1997-12-30 22:53:50, end 00:25:07 the
    following day). When the end ``HHMMSS`` is earlier than the start, the orbit
    crossed UTC midnight, so the end date is advanced by one day.

    Returns ``None`` if the filename has no start anchor. When the end anchor is
    absent, a nominal ~100-minute orbit length is assumed.
    """
    import os as _os

    name = _os.path.basename(reader._filename_of(granule))
    ms = _OBS_START_RE.search(name)
    if ms is None:
        return None
    d = _dt.datetime.strptime(ms.group(1), "%Y%m%d").date()
    s = ms.group(2)
    start = _dt.datetime(d.year, d.month, d.day,
                         int(s[:2]), int(s[2:4]), int(s[4:6]))
    me = _OBS_END_RE.search(name)
    if me is None:
        return start, start + _dt.timedelta(minutes=100)
    e = me.group(1)
    end = _dt.datetime(d.year, d.month, d.day,
                       int(e[:2]), int(e[2:4]), int(e[4:6]))
    if end < start:                      # crossed UTC midnight
        end += _dt.timedelta(days=1)
    return start, end


def granule_start_date(granule: Any, reader: Any) -> _dt.date | None:
    """Parse a granule's observation START date from its filename.

    Product filenames embed the observation start as ``.<YYYYMMDD>-S<HHMMSS>``
    (e.g. ``2A.TRMM.PR.V9-20220125.19971230-S225350-E002507...`` → 1997-12-30).
    Note the basename contains TWO 8-digit tokens — the processing date after
    ``V9-``/``V10-`` and the observation date before ``-S`` — and the
    ``-S<HHMMSS>`` anchor selects the observation start.

    Parameters
    ----------
    granule : object
        Granule (or filename) accepted by ``reader._filename_of``.
    reader : pf.readers.base.SwathReader
        Reader exposing ``_filename_of`` for this granule.

    Returns
    -------
    datetime.date or None
        The observation start date, or ``None`` if the filename does not match.
    """
    import os as _os

    name = _os.path.basename(reader._filename_of(granule))
    match = _OBS_START_RE.search(name)
    if match is None:
        return None
    return _dt.datetime.strptime(match.group(1), "%Y%m%d").date()


def _orbit_window(
    mission: str, orbit: int
) -> tuple[_dt.datetime, _dt.datetime]:
    """Return a coarse temporal window to search for a given orbit.

    Phase-1 has no orbit->time ephemeris, so we cannot tightly target an
    orbit's time from its number alone. The orchestrator/CLI generally supplies
    a date hint; absent that we fall back to a window centered on "now" minus a
    nominal lag. Callers that already know the date should instead use
    :func:`search_granules` directly and :func:`group_by_orbit`.

    Parameters
    ----------
    mission : str
        Mission name (unused in Phase-1; reserved for per-mission ephemeris).
    orbit : int
        Orbit number (unused in Phase-1; reserved for ephemeris lookup).

    Returns
    -------
    tuple of (datetime, datetime)
        ``(start, end)`` UTC bounds.
    """
    now = _dt.datetime.utcnow()
    half = _dt.timedelta(hours=_ORBIT_WINDOW_HOURS)
    return (now - half, now + half)


def granules_for_orbit(
    mission: str,
    orbit: int,
    short_name: str | None = None,
    *,
    imager_short_name: str | None = None,
    start: str | _dt.datetime | _dt.date | None = None,
    end: str | _dt.datetime | _dt.date | None = None,
) -> dict[str, Any]:
    """Find the granule(s) for a single orbit, keyed by product role.

    Resolves the radar product (GPM 2A-DPR) and, best-effort, the imager
    product (GPM GMI 1C) over the same temporal window. The returned dict has a
    stable shape (``{"radar": ..., "imager": ...}``).

    Parameters
    ----------
    mission : str
        Mission name (e.g. ``"GPM"``).
    orbit : int
        Orbit number to resolve.
    short_name : str, optional
        Radar product short-name. Defaults to the mission's Ku short-name
        (``SHORT_NAMES["GPM_KU"]`` for GPM).
    imager_short_name : str, optional
        Imager product short-name (keyword-only). Defaults to
        ``SHORT_NAMES["GPM_GMI"]``.
    start, end : str or datetime or date, optional
        Temporal bounds for the search. When omitted a coarse window from
        :func:`_orbit_window` is used; supplying a date hint that brackets the
        orbit is strongly recommended for an exact match.

    Returns
    -------
    dict[str, object or None]
        ``{"radar": <granule or None>, "imager": <granule or None>}`` — the
        granules whose parsed orbit equals ``orbit``, or ``None`` if not found.
        Imager resolution is best-effort: a failed imager search yields
        ``"imager": None`` and never disturbs the radar result.
    """
    radar_cls, imager_cls, radar_default, imager_default = _MISSION_PRODUCTS[mission]

    if short_name is None:
        short_name = radar_default
    if imager_short_name is None:
        imager_short_name = imager_default

    if not _LOGGED_IN:
        login()

    if start is None or end is None:
        win_start, win_end = _orbit_window(mission, orbit)
        start = start or win_start
        end = end or win_end

    reader = radar_cls()
    granules = search_granules(short_name, start, end)
    granules = prefer_version(granules, short_name, reader)
    radar_by_orbit = group_by_orbit(granules, reader)

    # Imager resolution is best-effort: never let it break the radar result.
    imager_by_orbit: dict[int, Any] = {}
    try:
        imager_reader = imager_cls()
        imager_granules = search_granules(imager_short_name, start, end)
        # Prefer the configured per-product data version (e.g. V08 imagers);
        # falls back to all granules if filtering leaves nothing.
        imager_granules = prefer_version(
            imager_granules, imager_short_name, imager_reader
        )
        imager_by_orbit = group_by_orbit(imager_granules, imager_reader)
    except Exception:  # noqa: BLE001 — imager search is best-effort
        imager_by_orbit = {}

    return {
        "radar": radar_by_orbit.get(int(orbit)),
        "imager": imager_by_orbit.get(int(orbit)),
    }
