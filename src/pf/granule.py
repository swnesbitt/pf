"""Per-orbit orchestration: download → read → label → build rows → write Parquet.

``process_orbit`` is the single unit of work for the parallel layer
(``scripts/run_orbits_parallel.py`` and the SLURM month-array). It is fully
worker-isolated: each call logs in to Earthdata, downloads to a unique
``/dev/shm`` directory, writes its own two Parquet files, and cleans up — so
many orbits run concurrently with no shared state or locking. It never raises
to the caller; all failures are returned as ``status="failed"`` with an error
string so a Pool can keep going and collect failures for retry.
"""

from __future__ import annotations

import shutil
import time as _time
from pathlib import Path
from types import ModuleType

import pandas as pd

from pf import colocate as _colocate
from pf import config as _config
from pf import features as _features
from pf import feature_id as _feature_id
from pf import pixels as _pixels
from pf import views as _views
from pf.catalog import write_orbit
from pf.label import label_rpf, touches_edge
from pf.readers.gpm_gmi import GpmGmiReader
from pf.readers.gpm_ku import GpmKuReader
from pf.readers.trmm_pr import TrmmPrReader
from pf.readers.trmm_tmi import TrmmTmiReader

# Mission-keyed reader registries (radar + imager).
_RADAR_READERS = {"GPM": GpmKuReader, "TRMM": TrmmPrReader}
_IMAGER_READERS = {"GPM": GpmGmiReader, "TRMM": TrmmTmiReader}


def _download_with_retry(handle, dest: Path, *, attempts: int = 4,
                         base_delay: float = 2.0) -> str:
    """Download one granule into ``dest`` with bounded exponential backoff.

    Returns the local file path. Raises the last exception if all attempts fail.
    Accepts either an earthaccess granule handle or an already-local path string.
    """
    # Already a local file? (used by tests / pre-staged granules)
    if isinstance(handle, (str, Path)) and Path(handle).exists():
        return str(handle)

    import earthaccess
    import os
    import socket

    # A stalled Earthdata socket (a hung read with no exception) would otherwise block
    # earthaccess.download forever and freeze the whole worker for hours — the retry
    # loop never fires because nothing is raised. A default socket timeout converts a
    # stall (no data for `read_timeout` s) into socket.timeout, which the loop retries.
    # Healthy transfers stream data continuously, so this only trips on true hangs.
    read_timeout = float(os.environ.get("PF_DOWNLOAD_TIMEOUT", "120"))
    socket.setdefaulttimeout(read_timeout)

    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            downloaded = earthaccess.download([handle], str(dest))
            if downloaded and Path(downloaded[0]).exists():
                return str(downloaded[0])
            raise RuntimeError("earthaccess.download returned no file")
        except Exception as exc:  # noqa: BLE001 — bounded retry around network I/O
            last_exc = exc
            if i < attempts - 1:
                _time.sleep(base_delay * (2 ** i))
    raise last_exc  # type: ignore[misc]


def process_orbit(mission: str,
                  orbit: int,
                  granule_handles: dict,
                  cfg: ModuleType = _config) -> dict:
    """Process one orbit end-to-end and write its feature (and pixel) Parquet.

    Parameters
    ----------
    mission : str
        Mission name, e.g. ``"GPM"``.
    orbit : int
        Orbit number (used for the deterministic ``feature_id`` and output path).
    granule_handles : dict
        ``{"radar": <earthaccess granule | local path>}`` (Phase 1 uses ``radar``
        only). A ``None`` / missing radar handle yields ``status="skipped_no_radar"``.
    cfg : module
        Configuration module supplying thresholds and ``PF_ROOT`` (defaults to
        :mod:`pf.config`).

    Returns
    -------
    dict
        ``{"orbit", "n_features", "n_pixels", "status"[, "error"]}`` where
        ``status`` ∈ {``ok``, ``skipped_no_radar``, ``empty``, ``failed``}.
        Never raises — exceptions are captured into the returned dict.
    """
    result = {"orbit": int(orbit), "n_features": 0, "n_pixels": 0, "status": "ok"}
    tmpdir = Path(cfg.DOWNLOAD_DIR) / f"pf_{mission}_{orbit}"

    try:
        radar = granule_handles.get("radar")
        if radar is None:
            result["status"] = "skipped_no_radar"
            return result

        reader_cls = _RADAR_READERS.get(mission)
        if reader_cls is None:
            result["status"] = "failed"
            result["error"] = f"no radar reader registered for mission {mission!r}"
            return result

        # Per-worker Earthdata auth (idempotent) — skipped if the handle is local.
        if not (isinstance(radar, (str, Path)) and Path(radar).exists()):
            import earthaccess
            earthaccess.login(strategy="netrc")

        tmpdir.mkdir(parents=True, exist_ok=True)
        path = _download_with_retry(radar, tmpdir)

        swath = reader_cls().read(path)

        # --- optional imager co-location (Phase 3) -----------------------
        # Best-effort: if a GMI handle is supplied, download/read it and fill
        # swath.pct_85_89 with co-located 89 GHz PCT. Any failure leaves PCT as
        # NaN; the radar PFs are still produced unchanged.
        imager_handle = granule_handles.get("imager")
        imager_cls = _IMAGER_READERS.get(mission)
        if imager_handle is not None and imager_cls is not None:
            try:
                if not (
                    isinstance(imager_handle, (str, Path))
                    and Path(imager_handle).exists()
                ):
                    import earthaccess
                    earthaccess.login(strategy="netrc")
                imager_path = _download_with_retry(imager_handle, tmpdir)
                imager = imager_cls().read(imager_path)
                swath.pct_85_89 = _colocate.colocate_pct(swath, imager, cfg)
                swath.pct_37 = _colocate.colocate_pct37(swath, imager, cfg)
            except Exception:  # noqa: BLE001 — imager is best-effort
                pass

        # --- per-orbit VIEWS (sampling denominator) ----------------------
        # Best-effort byproduct: grid every valid near-surface observation into
        # 0.05 deg cells. Runs BEFORE labeling so feature-less orbits still
        # record their sampling. A views failure must never fail the orbit or
        # block the feature/pixel tables.
        try:
            vdf = _views.grid_orbit_views(swath)
            written = _views.write_orbit_views(vdf, mission, cfg.PF_ROOT)
            result["n_views_cells"] = 0 if vdf is None else len(vdf)
            if written is None:
                result.setdefault("n_views_cells", 0)
        except Exception:  # noqa: BLE001 — views are best-effort
            result["n_views_cells"] = 0

        dbz = cfg.DBZ_THRESHOLD_BY_MISSION.get(mission, cfg.DBZ_THRESHOLD)
        labeled, kept = label_rpf(
            swath,
            dbz_thresh=dbz,
            min_area_km2=cfg.MIN_AREA_KM2,
            min_pixels=cfg.MIN_PIXELS,
            connectivity=cfg.CONNECTIVITY,
        )
        if not kept:
            result["status"] = "empty"
            return result

        rows = [
            _features.build_feature_row(
                swath, labeled, local_label, area_km2,
                touches_edge(labeled, local_label),
            )
            for (local_label, area_km2) in kept
        ]
        features_df = pd.DataFrame(rows, columns=[f.name for f in _features.FEATURE_SCHEMA])

        # Pixel (member) table: one row per member pixel of every kept feature.
        pixel_rows: list[dict] = []
        for (local_label, _area_km2) in kept:
            pixel_rows.extend(
                _pixels.build_pixel_rows(swath, labeled, local_label, mission, int(orbit))
            )
        if pixel_rows:
            pixels_df = pd.DataFrame(
                pixel_rows, columns=[f.name for f in _features.PIXEL_SCHEMA]
            )
        else:
            pixels_df = None

        write_orbit(features_df, pixels_df, mission, cfg.PF_ROOT)

        result["n_features"] = len(rows)
        result["n_pixels"] = 0 if pixels_df is None else len(pixels_df)
        result["status"] = "ok"
        return result

    except Exception as exc:  # noqa: BLE001 — must never raise to the Pool
        result["status"] = "failed"
        result["error"] = repr(exc)
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def cleanup_stale_downloads(download_dir: str | None = None) -> int:
    """Remove leftover ``pf_*`` download dirs from hard-killed workers.

    ``process_orbit`` deletes its own ``pf_<mission>_<orbit>`` dir in a
    ``finally`` block, so the cache is normally self-cleaning. But a worker that
    is hard-killed (SIGKILL / OOM / node failure) leaves its transient dir
    behind. This helper sweeps such stragglers before a new run so tmpfs does
    not slowly fill across re-runs.

    Parameters
    ----------
    download_dir : str, optional
        Directory to sweep. Defaults to :data:`pf.config.DOWNLOAD_DIR`.

    Returns
    -------
    int
        Number of ``pf_*`` directories removed.
    """
    root = Path(download_dir if download_dir is not None else _config.DOWNLOAD_DIR)
    if not root.is_dir():
        return 0
    removed = 0
    # Only touch dirs matching ``pf_*`` directly under root — never recurse into
    # arbitrary siblings or follow the glob outside this directory.
    for entry in root.glob("pf_*"):
        if entry.is_dir() and not entry.is_symlink() and entry.parent == root:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed


__all__ = ["process_orbit", "cleanup_stale_downloads"]
