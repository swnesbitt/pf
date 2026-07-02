#!/usr/bin/env python
"""End-to-end smoke test: process the first GPM 2A-DPR orbit in a date window.

Downloads one real granule, labels RPFs, writes Parquet, then re-reads the
feature dataset and asserts basic sanity. Intended to be run interactively on a
node with Earthdata access:

    python scripts/smoke_test_one_orbit.py --start 2018-07-01 --end 2018-07-01

It writes to a temporary root (``--root``) so it never pollutes the real DB.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pyarrow.dataset as ds

from pf import config as _config
from pf.granule import process_orbit
from pf.readers.gpm_ku import GpmKuReader
from pf import search as _search


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--short-name", default=None)
    ap.add_argument("--root", default=None, help="Output root (default: a temp dir)")
    args = ap.parse_args()

    root = args.root or tempfile.mkdtemp(prefix="pf_smoke_")
    _config.PF_ROOT = root

    short_name = args.short_name or _config.SHORT_NAMES["GPM_KU"]
    _search.login()
    granules = _search.search_granules(short_name, args.start, args.end)
    by_orbit = _search.group_by_orbit(granules, GpmKuReader())
    assert by_orbit, f"No {short_name} granules found in {args.start}..{args.end}"

    orbit = sorted(by_orbit)[0]
    print(f"Processing GPM orbit {orbit} -> {root}")
    result = process_orbit("GPM", orbit, {"radar": by_orbit[orbit]})
    print("result:", result)
    assert result["status"] in {"ok", "empty"}, result

    if result["status"] == "empty":
        print("No features in this orbit (valid outcome); pick a more convective window to exercise writes.")
        return

    feat = ds.dataset(Path(root) / "features", partitioning="hive").to_table().to_pandas()
    print(f"Read back {len(feat)} features.")
    assert len(feat) == result["n_features"]
    assert (feat["area_km2"] >= _config.MIN_AREA_KM2 - 1e-3).all(), "feature below min area!"
    assert feat["centroid_lat"].between(-90, 90).all()
    assert feat["centroid_lon"].between(-180, 180).all()
    assert (feat["npixels"] > 0).all()
    # feature_id round-trips
    from pf.feature_id import decode
    m, o, lab = decode(int(feat["feature_id"].iloc[0]))
    assert m == "GPM" and o == orbit
    print("SMOKE TEST PASSED:", dict(
        n_features=len(feat),
        max_area_km2=float(feat["area_km2"].max()),
        max_dbz=float(feat["max_near_sfc_dbz"].max()),
        n_mcs_by_area=int((feat["area_km2"] >= _config.MCS_AREA_KM2).sum()),
    ))


if __name__ == "__main__":
    main()
