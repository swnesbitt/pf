"""Offline tests for the Phase-5 ERA-5 co-location module (``pf.era5``).

Everything here is SYNTHETIC and OFFLINE: a tiny in-memory ``xarray.Dataset``
mimicking the ARCO ERA-5 structure (latitude DESCENDING, longitude 0..360,
pressure ``level``, hourly ``time``) is built and passed as the ``ds=`` argument
to :func:`pf.era5.era5_for_features`, so no GCS/network access ever happens.

The synthetic grid is centred on a known point (lat -23.0, lon 302.0 == -58.0)
with enough margin to fully cover the largest (5deg / radius 2.5) box, and KNOWN
values are placed so every expectation can be hand-checked. Hand-computed
expectations use the SAME nearest rule as the production code
(``xarray .sel(method='nearest')``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pf.era5 as era5


# --------------------------------------------------------------------------
# Synthetic ERA-5 dataset builder
# --------------------------------------------------------------------------
# Grid generous enough to fully contain the radius-2.5 box around (-23, 302):
#   lat box -> [-20.5, -25.5];  lon box -> [299.5, 304.5]
_LAT = np.round(np.arange(-19.0, -27.25, -0.25), 3)   # DESCENDING
_LON = np.round(np.arange(299.0, 305.25, 0.25), 3)    # 0..360 convention
_LEVELS = np.array([400, 500, 600, 700, 850, 925, 1000], dtype=float)

# The two hours the multi-hour test exercises.
_T0 = np.datetime64("1997-12-30T22:00:00", "ns")
_T1 = np.datetime64("1997-12-30T23:00:00", "ns")
_TIMES = np.array([_T0, _T1])

CENTROID_LAT = -23.0
CENTROID_LON = -58.0           # -> 302.0 in 0..360
CENTROID_LON360 = 302.0


def _const2d(value, ntime, nlat, nlon):
    return np.full((ntime, nlat, nlon), value, dtype=float)


def make_synthetic_ds(
    cape_field=None,
    *,
    times=_TIMES,
    lat=_LAT,
    lon=_LON,
    levels=_LEVELS,
):
    """Build a tiny ERA-5-like ``xarray.Dataset`` with KNOWN constant fields.

    ``cape_field`` (if given) is a 2-D ``(nlat, nlon)`` array broadcast over all
    times for ``convective_available_potential_energy``; otherwise CAPE is a
    constant. All other 2-D vars are distinct constants so centroid columns are
    individually identifiable. 3-D wind/geopotential are set so that geometric
    height ``= geopotential / 9.81`` is a known monotonic profile and the wind
    increases linearly with height (so shear is hand-computable).
    """
    ntime, nlat, nlon = len(times), len(lat), len(lon)
    coords = {"time": times, "level": levels, "latitude": lat, "longitude": lon}

    if cape_field is None:
        cape = _const2d(1000.0, ntime, nlat, nlon)
    else:
        cape = np.broadcast_to(
            np.asarray(cape_field, dtype=float)[None, :, :], (ntime, nlat, nlon)
        ).copy()

    data = {
        "convective_available_potential_energy": (
            ("time", "latitude", "longitude"), cape),
        "convective_inhibition": (
            ("time", "latitude", "longitude"), _const2d(-50.0, ntime, nlat, nlon)),
        "sea_surface_temperature": (
            ("time", "latitude", "longitude"), _const2d(300.0, ntime, nlat, nlon)),
        "skin_temperature": (
            ("time", "latitude", "longitude"), _const2d(301.0, ntime, nlat, nlon)),
        "total_precipitation": (
            ("time", "latitude", "longitude"), _const2d(0.002, ntime, nlat, nlon)),
        era5.WIND10[0]: (
            ("time", "latitude", "longitude"), _const2d(1.0, ntime, nlat, nlon)),
        era5.WIND10[1]: (
            ("time", "latitude", "longitude"), _const2d(2.0, ntime, nlat, nlon)),
    }

    # --- 3-D fields: known height profile + linear-with-height wind ---------
    # heights (m) descending with pressure; sorted ascending internally.
    heights = np.array([7000, 5500, 4200, 3000, 1500, 800, 100], dtype=float)
    geop_col = heights * era5.G
    u_col = 0.002 * heights          # u(1000m)=2, u(3000m)=6, u(6000m)=12
    v_col = np.full_like(heights, 5.0)

    def _col3d(col):
        # (time, level, lat, lon)
        out = np.empty((ntime, len(levels), nlat, nlon), dtype=float)
        out[:] = col[None, :, None, None]
        return out

    data["geopotential"] = (
        ("time", "level", "latitude", "longitude"), _col3d(geop_col))
    data["u_component_of_wind"] = (
        ("time", "level", "latitude", "longitude"), _col3d(u_col))
    data["v_component_of_wind"] = (
        ("time", "level", "latitude", "longitude"), _col3d(v_col))

    return xr.Dataset(data, coords=coords)


def _features_df(records):
    """Build a features DataFrame with the required columns."""
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# 1. Output shape / columns / input-order preservation
# --------------------------------------------------------------------------
def test_columns_and_input_order():
    ds = make_synthetic_ds()
    # Three features, deliberately NOT in feature_id order, two hours.
    df = _features_df([
        {"feature_id": 30, "mission": "TRMM", "orbit": 522,
         "time": "1997-12-30T23:10", "centroid_lat": CENTROID_LAT,
         "centroid_lon": CENTROID_LON},
        {"feature_id": 10, "mission": "TRMM", "orbit": 522,
         "time": "1997-12-30T22:20", "centroid_lat": CENTROID_LAT,
         "centroid_lon": CENTROID_LON},
        {"feature_id": 20, "mission": "TRMM", "orbit": 522,
         "time": "1997-12-30T23:25", "centroid_lat": CENTROID_LAT,
         "centroid_lon": CENTROID_LON},
    ])
    out = era5.era5_for_features(df, ds=ds)

    # one row per input feature
    assert len(out) == 3
    # INPUT order preserved (feature_id sequence 30, 10, 20)
    assert list(out["feature_id"]) == [30, 10, 20]

    # exactly ALL_COLUMNS, in order, 252 total
    assert list(out.columns) == era5.ALL_COLUMNS
    assert len(era5.ALL_COLUMNS) == 252

    # 4 meta + 8 centroid + 240 box-stat
    assert era5.META_COLUMNS == ["feature_id", "mission", "orbit", "time"]
    centroid_cols = [c for c in out.columns if c.endswith("_centroid")]
    assert len(centroid_cols) == 8
    box_cols = [c for c in out.columns
                if c not in era5.META_COLUMNS and not c.endswith("_centroid")]
    assert len(box_cols) == 240


# --------------------------------------------------------------------------
# 2. Centroid value == synthetic value at the nearest grid cell
# --------------------------------------------------------------------------
def test_centroid_value_matches_nearest_cell():
    # Hand-set a unique CAPE value at the centroid's nearest cell.
    cape = np.zeros((len(_LAT), len(_LON)), dtype=float)
    ilat = int(np.argmin(np.abs(_LAT - CENTROID_LAT)))
    ilon = int(np.argmin(np.abs(_LON - CENTROID_LON360)))
    cape[ilat, ilon] = 1550.0      # the validated orbit-522 CAPE value
    ds = make_synthetic_ds(cape_field=cape)

    df = _features_df([
        {"feature_id": 1, "mission": "TRMM", "orbit": 522,
         "time": "1997-12-30T23:00", "centroid_lat": CENTROID_LAT,
         "centroid_lon": CENTROID_LON},
    ])
    out = era5.era5_for_features(df, ds=ds)

    # Same nearest rule the production code uses.
    expected = float(
        ds["convective_available_potential_energy"]
        .sel(time=_T1, method="nearest")
        .sel(latitude=CENTROID_LAT, longitude=CENTROID_LON360, method="nearest")
        .values
    )
    assert expected == 1550.0
    assert out["cape_centroid"].iloc[0] == pytest.approx(1550.0)

    # The other constant 2-D vars also land on their known values.
    assert out["cin_centroid"].iloc[0] == pytest.approx(-50.0)
    assert out["sst_centroid"].iloc[0] == pytest.approx(300.0)
    assert out["skt_centroid"].iloc[0] == pytest.approx(301.0)
    assert out["tpr_centroid"].iloc[0] == pytest.approx(0.002)


# --------------------------------------------------------------------------
# 3. Box stats over a known pattern, NaN-safe
# --------------------------------------------------------------------------
def test_box_stats_known_pattern_nan_safe():
    # Build a CAPE field that is 0 everywhere except inside the 5deg box, where
    # we place a known pattern (a ramp), with one NaN that must be ignored.
    cape = np.zeros((len(_LAT), len(_LON)), dtype=float)

    lat_sel, lon_min, lon_max = era5._box_slices(
        CENTROID_LAT, CENTROID_LON360, 2.5)
    # boolean masks for the cells inside the 5deg box
    in_lat = (_LAT <= lat_sel.start) & (_LAT >= lat_sel.stop)
    in_lon = (_LON >= lon_min) & (_LON <= lon_max)
    lat_idx = np.where(in_lat)[0]
    lon_idx = np.where(in_lon)[0]

    # Fill the box with a deterministic ramp of distinct values.
    counter = 1.0
    box_vals = []
    for i in lat_idx:
        for j in lon_idx:
            cape[i, j] = counter
            box_vals.append(counter)
            counter += 1.0
    box_vals = np.array(box_vals)

    # Inject one NaN inside the box (must be ignored by NaN-safe stats).
    nan_i, nan_j = lat_idx[0], lon_idx[0]
    cape[nan_i, nan_j] = np.nan
    finite_box = box_vals[1:]   # drop the cell we set to NaN

    ds = make_synthetic_ds(cape_field=cape)
    df = _features_df([
        {"feature_id": 1, "mission": "TRMM", "orbit": 522,
         "time": "1997-12-30T23:00", "centroid_lat": CENTROID_LAT,
         "centroid_lon": CENTROID_LON},
    ])
    out = era5.era5_for_features(df, ds=ds).iloc[0]

    assert out["min_cape_5deg"] == pytest.approx(finite_box.min())
    assert out["max_cape_5deg"] == pytest.approx(finite_box.max())
    assert out["mean_cape_5deg"] == pytest.approx(finite_box.mean())
    # NaN was ignored: the box has (21*21 - 1) finite cells contributing.
    assert out["std_cape_5deg"] == pytest.approx(finite_box.std())
    assert out["p50_cape_5deg"] == pytest.approx(np.percentile(finite_box, 50))


# --------------------------------------------------------------------------
# 4. Fetch-once-per-hour + identical values for same hour+location
# --------------------------------------------------------------------------
def test_fetch_once_per_hour_and_identical_values():
    ds = make_synthetic_ds()
    # 3 features at 3 timestamps that round to 2 distinct hours (22:00, 23:00).
    df = _features_df([
        {"feature_id": 1, "mission": "TRMM", "orbit": 522,
         "time": "1997-12-30T22:20", "centroid_lat": CENTROID_LAT,
         "centroid_lon": CENTROID_LON},
        {"feature_id": 2, "mission": "TRMM", "orbit": 522,
         "time": "1997-12-30T23:10", "centroid_lat": CENTROID_LAT,
         "centroid_lon": CENTROID_LON},
        {"feature_id": 3, "mission": "TRMM", "orbit": 522,
         "time": "1997-12-30T23:25", "centroid_lat": CENTROID_LAT,
         "centroid_lon": CENTROID_LON},
    ])

    era5.FETCH_COUNT = 999  # ensure it gets reset by the call
    out = era5.era5_for_features(df, ds=ds)

    # exactly 2 unique hours fetched, NOT 3 features.
    assert era5.FETCH_COUNT == 2

    # features 2 and 3 share hour 23:00 AND the same location -> identical row.
    row2 = out[out["feature_id"] == 2].iloc[0]
    row3 = out[out["feature_id"] == 3].iloc[0]
    for col in era5.VALUE_COLUMNS:
        v2, v3 = row2[col], row3[col]
        if pd.isna(v2) and pd.isna(v3):
            continue
        assert v2 == pytest.approx(v3), col


# --------------------------------------------------------------------------
# 5. compute_stats unit test
# --------------------------------------------------------------------------
def test_compute_stats_known_array():
    values = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 5.0])
    finite = values[np.isfinite(values)]
    res = era5.compute_stats(values, "cape", "5deg")

    # All expected keys present.
    expected_keys = (
        ["min_cape_5deg", "max_cape_5deg", "mean_cape_5deg", "std_cape_5deg"]
        + [f"p{p}_cape_5deg" for p in era5.PERCENTILES]
    )
    assert set(res) == set(expected_keys)

    assert res["min_cape_5deg"] == pytest.approx(finite.min())
    assert res["max_cape_5deg"] == pytest.approx(finite.max())
    assert res["mean_cape_5deg"] == pytest.approx(finite.mean())
    assert res["std_cape_5deg"] == pytest.approx(finite.std())
    for p in era5.PERCENTILES:
        assert res[f"p{p}_cape_5deg"] == pytest.approx(np.percentile(finite, p))


def test_compute_stats_all_nan():
    values = np.array([np.nan, np.nan, np.nan])
    res = era5.compute_stats(values, "cin", "1p25deg")
    for stat in era5.STATS:
        assert np.isnan(res[f"{stat}_cin_1p25deg"])


# --------------------------------------------------------------------------
# 6. compute_shear_fields unit test
# --------------------------------------------------------------------------
def test_compute_shear_fields_known_profile():
    lat = np.array([CENTROID_LAT])
    lon = np.array([CENTROID_LON360])
    levels = _LEVELS
    heights = np.array([7000, 5500, 4200, 3000, 1500, 800, 100], dtype=float)
    geop = heights * era5.G
    u = 0.002 * heights          # u(1000)=2, u(3000)=6, u(6000)=12
    v = np.full_like(heights, 5.0)

    def _mk(col):
        return xr.DataArray(
            col[:, None, None] * np.ones((1, 1)),
            coords={"level": levels, "latitude": lat, "longitude": lon},
            dims=["level", "latitude", "longitude"],
        )

    u10, v10 = 1.0, 2.0
    subset = xr.Dataset({
        "geopotential": _mk(geop),
        "u_component_of_wind": _mk(u),
        "v_component_of_wind": _mk(v),
        era5.WIND10[0]: xr.DataArray(
            [[u10]], coords={"latitude": lat, "longitude": lon},
            dims=["latitude", "longitude"]),
        era5.WIND10[1]: xr.DataArray(
            [[v10]], coords={"latitude": lat, "longitude": lon},
            dims=["latitude", "longitude"]),
    })

    fields = era5.compute_shear_fields(subset)
    assert set(fields) == set(era5.SHEAR_HEIGHTS_M)
    for H in era5.SHEAR_HEIGHTS_M:
        u_h = 0.002 * H
        v_h = 5.0
        expected = np.sqrt((u_h - u10) ** 2 + (v_h - v10) ** 2)
        got = float(
            fields[H].sel(latitude=CENTROID_LAT, longitude=CENTROID_LON360).values
        )
        assert got == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------
# 7. ERA5_SCHEMA field count + column naming convention
# --------------------------------------------------------------------------
def test_era5_schema_fields_and_naming():
    # 4 meta + 252 value? Spec: 252 value cols + 4 meta. ALL_COLUMNS == 252.
    # The schema enumerates feature_id/mission/orbit/time + VALUE_COLUMNS.
    assert len(era5.ALL_COLUMNS) == 252
    assert len(era5.VALUE_COLUMNS) == 248          # 8 centroid + 240 box
    assert len(era5.ERA5_SCHEMA) == 252

    names = era5.ERA5_SCHEMA.names
    assert names[:4] == ["feature_id", "mission", "orbit", "time"]

    # centroid columns: one per STAT_VAR, named {var}_centroid
    for var in era5.STAT_VARS:
        assert f"{var}_centroid" in names

    # box columns: {stat}_{var}_{box} for every STATS x STAT_VARS x BOX_LABELS
    expected_box = set()
    for var in era5.STAT_VARS:
        for label in era5.BOX_LABELS.values():
            for stat in era5.STATS:
                expected_box.add(f"{stat}_{var}_{label}")
    assert len(expected_box) == 240
    name_set = set(names)
    assert expected_box <= name_set

    # dtypes: meta then float32 values
    import pyarrow as pa
    schema = era5.ERA5_SCHEMA
    assert schema.field("feature_id").type == pa.int64()
    assert schema.field("mission").type == pa.string()
    assert schema.field("orbit").type == pa.int32()
    assert schema.field("time").type == pa.timestamp("us")
    for col in era5.VALUE_COLUMNS:
        assert schema.field(col).type == pa.float32(), col


# --------------------------------------------------------------------------
# 8. write_era5 round-trip (offline, tmp_path)
# --------------------------------------------------------------------------
def test_write_era5_roundtrip(tmp_path):
    import pyarrow.dataset as pads

    # Build a small era5 DataFrame matching the schema columns.
    n = 3
    rows = []
    for i in range(n):
        row = {
            "feature_id": 1000 + i,
            "mission": "TRMM",
            "orbit": 522,
            "time": pd.Timestamp("1997-12-30T23:00:00"),
        }
        for col in era5.VALUE_COLUMNS:
            row[col] = float(i) + 0.5
        rows.append(row)
    df = pd.DataFrame(rows)[era5.ALL_COLUMNS]

    path = era5.write_era5(df, "TRMM", root=tmp_path)

    # Expected hive-partitioned path.
    expected = (
        tmp_path / "era5" / "mission=TRMM" / "year=1997" / "month=12"
        / "orbit=000522.parquet"
    )
    assert path == expected
    assert expected.exists()

    # Re-read via pyarrow dataset (hive partitioning) and check round-trip.
    dataset = pads.dataset(
        str(tmp_path / "era5"), format="parquet", partitioning="hive")
    table = dataset.to_table()
    back = table.to_pandas()

    assert len(back) == n
    assert set(back["feature_id"]) == {1000, 1001, 1002}

    # dtypes preserved: feature_id int64, value cols float32 in the file schema.
    import pyarrow as pa
    file_schema = dataset.schema
    assert file_schema.field("feature_id").type == pa.int64()
    assert file_schema.field("cape_centroid").type == pa.float32()

    # partition columns recovered from the hive path
    assert set(back["mission"].astype(str)) == {"TRMM"}

    # join to a features-like frame on feature_id
    feats = pd.DataFrame({
        "feature_id": [1000, 1001, 1002],
        "centroid_lat": [-23.0, -23.0, -23.0],
    })
    joined = feats.merge(
        back[["feature_id", "cape_centroid"]], on="feature_id", how="inner")
    assert len(joined) == n
    assert "cape_centroid" in joined.columns
