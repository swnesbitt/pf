"""Shared engine for the shear/CAPE hypothesis notebooks.

Every notebook does `from _shc import *`. Centralizes: the DuckDB connection over the
materialized analysis table (features ⨝ era5, |lat|<40, both missions), the binning /
2-D composite / fixed-CAPE-line helpers, region boxes, default predictor choices, and
the extreme-value thresholds (percentile + absolute Zipser cuts).
"""
from __future__ import annotations
import numpy as np, pandas as pd
import duckdb
import matplotlib.pyplot as plt

ANALYSIS = "/data/scratch/a/snesbitt/pf_db/analysis/shear_cape/**/*.parquet"

# --- predictor defaults (see plan §3) -------------------------------------------------
AMBIENT_CAPE = "p90_cape_2p50deg"     # pre-convective ambient CAPE (not depleted centroid)
SHEAR        = "shear_6000m_centroid" # deep-layer (0-6 km) bulk shear -> organization
SHEAR_LAYERS = ["shear_1000m_centroid", "shear_3000m_centroid", "shear_6000m_centroid"]
MOISTURE     = "mean_tpr_2p50deg"
CIN          = "mean_cin_2p50deg"

# fixed-width composite bins (physical units)
CAPE_W, CAPE_MAX = 250.0, 5000.0      # J/kg
SHEAR_W, SHEAR_MAX = 2.0, 30.0        # m/s
MIN_BIN_N = 50                        # mask thinner composite cells

# extreme-value thresholds (plan A7) — absolute (Zipser-style) cuts
ABS_EXTREME = {"ht40_km": (">=", 14.0), "ht30_km": (">=", 16.0),
               "min_pct_37": ("<=", 150.0), "min_pct_85_89": ("<=", 100.0)}

REGIONS = {
 "Amazon": (-15, 5, -75, -50), "Congo": (-10, 10, 10, 35),
 "Maritime Cont.": (-10, 10, 95, 150), "W Pacific pool": (0, 15, 130, 170),
 "E Pacific ITCZ": (2, 12, -140, -90), "Sahel/W Africa": (5, 18, -15, 30),
 "US Great Plains": (30, 40, -105, -85), "SE S.America": (-38, -20, -65, -45),
 "Bay of Bengal": (5, 22, 80, 100),
}

# convective-feature filter for the intensity/organization tests (compare storms, not drizzle)
CONV_FILTER = "(ht30_km > 0 OR conv_area_km2 > 0)"


def connect():
    """DuckDB connection with the `fe` view over the materialized analysis table."""
    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute(f"CREATE VIEW fe AS SELECT * FROM read_parquet('{ANALYSIS}', hive_partitioning=1)")
    return con


def q(con, sql, *params):
    return con.execute(sql, list(params)).df()


# --- 2-D composite in (CAPE, shear) phase space --------------------------------------
def composite_2d(con, response, *, mission, surf, stat="median",
                 cape_col=AMBIENT_CAPE, shear_col=SHEAR, conv_only=True, where=""):
    """median (or p90/p99/mean/frac) of `response` over (CAPE x shear) bins → (grid, ncount, extent)."""
    agg = {"median": f"quantile_cont({response},0.5)", "p90": f"quantile_cont({response},0.9)",
           "p99": f"quantile_cont({response},0.99)", "mean": f"avg({response})",
           "frac": f"avg(CAST({response} AS DOUBLE))"}[stat]
    filt = f"AND {CONV_FILTER}" if conv_only else ""
    df = q(con, f"""
        SELECT floor({cape_col}/{CAPE_W})*{CAPE_W} AS cb,
               floor({shear_col}/{SHEAR_W})*{SHEAR_W} AS sb,
               {agg} AS val, count(*) AS n
        FROM fe
        WHERE mission=? AND surf=? AND {response} IS NOT NULL
              AND {cape_col} IS NOT NULL AND {shear_col} IS NOT NULL
              AND {cape_col} < {CAPE_MAX} AND {shear_col} < {SHEAR_MAX} {filt} {where}
        GROUP BY 1,2
    """, mission, surf)
    ncb, nsb = int(CAPE_MAX/CAPE_W), int(SHEAR_MAX/SHEAR_W)
    G = np.full((nsb, ncb), np.nan); N = np.zeros((nsb, ncb))
    ci = (df.cb/CAPE_W).astype(int).clip(0, ncb-1); si = (df.sb/SHEAR_W).astype(int).clip(0, nsb-1)
    G[si, ci] = np.where(df.n.to_numpy() >= MIN_BIN_N, df.val.to_numpy(), np.nan)
    N[si, ci] = df.n.to_numpy()
    extent = [0, CAPE_MAX, 0, SHEAR_MAX]      # x=CAPE J/kg, y=shear m/s
    return np.ma.masked_invalid(G), N, extent


def plot_composite(con, response, *, mission, label, stat="median", cmap="turbo",
                   conv_only=True, where="", figsize=(13, 5)):
    """Side-by-side land/ocean composite heatmaps of `response` in (CAPE,shear) space."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax, surf in zip(axes, ["ocean", "land"]):
        G, N, ext = composite_2d(con, response, mission=mission, surf=surf, stat=stat,
                                 conv_only=conv_only, where=where)
        im = ax.imshow(G, origin="lower", aspect="auto", extent=ext, cmap=cmap)
        ax.set_xlabel("ambient CAPE [J kg$^{-1}$]"); ax.set_ylabel("0–6 km shear [m s$^{-1}$]")
        ax.set_title(f"{mission} {surf}: {stat} {label}")
        fig.colorbar(im, ax=ax, shrink=0.85, label=f"{stat} {label}")
    plt.show()


def fixed_cape_lines(con, response, *, mission, surf, stat="median",
                     cape_col=AMBIENT_CAPE, shear_col=SHEAR, conv_only=True):
    """response vs shear, one line per CAPE quartile — the 'shear effect at fixed CAPE' test."""
    filt = f"AND {CONV_FILTER}" if conv_only else ""
    agg = {"median": f"quantile_cont({response},0.5)", "p90": f"quantile_cont({response},0.9)",
           "p99": f"quantile_cont({response},0.99)", "mean": f"avg({response})"}[stat]
    df = q(con, f"""
        WITH s AS (SELECT *, ntile(4) OVER (ORDER BY {cape_col}) AS capeq FROM fe
                   WHERE mission=? AND surf=? AND {cape_col} IS NOT NULL AND {shear_col} IS NOT NULL
                         AND {response} IS NOT NULL AND {shear_col} < {SHEAR_MAX} {filt})
        SELECT capeq, floor({shear_col}/{SHEAR_W})*{SHEAR_W} AS sb, {agg} AS val, count(*) n
        FROM s GROUP BY 1,2 HAVING count(*) >= {MIN_BIN_N} ORDER BY 1,2
    """, mission, surf)
    rng = q(con, f"""SELECT capeq, min({cape_col}) lo, max({cape_col}) hi FROM
        (SELECT {cape_col}, ntile(4) OVER (ORDER BY {cape_col}) capeq FROM fe
         WHERE mission=? AND surf=? AND {cape_col} IS NOT NULL) GROUP BY 1 ORDER BY 1""", mission, surf)
    return df, rng


def plot_fixed_cape(con, response, *, mission, label, stat="median", figsize=(13, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax, surf in zip(axes, ["ocean", "land"]):
        df, rng = fixed_cape_lines(con, response, mission=mission, surf=surf, stat=stat)
        cmap = plt.cm.viridis(np.linspace(0, 1, 4))
        for k in sorted(df.capeq.unique()):
            s = df[df.capeq == k]; r = rng[rng.capeq == k].iloc[0]
            ax.plot(s.sb + SHEAR_W/2, s.val, "-o", ms=3, color=cmap[int(k)-1],
                    label=f"CAPE {r.lo:.0f}–{r.hi:.0f}")
        ax.set_xlabel("0–6 km shear [m s$^{-1}$]"); ax.set_ylabel(f"{stat} {label}")
        ax.set_title(f"{mission} {surf}: {label} vs shear at fixed CAPE"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    plt.show()


def load_sample(con, cols, *, mission=None, n=400_000, conv_only=True, where=""):
    """Pull a manageable per-feature sample to pandas for the ML/XAI notebooks.

    The reservoir sample is taken AFTER the filters (subquery), so `n` is the size of
    the returned filtered sample — not n rows of `fe` that then mostly fail the filter.
    """
    mfilt = f"AND mission='{mission}'" if mission else ""
    filt = f"AND {CONV_FILTER}" if conv_only else ""
    sel = ", ".join(cols)
    return q(con, f"""SELECT * FROM (
        SELECT {sel} FROM fe WHERE TRUE {mfilt} {filt} {where}
    ) USING SAMPLE {n} ROWS (reservoir)""")
