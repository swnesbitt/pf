"""``pf`` command-line interface (typer).

Thin orchestration over :mod:`pf.search` and :mod:`pf.granule`:

* ``pf search MISSION --start --end`` — list orbits/granules in a window (no download).
* ``pf process-orbit MISSION ORBIT --start --end`` — process one orbit end-to-end.
* ``pf era5 MISSION --start --end`` — co-locate ERA-5 to existing features (network).

Bulk parallel processing lives in ``scripts/run_orbits_parallel.py``
(radar/imager) and ``scripts/add_era5.py`` (ERA-5).
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from pf.config import PF_ROOT, SHORT_NAMES
from pf import granule as _granule
from pf import search as _search
from pf.readers.gpm_ku import GpmKuReader

app = typer.Typer(add_completion=False, help="Precipitation-Feature (PF) database CLI.")
console = Console()


def _granule_name(g) -> str:
    """Best-effort human-readable granule name for display."""
    try:
        links = g.data_links()
        if links:
            return links[0].rsplit("/", 1)[-1]
    except Exception:  # noqa: BLE001
        pass
    try:
        return str(g["umm"]["GranuleUR"])
    except Exception:  # noqa: BLE001
        return str(g)


@app.command("search")
def search_cmd(
    mission: str = typer.Argument(..., help="Mission name, e.g. GPM"),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD"),
    short_name: str = typer.Option(
        None, "--short-name", help="Override earthaccess short_name (default from config)"
    ),
) -> None:
    """Search Earthdata and print orbits/granules in a time window (no downloads)."""
    if short_name is None:
        short_name = SHORT_NAMES["GPM_KU"]
    _search.login()
    granules = _search.search_granules(short_name, start, end)
    by_orbit = _search.group_by_orbit(granules, GpmKuReader())

    table = Table(title=f"{mission} {short_name}  {start} .. {end}")
    table.add_column("orbit", justify="right")
    table.add_column("granule")
    table.add_column("products", justify="center")
    for orbit in sorted(by_orbit):
        table.add_row(str(orbit), _granule_name(by_orbit[orbit]), "radar")
    console.print(table)
    console.print(f"[bold]{len(by_orbit)}[/bold] orbits found.")


@app.command("process-orbit")
def process_orbit_cmd(
    mission: str = typer.Argument(..., help="Mission name, e.g. GPM"),
    orbit: int = typer.Argument(..., help="Orbit number (integer)"),
    start: str = typer.Option(
        None, "--start", help="Search-window start YYYY-MM-DD (brackets the orbit)"
    ),
    end: str = typer.Option(None, "--end", help="Search-window end YYYY-MM-DD"),
    root: str = typer.Option(PF_ROOT, "--root", help="Output dataset root"),
) -> None:
    """Resolve one orbit's granule, process it, and print the result dict.

    A ``--start``/``--end`` window that brackets the orbit is strongly
    recommended (there is no orbit→time ephemeris from the number alone).
    Exits non-zero if processing fails.
    """
    handles = _search.granules_for_orbit(mission, orbit, start=start, end=end)
    if handles.get("radar") is None:
        console.print(
            f"[red]No radar granule found for {mission} orbit {orbit} "
            f"in window {start}..{end}.[/red]"
        )
        raise typer.Exit(code=2)

    # process_orbit reads PF_ROOT from the config module; honor an override.
    if root != PF_ROOT:
        from pf import config as _config
        _config.PF_ROOT = root

    result = _granule.process_orbit(mission, orbit, handles)
    console.print(result)
    if result["status"] == "failed":
        raise typer.Exit(code=1)


@app.command("era5")
def era5_cmd(
    mission: str = typer.Argument(..., help="Mission name, e.g. GPM or TRMM"),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD (inclusive)"),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD (exclusive)"),
    root: str = typer.Option(PF_ROOT, "--root", help="Dataset root"),
) -> None:
    """Co-locate ERA-5 environment to features in a mission + time window.

    Reads the existing feature table (``{root}/features``, hive-partitioned),
    filters by mission and ``[start, end)``, co-locates ARCO ERA-5 at each
    feature centroid (+ box stats), and writes a per-orbit ERA-5 Parquet table
    under ``{root}/era5``. Requires network access to the ARCO ERA-5 store.
    """
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    from pf.era5 import era5_for_features, write_era5

    mission_key = str(mission).upper()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    features_root = f"{root}/features"
    dataset = pads.dataset(features_root, partitioning="hive")

    cols = ["feature_id", "mission", "orbit", "time", "centroid_lat", "centroid_lon"]
    filt = (
        (pc.field("mission") == mission_key)
        & (pc.field("time") >= pa_scalar(start_ts))
        & (pc.field("time") < pa_scalar(end_ts))
    )
    table = dataset.to_table(columns=cols, filter=filt)
    df = table.to_pandas()

    if len(df) == 0:
        console.print(
            f"[yellow]No features for {mission_key} in {start}..{end}.[/yellow]"
        )
        return

    console.print(
        f"Co-locating ERA-5 for [bold]{len(df)}[/bold] {mission_key} features "
        f"in {start}..{end} (network)."
    )

    era5_df = era5_for_features(df)

    n_written = 0
    for orbit, grp in era5_df.groupby("orbit"):
        path = write_era5(grp, mission_key, root=root)
        n_written += len(grp)
        console.print(f"  orbit={int(orbit):06d}: {len(grp)} rows -> {path}")

    console.print(f"[green]Done. {n_written} ERA-5 rows written.[/green]")


def pa_scalar(ts):
    """Convert a pandas Timestamp to a us-resolution pyarrow timestamp scalar."""
    import pyarrow as pa

    return pa.scalar(ts.to_pydatetime(), type=pa.timestamp("us"))


def main() -> None:
    """Console-script entry point (``pf = pf.cli:main``)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
