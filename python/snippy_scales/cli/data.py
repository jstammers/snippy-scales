"""CLI commands for data ingestion and management."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Data ingestion commands.")
console = Console()


def _fmt_cost(cost: float) -> str:
    """Format a cost float for display in the terminal.

    Returns ``"cached"`` for zero-cost (already up to date), ``"N/A"`` for
    NaN (estimation failed), and a dollar-formatted string otherwise.
    """
    if cost == 0.0:
        return "[dim]cached[/dim]"
    if math.isnan(cost):
        return "[yellow]N/A[/yellow]"
    return f"[bold yellow]${cost:.4f}[/bold yellow]"


@app.command()
def ingest(
    dataset: str = typer.Argument(..., help="Databento dataset identifier"),
    symbol: str = typer.Option(..., "--symbol", "-s", help="Instrument symbol"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    stype_in: Literal["raw_symbol", "continuous", "parent", "instrument_id"] = typer.Option(
        "raw_symbol", help="Databento symbology type"
    ),
    frequency: Annotated[
        str,
        typer.Option(
            "--frequency",
            "-f",
            help="Bar frequency: '1d', '1h', '1m', 'daily', etc.",
        ),
    ] = "1d",
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Root directory for raw Parquet files."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the cost-confirmation prompt."),
    ] = False,
) -> None:
    """Download and upsert bar data for a single symbol.

    Before downloading, the Databento metadata API is queried to estimate the
    cost of the request.  You will be asked to confirm unless ``--yes`` is
    passed.  Existing data is extended rather than re-downloaded, so symbols
    that are already up to date incur no cost and are skipped automatically.
    """
    from snippy_scales.data.config import frequency_to_schema  # noqa: PLC0415
    from snippy_scales.data.ingest import (  # noqa: PLC0415
        RAW_DIR,
        estimate_cost,
        upsert_symbol,
    )

    try:
        schema = frequency_to_schema(frequency)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    out_dir = output_dir or RAW_DIR

    # --- Cost estimation ---
    console.print(f"Estimating cost for [bold]{symbol}[/] ({schema})  {start} → {end} …")
    cost: float
    try:
        cost = estimate_cost(
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            start=start,
            end=end,
            output_dir=out_dir,
            stype_in=stype_in,
        )
    except Exception as exc:
        console.print(f"[yellow]Warning: cost estimation failed — {exc}[/]")
        cost = float("nan")

    if cost == 0.0:
        console.print("[dim]Symbol is already up to date — nothing to download.[/]")
        return

    console.print(f"Estimated cost: {_fmt_cost(cost)}")

    # --- Confirmation ---
    if not yes and not math.isnan(cost):
        if not typer.confirm("Proceed with download?"):
            raise typer.Abort()
    elif not yes and math.isnan(cost):
        console.print("[yellow]Cost estimate unavailable.[/]")
        if not typer.confirm("Proceed anyway?"):
            raise typer.Abort()

    # --- Download ---
    console.print(f"Ingesting [bold]{symbol}[/] ({schema})  {start} → {end}  [[dim]{dataset}[/]]")
    path = upsert_symbol(
        dataset=dataset,
        symbol=symbol,
        schema=schema,
        start=start,
        end=end,
        output_dir=out_dir,
    )
    console.print(f"[green]Saved:[/] {path}")


@app.command(name="ingest-config")
def ingest_config(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the YAML ingestion config file.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    frequency: Annotated[
        str | None,
        typer.Option(
            "--frequency",
            "-f",
            help=(
                "Override the tick_frequency set in the config file.  "
                "Accepts aliases like '1d', '1h', '1m', 'daily'."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Root directory for raw Parquet files."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show the ingestion plan and cost estimate without downloading.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the cost-confirmation prompt."),
    ] = False,
) -> None:
    """Batch-ingest all symbols defined in a YAML config file.

    Uses upsert semantics: only data not already on disk is fetched, so
    repeated runs are cheap.  Before any download the Databento metadata API
    is queried once per symbol to estimate per-asset and total costs.  You
    will be asked to confirm the total spend unless ``--yes`` is passed.

    Examples::

        algo data ingest-config configs/databento.yaml
        algo data ingest-config configs/databento.yaml --frequency 1h
        algo data ingest-config configs/databento.yaml --yes
        algo data ingest-config configs/databento.yaml --dry-run
    """
    import datetime  # noqa: PLC0415

    from snippy_scales.data.config import frequency_to_schema, load_config  # noqa: PLC0415
    from snippy_scales.data.ingest import (  # noqa: PLC0415
        RAW_DIR,
        estimate_costs_from_config,
        ingest_from_config,
    )

    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error loading config:[/] {exc}")
        raise typer.Exit(1) from exc

    if frequency is not None:
        try:
            frequency_to_schema(frequency)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(1) from exc

    effective_schema = frequency_to_schema(frequency or cfg.tick_frequency)
    out_dir = output_dir or RAW_DIR
    effective_end = cfg.end or datetime.date.today().isoformat()
    symbols = cfg.all_symbols

    # --- Cost estimation ---
    console.print(
        f"Estimating costs for [bold]{len(symbols)}[/] symbol(s) "
        f"[[dim]{cfg.dataset} / {effective_schema}[/]] …"
    )
    costs = estimate_costs_from_config(cfg, frequency_override=frequency, output_dir=out_dir)

    # --- Build plan table with per-symbol costs ---
    table = Table(title="Ingestion Plan", show_header=True, header_style="bold cyan")
    table.add_column("Asset Class")
    table.add_column("Symbol")
    table.add_column("Schema")
    table.add_column("Range")
    table.add_column("Cost (USD)", justify="right")

    for class_name, ac in cfg.asset_classes.items():
        for sym in ac.symbols:
            table.add_row(
                class_name,
                sym,
                effective_schema,
                f"{cfg.start} → {effective_end}",
                _fmt_cost(costs.get(sym, float("nan"))),
            )

    console.print(table)

    # --- Totals ---
    finite_costs = [c for c in costs.values() if not math.isnan(c)]
    has_nan = any(math.isnan(c) for c in costs.values())
    total = sum(finite_costs)
    total_str = f"[bold yellow]${total:.4f}[/bold yellow]"
    if has_nan:
        total_str += " [yellow]+ N/A[/yellow]"

    cached_count = sum(1 for c in costs.values() if c == 0.0)
    download_count = len(symbols) - cached_count

    console.print(
        f"\n[dim]Dataset:[/] {cfg.dataset}  "
        f"[dim]Schema:[/] {effective_schema}  "
        f"[dim]Output:[/] {out_dir}\n"
        f"[dim]Symbols to download:[/] {download_count}  "
        f"[dim]Already cached:[/] {cached_count}  "
        f"[dim]Total estimated cost:[/] {total_str}"
    )

    if dry_run:
        console.print("\n[yellow]Dry-run mode — no data downloaded.[/]")
        return

    # Skip download if everything is already cached
    if download_count == 0:
        console.print("\n[dim]All symbols are already up to date — nothing to download.[/]")
        return

    # --- Confirmation ---
    if not yes and not typer.confirm("\nProceed with download?"):
        raise typer.Abort()

    results = ingest_from_config(cfg, frequency_override=frequency, output_dir=out_dir)

    # --- Summary ---
    succeeded = len(results)
    failed = len(symbols) - succeeded
    console.print(
        f"\n[bold green]Done.[/]  "
        f"{succeeded} symbol(s) succeeded" + (f", [bold red]{failed} failed[/]" if failed else ".")
    )
