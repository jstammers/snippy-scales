"""CLI commands for data ingestion and management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Data ingestion commands.")
console = Console()


@app.command()
def ingest(
    dataset: str = typer.Argument(..., help="Databento dataset identifier"),
    symbol: str = typer.Option(..., "--symbol", "-s", help="Instrument symbol"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
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
) -> None:
    """Download and upsert bar data for a single symbol.

    Existing data is extended rather than re-downloaded, keeping API costs low.
    """
    from snippy_scales.data.config import frequency_to_schema  # noqa: PLC0415
    from snippy_scales.data.ingest import RAW_DIR, upsert_symbol  # noqa: PLC0415

    try:
        schema = frequency_to_schema(frequency)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    out_dir = output_dir or RAW_DIR
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
            help="Print the ingestion plan without downloading any data.",
        ),
    ] = False,
) -> None:
    """Batch-ingest all symbols defined in a YAML config file.

    Uses upsert semantics: only data not already on disk is fetched, so
    repeated runs are cheap.

    Example::

        algo data ingest-config configs/databento.yaml
        algo data ingest-config configs/databento.yaml --frequency 1h
    """
    from snippy_scales.data.config import frequency_to_schema, load_config  # noqa: PLC0415
    from snippy_scales.data.ingest import RAW_DIR, ingest_from_config  # noqa: PLC0415

    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error loading config:[/] {exc}")
        raise typer.Exit(1) from exc

    # Validate frequency override before starting
    if frequency is not None:
        try:
            frequency_to_schema(frequency)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(1) from exc

    effective_schema = frequency_to_schema(frequency or cfg.tick_frequency)
    out_dir = output_dir or RAW_DIR
    import datetime  # noqa: PLC0415

    effective_end = cfg.end or datetime.date.today().isoformat()
    symbols = cfg.all_symbols

    # --- Print ingestion plan ---
    table = Table(title="Ingestion Plan", show_header=True, header_style="bold cyan")
    table.add_column("Asset Class")
    table.add_column("Symbol")
    table.add_column("Schema")
    table.add_column("Range")

    for class_name, ac in cfg.asset_classes.items():
        for sym in ac.symbols:
            table.add_row(
                class_name,
                sym,
                effective_schema,
                f"{cfg.start} → {effective_end}",
            )

    console.print(table)
    console.print(
        f"\n[dim]Dataset:[/] {cfg.dataset}  "
        f"[dim]Symbols:[/] {len(symbols)}  "
        f"[dim]Output:[/] {out_dir}"
    )

    if dry_run:
        console.print("[yellow]Dry-run mode — no data downloaded.[/]")
        return

    results = ingest_from_config(
        cfg,
        frequency_override=frequency,
        output_dir=out_dir,
    )

    # --- Summary ---
    succeeded = len(results)
    failed = len(symbols) - succeeded
    console.print(
        f"\n[bold green]Done.[/]  "
        f"{succeeded} symbol(s) succeeded" + (f", [bold red]{failed} failed[/]" if failed else ".")
    )
