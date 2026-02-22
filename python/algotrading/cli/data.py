"""CLI commands for data ingestion and management."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Data ingestion commands.")
console = Console()


@app.command()
def ingest(
    dataset: str = typer.Argument(..., help="Databento dataset identifier"),
    symbol: str = typer.Option(..., "--symbol", "-s", help="Instrument symbol"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
) -> None:
    """Download and store bar data from Databento."""
    console.print(f"Ingesting [bold]{symbol}[/] from {start} to {end}  [{dataset}]")
    from snippy_scales.data.ingest import ingest_databento  # noqa: PLC0415

    ingest_databento(dataset=dataset, symbol=symbol, start=start, end=end)
