"""CLI commands for running backtests."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Backtest commands.")
console = Console()


@app.command()
def run(
    strategy: str = typer.Argument(..., help="Strategy name or config path"),
    data_path: Path = typer.Option(Path("data/derived"), help="Path to bar data"),  # noqa: B008
    initial_cash: float = typer.Option(1_000_000.0, help="Starting capital"),  # noqa: B008
) -> None:
    """Run a backtest for a named strategy."""
    console.print(f"[cyan]Running backtest:[/] {strategy}  cash={initial_cash:,.0f}")
    # TODO: load strategy, call Rust engine via _algo_core
    console.print("[yellow]Not yet implemented — wire strategy → Rust engine.[/]")
