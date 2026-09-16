"""CLI commands for research utilities."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Research utilities.")
console = Console()


@app.command()
def notebook(name: str = typer.Argument("scratch")) -> None:
    """Launch a Jupyter notebook for the given research name."""
    subprocess.run([sys.executable, "-m", "jupyter", "lab", f"research/{name}.ipynb"])  # noqa: S603


@app.command()
def roughness(
    symbol: str = typer.Argument(..., help="Symbol, e.g. ES.c.0"),
    data_dir: Path = typer.Option(  # noqa: B008
        Path("data/raw"), help="Root directory of raw Parquet data."
    ),
    schema: str = typer.Option("ohlcv-1d", help="Bar schema file to read."),
    n_folds: int = typer.Option(5, help="Chronological folds used to assess stability."),
) -> None:
    """Run the roughness gate — Step 1 of the SDE falsification plan.

    Decides whether an instrument's volatility path supports a rough or
    non-Markovian model, stably and out of sample.  Only a ``ROUGH`` verdict
    authorises progressing to a fractional or neural model.

    The gate is deliberately hard to pass: it refuses when observation noise
    makes the exponent unidentifiable, which is the common outcome on daily
    bars.  See ``docs/research/sde_market_dynamics.md``.
    """
    import polars as pl

    from snippy_scales.research.diagnostics import log_variance_proxy, roughness_gate

    path = data_dir / symbol.replace("/", "_") / f"{schema}.parquet"
    if not path.exists():
        console.print(f"[red]No data at[/] {path}")
        console.print("Ingest it first:  [cyan]algo data ingest-config configs/databento.yaml[/]")
        raise typer.Exit(code=1)

    bars = pl.read_parquet(path)
    log_rv = log_variance_proxy(bars)
    returns = (
        bars["close"].pct_change().drop_nulls().to_numpy() if "close" in bars.columns else None
    )

    report = roughness_gate(log_rv, returns=returns, symbol=symbol, n_folds=n_folds)
    console.print(report.summary())

    if schema.startswith("ohlcv-1d"):
        console.print(
            "\n[yellow]Note:[/] daily bars give a noisy realised-variance proxy. "
            "A NOISE_DOMINATED verdict here means the data is inadequate, not that "
            "the model is wrong — re-run on 1m bars before concluding."
        )

    raise typer.Exit(code=0 if report.proceed_to_neural else 2)
