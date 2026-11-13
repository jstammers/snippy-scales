"""CLI command for a single Nautilus backtest.

Usage::

    algo backtest run trend-following -s AAPL --start 2018-01-01 \\
        --params '{"fast_period": 10}' --fees 0.0005 --band 0.02

    # Real ES outrights, rolled on volume
    algo backtest run momentum-ts -s ES.FUT --per-contract 2.25
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from snippy_scales.cli._common import (
    ROLL_HELP,
    STRATEGY_NAMES,
    console,
    load_data,
    make_costs,
    make_roll,
    parse_json,
    resolve_strategy,
)

app = typer.Typer(help="Single backtests on NautilusTrader.")


@app.command()
def run(
    strategy: str = typer.Argument(..., help=f"Strategy name: {', '.join(STRATEGY_NAMES)}"),
    symbol: list[str] = typer.Option(..., "--symbol", "-s", help="Symbol; repeat for several"),  # noqa: B008
    schema: str = typer.Option("ohlcv-1d", "--schema", help="Bar schema"),
    start: str = typer.Option(None, "--start", help="First bar close traded (ISO date)"),
    end: str = typer.Option(None, "--end", help="Last bar close traded (ISO date)"),
    params: str = typer.Option(None, "--params", help="Strategy parameters as a JSON object"),
    capital: float = typer.Option(1_000_000.0, "--capital", help="Initial capital"),
    fees: float = typer.Option(0.001, "--fees", help="Commission as a fraction of notional"),
    per_contract: float = typer.Option(0.0, "--per-contract", help="Commission per contract"),
    slippage: float = typer.Option(0.0, "--slippage", help="Probability of one tick of slippage"),
    band: float = typer.Option(0.0, "--band", help="Skip trades below this fraction of equity"),
    catalog_path: Path = typer.Option(None, "--catalog", help="Catalog root"),  # noqa: B008
    roll: str = typer.Option("volume", "--roll", help=ROLL_HELP),
    roll_days: int = typer.Option(5, "--roll-days", help="Calendar roll: days before expiry"),
    roll_months: str = typer.Option(None, "--roll-months", help="Contract months, e.g. HMUZ"),
    engine: str = typer.Option(
        "nautilus", "--engine", help="nautilus (reference) or vector (numba screening kernel)"
    ),
) -> None:
    """Backtest one strategy and print Nautilus' statistics."""
    from snippy_scales.execution import ENGINES

    if engine not in ENGINES:
        console.print(f"[red]--engine must be one of {', '.join(ENGINES)}[/]")
        raise typer.Exit(1)
    run_backtest = ENGINES[engine]

    cls = resolve_strategy(strategy)
    research = cls(**parse_json(params, "--params"))
    data = load_data(symbol, schema, catalog_path, make_roll(roll, roll_days, roll_months))
    result = run_backtest(
        research,
        data,
        start=start,
        end=end,
        initial_capital=capital,
        costs=make_costs(fees, per_contract, slippage),
        rebalance_band=band,
    )
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column(justify="right")
    table.add_row("total return", f"{result.total_return:+.2%}")
    table.add_row("orders", str(result.total_orders))
    table.add_row("rolls", str(len(result.rolls)))
    for key, value in sorted(result.stats.items()):
        table.add_row(key, f"{value:,.4f}")
    console.print(table)
    if result.rejections:
        console.print(f"[yellow]{len(result.rejections)} orders were denied or rejected:[/]")
        for reason in result.rejections[:5]:
            console.print(f"  {reason}")
