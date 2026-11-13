"""Helpers shared by the ``backtest`` and ``eval`` CLI commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

import typer
from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path

    from snippy_scales.data.futures import RollRule
    from snippy_scales.execution import Costs, MarketData
    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

console = Console()

STRATEGY_NAMES = ("trend-following", "momentum-ts", "momentum-cs", "mean-reversion")


def registry() -> dict[str, type[Strategy] | type[MultiAssetStrategy]]:
    from snippy_scales.strategies import (
        CrossSectionalMomentum,
        MeanReversion,
        TimeSeriesMomentum,
        TrendFollowing,
    )

    return {
        "trend-following": TrendFollowing,
        "momentum-ts": TimeSeriesMomentum,
        "momentum-cs": CrossSectionalMomentum,
        "mean-reversion": MeanReversion,
    }


def resolve_strategy(name: str) -> type[Strategy] | type[MultiAssetStrategy]:
    strategies = registry()
    if name not in strategies:
        console.print(f"[red]Unknown strategy '{name}'.[/]  Available: {', '.join(strategies)}")
        raise typer.Exit(1)
    return strategies[name]


def parse_json(text: str | None, option: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid {option} JSON:[/] {exc}")
        raise typer.Exit(1) from exc
    if not isinstance(value, dict):
        console.print(f"[red]{option} must be a JSON object.[/]")
        raise typer.Exit(1)
    return value


def make_costs(commission_rate: float, commission_per_contract: float, slippage: float) -> Costs:
    from snippy_scales.execution import Costs

    try:
        return Costs(
            commission_rate=0.0 if commission_per_contract else commission_rate,
            commission_per_contract=commission_per_contract,
            slippage_prob=slippage,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


ROLL_HELP = "Roll rule for parent futures symbols (ES.FUT): volume or calendar"


def make_roll(method: str, days: int, months: str | None) -> RollRule:
    from snippy_scales.data.futures import RollRule

    try:
        rule = cast("Literal['volume', 'calendar']", method)  # RollRule validates it
        return RollRule(rule, days_before_expiry=days, months=months)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


def load_data(
    symbols: list[str], schema: str, catalog_path: Path | None, roll: RollRule | None = None
) -> MarketData:
    from snippy_scales.execution import load_market_data

    console.print(f"[cyan]Loading[/] {', '.join(symbols)} ({schema})")
    try:
        return load_market_data(symbols, schema, roll=roll, catalog_path=catalog_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
