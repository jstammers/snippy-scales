"""CLI commands for walk-forward strategy evaluation.

Usage examples::

    # Walk-forward with default parameters
    algo eval run trend-following --symbol ES.c.0 --splits 5

    # Grid sweep
    algo eval sweep trend-following --symbol ES.c.0 \\
        --params '{"fast_period": [10, 20, 40], "slow_period": [40, 60, 120]}'

    # Random search
    algo eval sweep trend-following --symbol ES.c.0 \\
        --params '{"fast_period": [10, 20, 40], "slow_period": [40, 60, 120]}' \\
        --search random --trials 10

    # Evaluate all strategies across all available symbols
    algo eval run-all

    # List all saved experiments
    algo eval list --db results.db

    # Regenerate tearsheet for a saved experiment
    algo eval tearsheet 1 --db results.db --output reports/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

app = typer.Typer(help="Walk-forward evaluation and hyperparameter sweep commands.")
console = Console()

# ── Strategy registry ─────────────────────────────────────────────────────────


def _build_registry() -> dict[str, type[Strategy] | type[MultiAssetStrategy]]:
    from snippy_scales.strategies.mean_reversion import MeanReversion
    from snippy_scales.strategies.momentum_cs import CrossSectionalMomentum
    from snippy_scales.strategies.momentum_ts import TimeSeriesMomentum
    from snippy_scales.strategies.trend import TrendFollowing

    return {
        "trend-following": TrendFollowing,
        "momentum-ts": TimeSeriesMomentum,
        "momentum-cs": CrossSectionalMomentum,
        "mean-reversion": MeanReversion,
    }


_STRATEGY_NAMES = [
    "trend-following",
    "momentum-ts",
    "momentum-cs",
    "mean-reversion",
]


def _resolve_strategy(name: str) -> type[Strategy] | type[MultiAssetStrategy]:
    registry = _build_registry()
    if name not in registry:
        console.print(
            f"[red]Unknown strategy '{name}'.[/]  Available: {', '.join(registry.keys())}"
        )
        raise typer.Exit(1)
    return registry[name]


# ── Commands ──────────────────────────────────────────────────────────────────


@app.command()
def run(
    strategy: str = typer.Argument(..., help=f"Strategy name: {', '.join(_STRATEGY_NAMES)}"),
    symbol: str = typer.Option("ES.c.0", "--symbol", "-s", help="Instrument symbol"),
    schema: str = typer.Option("ohlcv-1d", "--schema", help="Bar schema (e.g. ohlcv-1d)"),
    splits: int = typer.Option(5, "--splits", "-n", help="Number of walk-forward folds"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test fraction per fold"),
    window: str = typer.Option("expanding", "--window", help="expanding or rolling"),
    gap: int = typer.Option(0, "--gap", help="Bars between train end and test start"),
    initial_capital: float = typer.Option(1_000_000.0, "--capital", help="Initial capital"),
    fees: float = typer.Option(0.001, "--fees", help="Per-trade fee fraction"),
    db: Path = typer.Option(None, "--db", help="DuckDB results database path"),  # noqa: B008
    tearsheet_dir: Path = typer.Option(  # noqa: B008
        None, "--tearsheet-dir", help="Directory for tearsheet output"
    ),
    benchmark: str = typer.Option(
        "SPY",
        "--benchmark",
        "-b",
        help="Benchmark ticker for tearsheet (e.g. SPY, QQQ). Pass 'none' to disable.",
    ),  # noqa: E501
) -> None:
    """Run walk-forward evaluation for a strategy with its default parameters."""
    from snippy_scales.data.ingest import load_bars
    from snippy_scales.evaluation import EvaluationRunner

    strategy_class = _resolve_strategy(strategy)

    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy

    if issubclass(strategy_class, MultiAssetStrategy):
        console.print(
            f"[red]{strategy_class.__name__} is a multi-asset strategy.[/]  "
            "Use [cyan]algo eval sweep --symbols ...[/] (multi-asset support coming soon)."
        )
        raise typer.Exit(1)
    single_class = cast("type[Strategy]", strategy_class)

    console.print(f"[cyan]Loading bars:[/] {symbol} ({schema})")
    try:
        bars = load_bars(symbol, schema)
    except FileNotFoundError as exc:
        console.print(
            f"[red]No data found for {symbol}/{schema}.[/]  "
            "Run [cyan]algo data ingest-config[/] first."
        )
        raise typer.Exit(1) from exc

    console.print(
        f"[cyan]Evaluating:[/] {single_class.__name__}  "
        f"splits={splits}  window={window}  bars={len(bars):,}"
    )

    runner = EvaluationRunner(
        initial_capital=initial_capital,
        fees=fees,
        n_splits=splits,
        test_size=test_size,
        window=window,
        gap=gap,
        db_path=db,
        tearsheet_dir=tearsheet_dir,
        benchmark=None if benchmark.lower() == "none" else benchmark,
    )

    result = runner.evaluate(single_class, bars, symbol=symbol)
    _print_result_summary(result)


@app.command()
def sweep(
    strategy: str = typer.Argument(..., help=f"Strategy name: {', '.join(_STRATEGY_NAMES)}"),
    symbol: str = typer.Option("ES.c.0", "--symbol", "-s", help="Instrument symbol"),
    schema: str = typer.Option("ohlcv-1d", "--schema", help="Bar schema"),
    params: str = typer.Option(
        ...,
        "--params",
        help="JSON dict of param lists e.g. '{\"fast_period\": [10, 20, 40]}'",
    ),
    search: str = typer.Option("grid", "--search", help="grid, random, or optuna"),
    trials: int = typer.Option(20, "--trials", help="Iterations for random/optuna search"),
    splits: int = typer.Option(5, "--splits", "-n", help="Number of walk-forward folds"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test fraction per fold"),
    window: str = typer.Option("expanding", "--window", help="expanding or rolling"),
    gap: int = typer.Option(0, "--gap", help="Bars between train end and test start"),
    initial_capital: float = typer.Option(1_000_000.0, "--capital"),
    fees: float = typer.Option(0.001, "--fees"),
    db: Path = typer.Option(None, "--db", help="DuckDB results database path"),  # noqa: B008
    tearsheet_dir: Path = typer.Option(None, "--tearsheet-dir"),  # noqa: B008
    benchmark: str = typer.Option(
        "SPY", "--benchmark", "-b", help="Benchmark ticker for tearsheet. Pass 'none' to disable."
    ),  # noqa: E501
) -> None:
    """Run a parameter sweep with walk-forward evaluation."""
    from snippy_scales.data.ingest import load_bars
    from snippy_scales.evaluation import (
        EvaluationRunner,
        ParameterGrid,
        RandomSearch,
    )

    strategy_class = _resolve_strategy(strategy)

    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy

    if issubclass(strategy_class, MultiAssetStrategy):
        console.print(
            f"[red]{strategy_class.__name__} is a multi-asset strategy.[/]  "
            "Multi-asset sweep support is coming soon."
        )
        raise typer.Exit(1)
    single_class = cast("type[Strategy]", strategy_class)

    try:
        param_dict: dict = json.loads(params)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid --params JSON:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[cyan]Loading bars:[/] {symbol} ({schema})")
    try:
        bars = load_bars(symbol, schema)
    except FileNotFoundError as exc:
        console.print(
            f"[red]No data found for {symbol}/{schema}.[/]  "
            "Run [cyan]algo data ingest-config[/] first."
        )
        raise typer.Exit(1) from exc

    # Build param search object
    if search == "grid":
        param_search = ParameterGrid(param_dict)
        n_combos = len(param_search)
        console.print(
            f"[cyan]Grid search:[/] {n_combos} combinations × {splits} folds = "
            f"{n_combos * splits} backtests"
        )
    elif search == "random":
        param_search = RandomSearch(param_dict, n_iter=trials, seed=42)
        console.print(
            f"[cyan]Random search:[/] {trials} trials × {splits} folds = "
            f"{trials * splits} backtests"
        )
    elif search == "optuna":
        from snippy_scales.evaluation import OptunaSearch

        # For optuna, values are treated as (low, high) int ranges or lists
        optuna_space = {}
        for k, v in param_dict.items():
            if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, int | float) for x in v):  # noqa: PLR2004
                optuna_space[k] = (v[0], v[-1])
            else:
                optuna_space[k] = v
        param_search = OptunaSearch(optuna_space, n_trials=trials, seed=42)
        console.print(f"[cyan]Optuna search:[/] {trials} trials × {splits} folds")
    else:
        console.print(f"[red]Unknown search type '{search}'.[/]  Use grid, random, or optuna.")
        raise typer.Exit(1)

    runner = EvaluationRunner(
        initial_capital=initial_capital,
        fees=fees,
        n_splits=splits,
        test_size=test_size,
        window=window,
        gap=gap,
        db_path=db,
        tearsheet_dir=tearsheet_dir,
        benchmark=None if benchmark.lower() == "none" else benchmark,
    )

    result = runner.evaluate(single_class, bars, params=param_search, symbol=symbol)
    _print_result_summary(result)

    # Print top parameter sets
    console.print("\n[bold]Top parameter sets (by mean OOS Sharpe):[/]")
    df = result.summary_df()
    if len(df) > 0:
        _print_polars_table(df.head(10))


@app.command(name="run-all")
def run_all(
    schema: str = typer.Option("ohlcv-1d", "--schema", help="Bar schema (e.g. ohlcv-1d)"),
    data_dir: Path = typer.Option(  # noqa: B008
        None, "--data-dir", help="Raw data directory (default: data/raw)"
    ),
    splits: int = typer.Option(5, "--splits", "-n", help="Number of walk-forward folds"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test fraction per fold"),
    window: str = typer.Option("expanding", "--window", help="expanding or rolling"),
    gap: int = typer.Option(0, "--gap", help="Bars between train end and test start"),
    initial_capital: float = typer.Option(1_000_000.0, "--capital", help="Initial capital"),
    fees: float = typer.Option(0.001, "--fees", help="Per-trade fee fraction"),
    db: Path = typer.Option(None, "--db", help="DuckDB results database path"),  # noqa: B008
    tearsheet_dir: Path = typer.Option(  # noqa: B008
        None, "--tearsheet-dir", help="Directory for tearsheet output"
    ),
    benchmark: str = typer.Option(
        "SPY",
        "--benchmark",
        "-b",
        help="Benchmark ticker for tearsheet. Pass 'none' to disable.",
    ),
) -> None:
    """Evaluate every strategy against every available symbol in data/raw."""
    from snippy_scales.data.ingest import RAW_DIR, load_bars
    from snippy_scales.evaluation import EvaluationRunner
    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy

    raw_dir = data_dir if data_dir is not None else RAW_DIR

    # 1. Discover symbols that have data for the requested schema
    parquet_files = sorted(raw_dir.glob(f"*/{schema}.parquet"))
    if not parquet_files:
        console.print(
            f"[red]No {schema!r} data found under {raw_dir}.[/]  "
            "Run [cyan]algo data ingest-config[/] first."
        )
        raise typer.Exit(1)

    symbols = [p.parent.name for p in parquet_files]
    console.print(f"[cyan]Discovered {len(symbols)} symbol(s):[/] {', '.join(symbols)}")

    # 2. Load bars for every symbol
    bars_map: dict[str, Any] = {}
    for symbol in symbols:
        try:
            bars_map[symbol] = load_bars(symbol, schema, raw_dir)
            console.print(f"  [green]loaded[/] {symbol}  ({len(bars_map[symbol]):,} bars)")
        except FileNotFoundError:
            console.print(f"  [yellow]skip[/] {symbol}: file unreadable")

    if not bars_map:
        console.print("[red]No symbols loaded. Aborting.[/]")
        raise typer.Exit(1)

    # 3. Build shared runner and strategy registry
    registry = _build_registry()
    runner = EvaluationRunner(
        initial_capital=initial_capital,
        fees=fees,
        n_splits=splits,
        test_size=test_size,
        window=window,
        gap=gap,
        db_path=db,
        tearsheet_dir=tearsheet_dir,
        benchmark=None if benchmark.lower() == "none" else benchmark,
    )

    # 4. Iterate: single-asset strategies run per symbol; multi-asset run once
    summary: list[dict[str, Any]] = []

    for strategy_name, strategy_class in registry.items():
        is_multi = issubclass(strategy_class, MultiAssetStrategy)

        if is_multi:
            if len(bars_map) < 2:  # noqa: PLR2004
                console.print(
                    f"\n[yellow]Skipping {strategy_name}:[/] "
                    "multi-asset strategy requires ≥ 2 symbols."
                )
                continue

            console.print(
                f"\n[bold cyan]>> {strategy_name}[/]  "
                f"[dim](multi-asset · {len(bars_map)} symbols)[/]"
            )
            aligned = _align_bars(bars_map)
            if len(aligned) < 2:  # noqa: PLR2004
                console.print("  [yellow]skip:[/] fewer than 2 symbols share a common time range.")
                continue

            try:
                multi_class = cast("type[MultiAssetStrategy]", strategy_class)
                result = runner.evaluate_multi_asset(multi_class, aligned)
                best = result.best_result
                summary.append(
                    {
                        "strategy": strategy_name,
                        "symbols": "+".join(sorted(aligned.keys())),
                        "mean_sharpe": best.mean_test_sharpe,
                        "mean_return_pct": best.mean_test_return,
                        "mean_max_dd_pct": best.mean_test_max_dd,
                    }
                )
                console.print(
                    f"  [green]done[/]  sharpe={best.mean_test_sharpe:.3f}  "
                    f"return={best.mean_test_return:.2f}%  "
                    f"max_dd={best.mean_test_max_dd:.2f}%"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]error:[/] {exc}")

        else:
            single_class = cast("type[Strategy]", strategy_class)
            for symbol, bars in bars_map.items():
                console.print(f"\n[bold cyan]>> {strategy_name}[/]  [dim]({symbol})[/]")
                try:
                    result = runner.evaluate(single_class, bars, symbol=symbol)
                    best = result.best_result
                    summary.append(
                        {
                            "strategy": strategy_name,
                            "symbols": symbol,
                            "mean_sharpe": best.mean_test_sharpe,
                            "mean_return_pct": best.mean_test_return,
                            "mean_max_dd_pct": best.mean_test_max_dd,
                        }
                    )
                    console.print(
                        f"  [green]done[/]  sharpe={best.mean_test_sharpe:.3f}  "
                        f"return={best.mean_test_return:.2f}%  "
                        f"max_dd={best.mean_test_max_dd:.2f}%"
                    )
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]error:[/] {exc}")

    # 5. Print combined summary table sorted by mean OOS Sharpe
    if not summary:
        console.print("\n[yellow]No results to display.[/]")
        return

    console.print("\n[bold]── Summary (sorted by mean OOS Sharpe) ──────────────────[/]")
    table = Table(show_lines=False, header_style="bold cyan")
    table.add_column("strategy", style="white")
    table.add_column("symbol(s)", style="dim")
    table.add_column("mean_sharpe", justify="right")
    table.add_column("mean_return%", justify="right")
    table.add_column("mean_max_dd%", justify="right")

    for row in sorted(summary, key=lambda r: r["mean_sharpe"], reverse=True):
        table.add_row(
            row["strategy"],
            row["symbols"],
            f"{row['mean_sharpe']:.3f}",
            f"{row['mean_return_pct']:.2f}",
            f"{row['mean_max_dd_pct']:.2f}",
        )
    console.print(table)


def _align_bars(bars_map: dict[str, Any]) -> dict[str, Any]:
    """Trim each symbol's bars to the common ts_event intersection.

    Multi-asset strategies require all DataFrames to have identical length
    and time-aligned rows.  This helper finds the intersection of timestamps
    across all symbols and filters every DataFrame to that common set.

    Args:
        bars_map: Mapping of symbol → Polars bar DataFrame.

    Returns:
        Filtered mapping with only symbols that share at least one common
        timestamp.  Returns an empty dict if no common timestamps exist.
    """
    import polars as pl

    ts_col = "ts_event"
    ts_sets = [set(df[ts_col].to_list()) for df in bars_map.values() if ts_col in df.columns]
    if not ts_sets:
        return {}

    common_ts = ts_sets[0].intersection(*ts_sets[1:])
    if not common_ts:
        return {}

    common_ts_list = sorted(common_ts)
    return {
        sym: df.filter(pl.col(ts_col).is_in(common_ts_list)).sort(ts_col)
        for sym, df in bars_map.items()
        if ts_col in df.columns
    }


@app.command(name="list")
def list_experiments(
    db: Path = typer.Option(Path("analytics.duckdb"), "--db", help="DuckDB results database path"),  # noqa: B008
) -> None:
    """List all saved experiments from the results database."""
    from snippy_scales.evaluation import AnalyticsStore

    if not db.exists():
        console.print(f"[yellow]Database not found:[/] {db}")
        raise typer.Exit(0)

    store = AnalyticsStore(db)
    df = store.load_experiments()

    if len(df) == 0:
        console.print("[yellow]No experiments found.[/]")
        return

    table = Table(title=f"Experiments in {db}", show_lines=True)
    for col in df.columns:
        table.add_column(col, style="cyan" if col == "id" else "white")

    for row in df.iter_rows():
        table.add_row(*[str(v) for v in row])

    console.print(table)


@app.command()
def tearsheet(
    experiment_id: int = typer.Argument(..., help="Experiment ID from the results database"),
    db: Path = typer.Option(Path("analytics.duckdb"), "--db"),  # noqa: B008
    output: Path = typer.Option(Path("reports"), "--output", "-o"),  # noqa: B008
    benchmark: str = typer.Option(
        "SPY", "--benchmark", "-b", help="Benchmark ticker (e.g. SPY, QQQ). Pass 'none' to disable."
    ),  # noqa: E501
) -> None:
    """Regenerate a tearsheet for a saved experiment."""
    from snippy_scales.evaluation import AnalyticsStore, TearsheetGenerator
    from snippy_scales.evaluation.results import EvaluationResult, FoldResult, SweepResult

    if not db.exists():
        console.print(f"[red]Database not found:[/] {db}")
        raise typer.Exit(1)

    import numpy as np

    from snippy_scales.backtesting.domain import BacktestMetrics

    analytics_store = AnalyticsStore(db)
    exps = analytics_store.load_experiments()

    matching = exps.filter(exps["id"] == experiment_id) if len(exps) > 0 else exps
    if len(matching) == 0:
        console.print(f"[red]Experiment {experiment_id} not found in {db}[/]")
        raise typer.Exit(1)

    exp_row = matching.row(0, named=True)
    sweep_df = analytics_store.load_sweep_results(experiment_id)

    if len(sweep_df) == 0:
        console.print("[yellow]No sweep results for this experiment.[/]")
        raise typer.Exit(0)

    # Helpers to reconstruct BacktestMetrics from stored fold_results columns.
    # Train window: only 4 summary metrics are stored; remaining fields default to 0.
    # Test window: all 33 BacktestMetrics fields are stored with their canonical names.
    def _train_metrics(row: dict) -> BacktestMetrics:
        return BacktestMetrics(
            total_return_pct=row.get("train_return_pct") or 0.0,
            sharpe_ratio=row.get("train_sharpe") or 0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            omega_ratio=0.0,
            max_drawdown_pct=row.get("train_max_dd") or 0.0,
            max_drawdown_duration=0,
            total_trades=row.get("train_trades") or 0,
            total_closed_trades=0,
            total_open_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            sqn=0.0,
            avg_trade_return_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            payoff_ratio=0.0,
            recovery_factor=0.0,
            avg_holding_period=0.0,
            avg_winning_duration=0.0,
            avg_losing_duration=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            start_value=0.0,
            end_value=0.0,
            total_fees_paid=0.0,
            open_trade_pnl=0.0,
            exposure_pct=0.0,
        )

    def _test_metrics(row: dict) -> BacktestMetrics:
        return BacktestMetrics(
            total_return_pct=row.get("total_return_pct") or 0.0,
            sharpe_ratio=row.get("sharpe_ratio") or 0.0,
            sortino_ratio=row.get("sortino_ratio") or 0.0,
            calmar_ratio=row.get("calmar_ratio") or 0.0,
            omega_ratio=row.get("omega_ratio") or 0.0,
            max_drawdown_pct=row.get("max_drawdown_pct") or 0.0,
            max_drawdown_duration=row.get("max_drawdown_duration") or 0,
            total_trades=row.get("total_trades") or 0,
            total_closed_trades=row.get("total_closed_trades") or 0,
            total_open_trades=row.get("total_open_trades") or 0,
            winning_trades=row.get("winning_trades") or 0,
            losing_trades=row.get("losing_trades") or 0,
            win_rate_pct=row.get("win_rate_pct") or 0.0,
            profit_factor=row.get("profit_factor") or 0.0,
            expectancy=row.get("expectancy") or 0.0,
            sqn=row.get("sqn") or 0.0,
            avg_trade_return_pct=row.get("avg_trade_return_pct") or 0.0,
            avg_win_pct=row.get("avg_win_pct") or 0.0,
            avg_loss_pct=row.get("avg_loss_pct") or 0.0,
            best_trade_pct=row.get("best_trade_pct") or 0.0,
            worst_trade_pct=row.get("worst_trade_pct") or 0.0,
            payoff_ratio=row.get("payoff_ratio") or 0.0,
            recovery_factor=row.get("recovery_factor") or 0.0,
            avg_holding_period=row.get("avg_holding_period") or 0.0,
            avg_winning_duration=row.get("avg_winning_duration") or 0.0,
            avg_losing_duration=row.get("avg_losing_duration") or 0.0,
            max_consecutive_wins=row.get("max_consecutive_wins") or 0,
            max_consecutive_losses=row.get("max_consecutive_losses") or 0,
            start_value=row.get("start_value") or 0.0,
            end_value=row.get("end_value") or 0.0,
            total_fees_paid=row.get("total_fees_paid") or 0.0,
            open_trade_pnl=row.get("open_trade_pnl") or 0.0,
            exposure_pct=row.get("exposure_pct") or 0.0,
        )

    # Reconstruct a minimal EvaluationResult for tearsheet generation.
    sweep_results = []
    for sweep_row in sweep_df.iter_rows(named=True):
        folds_df = analytics_store.load_fold_results(sweep_row["id"])
        folds: list[FoldResult] = []
        for fr in folds_df.iter_rows(named=True):
            train_eq = np.array(json.loads(fr.get("train_equity_json") or "[]"))
            test_eq = np.array(json.loads(fr.get("test_equity_json") or "[]"))
            folds.append(
                FoldResult(
                    fold_idx=fr["fold_idx"],
                    params=json.loads(sweep_row["params_json"]),
                    train_metrics=_train_metrics(fr),
                    test_metrics=_test_metrics(fr),
                    train_equity_curve=train_eq,
                    test_equity_curve=test_eq,
                    train_start=fr.get("train_start") or "",
                    train_end=fr.get("train_end") or "",
                    test_start=fr.get("test_start") or "",
                    test_end=fr.get("test_end") or "",
                )
            )
        sweep_results.append(SweepResult(params=json.loads(sweep_row["params_json"]), folds=folds))

    result = EvaluationResult(
        experiment_name=exp_row["name"],
        strategy_class=exp_row["strategy_class"],
        symbols=json.loads(exp_row["symbols_json"]),
        sweep_results=sweep_results,
    )

    output.mkdir(parents=True, exist_ok=True)
    safe_name = exp_row["name"].replace(" ", "_").replace("/", "-")
    output_path = output / f"{safe_name}_tearsheet"

    gen = TearsheetGenerator(benchmark=None if benchmark.lower() == "none" else benchmark)
    path = gen.generate_walk_forward(result, output_path=output_path)
    console.print(f"[green]Saved:[/] {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _print_result_summary(result: Any) -> None:
    from snippy_scales.evaluation.results import EvaluationResult

    if not isinstance(result, EvaluationResult) or not result.sweep_results:
        return

    best = result.best_result
    console.print("\n[bold green]Best parameter set:[/]")
    for k, v in best.params.items():
        console.print(f"  {k} = {v}")

    console.print(
        f"\n  Mean OOS Sharpe : [cyan]{best.mean_test_sharpe:.3f}[/]\n"
        f"  Mean OOS Return : [cyan]{best.mean_test_return:.2f}%[/]\n"
        f"  Mean OOS Max DD : [cyan]{best.mean_test_max_dd:.2f}%[/]"
    )


def _print_polars_table(df: Any) -> None:
    """Render a Polars DataFrame as a Rich table."""
    table = Table(show_lines=False, header_style="bold cyan")
    for col in df.columns:
        table.add_column(col)
    for row in df.iter_rows():
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)


from typing import Any  # noqa: E402 — needed for _print_result_summary annotation
