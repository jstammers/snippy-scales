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
    db: Path = typer.Option(None, "--db", help="SQLite results database path"),  # noqa: B008
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
    db: Path = typer.Option(None, "--db", help="SQLite results database path"),  # noqa: B008
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


@app.command(name="list")
def list_experiments(
    db: Path = typer.Option(Path("results.db"), "--db", help="SQLite results database path"),  # noqa: B008
) -> None:
    """List all saved experiments from the results database."""
    from snippy_scales.evaluation import SQLiteStore

    if not db.exists():
        console.print(f"[yellow]Database not found:[/] {db}")
        raise typer.Exit(0)

    store = SQLiteStore(db)
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
    db: Path = typer.Option(Path("results.db"), "--db"),  # noqa: B008
    output: Path = typer.Option(Path("reports"), "--output", "-o"),  # noqa: B008
    benchmark: str = typer.Option(
        "SPY", "--benchmark", "-b", help="Benchmark ticker (e.g. SPY, QQQ). Pass 'none' to disable."
    ),  # noqa: E501
) -> None:
    """Regenerate a tearsheet for a saved experiment."""
    from snippy_scales.evaluation import SQLiteStore, TearsheetGenerator
    from snippy_scales.evaluation.results import EvaluationResult, FoldResult, SweepResult

    if not db.exists():
        console.print(f"[red]Database not found:[/] {db}")
        raise typer.Exit(1)

    store = SQLiteStore(db)
    exps = store.load_experiments()

    matching = exps.filter(exps["id"] == experiment_id) if len(exps) > 0 else exps
    if len(matching) == 0:
        console.print(f"[red]Experiment {experiment_id} not found in {db}[/]")
        raise typer.Exit(1)

    exp_row = matching.row(0, named=True)
    sweep_df = store.load_sweep_results(experiment_id)

    if len(sweep_df) == 0:
        console.print("[yellow]No sweep results for this experiment.[/]")
        raise typer.Exit(0)

    # Reconstruct a minimal EvaluationResult for tearsheet generation
    # (equity curves come from fold_results)
    sweep_results = []
    for sweep_row in sweep_df.iter_rows(named=True):
        folds_df = store.load_fold_results(sweep_row["id"])
        folds: list[FoldResult] = []
        for fr in folds_df.iter_rows(named=True):
            import numpy as np

            from snippy_scales.backtesting.domain import BacktestMetrics

            def _metrics(row: dict, prefix: str) -> BacktestMetrics:
                return BacktestMetrics(
                    total_return_pct=row.get(f"{prefix}_return_pct") or 0.0,
                    sharpe_ratio=row.get(f"{prefix}_sharpe") or 0.0,
                    sortino_ratio=row.get(f"{prefix}_sortino") or 0.0,
                    calmar_ratio=row.get(f"{prefix}_calmar") or 0.0,
                    max_drawdown_pct=row.get(f"{prefix}_max_dd") or 0.0,
                    win_rate_pct=row.get(f"{prefix}_win_rate") or 0.0,
                    profit_factor=row.get(f"{prefix}_profit_factor") or 0.0,
                    total_trades=row.get(f"{prefix}_trades") or 0,
                    expectancy=row.get(f"{prefix}_expectancy") or 0.0,
                )

            train_eq = np.array(json.loads(fr.get("train_equity_json") or "[]"))
            test_eq = np.array(json.loads(fr.get("test_equity_json") or "[]"))
            folds.append(
                FoldResult(
                    fold_idx=fr["fold_idx"],
                    params=json.loads(sweep_row["params_json"]),
                    train_metrics=_metrics(fr, "train"),
                    test_metrics=_metrics(fr, "test"),
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
