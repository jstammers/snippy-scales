"""CLI commands for walk-forward evaluation (Optuna studies on Nautilus backtests).

Usage examples::

    # Walk-forward with default parameters
    algo eval run trend-following -s ES.c.0 --splits 5

    # Grid sweep (lists are grid values)
    algo eval sweep trend-following -s ES.c.0 --search grid \\
        --params '{"fast_period": [10, 20, 40], "slow_period": [40, 60, 120]}'

    # TPE search over ranges
    algo eval sweep momentum-ts -s ES.c.0 --trials 50 \\
        --params '{"lookback": {"low": 20, "high": 260}}'

    # Every strategy on every symbol in the catalog
    algo eval run-all

    # Saved studies, a report, and a tearsheet
    algo eval list
    algo eval report <study>
    algo eval tearsheet <study> --output reports/<study>.html
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    registry,
    resolve_strategy,
)
from snippy_scales.data.paths import EXPERIMENTS_DIR
from snippy_scales.evaluation.walk_forward import DEFAULT_STORAGE

if TYPE_CHECKING:
    import optuna

    from snippy_scales.data.futures import RollRule
    from snippy_scales.evaluation import StudyReport, WalkForward
    from snippy_scales.execution import Costs, MarketData
    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

app = typer.Typer(help="Walk-forward evaluation: Optuna studies on Nautilus backtests.")

_STRATEGY_HELP = f"Strategy name: {', '.join(STRATEGY_NAMES)}"
_STORAGE_HELP = "Optuna storage URL, or 'memory' to keep nothing"


def _storage(value: str) -> str | None:
    return None if value == "memory" else value


def _walk_forward(
    data: MarketData,
    *,
    splits: int,
    test_size: float,
    min_train: int,
    costs: Costs,
    capital: float,
    band: float,
) -> WalkForward:
    from snippy_scales.evaluation import WalkForward, make_folds

    try:
        folds = make_folds(data.closes(), n_splits=splits, test_size=test_size, min_train=min_train)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(
        f"[cyan]Folds:[/] {len(folds)} test windows, "
        f"{folds[0].test_start:%Y-%m-%d} → {folds[-1].test_end:%Y-%m-%d}"
    )
    return WalkForward(data, folds, costs=costs, initial_capital=capital, rebalance_band=band)


def _print_report(report: StudyReport) -> None:
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column(justify="right")
    pbo = "n/a (1 trial)" if report.pbo is None else f"{report.pbo:.2f}"
    rows = [
        ("study", report.study_name),
        ("symbols", ", ".join(report.symbols)),
        ("trials", str(report.n_trials)),
        ("best params", str(report.best_params or "defaults")),
        ("mean fold Sharpe", f"{report.mean_fold_sharpe:.3f}"),
        ("fold Sharpes", ", ".join(f"{s:.2f}" for s in report.fold_sharpes)),
        ("OOS Sharpe (stitched)", f"{report.sharpe:.3f}"),
        ("OOS return", f"{report.total_return:+.2%}"),
        ("OOS max drawdown", f"{report.max_drawdown:.2%}"),
        ("PSR (Sharpe > 0)", f"{report.psr:.3f}"),
        (f"DSR (best of {report.n_trials})", f"{report.dsr:.3f}"),
        ("PBO", pbo),
        ("min track record", f"{report.min_track_record_years:.1f} years"),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


def _run(
    strategy: str,
    symbols: list[str],
    schema: str,
    catalog_path: Path | None,
    *,
    space: dict[str, Any],
    search: str,
    trials: int | None,
    splits: int,
    test_size: float,
    min_train: int,
    costs: Costs,
    capital: float,
    band: float,
    storage: str,
    study_name: str | None,
    tearsheet: Path | None,
    roll: RollRule,
) -> None:
    from snippy_scales.evaluation import run_study, summarize_study

    cls = resolve_strategy(strategy)
    data = load_data(symbols, schema, catalog_path, roll)
    wf = _walk_forward(
        data,
        splits=splits,
        test_size=test_size,
        min_train=min_train,
        costs=costs,
        capital=capital,
        band=band,
    )
    try:
        study = run_study(
            cls,
            wf,
            space=space,
            sampler=search,
            n_trials=trials,
            study_name=study_name,
            storage=_storage(storage),
            returns_dir=EXPERIMENTS_DIR,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    report = summarize_study(study)
    _print_report(report)
    if tearsheet is not None:
        _write_tearsheet(report, tearsheet, None, schema, catalog_path)


# ── Commands ──────────────────────────────────────────────────────────────────

_SYMBOLS = typer.Option(..., "--symbol", "-s", help="Symbol; repeat for several")
_SCHEMA = typer.Option("ohlcv-1d", "--schema", help="Bar schema (e.g. ohlcv-1d)")
_CATALOG = typer.Option(None, "--catalog", help="Catalog root (default: data/catalog)")
_SPLITS = typer.Option(5, "--splits", "-n", help="Number of walk-forward folds")
_TEST_SIZE = typer.Option(0.2, "--test-size", help="Bars per fold: fraction of all bars")
_MIN_TRAIN = typer.Option(252, "--min-train", help="Warm-up bars before the first fold")
_CAPITAL = typer.Option(1_000_000.0, "--capital", help="Initial capital")
_FEES = typer.Option(0.001, "--fees", help="Commission as a fraction of notional")
_PER_CONTRACT = typer.Option(
    0.0, "--per-contract", help="Commission per contract (overrides --fees)"
)
_SLIPPAGE = typer.Option(0.0, "--slippage", help="Probability of one tick of slippage")
_BAND = typer.Option(0.0, "--band", help="Skip trades below this fraction of equity")
_STORAGE = typer.Option(DEFAULT_STORAGE, "--storage", help=_STORAGE_HELP)
_STUDY = typer.Option(None, "--study", help="Study name (default: strategy-symbols-time)")
_TEARSHEET = typer.Option(None, "--tearsheet", help="Write a tearsheet (.html) here")
_ROLL = typer.Option("volume", "--roll", help=ROLL_HELP)
_ROLL_DAYS = typer.Option(5, "--roll-days", help="Calendar roll: days before expiry")
_ROLL_MONTHS = typer.Option(None, "--roll-months", help="Contract months to trade, e.g. HMUZ")


@app.command()
def run(
    strategy: str = typer.Argument(..., help=_STRATEGY_HELP),
    symbol: list[str] = _SYMBOLS,
    schema: str = _SCHEMA,
    catalog_path: Path = _CATALOG,
    params: str = typer.Option(None, "--params", help="Fixed parameters as a JSON object"),
    splits: int = _SPLITS,
    test_size: float = _TEST_SIZE,
    min_train: int = _MIN_TRAIN,
    capital: float = _CAPITAL,
    fees: float = _FEES,
    per_contract: float = _PER_CONTRACT,
    slippage: float = _SLIPPAGE,
    band: float = _BAND,
    storage: str = _STORAGE,
    study: str = _STUDY,
    tearsheet: Path = _TEARSHEET,
    roll: str = _ROLL,
    roll_days: int = _ROLL_DAYS,
    roll_months: str = _ROLL_MONTHS,
) -> None:
    """Walk-forward one parameter set (defaults unless --params is given)."""
    fixed = parse_json(params, "--params")
    if any(isinstance(v, list | dict) for v in fixed.values()):
        console.print("[red]--params for `run` must be fixed values; use `sweep` to search.[/]")
        raise typer.Exit(1)
    _run(
        strategy,
        symbol,
        schema,
        catalog_path,
        space=fixed,
        search="tpe",
        trials=1,
        splits=splits,
        test_size=test_size,
        min_train=min_train,
        costs=make_costs(fees, per_contract, slippage),
        capital=capital,
        band=band,
        storage=storage,
        study_name=study,
        tearsheet=tearsheet,
        roll=make_roll(roll, roll_days, roll_months),
    )


@app.command()
def sweep(
    strategy: str = typer.Argument(..., help=_STRATEGY_HELP),
    symbol: list[str] = _SYMBOLS,
    params: str = typer.Option(
        ...,
        "--params",
        help='Search space: {"name": [values]} or {"name": {"low": 1, "high": 9}}',
    ),
    search: str = typer.Option("tpe", "--search", help="tpe, random or grid"),
    trials: int = typer.Option(None, "--trials", help="Trials (default: grid size, else 20)"),
    schema: str = _SCHEMA,
    catalog_path: Path = _CATALOG,
    splits: int = _SPLITS,
    test_size: float = _TEST_SIZE,
    min_train: int = _MIN_TRAIN,
    capital: float = _CAPITAL,
    fees: float = _FEES,
    per_contract: float = _PER_CONTRACT,
    slippage: float = _SLIPPAGE,
    band: float = _BAND,
    storage: str = _STORAGE,
    study: str = _STUDY,
    tearsheet: Path = _TEARSHEET,
    roll: str = _ROLL,
    roll_days: int = _ROLL_DAYS,
    roll_months: str = _ROLL_MONTHS,
) -> None:
    """Search parameters by mean walk-forward Sharpe; report DSR and PBO."""
    _run(
        strategy,
        symbol,
        schema,
        catalog_path,
        space=parse_json(params, "--params"),
        search=search,
        trials=trials,
        splits=splits,
        test_size=test_size,
        min_train=min_train,
        costs=make_costs(fees, per_contract, slippage),
        capital=capital,
        band=band,
        storage=storage,
        study_name=study,
        tearsheet=tearsheet,
        roll=make_roll(roll, roll_days, roll_months),
    )


@app.command(name="run-all")
def run_all(
    schema: str = _SCHEMA,
    catalog_path: Path = _CATALOG,
    splits: int = _SPLITS,
    test_size: float = _TEST_SIZE,
    min_train: int = _MIN_TRAIN,
    capital: float = _CAPITAL,
    fees: float = _FEES,
    per_contract: float = _PER_CONTRACT,
    slippage: float = _SLIPPAGE,
    band: float = _BAND,
    storage: str = _STORAGE,
    roll: str = _ROLL,
    roll_days: int = _ROLL_DAYS,
    roll_months: str = _ROLL_MONTHS,
) -> None:
    """Walk-forward every strategy (default parameters) on every catalog symbol.

    Outright futures contracts are evaluated as one rolled parent symbol per root
    (``ES.FUT``). Single-asset strategies run per symbol; multi-asset strategies
    run once per venue on all its symbols, over the dates they share. Symbols
    that can't be loaded (e.g. no instrument definition) are skipped.
    """
    from snippy_scales.data import catalog as cat
    from snippy_scales.data.futures import tradable_symbols
    from snippy_scales.evaluation import run_study, summarize_study
    from snippy_scales.execution import load_market_data
    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy as _Multi

    series = cat.list_bar_series(schema, catalog_path)
    if not series:
        console.print(
            f"[red]No {schema!r} bars found in the catalog.[/]  Run [cyan]algo data sync[/] first."
        )
        raise typer.Exit(1)
    series = tradable_symbols(cat.open_catalog(catalog_path), series)
    console.print(f"[cyan]Discovered {len(series)} symbol(s)[/]")
    rule = make_roll(roll, roll_days, roll_months)
    data: MarketData | None = None
    by_venue: dict[str, list[str]] = {}
    for symbol, venue in series:
        if any(symbol in v for v in by_venue.values()):
            console.print(f"  [yellow]skip[/] {symbol}.{venue}: symbol already loaded")
            continue
        try:
            part = load_market_data(
                symbol, schema, venue=venue, roll=rule, catalog_path=catalog_path
            )
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"  [yellow]skip[/] {symbol}: {exc}")
            continue
        data = part if data is None else data.merge(part)
        by_venue.setdefault(venue, []).append(symbol)
    if data is None:
        console.print("[red]No symbol could be loaded.[/]")
        raise typer.Exit(1)
    costs = make_costs(fees, per_contract, slippage)

    jobs: list[tuple[str, type[Strategy] | type[MultiAssetStrategy], list[str]]] = []
    for name, cls in registry().items():
        if issubclass(cls, _Multi):
            for venue, symbols in by_venue.items():
                if len(symbols) < 2:  # noqa: PLR2004
                    console.print(f"[yellow]Skipping {name} on {venue}:[/] needs 2+ symbols.")
                    continue
                jobs.append((name, cls, symbols))
        else:
            jobs.extend((name, cls, [s]) for symbols in by_venue.values() for s in symbols)

    summary: list[tuple[str, str, StudyReport]] = []
    for name, cls, job_symbols in jobs:
        label = job_symbols[0] if len(job_symbols) == 1 else f"{len(job_symbols)} symbols"
        console.print(f"[bold cyan]>> {name}[/] [dim]({label})[/]")
        try:
            wf = _walk_forward(
                data.select(job_symbols),
                splits=splits,
                test_size=test_size,
                min_train=min_train,
                costs=costs,
                capital=capital,
                band=band,
            )
            study = run_study(cls, wf, storage=_storage(storage), returns_dir=EXPERIMENTS_DIR)
            report = summarize_study(study)
        except (ValueError, typer.Exit) as exc:
            console.print(f"  [red]error:[/] {exc}")
            continue
        summary.append((name, label, report))
        console.print(
            f"  [green]done[/]  sharpe={report.sharpe:.3f}  return={report.total_return:+.2%}"
        )

    if not summary:
        console.print("\n[yellow]No results to display.[/]")
        return
    console.print("\n[bold]── Summary (sorted by stitched OOS Sharpe) ──[/]")
    table = Table(header_style="bold cyan")
    for column in ("strategy", "symbol(s)", "oos_sharpe", "return", "max_dd", "psr", "study"):
        table.add_column(
            column, justify="left" if column in ("strategy", "symbol(s)", "study") else "right"
        )
    for name, label, report in sorted(summary, key=lambda r: r[2].sharpe, reverse=True):
        table.add_row(
            name,
            label,
            f"{report.sharpe:.3f}",
            f"{report.total_return:+.2%}",
            f"{report.max_drawdown:.2%}",
            f"{report.psr:.2f}",
            report.study_name,
        )
    console.print(table)


def _load_study(name: str, storage: str) -> optuna.Study:
    import optuna

    try:
        return optuna.load_study(study_name=name, storage=storage)
    except KeyError as exc:
        console.print(f"[red]No study {name!r} in {storage}.[/]  See [cyan]algo eval list[/].")
        raise typer.Exit(1) from exc


@app.command(name="list")
def list_studies(storage: str = typer.Option(DEFAULT_STORAGE, "--storage")) -> None:
    """List saved studies."""
    import optuna

    summaries = optuna.get_all_study_summaries(storage=storage)
    if not summaries:
        console.print("[yellow]No studies yet.[/]  Run [cyan]algo eval run[/] first.")
        return
    table = Table(header_style="bold cyan")
    for column in ("study", "trials", "best (mean fold Sharpe)", "started"):
        table.add_column(column)
    for s in sorted(summaries, key=lambda s: s.datetime_start or 0, reverse=True):
        best = s.best_trial.value if s.best_trial is not None else None
        table.add_row(
            s.study_name,
            str(s.n_trials),
            "—" if best is None else f"{best:.3f}",
            f"{s.datetime_start:%Y-%m-%d %H:%M}" if s.datetime_start else "—",
        )
    console.print(table)


@app.command()
def report(
    study: str = typer.Argument(..., help="Study name (see `algo eval list`)"),
    storage: str = typer.Option(DEFAULT_STORAGE, "--storage"),
) -> None:
    """Print the out-of-sample report (Sharpe, PSR, DSR, PBO) of a saved study."""
    from snippy_scales.evaluation import summarize_study

    _print_report(summarize_study(_load_study(study, storage)))


def _write_tearsheet(
    report: StudyReport,
    output: Path,
    benchmark: str | None,
    schema: str,
    catalog_path: Path | None,
) -> None:
    import polars as pl

    from snippy_scales.data.catalog import load_bars
    from snippy_scales.evaluation import write_tearsheet

    bench = None
    if benchmark:
        try:
            bars = load_bars(benchmark, schema, catalog_path=catalog_path)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Benchmark: {exc}[/]")
            raise typer.Exit(1) from exc
        bench = bars.select(
            pl.col("ts_init").dt.truncate("1d").alias("ts"),
            pl.col("close").pct_change().alias("return"),
        ).drop_nulls()
    write_tearsheet(report, output, benchmark=bench, benchmark_name=benchmark or "Benchmark")
    console.print(f"[green]Tearsheet:[/] {output}")


@app.command()
def tearsheet(
    study: str = typer.Argument(..., help="Study name (see `algo eval list`)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file (default: reports/)"),  # noqa: B008
    benchmark: str = typer.Option(None, "--benchmark", "-b", help="Catalog symbol to overlay"),
    schema: str = _SCHEMA,
    catalog_path: Path = _CATALOG,
    storage: str = typer.Option(DEFAULT_STORAGE, "--storage"),
) -> None:
    """Render a Nautilus tearsheet of a study's stitched out-of-sample returns."""
    from snippy_scales.evaluation import summarize_study

    report = summarize_study(_load_study(study, storage))
    _write_tearsheet(
        report, output or Path("reports") / f"{study}.html", benchmark, schema, catalog_path
    )
