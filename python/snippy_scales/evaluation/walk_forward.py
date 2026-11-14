"""Walk-forward evaluation on Optuna and Nautilus.

A **study** searches a strategy's parameters. Each Optuna **trial** builds the
strategy with one parameter set and backtests it on every out-of-sample
**fold** through :func:`~snippy_scales.execution.run_backtest`. The trial's
value is the mean fold Sharpe. Per-fold statistics go into the trial's user
attributes, and the stitched out-of-sample daily returns into
``<returns_dir>/<study>/trial_<n>.parquet`` (needed for PBO and tearsheets).

Folds are contiguous test windows at the end of the history. Signals for a
fold are computed on all bars up to its end, so strategies are warm on the
first test bar; the earlier bars are the fold's (expanding) training window.
The built-in strategies fit nothing, so parameters are chosen on the folds
themselves, as before the migration. :func:`summarize_study` therefore reports
the Deflated Sharpe Ratio (selection over N trials) and the Probability of
Backtest Overfitting alongside the raw numbers.

Usage::

    data = load_market_data("ES.c.0")
    wf = WalkForward(data, make_folds(data.closes(), n_splits=5))
    study = run_study(TrendFollowing, wf, space={"fast_period": [10, 20, 40]}, sampler="grid")
    report = summarize_study(study)
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nautilus_trader.analysis as analysis
import numpy as np
import optuna
import polars as pl
from scipy import stats as sps

from snippy_scales._constants import TRADING_DAYS_PER_YEAR
from snippy_scales.data.paths import EXPERIMENTS_DIR
from snippy_scales.evaluation.statistics import (
    deflated_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from snippy_scales.execution import ENGINES, Costs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from snippy_scales.execution import BacktestRun, MarketData
    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

    StrategyClass = type[Strategy] | type[MultiAssetStrategy]

#: Default Optuna storage: one SQLite file for every study.
DEFAULT_STORAGE = f"sqlite:///{EXPERIMENTS_DIR / 'optuna.db'}"

#: Nautilus statistics computed on stitched out-of-sample returns.
_RETURN_STATISTICS = (
    "SharpeRatio",
    "SortinoRatio",
    "CalmarRatio",
    "CAGR",
    "MaxDrawdown",
    "ReturnsVolatility",
    "ReturnsAverage",
    "ReturnsAverageWin",
    "ReturnsAverageLoss",
    "ReturnsSkewness",
    "ReturnsKurtosis",
    "ProfitFactor",
    "RiskReturnRatio",
    "TailRatio",
    "OmegaRatio",
    "UlcerIndex",
    "ValueAtRisk",
    "ExpectedShortfall",
)
_SHARPE = "Sharpe Ratio (252 days)"

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Folds ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, as inclusive bar-close bounds (UTC)."""

    index: int
    train_start: dt.datetime
    train_end: dt.datetime
    test_start: dt.datetime
    test_end: dt.datetime

    def to_json(self) -> dict[str, Any]:
        return {
            k: v.isoformat() if isinstance(v, dt.datetime) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Fold:
        return cls(
            index=int(raw["index"]),
            **{k: dt.datetime.fromisoformat(raw[k]) for k in raw if k != "index"},
        )


def make_folds(
    closes: pl.Series,
    *,
    n_splits: int = 5,
    test_size: float | int = 0.2,
    min_train: int = TRADING_DAYS_PER_YEAR,
) -> list[Fold]:
    """Split bar closes into *n_splits* contiguous test windows at the end.

    Args:
        closes: Sorted bar closes (e.g. :meth:`MarketData.closes`).
        n_splits: Number of folds.
        test_size: Bars per test window, as a fraction of all bars or a count.
            Capped so at least *min_train* bars precede the first test window.
        min_train: Bars reserved before the first fold for warm-up.

    Raises:
        ValueError: If there are too few bars for one bar per test window.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    n = closes.len()
    requested = max(1, int(n * test_size)) if isinstance(test_size, float) else int(test_size)
    test_n = min(requested, (n - min_train) // n_splits)
    if test_n < 1:
        raise ValueError(
            f"{n} bars are too few for {n_splits} folds after {min_train} warm-up bars"
        )
    ts = closes.to_list()
    folds = []
    for i in range(n_splits):
        first = n - (n_splits - i) * test_n
        folds.append(
            Fold(
                index=i,
                train_start=ts[0],
                train_end=ts[max(first - 1, 0)],
                test_start=ts[first],
                test_end=ts[first + test_n - 1],
            )
        )
    return folds


# ── Search space ──────────────────────────────────────────────────────────────


def suggest(trial: optuna.trial.BaseTrial, space: Mapping[str, Any]) -> dict[str, Any]:
    """Draw one parameter set from *space*.

    Each value is a list (categorical choice), a ``{"low", "high", "step"?,
    "log"?}`` range (integer if both bounds are ints, else float), or a scalar
    (fixed).
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        elif isinstance(spec, dict):
            low, high = spec["low"], spec["high"]
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(
                    name, low, high, step=spec.get("step", 1), log=spec.get("log", False)
                )
            else:
                params[name] = trial.suggest_float(
                    name, low, high, step=spec.get("step"), log=spec.get("log", False)
                )
        else:
            params[name] = spec
    return params


def _sampler(kind: str, space: Mapping[str, Any], seed: int) -> optuna.samplers.BaseSampler:
    if kind == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if kind == "grid":
        grid = {k: v for k, v in space.items() if isinstance(v, list)}
        if len(grid) != sum(1 for v in space.values() if isinstance(v, list | dict)):
            raise ValueError("grid search needs every searched parameter as a list of values")
        return optuna.samplers.GridSampler(grid, seed=seed)
    raise ValueError(f"unknown sampler {kind!r}; use tpe, random or grid")


# ── Running ───────────────────────────────────────────────────────────────────


def bar_returns(bars: pl.DataFrame) -> pl.DataFrame:
    """Close-to-close returns of *bars*, labelled as Nautilus labels daily returns.

    Nautilus (and therefore every :class:`BacktestRun.returns` and stitched
    study series) labels the return from one day's equity to the next with the
    day it **starts** on. A bar's return ends at its close (``ts_init``), so it
    is labelled with the day before that close's day. Labelling it with the
    close's own day puts a benchmark one day late against strategy returns: its
    correlation with a market-exposed strategy then collapses to ~0.

    Args:
        bars: At least ``ts_init`` (bar close) and ``close``.

    Returns:
        ``ts`` (``Datetime[ns, UTC]``, midnight), ``return``.
    """
    return (
        bars.sort("ts_init")
        .select(
            (pl.col("ts_init").dt.truncate("1d") - pl.duration(days=1))
            .cast(pl.Datetime("ns", "UTC"))
            .alias("ts"),
            pl.col("close").pct_change().alias("return"),
        )
        .drop_nulls()
    )


def stitch_returns(runs: Sequence[BacktestRun]) -> pl.DataFrame:
    """Concatenate fold returns into one daily series (compounding shared days)."""
    frames = [r.returns for r in runs if r.returns.height]
    if not frames:
        return pl.DataFrame(schema={"ts": pl.Datetime("ns", "UTC"), "return": pl.Float64})
    return (
        pl.concat(frames)
        .group_by("ts")
        .agg(((1.0 + pl.col("return")).product() - 1.0).alias("return"))
        .sort("ts")
    )


@dataclass(frozen=True)
class WalkForward:
    """Market data, folds and backtest settings shared by every trial.

    *engine* picks the backtester: ``"nautilus"`` (the event-driven reference) or
    ``"vector"`` (:mod:`snippy_scales.execution.vector`, a numba kernel that
    reproduces Nautilus's fills, returns and return statistics on bars with
    ample volume, for screening). Validate the trials you keep on Nautilus.
    """

    data: MarketData
    folds: Sequence[Fold]
    costs: Costs = field(default_factory=Costs)
    initial_capital: float = 1_000_000.0
    rebalance_band: float = 0.0
    leverage: float = 10.0
    engine: str = "nautilus"

    def __post_init__(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {sorted(ENGINES)}, got {self.engine!r}")

    def run_folds(self, research: Strategy | MultiAssetStrategy) -> list[BacktestRun]:
        """Backtest *research* on every fold's test window."""
        run_backtest = ENGINES[self.engine]
        return [
            run_backtest(
                research,
                self.data,
                start=fold.test_start,
                end=fold.test_end,
                initial_capital=self.initial_capital,
                costs=self.costs,
                rebalance_band=self.rebalance_band,
                leverage=self.leverage,
            )
            for fold in self.folds
        ]


def _fold_summary(run: BacktestRun) -> dict[str, float | int]:
    return {
        "sharpe": run.sharpe,
        "total_return": run.total_return,
        "max_drawdown": float(run.stats_returns.get("Max Drawdown", 0.0) or 0.0),
        "orders": run.total_orders,
        "rejections": len(run.rejections),
    }


def run_study(
    strategy_cls: StrategyClass,
    wf: WalkForward,
    *,
    space: Mapping[str, Any] | None = None,
    sampler: str = "tpe",
    n_trials: int | None = None,
    study_name: str | None = None,
    storage: str | None = DEFAULT_STORAGE,
    returns_dir: Path = EXPERIMENTS_DIR,
    seed: int = 42,
) -> optuna.Study:
    """Search *strategy_cls* parameters by walk-forward Sharpe.

    Args:
        strategy_cls: Research strategy class; trials call ``strategy_cls(**params)``.
        wf: Data, folds and backtest settings.
        space: Search space (see :func:`suggest`). ``None`` runs the defaults once.
        sampler: ``"tpe"``, ``"random"`` or ``"grid"``.
        n_trials: Trials to run. Defaults to the grid size for ``grid``, else 1
            without a space and 20 with one.
        study_name: Optuna study name. Defaults to ``<Class>-<symbols>-<UTC time>``.
        storage: Optuna storage URL; ``None`` keeps the study in memory. Several
            processes can share one storage (and study name) to run in parallel.
        returns_dir: Where per-trial OOS returns are written.
        seed: Sampler seed.

    Returns:
        The Optuna study.
    """
    space = dict(space or {})
    if n_trials is None and sampler != "grid":
        n_trials = 20 if space else 1
    if study_name is None:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
        study_name = f"{strategy_cls.__name__}-{'+'.join(wf.data.symbols)}-{stamp}"
    if storage is not None and storage.startswith("sqlite:///"):
        Path(storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    trial_dir = returns_dir / study_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=_sampler(sampler, space, seed),
        load_if_exists=True,
    )
    study.set_user_attr("strategy", f"{strategy_cls.__module__}.{strategy_cls.__qualname__}")
    study.set_user_attr("symbols", wf.data.symbols)
    study.set_user_attr("bar_types", list(wf.data.bar_types.values()))
    study.set_user_attr("folds", [f.to_json() for f in wf.folds])
    study.set_user_attr("costs", asdict(wf.costs))
    study.set_user_attr("initial_capital", wf.initial_capital)
    study.set_user_attr("rebalance_band", wf.rebalance_band)
    study.set_user_attr("engine", wf.engine)
    study.set_user_attr("space", json.loads(json.dumps(space, default=str)))
    study.set_user_attr("returns_dir", str(trial_dir))

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        runs = wf.run_folds(strategy_cls(**params))
        folds = [_fold_summary(r) for r in runs]
        path = trial_dir / f"trial_{trial.number}.parquet"
        stitch_returns(runs).write_parquet(path)
        trial.set_user_attr("params", params)
        trial.set_user_attr("folds", folds)
        trial.set_user_attr("returns_path", str(path))
        return float(np.mean([f["sharpe"] for f in folds]))

    study.optimize(objective, n_trials=n_trials)
    return study


# ── Reporting ─────────────────────────────────────────────────────────────────


def returns_statistics(returns: pl.DataFrame) -> dict[str, float]:
    """Nautilus return statistics for a daily ``ts``/``return`` series."""
    analyzer = analysis.PortfolioAnalyzer()
    for name in _RETURN_STATISTICS:
        analyzer.register_statistic(getattr(analysis, name)())
    ts = returns["ts"].dt.epoch("ns").to_list()
    for t, r in zip(ts, returns["return"].to_list(), strict=True):
        analyzer.add_return(int(t), float(r))
    out = analyzer.get_performance_stats_returns()
    return {k: float(v) for k, v in out.items() if v is not None and not math.isnan(float(v))}


def _trial_returns(trial: optuna.trial.FrozenTrial) -> pl.DataFrame:
    return pl.read_parquet(trial.user_attrs["returns_path"])


def periods_per_year(returns: pl.DataFrame) -> float:
    """Observed return periods per year of a ``ts``/``return`` series.

    Nautilus samples equity on **every calendar day** (weekends and holidays are
    zero returns), so a daily series has ~365 periods a year, not 252.
    Annualising it by √252, as Nautilus's ``Sharpe Ratio (252 days)`` does,
    understates the Sharpe of a weekday-traded asset by √(252/365) ≈ 0.83, and
    dividing its observation count by 252 overstates track-record years by ~45%.
    Falls back to :data:`TRADING_DAYS_PER_YEAR` for fewer than two returns.
    """
    if returns.height < 2:  # noqa: PLR2004
        return float(TRADING_DAYS_PER_YEAR)
    first, last = returns["ts"].min(), returns["ts"].max()
    days = (last - first).total_seconds() / 86_400 + 1  # type: ignore[operator, union-attr]
    return returns.height / (days / 365.25)


def _annualised_sharpe(returns: np.ndarray, periods: float = TRADING_DAYS_PER_YEAR) -> float:
    if returns.size < 2 or returns.std(ddof=1) == 0.0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(periods))


@dataclass(frozen=True)
class StudyReport:
    """Out-of-sample summary of a study's best trial, deflated for the search.

    Attributes:
        study_name: Optuna study name.
        strategy: Dotted path of the strategy class.
        symbols: Symbols traded.
        n_trials: Completed trials (the DSR's N).
        best_params: Parameters of the best trial.
        mean_fold_sharpe: The best trial's objective value.
        fold_sharpes: The best trial's Sharpe per fold.
        returns: The best trial's stitched OOS daily returns.
        stats_returns: Nautilus statistics of those returns.
        sharpe: Annualised Sharpe of the stitched returns, by their observed
            frequency (:func:`periods_per_year`). Nautilus's ``Sharpe Ratio
            (252 days)`` in *stats_returns* annualises the same calendar-day
            series by √252, so it is ~0.83× this for weekday-traded assets.
        total_return: Compounded stitched return, as a fraction.
        max_drawdown: Worst peak-to-trough fall of the stitched equity, ``<= 0``.
        psr: Probability the true Sharpe is above zero.
        dsr: PSR against the expected best Sharpe of ``n_trials`` skill-less trials.
        min_track_record_years: Years of returns needed for 95% confidence
            (observations needed / observed periods per year).
        pbo: Probability of Backtest Overfitting over all trials (``None`` if < 2).
    """

    study_name: str
    strategy: str
    symbols: list[str]
    n_trials: int
    best_params: dict[str, Any]
    mean_fold_sharpe: float
    fold_sharpes: list[float]
    returns: pl.DataFrame = field(repr=False)
    stats_returns: dict[str, float] = field(repr=False)
    sharpe: float
    total_return: float
    max_drawdown: float
    psr: float
    dsr: float
    min_track_record_years: float
    pbo: float | None


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE
    ]


def summarize_study(
    study: optuna.Study, *, pbo_partitions: int = 8, selection: optuna.Study | None = None
) -> StudyReport:
    """Compute the out-of-sample report for *study*'s best trial.

    Args:
        study: The study whose best trial is reported.
        pbo_partitions: CSCV partitions for the PBO.
        selection: The study the reported trials were *chosen from*, when that is
            a different one. Typically *study* re-runs a screening study's top
            trials on Nautilus: the DSR must then be deflated for every screened
            trial (``n_trials`` and the spread of their Sharpes), and the PBO
            computed over them, not over the handful re-run.

    Raises:
        ValueError: If the study (or *selection*) has no completed trial.
    """
    trials = _completed(study)
    if not trials:
        raise ValueError(f"study {study.study_name!r} has no completed trials")
    best = max(trials, key=lambda t: t.value if t.value is not None else -math.inf)
    returns = _trial_returns(best)
    values = returns["return"].to_numpy()

    if selection is not None:
        trials = _completed(selection)
        if not trials:
            raise ValueError(f"study {selection.study_name!r} has no completed trials")
    trial_returns = {t.number: _trial_returns(t) for t in trials}
    trial_sharpes = np.array(
        [
            _annualised_sharpe(r["return"].to_numpy(), periods_per_year(r))
            for r in trial_returns.values()
        ]
    )
    periods = periods_per_year(returns)
    sharpe = _annualised_sharpe(values, periods)
    n_obs = values.size
    skew = float(sps.skew(values)) if n_obs > 2 else 0.0
    kurt = float(sps.kurtosis(values, fisher=False)) if n_obs > 3 else 3.0
    trial_std = float(trial_sharpes.std(ddof=1)) if trial_sharpes.size > 1 else 0.0
    common = dict(skew=skew, kurtosis=kurt, periods_per_year=periods)

    pbo: float | None = None
    if len(trials) > 1:
        matrix: pl.DataFrame | None = None
        for number, frame in trial_returns.items():
            column = frame.rename({"return": f"t{number}"})
            matrix = column if matrix is None else matrix.join(column, on="ts", how="inner")
        assert matrix is not None  # noqa: S101
        pbo = probability_of_backtest_overfitting(
            matrix.drop("ts").to_numpy(), n_partitions=pbo_partitions
        )

    equity = np.cumprod(1.0 + values) if n_obs else np.array([1.0])
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0)) if n_obs else 0.0
    min_trl = min_track_record_length(sharpe, **common)
    return StudyReport(
        study_name=study.study_name,
        strategy=str(study.user_attrs.get("strategy", "")),
        symbols=list(study.user_attrs.get("symbols", [])),
        n_trials=len(trials),
        best_params=dict(best.user_attrs.get("params", best.params)),
        mean_fold_sharpe=float(best.value or 0.0),
        fold_sharpes=[float(f["sharpe"]) for f in best.user_attrs.get("folds", [])],
        returns=returns,
        stats_returns=returns_statistics(returns) if n_obs else {},
        sharpe=sharpe,
        total_return=float(equity[-1] - 1.0),
        max_drawdown=drawdown,
        psr=probabilistic_sharpe_ratio(sharpe, n_obs=n_obs, **common),
        dsr=deflated_sharpe_ratio(
            sharpe, n_trials=len(trials), n_obs=n_obs, trial_std=trial_std, **common
        ),
        min_track_record_years=min_trl / periods,
        pbo=pbo,
    )


def load_strategy_class(path: str) -> StrategyClass:
    """Import a strategy class from its dotted path (as stored on a study)."""
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def write_tearsheet(
    report: StudyReport,
    path: Path,
    *,
    benchmark: pl.DataFrame | None = None,
    benchmark_name: str = "Benchmark",
) -> Path:
    """Render a Nautilus tearsheet of the stitched OOS returns to *path* (``.html``).

    Args:
        report: Study report.
        path: Output file; the extension selects the format.
        benchmark: Optional daily ``ts``/``return`` series to overlay.
        benchmark_name: Label for the benchmark.
    """
    import pandas as pd

    def series(frame: pl.DataFrame) -> pd.Series:
        index = pd.DatetimeIndex(frame["ts"].to_list())
        return pd.Series(frame["return"].to_numpy(), index=index)

    path.parent.mkdir(parents=True, exist_ok=True)
    analysis.create_tearsheet_from_stats(
        {},
        report.stats_returns,
        {
            "DSR": report.dsr,
            "PSR": report.psr,
            "PBO": report.pbo if report.pbo is not None else math.nan,
        },
        series(report.returns),
        output_path=str(path),
        title=f"{report.study_name} — out-of-sample (best of {report.n_trials})",
        benchmark_returns=series(benchmark) if benchmark is not None else None,
        benchmark_name=benchmark_name,
    )
    return path
