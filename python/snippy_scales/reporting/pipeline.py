"""The computations behind a strategy report, kept out of the notebook so they are tested.

A report runs three stages on one :class:`~snippy_scales.reporting.ReportConfig`:

1. **Screening** — an Optuna walk-forward study on the fast engine (the numba
   kernel, :mod:`snippy_scales.execution.vector`), searching the parameter space.
2. **Validation** — the *top_k* screened parameter sets re-run, fold for fold,
   on Nautilus (the event-driven reference). The best of them is reported with
   its Deflated Sharpe Ratio deflated for **every screened trial**, not just the
   re-run ones, since that is the search it was selected from.
3. **Full backtest** — the validated best parameters on Nautilus over the whole
   evaluated period, for Nautilus's own statistics, fills and equity curve.

Studies live in the report's own SQLite file (``reports/<name>/optuna.db``) and
are resumed on re-runs: only missing trials are run. Each study records the
config's :meth:`~ReportConfig.fingerprint`; a changed config raises instead of
mixing trials run under different settings.
"""

from __future__ import annotations

import datetime as dt
import inspect
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import polars as pl

from snippy_scales._constants import TRADING_DAYS_PER_YEAR
from snippy_scales.data import catalog as cat
from snippy_scales.evaluation import (
    WalkForward,
    bar_returns,
    make_folds,
    run_study,
    summarize_study,
)
from snippy_scales.execution import Costs, load_market_data, run_backtest
from snippy_scales.reporting.config import REPO_ROOT

if TYPE_CHECKING:
    from snippy_scales.evaluation import Fold, StudyReport
    from snippy_scales.execution import BacktestRun, MarketData
    from snippy_scales.reporting.config import ReportConfig
    from snippy_scales.strategies import MultiAssetStrategy, Strategy

#: Where report instances live (git-ignored apart from ``templates/``).
REPORTS_DIR = Path("reports")

_COMPLETE = optuna.trial.TrialState.COMPLETE


# ── Paths ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReportPaths:
    """Files of one report instance, all under ``reports/<name>/``."""

    root: Path

    @classmethod
    def named(cls, name: str, reports_dir: Path = REPORTS_DIR) -> ReportPaths:
        return cls(reports_dir / name)

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def notebook(self) -> Path:
        return self.root / "report.py"

    @property
    def html(self) -> Path:
        return self.root / "report.html"

    @property
    def notes(self) -> Path:
        return self.root / "notes.md"

    @property
    def tearsheet(self) -> Path:
        return self.root / "tearsheet.html"

    @property
    def trials_dir(self) -> Path:
        return self.root / "trials"

    @property
    def storage(self) -> str:
        return f"sqlite:///{(self.root / 'optuna.db').resolve()}"


# ── Strategy ──────────────────────────────────────────────────────────────────


def theory_markdown(cfg: ReportConfig) -> str | None:
    """The theory write-up with its title dropped and headings demoted one level.

    ``docs/strategies/<strategy>.md`` is a standalone MkDocs page (``# Title``,
    ``## Section``); in the report it sits under the "Theory" section heading.
    Returns ``None`` if the file does not exist.
    """
    path = cfg.theory_path
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    fenced = False
    out = []
    for line in lines:
        if line.lstrip().startswith("```"):
            fenced = not fenced
        out.append("#" + line if line.startswith("#") and not fenced else line)
    return "\n".join(out).strip()


def strategy_defaults(cls: type) -> dict[str, Any]:
    """Constructor defaults of a strategy class (parameters not searched keep these)."""
    return {
        name: p.default
        for name, p in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and p.default is not inspect.Parameter.empty
    }


def space_table(cfg: ReportConfig) -> pl.DataFrame:
    """Every constructor parameter: how it is searched (or fixed) and its default."""
    defaults = strategy_defaults(strategy_class(cfg))
    space = cfg.screening.space
    rows = []
    for name in dict.fromkeys([*defaults, *space]):
        spec = space.get(name)
        if isinstance(spec, list):
            kind, values = "categorical", ", ".join(map(str, spec))
        elif isinstance(spec, dict):
            kind = "log range" if spec.get("log") else "range"
            step = f" step {spec['step']}" if spec.get("step") else ""
            values = f"[{spec['low']}, {spec['high']}]{step}"
        elif name in space:
            kind, values = "fixed", str(spec)
        else:
            kind, values = "default", str(defaults.get(name))
        rows.append(
            {
                "parameter": name,
                "search": kind,
                "values": values,
                "default": str(defaults.get(name)),
            }
        )
    return pl.DataFrame(rows)


# ── Data ──────────────────────────────────────────────────────────────────────


def strategy_class(cfg: ReportConfig) -> type[Strategy] | type[MultiAssetStrategy]:
    """The research strategy class registered under ``cfg.strategy``."""
    from snippy_scales.cli._common import registry

    strategies = registry()
    if cfg.strategy not in strategies:
        raise ValueError(f"unknown strategy {cfg.strategy!r}; available: {', '.join(strategies)}")
    return strategies[cfg.strategy]


def load_universe(cfg: ReportConfig, catalog_path: Path | None = None) -> MarketData:
    """Load every configured symbol that has bars in the catalog.

    Raises:
        FileNotFoundError: If none of them has.
    """
    d = cfg.data
    series = cat.list_bar_series(d.schema_, catalog_path)
    available = {s for s, v in series if d.venue is None or v == d.venue}
    symbols = d.resolve_symbols(available)
    if not symbols:
        raise FileNotFoundError(f"none of the configured symbols has {d.schema_} bars")
    return load_market_data(symbols, d.schema_, venue=d.venue, catalog_path=catalog_path)


def missing_symbols(cfg: ReportConfig, data: MarketData) -> list[str]:
    """Configured symbols that were not loaded (no bars in the catalog)."""
    wanted = cfg.data.resolve_symbols()
    return [s for s in wanted if s not in data.frames]


def calendar(cfg: ReportConfig, data: MarketData) -> pl.Series:
    """Every bar close of the universe within ``[start, end]`` (union over symbols)."""
    closes = data.closes(union=True)
    lo, hi = cat.to_ns(cfg.data.start), cat.to_ns(cfg.data.end)
    ns = closes.dt.epoch("ns")
    keep = pl.Series([True] * closes.len())
    if lo is not None:
        keep = keep & (ns >= lo)
    if hi is not None:
        keep = keep & (ns <= hi)
    return closes.filter(keep)


def data_summary(data: MarketData) -> pl.DataFrame:
    """One row per symbol: first and last bar close, bar count, zero-volume bars.

    Nautilus treats a bar's volume as its liquidity, so it fills nothing on a
    zero-volume bar ("No market") while the screening kernel fills everything
    at the close: zero-volume bars are where the two engines can disagree.
    """
    rows = [
        {
            "symbol": s,
            "first": f["ts_init"].min(),
            "last": f["ts_init"].max(),
            "bars": f.height,
            "zero_volume_bars": int((f["volume"] <= 0).sum()) if "volume" in f.columns else 0,
        }
        for s, f in data.frames.items()
    ]
    return pl.DataFrame(rows).sort("symbol")


def universe_size(data: MarketData) -> pl.DataFrame:
    """Symbols with a bar at each bar close (``ts``, ``symbols``)."""
    return (
        pl.concat([f.select(pl.col("ts_init").alias("ts")) for f in data.frames.values()])
        .group_by("ts")
        .len("symbols")
        .sort("ts")
    )


def daily_returns(frame: pl.DataFrame) -> pl.DataFrame:
    """Close-to-close returns labelled like strategy returns (see :func:`bar_returns`)."""
    return bar_returns(frame)


def benchmark_bars(cfg: ReportConfig, catalog_path: Path | None = None) -> pl.DataFrame | None:
    """Bars of ``cfg.data.benchmark`` (``None`` if not configured)."""
    if not cfg.data.benchmark:
        return None
    return cat.load_bars(
        cfg.data.benchmark, cfg.data.schema_, venue=cfg.data.venue, catalog_path=catalog_path
    )


def benchmark_returns(cfg: ReportConfig, catalog_path: Path | None = None) -> pl.DataFrame | None:
    """Buy-and-hold daily returns of ``cfg.data.benchmark`` (``None`` if not configured)."""
    bars = benchmark_bars(cfg, catalog_path)
    return None if bars is None else daily_returns(bars)


# ── Walk-forward ──────────────────────────────────────────────────────────────


def costs(cfg: ReportConfig) -> Costs:
    e = cfg.evaluation
    return Costs(
        commission_rate=e.commission_rate,
        commission_per_contract=e.commission_per_contract,
        slippage_prob=e.slippage_prob,
    )


def folds(cfg: ReportConfig, data: MarketData) -> list[Fold]:
    e = cfg.evaluation
    return make_folds(
        calendar(cfg, data), n_splits=e.n_splits, test_size=e.test_size, min_train=e.min_train
    )


def walk_forward(cfg: ReportConfig, data: MarketData, engine: str) -> WalkForward:
    e = cfg.evaluation
    return WalkForward(
        data,
        folds(cfg, data),
        costs=costs(cfg),
        initial_capital=e.initial_capital,
        rebalance_band=e.rebalance_band,
        leverage=e.leverage,
        engine=engine,
    )


def fold_frame(fold_list: list[Fold]) -> pl.DataFrame:
    """The folds as a table, with their test-window length in bars' calendar days."""
    return pl.DataFrame(
        [
            {
                "fold": f.index,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "test_days": (f.test_end - f.test_start).days,
            }
            for f in fold_list
        ]
    )


def _study(name: str, storage: str, fingerprint: str) -> optuna.Study:
    study = optuna.create_study(
        study_name=name, storage=storage, direction="maximize", load_if_exists=True
    )
    found = study.user_attrs.get("fingerprint")
    if found is not None and found != fingerprint:
        raise ValueError(
            f"study {name!r} was run with a different config (fingerprint {found}, now "
            f"{fingerprint}). Create a new report (`algo report new`) or delete "
            f"{storage.removeprefix('sqlite:///')} to start over."
        )
    study.set_user_attr("fingerprint", fingerprint)
    return study


def _grid_size(space: dict[str, Any]) -> int:
    return math.prod(len(v) for v in space.values() if isinstance(v, list))


def trials_frame(study: optuna.Study) -> pl.DataFrame:
    """Completed trials: number, objective, duration, parameters and per-fold statistics."""
    rows = []
    for t in study.get_trials(deepcopy=False, states=(_COMPLETE,)):
        folds_ = t.user_attrs.get("folds", [])
        row: dict[str, Any] = {
            "trial": t.number,
            "mean_fold_sharpe": float(t.value or 0.0),
            "duration_s": t.duration.total_seconds() if t.duration else math.nan,
            "orders": int(sum(f["orders"] for f in folds_)),
            "rejections": int(sum(f["rejections"] for f in folds_)),
            "positive_folds": float(np.mean([f["sharpe"] > 0 for f in folds_])) if folds_ else 0.0,
        }
        row.update({f"p_{k}": v for k, v in t.params.items()})
        row.update({f"fold_{i}": float(f["sharpe"]) for i, f in enumerate(folds_)})
        rows.append(row)
    return pl.DataFrame(rows).sort("trial") if rows else pl.DataFrame()


@dataclass(frozen=True)
class StageResult:
    """One stage's study, its trials as a table, and its out-of-sample report."""

    study: optuna.Study
    trials: pl.DataFrame
    report: StudyReport
    new_trials: int
    wall_seconds: float


def run_screening(cfg: ReportConfig, data: MarketData, paths: ReportPaths) -> StageResult:
    """Run (or resume) the screening search on ``cfg.screening.engine``."""
    s = cfg.screening
    name = f"{cfg.name}-screen"
    study = _study(name, paths.storage, cfg.fingerprint())
    done = len(study.get_trials(deepcopy=False, states=(_COMPLETE,)))
    target = _grid_size(s.space) if s.sampler == "grid" else (s.n_trials or 50)
    todo = max(target - done, 0)
    started = dt.datetime.now(dt.UTC)
    if todo:
        study = run_study(
            strategy_class(cfg),
            walk_forward(cfg, data, s.engine),
            space=s.space,
            sampler=s.sampler,
            n_trials=todo,
            study_name=name,
            storage=paths.storage,
            returns_dir=paths.trials_dir,
            seed=s.seed,
        )
    wall = (dt.datetime.now(dt.UTC) - started).total_seconds()
    return StageResult(study, trials_frame(study), summarize_study(study), todo, wall)


def top_trials(study: optuna.Study, k: int) -> list[optuna.trial.FrozenTrial]:
    """The *k* best completed trials with distinct parameters."""
    trials = sorted(
        study.get_trials(deepcopy=False, states=(_COMPLETE,)),
        key=lambda t: t.value if t.value is not None else -math.inf,
        reverse=True,
    )
    seen: set[str] = set()
    best = []
    for t in trials:
        key = repr(sorted(t.params.items()))
        if key not in seen:
            seen.add(key)
            best.append(t)
        if len(best) == k:
            break
    return best


def run_validation(
    cfg: ReportConfig, data: MarketData, screening: StageResult, paths: ReportPaths
) -> StageResult:
    """Re-run the top screened parameter sets on Nautilus (resuming), deflated for the search."""
    name = f"{cfg.name}-nautilus"
    study = _study(name, paths.storage, cfg.fingerprint())
    for t in top_trials(screening.study, cfg.validation.top_k):
        study.enqueue_trial(t.params, skip_if_exists=True)
    waiting = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,)))
    started = dt.datetime.now(dt.UTC)
    if waiting:
        study = run_study(
            strategy_class(cfg),
            walk_forward(cfg, data, "nautilus"),
            space=cfg.screening.space,
            sampler="tpe",
            n_trials=waiting,
            study_name=name,
            storage=paths.storage,
            returns_dir=paths.trials_dir,
            seed=cfg.screening.seed,
        )
    wall = (dt.datetime.now(dt.UTC) - started).total_seconds()
    report = summarize_study(study, selection=screening.study)
    return StageResult(study, trials_frame(study), report, waiting, wall)


def engine_comparison(screening: StageResult, validation: StageResult) -> pl.DataFrame:
    """Screened vs Nautilus results for every validated parameter set.

    Columns: ``trial`` (screening trial number), parameters, the two engines'
    mean fold Sharpe and runtime, their difference and the speed-up.
    """
    if validation.trials.is_empty() or screening.trials.is_empty():
        return pl.DataFrame()
    params = [c for c in validation.trials.columns if c.startswith("p_")]
    screen = screening.trials.select(
        "trial",
        *params,
        pl.col("mean_fold_sharpe").alias("screen_sharpe"),
        pl.col("duration_s").alias("screen_s"),
    )
    naut = validation.trials.select(
        *params,
        pl.col("mean_fold_sharpe").alias("nautilus_sharpe"),
        pl.col("duration_s").alias("nautilus_s"),
        pl.col("rejections").alias("nautilus_rejections"),
    )
    joined = screen.join(naut, on=params, how="inner").unique(params, keep="first")
    return joined.with_columns(
        (pl.col("nautilus_sharpe") - pl.col("screen_sharpe")).alias("sharpe_diff"),
        (pl.col("nautilus_s") / pl.col("screen_s")).alias("speedup"),
    ).sort("nautilus_sharpe", descending=True)


def fold_comparison(screening: StageResult, validation: StageResult) -> pl.DataFrame:
    """Per-fold Sharpe on both engines for the validated parameter sets (long format)."""
    params = [c for c in validation.trials.columns if c.startswith("p_")]
    folds_ = [c for c in validation.trials.columns if c.startswith("fold_")]
    if not params or not folds_:
        return pl.DataFrame()
    a = screening.trials.select(params + folds_).unique(params, keep="first")
    b = validation.trials.select(params + folds_)
    long_a = a.unpivot(folds_, index=params, variable_name="fold", value_name="screen")
    long_b = b.unpivot(folds_, index=params, variable_name="fold", value_name="nautilus")
    return long_a.join(long_b, on=[*params, "fold"], how="inner")


# ── Full backtest ─────────────────────────────────────────────────────────────


def full_backtest(cfg: ReportConfig, data: MarketData, params: dict[str, Any]) -> BacktestRun:
    """Nautilus backtest of *params* from the first post-warm-up bar to ``end``.

    The parameters were chosen on the folds this period contains, so this run
    is *in-sample* for them: it shows how Nautilus trades the strategy (fills,
    positions, statistics), not an unbiased performance estimate — the
    validated walk-forward report is that.
    """
    e = cfg.evaluation
    closes = calendar(cfg, data)
    start = closes[min(e.min_train, closes.len() - 1)]
    return run_backtest(
        strategy_class(cfg)(**params),
        data,
        start=start,
        end=cfg.data.end,
        initial_capital=e.initial_capital,
        costs=costs(cfg),
        rebalance_band=e.rebalance_band,
        leverage=e.leverage,
    )


# ── Attribution ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Attribution:
    """Where a strategy's (gross) returns come from, on a positions × returns panel.

    Rebuilt from the same targets the engines trade (``compute_targets``), held
    between a symbol's bars, weighted ``1/N`` and earning the next bar's return.
    Costs are excluded, so the total is close to a zero-cost engine run.

    Attributes:
        components: ``component, ann_return, ann_vol, sharpe``. *total* splits
            two ways. **long leg + short leg**. **market timing + selection**,
            where timing is the average position across listed names times
            the equal-weight universe return, and selection is the rest (picking
            which names to hold). *beta-hedged* is total minus beta × benchmark.
        exposure: Mean net and gross exposure (fractions of equity), beta and
            correlation to the benchmark.
        signal: Does the target's sign predict the next month? Pooled hit rate
            against the base rate of up-months, hit rates of longs and shorts,
            and the mean cross-sectional rank IC with its t-statistic
            (non-overlapping 21-bar horizons).
        contributors: Per-symbol total contribution (bp of equity), sorted.
        series: ``ts`` and the daily component returns, for charts.
    """

    components: pl.DataFrame
    exposure: dict[str, float]
    signal: dict[str, float]
    contributors: pl.DataFrame
    series: pl.DataFrame


def _sharpe(x: np.ndarray) -> float:
    if x.size < 2 or x.std(ddof=1) == 0.0:  # noqa: PLR2004
        return 0.0
    return float(x.mean() / x.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _ffill(panel: np.ndarray, alive: np.ndarray) -> np.ndarray:
    out = panel.copy()
    for i in range(1, out.shape[0]):
        gap = np.isnan(out[i]) & alive[i]
        out[i, gap] = out[i - 1, gap]
    return out


def _rate(hits: np.ndarray) -> float:
    return float(hits.mean()) if hits.size else math.nan


def _signal_quality(sign: np.ndarray, nxt: np.ndarray, ic: np.ndarray) -> dict[str, float]:
    t_stat = math.nan
    if ic.size > 2 and ic.std() > 0:  # noqa: PLR2004
        t_stat = float(ic.mean() / ic.std(ddof=1) * math.sqrt(ic.size))
    return {
        "hit rate": _rate(np.sign(nxt) == sign),
        "base rate of up months": _rate(nxt > 0),
        "hit rate of longs": _rate(nxt[sign > 0] > 0),
        "hit rate of shorts": _rate(nxt[sign < 0] < 0),
        "share of positions long": _rate(sign > 0),
        "rank IC (mean)": float(ic.mean()) if ic.size else math.nan,
        "rank IC t-stat": t_stat,
    }


def attribution(  # noqa: PLR0915
    cfg: ReportConfig,
    data: MarketData,
    params: dict[str, Any],
    *,
    start: dt.datetime | None = None,
    benchmark: pl.DataFrame | None = None,
    horizon: int = 21,
) -> Attribution:
    """Decompose *params*' gross returns from *start* to the end (see :class:`Attribution`).

    Args:
        cfg: Report config (strategy).
        data: Universe.
        params: Strategy parameters.
        start: First bar close of the window (default: the first fold's test start).
        benchmark: Benchmark bars (``ts_init``, ``close``) for beta; ``None`` uses
            the equal-weight universe.
        horizon: Bars ahead for the signal hit rate and rank IC.
    """
    from snippy_scales.execution import compute_targets

    symbols = list(data.frames)
    calendar_ = data.closes(union=True)
    ts = calendar_.dt.epoch("ns").to_numpy()
    t_n, n = ts.size, len(symbols)
    close = np.full((t_n, n), np.nan)
    target = np.full((t_n, n), np.nan)
    alive = np.zeros((t_n, n), bool)
    targets = compute_targets(strategy_class(cfg)(**params), data.frames)
    for j, s in enumerate(symbols):
        f = data.frames[s]
        idx = np.searchsorted(ts, f["ts_init"].dt.epoch("ns").to_numpy())
        close[idx, j] = f["close"].to_numpy()
        alive[idx[0] : idx[-1] + 1, j] = True
        keys = np.fromiter(targets[s].keys(), np.int64, len(targets[s]))
        vals = np.fromiter(targets[s].values(), np.float64, len(targets[s]))
        target[np.searchsorted(ts, keys), j] = vals
    close = _ffill(close, alive)
    target = np.nan_to_num(_ffill(target, alive))
    ret = np.zeros((t_n, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = close[1:] / close[:-1] - 1.0
    ret[~np.isfinite(ret) | ~alive] = 0.0
    n_alive = alive.sum(1)
    ew = np.where(n_alive > 0, ret.sum(1) / np.maximum(n_alive, 1), 0.0)
    if benchmark is not None:
        b = np.full(t_n, np.nan)
        bts = benchmark["ts_init"].dt.epoch("ns").to_numpy()
        inside = (bts >= ts[0]) & (bts <= ts[-1])
        b[np.searchsorted(ts, bts[inside])] = benchmark["close"].to_numpy()[inside]
        b = pl.Series(b).fill_nan(None).forward_fill().to_numpy()
        bench = np.zeros(t_n)
        with np.errstate(invalid="ignore", divide="ignore"):
            bench[1:] = b[1:] / b[:-1] - 1.0
        bench[~np.isfinite(bench)] = 0.0
    else:
        bench = ew

    pos = target[:-1] / n  # set at close t, earns bar t+1
    r = ret[1:]
    live = alive[:-1]
    total = (pos * r).sum(1)
    long_ = (np.clip(pos, 0.0, None) * r).sum(1)
    short = (np.clip(pos, None, 0.0) * r).sum(1)
    mean_pos = (pos * live).sum(1) / np.maximum(live.sum(1), 1)
    timing = mean_pos * (r * live).sum(1)
    selection = total - timing
    first = start if start is not None else folds(cfg, data)[0].test_start
    lo_ns = cat.to_ns(first)
    lo = int(np.searchsorted(ts[1:], lo_ns if lo_ns is not None else ts[1]))
    w = slice(lo, t_n - 1)
    x = bench[1:][w]
    y = total[w]
    beta = float(np.cov(y, x)[0, 1] / x.var(ddof=1)) if x.var() > 0 else 0.0
    hedged = y - beta * x
    parts = {
        "total": y,
        "long leg": long_[w],
        "short leg": short[w],
        "market timing": timing[w],
        "selection": selection[w],
        "beta-hedged": hedged,
    }
    components = pl.DataFrame(
        [
            {
                "component": k,
                "ann_return": float(v.mean() * TRADING_DAYS_PER_YEAR),
                "ann_vol": float(v.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)),
                "sharpe": _sharpe(v),
            }
            for k, v in parts.items()
        ]
    )
    net, gross = pos.sum(1)[w], np.abs(pos).sum(1)[w]
    exposure = {
        "mean net exposure": float(net.mean()),
        "mean gross exposure": float(gross.mean()),
        "beta to benchmark": beta,
        "correlation to benchmark": float(np.corrcoef(y, x)[0, 1]) if y.std() > 0 else 0.0,
    }

    # Signal quality on non-overlapping horizons.
    with np.errstate(invalid="ignore", divide="ignore"):
        logp = np.log(close)
    fwd = np.full((t_n, n), np.nan)
    fwd[:-horizon] = logp[horizon:] - logp[:-horizon]
    rows = np.arange(lo + 1, t_n - horizon, horizon)
    sign, nxt = np.sign(target[rows]), fwd[rows]
    ok = (sign != 0) & np.isfinite(nxt) & alive[rows]
    ics = []
    for i in rows:
        m = alive[i] & np.isfinite(fwd[i]) & (target[i] != 0)
        if m.sum() > 10:  # noqa: PLR2004
            a = pl.Series(target[i, m]).rank().to_numpy()
            b_ = pl.Series(fwd[i, m]).rank().to_numpy()
            if a.std() > 0 and b_.std() > 0:
                ics.append(float(np.corrcoef(a, b_)[0, 1]))
    signal = _signal_quality(sign[ok], nxt[ok], np.array(ics))
    contrib = (pos * r)[w].sum(0) * 1e4
    contributors = pl.DataFrame({"symbol": symbols, "contribution_bp": contrib}).sort(
        "contribution_bp"
    )
    series = pl.DataFrame(
        {"ts": calendar_[1:][w], **{k: v for k, v in parts.items()}, "benchmark": x}
    )
    return Attribution(components, exposure, signal, contributors, series)


# ── Summaries ─────────────────────────────────────────────────────────────────


def annualised_sharpe(returns: pl.DataFrame) -> float:
    """Sharpe of daily ``return`` values, annualised by :data:`TRADING_DAYS_PER_YEAR`."""
    values = returns["return"].to_numpy()
    if values.size < 2 or values.std(ddof=1) == 0.0:  # noqa: PLR2004
        return 0.0
    return float(values.mean() / values.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def drawdown(returns: pl.DataFrame) -> pl.DataFrame:
    """Compounded equity and drawdown of daily returns (``ts``, ``equity``, ``drawdown``)."""
    equity = (1.0 + returns["return"]).cum_prod()
    return returns.select("ts").with_columns(
        equity.alias("equity"), (equity / equity.cum_max() - 1.0).alias("drawdown")
    )


def calendar_year_returns(returns: pl.DataFrame) -> pl.DataFrame:
    """Compounded return per calendar year (``year``, ``return``)."""
    return (
        returns.group_by(pl.col("ts").dt.year().alias("year"))
        .agg(((1.0 + pl.col("return")).product() - 1.0).alias("return"))
        .sort("year")
    )


def within(returns: pl.DataFrame, reference: pl.DataFrame) -> pl.DataFrame:
    """*returns* restricted to the days of *reference*."""
    return returns.join(reference.select("ts"), on="ts", how="semi")


def checks(
    validation: StageResult,
    comparison: pl.DataFrame,
    benchmark: pl.DataFrame | None,
    full_run: BacktestRun | None = None,
) -> pl.DataFrame:
    """Pass/fail checks on the validated result, for the conclusions section.

    Orders Nautilus denied or rejected (validation folds and *full_run*) mean it
    did not trade as the strategy intended, so its numbers are suspect.
    """
    r = validation.report
    oos_years = r.returns.height / 365.25 if r.returns.height else 0.0
    rows: list[dict[str, Any]] = [
        {
            "check": "Deflated Sharpe ratio ≥ 0.95 (deflated for every screened trial)",
            "value": r.dsr,
            "passed": r.dsr >= 0.95,  # noqa: PLR2004
        },
        {
            "check": "Probability of backtest overfitting < 0.5",
            "value": math.nan if r.pbo is None else r.pbo,
            "passed": r.pbo is not None and r.pbo < 0.5,  # noqa: PLR2004
        },
        {
            "check": "At least 60% of folds have a positive Sharpe",
            "value": float(np.mean([s > 0 for s in r.fold_sharpes])) if r.fold_sharpes else 0.0,
            "passed": bool(r.fold_sharpes) and np.mean([s > 0 for s in r.fold_sharpes]) >= 0.6,  # noqa: PLR2004
        },
        {
            "check": "Minimum track record length ≤ out-of-sample years",
            "value": r.min_track_record_years,
            "passed": r.min_track_record_years <= oos_years,
        },
    ]
    if benchmark is not None and r.returns.height:
        bench = annualised_sharpe(within(benchmark, r.returns))
        rows.append(
            {
                "check": f"OOS Sharpe beats the benchmark's ({bench:.2f}) over the same days",
                "value": r.sharpe,
                "passed": r.sharpe > bench,
            }
        )
    if not comparison.is_empty():
        worst = float(np.abs(comparison["sharpe_diff"].to_numpy()).max())
        rows.append(
            {
                "check": "Screening engine within 0.05 Sharpe of Nautilus (validated trials)",
                "value": worst,
                "passed": worst <= 0.05,  # noqa: PLR2004
            }
        )
        rejections = int(comparison["nautilus_rejections"].sum())
        if full_run is not None:
            rejections += len(full_run.rejections)
        rows.append(
            {
                "check": "No orders denied or rejected by Nautilus",
                "value": float(rejections),
                "passed": rejections == 0,
            }
        )
    return pl.DataFrame(
        [{**r, "value": float(r["value"]), "passed": bool(r["passed"])} for r in rows],
        schema={"check": pl.String, "value": pl.Float64, "passed": pl.Boolean},
    )


def rejection_summary(run: BacktestRun) -> pl.DataFrame:
    """Denied/rejected orders grouped by reason (instrument IDs folded away), most first."""
    if not run.rejections:
        return pl.DataFrame(schema={"reason": pl.String, "orders": pl.UInt32})
    reasons = pl.Series("reason", list(run.rejections)).str.replace_all(
        r"\b[A-Z0-9.]+\.[A-Z]+\b", "<instrument>"
    )
    return reasons.to_frame().group_by("reason").len("orders").sort("orders", descending=True)


def environment() -> dict[str, str]:
    """Git revision, interpreter and key package versions, for reproducibility."""

    def git(*args: str) -> str:
        try:
            out = subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return out.stdout.strip()

    env = {
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git commit": git("rev-parse", "--short", "HEAD"),
        "git branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted changes": "yes" if git("status", "--porcelain") else "no",
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
    }
    for package in ("nautilus_trader", "optuna", "numba", "polars", "marimo"):
        try:
            env[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            env[package] = "not installed"
    return env


def commands(cfg: ReportConfig, paths: ReportPaths) -> str:
    """Shell commands that reproduce or extend this report."""
    return "\n".join(
        [
            "# Rebuild this report (resumes its studies; re-exports report.html)",
            f"algo report build {cfg.name}",
            "",
            "# Open it interactively",
            f"algo report edit {cfg.name}",
            "",
            "# Start a variant: new asset class, period or search space",
            f"algo report new {cfg.name}-v2 --from {paths.config}",
            f"$EDITOR reports/{cfg.name}-v2/config.yaml",
            f"algo report build {cfg.name}-v2",
            "",
            "# Inspect the studies with Optuna's dashboard",
            f"optuna-dashboard {paths.storage}",
        ]
    )
