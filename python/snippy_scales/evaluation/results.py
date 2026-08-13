"""Result types for the evaluation framework.

Three nested types form the result hierarchy:

* :class:`FoldResult` — metrics and equity curves for one (params, fold) pair.
* :class:`SweepResult` — aggregated statistics across folds for a single
  parameter set.
* :class:`EvaluationResult` — top-level container holding all sweep results,
  exposing ``best_params`` and a tabular ``summary_df()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from snippy_scales._constants import TRADING_DAYS_PER_YEAR

if TYPE_CHECKING:
    import polars as pl

    from snippy_scales.backtesting.domain import BacktestMetrics


# ── FoldResult ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FoldResult:
    """Metrics and equity curves for one (parameter set, fold) combination.

    Attributes:
        fold_idx: Zero-based fold index.
        params: Strategy constructor kwargs used in this fold.
        train_metrics: Performance metrics on the training window.
        test_metrics: Out-of-sample performance metrics.
        train_equity_curve: Portfolio equity over the training window.
        test_equity_curve: Portfolio equity over the test window.
        train_start: ISO date string for the training window start.
        train_end: ISO date string for the training window end.
        test_start: ISO date string for the test window start.
        test_end: ISO date string for the test window end.
    """

    fold_idx: int
    params: dict[str, Any]
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    train_equity_curve: np.ndarray
    test_equity_curve: np.ndarray
    train_start: str
    train_end: str
    test_start: str
    test_end: str


# ── SweepResult ───────────────────────────────────────────────────────────────


@dataclass
class SweepResult:
    """Aggregated walk-forward statistics for a single parameter set.

    Attributes:
        params: Strategy constructor kwargs.
        folds: Individual fold results.
    """

    params: dict[str, Any]
    folds: list[FoldResult] = field(default_factory=list)

    # ── Aggregate test metrics ──────────────────────────────────────────

    @property
    def mean_test_sharpe(self) -> float:
        """Mean Sharpe ratio across test folds."""
        vals = [f.test_metrics.sharpe_ratio for f in self.folds]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_test_return(self) -> float:
        """Mean total return (%) across test folds."""
        vals = [f.test_metrics.total_return_pct for f in self.folds]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_test_max_dd(self) -> float:
        """Mean maximum drawdown (%) across test folds."""
        vals = [f.test_metrics.max_drawdown_pct for f in self.folds]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_test_sortino(self) -> float:
        """Mean Sortino ratio across test folds."""
        vals = [f.test_metrics.sortino_ratio for f in self.folds]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def std_test_sharpe(self) -> float:
        """Population std-dev of Sharpe ratio across test folds."""
        vals = [f.test_metrics.sharpe_ratio for f in self.folds]
        return float(np.std(vals)) if vals else float("nan")

    @property
    def std_test_return(self) -> float:
        """Population std-dev of total return (%) across test folds."""
        vals = [f.test_metrics.total_return_pct for f in self.folds]
        return float(np.std(vals)) if vals else float("nan")

    @property
    def std_test_max_dd(self) -> float:
        """Population std-dev of maximum drawdown (%) across test folds."""
        vals = [f.test_metrics.max_drawdown_pct for f in self.folds]
        return float(np.std(vals)) if vals else float("nan")

    @property
    def std_test_sortino(self) -> float:
        """Population std-dev of Sortino ratio across test folds."""
        vals = [f.test_metrics.sortino_ratio for f in self.folds]
        return float(np.std(vals)) if vals else float("nan")

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def n_test_observations(self) -> int:
        """Total out-of-sample bars across all test folds."""
        return int(sum(len(f.test_equity_curve) for f in self.folds))

    def test_returns(self) -> np.ndarray:
        """Return the concatenated out-of-sample bar returns across test folds.

        Each fold's equity curve is differenced independently so that the join
        between folds does not manufacture a spurious return.  Used to estimate
        the higher moments the Deflated Sharpe Ratio needs.

        Returns:
            1-D float64 array of per-bar returns; empty if no folds have data.
        """
        segments: list[np.ndarray] = []
        for fold in sorted(self.folds, key=lambda f: f.fold_idx):
            equity = np.asarray(fold.test_equity_curve, dtype=np.float64)
            if equity.size < 2:
                continue
            prev = equity[:-1]
            segments.append(
                np.where(prev != 0.0, equity[1:] / np.where(prev != 0.0, prev, 1.0) - 1.0, 0.0)
            )

        if not segments:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(segments)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict for serialisation."""
        return {
            "params": self.params,
            "mean_test_sharpe": self.mean_test_sharpe,
            "std_test_sharpe": self.std_test_sharpe,
            "mean_test_return": self.mean_test_return,
            "std_test_return": self.std_test_return,
            "mean_test_max_dd": self.mean_test_max_dd,
            "std_test_max_dd": self.std_test_max_dd,
            "mean_test_sortino": self.mean_test_sortino,
            "std_test_sortino": self.std_test_sortino,
            "n_folds": self.n_folds,
        }


# ── EvaluationResult ──────────────────────────────────────────────────────────


@dataclass
class EvaluationResult:
    """Top-level container for a completed evaluation run.

    Holds all sweep results and exposes helpers to identify the best
    parameter set and produce a summary DataFrame.

    Attributes:
        experiment_name: Human-readable name for this evaluation run.
        strategy_class: Fully-qualified class name of the strategy.
        symbols: List of symbols evaluated.
        sweep_results: One :class:`SweepResult` per parameter set.
    """

    experiment_name: str
    strategy_class: str
    symbols: list[str]
    sweep_results: list[SweepResult] = field(default_factory=list)

    # ── Best parameter selection ────────────────────────────────────────

    @property
    def best_result(self) -> SweepResult:
        """SweepResult with the highest mean out-of-sample Sharpe ratio.

        Raises:
            ValueError: If there are no sweep results.
        """
        if not self.sweep_results:
            raise ValueError("No sweep results available")
        return max(self.sweep_results, key=lambda r: r.mean_test_sharpe)

    @property
    def best_params(self) -> dict[str, Any]:
        """Parameter dict that achieved the highest mean OOS Sharpe ratio."""
        return self.best_result.params

    # ── Selection-bias correction ───────────────────────────────────────

    @property
    def n_trials(self) -> int:
        """Number of parameter sets searched — the sweep's selection breadth."""
        return len(self.sweep_results)

    @property
    def sharpe_dispersion(self) -> float:
        """Std-dev of mean OOS Sharpe across parameter sets.

        This is the scale on which selection noise operates, and the estimator
        the Deflated Sharpe Ratio uses for ``trial_std``.
        """
        vals = [r.mean_test_sharpe for r in self.sweep_results]
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.std(finite)) if len(finite) > 1 else 1.0

    def deflated_sharpe(
        self,
        sweep: SweepResult | None = None,
        *,
        periods_per_year: float = TRADING_DAYS_PER_YEAR,
    ) -> float:
        """Return the Deflated Sharpe Ratio for *sweep* (default: the best).

        Corrects the headline Sharpe for the fact that it was *selected* as the
        maximum of :attr:`n_trials` candidates.  A value above 0.95 is the
        conventional bar for treating a swept result as evidence of skill; a
        value near 0.5 means it is indistinguishable from the luckiest of N
        coin flips.

        Args:
            sweep: Parameter set to evaluate.  Defaults to :attr:`best_result`.
            periods_per_year: Annualisation factor behind the fold Sharpe
                ratios.  Defaults to ``252``; override for intraday bars.

        Returns:
            Probability in ``[0, 1]``; ``0.0`` when there is insufficient data.
        """
        from snippy_scales.evaluation.statistics import deflated_sharpe_ratio

        if not self.sweep_results:
            return 0.0

        target = sweep if sweep is not None else self.best_result
        returns = target.test_returns()
        if returns.size < 2:
            return 0.0

        observed = target.mean_test_sharpe
        if not np.isfinite(observed):
            return 0.0

        return deflated_sharpe_ratio(
            observed,
            n_trials=self.n_trials,
            n_obs=int(returns.size),
            trial_std=self.sharpe_dispersion,
            skew=float(stats.skew(returns)),
            kurtosis=float(stats.kurtosis(returns, fisher=False)),
            periods_per_year=periods_per_year,
        )

    # ── Summary table ───────────────────────────────────────────────────

    def summary_df(self) -> pl.DataFrame:
        """Return a Polars DataFrame with one row per parameter set.

        Columns include all parameter names, aggregate test-window metrics, and
        ``deflated_sharpe`` — the probability each result reflects skill rather
        than the selection bias induced by searching :attr:`n_trials`
        configurations.  Read that column before the Sharpe column.
        """
        import polars as pl

        rows = []
        for sr in self.sweep_results:
            row: dict[str, Any] = dict(sr.params)
            row["mean_test_sharpe"] = sr.mean_test_sharpe
            row["std_test_sharpe"] = sr.std_test_sharpe
            row["mean_test_return_pct"] = sr.mean_test_return
            row["std_test_return_pct"] = sr.std_test_return
            row["mean_test_max_dd_pct"] = sr.mean_test_max_dd
            row["std_test_max_dd_pct"] = sr.std_test_max_dd
            row["mean_test_sortino"] = sr.mean_test_sortino
            row["std_test_sortino"] = sr.std_test_sortino
            row["deflated_sharpe"] = self.deflated_sharpe(sr)
            row["n_folds"] = sr.n_folds
            rows.append(row)

        if not rows:
            return pl.DataFrame()

        return pl.DataFrame(rows).sort("mean_test_sharpe", descending=True)

    def oos_equity_curve(self) -> np.ndarray:
        """Stitch together the test-fold equity curves of the best parameter set.

        Returns the out-of-sample equity curve as a 1-D float64 array,
        normalised so it starts at 1.0.  Each fold's equity curve is
        rescaled to continue from where the previous fold ended.
        """
        folds = sorted(self.best_result.folds, key=lambda f: f.fold_idx)
        if not folds:
            return np.array([1.0])

        segments: list[np.ndarray] = []
        running_end = 1.0

        for fold in folds:
            eq = np.asarray(fold.test_equity_curve, dtype=np.float64)
            if len(eq) == 0:
                continue
            # Normalise this segment so it starts at running_end
            start = eq[0] if eq[0] != 0 else 1.0
            scaled = eq / start * running_end
            segments.append(scaled)
            running_end = float(scaled[-1])

        return np.concatenate(segments) if segments else np.array([1.0])
