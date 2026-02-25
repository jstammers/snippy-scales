"""Tests for walk-forward domain types: FoldResult, WalkForwardSummary,
WalkForwardResult.

Covers:
* FoldResult basic construction
* WalkForwardSummary.from_folds — correct mean and population std
* WalkForwardSummary.from_folds — std is 0.0 for a single fold
* WalkForwardSummary.from_folds — raises ValueError for empty list
* WalkForwardResult.from_fold_results — summary is auto-computed
* All 14 key metrics are represented in the summary
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from snippy_scales.backtesting.domain import (
    _SUMMARY_METRICS,
    BacktestMetrics,
    BacktestResult,
    FoldResult,
    WalkForwardResult,
    WalkForwardSummary,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_BASE_NS = 1_577_836_800_000_000_000  # 2020-01-01 UTC in nanoseconds
_DAY_NS = 86_400_000_000_000


def _make_metrics(**overrides: float | int) -> BacktestMetrics:
    """Construct a BacktestMetrics with sensible defaults, accepting overrides."""
    defaults: dict[str, float | int] = {
        "total_return_pct": 10.0,
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.5,
        "calmar_ratio": 0.8,
        "omega_ratio": 1.4,
        "max_drawdown_pct": -12.0,
        "max_drawdown_duration": 30,
        "total_trades": 50,
        "total_closed_trades": 48,
        "total_open_trades": 2,
        "winning_trades": 28,
        "losing_trades": 20,
        "win_rate_pct": 56.0,
        "profit_factor": 1.6,
        "expectancy": 250.0,
        "sqn": 2.1,
        "avg_trade_return_pct": 0.5,
        "avg_win_pct": 1.2,
        "avg_loss_pct": -0.8,
        "best_trade_pct": 4.5,
        "worst_trade_pct": -3.0,
        "payoff_ratio": 1.5,
        "recovery_factor": 3.2,
        "avg_holding_period": 5.0,
        "avg_winning_duration": 6.0,
        "avg_losing_duration": 4.0,
        "max_consecutive_wins": 7,
        "max_consecutive_losses": 5,
        "start_value": 100_000.0,
        "end_value": 110_000.0,
        "total_fees_paid": 500.0,
        "open_trade_pnl": 0.0,
        "exposure_pct": 60.0,
    }
    defaults.update(overrides)
    return BacktestMetrics(**defaults)  # type: ignore[arg-type]


def _make_result(**metric_overrides: float | int) -> BacktestResult:
    """Construct a minimal BacktestResult with a short random equity curve."""
    n = 50
    rng = np.random.default_rng(0)
    equity = 100_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, n))
    return BacktestResult(
        symbol="SIM",
        metrics=_make_metrics(**metric_overrides),
        equity_curve=equity,
        drawdown_curve=np.zeros(n),
        returns=np.diff(equity) / equity[:-1],
    )


def _make_fold(index: int, **metric_overrides: float | int) -> FoldResult:
    oos_start = _BASE_NS + index * 90 * _DAY_NS
    oos_end = oos_start + 90 * _DAY_NS
    return FoldResult(
        fold_index=index,
        oos_start=oos_start,
        oos_end=oos_end,
        result=_make_result(**metric_overrides),
    )


# ── FoldResult ────────────────────────────────────────────────────────────────


def test_fold_result_is_frozen() -> None:
    fold = _make_fold(0)
    with pytest.raises((AttributeError, TypeError)):
        fold.fold_index = 99  # type: ignore[misc]


def test_fold_result_stores_window() -> None:
    fold = _make_fold(2)
    assert fold.fold_index == 2
    assert fold.oos_end > fold.oos_start


def test_fold_result_result_is_backtest_result() -> None:
    fold = _make_fold(0)
    assert isinstance(fold.result, BacktestResult)


# ── WalkForwardSummary ────────────────────────────────────────────────────────


def test_summary_from_empty_folds_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        WalkForwardSummary.from_folds([])


def test_summary_n_folds() -> None:
    folds = [_make_fold(i) for i in range(4)]
    summary = WalkForwardSummary.from_folds(folds)
    assert summary.n_folds == 4


def test_summary_mean_is_correct() -> None:
    """Mean of sharpe_ratio across two folds with known values."""
    folds = [
        _make_fold(0, sharpe_ratio=1.0),
        _make_fold(1, sharpe_ratio=3.0),
    ]
    summary = WalkForwardSummary.from_folds(folds)
    assert summary.sharpe_ratio_mean == pytest.approx(2.0)


def test_summary_std_is_population_std() -> None:
    """Std-dev uses ddof=0 (population) so single-fold std is always 0.0."""
    folds = [
        _make_fold(0, sharpe_ratio=1.0),
        _make_fold(1, sharpe_ratio=3.0),
    ]
    summary = WalkForwardSummary.from_folds(folds)
    # Population std of [1.0, 3.0] = 1.0
    assert summary.sharpe_ratio_std == pytest.approx(1.0)


def test_summary_single_fold_std_is_zero() -> None:
    summary = WalkForwardSummary.from_folds([_make_fold(0)])
    assert summary.sharpe_ratio_std == pytest.approx(0.0)


def test_summary_all_key_metrics_present() -> None:
    """Every metric in _SUMMARY_METRICS has a _mean and _std attribute."""
    summary = WalkForwardSummary.from_folds([_make_fold(0)])
    for metric in _SUMMARY_METRICS:
        assert hasattr(summary, f"{metric}_mean"), f"missing {metric}_mean"
        assert hasattr(summary, f"{metric}_std"), f"missing {metric}_std"


def test_summary_std_finite_for_all_metrics() -> None:
    folds = [_make_fold(i) for i in range(3)]
    summary = WalkForwardSummary.from_folds(folds)
    for metric in _SUMMARY_METRICS:
        std = getattr(summary, f"{metric}_std")
        assert math.isfinite(std), f"{metric}_std is not finite: {std}"


def test_summary_max_drawdown_std_with_varied_folds() -> None:
    folds = [
        _make_fold(0, max_drawdown_pct=-5.0),
        _make_fold(1, max_drawdown_pct=-10.0),
        _make_fold(2, max_drawdown_pct=-15.0),
    ]
    summary = WalkForwardSummary.from_folds(folds)
    assert summary.max_drawdown_pct_mean == pytest.approx(-10.0)
    # Population std of [-5, -10, -15]: mean=-10, deviations=[5,0,-5], variance=50/3
    expected_std = math.sqrt((25 + 0 + 25) / 3)
    assert summary.max_drawdown_pct_std == pytest.approx(expected_std)


# ── WalkForwardResult ─────────────────────────────────────────────────────────


def test_walk_forward_result_from_fold_results() -> None:
    folds = [_make_fold(i) for i in range(3)]
    wf = WalkForwardResult.from_fold_results("SIM", folds)
    assert wf.symbol == "SIM"
    assert len(wf.folds) == 3
    assert isinstance(wf.summary, WalkForwardSummary)
    assert wf.summary.n_folds == 3


def test_walk_forward_result_basket_symbol() -> None:
    folds = [_make_fold(0)]
    wf = WalkForwardResult.from_fold_results(["ES.c.0", "NQ.c.0"], folds)
    assert wf.symbol == ["ES.c.0", "NQ.c.0"]


def test_walk_forward_result_summary_matches_folds() -> None:
    """The auto-computed summary matches manually computing the mean."""
    folds = [
        _make_fold(0, total_return_pct=5.0),
        _make_fold(1, total_return_pct=15.0),
    ]
    wf = WalkForwardResult.from_fold_results("SIM", folds)
    assert wf.summary.total_return_pct_mean == pytest.approx(10.0)
    assert wf.summary.total_return_pct_std == pytest.approx(5.0)
