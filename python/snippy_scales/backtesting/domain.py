"""Domain types for the backtesting layer.

Defines the canonical result types and the ``TRADING_DAYS_PER_YEAR`` constant
used throughout the backtesting infrastructure.  These types are engine-agnostic:
they carry no references to raptorbt and can be used against any execution backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR: int = 252
"""Conventional trading-day count used for annualising volatility and ratios."""

# ── Trade ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Trade:
    """Engine-agnostic representation of a completed round-trip trade.

    Attributes:
        symbol: Instrument identifier.
        entry_time: Bar timestamp at position open (nanoseconds since epoch).
        exit_time: Bar timestamp at position close (nanoseconds since epoch).
        direction: ``1`` for long, ``-1`` for short.
        entry_price: Fill price on entry.
        exit_price: Fill price on exit.
        pnl: Realised profit / loss in cash units.
        return_pct: Percentage return for this trade.
    """

    symbol: str
    entry_time: int
    exit_time: int
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float

    @classmethod
    def from_raptorbt(cls, t: Any, symbol: str) -> Trade:
        """Convert a ``raptorbt.PyTrade`` object to a domain :class:`Trade`.

        Uses :func:`getattr` with safe defaults so that minor raptorbt API
        changes do not cause hard failures.

        Args:
            t: A ``raptorbt.PyTrade`` instance.
            symbol: Instrument identifier to attach to the result.

        Returns:
            A typed, engine-agnostic :class:`Trade`.
        """
        return cls(
            symbol=symbol,
            entry_time=int(getattr(t, "entry_time", 0)),
            exit_time=int(getattr(t, "exit_time", 0)),
            direction=int(getattr(t, "direction", 1)),
            entry_price=float(getattr(t, "entry_price", 0.0)),
            exit_price=float(getattr(t, "exit_price", 0.0)),
            pnl=float(getattr(t, "pnl", 0.0)),
            return_pct=float(getattr(t, "return_pct", 0.0)),
        )


# ── BacktestMetrics ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BacktestMetrics:
    """Key performance metrics from a backtest run.

    A frozen, typed subset of ``raptorbt.PyBacktestMetrics`` for ergonomic
    access and easy serialisation.

    Attributes:
        total_return_pct: Total strategy return as a percentage.
        sharpe_ratio: Annualised Sharpe ratio (risk-free = 0).
        sortino_ratio: Annualised Sortino ratio.
        calmar_ratio: Return / max-drawdown ratio.
        omega_ratio: Probability-weighted ratio of gains to losses.
        max_drawdown_pct: Maximum peak-to-trough drawdown as a percentage.
        max_drawdown_duration: Longest drawdown period in bars.
        total_trades: Total number of trades executed.
        total_closed_trades: Number of positions that have been closed.
        total_open_trades: Number of currently active positions.
        winning_trades: Number of profitable transactions.
        losing_trades: Number of unprofitable transactions.
        win_rate_pct: Percentage of trades that were profitable.
        profit_factor: Gross profit / gross loss.
        expectancy: Average profit per trade in cash units.
        sqn: System Quality Number measuring trade consistency.
        avg_trade_return_pct: Average return across all trades.
        avg_win_pct: Mean return of winning trades.
        avg_loss_pct: Mean return of losing trades.
        best_trade_pct: Maximum single-trade return achieved.
        worst_trade_pct: Minimum single-trade return achieved.
        payoff_ratio: Average winning return relative to average losing return.
        recovery_factor: Net profit divided by maximum drawdown.
        avg_holding_period: Average trade duration in bars.
        avg_winning_duration: Mean duration of profitable trades in bars.
        avg_losing_duration: Mean duration of losing trades in bars.
        max_consecutive_wins: Longest winning streak length.
        max_consecutive_losses: Longest losing streak length.
        start_value: Initial portfolio capital.
        end_value: Final portfolio value after all trades.
        total_fees_paid: Cumulative transaction costs incurred.
        open_trade_pnl: Unrealised profit/loss from active positions.
        exposure_pct: Percentage of time with active market positions.
    """

    # Core performance
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    # Drawdown
    max_drawdown_pct: float
    max_drawdown_duration: int
    # Trade counts
    total_trades: int
    total_closed_trades: int
    total_open_trades: int
    winning_trades: int
    losing_trades: int
    # Trade performance
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    sqn: float
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    payoff_ratio: float
    recovery_factor: float
    # Duration
    avg_holding_period: float
    avg_winning_duration: float
    avg_losing_duration: float
    # Streaks
    max_consecutive_wins: int
    max_consecutive_losses: int
    # Portfolio
    start_value: float
    end_value: float
    total_fees_paid: float
    open_trade_pnl: float
    exposure_pct: float

    @classmethod
    def from_raptorbt(cls, m: Any) -> BacktestMetrics:
        """Construct from a ``raptorbt.PyBacktestMetrics`` object.

        Uses :func:`getattr` with safe defaults so that minor raptorbt API
        changes do not cause hard failures.

        Args:
            m: raptorbt metrics object with the expected attribute names.

        Returns:
            Typed :class:`BacktestMetrics` instance.
        """
        return cls(
            # Core performance
            total_return_pct=float(getattr(m, "total_return_pct", 0.0)),
            sharpe_ratio=float(getattr(m, "sharpe_ratio", 0.0)),
            sortino_ratio=float(getattr(m, "sortino_ratio", 0.0)),
            calmar_ratio=float(getattr(m, "calmar_ratio", 0.0)),
            omega_ratio=float(getattr(m, "omega_ratio", 0.0)),
            # Drawdown
            max_drawdown_pct=float(getattr(m, "max_drawdown_pct", 0.0)),
            max_drawdown_duration=int(getattr(m, "max_drawdown_duration", 0)),
            # Trade counts
            total_trades=int(getattr(m, "total_trades", 0)),
            total_closed_trades=int(getattr(m, "total_closed_trades", 0)),
            total_open_trades=int(getattr(m, "total_open_trades", 0)),
            winning_trades=int(getattr(m, "winning_trades", 0)),
            losing_trades=int(getattr(m, "losing_trades", 0)),
            # Trade performance
            win_rate_pct=float(getattr(m, "win_rate_pct", 0.0)),
            profit_factor=float(getattr(m, "profit_factor", 0.0)),
            expectancy=float(getattr(m, "expectancy", 0.0)),
            sqn=float(getattr(m, "sqn", 0.0)),
            avg_trade_return_pct=float(getattr(m, "avg_trade_return_pct", 0.0)),
            avg_win_pct=float(getattr(m, "avg_win_pct", 0.0)),
            avg_loss_pct=float(getattr(m, "avg_loss_pct", 0.0)),
            best_trade_pct=float(getattr(m, "best_trade_pct", 0.0)),
            worst_trade_pct=float(getattr(m, "worst_trade_pct", 0.0)),
            payoff_ratio=float(getattr(m, "payoff_ratio", 0.0)),
            recovery_factor=float(getattr(m, "recovery_factor", 0.0)),
            # Duration
            avg_holding_period=float(getattr(m, "avg_holding_period", 0.0)),
            avg_winning_duration=float(getattr(m, "avg_winning_duration", 0.0)),
            avg_losing_duration=float(getattr(m, "avg_losing_duration", 0.0)),
            # Streaks
            max_consecutive_wins=int(getattr(m, "max_consecutive_wins", 0)),
            max_consecutive_losses=int(getattr(m, "max_consecutive_losses", 0)),
            # Portfolio
            start_value=float(getattr(m, "start_value", 0.0)),
            end_value=float(getattr(m, "end_value", 0.0)),
            total_fees_paid=float(getattr(m, "total_fees_paid", 0.0)),
            open_trade_pnl=float(getattr(m, "open_trade_pnl", 0.0)),
            exposure_pct=float(getattr(m, "exposure_pct", 0.0)),
        )


# ── BacktestResult ────────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Full result from a backtest run.

    Attributes:
        symbol: Instrument symbol or list of symbols for basket runs.
        metrics: Aggregated performance metrics.
        equity_curve: Portfolio equity over time as a float64 NumPy array.
        drawdown_curve: Drawdown time series as a float64 NumPy array.
        returns: Period returns as a float64 NumPy array.
        trades: Engine-agnostic list of completed :class:`Trade` objects.
    """

    symbol: str | list[str]
    metrics: BacktestMetrics
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    returns: np.ndarray
    trades: list[Trade] = field(default_factory=list)

    @classmethod
    def from_raptorbt(cls, result: Any, symbol: str | list[str]) -> BacktestResult:
        """Construct from a ``raptorbt.PyBacktestResult`` object.

        Args:
            result: raptorbt result object.
            symbol: Symbol or list of symbols associated with this result.

        Returns:
            Typed :class:`BacktestResult` instance with domain :class:`Trade` objects.
        """
        sym_str = symbol if isinstance(symbol, str) else (symbol[0] if symbol else "BASKET")
        raw_trades = list(result.trades())
        trades = [Trade.from_raptorbt(t, sym_str) for t in raw_trades]
        return cls(
            symbol=symbol,
            metrics=BacktestMetrics.from_raptorbt(result.metrics),
            equity_curve=np.asarray(result.equity_curve(), dtype=np.float64),
            drawdown_curve=np.asarray(result.drawdown_curve(), dtype=np.float64),
            returns=np.asarray(result.returns(), dtype=np.float64),
            trades=trades,
        )


# ── Walk-forward / fold types ─────────────────────────────────────────────────

#: Ordered list of the key float metrics aggregated in :class:`WalkForwardSummary`.
_SUMMARY_METRICS: tuple[str, ...] = (
    "total_return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "max_drawdown_pct",
    "win_rate_pct",
    "profit_factor",
    "expectancy",
    "sqn",
    "recovery_factor",
    "payoff_ratio",
    "avg_trade_return_pct",
    "exposure_pct",
)


@dataclass(frozen=True)
class FoldResult:
    """Out-of-sample backtest result for a single walk-forward fold.

    Attributes:
        fold_index: Zero-based fold number within the walk-forward analysis.
        oos_start: Out-of-sample window start (nanoseconds since epoch).
        oos_end: Out-of-sample window end (nanoseconds since epoch).
        result: Full :class:`BacktestResult` for this fold's OOS period.
    """

    fold_index: int
    oos_start: int
    oos_end: int
    result: BacktestResult


@dataclass(frozen=True)
class WalkForwardSummary:
    """Mean and standard deviation of key metrics across walk-forward folds.

    Each of the 14 key metrics has a ``_mean`` and ``_std`` pair.  The std is
    the **population** standard deviation (``ddof=0``) so it is always finite
    even for a single fold.

    Attributes:
        n_folds: Number of folds that were aggregated.
        total_return_pct_mean: Mean total return across folds.
        total_return_pct_std: Std-dev of total return across folds.
        sharpe_ratio_mean: Mean annualised Sharpe ratio.
        sharpe_ratio_std: Std-dev of Sharpe ratio.
        sortino_ratio_mean: Mean annualised Sortino ratio.
        sortino_ratio_std: Std-dev of Sortino ratio.
        calmar_ratio_mean: Mean Calmar ratio.
        calmar_ratio_std: Std-dev of Calmar ratio.
        omega_ratio_mean: Mean Omega ratio.
        omega_ratio_std: Std-dev of Omega ratio.
        max_drawdown_pct_mean: Mean maximum drawdown.
        max_drawdown_pct_std: Std-dev of maximum drawdown.
        win_rate_pct_mean: Mean win rate.
        win_rate_pct_std: Std-dev of win rate.
        profit_factor_mean: Mean profit factor.
        profit_factor_std: Std-dev of profit factor.
        expectancy_mean: Mean expectancy.
        expectancy_std: Std-dev of expectancy.
        sqn_mean: Mean System Quality Number.
        sqn_std: Std-dev of SQN.
        recovery_factor_mean: Mean recovery factor.
        recovery_factor_std: Std-dev of recovery factor.
        payoff_ratio_mean: Mean payoff ratio.
        payoff_ratio_std: Std-dev of payoff ratio.
        avg_trade_return_pct_mean: Mean average trade return.
        avg_trade_return_pct_std: Std-dev of average trade return.
        exposure_pct_mean: Mean market exposure.
        exposure_pct_std: Std-dev of market exposure.
    """

    n_folds: int
    total_return_pct_mean: float
    total_return_pct_std: float
    sharpe_ratio_mean: float
    sharpe_ratio_std: float
    sortino_ratio_mean: float
    sortino_ratio_std: float
    calmar_ratio_mean: float
    calmar_ratio_std: float
    omega_ratio_mean: float
    omega_ratio_std: float
    max_drawdown_pct_mean: float
    max_drawdown_pct_std: float
    win_rate_pct_mean: float
    win_rate_pct_std: float
    profit_factor_mean: float
    profit_factor_std: float
    expectancy_mean: float
    expectancy_std: float
    sqn_mean: float
    sqn_std: float
    recovery_factor_mean: float
    recovery_factor_std: float
    payoff_ratio_mean: float
    payoff_ratio_std: float
    avg_trade_return_pct_mean: float
    avg_trade_return_pct_std: float
    exposure_pct_mean: float
    exposure_pct_std: float

    @classmethod
    def from_folds(cls, folds: list[FoldResult]) -> WalkForwardSummary:
        """Compute mean and population std for key metrics across *folds*.

        Args:
            folds: Non-empty list of :class:`FoldResult` objects.

        Returns:
            :class:`WalkForwardSummary` with mean and std for all key metrics.

        Raises:
            ValueError: If *folds* is empty.
        """
        if not folds:
            raise ValueError("Cannot compute walk-forward summary from an empty fold list.")

        kwargs: dict[str, Any] = {"n_folds": len(folds)}
        for metric in _SUMMARY_METRICS:
            values = [getattr(f.result.metrics, metric) for f in folds]
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            kwargs[f"{metric}_mean"] = mean
            kwargs[f"{metric}_std"] = math.sqrt(variance)
        return cls(**kwargs)


@dataclass
class WalkForwardResult:
    """Aggregated result of a walk-forward (out-of-sample) analysis.

    Attributes:
        symbol: Instrument symbol or list of symbols.
        folds: Ordered list of per-fold :class:`FoldResult` objects.
        summary: Cross-fold mean and std-dev for the 14 key metrics.
    """

    symbol: str | list[str]
    folds: list[FoldResult]
    summary: WalkForwardSummary

    @classmethod
    def from_fold_results(
        cls, symbol: str | list[str], folds: list[FoldResult]
    ) -> WalkForwardResult:
        """Construct from a list of fold results, computing the summary automatically.

        Args:
            symbol: Instrument symbol(s).
            folds: Non-empty list of :class:`FoldResult` objects.

        Returns:
            :class:`WalkForwardResult` with populated summary.
        """
        return cls(
            symbol=symbol,
            folds=folds,
            summary=WalkForwardSummary.from_folds(folds),
        )
