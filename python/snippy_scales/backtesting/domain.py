"""Domain types for the backtesting layer.

Defines the canonical result types and the ``TRADING_DAYS_PER_YEAR`` constant
used throughout the backtesting infrastructure.  These types are engine-agnostic:
they carry no references to raptorbt and can be used against any execution backend.
"""

from __future__ import annotations

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
        max_drawdown_pct: Maximum peak-to-trough drawdown as a percentage.
        win_rate_pct: Percentage of trades that were profitable.
        profit_factor: Gross profit / gross loss.
        total_trades: Number of completed round-trip trades.
        expectancy: Average profit per trade in cash units.
    """

    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    expectancy: float

    @classmethod
    def from_raptorbt(cls, m: Any) -> BacktestMetrics:
        """Construct from a ``raptorbt.PyBacktestMetrics`` object.

        Args:
            m: raptorbt metrics object with the expected attribute names.

        Returns:
            Typed :class:`BacktestMetrics` instance.
        """
        return cls(
            total_return_pct=float(m.total_return_pct),
            sharpe_ratio=float(m.sharpe_ratio),
            sortino_ratio=float(m.sortino_ratio),
            calmar_ratio=float(m.calmar_ratio),
            max_drawdown_pct=float(m.max_drawdown_pct),
            win_rate_pct=float(m.win_rate_pct),
            profit_factor=float(m.profit_factor),
            total_trades=int(m.total_trades),
            expectancy=float(m.expectancy),
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
