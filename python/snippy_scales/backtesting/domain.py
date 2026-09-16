"""Domain types for the backtesting layer.

Defines the canonical result types and the ``TRADING_DAYS_PER_YEAR`` constant
used throughout the backtesting infrastructure.  These types are engine-agnostic:
they carry no references to raptorbt and can be used against any execution backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from snippy_scales._constants import TRADING_DAYS_PER_YEAR

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "Trade",
    "BacktestMetrics",
    "BacktestResult",
    "drawdown_curve",
]


# ── Numeric helpers ───────────────────────────────────────────────────────────


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning ``0.0`` when the denominator is zero or non-finite.

    Financial metrics are riddled with degenerate cases — no losing trades, a
    flat equity curve, an empty fold.  Propagating ``inf``/``nan`` from those
    into a metrics table poisons downstream aggregation (means, database
    columns), so they collapse to zero here.

    Args:
        numerator: Dividend.
        denominator: Divisor.

    Returns:
        ``numerator / denominator``, or ``0.0`` if that is not finite.
    """
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    result = numerator / denominator
    return float(result) if np.isfinite(result) else 0.0


def drawdown_curve(equity: np.ndarray) -> np.ndarray:
    """Return the fractional drawdown series for an equity curve.

    Args:
        equity: Portfolio value per bar.

    Returns:
        Array of the same length where each element is
        ``equity / running_peak - 1`` (so values are ``<= 0``).
    """
    equity_arr = np.asarray(equity, dtype=np.float64)
    if equity_arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    peak = np.maximum.accumulate(equity_arr)
    safe_peak = np.where(peak == 0.0, 1.0, peak)
    return equity_arr / safe_peak - 1.0


def _drawdown_stats(equity: np.ndarray) -> tuple[float, int]:
    """Return ``(max_drawdown_pct, max_drawdown_duration)`` for an equity curve.

    The drawdown percentage is reported as a **positive** number in percentage
    points, matching raptorbt.  Duration is the longest run of consecutive bars
    spent below a previous peak.

    Args:
        equity: Portfolio value per bar.

    Returns:
        Tuple of maximum drawdown (percentage points) and its duration in bars.
    """
    drawdown = drawdown_curve(equity)
    if drawdown.size == 0:
        return 0.0, 0

    max_dd_pct = float(-np.min(drawdown)) * 100.0

    underwater = drawdown < 0.0
    longest = 0
    current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)

    return max_dd_pct, longest


def _max_streak(flags: np.ndarray) -> int:
    """Return the length of the longest run of ``True`` in *flags*.

    Args:
        flags: Boolean array (e.g. per-trade win indicators).

    Returns:
        Longest consecutive run length; ``0`` for an empty array.
    """
    longest = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _num(obj: Any, name: str, default: float = 0.0) -> float:
    """Read a numeric attribute that may be absent *or* ``None``.

    raptorbt returns ``None`` for metrics that are mathematically undefined —
    ``profit_factor`` when there are no losing trades, for example.  A plain
    ``getattr(obj, name, default)`` only covers the *absent* case and passes
    ``None`` straight through to ``float()``, which raises.  This covers both,
    and also collapses non-finite values so they cannot poison downstream
    aggregation.

    Args:
        obj: Source object, typically a raptorbt metrics struct.
        name: Attribute name.
        default: Value to use when the attribute is missing, ``None``, or
            non-finite.

    Returns:
        A finite float.
    """
    value = getattr(obj, name, None)
    if value is None:
        return default
    result = float(value)
    return result if np.isfinite(result) else default


def _int(obj: Any, name: str, default: int = 0) -> int:
    """Read an integer attribute that may be absent or ``None``.

    Args:
        obj: Source object.
        name: Attribute name.
        default: Value to use when the attribute is missing or ``None``.

    Returns:
        An int.
    """
    value = getattr(obj, name, None)
    return default if value is None else int(value)


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
            entry_time=_int(t, "entry_time", 0),
            exit_time=_int(t, "exit_time", 0),
            direction=_int(t, "direction", 1),
            entry_price=_num(t, "entry_price", 0.0),
            exit_price=_num(t, "exit_price", 0.0),
            pnl=_num(t, "pnl", 0.0),
            return_pct=_num(t, "return_pct", 0.0),
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
    def from_returns(
        cls,
        *,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trades: list[Trade],
        durations: np.ndarray | None = None,
        periods_per_year: float = TRADING_DAYS_PER_YEAR,
        total_fees_paid: float = 0.0,
        exposure_pct: float = 0.0,
        open_trade_pnl: float = 0.0,
    ) -> BacktestMetrics:
        """Compute all 33 metrics from a return stream and a list of trades.

        This is the engine-agnostic counterpart to :meth:`from_raptorbt`, used
        by :class:`~snippy_scales.backtesting.continuous.TargetPositionEngine`.
        Unit conventions match raptorbt exactly: every ``*_pct`` field is in
        **percentage points** (``3.45`` means 3.45%), and ``max_drawdown_pct``
        is reported as a positive number.

        .. warning::
           **Annualisation differs from raptorbt.**  raptorbt annualises
           ``sharpe_ratio``, ``sortino_ratio`` and ``calmar_ratio`` with **365**
           periods per year regardless of bar frequency (verified empirically;
           see ``test_domain_metrics.py::test_raptorbt_annualises_with_365``).
           On daily bars that overstates Sharpe by ``sqrt(365/252) ≈ 1.20`` —
           roughly **20%**.  This method uses *periods_per_year* instead, which
           defaults to the repo-wide ``TRADING_DAYS_PER_YEAR = 252``.  Ratios
           produced here will therefore be ~20% *lower* than raptorbt's on the
           same equity curve, and they are the correct ones.  All 29 non-ratio
           fields agree exactly.

        Args:
            returns: Per-bar portfolio returns as fractions (``0.01`` = 1%).
            equity_curve: Portfolio value per bar; ``equity_curve[0]`` is the
                initial capital.
            trades: Completed round-trip trades used for trade-level metrics.
            durations: Holding period of each trade in bars, aligned with
                *trades*.  When ``None``, duration metrics report ``0.0``.
            periods_per_year: Annualisation factor.  ``252`` for daily bars;
                for intraday bars use bars-per-day × trading-days-per-year.
            total_fees_paid: Cumulative transaction costs in cash units.
            exposure_pct: Percentage of bars holding a non-zero position.
            open_trade_pnl: Unrealised P&L of any position still open at the end.

        Returns:
            Fully populated :class:`BacktestMetrics`.
        """
        rets = np.asarray(returns, dtype=np.float64)
        equity = np.asarray(equity_curve, dtype=np.float64)

        start_value = float(equity[0]) if equity.size else 0.0
        end_value = float(equity[-1]) if equity.size else 0.0
        total_return_pct = _safe_div(end_value - start_value, start_value) * 100.0

        # ── Risk-adjusted ratios ──────────────────────────────────────────
        ann = float(np.sqrt(periods_per_year))
        mean_ret = float(np.mean(rets)) if rets.size else 0.0
        std_ret = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
        sharpe_ratio = _safe_div(mean_ret, std_ret) * ann

        downside = np.minimum(rets, 0.0)
        downside_std = float(np.sqrt(np.mean(downside**2))) if rets.size else 0.0
        sortino_ratio = _safe_div(mean_ret, downside_std) * ann

        gains = float(np.sum(rets[rets > 0.0])) if rets.size else 0.0
        losses = float(-np.sum(rets[rets < 0.0])) if rets.size else 0.0
        omega_ratio = _safe_div(gains, losses)

        # ── Drawdown ──────────────────────────────────────────────────────
        max_drawdown_pct, max_drawdown_duration = _drawdown_stats(equity)

        n_bars = int(rets.size)
        if n_bars > 0 and start_value > 0.0 and end_value > 0.0:
            years = n_bars / periods_per_year
            ann_return_pct = ((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0
        else:
            ann_return_pct = 0.0
        calmar_ratio = _safe_div(ann_return_pct, max_drawdown_pct)
        recovery_factor = _safe_div(total_return_pct, max_drawdown_pct)

        # ── Trade-level statistics ────────────────────────────────────────
        pnls = np.array([t.pnl for t in trades], dtype=np.float64)
        trade_rets = np.array([t.return_pct for t in trades], dtype=np.float64)
        is_win = pnls > 0.0
        is_loss = pnls < 0.0
        n_closed = int(pnls.size)
        n_win = int(np.count_nonzero(is_win))
        n_loss = int(np.count_nonzero(is_loss))

        gross_profit = float(np.sum(pnls[is_win])) if n_win else 0.0
        gross_loss = float(-np.sum(pnls[is_loss])) if n_loss else 0.0

        avg_win_pct = float(np.mean(trade_rets[is_win])) if n_win else 0.0
        avg_loss_pct = float(np.mean(trade_rets[is_loss])) if n_loss else 0.0
        pnl_std = float(np.std(pnls, ddof=1)) if n_closed > 1 else 0.0

        durs = (
            np.asarray(durations, dtype=np.float64)
            if durations is not None
            else np.zeros(n_closed, dtype=np.float64)
        )

        return cls(
            # Core performance
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            # Drawdown
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration=max_drawdown_duration,
            # Trade counts
            total_trades=n_closed,
            total_closed_trades=n_closed,
            total_open_trades=1 if open_trade_pnl != 0.0 else 0,
            winning_trades=n_win,
            losing_trades=n_loss,
            # Trade performance
            win_rate_pct=_safe_div(float(n_win), float(n_closed)) * 100.0,
            profit_factor=_safe_div(gross_profit, gross_loss),
            expectancy=float(np.mean(pnls)) if n_closed else 0.0,
            sqn=_safe_div(float(np.mean(pnls)) if n_closed else 0.0, pnl_std)
            * float(np.sqrt(n_closed)),
            avg_trade_return_pct=float(np.mean(trade_rets)) if n_closed else 0.0,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            best_trade_pct=float(np.max(trade_rets)) if n_closed else 0.0,
            worst_trade_pct=float(np.min(trade_rets)) if n_closed else 0.0,
            payoff_ratio=_safe_div(avg_win_pct, abs(avg_loss_pct)),
            recovery_factor=recovery_factor,
            # Duration
            avg_holding_period=float(np.mean(durs)) if durs.size else 0.0,
            avg_winning_duration=float(np.mean(durs[is_win])) if n_win and durs.size else 0.0,
            avg_losing_duration=float(np.mean(durs[is_loss])) if n_loss and durs.size else 0.0,
            # Streaks
            max_consecutive_wins=_max_streak(is_win),
            max_consecutive_losses=_max_streak(is_loss),
            # Portfolio
            start_value=start_value,
            end_value=end_value,
            total_fees_paid=total_fees_paid,
            open_trade_pnl=open_trade_pnl,
            exposure_pct=exposure_pct,
        )

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
            total_return_pct=_num(m, "total_return_pct", 0.0),
            sharpe_ratio=_num(m, "sharpe_ratio", 0.0),
            sortino_ratio=_num(m, "sortino_ratio", 0.0),
            calmar_ratio=_num(m, "calmar_ratio", 0.0),
            omega_ratio=_num(m, "omega_ratio", 0.0),
            # Drawdown
            max_drawdown_pct=_num(m, "max_drawdown_pct", 0.0),
            max_drawdown_duration=_int(m, "max_drawdown_duration", 0),
            # Trade counts
            total_trades=_int(m, "total_trades", 0),
            total_closed_trades=_int(m, "total_closed_trades", 0),
            total_open_trades=_int(m, "total_open_trades", 0),
            winning_trades=_int(m, "winning_trades", 0),
            losing_trades=_int(m, "losing_trades", 0),
            # Trade performance
            win_rate_pct=_num(m, "win_rate_pct", 0.0),
            profit_factor=_num(m, "profit_factor", 0.0),
            expectancy=_num(m, "expectancy", 0.0),
            sqn=_num(m, "sqn", 0.0),
            avg_trade_return_pct=_num(m, "avg_trade_return_pct", 0.0),
            avg_win_pct=_num(m, "avg_win_pct", 0.0),
            avg_loss_pct=_num(m, "avg_loss_pct", 0.0),
            best_trade_pct=_num(m, "best_trade_pct", 0.0),
            worst_trade_pct=_num(m, "worst_trade_pct", 0.0),
            payoff_ratio=_num(m, "payoff_ratio", 0.0),
            recovery_factor=_num(m, "recovery_factor", 0.0),
            # Duration
            avg_holding_period=_num(m, "avg_holding_period", 0.0),
            avg_winning_duration=_num(m, "avg_winning_duration", 0.0),
            avg_losing_duration=_num(m, "avg_losing_duration", 0.0),
            # Streaks
            max_consecutive_wins=_int(m, "max_consecutive_wins", 0),
            max_consecutive_losses=_int(m, "max_consecutive_losses", 0),
            # Portfolio
            start_value=_num(m, "start_value", 0.0),
            end_value=_num(m, "end_value", 0.0),
            total_fees_paid=_num(m, "total_fees_paid", 0.0),
            open_trade_pnl=_num(m, "open_trade_pnl", 0.0),
            exposure_pct=_num(m, "exposure_pct", 0.0),
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
