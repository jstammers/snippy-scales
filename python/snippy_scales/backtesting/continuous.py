"""Continuous target-position execution engine.

The default :class:`~snippy_scales.backtesting.engine.RaptorExecutionEngine`
consumes boolean entry/exit arrays, which are produced by
:class:`~snippy_scales.backtesting.signals.SignFlipInterpreter` from a signed
position series.  That conversion reads only the **sign** of each position, so
the magnitude a strategy computes is silently discarded — a vol-scaled signal
of ``0.3`` and one of ``3.0`` produce byte-identical backtests.

That is fatal for any model whose output is a *distribution* rather than a
direction: the whole point of estimating conditional variance is to size
positions continuously.  :class:`TargetPositionEngine` closes that gap.  It
consumes ``InstrumentSpec.positions`` directly and computes

.. code-block:: text

    gross_t = Σ_i  w_i · pos_{i,t-lag} · ret_{i,t}
    cost_t  = Σ_i  w_i · cost_model(|Δpos_{i,t-lag}|, price_{i,t})
    net_t   = gross_t - cost_t
    equity_t = equity_{t-1} · (1 + net_t)

Positions are expressed in units of equity: ``1.0`` means fully invested,
``-0.5`` means short half the portfolio, ``2.0`` means 2× levered.

Timing convention matches raptorbt: a position established at the close of bar
``t`` earns the return of bar ``t+1`` (``execution_lag=1``).  Setting
``execution_lag=0`` deliberately introduces lookahead and exists only so tests
can assert that lookahead *does* inflate performance.

The engine satisfies the :class:`~snippy_scales.backtesting.engine.ExecutionEngine`
Protocol, so it drops into existing runners without any change to
``evaluation/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from snippy_scales._constants import TRADING_DAYS_PER_YEAR
from snippy_scales.backtesting.costs import CostModel, ProportionalCost
from snippy_scales.backtesting.domain import (
    BacktestMetrics,
    BacktestResult,
    Trade,
    drawdown_curve,
)

if TYPE_CHECKING:
    from snippy_scales.backtesting.engine import InstrumentSpec

__all__ = [
    "ContinuousConfig",
    "make_continuous_config",
    "TargetPositionEngine",
    "positions_from_signals",
]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContinuousConfig:
    """Configuration for :class:`TargetPositionEngine`.

    Attributes:
        initial_capital: Starting portfolio value in currency units.
        cost_model: Transaction-cost model.  Defaults to
            :class:`~snippy_scales.backtesting.costs.ProportionalCost` with the
            same 10 bp / 5 bp defaults as
            :func:`~snippy_scales.backtesting.engine.make_config`.
        periods_per_year: Annualisation factor.  ``252`` for daily bars; for
            intraday bars use bars-per-day × trading-days-per-year (e.g. 1-hour
            CME bars ≈ ``23 * 252``).
        execution_lag: Bars between a position being decided and it earning a
            return.  ``1`` (the default) is the honest setting.
    """

    initial_capital: float = 100_000.0
    cost_model: CostModel = field(default_factory=ProportionalCost)
    periods_per_year: float = TRADING_DAYS_PER_YEAR
    execution_lag: int = 1

    def __post_init__(self) -> None:
        """Validate configuration invariants.

        Raises:
            ValueError: If capital is non-positive, the annualisation factor is
                non-positive, or the execution lag is negative.
        """
        if self.initial_capital <= 0.0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        if self.periods_per_year <= 0.0:
            raise ValueError(f"periods_per_year must be positive, got {self.periods_per_year}")
        if self.execution_lag < 0:
            raise ValueError(f"execution_lag must be non-negative, got {self.execution_lag}")


def make_continuous_config(
    *,
    initial_capital: float = 100_000.0,
    cost_model: CostModel | None = None,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    execution_lag: int = 1,
) -> ContinuousConfig:
    """Create a :class:`ContinuousConfig`, mirroring :func:`make_config`.

    Args:
        initial_capital: Starting portfolio value.
        cost_model: Cost model; defaults to :class:`ProportionalCost`.
        periods_per_year: Annualisation factor.
        execution_lag: Bars between decision and return accrual.

    Returns:
        A validated :class:`ContinuousConfig`.
    """
    return ContinuousConfig(
        initial_capital=initial_capital,
        cost_model=cost_model if cost_model is not None else ProportionalCost(),
        periods_per_year=periods_per_year,
        execution_lag=execution_lag,
    )


# ── Signal fallback ───────────────────────────────────────────────────────────


def positions_from_signals(
    entries: np.ndarray,
    exits: np.ndarray,
    direction: int = 1,
) -> np.ndarray:
    """Reconstruct a ±1 position series from boolean entry/exit arrays.

    This is the inverse of
    :class:`~snippy_scales.backtesting.signals.SignFlipInterpreter` and lets
    :class:`TargetPositionEngine` run on an
    :class:`~snippy_scales.backtesting.engine.InstrumentSpec` that carries only
    boolean signals — which is what makes engine-to-engine parity testing
    possible.

    An exit on the same bar as an entry is resolved as a *flat* bar: the exit
    wins, matching the sign-flip interpreter's treatment of a direct reversal.

    Args:
        entries: Boolean array; ``True`` opens a position.
        exits: Boolean array; ``True`` closes it.
        direction: ``1`` for a long book, ``-1`` for a short book.

    Returns:
        Float array of positions in ``{-1, 0, +1}`` scaled by *direction*.
    """
    entries_arr = np.asarray(entries, dtype=bool)
    exits_arr = np.asarray(exits, dtype=bool)

    state = np.zeros(entries_arr.size, dtype=np.float64)
    holding = False
    for i in range(entries_arr.size):
        if exits_arr[i]:
            holding = False
        elif entries_arr[i]:
            holding = True
        state[i] = 1.0 if holding else 0.0

    return state * float(direction)


def _resolve_positions(spec: InstrumentSpec) -> np.ndarray:
    """Return the continuous position series for *spec*.

    Uses ``spec.positions`` when present; otherwise reconstructs a ±1 series
    from the boolean signals.

    Args:
        spec: Instrument leg.

    Returns:
        Float position array aligned with the spec's bars.
    """
    positions = getattr(spec, "positions", None)
    if positions is not None:
        return np.asarray(positions, dtype=np.float64)
    return positions_from_signals(spec.entries, spec.exits, spec.direction)


# ── Engine ────────────────────────────────────────────────────────────────────


class TargetPositionEngine:
    """:class:`ExecutionEngine` that honours continuous position sizing.

    Args:
        config: Default configuration used when :meth:`execute` is called
            without one.
    """

    def __init__(self, config: ContinuousConfig | None = None) -> None:
        self._default_config = config if config is not None else ContinuousConfig()

    def execute(
        self,
        instruments: list[InstrumentSpec],
        *,
        config: Any | None = None,
        sync_mode: str = "any",
    ) -> BacktestResult:
        """Run a continuous-position backtest over one or more instrument legs.

        Args:
            instruments: One :class:`InstrumentSpec` per leg.  Every leg must
                have the same number of bars.
            config: :class:`ContinuousConfig`, or ``None`` to use the instance
                default.
            sync_mode: Accepted for Protocol compatibility and ignored — legs
                are always combined additively by weight.

        Returns:
            :class:`BacktestResult` for the combined portfolio.

        Raises:
            ValueError: If *instruments* is empty, if legs disagree on bar
                count, or if *config* is not a :class:`ContinuousConfig`.
        """
        del sync_mode  # legs are combined additively; no synchronisation modes

        if not instruments:
            raise ValueError("instruments must not be empty")

        cfg = config if config is not None else self._default_config
        if not isinstance(cfg, ContinuousConfig):
            raise ValueError(
                "TargetPositionEngine requires a ContinuousConfig "
                f"(got {type(cfg).__name__}); build one with make_continuous_config()"
            )

        n_bars = int(instruments[0].close.size)
        mismatched = [s.symbol for s in instruments if int(s.close.size) != n_bars]
        if mismatched:
            raise ValueError(
                f"all instrument legs must have {n_bars} bars; mismatched: {', '.join(mismatched)}"
            )
        if n_bars == 0:
            raise ValueError("instrument legs must contain at least one bar")

        gross = np.zeros(n_bars, dtype=np.float64)
        costs = np.zeros(n_bars, dtype=np.float64)
        exposed = np.zeros(n_bars, dtype=bool)
        per_leg: list[tuple[InstrumentSpec, np.ndarray, np.ndarray]] = []

        for spec in instruments:
            close = np.asarray(spec.close, dtype=np.float64)
            positions = _resolve_positions(spec)
            held = _lag(positions, cfg.execution_lag)

            bar_return = _bar_returns(close)
            leg_gross = spec.weight * held * bar_return

            traded = np.abs(np.diff(held, prepend=0.0))
            leg_cost = spec.weight * np.asarray(
                cfg.cost_model.cost_fraction(traded, close), dtype=np.float64
            )

            gross += leg_gross
            costs += leg_cost
            exposed |= held != 0.0
            per_leg.append((spec, positions, leg_gross - leg_cost))

        net = gross - costs
        equity = cfg.initial_capital * np.cumprod(1.0 + net)
        prior_equity = np.concatenate(([cfg.initial_capital], equity[:-1]))

        trades: list[Trade] = []
        durations: list[float] = []
        for spec, positions, leg_net in per_leg:
            leg_trades, leg_durations = _synthesise_trades(
                spec=spec,
                positions=positions,
                leg_net=leg_net,
                prior_equity=prior_equity,
                execution_lag=cfg.execution_lag,
            )
            trades.extend(leg_trades)
            durations.extend(leg_durations)

        total_fees_paid = float(np.sum(costs * prior_equity))
        exposure_pct = float(np.count_nonzero(exposed)) / n_bars * 100.0

        metrics = BacktestMetrics.from_returns(
            returns=net,
            equity_curve=equity,
            trades=trades,
            durations=np.asarray(durations, dtype=np.float64),
            periods_per_year=cfg.periods_per_year,
            total_fees_paid=total_fees_paid,
            exposure_pct=exposure_pct,
        )

        symbols = [spec.symbol for spec in instruments]
        return BacktestResult(
            symbol=symbols[0] if len(symbols) == 1 else symbols,
            metrics=metrics,
            equity_curve=equity,
            drawdown_curve=drawdown_curve(equity),
            returns=net,
            trades=trades,
        )


# ── Internals ─────────────────────────────────────────────────────────────────


def _bar_returns(close: np.ndarray) -> np.ndarray:
    """Return simple bar-over-bar returns with a leading zero.

    Args:
        close: Close prices.

    Returns:
        Array of the same length; element ``0`` is ``0.0``.
    """
    out = np.zeros(close.size, dtype=np.float64)
    if close.size < 2:
        return out
    prev = close[:-1]
    out[1:] = np.where(prev != 0.0, close[1:] / np.where(prev != 0.0, prev, 1.0) - 1.0, 0.0)
    return out


def _lag(positions: np.ndarray, lag: int) -> np.ndarray:
    """Shift *positions* forward by *lag* bars, padding with zeros.

    Args:
        positions: Target position series.
        lag: Number of bars to delay. ``0`` leaves the series unchanged.

    Returns:
        Lagged copy of *positions*.
    """
    if lag == 0:
        return np.asarray(positions, dtype=np.float64)
    held = np.zeros_like(positions, dtype=np.float64)
    if lag < positions.size:
        held[lag:] = positions[:-lag]
    return held


def _synthesise_trades(
    *,
    spec: InstrumentSpec,
    positions: np.ndarray,
    leg_net: np.ndarray,
    prior_equity: np.ndarray,
    execution_lag: int,
) -> tuple[list[Trade], list[float]]:
    """Derive round-trip trades from a continuous position series.

    A trade opens when the position leaves zero and closes when it returns to
    zero or changes sign.  Its cash P&L is the equity-weighted sum of the leg's
    net per-bar contributions across the holding period, so trade P&L reconciles
    with the portfolio equity curve by construction.  ``return_pct`` is the
    direction-signed price move between entry and exit close, matching the
    convention raptorbt reports.

    Args:
        spec: The instrument leg.
        positions: Decided (un-lagged) position series.
        leg_net: This leg's net return contribution per bar.
        prior_equity: Portfolio equity at the start of each bar.
        execution_lag: Bars between decision and return accrual.

    Returns:
        Tuple of the trade list and a parallel list of holding periods in bars.
    """
    close = np.asarray(spec.close, dtype=np.float64)
    timestamps = np.asarray(spec.timestamps)
    n_bars = positions.size

    trades: list[Trade] = []
    durations: list[float] = []

    sign = np.sign(positions)
    entry_idx: int | None = None

    for i in range(n_bars):
        current = sign[i]
        previous = sign[i - 1] if i > 0 else 0.0

        if current != previous and previous != 0.0 and entry_idx is not None:
            trades.append(
                _build_trade(
                    spec=spec,
                    close=close,
                    timestamps=timestamps,
                    leg_net=leg_net,
                    prior_equity=prior_equity,
                    entry_idx=entry_idx,
                    exit_idx=i,
                    direction=int(previous),
                    execution_lag=execution_lag,
                )
            )
            durations.append(float(i - entry_idx))
            entry_idx = None

        if current != 0.0 and entry_idx is None:
            entry_idx = i

    # A position still open on the final bar is marked to market and closed
    # there, so its contribution is not silently dropped from trade statistics.
    if entry_idx is not None and entry_idx < n_bars - 1:
        trades.append(
            _build_trade(
                spec=spec,
                close=close,
                timestamps=timestamps,
                leg_net=leg_net,
                prior_equity=prior_equity,
                entry_idx=entry_idx,
                exit_idx=n_bars - 1,
                direction=int(sign[n_bars - 1]),
                execution_lag=execution_lag,
            )
        )
        durations.append(float(n_bars - 1 - entry_idx))

    return trades, durations


def _build_trade(
    *,
    spec: InstrumentSpec,
    close: np.ndarray,
    timestamps: np.ndarray,
    leg_net: np.ndarray,
    prior_equity: np.ndarray,
    entry_idx: int,
    exit_idx: int,
    direction: int,
    execution_lag: int,
) -> Trade:
    """Build a single :class:`Trade` spanning ``[entry_idx, exit_idx]``.

    Args:
        spec: The instrument leg.
        close: Close prices.
        timestamps: Bar timestamps in nanoseconds.
        leg_net: Net per-bar return contribution for this leg.
        prior_equity: Portfolio equity at the start of each bar.
        entry_idx: Bar index at which the position was established.
        exit_idx: Bar index at which it was closed.
        direction: ``1`` for long, ``-1`` for short.
        execution_lag: Bars between decision and return accrual.

    Returns:
        A populated :class:`Trade`.
    """
    # Returns accrue from the bar the position becomes effective through to the
    # bar the exit itself earns.
    start = min(entry_idx + execution_lag, leg_net.size)
    stop = min(exit_idx + execution_lag, leg_net.size)
    pnl = float(np.sum(leg_net[start:stop] * prior_equity[start:stop]))

    entry_price = float(close[entry_idx])
    exit_price = float(close[exit_idx])
    return_pct = direction * (exit_price / entry_price - 1.0) * 100.0 if entry_price != 0.0 else 0.0

    return Trade(
        symbol=spec.symbol,
        entry_time=int(timestamps[entry_idx]),
        exit_time=int(timestamps[exit_idx]),
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        return_pct=return_pct,
    )
