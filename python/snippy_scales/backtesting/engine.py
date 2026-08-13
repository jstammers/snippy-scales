"""Execution engine abstraction and raptorbt implementation.

Defines:

* :class:`InstrumentSpec` — a frozen dataclass describing one instrument leg
  (OHLCV data + entry/exit signals + direction + weight).
* :class:`ExecutionEngine` — a :class:`~typing.Protocol` for swappable
  execution backends (e.g. raptorbt, vectorbt, paper-trading).
* :class:`RaptorExecutionEngine` — the default implementation backed by
  ``raptorbt.run_basket_backtest``.
* :func:`make_config` — factory for ``raptorbt.PyBacktestConfig``.
* Low-level wrappers :func:`run_single`, :func:`run_long_short`,
  :func:`run_basket` — thin typed wrappers kept for direct use and backward
  compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np  # noqa: TC002
import raptorbt

from snippy_scales.backtesting.domain import BacktestResult

# ── InstrumentSpec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InstrumentSpec:
    """Immutable specification for one instrument leg in a basket backtest.

    Each instrument in a basket backtest (including long and short legs of
    the same underlying asset) is represented as a separate ``InstrumentSpec``.
    The dataclass is frozen to prevent accidental mutation after construction.

    Attributes:
        symbol: Unique identifier (e.g. ``"ES.c.0"`` or ``"AAPL_long"``).
        timestamps: Bar timestamps as int64 nanoseconds.
        open: Bar open prices (float64).
        high: Bar high prices (float64).
        low: Bar low prices (float64).
        close: Bar close prices (float64).
        volume: Bar volume (float64).
        entries: Boolean array; ``True`` on the bar where the position opens.
        exits: Boolean array; ``True`` on the bar where the position closes.
        direction: ``1`` for long positions, ``-1`` for short positions.
        weight: Capital allocation fraction for this leg (0 < w ≤ 1).
        positions: Optional signed, continuously-sized target positions in units
            of equity (``0.5`` = half the portfolio long, ``-2.0`` = 2× short).
            Consumed by
            :class:`~snippy_scales.backtesting.continuous.TargetPositionEngine`
            and **ignored by** :class:`RaptorExecutionEngine`, which reads only
            ``entries``/``exits``.  Leave as ``None`` for sign-only strategies.
    """

    symbol: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    entries: np.ndarray
    exits: np.ndarray
    direction: int
    weight: float
    positions: np.ndarray | None = None

    def to_tuple(self) -> tuple[Any, ...]:
        """Return the 11-element tuple expected by ``raptorbt.run_basket_backtest``.

        Returns:
            Tuple of ``(timestamps, open, high, low, close, volume,
            entries, exits, direction, weight, symbol)``.
        """
        return (
            self.timestamps,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.entries,
            self.exits,
            self.direction,
            self.weight,
            self.symbol,
        )


# ── Config factory ────────────────────────────────────────────────────────────


def make_config(
    *,
    initial_capital: float = 100_000.0,
    fees: float = 0.001,
    slippage: float = 0.0005,
    upon_bar_close: bool = True,
) -> Any:
    """Create a ``raptorbt.PyBacktestConfig`` with sensible defaults.

    Args:
        initial_capital: Starting portfolio value in currency units.
        fees: Per-trade commission as a fraction of notional
            (e.g. ``0.001`` = 10 bps).
        slippage: Round-trip slippage as a fraction of price
            (e.g. ``0.0005`` = 5 bps).
        upon_bar_close: If ``True`` fills execute at bar close; else at open.

    Returns:
        Configured ``raptorbt.PyBacktestConfig`` instance.
    """
    return raptorbt.PyBacktestConfig(
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage,
        upon_bar_close=upon_bar_close,
    )


# ── Low-level raptorbt wrappers ───────────────────────────────────────────────


def run_single(
    *,
    symbol: str,
    timestamps: np.ndarray,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    volume: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    direction: int,
    weight: float = 1.0,
    config: Any | None = None,
) -> BacktestResult:
    """Run a single-instrument, single-direction backtest via raptorbt.

    Args:
        symbol: Instrument identifier.
        timestamps: Bar timestamps as int64 nanoseconds.
        open_prices: Open prices (float64).
        high_prices: High prices (float64).
        low_prices: Low prices (float64).
        close_prices: Close prices (float64).
        volume: Bar volume (float64).
        entries: Boolean entry array.
        exits: Boolean exit array.
        direction: ``1`` = long, ``-1`` = short.
        weight: Capital allocation fraction (default ``1.0``).
        config: raptorbt config; uses :func:`make_config` defaults if ``None``.

    Returns:
        :class:`BacktestResult` wrapping the raptorbt output.
    """
    cfg = config if config is not None else make_config()
    raw = raptorbt.run_single_backtest(
        timestamps=timestamps,
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume,
        entries=entries,
        exits=exits,
        direction=direction,
        weight=weight,
        symbol=symbol,
        config=cfg,
    )
    return BacktestResult.from_raptorbt(raw, symbol=symbol)


def run_long_short(
    *,
    symbol: str,
    timestamps: np.ndarray,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    volume: np.ndarray,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
    long_weight: float = 0.5,
    short_weight: float = 0.5,
    config: Any | None = None,
) -> BacktestResult:
    """Run a simultaneous long/short strategy on one instrument via raptorbt.

    Uses ``run_multi_backtest`` with ``combine_mode="independent"`` so the long
    and short books each receive their own capital allocation.

    Args:
        symbol: Instrument identifier.
        timestamps: Bar timestamps as int64 nanoseconds.
        open_prices: Open prices (float64).
        high_prices: High prices (float64).
        low_prices: Low prices (float64).
        close_prices: Close prices (float64).
        volume: Bar volume (float64).
        long_entries: Entry signals for the long book.
        long_exits: Exit signals for the long book.
        short_entries: Entry signals for the short book.
        short_exits: Exit signals for the short book.
        long_weight: Capital fraction for the long book.
        short_weight: Capital fraction for the short book.
        config: raptorbt config; uses :func:`make_config` defaults if ``None``.

    Returns:
        :class:`BacktestResult` with combined long/short portfolio metrics.
    """
    cfg = config if config is not None else make_config()
    strategies: list[tuple[Any, ...]] = [
        (long_entries, long_exits, 1, long_weight, f"{symbol}_long"),
        (short_entries, short_exits, -1, short_weight, f"{symbol}_short"),
    ]
    raw = raptorbt.run_multi_backtest(
        timestamps=timestamps,
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume,
        strategies=strategies,
        config=cfg,
        combine_mode="independent",
    )
    return BacktestResult.from_raptorbt(raw, symbol=symbol)


def run_basket(
    *,
    instruments: list[InstrumentSpec],
    config: Any | None = None,
    sync_mode: str = "any",
) -> BacktestResult:
    """Run a multi-instrument basket backtest via raptorbt.

    Args:
        instruments: List of :class:`InstrumentSpec`, one per (asset, direction)
            pair.  Must not be empty.
        config: raptorbt config; uses :func:`make_config` defaults if ``None``.
        sync_mode: Basket synchronisation mode: ``"any"``, ``"all"``,
            ``"majority"``, or ``"master"`` (default ``"any"``).

    Returns:
        :class:`BacktestResult` for the combined basket portfolio.

    Raises:
        ValueError: If *instruments* is empty.
    """
    if not instruments:
        raise ValueError("instruments must not be empty")
    cfg = config if config is not None else make_config()
    tuples = [spec.to_tuple() for spec in instruments]
    raw = raptorbt.run_basket_backtest(
        instruments=tuples,
        config=cfg,
        sync_mode=sync_mode,
    )
    symbols = [spec.symbol for spec in instruments]
    return BacktestResult.from_raptorbt(raw, symbol=symbols)


# ── ExecutionEngine Protocol ──────────────────────────────────────────────────


@runtime_checkable
class ExecutionEngine(Protocol):
    """Protocol for swappable execution backends.

    Implement this protocol to replace raptorbt with an alternative engine
    (e.g. vectorbt, paper-trading simulator, live order router) without
    changing any runner code.

    The engine receives a fully-prepared list of :class:`InstrumentSpec`
    objects and a raptorbt-compatible config, and returns a
    :class:`BacktestResult`.
    """

    def execute(
        self,
        instruments: list[InstrumentSpec],
        *,
        config: Any | None = None,
        sync_mode: str = "any",
    ) -> BacktestResult:
        """Execute a basket backtest for the given instruments.

        Args:
            instruments: One :class:`InstrumentSpec` per (asset, direction) leg.
            config: Engine-specific configuration object.
            sync_mode: Synchronisation strategy across instruments.

        Returns:
            :class:`BacktestResult` for the combined portfolio.
        """
        ...


class RaptorExecutionEngine:
    """Default :class:`ExecutionEngine` backed by ``raptorbt``.

    Delegates directly to :func:`run_basket` which calls
    ``raptorbt.run_basket_backtest`` under the hood.
    """

    def execute(
        self,
        instruments: list[InstrumentSpec],
        *,
        config: Any | None = None,
        sync_mode: str = "any",
    ) -> BacktestResult:
        """Execute a basket backtest via raptorbt.

        Args:
            instruments: One :class:`InstrumentSpec` per (asset, direction) leg.
            config: ``raptorbt.PyBacktestConfig`` or ``None`` to use defaults.
            sync_mode: Basket synchronisation mode.

        Returns:
            :class:`BacktestResult` from raptorbt.
        """
        return run_basket(instruments=instruments, config=config, sync_mode=sync_mode)
