"""Backtesting infrastructure using raptorbt as the execution engine.

This module provides:

* **Result types** – :class:`BacktestMetrics` and :class:`BacktestResult`
  wrap raptorbt output with full Python type safety.
* **Signal conversion** – :func:`positions_to_signals` converts signed-float
  position arrays (the native strategy output format) to the boolean
  entry/exit arrays that raptorbt expects.
* **OHLCV extraction** – :func:`extract_ohlcv` normalises Polars bar DataFrames
  to aligned NumPy arrays, synthesising missing columns (open/high/low/volume)
  from close prices where necessary.
* **Low-level runners** – :func:`run_single`, :func:`run_long_short`, and
  :func:`run_basket` are thin, typed wrappers around the three main raptorbt
  entry points.
* **High-level runners** – :class:`BacktestRunner` (single-asset) and
  :class:`BasketRunner` (multi-asset) accept strategy objects directly and
  handle all signal conversion and raptorbt wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import raptorbt

if TYPE_CHECKING:
    import polars as pl

    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

# ── Synthetic OHLCV helpers ────────────────────────────────────────────────────

# Base timestamp: 2020-01-01 00:00:00 UTC in nanoseconds (int64).
_BASE_NS: int = 1_577_836_800_000_000_000
_DAY_NS: int = 86_400_000_000_000


def extract_ohlcv(bars: pl.DataFrame) -> dict[str, np.ndarray]:
    """Extract aligned OHLCV NumPy arrays from a Polars bar DataFrame.

    If the DataFrame is missing open/high/low/volume columns they are
    synthesised deterministically from the close price so that the result
    is always a valid OHLCV dataset.  Timestamps are taken from a ``ts``
    column (expected as int64 nanoseconds) or generated as sequential
    daily timestamps starting from 2020-01-01 UTC.

    Args:
        bars: Polars DataFrame with at minimum a ``close`` column.

    Returns:
        Dictionary with keys ``timestamps``, ``open``, ``high``, ``low``,
        ``close``, and ``volume`` mapped to float64 (or int64 for timestamps)
        NumPy arrays of equal length.
    """
    close = bars["close"].to_numpy().astype(np.float64)
    n = len(close)

    # Timestamps ──────────────────────────────────────────────────────────────
    if "ts" in bars.columns:
        timestamps = bars["ts"].to_numpy().astype(np.int64)
    else:
        timestamps = _BASE_NS + np.arange(n, dtype=np.int64) * _DAY_NS

    # Open ────────────────────────────────────────────────────────────────────
    if "open" in bars.columns:
        open_arr = bars["open"].to_numpy().astype(np.float64)
    else:
        # Assume bar opens at the previous close (no gap by default).
        open_arr = np.empty(n, dtype=np.float64)
        open_arr[0] = close[0]
        open_arr[1:] = close[:-1]

    # High / Low ──────────────────────────────────────────────────────────────
    if "high" in bars.columns:
        high_arr = bars["high"].to_numpy().astype(np.float64)
    else:
        high_arr = np.maximum(open_arr, close) * 1.001

    if "low" in bars.columns:
        low_arr = bars["low"].to_numpy().astype(np.float64)
    else:
        low_arr = np.minimum(open_arr, close) * 0.999

    # Volume ──────────────────────────────────────────────────────────────────
    if "volume" in bars.columns:
        volume = bars["volume"].to_numpy().astype(np.float64)
    else:
        volume = np.ones(n, dtype=np.float64)

    return {
        "timestamps": timestamps,
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close,
        "volume": volume,
    }


# ── Signal conversion ─────────────────────────────────────────────────────────


def positions_to_signals(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a signed-position array to boolean entry/exit arrays.

    Strategy ``generate_signals()`` methods return signed floats where a
    positive value indicates a long position and a negative value indicates
    a short position.  raptorbt consumes boolean entry/exit arrays.  This
    function performs that conversion by detecting sign changes.

    Args:
        positions: Float array of signed target positions. Positive → long,
            negative → short, zero → flat.

    Returns:
        A 4-tuple of ``(long_entries, long_exits, short_entries, short_exits)``,
        all boolean arrays of the same length as *positions*.

    Examples:
        >>> pos = np.array([0.0, 0.8, 0.9, 0.0, -0.7, 0.0])
        >>> le, lx, se, sx = positions_to_signals(pos)
        >>> le.tolist()  # long entry at index 1
        [False, True, False, False, False, False]
    """
    is_long = positions > 0.0
    is_short = positions < 0.0

    prev_long = np.empty_like(is_long)
    prev_long[0] = False
    prev_long[1:] = is_long[:-1]

    prev_short = np.empty_like(is_short)
    prev_short[0] = False
    prev_short[1:] = is_short[:-1]

    long_entries = is_long & ~prev_long
    long_exits = prev_long & ~is_long
    short_entries = is_short & ~prev_short
    short_exits = prev_short & ~is_short

    return long_entries, long_exits, short_entries, short_exits


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BacktestMetrics:
    """Key performance metrics extracted from a raptorbt result.

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


@dataclass
class BacktestResult:
    """Full result from a raptorbt backtest run.

    Attributes:
        symbol: Instrument symbol or list of symbols for basket runs.
        metrics: Aggregated performance metrics.
        equity_curve: Portfolio equity over time as a float64 NumPy array.
        drawdown_curve: Drawdown time series as a float64 NumPy array.
        returns: Period returns as a float64 NumPy array.
        trades: List of ``raptorbt.PyTrade`` objects.
    """

    symbol: str | list[str]
    metrics: BacktestMetrics
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    returns: np.ndarray
    trades: list[Any] = field(default_factory=list)

    @classmethod
    def from_raptorbt(cls, result: Any, symbol: str | list[str]) -> BacktestResult:
        """Construct from a ``raptorbt.PyBacktestResult`` object.

        Args:
            result: raptorbt result object.
            symbol: Symbol or list of symbols associated with this result.

        Returns:
            Typed :class:`BacktestResult` instance.
        """
        return cls(
            symbol=symbol,
            metrics=BacktestMetrics.from_raptorbt(result.metrics),
            equity_curve=np.asarray(result.equity_curve(), dtype=np.float64),
            drawdown_curve=np.asarray(result.drawdown_curve(), dtype=np.float64),
            returns=np.asarray(result.returns(), dtype=np.float64),
            trades=list(result.trades()),
        )


# ── Instrument specification ──────────────────────────────────────────────────


@dataclass
class InstrumentSpec:
    """Specifies one instrument's data and signals for a basket backtest.

    Each instrument in a basket backtest (including long and short legs of
    the same underlying asset) is represented as a separate ``InstrumentSpec``.

    Attributes:
        symbol: Unique identifier (e.g. ``"ES.c.0"`` or ``"AAPL_long"``).
        timestamps: Bar timestamps as int64 nanoseconds.
        open: Bar open prices (float64).
        high: Bar high prices (float64).
        low: Bar low prices (float64).
        close: Bar close prices (float64).
        volume: Bar volume (float64).
        entries: Boolean array; ``True`` on the bar where position opens.
        exits: Boolean array; ``True`` on the bar where position closes.
        direction: ``1`` for long positions, ``-1`` for short positions.
        weight: Capital allocation fraction for this instrument (0 < w ≤ 1).
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
        entries: Boolean array; ``True`` on entry bars.
        exits: Boolean array; ``True`` on exit bars.
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
    and short books each receive their own capital allocation and their P&Ls are
    aggregated into a single result.

    Args:
        symbol: Instrument identifier.
        timestamps: Bar timestamps as int64 nanoseconds.
        open_prices: Open prices (float64).
        high_prices: High prices (float64).
        low_prices: Low prices (float64).
        close_prices: Close prices (float64).
        volume: Bar volume (float64).
        long_entries: ``True`` on bars where a long position should open.
        long_exits: ``True`` on bars where a long position should close.
        short_entries: ``True`` on bars where a short position should open.
        short_exits: ``True`` on bars where a short position should close.
        long_weight: Capital fraction allocated to the long book.
        short_weight: Capital fraction allocated to the short book.
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

    Each :class:`InstrumentSpec` carries its own OHLCV data, entry/exit
    signals, direction, and weight.

    Sync mode controls rebalancing behaviour:

    * ``"any"`` – act whenever *any* instrument signals (most liquid).
    * ``"all"`` – act only when *all* instruments signal simultaneously.
    * ``"majority"`` – act when >50% of instruments signal.
    * ``"master"`` – the first instrument drives rebalancing.

    Args:
        instruments: List of :class:`InstrumentSpec`, one per (asset, direction)
            pair.  The same underlying asset can appear twice (once with
            ``direction=1``, once with ``direction=-1``) to model a long/short
            portfolio.
        config: raptorbt config; uses :func:`make_config` defaults if ``None``.
        sync_mode: Rebalancing synchronisation mode (default ``"any"``).

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


# ── High-level runner classes ─────────────────────────────────────────────────


def _vol_adjusted_weight(
    close: np.ndarray,
    vol_target: float,
    max_leverage: float,
) -> float:
    """Compute a static vol-adjusted allocation weight.

    Estimates the average realised daily volatility (annualised) over the
    full price series and derives the weight required to hit *vol_target*,
    capped at *max_leverage*.

    The result is halved because the same capital is split between the long
    and short book in a long/short strategy.

    Args:
        close: Close price array.
        vol_target: Target annualised volatility.
        max_leverage: Maximum allowed leverage multiplier.

    Returns:
        A positive float representing the per-book weight (≤ 0.5).
    """
    if len(close) < 2:  # noqa: PLR2004
        return vol_target / 2.0
    daily_ret = np.diff(close) / np.where(close[:-1] == 0, 1.0, close[:-1])
    avg_vol = float(np.nanstd(daily_ret)) * (252**0.5)
    raw_weight = vol_target / max(avg_vol, 1e-6)
    return min(raw_weight, max_leverage) / 2.0


class BacktestRunner:
    """High-level runner for single-asset ``Strategy`` implementations.

    Chains together signal generation, OHLCV extraction, signal conversion, and
    the raptorbt long/short runner into a single ``run()`` call.

    Args:
        initial_capital: Starting portfolio equity.
        fees: Per-trade commission fraction (default 10 bps).
        slippage: Round-trip slippage fraction (default 5 bps).
        vol_target: Target annualised portfolio volatility used to derive the
            static capital allocation weight passed to raptorbt.
        max_leverage: Hard leverage cap on the vol-adjusted weight.

    Example::

        runner = BacktestRunner(initial_capital=1_000_000.0, fees=0.001)
        result = runner.run(strategy, bars, symbol="ES.c.0")
        print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
    """

    def __init__(
        self,
        *,
        initial_capital: float = 100_000.0,
        fees: float = 0.001,
        slippage: float = 0.0005,
        vol_target: float = 0.10,
        max_leverage: float = 2.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.vol_target = vol_target
        self.max_leverage = max_leverage

    def run(
        self,
        strategy: Strategy,
        bars: pl.DataFrame,
        *,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run a single-asset backtest for the given strategy.

        Args:
            strategy: Any :class:`~snippy_scales.strategies.trend.Strategy`
                implementation.
            bars: OHLCV bar DataFrame (must include a ``close`` column).
            symbol: Instrument name used in result reporting.

        Returns:
            :class:`BacktestResult` with equity curve, metrics, and trade list.
        """
        positions = strategy.generate_signals(bars).to_numpy().astype(np.float64)
        long_entries, long_exits, short_entries, short_exits = positions_to_signals(positions)

        ohlcv = extract_ohlcv(bars)
        weight = _vol_adjusted_weight(ohlcv["close"], self.vol_target, self.max_leverage)

        cfg = make_config(
            initial_capital=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
        )
        return run_long_short(
            symbol=symbol,
            timestamps=ohlcv["timestamps"],
            open_prices=ohlcv["open"],
            high_prices=ohlcv["high"],
            low_prices=ohlcv["low"],
            close_prices=ohlcv["close"],
            volume=ohlcv["volume"],
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            long_weight=weight,
            short_weight=weight,
            config=cfg,
        )


class BasketRunner:
    """High-level runner for multi-asset ``MultiAssetStrategy`` implementations.

    Chains together multi-asset signal generation, per-asset OHLCV extraction,
    signal conversion, and the raptorbt basket runner.

    Long and short legs of the same underlying asset are submitted as separate
    instruments (suffixed ``_long`` / ``_short``) so that raptorbt correctly
    models both books within a single basket backtest.

    Args:
        initial_capital: Starting portfolio equity.
        fees: Per-trade commission fraction.
        slippage: Round-trip slippage fraction.
        sync_mode: raptorbt basket synchronisation mode (default ``"any"``).

    Example::

        runner = BasketRunner(initial_capital=1_000_000.0, fees=0.001)
        result = runner.run(cs_strategy, multi_bars)
        print(f"Max DD: {result.metrics.max_drawdown_pct:.1f}%")
    """

    def __init__(
        self,
        *,
        initial_capital: float = 100_000.0,
        fees: float = 0.001,
        slippage: float = 0.0005,
        sync_mode: str = "any",
    ) -> None:
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.sync_mode = sync_mode

    def run(
        self,
        strategy: MultiAssetStrategy,
        multi_bars: dict[str, pl.DataFrame],
    ) -> BacktestResult:
        """Run a multi-asset basket backtest for the given strategy.

        Args:
            strategy: A :class:`~snippy_scales.strategies.momentum_cs.MultiAssetStrategy`
                implementation.
            multi_bars: Mapping of ``symbol → bars DataFrame``.  All DataFrames
                must have the same number of rows (aligned in time).

        Returns:
            :class:`BacktestResult` for the combined basket portfolio.

        Raises:
            ValueError: If no signals are generated (all positions are zero).
        """
        positions_dict = strategy.generate_signals(multi_bars)
        n_assets = len(multi_bars)

        instruments: list[InstrumentSpec] = []
        for sym, pos_series in positions_dict.items():
            bars = multi_bars[sym]
            ohlcv = extract_ohlcv(bars)
            positions = pos_series.to_numpy().astype(np.float64)

            long_entries, long_exits, short_entries, short_exits = positions_to_signals(positions)

            # Equal-weight allocation across all assets (long and short books).
            weight = 1.0 / n_assets

            if long_entries.any():
                instruments.append(
                    InstrumentSpec(
                        symbol=f"{sym}_long",
                        timestamps=ohlcv["timestamps"],
                        open=ohlcv["open"],
                        high=ohlcv["high"],
                        low=ohlcv["low"],
                        close=ohlcv["close"],
                        volume=ohlcv["volume"],
                        entries=long_entries,
                        exits=long_exits,
                        direction=1,
                        weight=weight,
                    )
                )

            if short_entries.any():
                instruments.append(
                    InstrumentSpec(
                        symbol=f"{sym}_short",
                        timestamps=ohlcv["timestamps"],
                        open=ohlcv["open"],
                        high=ohlcv["high"],
                        low=ohlcv["low"],
                        close=ohlcv["close"],
                        volume=ohlcv["volume"],
                        entries=short_entries,
                        exits=short_exits,
                        direction=-1,
                        weight=weight,
                    )
                )

        if not instruments:
            raise ValueError(
                "No signals generated by strategy; cannot run basket backtest.  "
                "Ensure the input data is long enough for the strategy warmup period."
            )

        cfg = make_config(
            initial_capital=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
        )
        return run_basket(instruments=instruments, config=cfg, sync_mode=self.sync_mode)
