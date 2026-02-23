"""High-level backtesting runners.

Provides two runner classes that chain strategy signal generation, OHLCV
extraction, signal conversion, and engine execution into a single ``run()``
call:

* :class:`BacktestRunner` — single-asset strategies
  (:class:`~snippy_scales.strategies.trend.Strategy`).
* :class:`BasketRunner` — multi-asset strategies
  (:class:`~snippy_scales.strategies.momentum_cs.MultiAssetStrategy`).

Both runners accept an optional :class:`~snippy_scales.backtesting.engine.ExecutionEngine`
so that the execution backend can be swapped (e.g. for testing or paper trading)
without modifying strategy code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from snippy_scales.backtesting.allocation import VolTargetAllocator
from snippy_scales.backtesting.data import extract_ohlcv
from snippy_scales.backtesting.engine import (
    ExecutionEngine,
    InstrumentSpec,
    RaptorExecutionEngine,
    make_config,
    run_long_short,
)
from snippy_scales.backtesting.signals import SignFlipInterpreter

if TYPE_CHECKING:
    import polars as pl

    from snippy_scales.backtesting.domain import BacktestResult
    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

_DEFAULT_ENGINE = RaptorExecutionEngine()
_DEFAULT_INTERPRETER = SignFlipInterpreter()


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
        engine: Execution engine to use (default :class:`RaptorExecutionEngine`).

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
        engine: ExecutionEngine | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self._allocator = VolTargetAllocator(
            vol_target=vol_target,
            max_leverage=max_leverage,
            long_short=True,
        )
        self._engine: ExecutionEngine = engine if engine is not None else _DEFAULT_ENGINE

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
        bundle = _DEFAULT_INTERPRETER.interpret(positions)
        ohlcv = extract_ohlcv(bars)
        weight = self._allocator.weight(ohlcv.close)

        cfg = make_config(
            initial_capital=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
        )
        return run_long_short(
            symbol=symbol,
            timestamps=ohlcv.timestamps,
            open_prices=ohlcv.open,
            high_prices=ohlcv.high,
            low_prices=ohlcv.low,
            close_prices=ohlcv.close,
            volume=ohlcv.volume,
            long_entries=bundle.long_entries,
            long_exits=bundle.long_exits,
            short_entries=bundle.short_entries,
            short_exits=bundle.short_exits,
            long_weight=weight,
            short_weight=weight,
            config=cfg,
        )


class BasketRunner:
    """High-level runner for multi-asset ``MultiAssetStrategy`` implementations.

    Chains together multi-asset signal generation, per-asset OHLCV extraction,
    signal conversion, and the execution engine's basket runner.

    Long and short legs of the same underlying asset are submitted as separate
    instruments (suffixed ``_long`` / ``_short``) so that the engine correctly
    models both books.

    Args:
        initial_capital: Starting portfolio equity.
        fees: Per-trade commission fraction.
        slippage: Round-trip slippage fraction.
        sync_mode: raptorbt basket synchronisation mode (default ``"any"``).
        engine: Execution engine to use (default :class:`RaptorExecutionEngine`).

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
        engine: ExecutionEngine | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.sync_mode = sync_mode
        self._engine: ExecutionEngine = engine if engine is not None else _DEFAULT_ENGINE

    def run(
        self,
        strategy: MultiAssetStrategy,
        multi_bars: dict[str, pl.DataFrame],
    ) -> BacktestResult:
        """Run a multi-asset basket backtest for the given strategy.

        Args:
            strategy: A :class:`~snippy_scales.strategies.momentum_cs.MultiAssetStrategy`
                implementation.
            multi_bars: Mapping of ``symbol → bars DataFrame``.

        Returns:
            :class:`BacktestResult` for the combined basket portfolio.

        Raises:
            ValueError: If no signals are generated (all positions are zero).
        """
        positions_dict = strategy.generate_signals(multi_bars)
        n_assets = len(multi_bars)
        weight = 1.0 / n_assets

        instruments: list[InstrumentSpec] = []
        for sym, pos_series in positions_dict.items():
            ohlcv = extract_ohlcv(multi_bars[sym])
            positions = pos_series.to_numpy().astype(np.float64)
            bundle = _DEFAULT_INTERPRETER.interpret(positions)

            if bundle.long_entries.any():
                instruments.append(
                    InstrumentSpec(
                        symbol=f"{sym}_long",
                        timestamps=ohlcv.timestamps,
                        open=ohlcv.open,
                        high=ohlcv.high,
                        low=ohlcv.low,
                        close=ohlcv.close,
                        volume=ohlcv.volume,
                        entries=bundle.long_entries,
                        exits=bundle.long_exits,
                        direction=1,
                        weight=weight,
                    )
                )

            if bundle.short_entries.any():
                instruments.append(
                    InstrumentSpec(
                        symbol=f"{sym}_short",
                        timestamps=ohlcv.timestamps,
                        open=ohlcv.open,
                        high=ohlcv.high,
                        low=ohlcv.low,
                        close=ohlcv.close,
                        volume=ohlcv.volume,
                        entries=bundle.short_entries,
                        exits=bundle.short_exits,
                        direction=-1,
                        weight=weight,
                    )
                )

        if not instruments:
            raise ValueError(
                "No signals generated by strategy; cannot run basket backtest.  "
                "Ensure the input data is long enough for the strategy warmup period."
            )

        cfg: Any = make_config(
            initial_capital=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
        )
        return self._engine.execute(instruments, config=cfg, sync_mode=self.sync_mode)
