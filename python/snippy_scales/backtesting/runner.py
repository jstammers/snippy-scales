"""Backtesting infrastructure — public re-export surface.

This module is the canonical import path for the backtesting layer.  All
symbols are implemented in focused submodules and re-exported here so that
existing code that imports from ``snippy_scales.backtesting.runner`` continues
to work unchanged.

Submodule layout:

* :mod:`.domain`     — ``TRADING_DAYS_PER_YEAR``, ``Trade``, ``BacktestMetrics``,
  ``BacktestResult``
* :mod:`.data`       — ``OhlcvArrays``, ``normalize_ohlcv``, ``to_numpy_ohlcv``,
  ``extract_ohlcv``
* :mod:`.signals`    — ``SignalBundle``, ``PositionInterpreter``,
  ``SignFlipInterpreter``, ``positions_to_signals``
* :mod:`.allocation` — ``VolTargetAllocator``
* :mod:`.engine`     — ``ExecutionEngine``, ``RaptorExecutionEngine``,
  ``InstrumentSpec``, ``make_config``, ``run_single``, ``run_long_short``,
  ``run_basket``
* :mod:`.runners`    — ``BacktestRunner``, ``BasketRunner``
"""

from __future__ import annotations

from snippy_scales.backtesting.allocation import VolTargetAllocator
from snippy_scales.backtesting.data import (
    OhlcvArrays,
    extract_ohlcv,
    normalize_ohlcv,
    to_numpy_ohlcv,
)
from snippy_scales.backtesting.domain import (
    TRADING_DAYS_PER_YEAR,
    BacktestMetrics,
    BacktestResult,
    Trade,
)
from snippy_scales.backtesting.engine import (
    ExecutionEngine,
    InstrumentSpec,
    RaptorExecutionEngine,
    make_config,
    run_basket,
    run_long_short,
    run_single,
)
from snippy_scales.backtesting.runners import BacktestRunner, BasketRunner
from snippy_scales.backtesting.signals import (
    PositionInterpreter,
    SignalBundle,
    SignFlipInterpreter,
    positions_to_signals,
)

__all__ = [
    # domain
    "TRADING_DAYS_PER_YEAR",
    "Trade",
    "BacktestMetrics",
    "BacktestResult",
    # data
    "OhlcvArrays",
    "normalize_ohlcv",
    "to_numpy_ohlcv",
    "extract_ohlcv",
    # signals
    "SignalBundle",
    "PositionInterpreter",
    "SignFlipInterpreter",
    "positions_to_signals",
    # allocation
    "VolTargetAllocator",
    # engine
    "ExecutionEngine",
    "RaptorExecutionEngine",
    "InstrumentSpec",
    "make_config",
    "run_single",
    "run_long_short",
    "run_basket",
    # runners
    "BacktestRunner",
    "BasketRunner",
]
