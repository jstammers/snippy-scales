"""Execution on NautilusTrader — the only package that imports ``nautilus_trader``
for trading (``data/`` uses it for storage).

* :class:`SignalStrategy` trades the target positions of a research strategy.
* :func:`load_market_data` / :meth:`MarketData.from_frames` prepare bars and instruments.
* :func:`run_backtest` backtests a research strategy on them over a time window.
"""

from snippy_scales.execution.backtest import (
    SHARPE,
    BacktestRun,
    Costs,
    MarketData,
    load_market_data,
    run_backtest,
)
from snippy_scales.execution.strategy import SignalStrategy, compute_targets

__all__ = [
    "SHARPE",
    "BacktestRun",
    "Costs",
    "MarketData",
    "SignalStrategy",
    "compute_targets",
    "load_market_data",
    "run_backtest",
]
