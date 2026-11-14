"""Execution on NautilusTrader — the only package that imports ``nautilus_trader``
for trading (``data/`` uses it for storage).

* :class:`SignalStrategy` trades the target positions of a research strategy.
* :func:`load_market_data` / :meth:`MarketData.from_frames` prepare bars and instruments.
* :func:`run_backtest` backtests a research strategy on them over a time window.
* :func:`vector.run_backtest` is the same call on a numba kernel, for screening
  (:data:`ENGINES` maps ``"nautilus"`` / ``"vector"`` to the two).
"""

from snippy_scales.execution import vector  # noqa: E402, I001 — needs backtest/strategy first
from snippy_scales.execution.backtest import (
    SHARPE,
    BacktestRun,
    Costs,
    MarketData,
    load_market_data,
    run_backtest,
)
from snippy_scales.execution.strategy import SignalStrategy, compute_targets

#: Backtest engines by name; both share one signature and return a BacktestRun.
ENGINES = {"nautilus": run_backtest, "vector": vector.run_backtest}

__all__ = [
    "ENGINES",
    "SHARPE",
    "BacktestRun",
    "Costs",
    "MarketData",
    "SignalStrategy",
    "compute_targets",
    "load_market_data",
    "run_backtest",
    "vector",
]
