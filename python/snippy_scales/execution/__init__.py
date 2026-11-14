"""Execution on NautilusTrader — the only package that imports ``nautilus_trader``
for trading (``data/`` uses it for storage).

* :class:`SignalStrategy` trades the target positions of a research strategy.
* :func:`panel_targets` computes a strategy's targets for a whole universe as
  arrays (:class:`Targets`) from a stacked :class:`Panel` of bars.
* :func:`load_market_data` / :meth:`MarketData.from_frames` prepare bars and instruments.
* :func:`run_backtest` backtests a research strategy on them over a time window.
* :func:`vector.run_backtest` is the same call on a numba kernel, for screening
  (:data:`ENGINES` maps ``"nautilus"`` / ``"vector"`` to the two).
* :class:`pool.BacktestPool` runs Nautilus backtests in worker processes.
"""

from snippy_scales.execution import pool, vector  # noqa: E402, I001 — needs backtest/strategy first
from snippy_scales.execution.backtest import (
    SHARPE,
    BacktestRun,
    Costs,
    MarketData,
    load_market_data,
    run_backtest,
)
from snippy_scales.execution.strategy import (
    Panel,
    SignalStrategy,
    Targets,
    compute_targets,
    panel_targets,
    stack_frames,
    target_frame,
)

#: Backtest engines by name; both share one signature and return a BacktestRun.
ENGINES = {"nautilus": run_backtest, "vector": vector.run_backtest}

__all__ = [
    "ENGINES",
    "SHARPE",
    "BacktestRun",
    "Costs",
    "MarketData",
    "Panel",
    "SignalStrategy",
    "Targets",
    "compute_targets",
    "load_market_data",
    "panel_targets",
    "pool",
    "run_backtest",
    "stack_frames",
    "target_frame",
    "vector",
]
