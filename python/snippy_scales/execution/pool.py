"""Run backtests in worker processes, each holding its own copy of the market data.

A Nautilus ``BacktestEngine`` run is single-threaded Python/Cython: on a large
universe one walk-forward fold takes seconds, and the folds of a trial (and the
trials of a validation stage) are independent. :class:`BacktestPool` spreads
them over processes::

    with BacktestPool(data, workers=6) as pool:
        futures = [pool.submit(strategy, start=f.test_start, end=f.test_end) for f in folds]
        runs = [f.result() for f in futures]

Nautilus instruments cannot be pickled, so the data is not copied to the
workers: each one reloads it from the catalog once, at start-up, with the
recipe :func:`~snippy_scales.execution.backtest.load_market_data` stored on it
(:attr:`MarketData.source`), and checks it got the same bars. Results are those
of :func:`~snippy_scales.execution.backtest.run_backtest` in this process —
every run builds a fresh engine from the same inputs.

Workers are started with ``spawn`` (Nautilus runs Rust threads, which ``fork``
does not copy safely). Each holds the whole universe: about 3 GB for 800 daily
S&P 500 series, so size *workers* to memory as well as cores.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import threading
import weakref
from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

from snippy_scales.execution.backtest import BacktestRun, MarketData, run_backtest

if TYPE_CHECKING:
    from types import TracebackType

    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

_DATA: MarketData | None = None


def _init(data: MarketData) -> None:
    # Unpickling *data* reloads it from the catalog (MarketData.__reduce__).
    global _DATA  # noqa: PLW0603 — one dataset per worker process
    _DATA = data


def _run(research: Strategy | MultiAssetStrategy, kwargs: dict[str, Any]) -> BacktestRun:
    assert _DATA is not None  # noqa: S101 — set by the pool initializer
    return run_backtest(research, _DATA, **kwargs)


def default_workers(tasks: int | None = None) -> int:
    """Half the CPUs (each worker holds a copy of the data), at most *tasks*."""
    n = max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(n, tasks)) if tasks is not None else n


class BacktestPool:
    """Processes running :func:`run_backtest` on one dataset.

    Args:
        data: Market data loaded with :func:`load_market_data` (or derived from
            such data with ``select``/``window``/``merge``).
        workers: Number of processes.

    Raises:
        TypeError: If *data* was built from frames and so cannot be reloaded
            in a worker.
    """

    def __init__(self, data: MarketData, workers: int) -> None:
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if data.source is None:
            raise TypeError(
                "BacktestPool needs data from load_market_data (workers reload it from "
                "the catalog); data built from frames has no source"
            )
        self.workers = workers
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init,
            initargs=(data,),
        )

    def submit(self, research: Strategy | MultiAssetStrategy, **kwargs: Any) -> Future[BacktestRun]:  # noqa: ANN401
        """Queue ``run_backtest(research, data, **kwargs)``."""
        return self._executor.submit(_run, research, kwargs)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> BacktestPool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown()


#: Pools kept alive between calls, by ``(id(data), workers)``: starting one
#: reloads the data in every worker, so trials of a study share it.
_POOLS: dict[tuple[int, int], BacktestPool] = {}
_LOCK = threading.Lock()


def shared_pool(data: MarketData, workers: int) -> BacktestPool:
    """A :class:`BacktestPool` for *data*, reused until *data* is garbage collected."""
    key = (id(data), workers)
    with _LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = _POOLS[key] = BacktestPool(data, workers)
            weakref.finalize(data, _close, key)
    return pool


def _close(key: tuple[int, int]) -> None:
    pool = _POOLS.pop(key, None)
    if pool is not None:
        pool.shutdown()


def close_pools() -> None:
    """Shut down every shared pool."""
    for key in list(_POOLS):
        _close(key)


atexit.register(close_pools)
