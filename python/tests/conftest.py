"""Shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
from nautilus_trader.model import (
    AssetClass,
    Currency,
    FuturesContract,
    InstrumentId,
    Price,
    Quantity,
    Symbol,
)

from snippy_scales.execution import BacktestRun, MarketData, run_backtest

if TYPE_CHECKING:
    from collections.abc import Callable

    from snippy_scales.strategies import MultiAssetStrategy, Strategy

_DAY = 86_400_000_000_000
_BASE = 1_577_836_800_000_000_000  # 2020-01-01 UTC


def synthetic_future(symbol: str, venue: str = "SIM", multiplier: int = 1) -> FuturesContract:
    """A USD futures contract with tick 0.01 and lot size 1."""
    return FuturesContract(
        InstrumentId.from_str(f"{symbol}.{venue}"),
        Symbol(symbol),
        AssetClass.INDEX,
        symbol,
        0,
        2**63 - 1,
        Currency.from_str("USD"),
        2,
        Price.from_str("0.01"),
        Quantity.from_int(multiplier),
        Quantity.from_int(1),
        0,
        0,
    )


def daily_bars(close: np.ndarray) -> pl.DataFrame:
    """Daily bars from 2020-01-01 with O=H=L=C=*close* (rounded to the tick)."""
    close = np.round(np.asarray(close, dtype=np.float64), 2)
    ts_open = _BASE + np.arange(close.size, dtype=np.int64) * _DAY
    return pl.DataFrame(
        {
            "ts_event": ts_open,
            "ts_init": ts_open + _DAY,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(close.size, 1e12),
        }
    ).with_columns(pl.col("ts_event", "ts_init").cast(pl.Datetime("ns", "UTC")))


def synthetic_data(closes: dict[str, pl.DataFrame | np.ndarray]) -> MarketData:
    """MarketData for ``symbol -> close prices`` (a frame with ``close`` or an array)."""
    frames = {
        s: daily_bars(c["close"].to_numpy() if isinstance(c, pl.DataFrame) else c)
        for s, c in closes.items()
    }
    return MarketData.from_frames(frames, {s: synthetic_future(s) for s in frames})


@pytest.fixture
def backtest() -> Callable[..., BacktestRun]:
    """``backtest(research, {symbol: closes}, **run_backtest_kwargs)`` on synthetic futures."""

    def _run(
        research: Strategy | MultiAssetStrategy,
        closes: dict[str, pl.DataFrame | np.ndarray],
        **kwargs: Any,
    ) -> BacktestRun:
        return run_backtest(research, synthetic_data(closes), **kwargs)

    return _run
