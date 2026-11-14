"""Nautilus backtests in worker processes match backtests in this process."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

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

from snippy_scales.data import catalog as cat
from snippy_scales.evaluation import WalkForward, make_folds
from snippy_scales.execution import Costs, MarketData, load_market_data, pool
from snippy_scales.strategies import CrossSectionalMomentum, TrendFollowing

if TYPE_CHECKING:
    from pathlib import Path

    from snippy_scales.strategies import MultiAssetStrategy, Strategy

_DAY = 86_400_000_000_000
_BASE = 1_577_836_800_000_000_000  # 2020-01-01 UTC
_SYMBOLS = ["X", "Y", "Z"]


def _future(symbol: str) -> FuturesContract:
    return FuturesContract(
        InstrumentId.from_str(f"{symbol}.SIM"),
        Symbol(symbol),
        AssetClass.INDEX,
        symbol,
        0,
        2**63 - 1,
        Currency.from_str("USD"),
        2,
        Price.from_str("0.01"),
        Quantity.from_int(1),
        Quantity.from_int(1),
        0,
        0,
    )


def _frame(n: int, seed: int, offset: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.round(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), 2)
    ts_open = _BASE + (offset + np.arange(n, dtype=np.int64)) * _DAY
    return pl.DataFrame(
        {
            "ts_event": ts_open,
            "ts_init": ts_open + _DAY,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1e12),
        }
    ).with_columns(pl.col("ts_event", "ts_init").cast(pl.Datetime("ns", "UTC")))


def _write(catalog_path: Path, symbol: str, frame: pl.DataFrame) -> None:
    bt = cat.bar_type(symbol, "SIM", "ohlcv-1d")
    bars = cat.frame_to_bars(frame, bar_type=bt, price_precision=2, interval_ns=_DAY)
    cat.write_bars(cat.open_catalog(catalog_path), bars)


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    cat.open_catalog(tmp_path).write_instruments([_future(s) for s in _SYMBOLS])
    for i, s in enumerate(_SYMBOLS):
        _write(tmp_path, s, _frame(300 - 40 * i, seed=30 + i, offset=20 * i))
    return tmp_path


def _same(a: MarketData, b: MarketData) -> None:
    assert a.symbols == b.symbols
    assert a.signature() == b.signature()
    for s in a.symbols:
        assert a.frames[s].equals(b.frames[s])
    for iid in a.bars:
        assert [str(x) for x in a.bars[iid]] == [str(x) for x in b.bars[iid]]


def test_pickle_reloads_from_catalog(catalog_path: Path) -> None:
    data = load_market_data(_SYMBOLS, catalog_path=catalog_path)
    _same(pickle.loads(pickle.dumps(data)), data)  # noqa: S301 — our own bytes
    part = data.select(["Y", "Z"]).window("2020-03-01", "2020-08-01")
    _same(pickle.loads(pickle.dumps(part)), part)  # noqa: S301


def test_pickle_detects_a_changed_catalog(catalog_path: Path) -> None:
    data = load_market_data(_SYMBOLS, catalog_path=catalog_path)
    raw = pickle.dumps(data)
    _write(catalog_path, "X", _frame(5, seed=1, offset=400))  # bars after the existing ones
    with pytest.raises(RuntimeError, match="catalog changed"):
        pickle.loads(raw)  # noqa: S301


def test_frames_data_cannot_go_to_a_pool(catalog_path: Path) -> None:
    data = load_market_data(_SYMBOLS, catalog_path=catalog_path)
    frames_only = MarketData.from_frames({"X": data.frames["X"]}, {"X": data.instruments["X.SIM"]})
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(frames_only)
    with pytest.raises(TypeError, match="load_market_data"):
        pool.BacktestPool(frames_only, workers=2)


@pytest.mark.parametrize(
    "research",
    [
        TrendFollowing(fast_period=5, slow_period=20),
        CrossSectionalMomentum(lookback=40, skip_recent=5, vol_lookback=20, rebal_freq=5),
    ],
    ids=["trend", "csmom"],
)
def test_parallel_folds_match_in_process(
    catalog_path: Path, research: Strategy | MultiAssetStrategy
) -> None:
    data = load_market_data(_SYMBOLS, catalog_path=catalog_path)
    folds = make_folds(data.closes(union=True), n_splits=3, min_train=60)
    costs = Costs(commission_rate=5e-4)
    serial = WalkForward(data, folds, costs=costs).run_folds(research)
    try:
        parallel = WalkForward(data, folds, costs=costs, workers=2).run_folds(research)
    finally:
        pool.close_pools()
    assert len(parallel) == len(serial) == 3
    for a, b in zip(serial, parallel, strict=True):
        assert a.total_orders == b.total_orders > 0
        assert a.rejections == b.rejections
        assert a.stats_returns == pytest.approx(b.stats_returns, nan_ok=True, abs=0)
        assert a.equity.equals(b.equity)
        assert a.returns.equals(b.returns)
