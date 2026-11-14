"""Tests for ``execution/``: SignalStrategy sizing and timing, MarketData, run_backtest.

The one-time parity check against the deleted ``TargetPositionEngine`` is
recorded in docs/architecture/nautilus_migration.md (phase 2).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

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
from snippy_scales.data.futures import RollSchedule
from snippy_scales.execution import (
    BacktestRun,
    Costs,
    MarketData,
    compute_targets,
    load_market_data,
    run_backtest,
)
from snippy_scales.strategies import (
    CrossSectionalMomentum,
    MultiAssetStrategy,
    Strategy,
    TrendFollowing,
)

_DAY = 86_400_000_000_000
_BASE = 1_577_836_800_000_000_000  # 2020-01-01 UTC


def _future(symbol: str, venue: str = "SIM", multiplier: int = 1) -> FuturesContract:
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


def _frame(n: int, seed: int, drift: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.round(100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), 2)
    ts_open = _BASE + np.arange(n, dtype=np.int64) * _DAY
    return pl.DataFrame(
        {
            "ts_event": ts_open,
            "ts_init": ts_open + _DAY,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1e12),  # ample liquidity: no book-walking on fills
        }
    ).with_columns(pl.col("ts_event", "ts_init").cast(pl.Datetime("ns", "UTC")))


class _Fixed(Strategy):
    """Replays a precomputed target array (for exact sizing assertions)."""

    def __init__(self, targets: np.ndarray) -> None:
        self.targets = targets

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        return pl.Series("position", self.targets[: bars.height])


def run_backtest_on_frames(
    research: Strategy | MultiAssetStrategy,
    frames: dict[str, pl.DataFrame],
    instruments: dict[str, FuturesContract],
    **kwargs: Any,
) -> BacktestRun:
    return run_backtest(research, MarketData.from_frames(frames, instruments), **kwargs)


# ── Sizing and timing ─────────────────────────────────────────────────────────


def test_position_follows_target_with_fill_at_close() -> None:
    frame = _frame(40, seed=1)
    targets = np.zeros(40)
    targets[5:20] = 0.5
    targets[20:30] = -1.0
    run = run_backtest_on_frames(
        _Fixed(targets), {"X": frame}, {"X": _future("X")}, initial_capital=1_000_000.0
    )
    assert run.rejections == ()
    close = frame["close"].to_numpy()
    equity = run.equity["equity"].to_numpy()
    # Flat until the first target fills at bar 5's close; first P&L lands on bar 6.
    assert np.all(equity[:6] == 1_000_000.0)
    contracts = round(0.5 * 1_000_000.0 / close[5])
    assert equity[6] == pytest.approx(1_000_000.0 + contracts * (close[6] - close[5]), abs=0.01)
    # One order to open, one per bar to rebalance while the target is held, ...
    assert run.total_orders > 3


def test_rebalance_band_cuts_orders() -> None:
    frame = _frame(120, seed=2)
    research = TrendFollowing(fast_period=5, slow_period=20)
    every_bar = run_backtest_on_frames(research, {"X": frame}, {"X": _future("X")})
    banded = run_backtest_on_frames(
        research, {"X": frame}, {"X": _future("X")}, rebalance_band=0.05
    )
    assert banded.total_orders < every_bar.total_orders / 2


def test_multiplier_scales_contracts() -> None:
    frame = _frame(30, seed=3)
    targets = np.full(30, 1.0)
    run = run_backtest_on_frames(_Fixed(targets), {"X": frame}, {"X": _future("X", multiplier=50)})
    first = run.fills.sort_values("ts_event").iloc[0]
    assert float(first["last_qty"]) == round(1_000_000.0 / (frame["close"][0] * 50))


def test_start_trades_only_after_warmup() -> None:
    frame = _frame(100, seed=4)
    start = dt.datetime(2020, 2, 15, tzinfo=dt.UTC)
    run = run_backtest_on_frames(
        TrendFollowing(fast_period=5, slow_period=20),
        {"X": frame},
        {"X": _future("X")},
        start=start,
    )
    assert run.equity.filter(pl.col("ts") < start).height == 0
    # Signals were warm on the first replayed bar, so it traded immediately.
    assert run.fills["ts_event"].min() == run.equity["ts"].min()


def test_commission_rate_charges_notional() -> None:
    frame = _frame(30, seed=5)
    targets = np.full(30, 1.0)
    run = run_backtest_on_frames(
        _Fixed(targets),
        {"X": frame},
        {"X": _future("X")},
        costs=Costs(commission_rate=0.001),
    )
    first = run.fills.sort_values("ts_event").iloc[0]
    notional = float(first["last_qty"]) * float(first["last_px"])
    assert float(str(first["commission"]).split()[0]) == pytest.approx(0.001 * notional, abs=0.01)


def test_costs_validation() -> None:
    with pytest.raises(ValueError, match="not both"):
        Costs(commission_per_contract=1.0, commission_rate=0.001)
    with pytest.raises(ValueError, match="slippage_prob"):
        Costs(slippage_prob=2.0)


def test_mismatched_venues_rejected() -> None:
    frame = _frame(10, seed=6)
    with pytest.raises(ValueError, match="one venue"):
        run_backtest_on_frames(
            _Fixed(np.zeros(10)),
            {"X": frame, "Y": frame},
            {"X": _future("X"), "Y": _future("Y", venue="OTHER")},
        )


def test_frames_must_be_stamped_at_close() -> None:
    frame = _frame(10, seed=7).with_columns(pl.col("ts_event").alias("ts_init"))
    with pytest.raises(ValueError, match="ts_init must equal"):
        run_backtest_on_frames(_Fixed(np.zeros(10)), {"X": frame}, {"X": _future("X")})


def test_run_backtest_reads_catalog(tmp_path: Path) -> None:
    frame = _frame(80, seed=8)
    catalog = cat.open_catalog(tmp_path)
    instrument = _future("X")
    catalog.write_instruments([instrument])
    bt = cat.bar_type("X", "SIM", "ohlcv-1d")
    cat.write_bars(
        catalog,
        cat.frame_to_bars(frame, bar_type=bt, price_precision=2, interval_ns=_DAY),
    )
    research = TrendFollowing(fast_period=5, slow_period=20)
    from_catalog = run_backtest(research, load_market_data("X", catalog_path=tmp_path))
    from_frames = run_backtest_on_frames(research, {"X": frame}, {"X": instrument})
    np.testing.assert_allclose(
        from_catalog.equity["equity"].to_numpy(), from_frames.equity["equity"].to_numpy()
    )
    assert "Sharpe Ratio (252 days)" in from_catalog.stats
    assert from_catalog.sharpe == from_catalog.stats["Sharpe Ratio (252 days)"]
    assert from_catalog.returns.height > 0


def test_load_market_data_batches_many_symbols(tmp_path: Path) -> None:
    """One query for all symbols returns what loading each symbol alone does."""
    catalog = cat.open_catalog(tmp_path)
    symbols = ["X", "Y", "Z"]
    catalog.write_instruments([_future(s) for s in symbols])
    for i, s in enumerate(symbols):
        bt = cat.bar_type(s, "SIM", "ohlcv-1d")
        frame = _frame(40 + 10 * i, seed=20 + i)
        cat.write_bars(
            catalog, cat.frame_to_bars(frame, bar_type=bt, price_precision=2, interval_ns=_DAY)
        )
    together = load_market_data(symbols, catalog_path=tmp_path)
    assert together.symbols == symbols
    for s in symbols:
        alone = load_market_data(s, catalog_path=tmp_path)
        iid = f"{s}.SIM"
        assert together.frames[s].equals(alone.frames[s])
        assert together.frames[s].equals(cat.load_bars(s, "ohlcv-1d", catalog_path=tmp_path))
        assert [str(b) for b in together.bars[iid]] == [str(b) for b in alone.bars[iid]]
        assert together.instruments[iid] == alone.instruments[iid]
        assert together.bar_types[iid] == alone.bar_types[iid]


def test_run_backtest_missing_instrument(tmp_path: Path) -> None:
    catalog = cat.open_catalog(tmp_path)
    bt = cat.bar_type("X", "SIM", "ohlcv-1d")
    cat.write_bars(
        catalog,
        cat.frame_to_bars(_frame(10, seed=9), bar_type=bt, price_precision=2, interval_ns=_DAY),
    )
    with pytest.raises(FileNotFoundError, match="instrument definition"):
        load_market_data("X", catalog_path=tmp_path)


def test_live_bar_beyond_history_recomputes() -> None:
    """Bars past the precomputed history still get targets (the live path)."""
    from snippy_scales.execution.strategy import SignalStrategy

    frame = _frame(60, seed=10)
    instrument = _future("X")
    strategy = SignalStrategy(
        TrendFollowing(fast_period=5, slow_period=20),
        history={"X": frame.head(40)},
        instruments={"X.SIM": instrument},
        bar_types={"X.SIM": cat.bar_type("X", "SIM", "ohlcv-1d")},
        schedules={"X": RollSchedule.single("X", "X.SIM")},
    )
    full = compute_targets(TrendFollowing(fast_period=5, slow_period=20), {"X": frame})
    bar = cat.frame_to_bars(
        frame.slice(40, 1),
        bar_type=cat.bar_type("X", "SIM", "ohlcv-1d"),
        price_precision=2,
        interval_ns=_DAY,
    )[0]
    strategy._extend_history({"X": bar})  # keyed by symbol
    assert strategy._targets["X"][bar.ts_init] == pytest.approx(full["X"][bar.ts_init])


def test_unaligned_multi_asset_bars_do_not_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """A close only some symbols trade at is skipped, not treated as a live bar."""
    from snippy_scales.execution import strategy as strategy_module

    frames = {f"A{i}": _frame(120, seed=200 + i) for i in range(3)}
    frames["A0"] = frames["A0"].filter(pl.int_range(pl.len()) % 7 != 3)  # A0 misses some days
    calls = 0
    original = strategy_module.compute_targets

    def counting(*args: object, **kwargs: object) -> dict[str, dict[int, float]]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(strategy_module, "compute_targets", counting)
    run = run_backtest_on_frames(
        CrossSectionalMomentum(lookback=20, skip_recent=2, vol_lookback=10, rebal_freq=5),
        frames,
        {s: _future(s) for s in frames},
    )
    assert calls == 1
    assert run.rejections == ()
    assert run.total_orders > 0


def test_window_slices_bars_and_frames() -> None:
    frame = _frame(50, seed=11)
    data = MarketData.from_frames({"X": frame}, {"X": _future("X")})
    start = dt.datetime(2020, 1, 11, tzinfo=dt.UTC)
    end = dt.datetime(2020, 1, 31, tzinfo=dt.UTC)
    sub = data.window(start, end)
    ts = [b.ts_init for b in sub.bars["X.SIM"]]
    lo, hi = cat.to_ns(start), cat.to_ns(end)
    assert lo is not None and hi is not None
    assert min(ts) >= lo and max(ts) <= hi
    assert len(ts) == 21
    assert sub.frames["X"]["ts_init"].max() == end
    assert sub.frames["X"].height == 30  # history from the first bar, for warm-up
    assert data.closes().len() == 50
