"""Unit tests for the backtesting runner infrastructure.

Tests cover:
* positions_to_signals() conversion correctness
* extract_ohlcv() column synthesis and passthrough
* BacktestMetrics and BacktestResult construction (via raptorbt integration)
* run_single(), run_long_short(), run_basket() happy-path execution
* BacktestRunner.run() end-to-end with a trivial strategy
* BasketRunner.run() end-to-end with a trivial multi-asset strategy
* Edge cases: empty signals, single-bar series, all-zeros positions
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from snippy_scales.backtesting.runner import (
    BacktestMetrics,
    BacktestResult,
    BacktestRunner,
    BasketRunner,
    InstrumentSpec,
    OhlcvArrays,
    extract_ohlcv,
    make_config,
    positions_to_signals,
    run_basket,
    run_long_short,
    run_single,
)
from snippy_scales.strategies.mean_reversion import MeanReversion
from snippy_scales.strategies.momentum_cs import CrossSectionalMomentum
from snippy_scales.strategies.momentum_ts import TimeSeriesMomentum
from snippy_scales.strategies.trend import TrendFollowing

# ── Fixtures ──────────────────────────────────────────────────────────────────

_BASE_NS = 1_577_836_800_000_000_000  # 2020-01-01 UTC in nanoseconds
_DAY_NS = 86_400_000_000_000


def _make_bars(n: int = 500, seed: int = 0, trend: float = 0.0) -> pl.DataFrame:
    """Generate synthetic OHLCV bar data.

    Args:
        n: Number of bars.
        seed: NumPy random seed.
        trend: Per-bar drift added to returns (positive = upward trend).
    """
    rng = np.random.default_rng(seed)
    close = 1000.0 * np.cumprod(1.0 + rng.normal(trend, 0.01, n))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = rng.uniform(1_000, 10_000, n)
    timestamps = _BASE_NS + np.arange(n, dtype=np.int64) * _DAY_NS
    return pl.DataFrame(
        {
            "ts": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_ohlcv_arrays(n: int = 300, seed: int = 42) -> OhlcvArrays:
    """Build aligned NumPy OHLCV arrays ready for raptorbt."""
    bars = _make_bars(n, seed=seed)
    return extract_ohlcv(bars)


# ── positions_to_signals ──────────────────────────────────────────────────────


def test_positions_to_signals_shape() -> None:
    pos = np.array([0.0, 1.0, 1.0, 0.0, -1.0, 0.0])
    le, lx, se, sx = positions_to_signals(pos)
    assert le.shape == pos.shape
    assert lx.shape == pos.shape
    assert se.shape == pos.shape
    assert sx.shape == pos.shape


def test_positions_to_signals_correct_transitions() -> None:
    pos = np.array([0.0, 0.8, 0.9, 0.0, -0.7, -0.5, 0.0])
    le, lx, se, sx = positions_to_signals(pos)

    assert le.tolist() == [False, True, False, False, False, False, False]
    assert lx.tolist() == [False, False, False, True, False, False, False]
    assert se.tolist() == [False, False, False, False, True, False, False]
    assert sx.tolist() == [False, False, False, False, False, False, True]


def test_positions_to_signals_long_to_short_direct() -> None:
    """Test transition directly from long to short (no flat bar in between)."""
    pos = np.array([0.0, 1.0, -1.0, 0.0])
    le, lx, se, sx = positions_to_signals(pos)
    # At index 2: was long (exit long), now short (entry short) – both triggered.
    assert le[1] and lx[2] and se[2] and sx[3]


def test_positions_to_signals_all_zeros() -> None:
    pos = np.zeros(10)
    le, lx, se, sx = positions_to_signals(pos)
    assert not le.any()
    assert not lx.any()
    assert not se.any()
    assert not sx.any()


# ── extract_ohlcv ─────────────────────────────────────────────────────────────


def test_extract_ohlcv_with_full_columns() -> None:
    bars = _make_bars(50)
    ohlcv = extract_ohlcv(bars)
    n = 50
    assert ohlcv.close.shape == (n,)
    assert ohlcv.open.shape == (n,)
    assert ohlcv.high.shape == (n,)
    assert ohlcv.low.shape == (n,)
    assert ohlcv.volume.shape == (n,)
    assert ohlcv.timestamps.shape == (n,)
    assert ohlcv.timestamps.dtype == np.int64


def test_extract_ohlcv_synthesises_missing_columns() -> None:
    bars = pl.DataFrame({"close": [100.0, 101.0, 99.0, 102.0]})
    ohlcv = extract_ohlcv(bars)
    assert ohlcv.open.shape == (4,)
    assert ohlcv.high.shape == (4,)
    assert ohlcv.low.shape == (4,)
    assert ohlcv.volume.shape == (4,)
    # Synthesised high must be ≥ close; low must be ≤ close.
    assert (ohlcv.high >= ohlcv.close).all()
    assert (ohlcv.low <= ohlcv.close).all()


def test_extract_ohlcv_timestamps_monotone_when_synthesised() -> None:
    bars = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
    ohlcv = extract_ohlcv(bars)
    ts = ohlcv.timestamps
    assert (np.diff(ts) > 0).all()


# ── make_config ───────────────────────────────────────────────────────────────


def test_make_config_returns_raptorbt_config() -> None:
    cfg = make_config(initial_capital=50_000.0, fees=0.002)
    assert cfg is not None


# ── run_single ────────────────────────────────────────────────────────────────


def test_run_single_returns_backtest_result() -> None:
    ohlcv = _make_ohlcv_arrays(300)
    n = 300
    # Simple: long the whole period.
    entries = np.zeros(n, dtype=bool)
    entries[50] = True
    exits = np.zeros(n, dtype=bool)
    exits[250] = True

    result = run_single(
        symbol="TEST",
        timestamps=ohlcv.timestamps,
        open_prices=ohlcv.open,
        high_prices=ohlcv.high,
        low_prices=ohlcv.low,
        close_prices=ohlcv.close,
        volume=ohlcv.volume,
        entries=entries,
        exits=exits,
        direction=1,
        weight=1.0,
        config=make_config(),
    )
    assert isinstance(result, BacktestResult)
    assert result.symbol == "TEST"
    assert result.equity_curve.shape[0] > 0
    assert isinstance(result.metrics, BacktestMetrics)


# ── run_long_short ────────────────────────────────────────────────────────────


def test_run_long_short_returns_backtest_result() -> None:
    ohlcv = _make_ohlcv_arrays(300)
    n = 300
    long_entries = np.zeros(n, dtype=bool)
    long_entries[20] = True
    long_exits = np.zeros(n, dtype=bool)
    long_exits[100] = True
    short_entries = np.zeros(n, dtype=bool)
    short_entries[150] = True
    short_exits = np.zeros(n, dtype=bool)
    short_exits[250] = True

    result = run_long_short(
        symbol="TEST",
        timestamps=ohlcv.timestamps,
        open_prices=ohlcv.open,
        high_prices=ohlcv.high,
        low_prices=ohlcv.low,
        close_prices=ohlcv.close,
        volume=ohlcv.volume,
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        long_weight=0.5,
        short_weight=0.5,
        config=make_config(),
    )
    assert isinstance(result, BacktestResult)
    assert result.equity_curve.shape[0] > 0


# ── run_basket ────────────────────────────────────────────────────────────────


def test_run_basket_raises_on_empty_instruments() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_basket(instruments=[])


def test_run_basket_returns_backtest_result() -> None:
    ohlcv = _make_ohlcv_arrays(300)
    n = 300
    entries = np.zeros(n, dtype=bool)
    entries[50] = True
    exits = np.zeros(n, dtype=bool)
    exits[200] = True

    specs = [
        InstrumentSpec(
            symbol="A_long",
            timestamps=ohlcv.timestamps,
            open=ohlcv.open,
            high=ohlcv.high,
            low=ohlcv.low,
            close=ohlcv.close,
            volume=ohlcv.volume,
            entries=entries,
            exits=exits,
            direction=1,
            weight=0.5,
        )
    ]
    result = run_basket(instruments=specs, config=make_config())
    assert isinstance(result, BacktestResult)
    assert isinstance(result.symbol, list)


# ── BacktestRunner ────────────────────────────────────────────────────────────


def test_backtest_runner_with_trend_following() -> None:
    bars = _make_bars(500, seed=1, trend=0.001)
    strategy = TrendFollowing(fast_period=10, slow_period=30)
    runner = BacktestRunner(initial_capital=100_000.0, fees=0.001)
    result = runner.run(strategy, bars, symbol="SIM")
    assert isinstance(result, BacktestResult)
    assert result.metrics.total_trades >= 0


def test_backtest_runner_with_momentum_ts() -> None:
    bars = _make_bars(500, seed=2, trend=0.0005)
    strategy = TimeSeriesMomentum(lookback=60, skip_recent=0, vol_target=0.10)
    runner = BacktestRunner(initial_capital=100_000.0, fees=0.001)
    result = runner.run(strategy, bars, symbol="SIM")
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) > 0


def test_backtest_runner_with_mean_reversion() -> None:
    bars = _make_bars(400, seed=3)
    strategy = MeanReversion(lookback=20, entry_z=2.0, exit_z=0.5)
    runner = BacktestRunner(initial_capital=100_000.0, fees=0.001, slippage=0.001)
    result = runner.run(strategy, bars, symbol="SIM")
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) > 0


# ── BasketRunner ──────────────────────────────────────────────────────────────


def test_basket_runner_with_cs_momentum() -> None:
    multi_bars = {f"SYM_{i}": _make_bars(500, seed=i, trend=(-1) ** i * 0.0003) for i in range(5)}
    strategy = CrossSectionalMomentum(
        lookback=60,
        skip_recent=5,
        top_quantile=0.4,
        bottom_quantile=0.4,
        rebal_freq=10,
    )
    runner = BasketRunner(initial_capital=500_000.0, fees=0.001)
    result = runner.run(strategy, multi_bars)
    assert isinstance(result, BacktestResult)
    assert isinstance(result.symbol, list)
    assert len(result.equity_curve) > 0


def test_basket_runner_raises_when_no_signals() -> None:
    """BasketRunner must raise when the strategy produces only flat positions."""
    # Only 10 bars – not enough for the default 252-bar lookback.
    multi_bars = {
        f"SYM_{i}": pl.DataFrame({"close": [float(100 + i + t) for t in range(10)]})
        for i in range(3)
    }
    strategy = CrossSectionalMomentum(lookback=252)
    runner = BasketRunner()
    with pytest.raises(ValueError, match="No signals"):
        runner.run(strategy, multi_bars)
