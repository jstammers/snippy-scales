"""Unit tests for the Time-Series Momentum strategy.

Tests cover:
* Signal array shape and type invariants.
* Warmup period: no non-zero values before lookback + skip_recent bars.
* Null handling: no nulls after the warmup period.
* Monotone trend detection: strong upward trend → persistent long signals.
* Parameter validation (ValueError for out-of-range inputs).
* Rebalancing frequency: positions freeze correctly between rebal dates.
* Volatility targeting: position magnitudes reflect vol_target / realized_vol.
* Integration with BacktestRunner via raptorbt.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from snippy_scales.backtesting.runner import BacktestResult, BacktestRunner
from snippy_scales.strategies.momentum_ts import TimeSeriesMomentum

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_bars(n: int, *, trend: float = 0.0, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1000.0 * np.cumprod(1.0 + rng.normal(trend, 0.01, n))
    return pl.DataFrame({"close": close})


def _make_trending_bars(n: int = 500) -> pl.DataFrame:
    """Strong deterministic upward trend."""
    t = np.arange(n, dtype=float)
    close = 1000.0 * np.exp(0.001 * t)  # 0.1% per bar
    return pl.DataFrame({"close": close})


def _make_downtrending_bars(n: int = 500) -> pl.DataFrame:
    """Strong deterministic downward trend."""
    t = np.arange(n, dtype=float)
    close = 1000.0 * np.exp(-0.001 * t)
    return pl.DataFrame({"close": close})


# ── Parameter validation ──────────────────────────────────────────────────────


def test_invalid_lookback_raises() -> None:
    with pytest.raises(ValueError, match="lookback"):
        TimeSeriesMomentum(lookback=0)


def test_invalid_skip_recent_raises() -> None:
    with pytest.raises(ValueError, match="skip_recent"):
        TimeSeriesMomentum(skip_recent=-1)


def test_invalid_vol_target_raises() -> None:
    with pytest.raises(ValueError, match="vol_target"):
        TimeSeriesMomentum(vol_target=0.0)


def test_invalid_max_leverage_raises() -> None:
    with pytest.raises(ValueError, match="max_leverage"):
        TimeSeriesMomentum(max_leverage=-1.0)


def test_invalid_rebal_freq_raises() -> None:
    with pytest.raises(ValueError, match="rebal_freq"):
        TimeSeriesMomentum(rebal_freq=0)


# ── Signal shape and type ──────────────────────────────────────────────────────


def test_signal_length_matches_input() -> None:
    bars = _make_bars(300)
    strategy = TimeSeriesMomentum(lookback=60, vol_lookback=20)
    signals = strategy.generate_signals(bars)
    assert len(signals) == len(bars)


def test_signal_is_polars_series() -> None:
    bars = _make_bars(200)
    strategy = TimeSeriesMomentum(lookback=50)
    signals = strategy.generate_signals(bars)
    assert isinstance(signals, pl.Series)
    assert signals.name == "position"


def test_no_nulls_in_output() -> None:
    bars = _make_bars(300)
    strategy = TimeSeriesMomentum(lookback=60, vol_lookback=20)
    signals = strategy.generate_signals(bars)
    assert signals.null_count() == 0


# ── Warmup period ─────────────────────────────────────────────────────────────


def test_warmup_period_is_flat() -> None:
    """Positions must be zero for the first lookback bars."""
    lookback = 60
    bars = _make_bars(300)
    strategy = TimeSeriesMomentum(lookback=lookback, skip_recent=0)
    signals = strategy.generate_signals(bars)
    # First lookback bars should be zero (warmup).
    warmup_signals = signals.slice(0, lookback)
    assert (warmup_signals == 0.0).all()


def test_skip_recent_extends_warmup() -> None:
    """With skip_recent > 0 the warmup extends by skip_recent bars."""
    lookback = 40
    skip = 10
    bars = _make_bars(300)
    strategy = TimeSeriesMomentum(lookback=lookback, skip_recent=skip)
    signals = strategy.generate_signals(bars)
    warmup_signals = signals.slice(0, lookback + skip)
    assert (warmup_signals == 0.0).all()


# ── Direction correctness ─────────────────────────────────────────────────────


def test_uptrend_produces_long_signals_after_warmup() -> None:
    """A clean upward trend should result in predominantly positive positions."""
    bars = _make_trending_bars(400)
    strategy = TimeSeriesMomentum(lookback=60, vol_lookback=20)
    signals = strategy.generate_signals(bars)
    after_warmup = signals.slice(80).to_numpy()  # safely past warmup
    assert float(np.mean(after_warmup > 0)) > 0.8  # at least 80% long


def test_downtrend_produces_short_signals_after_warmup() -> None:
    """A clean downward trend should result in predominantly negative positions."""
    bars = _make_downtrending_bars(400)
    strategy = TimeSeriesMomentum(lookback=60, vol_lookback=20)
    signals = strategy.generate_signals(bars)
    after_warmup = signals.slice(80).to_numpy()
    assert float(np.mean(after_warmup < 0)) > 0.8  # at least 80% short


# ── Volatility targeting ──────────────────────────────────────────────────────


def test_positions_bounded_by_max_leverage() -> None:
    bars = _make_bars(300, seed=7)
    vol_target = 0.10
    max_leverage = 2.0
    strategy = TimeSeriesMomentum(lookback=50, vol_target=vol_target, max_leverage=max_leverage)
    signals = strategy.generate_signals(bars)
    assert (signals.abs() <= max_leverage + 1e-9).all()


# ── Rebalancing frequency ─────────────────────────────────────────────────────


def test_rebal_freq_positions_change_only_at_rebal_dates() -> None:
    """Between rebalancing dates the position value must be unchanged."""
    rebal_freq = 5
    bars = _make_bars(200, seed=42)
    strategy = TimeSeriesMomentum(lookback=30, vol_lookback=15, rebal_freq=rebal_freq)
    signals = strategy.generate_signals(bars).to_numpy()
    warmup = 30

    for i in range(warmup + 1, len(signals)):
        bars_since_rebal = (i - warmup) % rebal_freq
        if bars_since_rebal != 0:
            # Position must equal the value at the last rebalance date.
            assert signals[i] == signals[i - bars_since_rebal], (
                f"Position changed at bar {i} (bars_since_rebal={bars_since_rebal})"
            )


# ── Integration with BacktestRunner ───────────────────────────────────────────


def test_run_backtest_returns_result() -> None:
    bars = _make_bars(500, trend=0.0005, seed=10)
    strategy = TimeSeriesMomentum(lookback=60, vol_target=0.10, max_leverage=2.0)
    runner = BacktestRunner(initial_capital=100_000.0, fees=0.001)
    result = runner.run(strategy, bars, symbol="SIM_TS")
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) > 0
    assert np.isfinite(result.metrics.sharpe_ratio)


def test_trending_asset_gives_positive_return() -> None:
    """On a clean trend with moderate costs, TSMOM should be profitable."""
    bars = _make_trending_bars(600)
    strategy = TimeSeriesMomentum(lookback=60, vol_target=0.15, max_leverage=2.0)
    runner = BacktestRunner(initial_capital=100_000.0, fees=0.0005)
    result = runner.run(strategy, bars, symbol="TREND")
    assert result.metrics.total_return_pct > 0
