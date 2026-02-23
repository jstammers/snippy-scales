"""Tests for strategy signal generation."""

from __future__ import annotations

import polars as pl
import pytest

from snippy_scales.strategies.trend import TrendFollowing


@pytest.fixture()
def sample_bars() -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    close = 4000.0 + np.cumsum(rng.normal(0, 10, n))
    return pl.DataFrame({"close": close, "volume": rng.integers(1000, 10000, n).astype(float)})


def test_trend_following_signal_length(sample_bars: pl.DataFrame) -> None:
    strategy = TrendFollowing(fast_period=10, slow_period=30)
    signals = strategy.generate_signals(sample_bars)
    assert len(signals) == len(sample_bars)


def test_trend_following_no_nulls_after_warmup(sample_bars: pl.DataFrame) -> None:
    strategy = TrendFollowing(fast_period=10, slow_period=30, vol_lookback=15)
    signals = strategy.generate_signals(sample_bars)
    warmup = 30  # slow period
    tail = signals.slice(warmup)
    assert tail.null_count() == 0
