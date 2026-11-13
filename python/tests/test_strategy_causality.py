"""Causality: a strategy's signal at bar k must not depend on later bars.

``SignalStrategy`` precomputes signals on the whole history and looks them up
bar by bar. That equals computing them live only if, for every prefix length k,

    generate_signals(bars[:k]) == generate_signals(bars)[:k]

This also catches lookahead in research code in general. Every built-in
strategy is listed here; add new strategies to ``_SINGLE`` / ``_MULTI``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from snippy_scales.strategies import (
    CrossSectionalMomentum,
    MeanReversion,
    MultiAssetStrategy,
    Strategy,
    TimeSeriesMomentum,
    TrendFollowing,
)

_N = 160

_SINGLE: list[Strategy] = [
    TrendFollowing(fast_period=5, slow_period=20, vol_lookback=10),
    TimeSeriesMomentum(lookback=20, skip_recent=3, vol_lookback=10),
    TimeSeriesMomentum(lookback=20, vol_lookback=10, rebal_freq=5),
    MeanReversion(lookback=15, entry_z=1.0, exit_z=0.3, vol_lookback=10),
]
_MULTI: list[MultiAssetStrategy] = [
    CrossSectionalMomentum(lookback=20, skip_recent=3, vol_lookback=10, rebal_freq=5),
    CrossSectionalMomentum(lookback=10, skip_recent=0, vol_lookback=5, rebal_freq=1),
]


def _bars(seed: int, n: int = _N) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
    return pl.DataFrame({"open": close, "high": close, "low": close, "close": close})


def _assert_prefix_equal(prefix: pl.Series, full: pl.Series) -> None:
    np.testing.assert_allclose(
        prefix.cast(pl.Float64).fill_null(np.nan).to_numpy(),
        full.cast(pl.Float64).fill_null(np.nan).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize("strategy", _SINGLE, ids=lambda s: type(s).__name__)
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(1, _N))
def test_single_asset_strategies_are_causal(strategy: Strategy, seed: int, k: int) -> None:
    bars = _bars(seed)
    full = strategy.generate_signals(bars)
    prefix = strategy.generate_signals(bars.head(k))
    _assert_prefix_equal(prefix, full.head(k))


@pytest.mark.parametrize("strategy", _MULTI, ids=lambda s: type(s).__name__)
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(1, _N))
def test_multi_asset_strategies_are_causal(strategy: MultiAssetStrategy, seed: int, k: int) -> None:
    universe = {f"A{i}": _bars(seed + i) for i in range(5)}
    full = strategy.generate_signals(universe)
    prefix = strategy.generate_signals({s: b.head(k) for s, b in universe.items()})
    for symbol in universe:
        _assert_prefix_equal(prefix[symbol], full[symbol].head(k))


def test_lookahead_is_detected() -> None:
    """Sanity check: a strategy that peeks at the next bar fails the property."""

    class Peeking(Strategy):
        def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
            return (bars["close"].shift(-1) / bars["close"] - 1.0).sign().fill_null(0.0)

    bars = _bars(0)
    with pytest.raises(AssertionError):
        _assert_prefix_equal(
            Peeking().generate_signals(bars.head(50)), Peeking().generate_signals(bars).head(50)
        )
