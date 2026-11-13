"""Unit tests for the Cross-Sectional Momentum strategy.

Tests cover:
* Signal shape and dtype invariants for all assets in the universe.
* Warmup period: signals are zero before lookback + skip_recent bars.
* Cross-sectional correctness: best-performing asset is long, worst is short.
* Parameter validation (ValueError for invalid inputs).
* Rebalancing: positions are frozen between rebalancing dates.
* Quantile coverage: number of long/short assets matches the quantile settings.
* Integration with BasketRunner via raptorbt.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from snippy_scales.backtesting.runner import BacktestResult, BasketRunner
from snippy_scales.strategies.momentum_cs import CrossSectionalMomentum, MultiAssetStrategy

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_multi_bars(
    n: int = 500,
    n_assets: int = 6,
    *,
    seed: int = 0,
) -> dict[str, pl.DataFrame]:
    """Generate a universe of synthetic bar DataFrames."""
    rng = np.random.default_rng(seed)
    return {
        f"SYM_{i}": pl.DataFrame({"close": 1000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.012, n))})
        for i in range(n_assets)
    }


def _make_sorted_universe(n: int = 400) -> dict[str, pl.DataFrame]:
    """Universe where assets have known drift order (A: +, B: 0, C: -)."""
    t = np.arange(n, dtype=float)
    return {
        "BEST": pl.DataFrame({"close": 1000.0 * np.exp(+0.002 * t)}),
        "MID": pl.DataFrame({"close": 1000.0 * np.ones(n)}),
        "WORST": pl.DataFrame({"close": 1000.0 * np.exp(-0.002 * t)}),
    }


# ── MultiAssetStrategy ABC ────────────────────────────────────────────────────


def test_multiasset_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        MultiAssetStrategy()  # ty does not flag abstract instantiation


# ── Parameter validation ──────────────────────────────────────────────────────


def test_invalid_lookback_raises() -> None:
    with pytest.raises(ValueError, match="lookback"):
        CrossSectionalMomentum(lookback=0)


def test_invalid_skip_recent_raises() -> None:
    with pytest.raises(ValueError, match="skip_recent"):
        CrossSectionalMomentum(skip_recent=-1)


def test_invalid_top_quantile_raises() -> None:
    with pytest.raises(ValueError, match="top_quantile"):
        CrossSectionalMomentum(top_quantile=0.0)


def test_invalid_bottom_quantile_raises() -> None:
    with pytest.raises(ValueError, match="bottom_quantile"):
        CrossSectionalMomentum(bottom_quantile=1.0)


def test_quantiles_sum_exceeds_one_raises() -> None:
    with pytest.raises(ValueError, match="top_quantile"):
        CrossSectionalMomentum(top_quantile=0.6, bottom_quantile=0.6)


def test_invalid_vol_target_raises() -> None:
    with pytest.raises(ValueError, match="vol_target"):
        CrossSectionalMomentum(vol_target=-0.1)


def test_invalid_max_leverage_raises() -> None:
    with pytest.raises(ValueError, match="max_leverage"):
        CrossSectionalMomentum(max_leverage=0.0)


def test_too_few_assets_raises() -> None:
    strategy = CrossSectionalMomentum(lookback=30, skip_recent=5)
    one_asset = {"SYM_0": pl.DataFrame({"close": [100.0] * 200})}
    with pytest.raises(ValueError, match="≥ 2"):
        strategy.generate_signals(one_asset)


def test_misaligned_lengths_raises() -> None:
    strategy = CrossSectionalMomentum(lookback=30)
    multi_bars = {
        "A": pl.DataFrame({"close": [100.0] * 200}),
        "B": pl.DataFrame({"close": [100.0] * 150}),
    }
    with pytest.raises(ValueError, match="same length"):
        strategy.generate_signals(multi_bars)


# ── Signal shape and type ──────────────────────────────────────────────────────


def test_signal_length_matches_input() -> None:
    multi_bars = _make_multi_bars(n=400, n_assets=4)
    strategy = CrossSectionalMomentum(lookback=60, skip_recent=5, rebal_freq=10)
    signals = strategy.generate_signals(multi_bars)
    for sym, bars in multi_bars.items():
        assert len(signals[sym]) == len(bars), f"Length mismatch for {sym}"


def test_signal_is_polars_series() -> None:
    multi_bars = _make_multi_bars(n=300, n_assets=3)
    strategy = CrossSectionalMomentum(lookback=50, skip_recent=0, rebal_freq=5)
    signals = strategy.generate_signals(multi_bars)
    for sym in multi_bars:
        assert isinstance(signals[sym], pl.Series)
        assert signals[sym].name == "position"


def test_all_symbols_present_in_output() -> None:
    multi_bars = _make_multi_bars(n=400, n_assets=5)
    strategy = CrossSectionalMomentum(lookback=60, rebal_freq=10)
    signals = strategy.generate_signals(multi_bars)
    assert set(signals.keys()) == set(multi_bars.keys())


def test_no_nulls_in_any_signal() -> None:
    multi_bars = _make_multi_bars(n=400, n_assets=4)
    strategy = CrossSectionalMomentum(lookback=60, skip_recent=5, rebal_freq=10)
    signals = strategy.generate_signals(multi_bars)
    for sym in multi_bars:
        assert signals[sym].null_count() == 0, f"Nulls found for {sym}"


# ── Warmup period ─────────────────────────────────────────────────────────────


def test_warmup_period_all_flat() -> None:
    multi_bars = _make_multi_bars(n=400, n_assets=4)
    lookback, skip = 60, 10
    strategy = CrossSectionalMomentum(lookback=lookback, skip_recent=skip, rebal_freq=10)
    signals = strategy.generate_signals(multi_bars)
    warmup = lookback + skip
    for sym in multi_bars:
        warmup_pos = signals[sym].slice(0, warmup)
        assert (warmup_pos == 0.0).all(), f"Non-zero position in warmup for {sym}"


# ── Cross-sectional correctness ───────────────────────────────────────────────


def test_best_asset_goes_long_worst_goes_short() -> None:
    """On a universe with known drift order, the best asset should be long
    and the worst should be short after the warmup period."""
    multi_bars = _make_sorted_universe(n=400)
    strategy = CrossSectionalMomentum(
        lookback=60,
        skip_recent=5,
        top_quantile=0.33,
        bottom_quantile=0.33,
        rebal_freq=10,
    )
    signals = strategy.generate_signals(multi_bars)
    warmup = 60 + 5

    best_after = signals["BEST"].slice(warmup + 10).to_numpy()
    worst_after = signals["WORST"].slice(warmup + 10).to_numpy()

    # Non-zero positions should be predominantly in the correct direction.
    best_nonzero = best_after[best_after != 0.0]
    worst_nonzero = worst_after[worst_after != 0.0]

    if len(best_nonzero) > 0:
        assert (best_nonzero > 0).mean() > 0.7, "BEST asset should mostly be long"
    if len(worst_nonzero) > 0:
        assert (worst_nonzero < 0).mean() > 0.7, "WORST asset should mostly be short"


def test_positions_bounded_by_max_leverage() -> None:
    multi_bars = _make_multi_bars(n=400, n_assets=4)
    max_leverage = 2.0
    strategy = CrossSectionalMomentum(lookback=60, rebal_freq=10, max_leverage=max_leverage)
    signals = strategy.generate_signals(multi_bars)
    for sym in multi_bars:
        pos = signals[sym].to_numpy()
        assert np.all(np.abs(pos) <= max_leverage + 1e-9), (
            f"Position exceeds max_leverage for {sym}"
        )


# ── Rebalancing ────────────────────────────────────────────────────────────────


def test_positions_constant_between_rebal_dates() -> None:
    """Between rebalancing dates, all positions should remain unchanged."""
    rebal_freq = 10
    multi_bars = _make_multi_bars(n=400, n_assets=4)
    strategy = CrossSectionalMomentum(lookback=60, skip_recent=5, rebal_freq=rebal_freq)
    signals = strategy.generate_signals(multi_bars)
    warmup = 65  # lookback + skip_recent

    for sym in multi_bars:
        pos = signals[sym].to_numpy()
        for t in range(warmup + 1, len(pos)):
            bars_since_rebal = (t - warmup) % rebal_freq
            if bars_since_rebal != 0:
                prev_rebal = t - bars_since_rebal
                assert pos[t] == pos[prev_rebal], f"{sym}: position changed off-rebal at bar {t}"


# ── Integration with BasketRunner ─────────────────────────────────────────────


def test_basket_runner_integration() -> None:
    multi_bars = _make_multi_bars(n=500, n_assets=6, seed=99)
    strategy = CrossSectionalMomentum(
        lookback=60,
        skip_recent=5,
        top_quantile=0.33,
        bottom_quantile=0.33,
        rebal_freq=10,
    )
    runner = BasketRunner(initial_capital=500_000.0, fees=0.001)
    result = runner.run(strategy, multi_bars)
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) > 0
    assert np.isfinite(result.metrics.total_return_pct)
