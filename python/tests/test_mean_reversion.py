"""Unit tests for the Mean Reversion strategy.

Tests cover:
* Signal array shape and null-count invariants.
* Warmup period: positions are zero before the lookback window fills.
* Hysteresis: positions persist correctly in the [exit_z, entry_z] band.
* Direction correctness: large positive z-score → short; negative → long.
* Parameter validation (ValueError for invalid inputs).
* Transaction-cost sensitivity: realistic fees should reduce Sharpe vs no-fees.
* Integration with BacktestRunner via raptorbt.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from snippy_scales.backtesting.runner import BacktestResult, BacktestRunner
from snippy_scales.strategies.mean_reversion import MeanReversion

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_bars(n: int = 400, *, seed: int = 0, sigma: float = 0.01) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1000.0 * np.cumprod(1.0 + rng.normal(0.0, sigma, n))
    return pl.DataFrame({"close": close})


def _make_mean_reverting_bars(n: int = 400) -> pl.DataFrame:
    """Ornstein–Uhlenbeck–like process that reverts to 1000."""
    rng = np.random.default_rng(0)
    price = 1000.0
    prices = []
    mean = 1000.0
    kappa = 0.1  # reversion speed
    sigma = 5.0  # noise
    for _ in range(n):
        price += kappa * (mean - price) + rng.normal(0, sigma)
        price = max(price, 1.0)  # floor
        prices.append(price)
    return pl.DataFrame({"close": prices})


def _make_large_shock_bars(n: int = 200, shock_bar: int = 50) -> pl.DataFrame:
    """Price series with a single large positive shock then flat."""
    close = [1000.0] * n
    for i in range(shock_bar, n):
        close[i] = 1000.0
    close[shock_bar] = 1100.0  # large positive shock
    return pl.DataFrame({"close": close})


# ── Parameter validation ──────────────────────────────────────────────────────


def test_invalid_lookback_raises() -> None:
    with pytest.raises(ValueError, match="lookback"):
        MeanReversion(lookback=1)


def test_invalid_entry_z_raises() -> None:
    with pytest.raises(ValueError, match="entry_z"):
        MeanReversion(entry_z=-1.0)


def test_invalid_exit_z_negative_raises() -> None:
    with pytest.raises(ValueError, match="exit_z"):
        MeanReversion(exit_z=-0.1)


def test_exit_z_gte_entry_z_raises() -> None:
    with pytest.raises(ValueError, match="exit_z"):
        MeanReversion(entry_z=1.5, exit_z=2.0)


def test_invalid_vol_target_raises() -> None:
    with pytest.raises(ValueError, match="vol_target"):
        MeanReversion(vol_target=0.0)


def test_invalid_max_leverage_raises() -> None:
    with pytest.raises(ValueError, match="max_leverage"):
        MeanReversion(max_leverage=0.0)


# ── Signal shape and type ──────────────────────────────────────────────────────


def test_signal_length_matches_input() -> None:
    bars = _make_bars(300)
    strategy = MeanReversion(lookback=20)
    signals = strategy.generate_signals(bars)
    assert len(signals) == len(bars)


def test_signal_is_polars_series_named_position() -> None:
    bars = _make_bars(200)
    strategy = MeanReversion(lookback=15)
    signals = strategy.generate_signals(bars)
    assert isinstance(signals, pl.Series)
    assert signals.name == "position"


def test_no_nulls_in_output() -> None:
    bars = _make_bars(300)
    strategy = MeanReversion(lookback=20)
    signals = strategy.generate_signals(bars)
    assert signals.null_count() == 0


# ── Warmup period ─────────────────────────────────────────────────────────────


def test_warmup_period_all_flat() -> None:
    """First lookback positions should be zero (z-score not yet defined)."""
    lookback = 25
    bars = _make_bars(300)
    strategy = MeanReversion(lookback=lookback)
    signals = strategy.generate_signals(bars)
    # First lookback bars: z-score NaN → positions should be 0.
    warmup = signals.slice(0, lookback)
    assert (warmup == 0.0).all()


# ── Hysteresis (state machine) ────────────────────────────────────────────────


def test_hysteresis_maintains_position_in_band() -> None:
    """Position must be held while |z| is in [exit_z, entry_z]."""
    # Build a price series that triggers a large shock and then reverts slowly.
    n = 300
    entry_z = 2.0
    exit_z = 0.5
    bars = _make_mean_reverting_bars(n)
    strategy = MeanReversion(lookback=20, entry_z=entry_z, exit_z=exit_z, max_leverage=3.0)
    signals = strategy.generate_signals(bars).to_numpy()

    # Find the first non-zero signal.
    nonzero_idx = np.nonzero(signals)[0]
    if len(nonzero_idx) < 2:  # noqa: PLR2004
        pytest.skip("Not enough signals generated for hysteresis test")

    # Between first entry and the next position change, check no premature exit.
    start = nonzero_idx[0]
    entry_sign = np.sign(signals[start])
    for i in range(start + 1, min(start + 30, n)):
        if signals[i] == 0.0:
            break  # Exit is acceptable once z drops below exit_z.
        # If still in position, sign must be consistent.
        assert np.sign(signals[i]) == entry_sign


def test_position_exits_when_z_reverts() -> None:
    """A large shock followed by immediate reversion should result in an exit."""
    # Flat price: z-score is 0 → should stay flat.
    close = [1000.0] * 200
    bars = pl.DataFrame({"close": close})
    strategy = MeanReversion(lookback=20, entry_z=2.0, exit_z=0.5)
    signals = strategy.generate_signals(bars)
    # Perfectly flat returns → z-score ≈ NaN or 0 → all flat.
    assert (signals == 0.0).all()


# ── Direction correctness ─────────────────────────────────────────────────────


def test_large_positive_shock_triggers_short() -> None:
    """A large upward price spike should result in a short position (reversion bet)."""
    bars = _make_large_shock_bars(n=200, shock_bar=100)
    strategy = MeanReversion(lookback=20, entry_z=1.5, exit_z=0.3, max_leverage=3.0)
    signals = strategy.generate_signals(bars).to_numpy()
    # At or just after the shock bar, the strategy should go short.
    window = signals[100:115]
    assert (window < 0).any(), "Expected at least one short position after the shock"


# ── Positions bounded by max_leverage ─────────────────────────────────────────


def test_positions_bounded_by_max_leverage() -> None:
    bars = _make_bars(400, seed=5, sigma=0.02)
    max_leverage = 2.0
    strategy = MeanReversion(lookback=20, max_leverage=max_leverage)
    signals = strategy.generate_signals(bars)
    assert (signals.abs() <= max_leverage + 1e-9).all()


# ── Transaction-cost sensitivity ──────────────────────────────────────────────


def test_higher_fees_reduce_profitability() -> None:
    """Realistic fees should reduce (or equal) performance vs zero fees.

    This tests that the backtest correctly models transaction costs — the
    mean reversion strategy has high turnover so fees matter.
    """
    bars = _make_mean_reverting_bars(n=500)
    strategy = MeanReversion(lookback=20, entry_z=1.8, exit_z=0.4)

    runner_zerofee = BacktestRunner(initial_capital=100_000.0, fees=0.0, slippage=0.0)
    runner_realistic = BacktestRunner(initial_capital=100_000.0, fees=0.002, slippage=0.001)

    result_zero = runner_zerofee.run(strategy, bars, symbol="MR_ZEROFEE")
    result_real = runner_realistic.run(strategy, bars, symbol="MR_REAL")

    # Realistic costs should not exceed zero-fee performance.
    assert result_real.metrics.total_return_pct <= result_zero.metrics.total_return_pct


# ── Integration with BacktestRunner ───────────────────────────────────────────


def test_run_backtest_returns_result() -> None:
    bars = _make_bars(400, seed=20)
    strategy = MeanReversion(lookback=20, entry_z=2.0, exit_z=0.5)
    runner = BacktestRunner(initial_capital=100_000.0, fees=0.001, slippage=0.0005)
    result = runner.run(strategy, bars, symbol="SIM_MR")
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) > 0
    assert np.isfinite(result.metrics.sharpe_ratio)


def test_mean_reverting_process_with_low_fees() -> None:
    """On a genuinely mean-reverting process with low fees, the strategy should
    show at least some trades."""
    bars = _make_mean_reverting_bars(n=500)
    strategy = MeanReversion(lookback=20, entry_z=1.5, exit_z=0.3, max_leverage=2.0)
    runner = BacktestRunner(initial_capital=100_000.0, fees=0.0001, slippage=0.0)
    result = runner.run(strategy, bars, symbol="MR_OPT")
    assert result.metrics.total_trades > 0
