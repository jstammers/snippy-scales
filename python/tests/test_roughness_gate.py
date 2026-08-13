"""Tests for the roughness gate and the noise-aware Hurst estimator.

The gate's job is to *refuse*, cheaply, before an expensive modelling
programme starts. So the load-bearing tests here are the rejections — and
above all :func:`test_gate_rejects_noisy_markovian_data`, which covers the case
where the naive estimator would have waved a Markovian series through as
"rough".

A gate biased toward authorising the work it guards is worse than no gate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from snippy_scales.research.diagnostics import (
    GateThresholds,
    RoughnessVerdict,
    log_variance_proxy,
    roughness_gate,
)
from snippy_scales.research.roughness import hurst_exponent, hurst_with_nugget
from snippy_scales.research.sde import (
    HestonParams,
    RoughBergomiParams,
    fractional_gaussian_noise,
    simulate_heston,
    simulate_rough_bergomi,
)

_TRADING_DAYS = 252
_DT = 1.0 / _TRADING_DAYS
_N = 6000


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _rough_log_variance(hurst: float = 0.12, seed: int = 1) -> np.ndarray:
    """Return log variance from a rough Bergomi simulation.

    Args:
        hurst: True Hurst exponent.
        seed: RNG seed.

    Returns:
        Log variance path.
    """
    rng = np.random.default_rng(seed)
    _, variances = simulate_rough_bergomi(
        RoughBergomiParams(hurst=hurst, eta=1.5, rho=-0.7, xi0=0.04),
        _N,
        _DT,
        n_paths=1,
        rng=rng,
    )
    return np.log(variances[0, 1:])


def _heston_log_variance(seed: int = 2) -> np.ndarray:
    """Return log variance from a Heston (Markovian) simulation.

    Args:
        seed: RNG seed.

    Returns:
        Log variance path.
    """
    rng = np.random.default_rng(seed)
    _, variances = simulate_heston(
        HestonParams(kappa=3.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.04),
        _N,
        _DT,
        n_paths=1,
        rng=rng,
    )
    return np.log(np.maximum(variances[0, 1:], 1e-12))


# ── The noise trap ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("true_hurst", [0.3, 0.5, 0.7])
def test_measurement_noise_biases_naive_hurst_toward_rough(true_hurst: float) -> None:
    """Modest observation noise must collapse the naive estimate toward zero.

    This documents *why* the naive log-log slope cannot be used for a gating
    decision: white measurement noise adds a constant to the variogram, which
    flattens the short-lag slope. The bias runs toward the "rough" verdict —
    the direction that would authorise expensive work.
    """
    rng = np.random.default_rng(3)
    signal = np.cumsum(fractional_gaussian_noise(8192, true_hurst, n_paths=1, rng=rng)[0])
    signal = signal / signal.std()
    noisy = signal + rng.normal(0.0, 0.25, signal.size)

    assert hurst_exponent(signal) == pytest.approx(true_hurst, abs=0.06)
    assert hurst_exponent(noisy) < 0.15


@pytest.mark.parametrize("true_hurst", [0.3, 0.5, 0.7])
def test_nugget_estimator_survives_measurement_noise(true_hurst: float) -> None:
    """The nugget fit must stay near the truth where the naive estimate collapses."""
    rng = np.random.default_rng(3)
    signal = np.cumsum(fractional_gaussian_noise(8192, true_hurst, n_paths=1, rng=rng)[0])
    signal = signal / signal.std()
    noisy = signal + rng.normal(0.0, 0.1, signal.size)

    naive = hurst_exponent(noisy)
    estimate = hurst_with_nugget(noisy)

    assert estimate.hurst > true_hurst - 0.12
    # The property that matters: strictly closer to the truth than the naive fit.
    assert abs(estimate.hurst - true_hurst) < abs(naive - true_hurst)


def test_nugget_estimator_recovers_the_noise_level() -> None:
    """The fitted nugget must match the noise actually injected."""
    rng = np.random.default_rng(7)
    signal = np.cumsum(fractional_gaussian_noise(8192, 0.5, n_paths=1, rng=rng)[0])
    signal = signal / signal.std()

    for noise in (0.1, 0.25, 0.5):
        estimate = hurst_with_nugget(signal + rng.normal(0.0, noise, signal.size))
        assert estimate.noise_std == pytest.approx(noise, rel=0.15)


def test_nugget_noise_share_rises_with_noise() -> None:
    """The noise-share diagnostic must track how much signal is left."""
    rng = np.random.default_rng(8)
    signal = np.cumsum(fractional_gaussian_noise(8192, 0.4, n_paths=1, rng=rng)[0])
    signal = signal / signal.std()

    shares = [
        hurst_with_nugget(signal + rng.normal(0.0, s, signal.size)).noise_share
        for s in (0.0, 0.1, 0.3, 0.8)
    ]
    assert shares == sorted(shares)
    assert shares[0] < 0.5
    assert shares[-1] > 0.9


def test_nugget_on_clean_signal_matches_naive() -> None:
    """With no noise the two estimators must broadly agree."""
    rng = np.random.default_rng(9)
    signal = np.cumsum(fractional_gaussian_noise(8192, 0.45, n_paths=1, rng=rng)[0])
    estimate = hurst_with_nugget(signal)

    assert estimate.converged
    assert estimate.hurst == pytest.approx(hurst_exponent(signal), abs=0.06)
    assert estimate.noise_share < 0.5


# ── Gate verdicts against known ground truth ──────────────────────────────────


def test_gate_passes_genuinely_rough_volatility() -> None:
    """Clean rough Bergomi at H = 0.12 must be the one case that proceeds."""
    report = roughness_gate(_rough_log_variance(), symbol="ROUGH-SIM", n_folds=5)

    assert report.verdict is RoughnessVerdict.ROUGH
    assert report.proceed_to_neural
    assert report.mean_hurst < 0.25
    assert len(report.folds) == 5


def test_gate_refuses_markovian_volatility() -> None:
    """Heston must not authorise a rough or neural model."""
    report = roughness_gate(_heston_log_variance(), symbol="HESTON-SIM", n_folds=5)

    assert not report.proceed_to_neural
    assert report.verdict in {RoughnessVerdict.MARKOVIAN, RoughnessVerdict.INTERMEDIATE}
    assert report.mean_hurst > 0.25


def test_gate_rejects_noisy_markovian_data() -> None:
    """The trap: a Markovian series plus RV noise must not read as rough.

    The naive estimator reports H ≈ 0.13 here — squarely in "volatility is
    rough" territory — and would have authorised the entire neural programme.
    The gate must refuse, and must say the data is inadequate rather than
    blaming the model.
    """
    rng = np.random.default_rng(3)
    clean = _heston_log_variance()
    noisy = clean + rng.normal(0.0, 0.5, clean.size)

    report = roughness_gate(noisy, symbol="TRAP", n_folds=5)

    assert not report.proceed_to_neural, "gate authorised work on noise-dominated data"
    assert report.verdict is RoughnessVerdict.NOISE_DOMINATED
    assert report.mean_hurst_naive < 0.25  # what the naive estimator would have claimed
    assert report.mean_noise_share > 0.5
    assert "higher-frequency" in report.rationale


def test_gate_rejects_unstable_hurst() -> None:
    """A series whose roughness regime-switches must be refused as unstable."""
    rng = np.random.default_rng(4)
    segments = [
        np.cumsum(fractional_gaussian_noise(_N // 5, h, n_paths=1, rng=rng)[0])
        for h in (0.10, 0.45, 0.15, 0.50, 0.12)
    ]
    report = roughness_gate(np.concatenate(segments), symbol="UNSTABLE", n_folds=5)

    assert report.verdict is RoughnessVerdict.UNSTABLE
    assert not report.proceed_to_neural
    assert report.std_hurst > report.thresholds.max_hurst_std


def test_gate_rejects_jump_dominated_series() -> None:
    """Jump domination must veto even a genuinely rough volatility path."""
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.01, _N)
    returns[::80] += 0.20

    report = roughness_gate(_rough_log_variance(), returns=returns, symbol="JUMPY", n_folds=5)

    assert report.verdict is RoughnessVerdict.JUMP_DOMINATED
    assert not report.proceed_to_neural
    assert report.jump_fraction > report.thresholds.max_jump_fraction


def test_gate_reports_insufficient_data() -> None:
    """Too few observations must be reported, not silently estimated."""
    report = roughness_gate(np.random.default_rng(6).normal(size=200), symbol="TINY", n_folds=5)

    assert report.verdict is RoughnessVerdict.INSUFFICIENT_DATA
    assert not report.proceed_to_neural
    assert report.folds == []


def test_gate_requires_at_least_two_folds() -> None:
    """Stability is the point, so a single fold must be rejected."""
    with pytest.raises(ValueError, match="at least 2"):
        roughness_gate(np.zeros(5000), n_folds=1)


def test_thresholds_are_configurable() -> None:
    """A stricter roughness threshold must be able to flip a passing verdict."""
    log_rv = _rough_log_variance()

    default = roughness_gate(log_rv, symbol="S", n_folds=5)
    strict = roughness_gate(
        log_rv, symbol="S", n_folds=5, thresholds=GateThresholds(rough_below=0.05)
    )

    assert default.proceed_to_neural
    assert not strict.proceed_to_neural


def test_summary_is_readable_and_shows_both_estimators() -> None:
    """The summary must expose the naive estimate alongside the corrected one."""
    text = roughness_gate(_rough_log_variance(), symbol="ES.c.0", n_folds=5).summary()

    assert "ES.c.0" in text
    assert "ROUGH" in text
    assert "naive" in text
    assert "noise share" in text
    assert text.count("[") >= 5  # one line per fold


# ── Log-variance proxy ────────────────────────────────────────────────────────


def _make_bars(n: int = 500, seed: int = 0) -> pl.DataFrame:
    """Build a small OHLC frame for proxy tests.

    Args:
        n: Number of bars.
        seed: RNG seed.

    Returns:
        Polars DataFrame with OHLC columns.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    spread = np.abs(rng.normal(0.0, 0.004, n))
    return pl.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1.0 + spread),
            "low": np.minimum(open_, close) * (1.0 - spread),
            "close": close,
        }
    )


def test_log_variance_proxy_shape_and_finiteness() -> None:
    """The proxy must be finite and aligned with the input bars."""
    bars = _make_bars(400)
    proxy = log_variance_proxy(bars)

    assert proxy.shape == (400,)
    assert bool(np.all(np.isfinite(proxy)))


def test_log_variance_proxy_tracks_volatility_level() -> None:
    """A higher-volatility series must produce a higher mean log variance."""
    rng = np.random.default_rng(11)

    def bars_at(vol: float) -> pl.DataFrame:
        n = 2000
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, vol, n)))
        open_ = np.concatenate(([close[0]], close[:-1]))
        spread = np.abs(rng.normal(0.0, vol * 0.4, n))
        return pl.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * (1.0 + spread),
                "low": np.minimum(open_, close) * (1.0 - spread),
                "close": close,
            }
        )

    assert log_variance_proxy(bars_at(0.02)).mean() > log_variance_proxy(bars_at(0.005)).mean()


def test_log_variance_proxy_survives_zero_range_bars() -> None:
    """Flat bars must be floored rather than producing -inf."""
    bars = _make_bars(300).with_columns(
        pl.when(pl.int_range(pl.len()) % 50 == 0)
        .then(pl.col("close"))
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.int_range(pl.len()) % 50 == 0)
        .then(pl.col("close"))
        .otherwise(pl.col("low"))
        .alias("low"),
    )
    proxy = log_variance_proxy(bars)
    assert bool(np.all(np.isfinite(proxy)))


def test_log_variance_proxy_requires_ohlc() -> None:
    """A missing column must be named explicitly."""
    with pytest.raises(ValueError, match="high"):
        log_variance_proxy(pl.DataFrame({"open": [1.0], "low": [1.0], "close": [1.0]}))
