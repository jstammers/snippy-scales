"""Tests for selection-bias corrections and probabilistic-forecast scoring.

The load-bearing test is :func:`test_deflated_sharpe_rejects_pure_noise`: a
sweep over hundreds of *worthless* random strategies produces a headline Sharpe
that looks respectable, and the Deflated Sharpe Ratio must refuse to endorse
it.  If that assertion ever fails, the sweep machinery is licensing false
discoveries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from snippy_scales.evaluation.scoring import (
    calibration_error,
    coverage,
    crps_ensemble,
    crps_gaussian,
    pinball_loss,
    pit_values,
)
from snippy_scales.evaluation.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)

_TRADING_DAYS = 252


# ── Expected maximum under the null ───────────────────────────────────────────


def test_expected_max_sharpe_grows_with_trials() -> None:
    """Searching harder must raise the bar a result has to clear."""
    values = [expected_max_sharpe(n) for n in (1, 10, 100, 1000)]
    assert values[0] == 0.0
    assert values == sorted(values)


def test_expected_max_sharpe_matches_simulation() -> None:
    """The extreme-value approximation must track a Monte-Carlo maximum."""
    rng = np.random.default_rng(0)
    n_trials = 50
    draws = rng.standard_normal((20_000, n_trials)).max(axis=1)
    assert expected_max_sharpe(n_trials) == pytest.approx(float(draws.mean()), abs=0.05)


def test_expected_max_sharpe_scales_with_dispersion() -> None:
    """Doubling trial dispersion must double the expected maximum."""
    assert expected_max_sharpe(100, trial_std=2.0) == pytest.approx(
        2.0 * expected_max_sharpe(100, trial_std=1.0)
    )


# ── The synthetic null ────────────────────────────────────────────────────────


def test_deflated_sharpe_rejects_pure_noise() -> None:
    """A sweep over worthless strategies must not survive deflation.

    500 zero-edge return series are generated, the best is selected exactly as
    ``EvaluationResult.best_result`` would, and its raw Sharpe is shown to look
    plausible.  The DSR must nonetheless stay far below the 0.95 bar.
    """
    rng = np.random.default_rng(42)
    n_trials, n_obs = 500, 4 * _TRADING_DAYS
    returns = rng.normal(0.0, 0.01, size=(n_obs, n_trials))

    sharpes = returns.mean(axis=0) / returns.std(axis=0, ddof=1) * math.sqrt(_TRADING_DAYS)
    best = int(np.argmax(sharpes))
    observed = float(sharpes[best])

    # The naive headline number looks like a real strategy.
    assert observed > 1.0

    dsr = deflated_sharpe_ratio(
        observed,
        n_trials=n_trials,
        n_obs=n_obs,
        trial_std=float(np.std(sharpes)),
        skew=float(stats.skew(returns[:, best])),
        kurtosis=float(stats.kurtosis(returns[:, best], fisher=False)),
        periods_per_year=_TRADING_DAYS,
    )
    assert dsr < 0.5, f"DSR endorsed a pure-noise sweep (observed Sharpe {observed:.2f})"


def test_deflated_sharpe_accepts_genuine_edge() -> None:
    """A strong, lightly-searched result must survive deflation."""
    rng = np.random.default_rng(7)
    n_obs = 10 * _TRADING_DAYS
    returns = rng.normal(0.0010, 0.008, size=n_obs)
    observed = returns.mean() / returns.std(ddof=1) * math.sqrt(_TRADING_DAYS)

    dsr = deflated_sharpe_ratio(
        observed,
        n_trials=5,
        n_obs=n_obs,
        trial_std=0.3,
        periods_per_year=_TRADING_DAYS,
    )
    assert dsr > 0.95


def test_broad_search_can_bury_a_real_edge() -> None:
    """A modest but genuine edge searched 50 ways must fail deflation.

    The deflation is not merely a formality applied to noise: a real Sharpe-0.7
    strategy, discovered by a 50-point sweep over 10 years of daily data, is
    statistically indistinguishable from the best of 50 coin flips.  Search
    breadth has to be budgeted, not maximised.
    """
    rng = np.random.default_rng(7)
    n_obs = 10 * _TRADING_DAYS
    returns = rng.normal(0.0006, 0.008, size=n_obs)
    observed = returns.mean() / returns.std(ddof=1) * math.sqrt(_TRADING_DAYS)
    assert observed > 0.5  # the edge is genuinely there

    narrow = deflated_sharpe_ratio(
        observed, n_trials=3, n_obs=n_obs, trial_std=0.3, periods_per_year=_TRADING_DAYS
    )
    broad = deflated_sharpe_ratio(
        observed, n_trials=50, n_obs=n_obs, trial_std=0.3, periods_per_year=_TRADING_DAYS
    )
    assert narrow > 0.85
    assert broad < 0.6
    assert narrow - broad > 0.25


def test_deflated_sharpe_decreases_with_more_trials() -> None:
    """Holding the result fixed, searching more must lower confidence in it."""
    scores = [
        deflated_sharpe_ratio(
            1.5, n_trials=n, n_obs=1000, trial_std=0.5, periods_per_year=_TRADING_DAYS
        )
        for n in (1, 10, 100, 10_000)
    ]
    assert scores == sorted(scores, reverse=True)


def test_deflated_sharpe_handles_degenerate_input() -> None:
    """Too few observations must return 0.0 rather than raise."""
    assert deflated_sharpe_ratio(1.0, n_trials=10, n_obs=1) == 0.0


# ── Probabilistic Sharpe ──────────────────────────────────────────────────────


def test_probabilistic_sharpe_increases_with_sample_length() -> None:
    """The same Sharpe observed for longer must be more convincing."""
    scores = [probabilistic_sharpe_ratio(0.8, n_obs=n) for n in (30, 250, 2500)]
    assert scores == sorted(scores)


def test_negative_skew_and_fat_tails_reduce_confidence() -> None:
    """Crash-prone return profiles must be penalised."""
    normal = probabilistic_sharpe_ratio(1.0, n_obs=500, periods_per_year=_TRADING_DAYS)
    skewed = probabilistic_sharpe_ratio(
        1.0, n_obs=500, skew=-1.5, kurtosis=9.0, periods_per_year=_TRADING_DAYS
    )
    assert skewed < normal


def test_annualised_sharpe_must_declare_its_frequency() -> None:
    """Forgetting periods_per_year must not quietly manufacture confidence.

    Guards the units trap directly: an annualised Sharpe passed as if it were
    per-observation produces near-certainty from a sample far too short to
    support it.
    """
    honest = probabilistic_sharpe_ratio(0.5, n_obs=60, periods_per_year=_TRADING_DAYS)
    mistaken = probabilistic_sharpe_ratio(0.5, n_obs=60)
    assert honest < 0.7
    assert mistaken > 0.99


def test_probabilistic_sharpe_at_benchmark_is_half() -> None:
    """Observing exactly the benchmark gives a coin-flip verdict."""
    assert probabilistic_sharpe_ratio(0.5, n_obs=500, benchmark_sharpe=0.5) == pytest.approx(0.5)


# ── Minimum track record ──────────────────────────────────────────────────────


def test_min_track_record_length_shrinks_with_stronger_edge() -> None:
    """A better strategy needs less time to prove itself."""
    lengths = [
        min_track_record_length(s, periods_per_year=_TRADING_DAYS) for s in (0.3, 0.6, 1.2, 2.0)
    ]
    assert lengths == sorted(lengths, reverse=True)
    assert all(math.isfinite(x) for x in lengths)


def test_min_track_record_length_infinite_without_edge() -> None:
    """No sample size establishes an edge that is not there."""
    assert min_track_record_length(0.0) == math.inf
    assert min_track_record_length(0.4, benchmark_sharpe=0.6) == math.inf


def test_min_track_record_length_is_sobering_for_realistic_sharpe() -> None:
    """A Sharpe-0.5 daily strategy needs roughly a decade to prove itself.

    This is the number that governs whether a retail research programme can
    ever confirm its own edge from live trading alone.
    """
    years = min_track_record_length(0.5, periods_per_year=_TRADING_DAYS) / _TRADING_DAYS
    assert 8.0 < years < 15.0


def test_min_track_record_length_validates_confidence() -> None:
    """Confidence outside (0, 1) must raise."""
    with pytest.raises(ValueError, match="confidence"):
        min_track_record_length(1.0, confidence=1.5)


# ── Probability of backtest overfitting ───────────────────────────────────────


def test_pbo_centres_on_half_for_indistinguishable_strategies() -> None:
    """With no real edge, in-sample selection must be a coin flip.

    Averaged across seeds because the single-sample spread is wide (sd ≈ 0.2):
    the CSCV splits overlap heavily, so one draw is a weak estimate of the
    procedure's behaviour even though its expectation is exactly 0.5.
    """
    scores = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        scores.append(
            probability_of_backtest_overfitting(
                rng.normal(0.0, 0.01, size=(600, 20)), n_partitions=8
            )
        )
    assert float(np.mean(scores)) == pytest.approx(0.5, abs=0.1)


def test_pbo_low_when_one_strategy_dominates() -> None:
    """A persistently superior configuration must be selected reliably."""
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, size=(600, 20))
    returns[:, 3] += 0.004  # a genuinely, consistently better configuration
    assert probability_of_backtest_overfitting(returns, n_partitions=8) < 0.1


def test_pbo_rejects_bad_shapes() -> None:
    """Malformed inputs must raise rather than silently return a number."""
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="2-D"):
        probability_of_backtest_overfitting(rng.normal(size=(100, 1)))
    with pytest.raises(ValueError, match="n_partitions"):
        probability_of_backtest_overfitting(rng.normal(size=(100, 5)), n_partitions=7)


# ── CRPS ──────────────────────────────────────────────────────────────────────


def test_crps_gaussian_reduces_to_absolute_error_at_zero_spread() -> None:
    """A point forecast must score as MAE, making CRPS directly comparable."""
    obs = np.array([1.0, -2.0, 0.5])
    mu = np.array([0.0, 0.0, 0.0])
    scores = crps_gaussian(obs, mu, np.zeros(3))
    np.testing.assert_allclose(scores, np.abs(obs - mu))


def test_crps_gaussian_minimised_at_true_parameters() -> None:
    """CRPS is proper: the true distribution must score best."""
    rng = np.random.default_rng(1)
    obs = rng.normal(0.0, 1.0, 20_000)
    truth = crps_gaussian(obs, np.zeros_like(obs), np.ones_like(obs)).mean()
    too_wide = crps_gaussian(obs, np.zeros_like(obs), np.full_like(obs, 2.0)).mean()
    too_narrow = crps_gaussian(obs, np.zeros_like(obs), np.full_like(obs, 0.4)).mean()
    biased = crps_gaussian(obs, np.full_like(obs, 1.0), np.ones_like(obs)).mean()

    assert truth < too_wide
    assert truth < too_narrow
    assert truth < biased


def test_crps_ensemble_matches_gaussian_closed_form() -> None:
    """The ensemble estimator must converge to the analytic Gaussian CRPS."""
    rng = np.random.default_rng(2)
    obs = np.array([0.0, 1.0, -1.5])
    samples = rng.normal(0.0, 1.0, size=(3, 60_000))
    approx = crps_ensemble(obs, samples)
    exact = crps_gaussian(obs, np.zeros(3), np.ones(3))
    np.testing.assert_allclose(approx, exact, atol=0.02)


def test_crps_ensemble_rejects_misaligned_input() -> None:
    """Shape errors must be caught explicitly."""
    with pytest.raises(ValueError, match="2-D"):
        crps_ensemble(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="rows"):
        crps_ensemble(np.zeros(3), np.zeros((5, 10)))


# ── Pinball loss ──────────────────────────────────────────────────────────────


def test_pinball_loss_is_asymmetric() -> None:
    """A low quantile must punish over-prediction far harder."""
    over = pinball_loss(np.array([0.0]), np.array([1.0]), quantile=0.05)
    under = pinball_loss(np.array([1.0]), np.array([0.0]), quantile=0.05)
    assert over[0] == pytest.approx(0.95)
    assert under[0] == pytest.approx(0.05)


def test_pinball_loss_minimised_at_true_quantile() -> None:
    """The optimal constant prediction is the distribution's true quantile."""
    rng = np.random.default_rng(6)
    obs = rng.normal(0.0, 1.0, 40_000)
    q = 0.25
    truth = float(stats.norm.ppf(q))
    best = pinball_loss(obs, np.full_like(obs, truth), q).mean()
    for offset in (-0.5, -0.2, 0.2, 0.5):
        assert best < pinball_loss(obs, np.full_like(obs, truth + offset), q).mean()


def test_pinball_loss_validates_quantile() -> None:
    """Quantiles outside (0, 1) must raise."""
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss(np.zeros(3), np.zeros(3), quantile=0.0)


# ── Calibration ───────────────────────────────────────────────────────────────


def test_pit_uniform_for_correct_forecast() -> None:
    """A correctly specified forecast must produce uniform PIT values."""
    rng = np.random.default_rng(8)
    n = 4000
    obs = rng.normal(0.0, 1.0, n)
    samples = rng.normal(0.0, 1.0, size=(n, 400))
    assert calibration_error(pit_values(obs, samples)) < 0.05


def test_pit_detects_overconfidence() -> None:
    """Too-narrow forecast intervals must show up as miscalibration."""
    rng = np.random.default_rng(9)
    n = 4000
    obs = rng.normal(0.0, 1.0, n)
    narrow = rng.normal(0.0, 0.3, size=(n, 400))

    honest = calibration_error(pit_values(obs, rng.normal(0.0, 1.0, size=(n, 400))))
    assert calibration_error(pit_values(obs, narrow)) > honest + 0.1


def test_pit_rejects_misaligned_input() -> None:
    """Shape errors must be caught explicitly."""
    with pytest.raises(ValueError, match="2-D"):
        pit_values(np.zeros(3), np.zeros(3))


def test_coverage_matches_nominal_for_correct_intervals() -> None:
    """A correctly specified 90% interval must cover about 90%."""
    rng = np.random.default_rng(10)
    obs = rng.normal(0.0, 1.0, 20_000)
    z = float(stats.norm.ppf(0.95))
    assert coverage(obs, np.full_like(obs, -z), np.full_like(obs, z)) == pytest.approx(
        0.90, abs=0.01
    )


def test_coverage_empty_input() -> None:
    """An empty input must return 0.0 rather than raise."""
    assert coverage(np.zeros(0), np.zeros(0), np.zeros(0)) == 0.0
