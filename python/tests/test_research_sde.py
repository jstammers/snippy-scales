"""Tests for volatility estimators, roughness diagnostics and SDE models.

Every estimator here is validated against **simulated data with known ground
truth** before it is trusted on market data.  That ordering is the whole point
of this layer: an estimator that cannot recover a Hurst exponent it was handed
cannot be used to claim volatility is rough.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from snippy_scales.research.roughness import (
    bipower_variation,
    hurst_exponent,
    jump_ratio,
    realised_variance,
    variogram,
)
from snippy_scales.research.sde import (
    HestonParams,
    OUParams,
    RoughBergomiParams,
    fit_ou,
    fractional_gaussian_noise,
    simulate_heston,
    simulate_ou,
    simulate_rough_bergomi,
)
from snippy_scales.research.volatility import (
    garman_klass_vol,
    parkinson_vol,
    rogers_satchell_vol,
    yang_zhang_vol,
)

_TRADING_DAYS = 252

#: Intrabar path resolution used when synthesising OHLC bars.  See the note in
#: :func:`_make_ohlc` — too few steps biases every range estimator downward.
_INTRABAR_STEPS = 500


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_ohlc(
    n: int = 500,
    seed: int = 0,
    daily_vol: float = 0.01,
    drift: float = 0.0,
    gap_vol: float = 0.0,
) -> dict[str, pl.Series]:
    """Generate OHLC bars with a known daily volatility.

    Args:
        n: Number of bars.
        seed: RNG seed.
        daily_vol: Standard deviation of the open-to-close move.
        drift: Per-bar drift added to the open-to-close move.
        gap_vol: Standard deviation of the overnight close-to-open gap.

    Returns:
        Mapping of ``open``/``high``/``low``/``close`` to Polars series.

    Note:
        ``_INTRABAR_STEPS`` must be large.  Range estimators target the
        *continuous* high and low, so a coarsely sampled intrabar path
        understates the range and biases every range estimator downward — at
        20 steps they read only ~80% of the true volatility, which would look
        like an estimator bug rather than a fixture artefact.
    """
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    open_ = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)

    steps_per_bar = _INTRABAR_STEPS
    prev_close = 100.0
    for i in range(n):
        gap = rng.normal(0.0, gap_vol) if gap_vol > 0.0 else 0.0
        o = prev_close * math.exp(gap)
        # Simulate the intrabar path so the high/low are internally consistent.
        steps = rng.normal(
            drift / steps_per_bar, daily_vol / math.sqrt(steps_per_bar), steps_per_bar
        )
        path = o * np.exp(np.cumsum(steps))
        open_[i] = o
        close[i] = path[-1]
        high[i] = max(o, float(path.max()))
        low[i] = min(o, float(path.min()))
        prev_close = close[i]

    return {
        "open": pl.Series("open", open_),
        "high": pl.Series("high", high),
        "low": pl.Series("low", low),
        "close": pl.Series("close", close),
    }


def _mean(series: pl.Series) -> float:
    """Return the mean of a series' non-null values as a plain float.

    Polars' ``mean()`` is typed as optionally ``None``, so this narrows it once
    here rather than at every call site.

    Args:
        series: Series to average.

    Returns:
        Mean of the non-null values; ``nan`` when the series is entirely null.
    """
    values = series.drop_nulls().to_numpy()
    return float(np.mean(values)) if values.size else float("nan")


def _std(series: pl.Series) -> float:
    """Return the standard deviation of a series' non-null values as a float.

    Args:
        series: Series to measure.

    Returns:
        Standard deviation of the non-null values; ``nan`` when undefined.
    """
    values = series.drop_nulls().to_numpy()
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


# ── Volatility estimators ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "estimator",
    ["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"],
)
def test_estimators_recover_known_volatility(estimator: str) -> None:
    """Each estimator must recover the volatility it was handed."""
    daily_vol = 0.012
    bars = _make_ohlc(n=3000, seed=1, daily_vol=daily_vol)
    expected = daily_vol * math.sqrt(_TRADING_DAYS)

    if estimator == "parkinson":
        series = parkinson_vol(bars["high"], bars["low"], window=250)
    elif estimator == "garman_klass":
        series = garman_klass_vol(
            bars["high"], bars["low"], bars["open"], bars["close"], window=250
        )
    elif estimator == "rogers_satchell":
        series = rogers_satchell_vol(
            bars["high"], bars["low"], bars["open"], bars["close"], window=250
        )
    else:
        series = yang_zhang_vol(bars["high"], bars["low"], bars["open"], bars["close"], window=250)

    estimate = _mean(series)
    assert estimate == pytest.approx(expected, rel=0.15), (
        f"{estimator} estimated {estimate:.4f}, expected {expected:.4f}"
    )


def test_estimators_return_full_length_series() -> None:
    """Rolling estimators must not silently shorten the series."""
    bars = _make_ohlc(n=300, seed=2)
    for series in (
        parkinson_vol(bars["high"], bars["low"], window=20),
        garman_klass_vol(bars["high"], bars["low"], bars["open"], bars["close"], window=20),
        rogers_satchell_vol(bars["high"], bars["low"], bars["open"], bars["close"], window=20),
        yang_zhang_vol(bars["high"], bars["low"], bars["open"], bars["close"], window=20),
    ):
        assert len(series) == 300


def test_range_estimators_beat_close_to_close_efficiency() -> None:
    """Range estimators must be less noisy than close-to-close at equal window.

    This is the reason to use them at all: same data, same window, materially
    lower estimator variance.
    """
    daily_vol = 0.012
    bars = _make_ohlc(n=4000, seed=3, daily_vol=daily_vol)

    close_to_close = bars["close"].pct_change().rolling_std(20) * math.sqrt(_TRADING_DAYS)
    parkinson = parkinson_vol(bars["high"], bars["low"], window=20)

    assert _std(parkinson) < _std(close_to_close)


def test_rogers_satchell_is_drift_robust() -> None:
    """Adding drift must move Rogers–Satchell far less than Parkinson.

    Compares each estimator against *itself* with and without drift, holding
    everything else fixed.  Measuring against the true volatility instead would
    confound the drift effect with the downward discretisation bias both
    estimators share, and the two happen to partly cancel for Parkinson.
    """
    daily_vol = 0.01
    flat = _make_ohlc(n=4000, seed=4, daily_vol=daily_vol, drift=0.0)
    trending = _make_ohlc(n=4000, seed=4, daily_vol=daily_vol, drift=0.004)

    def mean_of(estimator: str, bars: dict[str, pl.Series]) -> float:
        if estimator == "parkinson":
            series = parkinson_vol(bars["high"], bars["low"], window=250)
        else:
            series = rogers_satchell_vol(
                bars["high"], bars["low"], bars["open"], bars["close"], window=250
            )
        return _mean(series)

    pk_shift = abs(mean_of("parkinson", trending) / mean_of("parkinson", flat) - 1.0)
    rs_shift = abs(mean_of("rs", trending) / mean_of("rs", flat) - 1.0)

    assert rs_shift < 0.01, f"Rogers-Satchell moved {rs_shift:.1%} under drift"
    assert pk_shift > 2.0 * rs_shift


def test_yang_zhang_captures_overnight_gaps() -> None:
    """Yang–Zhang must account for gap variance that range estimators miss.

    CME futures gap across the settlement break, so an estimator blind to gaps
    systematically understates risk.
    """
    bars = _make_ohlc(n=3000, seed=5, daily_vol=0.008, gap_vol=0.010)

    yz = _mean(yang_zhang_vol(bars["high"], bars["low"], bars["open"], bars["close"], window=250))
    pk = _mean(parkinson_vol(bars["high"], bars["low"], window=250))

    # Total variance combines the intraday and overnight components.
    expected_total = math.sqrt(0.008**2 + 0.010**2) * math.sqrt(_TRADING_DAYS)
    assert yz == pytest.approx(expected_total, rel=0.15)
    assert yz > pk * 1.2


def test_yang_zhang_rejects_short_window() -> None:
    """A window below 2 makes the weighting undefined and must raise."""
    bars = _make_ohlc(n=50, seed=6)
    with pytest.raises(ValueError, match="window"):
        yang_zhang_vol(bars["high"], bars["low"], bars["open"], bars["close"], window=1)


# ── Hurst exponent ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("true_hurst", [0.1, 0.3, 0.5, 0.7])
def test_hurst_recovers_known_exponent(true_hurst: float) -> None:
    """The estimator must recover the H it was handed on exact fBm.

    This is the gate: without it, an ``H ≈ 0.1`` reading on market data cannot
    be distinguished from an estimator artefact.
    """
    rng = np.random.default_rng(11)
    fgn = fractional_gaussian_noise(8192, true_hurst, n_paths=1, rng=rng)
    fbm = np.cumsum(fgn[0])

    assert hurst_exponent(fbm) == pytest.approx(true_hurst, abs=0.06)


def test_hurst_of_brownian_motion_is_half() -> None:
    """Ordinary Brownian motion must estimate H ≈ 0.5."""
    rng = np.random.default_rng(12)
    bm = np.cumsum(rng.normal(size=20_000))
    assert hurst_exponent(bm) == pytest.approx(0.5, abs=0.05)


def test_hurst_of_white_noise_is_near_zero() -> None:
    """Uncorrelated noise is maximally rough."""
    rng = np.random.default_rng(13)
    assert hurst_exponent(rng.normal(size=20_000)) < 0.1


def test_hurst_constant_series_returns_half() -> None:
    """A degenerate constant series must not raise or return a wild value."""
    assert hurst_exponent(np.ones(100)) == 0.5


def test_variogram_rejects_short_series() -> None:
    """Too-short inputs must raise rather than return noise."""
    with pytest.raises(ValueError, match="at least 8"):
        variogram(np.arange(4, dtype=float))
    with pytest.raises(ValueError, match="moment"):
        variogram(np.arange(20, dtype=float), moment=0.0)


# ── Jumps ─────────────────────────────────────────────────────────────────────


def test_jump_ratio_near_zero_for_continuous_path() -> None:
    """A pure diffusion must show almost no jump component."""
    rng = np.random.default_rng(14)
    assert jump_ratio(rng.normal(0.0, 0.01, 5000)) < 0.15


def test_jump_ratio_detects_injected_jumps() -> None:
    """Injecting large discrete moves must raise the jump fraction."""
    rng = np.random.default_rng(15)
    returns = rng.normal(0.0, 0.01, 5000)
    continuous = jump_ratio(returns)

    returns[::200] += 0.15  # rare, large jumps
    assert jump_ratio(returns) > continuous + 0.2


def test_bipower_is_more_robust_than_realised_variance() -> None:
    """A single jump must move RV far more than BV."""
    rng = np.random.default_rng(16)
    returns = rng.normal(0.0, 0.01, 2000)
    rv_before, bv_before = realised_variance(returns), bipower_variation(returns)

    returns[1000] += 0.5
    rv_after, bv_after = realised_variance(returns), bipower_variation(returns)

    assert (rv_after - rv_before) > 10.0 * (bv_after - bv_before)


def test_bipower_handles_degenerate_input() -> None:
    """Short or empty inputs must return 0.0 rather than raise."""
    assert bipower_variation(np.array([0.01])) == 0.0
    assert jump_ratio(np.zeros(10)) == 0.0


# ── Ornstein–Uhlenbeck ────────────────────────────────────────────────────────


def test_fit_ou_recovers_simulated_parameters() -> None:
    """Exact-MLE fitting must recover the parameters used to simulate."""
    truth = OUParams(kappa=2.5, theta=0.04, sigma=0.30)
    dt = 1.0 / _TRADING_DAYS
    rng = np.random.default_rng(17)
    path = simulate_ou(truth, n_steps=50_000, dt=dt, n_paths=1, rng=rng)[0]

    fitted = fit_ou(path, dt=dt)
    assert fitted.kappa == pytest.approx(truth.kappa, rel=0.12)
    assert fitted.theta == pytest.approx(truth.theta, abs=0.01)
    assert fitted.sigma == pytest.approx(truth.sigma, rel=0.05)


def test_fit_ou_beats_euler_approximation_at_coarse_sampling() -> None:
    """Exact MLE must be less biased than the Euler/OLS approximation.

    The Euler estimator uses ``kappa ≈ (1 - b) / dt`` instead of
    ``-ln(b) / dt``.  At daily sampling with fast mean reversion the difference
    is a systematic upward bias, which is exactly the regime a half-life
    estimate lives in.
    """
    truth = OUParams(kappa=8.0, theta=0.0, sigma=0.5)
    dt = 1.0 / _TRADING_DAYS
    rng = np.random.default_rng(18)
    path = simulate_ou(truth, n_steps=80_000, dt=dt, n_paths=1, rng=rng)[0]

    exact = fit_ou(path, dt=dt).kappa

    x_prev, x_next = path[:-1], path[1:]
    slope = float(np.cov(x_prev, x_next, ddof=1)[0, 1] / np.var(x_prev, ddof=1))
    euler = (1.0 - slope) / dt

    assert abs(exact - truth.kappa) < abs(euler - truth.kappa)


def test_ou_half_life_matches_kappa() -> None:
    """Half-life must be ln(2)/kappa, and infinite without mean reversion."""
    assert OUParams(kappa=math.log(2.0), theta=0.0, sigma=1.0).half_life == pytest.approx(1.0)
    assert OUParams(kappa=0.0, theta=0.0, sigma=1.0).half_life == math.inf
    assert OUParams(kappa=0.0, theta=0.0, sigma=1.0).stationary_std == math.inf


def test_fit_ou_finds_no_mean_reversion_in_random_walk() -> None:
    """A driftless random walk must yield a half-life of the sample's own order.

    Sampling noise puts the lag-1 slope marginally below 1, so ``kappa`` comes
    out tiny but non-zero rather than exactly zero.  The meaningful assertion
    is that the implied half-life is not short enough to trade — mistaking that
    residual for mean reversion is the classic spurious-cointegration error.
    """
    rng = np.random.default_rng(19)
    fitted = fit_ou(np.cumsum(rng.normal(size=5000)))
    assert fitted.half_life > 100.0


def test_fit_ou_validates_input() -> None:
    """Degenerate inputs must raise."""
    with pytest.raises(ValueError, match="at least 3"):
        fit_ou(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="dt"):
        fit_ou(np.arange(10, dtype=float), dt=0.0)


def test_simulate_ou_matches_stationary_distribution() -> None:
    """Long simulated paths must match the analytic stationary moments."""
    params = OUParams(kappa=3.0, theta=0.05, sigma=0.2)
    rng = np.random.default_rng(20)
    paths = simulate_ou(params, n_steps=2000, dt=1.0 / _TRADING_DAYS, n_paths=400, rng=rng)

    tail = paths[:, 1000:]
    assert float(tail.mean()) == pytest.approx(params.theta, abs=0.005)
    assert float(tail.std()) == pytest.approx(params.stationary_std, rel=0.10)


# ── Heston ────────────────────────────────────────────────────────────────────


def test_heston_variance_reverts_to_theta() -> None:
    """Simulated variance must mean-revert to its long-run level."""
    params = HestonParams(kappa=3.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.16)
    rng = np.random.default_rng(21)
    _, variances = simulate_heston(
        params, n_steps=2000, dt=1.0 / _TRADING_DAYS, n_paths=300, rng=rng
    )
    assert float(variances[:, -500:].mean()) == pytest.approx(params.theta, rel=0.20)


def test_heston_variance_stays_non_negative() -> None:
    """Full truncation must hold even when the Feller condition fails."""
    params = HestonParams(kappa=1.0, theta=0.04, xi=1.2, rho=-0.7, v0=0.04)
    assert not params.feller_satisfied

    rng = np.random.default_rng(22)
    prices, variances = simulate_heston(
        params, n_steps=1500, dt=1.0 / _TRADING_DAYS, n_paths=200, rng=rng
    )
    assert bool(np.all(variances >= 0.0))
    assert bool(np.all(np.isfinite(prices)))
    assert bool(np.all(prices > 0.0))


def test_heston_negative_rho_produces_leverage_skew() -> None:
    """Negative correlation must produce left-skewed returns.

    The leverage effect is the reason rho is strongly negative for equity
    indices; a simulator that does not reproduce it is miswired.
    """
    rng = np.random.default_rng(23)
    common = {"kappa": 2.0, "theta": 0.04, "xi": 0.5, "v0": 0.04}
    dt = 1.0 / _TRADING_DAYS

    down, _ = simulate_heston(
        HestonParams(rho=-0.8, **common), n_steps=2000, dt=dt, n_paths=400, rng=rng
    )
    up, _ = simulate_heston(
        HestonParams(rho=0.8, **common), n_steps=2000, dt=dt, n_paths=400, rng=rng
    )

    def terminal_skew(paths: np.ndarray) -> float:
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        centred = log_returns - log_returns.mean()
        return float(np.mean(centred**3) / np.mean(centred**2) ** 1.5)

    assert terminal_skew(down) < terminal_skew(up)


def test_heston_feller_condition() -> None:
    """The Feller flag must reflect 2*kappa*theta > xi^2."""
    assert HestonParams(kappa=3.0, theta=0.04, xi=0.3, rho=0.0, v0=0.04).feller_satisfied
    assert not HestonParams(kappa=0.5, theta=0.01, xi=0.9, rho=0.0, v0=0.01).feller_satisfied


@pytest.mark.parametrize("bad", [{"n_steps": 0}, {"n_paths": 0}, {"dt": 0.0}])
def test_heston_validates_input(bad: dict[str, float]) -> None:
    """Non-positive simulation arguments must raise."""
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
    kwargs: dict[str, float] = {"n_steps": 100, "dt": 0.01, "n_paths": 1}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        simulate_heston(params, **kwargs)  # type: ignore[arg-type]


# ── Rough Bergomi ─────────────────────────────────────────────────────────────


def test_rough_bergomi_log_variance_is_rough() -> None:
    """Simulated log-variance must exhibit the Hurst exponent it was given.

    Closes the loop end to end: simulator and estimator are independent
    implementations, so agreement is meaningful evidence both are right.
    """
    params = RoughBergomiParams(hurst=0.12, eta=1.5, rho=-0.7, xi0=0.04)
    rng = np.random.default_rng(24)
    _, variances = simulate_rough_bergomi(
        params, n_steps=8000, dt=1.0 / _TRADING_DAYS, n_paths=1, rng=rng
    )
    assert hurst_exponent(np.log(variances[0, 1:])) == pytest.approx(params.hurst, abs=0.07)


def test_rough_bergomi_produces_valid_paths() -> None:
    """Prices must stay positive and finite, variances non-negative."""
    params = RoughBergomiParams(hurst=0.1, eta=1.9, rho=-0.9, xi0=0.04)
    rng = np.random.default_rng(25)
    prices, variances = simulate_rough_bergomi(
        params, n_steps=1000, dt=1.0 / _TRADING_DAYS, n_paths=50, rng=rng
    )
    assert bool(np.all(np.isfinite(prices)))
    assert bool(np.all(prices > 0.0))
    assert bool(np.all(variances >= 0.0))


def test_fgn_has_unit_variance_and_correct_shape() -> None:
    """fGn increments must be standardised, so H is the only free parameter."""
    rng = np.random.default_rng(26)
    fgn = fractional_gaussian_noise(4096, 0.3, n_paths=8, rng=rng)
    assert fgn.shape == (8, 4096)
    assert float(fgn.std()) == pytest.approx(1.0, rel=0.15)


def test_fgn_validates_input() -> None:
    """Out-of-range Hurst values and sizes must raise."""
    with pytest.raises(ValueError, match="hurst"):
        fractional_gaussian_noise(100, 0.0)
    with pytest.raises(ValueError, match="hurst"):
        fractional_gaussian_noise(100, 1.0)
    with pytest.raises(ValueError, match="n must be positive"):
        fractional_gaussian_noise(0, 0.3)
