"""Black-76 in JAX: prices, automatic-differentiation Greeks, implied volatility."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from scipy.stats import norm  # noqa: E402

from snippy_scales.research import options as opt  # noqa: E402

F, K, T, VOL, R = 5000.0, 5100.0, 0.25, 0.18, 0.04


def _closed_form(is_call: bool) -> tuple[float, float, float, float]:
    """Price, delta, gamma and vega from the textbook Black-76 formulas."""
    sd = VOL * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    df = np.exp(-R * T)
    if is_call:
        value, delta = df * (F * norm.cdf(d1) - K * norm.cdf(d2)), df * norm.cdf(d1)
    else:
        value, delta = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1)), -df * norm.cdf(-d1)
    return value, delta, df * norm.pdf(d1) / (F * sd), df * F * norm.pdf(d1) * np.sqrt(T)


@pytest.mark.parametrize("is_call", [True, False])
def test_price_and_greeks_match_the_closed_form(is_call: bool) -> None:
    value, delta, gamma, vega = _closed_form(is_call)
    assert float(opt.price(F, K, T, VOL, R, is_call)) == pytest.approx(value, rel=1e-12)
    g = opt.greeks(F, K, T, VOL, R, is_call)
    assert float(g.delta) == pytest.approx(delta, rel=1e-10)
    assert float(g.gamma) == pytest.approx(gamma, rel=1e-10)
    assert float(g.vega) == pytest.approx(vega, rel=1e-10)


def test_put_call_parity() -> None:
    strikes = np.linspace(4000.0, 6000.0, 21)
    call = np.asarray(opt.price(F, strikes, T, VOL, R, True))
    put = np.asarray(opt.price(F, strikes, T, VOL, R, False))
    np.testing.assert_allclose(call - put, np.exp(-R * T) * (F - strikes), rtol=1e-10, atol=1e-9)


def test_theta_is_the_price_change_over_time() -> None:
    day = 1.0 / 365.0
    theta = float(opt.greeks(F, K, T, VOL, R, True).theta)
    decay = float(opt.price(F, K, T - day, VOL, R, True) - opt.price(F, K, T, VOL, R, True))
    assert theta < 0  # a long option loses value as time passes
    assert theta * day == pytest.approx(decay, rel=2e-2)


def test_greeks_broadcast_over_a_chain() -> None:
    strikes = np.array([4800.0, 5000.0, 5200.0])
    calls = np.array([True, False, True])
    g = opt.greeks(F, strikes, T, VOL, R, calls)
    assert g.delta.shape == (3,)
    assert float(g.delta[0]) > 0 > float(g.delta[1])  # call, put
    assert np.all(np.asarray(g.gamma) > 0) and np.all(np.asarray(g.vega) > 0)


def test_implied_vol_recovers_the_volatility_across_a_chain() -> None:
    strikes = np.linspace(3500.0, 6500.0, 31)
    vols = 0.15 + 0.1 * ((strikes - F) / F) ** 2 + 0.05 * (F - strikes).clip(0) / F  # a skew
    calls = strikes >= F  # out-of-the-money options, as a surface would use
    premiums = opt.price(F, strikes, T, vols, R, calls)
    np.testing.assert_allclose(opt.implied_vol(premiums, F, strikes, T, R, calls), vols, atol=1e-8)


def test_implied_vol_is_nan_for_impossible_premiums() -> None:
    intrinsic = np.exp(-R * T) * (F - 4000.0)
    below = opt.implied_vol(intrinsic - 1.0, F, 4000.0, T, R, True)  # under intrinsic value
    above = opt.implied_vol(F * 2.0, F, 5000.0, T, R, True)  # worth more than the future
    assert np.isnan(float(below)) and np.isnan(float(above))
