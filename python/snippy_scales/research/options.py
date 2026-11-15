"""Black-76 pricing of options on futures in JAX: prices, Greeks, implied volatility.

Black (1976) prices a European option on a futures price :math:`F` with strike
:math:`K`, time to expiry :math:`T` (years), volatility :math:`\\sigma` and
continuously compounded rate :math:`r`:

.. math::

    C = e^{-rT}\\,[F\\,\\Phi(d_1) - K\\,\\Phi(d_2)], \\qquad
    P = e^{-rT}\\,[K\\,\\Phi(-d_2) - F\\,\\Phi(-d_1)], \\qquad
    d_{1,2} = \\frac{\\ln(F/K) \\pm \\tfrac12\\sigma^2 T}{\\sigma\\sqrt{T}}.

Everything is a pure, jit-compiled function of arrays that broadcast against each
other, so a whole chain is priced in one call. The Greeks are not typed in: they are
derivatives of :func:`price` taken by automatic differentiation, so they cannot
disagree with it.

CME's quarterly options on equity-index futures are American. With no dividend on a
future and a short time to expiry, the early-exercise premium is a few basis points
of the price, so Black-76 is a close approximation for them; it is not for
deep-in-the-money options when rates are high.

Requires the ``options`` extra (``uv sync --extra options``). Importing this module
switches JAX to 64-bit floats, which implied volatilities need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 — JAX's own flag

_VOL_LO, _VOL_HI = 1e-4, 5.0
_BISECTIONS = 60  # halves [1e-4, 5] to below 1e-17


@jax.jit
def price(  # noqa: PLR0913 — the six Black-76 inputs
    forward: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    vol: ArrayLike,
    rate: ArrayLike = 0.0,
    is_call: ArrayLike = True,  # noqa: FBT002 — an array of booleans, one per option
) -> Array:
    """Black-76 price.

    Args:
        forward: Futures price.
        strike: Strike.
        expiry: Time to expiry in years (positive).
        vol: Annualised volatility.
        rate: Continuously compounded discount rate.
        is_call: ``True`` for a call, ``False`` for a put.
    """
    forward, strike, expiry, vol = (
        jnp.asarray(x, jnp.float64) for x in (forward, strike, expiry, vol)
    )
    total = vol * jnp.sqrt(expiry)
    d1 = (jnp.log(forward / strike) + 0.5 * total**2) / total
    d2 = d1 - total
    call = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    put = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    return jnp.exp(-jnp.asarray(rate) * expiry) * jnp.where(jnp.asarray(is_call), call, put)


class Greeks(NamedTuple):
    """Sensitivities of the Black-76 price, per unit of each input.

    Attributes:
        delta: To the futures price.
        gamma: Of delta to the futures price.
        vega: To volatility (per 1.00 of volatility; divide by 100 for a vol point).
        theta: To the passage of time, per year (negative for a long option; divide
            by 365 for a calendar day).
    """

    delta: Array
    gamma: Array
    vega: Array
    theta: Array


def _scalar(  # noqa: PLR0913
    forward: Array, strike: Array, expiry: Array, vol: Array, rate: Array, is_call: Array
) -> Array:
    return price(forward, strike, expiry, vol, rate, is_call)


_delta = jax.grad(_scalar, argnums=0)
_gamma = jax.grad(_delta, argnums=0)
_vega = jax.grad(_scalar, argnums=3)
_dexpiry = jax.grad(_scalar, argnums=2)


@jax.jit
def greeks(  # noqa: PLR0913
    forward: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    vol: ArrayLike,
    rate: ArrayLike = 0.0,
    is_call: ArrayLike = True,  # noqa: FBT002
) -> Greeks:
    """Delta, gamma, vega and theta by automatic differentiation of :func:`price`."""
    args = jnp.broadcast_arrays(
        *(jnp.asarray(x, jnp.float64) for x in (forward, strike, expiry, vol, rate)),
        jnp.asarray(is_call),
    )
    shape = args[0].shape
    flat = [a.ravel() for a in args]

    def each(fn):  # noqa: ANN001, ANN202 — a jax.grad of _scalar
        return jax.vmap(fn)(*flat).reshape(shape)

    return Greeks(each(_delta), each(_gamma), each(_vega), -each(_dexpiry))


@jax.jit
def implied_vol(  # noqa: PLR0913
    premium: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    rate: ArrayLike = 0.0,
    is_call: ArrayLike = True,  # noqa: FBT002
) -> Array:
    """The volatility at which :func:`price` equals *premium*, by bisection.

    Bisection is used, not Newton's method: the price is monotone in volatility, so
    it cannot diverge on deep out-of-the-money options where vega is almost zero.

    Returns:
        Implied volatility, ``NaN`` where *premium* is outside the prices that
        volatilities between 0.01% and 500% can produce (below intrinsic value, or
        absurdly high).
    """
    premium = jnp.asarray(premium, jnp.float64)
    lo = jnp.full(
        jnp.broadcast_shapes(premium.shape, jnp.shape(forward), jnp.shape(strike)), _VOL_LO
    )
    hi = jnp.full(lo.shape, _VOL_HI)
    bad = (premium < price(forward, strike, expiry, lo, rate, is_call)) | (
        premium > price(forward, strike, expiry, hi, rate, is_call)
    )

    def step(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        above = price(forward, strike, expiry, mid, rate, is_call) > premium
        return jnp.where(above, lo, mid), jnp.where(above, mid, hi)

    lo, hi = jax.lax.fori_loop(0, _BISECTIONS, step, (lo, hi))
    return jnp.where(bad, jnp.nan, 0.5 * (lo + hi))
