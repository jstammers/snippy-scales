"""Roughness and jump diagnostics for volatility paths.

Before fitting any stochastic-volatility model — classical or neural — it is
worth asking whether the data supports the model class at all.  These are the
cheap tests that answer that, and they should be run *before* a deep-learning
dependency is added to the project.

**Hurst exponent.**  The rough-volatility literature (Gatheral, Jaisson and
Rosenbaum, *Volatility is rough*) reports that log-realised-variance behaves
like fractional Brownian motion with :math:`H \\approx 0.1`, far below the
:math:`H = 0.5` of a standard diffusion.  If a series really is that rough,
classical Markovian models (Heston, SABR, GARCH) are structurally misspecified
and a fractional or neural model can do better.  If the estimate comes out
near 0.5, that motivation evaporates and the simpler model should win.

:func:`hurst_exponent` estimates :math:`H` from the log-log slope of the
**variogram** (structure function)

.. math::

    m(\\Delta) = E\\left[|X_{t+\\Delta} - X_t|^q\\right] \\propto \\Delta^{qH}

which is the estimator used in that literature and is robust to the level of
the series.

**Jumps.**  Bipower variation is robust to jumps while realised variance is
not, so their ratio isolates the jump component.  A series dominated by jumps
is badly served by a continuous SDE no matter how flexible its drift and
diffusion, so this is the second gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

__all__ = [
    "hurst_exponent",
    "hurst_with_nugget",
    "RoughnessEstimate",
    "variogram",
    "default_lags",
    "realised_variance",
    "bipower_variation",
    "jump_ratio",
]

#: Scaling constant for bipower variation: mu_1^{-2} = pi/2, where
#: mu_1 = E|Z| = sqrt(2/pi) for a standard normal.  This normalises BV so that
#: BV/RV -> 1 for a jump-free path.
_BIPOWER_SCALE = np.pi / 2.0

#: Upper bound on the lags used for the default variogram grid.
#:
#: The variogram of a *finite* sample saturates at long lags — there are only
#: ~n/lag effectively independent differences available — which biases the
#: log-log slope, and hence H, **downward**.  The bias is severe: on exact fBm
#: of length 8192, allowing lags up to n/4 recovers H = 0.7 as 0.61, while
#: capping at 64 recovers 0.70.  Empirically 32-128 is a flat optimum across
#: H in [0.1, 0.9], so the cap is fixed rather than scaled with n.
_MAX_DEFAULT_LAG = 64


def default_lags(n: int) -> np.ndarray:
    """Return the default log-spaced lag grid for a series of length *n*.

    Args:
        n: Number of observations in the series.

    Returns:
        Sorted unique integer lags, log-spaced from 1 up to
        ``min(n // 4, 64)`` — see :data:`_MAX_DEFAULT_LAG` for why the cap
        matters.
    """
    max_lag = max(2, min(n // 4, _MAX_DEFAULT_LAG))
    return np.unique(np.round(np.logspace(0.0, np.log10(max_lag), 20)).astype(int))


def variogram(
    series: np.ndarray,
    lags: np.ndarray | None = None,
    moment: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical variogram (structure function) of *series*.

    Args:
        series: 1-D path, typically log realised variance.
        lags: Lags at which to evaluate.  Defaults to a log-spaced grid from
            1 to ``len(series) // 4``.
        moment: Absolute moment order :math:`q`.  ``2.0`` gives the classical
            variogram; lower orders are more robust to outliers.

    Returns:
        Tuple ``(lags, values)`` where ``values[i]`` is
        :math:`E[|X_{t+\\Delta_i} - X_t|^q]`.

    Raises:
        ValueError: If *series* has fewer than 8 points or *moment* is not
            positive.
    """
    path = np.asarray(series, dtype=np.float64)
    if path.size < 8:
        raise ValueError(f"series must have at least 8 points, got {path.size}")
    if moment <= 0.0:
        raise ValueError(f"moment must be positive, got {moment}")

    if lags is None:
        lags = default_lags(path.size)
    lag_array = np.asarray(lags, dtype=int)
    lag_array = lag_array[(lag_array >= 1) & (lag_array < path.size)]

    values = np.array(
        [np.mean(np.abs(path[lag:] - path[:-lag]) ** moment) for lag in lag_array],
        dtype=np.float64,
    )
    return lag_array, values


def hurst_exponent(
    series: np.ndarray,
    lags: np.ndarray | None = None,
    moment: float = 2.0,
) -> float:
    """Estimate the Hurst exponent from the variogram's log-log slope.

    Fits :math:`\\log m(\\Delta) = qH \\log \\Delta + c` by least squares and
    returns :math:`H`.

    Interpretation:

    * :math:`H \\approx 0.5` — standard Brownian motion; a Markovian diffusion
      is adequate.
    * :math:`H < 0.5` — rough / anti-persistent.  Log-volatility of liquid
      futures typically lands near 0.1, which is the empirical case for rough
      volatility models.
    * :math:`H > 0.5` — persistent, long-memory behaviour.

    Apply this to **log realised variance**, not to prices: it is the
    volatility path whose roughness is at issue.

    .. warning::
       The estimate is sensitive to the lag range.  Long lags bias H
       *downward* because the finite-sample variogram saturates, so a naive
       grid running to ``n/4`` can turn a true 0.7 into 0.61 — and, more
       dangerously here, can manufacture an apparently "rough" reading from a
       series that is not.  :func:`default_lags` caps the grid accordingly.
       When passing *lags* explicitly, check the answer's stability across
       ranges before believing it.

    Args:
        series: 1-D path to analyse.
        lags: Lags passed to :func:`variogram`.  Defaults to
            :func:`default_lags`.
        moment: Absolute moment order used in the variogram.

    Returns:
        Estimated Hurst exponent.  Returns ``0.5`` when the variogram is
        degenerate (constant series), since that carries no roughness
        information either way.

    Raises:
        ValueError: If *series* is too short or *moment* is not positive.
    """
    lag_array, values = variogram(series, lags=lags, moment=moment)

    positive = values > 0.0
    if np.count_nonzero(positive) < 2:
        return 0.5

    log_lags = np.log(lag_array[positive].astype(np.float64))
    log_values = np.log(values[positive])
    slope = float(np.polyfit(log_lags, log_values, 1)[0])
    return slope / moment


@dataclass(frozen=True)
class RoughnessEstimate:
    """Result of a nugget-aware Hurst fit.

    Attributes:
        hurst: Estimated Hurst exponent of the underlying signal, with the
            measurement-noise contribution separated out.
        noise_std: Estimated standard deviation of the additive observation
            noise, in the units of the input series.
        signal_scale: Fitted variogram scale ``c`` in
            ``m(d) = c * d^(2H) + 2 * noise_std^2``.
        noise_share: Fraction of the variogram at the shortest lag explained by
            noise rather than signal.  Above ~0.5 the Hurst estimate is barely
            identified and should not be trusted.
        converged: Whether the non-linear fit converged.
    """

    hurst: float
    noise_std: float
    signal_scale: float
    noise_share: float
    converged: bool


def hurst_with_nugget(
    series: np.ndarray,
    lags: np.ndarray | None = None,
) -> RoughnessEstimate:
    """Estimate the Hurst exponent while separating measurement noise.

    :func:`hurst_exponent` fits a straight line to the log-log variogram, which
    silently assumes the series is observed *exactly*.  Realised variance never
    is — it is an estimate from a finite number of intraday returns, and a
    range-based daily proxy is noisier still.  That estimation noise is white,
    so it adds a constant to the variogram:

    .. math::

        m(\\Delta) = c\\,\\Delta^{2H} + 2\\sigma_\\varepsilon^2

    The constant dominates at short lags, flattening the log-log slope and
    dragging :math:`H` toward zero — **toward the "rough" verdict**.  The
    effect is not subtle: an exactly-Markovian :math:`H = 0.5` path observed
    with noise of one tenth the signal's standard deviation estimates as
    :math:`H \\approx 0.07` under the log-log slope, which is indistinguishable
    from the canonical rough-volatility finding.

    This function fits the three parameters :math:`(c, H, \\sigma_\\varepsilon)`
    directly by non-linear least squares on the variogram, so the noise floor is
    absorbed by the nugget term instead of corrupting the exponent.

    This is the estimator to use for any gating decision.  Compare its output
    against :func:`hurst_exponent` — a large gap between them means the answer
    is being driven by observation noise, not by the volatility path.

    Args:
        series: 1-D path to analyse, typically log realised variance.
        lags: Lags passed to :func:`variogram`.  Defaults to
            :func:`default_lags`.

    Returns:
        A :class:`RoughnessEstimate`.  On a failed fit, ``converged`` is
        ``False`` and ``hurst`` falls back to the log-log slope.

    Raises:
        ValueError: If *series* is too short for a variogram.
    """
    lag_array, values = variogram(series, lags=lags, moment=2.0)

    positive = values > 0.0
    if np.count_nonzero(positive) < 4:
        fallback = hurst_exponent(series, lags=lags)
        return RoughnessEstimate(
            hurst=fallback,
            noise_std=0.0,
            signal_scale=0.0,
            noise_share=0.0,
            converged=False,
        )

    x = lag_array[positive].astype(np.float64)
    y = values[positive]

    def model(params: np.ndarray) -> np.ndarray:
        scale, hurst, noise_var = params
        return scale * x ** (2.0 * hurst) + 2.0 * noise_var

    def residual(params: np.ndarray) -> np.ndarray:
        # Fit in log space: the variogram spans orders of magnitude across
        # lags, and an unweighted linear fit would be dominated by the
        # longest lags — the noisiest and least informative part.
        return np.log(np.maximum(model(params), 1e-300)) - np.log(y)

    # Seed from the naive slope so the optimiser starts in the right basin.
    naive_h = float(np.clip(hurst_exponent(series, lags=lags), 0.01, 0.99))
    seed = np.array([max(y[0], 1e-12), naive_h, y[0] * 0.25])

    try:
        fit = optimize.least_squares(
            residual,
            seed,
            bounds=(np.array([1e-15, 0.001, 0.0]), np.array([np.inf, 0.999, np.inf])),
            max_nfev=2000,
        )
    except (ValueError, RuntimeError):
        return RoughnessEstimate(
            hurst=naive_h,
            noise_std=0.0,
            signal_scale=0.0,
            noise_share=0.0,
            converged=False,
        )

    scale, hurst, noise_var = (float(v) for v in fit.x)
    noise_term = 2.0 * noise_var
    shortest = scale * float(x[0]) ** (2.0 * hurst) + noise_term
    noise_share = noise_term / shortest if shortest > 0.0 else 0.0

    return RoughnessEstimate(
        hurst=hurst,
        noise_std=float(np.sqrt(max(noise_var, 0.0))),
        signal_scale=scale,
        noise_share=float(np.clip(noise_share, 0.0, 1.0)),
        converged=bool(fit.success),
    )


def realised_variance(returns: np.ndarray) -> float:
    """Return realised variance — the sum of squared returns.

    Args:
        returns: Intraperiod returns.

    Returns:
        Sum of squared returns.  Includes any jump contribution.
    """
    r = np.asarray(returns, dtype=np.float64)
    return float(np.sum(r**2))


def bipower_variation(returns: np.ndarray) -> float:
    """Return realised bipower variation — a jump-robust variance estimate.

    .. math::

        BV = \\frac{2}{\\pi} \\sum_{i=2}^{n} |r_{i-1}| \\, |r_i|

    Multiplying *adjacent* absolute returns means a single large jump enters
    only linearly rather than quadratically, so ``BV`` estimates the continuous
    part of quadratic variation while
    :func:`realised_variance` estimates the total.

    Args:
        returns: Intraperiod returns.

    Returns:
        Bipower variation; ``0.0`` for fewer than two returns.
    """
    r = np.abs(np.asarray(returns, dtype=np.float64))
    if r.size < 2:
        return 0.0
    return float(_BIPOWER_SCALE * np.sum(r[:-1] * r[1:]))


def jump_ratio(returns: np.ndarray) -> float:
    """Return the fraction of quadratic variation attributable to jumps.

    .. math::

        J = \\max\\left(0, \\; 1 - BV / RV\\right)

    A value near ``0`` means the path is essentially continuous and a diffusion
    model is appropriate.  A large value means jumps dominate, and no amount of
    flexibility in a continuous SDE's drift and diffusion will capture the
    dynamics — a jump-diffusion or Hawkes-type model is needed instead.

    Args:
        returns: Intraperiod returns.

    Returns:
        Jump fraction in ``[0, 1]``; ``0.0`` when realised variance is zero.
    """
    rv = realised_variance(returns)
    if rv <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - bipower_variation(returns) / rv))
