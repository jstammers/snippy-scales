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

import numpy as np

__all__ = [
    "hurst_exponent",
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
