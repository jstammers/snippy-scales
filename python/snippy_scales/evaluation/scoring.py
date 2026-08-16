"""Scoring rules for probabilistic forecasts.

An SDE — neural or classical — does not emit a point forecast.  It emits a
*distribution*, and its quality has to be judged as a distribution.  Two
questions must be kept apart:

1. **Is the model's distribution any good?**  Answered here, by proper scoring
   rules and calibration diagnostics.
2. **Does a strategy built on it make money?**  Answered by the backtest.

Conflating them is the standard way to waste months: a model can produce a
beautifully calibrated conditional variance and still lose money after costs,
and a badly calibrated model can look profitable on one lucky path.  Scoring
the forecast directly gives a signal with far more statistical power than P&L,
because every bar contributes an observation rather than every trade.

Provided:

* :func:`crps_ensemble` / :func:`crps_gaussian` — Continuous Ranked
  Probability Score, a *proper* scoring rule that generalises MAE to
  distributions (lower is better).  Thin wrappers over ``scoringrules``.
* :func:`pit_values` / :func:`calibration_error` — Probability Integral
  Transform.  If the forecast distribution is correct, PIT values are uniform;
  deviation from uniformity is exactly miscalibration.
* :func:`coverage` — realised hit rate of a central prediction interval.

Deliberately *not* provided:

* **Quantile / pinball loss** — use :func:`sklearn.metrics.mean_pinball_loss`,
  which scikit-learn already ships and this project already depends on.
* **Rank histograms, threshold-weighted and multivariate scores** — reach for
  ``scoringrules`` directly; it carries the full catalogue from R's
  ``scoringRules``.

The CRPS wrappers are kept as named functions rather than inlined call sites so
that the argument-order and axis conventions live in one place, and so the
rationale above stays attached to the code that uses it.
"""

from __future__ import annotations

import numpy as np
import scoringrules as sr
from scipy import stats

__all__ = [
    "crps_ensemble",
    "crps_gaussian",
    "pit_values",
    "calibration_error",
    "coverage",
]


def crps_gaussian(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Return the CRPS of a Gaussian forecast, in closed form.

    For a normal forecast :math:`N(\\mu, \\sigma^2)` and observation :math:`y`,
    with :math:`z = (y - \\mu)/\\sigma`:

    .. math::

        \\mathrm{CRPS} = \\sigma \\left[ z(2\\Phi(z) - 1)
                        + 2\\phi(z) - \\pi^{-1/2} \\right]

    Lower is better.  CRPS has the same units as the observation, and for a
    point forecast (``std → 0``) it reduces to absolute error — which makes it
    directly comparable against an MAE baseline.

    Delegates to :func:`scoringrules.crps_normal`; the degenerate ``std <= 0``
    case is handled here, since a zero-width forecast is a point forecast and
    should score as plain absolute error rather than divide by zero.

    Args:
        observations: Realised values.
        mean: Forecast means, broadcastable against *observations*.
        std: Forecast standard deviations.  Non-positive entries are treated as
            a degenerate point forecast and score as absolute error.

    Returns:
        Elementwise CRPS array.
    """
    obs = np.asarray(observations, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    sigma = np.asarray(std, dtype=np.float64)

    safe_sigma = np.where(sigma > 0.0, sigma, 1.0)
    score = np.asarray(sr.crps_normal(obs, mu, safe_sigma), dtype=np.float64)
    return np.where(sigma > 0.0, score, np.abs(obs - mu))


def crps_ensemble(observations: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Return the CRPS of a Monte-Carlo forecast ensemble.

    This is the estimator to use for a neural SDE, whose forecast distribution
    is only available as simulated paths.  Delegates to
    :func:`scoringrules.crps_ensemble`.

    Args:
        observations: Realised values, shape ``(n,)``.
        samples: Forecast samples, shape ``(n, m)`` — ``m`` draws per
            observation.  The sample axis is the last one.

    Returns:
        CRPS array of shape ``(n,)``.

    Raises:
        ValueError: If *samples* is not 2-D, or its leading dimension does not
            match *observations*.
    """
    obs = np.asarray(observations, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)

    if draws.ndim != 2:
        raise ValueError(f"samples must be 2-D (n, m), got shape {draws.shape}")
    if draws.shape[0] != obs.shape[0]:
        raise ValueError(f"samples has {draws.shape[0]} rows but observations has {obs.shape[0]}")

    if draws.shape[1] == 0:
        return np.zeros_like(obs)

    return np.asarray(sr.crps_ensemble(obs, draws, m_axis=-1), dtype=np.float64)


def pit_values(observations: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Return Probability Integral Transform values for an ensemble forecast.

    Each value is the fraction of forecast samples falling at or below the
    realisation.  **If the forecast distribution is correct these are uniform
    on [0, 1]** — so a histogram of PIT values is the single most informative
    diagnostic available for a generative model:

    * U-shaped → forecast is over-confident (intervals too narrow)
    * hump-shaped → under-confident (intervals too wide)
    * sloped → biased

    Args:
        observations: Realised values, shape ``(n,)``.
        samples: Forecast samples, shape ``(n, m)``.

    Returns:
        PIT values in ``[0, 1]``, shape ``(n,)``.

    Raises:
        ValueError: If *samples* is not 2-D or misaligned with *observations*.
    """
    obs = np.asarray(observations, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)

    if draws.ndim != 2:
        raise ValueError(f"samples must be 2-D (n, m), got shape {draws.shape}")
    if draws.shape[0] != obs.shape[0]:
        raise ValueError(f"samples has {draws.shape[0]} rows but observations has {obs.shape[0]}")

    return (draws <= obs[:, None]).mean(axis=1)


def calibration_error(pit: np.ndarray) -> float:
    """Return the Kolmogorov–Smirnov distance of PIT values from uniformity.

    A single scalar summarising a PIT histogram: ``0.0`` is perfect
    calibration, larger is worse.  Useful as an early-stopping or
    model-selection criterion where a histogram cannot be eyeballed.

    Args:
        pit: PIT values in ``[0, 1]``, typically from :func:`pit_values`.

    Returns:
        KS statistic in ``[0, 1]``; ``0.0`` for an empty input.
    """
    values = np.asarray(pit, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(stats.kstest(values, "uniform").statistic)


def coverage(
    observations: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Return the fraction of observations inside a prediction interval.

    Compare against the interval's nominal level: a 90% interval that covers
    70% of realisations is badly over-confident, which in a sizing model means
    systematically over-levered positions.

    Args:
        observations: Realised values.
        lower: Interval lower bounds.
        upper: Interval upper bounds.

    Returns:
        Empirical coverage in ``[0, 1]``; ``0.0`` for an empty input.
    """
    obs = np.asarray(observations, dtype=np.float64)
    if obs.size == 0:
        return 0.0
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    return float(np.mean((obs >= lo) & (obs <= hi)))
