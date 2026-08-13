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
  distributions (lower is better).
* :func:`pinball_loss` — quantile loss, for when only specific tail quantiles
  matter (e.g. sizing against a 5% VaR).
* :func:`pit_values` / :func:`calibration_error` — Probability Integral
  Transform.  If the forecast distribution is correct, PIT values are uniform;
  deviation from uniformity is exactly miscalibration.
* :func:`coverage` — realised hit rate of a central prediction interval.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "crps_ensemble",
    "crps_gaussian",
    "pinball_loss",
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
    z = (obs - mu) / safe_sigma
    score = safe_sigma * (
        z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)
    )
    return np.where(sigma > 0.0, score, np.abs(obs - mu))


def crps_ensemble(observations: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Return the CRPS of a Monte-Carlo forecast ensemble.

    This is the estimator to use for a neural SDE, whose forecast distribution
    is only available as simulated paths.  Uses the energy form

    .. math::

        \\mathrm{CRPS} = E|X - y| - \\tfrac{1}{2} E|X - X'|

    computed exactly from the sorted sample, which costs
    :math:`O(m \\log m)` per observation rather than the :math:`O(m^2)` of the
    naive pairwise expansion.

    Args:
        observations: Realised values, shape ``(n,)``.
        samples: Forecast samples, shape ``(n, m)`` — ``m`` draws per
            observation.

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

    m = draws.shape[1]
    if m == 0:
        return np.zeros_like(obs)

    ordered = np.sort(draws, axis=1)
    term_obs = np.abs(ordered - obs[:, None]).mean(axis=1)

    # E|X - X'| for a sorted sample equals
    #   (2 / m^2) * sum_i (2i - m + 1) * x_(i)   with i zero-indexed.
    weights = 2.0 * np.arange(m, dtype=np.float64) - m + 1.0
    term_spread = 2.0 * (ordered * weights).sum(axis=1) / (m * m)

    return term_obs - 0.5 * term_spread


def pinball_loss(
    observations: np.ndarray,
    predictions: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Return the pinball (quantile) loss.

    Asymmetric by design: at ``quantile=0.05`` an over-prediction is penalised
    19× more heavily than an under-prediction, which is the right shape when
    the forecast feeds a downside risk limit.

    Args:
        observations: Realised values.
        predictions: Predicted quantile values.
        quantile: Target quantile, strictly between 0 and 1.

    Returns:
        Elementwise loss array.

    Raises:
        ValueError: If *quantile* is not strictly between 0 and 1.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    obs = np.asarray(observations, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    error = obs - pred
    return np.maximum(quantile * error, (quantile - 1.0) * error)


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
