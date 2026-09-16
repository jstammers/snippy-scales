"""Selection-bias corrections for backtest performance statistics.

A parameter sweep does not just find the best strategy — it finds the
luckiest one.  Searching :math:`N` configurations and reporting the maximum
in-sample Sharpe produces an inflated number even when every configuration is
worthless, because the maximum of :math:`N` draws from a zero-mean distribution
grows like :math:`\\sqrt{2 \\ln N}`.  With a 100-point grid that expected
maximum is roughly 3 standard errors above zero.

This module implements the standard corrections:

* :func:`expected_max_sharpe` — the Sharpe a *null* strategy is expected to
  achieve as the best of ``n_trials``.  The number your result must beat.
* :func:`deflated_sharpe_ratio` — Bailey & López de Prado's Deflated Sharpe
  Ratio: the probability the true Sharpe is positive, after adjusting for
  trial count, non-normality (skew and kurtosis), and sample length.
* :func:`probability_of_backtest_overfitting` — CSCV: the probability that the
  configuration selected in-sample ranks below median out-of-sample.
* :func:`min_track_record_length` — how many observations are needed before a
  Sharpe is statistically distinguishable from a benchmark.

References:
    Bailey, D. and López de Prado, M. (2014). *The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
    Journal of Portfolio Management.

    Bailey, D., Borwein, J., López de Prado, M. and Zhu, Q. (2016).
    *The Probability of Backtest Overfitting*. Journal of Computational Finance.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

__all__ = [
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "min_track_record_length",
]

#: Euler–Mascheroni constant, used in the expected-maximum approximation.
_EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, trial_std: float = 1.0) -> float:
    """Return the expected maximum Sharpe across *n_trials* worthless strategies.

    Uses the standard extreme-value approximation for the maximum of ``N``
    independent standard normals:

    .. math::

        E[\\max] \\approx \\sigma \\left[ (1-\\gamma) Z^{-1}(1 - 1/N)
                        + \\gamma Z^{-1}(1 - 1/(Ne)) \\right]

    where :math:`\\gamma` is the Euler–Mascheroni constant and :math:`Z^{-1}`
    is the inverse standard normal CDF.

    This is the benchmark a sweep result must clear before it means anything:
    a grid of 100 configurations produces a best-in-sweep Sharpe around 2.5
    standard errors above zero **by construction**.

    Args:
        n_trials: Number of independent configurations searched.  Values below
            ``1`` are treated as ``1``.
        trial_std: Standard deviation of the Sharpe estimates across trials.
            The default of ``1.0`` expresses the answer in standard errors.

    Returns:
        Expected maximum Sharpe under the null hypothesis of no skill.
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0

    quantile_a = stats.norm.ppf(1.0 - 1.0 / n)
    quantile_b = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(
        trial_std * ((1.0 - _EULER_MASCHERONI) * quantile_a + _EULER_MASCHERONI * quantile_b)
    )


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_obs: int,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: float = 1.0,
) -> float:
    """Return the probability that the true Sharpe exceeds *benchmark_sharpe*.

    The Probabilistic Sharpe Ratio corrects the naive standard error
    :math:`1/\\sqrt{n}` for the non-normality of returns.  Negative skew and
    fat tails both *widen* the confidence interval, so a strategy with crash
    risk needs a higher observed Sharpe to reach the same confidence.

    .. important::
       The underlying formula is defined on **per-observation** Sharpe, not
       annualised Sharpe.  Pass *periods_per_year* whenever the Sharpe
       arguments are annualised (``252`` for daily bars) and they will be
       de-annualised internally.  Getting this wrong inflates confidence
       enormously — an annualised Sharpe fed in as if per-period makes a
       13-day track record look conclusive.

    Args:
        observed_sharpe: Sharpe estimated from the sample.
        n_obs: Number of return observations.
        benchmark_sharpe: Threshold to beat (``0.0`` tests for any skill).
        skew: Sample skewness of returns.
        kurtosis: Sample kurtosis of returns (``3.0`` is the normal value).
        periods_per_year: Annualisation factor already applied to the Sharpe
            arguments.  ``1.0`` means they are already per-observation.

    Returns:
        Probability in ``[0, 1]``.  Returns ``0.0`` when ``n_obs < 2``.

    Raises:
        ValueError: If *periods_per_year* is not positive.
    """
    if periods_per_year <= 0.0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    if n_obs < 2:
        return 0.0

    scale = math.sqrt(periods_per_year)
    observed = observed_sharpe / scale
    benchmark = benchmark_sharpe / scale

    variance = 1.0 - skew * observed + 0.25 * (kurtosis - 1.0) * observed**2
    if variance <= 0.0:
        return 0.0

    statistic = (observed - benchmark) * math.sqrt(n_obs - 1)
    return float(stats.norm.cdf(statistic / math.sqrt(variance)))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_obs: int,
    trial_std: float = 1.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: float = 1.0,
) -> float:
    """Return the Deflated Sharpe Ratio for a best-in-sweep result.

    The DSR is the Probabilistic Sharpe Ratio benchmarked against
    :func:`expected_max_sharpe` rather than against zero.  It answers the
    question that matters after a parameter search: *given that I searched
    this hard, is this result still evidence of skill?*

    Interpretation: a DSR above 0.95 is the conventional bar for treating a
    swept result as more than selection noise.  A DSR near 0.5 means the
    result is indistinguishable from the luckiest of N coin flips.

    ``observed_sharpe`` and ``trial_std`` must be expressed in the same units.
    When those units are annualised, pass *periods_per_year* so both are
    de-annualised consistently.

    Args:
        observed_sharpe: Best Sharpe found across the sweep.
        n_trials: Number of configurations searched.
        n_obs: Number of return observations backing the winning result.
        trial_std: Standard deviation of Sharpe estimates across the sweep.
            Using the sweep's own dispersion is the standard estimator.
        skew: Sample skewness of the winning strategy's returns.
        kurtosis: Sample kurtosis of the winning strategy's returns.
        periods_per_year: Annualisation factor already applied to
            *observed_sharpe* and *trial_std*.  ``1.0`` means per-observation.

    Returns:
        Probability in ``[0, 1]`` that the true Sharpe is positive after
        deflation.  Returns ``0.0`` when ``n_obs < 2``.
    """
    benchmark = expected_max_sharpe(n_trials, trial_std=trial_std)
    return probabilistic_sharpe_ratio(
        observed_sharpe,
        n_obs=n_obs,
        benchmark_sharpe=benchmark,
        skew=skew,
        kurtosis=kurtosis,
        periods_per_year=periods_per_year,
    )


def min_track_record_length(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: float = 1.0,
) -> float:
    """Return the observations needed to establish skill at *confidence*.

    Answers "how long must I run this before I know it works?".  Pass
    ``periods_per_year=252`` with an annualised Sharpe and the answer comes
    back in trading days; divide by 252 for years — usually a sobering number.
    A Sharpe-0.5 daily strategy needs roughly a decade.

    Args:
        observed_sharpe: Sharpe estimated from the sample.
        benchmark_sharpe: Threshold to beat.
        confidence: Required confidence level, strictly between 0 and 1.
        skew: Sample skewness of returns.
        kurtosis: Sample kurtosis of returns.
        periods_per_year: Annualisation factor already applied to the Sharpe
            arguments.  ``1.0`` means they are already per-observation.

    Returns:
        Required number of observations.  Returns ``inf`` when the observed
        Sharpe does not exceed the benchmark, since no sample size would
        establish skill that is not there.

    Raises:
        ValueError: If *confidence* is not strictly between 0 and 1, or if
            *periods_per_year* is not positive.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if periods_per_year <= 0.0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")

    scale = math.sqrt(periods_per_year)
    observed = observed_sharpe / scale
    excess = (observed_sharpe - benchmark_sharpe) / scale
    if excess <= 0.0:
        return math.inf

    variance = 1.0 - skew * observed + 0.25 * (kurtosis - 1.0) * observed**2
    if variance <= 0.0:
        return math.inf

    z = stats.norm.ppf(confidence)
    return float(1.0 + variance * (z / excess) ** 2)


def probability_of_backtest_overfitting(
    fold_returns: np.ndarray,
    *,
    n_partitions: int = 8,
) -> float:
    """Return the Probability of Backtest Overfitting via CSCV.

    Combinatorially Symmetric Cross-Validation splits the return history into
    ``n_partitions`` blocks and, for every balanced split into in-sample and
    out-of-sample halves, checks whether the configuration that ranked best
    in-sample lands below the *median* out-of-sample.  The fraction of splits
    where it does is the PBO.

    A PBO above ~0.5 means the selection procedure is worse than useless: the
    configuration you would have chosen tends to underperform the median one
    you rejected.

    Args:
        fold_returns: 2-D array shaped ``(n_observations, n_configurations)``
            holding each swept configuration's return series.  At least two
            configurations are required.
        n_partitions: Number of blocks to split the history into.  Must be
            even and at least 4; the number of splits grows as
            ``C(n_partitions, n_partitions/2)``, so values above ~16 get slow.

    Returns:
        Probability in ``[0, 1]``.  Returns ``0.0`` when there are too few
        observations to form the partitions.

    Raises:
        ValueError: If *fold_returns* is not 2-D with at least 2 columns, or
            if *n_partitions* is odd or below 4.
    """
    matrix = np.asarray(fold_returns, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(
            f"fold_returns must be 2-D with at least 2 configurations, got shape {matrix.shape}"
        )
    if n_partitions < 4 or n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even and >= 4, got {n_partitions}")

    n_obs = matrix.shape[0]
    if n_obs < n_partitions:
        return 0.0

    blocks = np.array_split(np.arange(n_obs), n_partitions)
    half = n_partitions // 2
    logits: list[float] = []

    for combo in itertools.combinations(range(n_partitions), half):
        in_sample_idx = np.concatenate([blocks[b] for b in combo])
        out_idx = np.concatenate([blocks[b] for b in range(n_partitions) if b not in combo])

        in_sharpe = _column_sharpe(matrix[in_sample_idx])
        out_sharpe = _column_sharpe(matrix[out_idx])

        best = int(np.argmax(in_sharpe))
        # Relative rank of the in-sample winner within the OOS distribution.
        rank = float(stats.rankdata(out_sharpe)[best]) / (out_sharpe.size + 1)
        logits.append(math.log(rank / (1.0 - rank)))

    if not logits:
        return 0.0
    return float(np.mean(np.asarray(logits) <= 0.0))


def _column_sharpe(matrix: np.ndarray) -> np.ndarray:
    """Return the per-column Sharpe of a return matrix.

    Annualisation is omitted deliberately: PBO depends only on the *ranking*
    of configurations, which any positive scaling leaves unchanged.

    Args:
        matrix: 2-D array shaped ``(n_observations, n_configurations)``.

    Returns:
        1-D array of per-column Sharpe estimates, zero where the column is flat.
    """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    return np.divide(mean, std, out=np.zeros_like(mean), where=std > 0.0)
