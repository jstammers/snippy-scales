"""The roughness gate — decide whether a stochastic-volatility model is warranted.

This is Step 1 of the falsification plan in ``docs/research/sde_market_dynamics.md``,
and it exists to be *failed cheaply*.  It answers one question before any
deep-learning dependency is added to the project:

    Does this instrument's volatility path actually support a rough or
    non-Markovian model, stably, out of sample?

The gate returns one of several verdicts and only one of them authorises further
work.  Three of them are refusals for distinct reasons, which matters because
the remedies differ: unstable parameters mean the model class is wrong, a
noise-dominated estimate means the *data* is inadequate, and jump domination
means a continuous SDE is the wrong object entirely.

.. important::
   **The naive estimator is biased toward passing this gate.**  Realised
   variance is estimated, never observed, and its estimation noise is white —
   which flattens the short-lag variogram and drags the log-log-slope Hurst
   estimate toward zero, i.e. toward "rough".  An exactly Markovian
   :math:`H = 0.5` path observed with noise one tenth of the signal's standard
   deviation reads as :math:`H \\approx 0.07` under the naive estimator.

   A gate that is biased toward authorising the expensive work is worse than no
   gate.  So this module decides on
   :func:`~snippy_scales.research.roughness.hurst_with_nugget`, which fits the
   noise floor explicitly, and refuses to decide at all when the noise share is
   too high to identify the exponent.

Usage::

    from snippy_scales.research.diagnostics import roughness_gate, log_variance_proxy

    log_rv = log_variance_proxy(bars)
    report = roughness_gate(log_rv, returns=returns, symbol="ES.c.0")
    print(report.summary())

    if report.proceed_to_neural:
        ...  # Step 4+ is authorised
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from snippy_scales.research.roughness import (
    hurst_exponent,
    hurst_with_nugget,
    jump_ratio,
)

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "RoughnessVerdict",
    "FoldRoughness",
    "RoughnessReport",
    "GateThresholds",
    "roughness_gate",
    "log_variance_proxy",
]


class RoughnessVerdict(StrEnum):
    """Outcome of the roughness gate.

    Only :attr:`ROUGH` authorises progressing to a fractional or neural model.
    """

    ROUGH = "rough"
    """H is stably well below 0.5.  A rough / non-Markovian model is justified."""

    MARKOVIAN = "markovian"
    """H is stably near 0.5.  Use Heston or HAR; the neural motivation is absent."""

    INTERMEDIATE = "intermediate"
    """H is stable but between the thresholds.  Weak support; prefer the simpler model."""

    UNSTABLE = "unstable"
    """H varies too much across folds.  The data supports no fitted volatility model."""

    NOISE_DOMINATED = "noise_dominated"
    """Observation noise swamps the signal — H is not identified.  Needs better data,
    not a different model."""

    JUMP_DOMINATED = "jump_dominated"
    """Jumps dominate quadratic variation.  No continuous SDE will fit well."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Too few observations to estimate anything per fold."""


@dataclass(frozen=True)
class GateThresholds:
    """Decision thresholds for :func:`roughness_gate`.

    Attributes:
        rough_below: H at or below this counts as rough.
        markovian_above: H at or above this counts as Markovian.
        max_hurst_std: Maximum std-dev of H across folds before the estimate is
            called unstable.
        max_noise_share: Maximum share of the shortest-lag variogram that may be
            explained by observation noise before H is treated as unidentified.
            The default of ``0.5`` is the point at which noise explains more of
            the short-lag variogram than the signal does — past there, the
            exponent is being inferred mostly from measurement error.
        max_jump_fraction: Maximum share of quadratic variation attributable to
            jumps before a continuous SDE is ruled out.
        min_obs_per_fold: Minimum observations required in each fold.
    """

    rough_below: float = 0.25
    markovian_above: float = 0.40
    max_hurst_std: float = 0.10
    max_noise_share: float = 0.50
    max_jump_fraction: float = 0.30
    min_obs_per_fold: int = 250


@dataclass(frozen=True)
class FoldRoughness:
    """Per-fold roughness estimates.

    Attributes:
        fold_idx: Zero-based fold index, chronological.
        n_obs: Observations in this fold.
        hurst: Nugget-aware Hurst estimate — the one decisions are made on.
        hurst_naive: Log-log-slope estimate, retained for comparison.  A large
            gap from *hurst* means observation noise is driving the naive number.
        noise_std: Estimated observation-noise standard deviation.
        noise_share: Share of the shortest-lag variogram explained by noise.
        converged: Whether the nugget fit converged.
    """

    fold_idx: int
    n_obs: int
    hurst: float
    hurst_naive: float
    noise_std: float
    noise_share: float
    converged: bool


@dataclass(frozen=True)
class RoughnessReport:
    """Full gate result for one instrument.

    Attributes:
        symbol: Instrument identifier.
        verdict: The gate's decision.
        rationale: Human-readable explanation of the decision.
        folds: Per-fold estimates, chronological.
        jump_fraction: Share of quadratic variation attributable to jumps.
        thresholds: Thresholds the decision was made against.
    """

    symbol: str
    verdict: RoughnessVerdict
    rationale: str
    folds: list[FoldRoughness] = field(default_factory=list)
    jump_fraction: float = 0.0
    thresholds: GateThresholds = field(default_factory=GateThresholds)

    @property
    def proceed_to_neural(self) -> bool:
        """Whether the gate authorises building a rough or neural model.

        Returns:
            ``True`` only for :attr:`RoughnessVerdict.ROUGH`.
        """
        return self.verdict is RoughnessVerdict.ROUGH

    @property
    def mean_hurst(self) -> float:
        """Mean nugget-aware Hurst across folds; ``nan`` when there are none."""
        values = [f.hurst for f in self.folds]
        return float(np.mean(values)) if values else float("nan")

    @property
    def std_hurst(self) -> float:
        """Std-dev of Hurst across folds; ``nan`` with fewer than two folds."""
        values = [f.hurst for f in self.folds]
        return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

    @property
    def mean_hurst_naive(self) -> float:
        """Mean naive Hurst across folds; ``nan`` when there are none."""
        values = [f.hurst_naive for f in self.folds]
        return float(np.mean(values)) if values else float("nan")

    @property
    def mean_noise_share(self) -> float:
        """Mean noise share across folds; ``nan`` when there are none."""
        values = [f.noise_share for f in self.folds]
        return float(np.mean(values)) if values else float("nan")

    def summary(self) -> str:
        """Return a readable multi-line summary of the gate result.

        Returns:
            Formatted report suitable for printing to a terminal or log.
        """
        lines = [
            f"Roughness gate — {self.symbol}",
            f"  verdict          : {self.verdict.value.upper()}",
            f"  proceed to neural: {'YES' if self.proceed_to_neural else 'NO'}",
            f"  Hurst (nugget)   : {self.mean_hurst:.3f} ± {self.std_hurst:.3f} across "
            f"{len(self.folds)} folds",
            f"  Hurst (naive)    : {self.mean_hurst_naive:.3f}  "
            f"[biased toward 0 by observation noise]",
            f"  noise share      : {self.mean_noise_share:.2f}",
            f"  jump fraction    : {self.jump_fraction:.3f}",
            f"  rationale        : {self.rationale}",
        ]
        if self.folds:
            lines.append("  per fold:")
            for fold in self.folds:
                lines.append(
                    f"    [{fold.fold_idx}] n={fold.n_obs:<6d} "
                    f"H={fold.hurst:.3f} (naive {fold.hurst_naive:.3f})  "
                    f"noise_share={fold.noise_share:.2f}"
                )
        return "\n".join(lines)


def log_variance_proxy(
    bars: pl.DataFrame,
    floor_quantile: float = 0.01,
) -> np.ndarray:
    """Build a per-bar log realised-variance series from OHLC bars.

    Uses the Rogers–Satchell per-bar estimator, which is drift-independent and
    therefore does not confuse a trending market for a volatile one.

    .. warning::
       A **daily range** estimator is a noisy proxy for realised variance.  The
       rough-volatility literature uses 5-minute realised variance, whose
       estimation noise is far smaller.  Expect
       :attr:`FoldRoughness.noise_share` to be high on daily bars — often high
       enough that the gate correctly refuses to decide.  When that happens the
       remedy is higher-frequency data, not a different estimator.

    Args:
        bars: DataFrame with ``open``, ``high``, ``low``, ``close`` columns.
        floor_quantile: Non-positive and near-zero variances are floored at this
            quantile of the positive values before taking logs, so that a single
            zero-range bar cannot produce ``-inf``.

    Returns:
        1-D array of log per-bar variance, same length as *bars*.

    Raises:
        ValueError: If a required column is missing, or no bar has positive
            estimated variance.
    """
    required = ("open", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing required column(s): {', '.join(missing)}")

    o = bars["open"].to_numpy().astype(np.float64)
    h = bars["high"].to_numpy().astype(np.float64)
    low = bars["low"].to_numpy().astype(np.float64)
    c = bars["close"].to_numpy().astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.log(h / c) * np.log(h / o) + np.log(low / c) * np.log(low / o)

    variance = np.where(np.isfinite(variance), variance, 0.0)
    positive = variance[variance > 0.0]
    if positive.size == 0:
        raise ValueError("no bar produced a positive variance estimate")

    floor = float(np.quantile(positive, floor_quantile))
    floor = max(floor, 1e-12)
    return np.log(np.maximum(variance, floor))


def roughness_gate(
    log_rv: np.ndarray,
    *,
    returns: np.ndarray | None = None,
    symbol: str = "UNKNOWN",
    n_folds: int = 5,
    thresholds: GateThresholds | None = None,
) -> RoughnessReport:
    """Run the roughness gate on a log realised-variance series.

    Splits *log_rv* into contiguous chronological folds, estimates the Hurst
    exponent in each with the noise-aware estimator, and decides whether the
    evidence supports a rough or non-Markovian volatility model.

    Decision order matters — each refusal has a different remedy, so the most
    fundamental objection is raised first:

    1. Too little data → :attr:`RoughnessVerdict.INSUFFICIENT_DATA`
    2. Jumps dominate → :attr:`RoughnessVerdict.JUMP_DOMINATED` (a continuous
       SDE is the wrong object)
    3. Noise dominates → :attr:`RoughnessVerdict.NOISE_DOMINATED` (get better
       data before modelling anything)
    4. H unstable across folds → :attr:`RoughnessVerdict.UNSTABLE` (no fitted
       volatility model will generalise)
    5. Otherwise classify H as rough / intermediate / Markovian.

    Args:
        log_rv: Log realised variance, e.g. from :func:`log_variance_proxy`.
        returns: Optional bar returns used for the jump check.  When ``None``
            the jump test is skipped and :attr:`RoughnessReport.jump_fraction`
            is ``0.0``.
        symbol: Instrument identifier for reporting.
        n_folds: Number of contiguous chronological folds.  Must be at least 2,
            since stability across folds is the whole point.
        thresholds: Decision thresholds; defaults to :class:`GateThresholds`.

    Returns:
        A :class:`RoughnessReport`.

    Raises:
        ValueError: If *n_folds* is less than 2.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2 to assess stability, got {n_folds}")

    limits = thresholds if thresholds is not None else GateThresholds()
    series = np.asarray(log_rv, dtype=np.float64)
    series = series[np.isfinite(series)]

    jump_fraction = float(jump_ratio(np.asarray(returns))) if returns is not None else 0.0

    if series.size < limits.min_obs_per_fold * n_folds:
        return RoughnessReport(
            symbol=symbol,
            verdict=RoughnessVerdict.INSUFFICIENT_DATA,
            rationale=(
                f"{series.size} usable observations, need at least "
                f"{limits.min_obs_per_fold * n_folds} for {n_folds} folds of "
                f"{limits.min_obs_per_fold}"
            ),
            jump_fraction=jump_fraction,
            thresholds=limits,
        )

    folds: list[FoldRoughness] = []
    for idx, chunk in enumerate(np.array_split(series, n_folds)):
        estimate = hurst_with_nugget(chunk)
        folds.append(
            FoldRoughness(
                fold_idx=idx,
                n_obs=int(chunk.size),
                hurst=estimate.hurst,
                hurst_naive=hurst_exponent(chunk),
                noise_std=estimate.noise_std,
                noise_share=estimate.noise_share,
                converged=estimate.converged,
            )
        )

    report = RoughnessReport(
        symbol=symbol,
        verdict=RoughnessVerdict.ROUGH,  # provisional; replaced below
        rationale="",
        folds=folds,
        jump_fraction=jump_fraction,
        thresholds=limits,
    )
    verdict, rationale = _decide(report, limits)

    return RoughnessReport(
        symbol=symbol,
        verdict=verdict,
        rationale=rationale,
        folds=folds,
        jump_fraction=jump_fraction,
        thresholds=limits,
    )


def _decide(
    report: RoughnessReport,
    limits: GateThresholds,
) -> tuple[RoughnessVerdict, str]:
    """Apply the gate's decision rules to a populated report.

    Args:
        report: Report carrying the per-fold estimates and jump fraction.
        limits: Decision thresholds.

    Returns:
        Tuple of the verdict and its rationale.
    """
    if report.jump_fraction > limits.max_jump_fraction:
        return (
            RoughnessVerdict.JUMP_DOMINATED,
            f"jumps account for {report.jump_fraction:.0%} of quadratic variation "
            f"(limit {limits.max_jump_fraction:.0%}); a continuous SDE is the wrong "
            "model class regardless of its roughness",
        )

    noise_share = report.mean_noise_share
    if noise_share > limits.max_noise_share:
        return (
            RoughnessVerdict.NOISE_DOMINATED,
            f"observation noise explains {noise_share:.0%} of the short-lag variogram "
            f"(limit {limits.max_noise_share:.0%}), so H is not identified. The naive "
            f"estimator reports {report.mean_hurst_naive:.2f}, which would look 'rough' "
            "but is an artefact. Use higher-frequency realised variance before "
            "concluding anything",
        )

    std_hurst = report.std_hurst
    if np.isfinite(std_hurst) and std_hurst > limits.max_hurst_std:
        return (
            RoughnessVerdict.UNSTABLE,
            f"H varies across folds with std {std_hurst:.3f} (limit "
            f"{limits.max_hurst_std:.3f}); the data supports no stable volatility model, "
            "so a more flexible one will fit noise",
        )

    mean_hurst = report.mean_hurst
    if mean_hurst <= limits.rough_below:
        return (
            RoughnessVerdict.ROUGH,
            f"H = {mean_hurst:.3f} ± {std_hurst:.3f}, stably below "
            f"{limits.rough_below}, with noise share {noise_share:.2f}. A rough or "
            "non-Markovian model is justified",
        )
    if mean_hurst >= limits.markovian_above:
        return (
            RoughnessVerdict.MARKOVIAN,
            f"H = {mean_hurst:.3f} ± {std_hurst:.3f}, at or above "
            f"{limits.markovian_above}; volatility is adequately Markovian. Use Heston "
            "or HAR — the rough/neural motivation is absent",
        )
    return (
        RoughnessVerdict.INTERMEDIATE,
        f"H = {mean_hurst:.3f} ± {std_hurst:.3f}, between {limits.rough_below} and "
        f"{limits.markovian_above}; support for a rough model is weak. Prefer the "
        "simpler model until a cheaper experiment says otherwise",
    )
