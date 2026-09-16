"""Range-based volatility estimators.

:func:`~snippy_scales.research.features.realised_vol` uses close-to-close
returns, which throws away the high and low of every bar — roughly 80% of the
information the OHLC data already carries.  Range-based estimators recover it,
and the efficiency gain is large: Parkinson is about 5× more efficient than
close-to-close, Garman–Klass about 7×, Yang–Zhang up to 14×.

For volatility *forecasting* this matters more than any modelling
sophistication downstream.  A better estimate of today's realised volatility
improves tomorrow's forecast far more reliably than a more expressive model
fitted to a noisier target, and it costs nothing — the data is already on disk.

Estimator selection:

============= ======================================== ==================
Estimator     Handles                                  Efficiency
============= ======================================== ==================
Parkinson     range only; no drift, no gaps            ~5×
Garman–Klass  range + close; no drift, no gaps         ~7×
Rogers–Satchell range + close + **drift**              ~8×
Yang–Zhang    range + close + drift + **overnight gaps** ~14×
============= ======================================== ==================

Yang–Zhang is the default choice for futures, which gap across the daily
settlement break.  Rogers–Satchell is the right pick when a series trends
strongly but does not gap.

All functions take and return :class:`polars.Series` and are annualised by
default with ``TRADING_DAYS_PER_YEAR``; pass ``periods_per_year`` for intraday
bars.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from snippy_scales._constants import TRADING_DAYS_PER_YEAR

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "parkinson_vol",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "yang_zhang_vol",
]

#: 1 / (4 ln 2) — the Parkinson scaling constant.
_PARKINSON_SCALE = 1.0 / (4.0 * math.log(2.0))

#: 2 ln 2 - 1 — the Garman–Klass close-term coefficient.
_GK_CLOSE_COEFF = 2.0 * math.log(2.0) - 1.0


def _annualise(variance: pl.Series, periods_per_year: float, name: str) -> pl.Series:
    """Convert a rolling variance to an annualised volatility series.

    Args:
        variance: Rolling variance per bar.
        periods_per_year: Annualisation factor; ``1.0`` leaves the result
            per-bar.
        name: Name for the returned series.

    Returns:
        Annualised volatility, renamed to *name*.
    """
    clipped = variance.clip(lower_bound=0.0)
    return (clipped.sqrt() * math.sqrt(periods_per_year)).rename(name)


def parkinson_vol(
    high: pl.Series,
    low: pl.Series,
    window: int = 20,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pl.Series:
    """Return rolling Parkinson volatility from the high-low range.

    .. math::

        \\sigma^2 = \\frac{1}{4 \\ln 2} \\; \\overline{(\\ln(H/L))^2}

    Assumes zero drift and continuous trading, so it *understates* volatility
    for a gapping instrument.

    Args:
        high: Bar high prices.
        low: Bar low prices.
        window: Rolling window length in bars.
        periods_per_year: Annualisation factor.

    Returns:
        Annualised volatility series named ``parkinson_<window>``.
    """
    log_hl = (high / low).log()
    variance = (log_hl**2 * _PARKINSON_SCALE).rolling_mean(window)
    return _annualise(variance, periods_per_year, f"parkinson_{window}")


def garman_klass_vol(
    high: pl.Series,
    low: pl.Series,
    open_: pl.Series,
    close: pl.Series,
    window: int = 20,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pl.Series:
    """Return rolling Garman–Klass volatility.

    .. math::

        \\sigma^2 = \\overline{\\tfrac{1}{2}(\\ln(H/L))^2
                    - (2\\ln 2 - 1)(\\ln(C/O))^2}

    Adds the open-to-close move to Parkinson's range term.  Still assumes zero
    drift and no overnight gap.

    Args:
        high: Bar high prices.
        low: Bar low prices.
        open_: Bar open prices.
        close: Bar close prices.
        window: Rolling window length in bars.
        periods_per_year: Annualisation factor.

    Returns:
        Annualised volatility series named ``garman_klass_<window>``.
    """
    log_hl = (high / low).log()
    log_co = (close / open_).log()
    per_bar = 0.5 * log_hl**2 - _GK_CLOSE_COEFF * log_co**2
    return _annualise(per_bar.rolling_mean(window), periods_per_year, f"garman_klass_{window}")


def rogers_satchell_vol(
    high: pl.Series,
    low: pl.Series,
    open_: pl.Series,
    close: pl.Series,
    window: int = 20,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pl.Series:
    """Return rolling Rogers–Satchell volatility.

    .. math::

        \\sigma^2 = \\overline{\\ln(H/C)\\ln(H/O) + \\ln(L/C)\\ln(L/O)}

    **Drift-independent**: unlike Parkinson and Garman–Klass it stays unbiased
    when the series trends, which makes it the right range estimator for a
    strongly trending future.  It still ignores overnight gaps.

    Args:
        high: Bar high prices.
        low: Bar low prices.
        open_: Bar open prices.
        close: Bar close prices.
        window: Rolling window length in bars.
        periods_per_year: Annualisation factor.

    Returns:
        Annualised volatility series named ``rogers_satchell_<window>``.
    """
    per_bar = (high / close).log() * (high / open_).log() + (low / close).log() * (
        low / open_
    ).log()
    return _annualise(per_bar.rolling_mean(window), periods_per_year, f"rogers_satchell_{window}")


def yang_zhang_vol(
    high: pl.Series,
    low: pl.Series,
    open_: pl.Series,
    close: pl.Series,
    window: int = 20,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pl.Series:
    """Return rolling Yang–Zhang volatility.

    Combines three components — overnight (close-to-open), open-to-close, and
    the drift-independent Rogers–Satchell term:

    .. math::

        \\sigma^2 = \\sigma^2_{\\text{overnight}}
                    + k\\,\\sigma^2_{\\text{open→close}}
                    + (1-k)\\,\\sigma^2_{\\text{RS}}

    with :math:`k = 0.34 / (1.34 + (n+1)/(n-1))`.

    This is the estimator to reach for on CME futures: it is the only one here
    that accounts for the **overnight gap** across the daily settlement break,
    which for equity index futures carries a large share of total variance.

    Args:
        high: Bar high prices.
        low: Bar low prices.
        open_: Bar open prices.
        close: Bar close prices.
        window: Rolling window length in bars.  Must be at least 2.
        periods_per_year: Annualisation factor.

    Returns:
        Annualised volatility series named ``yang_zhang_<window>``.

    Raises:
        ValueError: If *window* is less than 2, which would make the
            weighting coefficient undefined.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")

    prev_close = close.shift(1)
    log_overnight = (open_ / prev_close).log()
    log_open_close = (close / open_).log()

    # Rolling *variances* about their own rolling means (ddof=1).
    overnight_var = log_overnight.rolling_std(window, ddof=1) ** 2
    open_close_var = log_open_close.rolling_std(window, ddof=1) ** 2

    rs_per_bar = (high / close).log() * (high / open_).log() + (low / close).log() * (
        low / open_
    ).log()
    rs_var = rs_per_bar.rolling_mean(window)

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    variance = overnight_var + k * open_close_var + (1.0 - k) * rs_var
    return _annualise(variance, periods_per_year, f"yang_zhang_{window}")
