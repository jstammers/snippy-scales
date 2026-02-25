"""Feature engineering helpers for research notebooks and strategy development."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from snippy_scales._constants import TRADING_DAYS_PER_YEAR

if TYPE_CHECKING:
    import polars as pl


def realised_vol(close: pl.Series, window: int = 20, annualise: bool = True) -> pl.Series:
    """Rolling realised volatility from close prices."""
    ret = close.pct_change()
    rv = ret.rolling_std(window)
    return (rv * math.sqrt(TRADING_DAYS_PER_YEAR) if annualise else rv).rename(f"rv_{window}")


def zscore(series: pl.Series, window: int = 60) -> pl.Series:
    """Rolling z-score."""
    mu = series.rolling_mean(window)
    sigma = series.rolling_std(window)
    return ((series - mu) / sigma.clip(lower_bound=1e-9)).rename(f"z_{series.name}")


def carry(front: pl.Series, back: pl.Series, front_expiry_days: pl.Series) -> pl.Series:
    """Approximate futures carry (annualised roll yield)."""
    roll_yield = (back - front) / front.clip(lower_bound=1e-9)
    annualised = roll_yield / (front_expiry_days / 365.0).clip(lower_bound=1e-3)
    return annualised.rename("carry")
