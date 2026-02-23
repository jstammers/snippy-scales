"""Strategy interfaces and built-in strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


class Strategy(ABC):
    """Abstract base for all strategies.

    A strategy consumes a bar DataFrame and returns a Series of target
    positions (float, signed; e.g. 1.0 = long 1 unit, -0.5 = short 0.5).
    """

    @abstractmethod
    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        """Return a Series of target positions, same length as bars."""
        ...


class TrendFollowing(Strategy):
    """Simple moving-average crossover trend strategy.

    Volatility-normalised position sizing: target Δ = sign(signal) * (vol_target / realised_vol).
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 60,
        vol_target: float = 0.10,
        vol_lookback: int = 20,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        close = bars["close"]
        fast_ma = close.rolling_mean(self.fast_period)
        slow_ma = close.rolling_mean(self.slow_period)
        daily_ret = close.pct_change()
        realised_vol = daily_ret.rolling_std(self.vol_lookback) * (252**0.5)

        raw_signal = (fast_ma - slow_ma).sign()
        vol_scalar = (self.vol_target / realised_vol.clip(lower_bound=1e-6)).clip(upper_bound=5.0)
        positions = raw_signal * vol_scalar

        return positions.fill_null(0.0).rename("position")
