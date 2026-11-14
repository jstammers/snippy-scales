"""Strategy interfaces and built-in strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

#: Column naming the symbol of each row in a panel (see :meth:`Strategy.generate_signals_panel`).
SYMBOL = "symbol"


class Strategy(ABC):
    """Abstract base for all strategies.

    A strategy consumes a bar DataFrame and returns a Series of target
    positions (float, signed; e.g. 1.0 = long 1 unit, -0.5 = short 0.5).
    """

    @abstractmethod
    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        """Return a Series of target positions, same length as bars."""
        ...

    def generate_signals_panel(self, bars: pl.DataFrame) -> pl.Series:
        """Targets for many symbols at once: :meth:`generate_signals` per symbol.

        Args:
            bars: Long frame of several symbols' bars, the symbol in column
                :data:`SYMBOL`. Each symbol's rows are **contiguous and sorted by
                bar close**.

        Returns:
            One target per row of *bars*, in row order; each symbol's slice
            equals ``generate_signals`` on that symbol's bars alone.

        The default calls :meth:`generate_signals` once per symbol. Override it
        with per-symbol window expressions (``expr.over(SYMBOL)``) to compute a
        whole universe in one polars query; ``tests/test_strategy_panel.py``
        checks an override against the default.
        """
        parts = []
        for frame in bars.partition_by(SYMBOL, maintain_order=True):
            values = self.generate_signals(frame)
            if values.len() != frame.height:
                raise ValueError(
                    f"generate_signals returned {values.len()} values for {frame.height} bars "
                    f"of {frame[SYMBOL][0]}"
                )
            parts.append(values.cast(pl.Float64).rename("position"))
        return pl.concat(parts) if parts else pl.Series("position", [], pl.Float64)


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

    def _positions(self, close: pl.Expr) -> pl.Expr:
        fast_ma = close.rolling_mean(self.fast_period)
        slow_ma = close.rolling_mean(self.slow_period)
        daily_ret = close.pct_change()
        realised_vol = daily_ret.rolling_std(self.vol_lookback) * (252**0.5)

        raw_signal = (fast_ma - slow_ma).sign()
        vol_scalar = (self.vol_target / realised_vol.clip(lower_bound=1e-6)).clip(upper_bound=5.0)
        return raw_signal * vol_scalar

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        positions = bars.select(self._positions(pl.col("close")).alias("position"))["position"]
        return positions.fill_null(0.0)

    def generate_signals_panel(self, bars: pl.DataFrame) -> pl.Series:
        positions = bars.select(self._positions(pl.col("close")).over(SYMBOL).alias("position"))[
            "position"
        ]
        return positions.fill_null(0.0)
