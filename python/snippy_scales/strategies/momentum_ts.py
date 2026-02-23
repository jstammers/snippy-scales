"""Time-Series Momentum (TSMOM) strategy.

Implements the canonical trend-following signal from Moskowitz, Ooi & Pedersen
(2012) *Time Series Momentum*:

.. math::

    \\text{signal}_t = \\operatorname{sign}\\!\\left(r_{t-L-S,\\,t-S}\\right)

    \\text{position}_t = \\text{signal}_t \\times
        \\frac{\\sigma^*}{\\hat{\\sigma}_t}

where :math:`\\sigma^*` is the target annualised volatility, :math:`\\hat{\\sigma}_t`
is the rolling realised volatility, *L* is the lookback window, and *S* is the
number of recent bars to skip (to avoid short-term reversal contamination).

The strategy is also compatible with the standard moving-average crossover
formulation via :class:`snippy_scales.strategies.trend.TrendFollowing`.

Design rationale
----------------
Time-series momentum is the first structural test of portfolio path dependency:
drawdown behaviour and leverage dynamics are highly sensitive to the choice of
lookback and volatility-targeting parameters.  A correct implementation should:

* Produce positive out-of-sample Sharpe for trend-following regimes.
* Show graceful degradation (not blow-up) when slippage is added.
* Have turnover that scales predictably with *lookback* and *rebal_freq*.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from snippy_scales.strategies.trend import Strategy


class TimeSeriesMomentum(Strategy):
    """Time-Series Momentum strategy with volatility-targeted position sizing.

    The raw signal is the sign of the realised return over a rolling *lookback*
    window, optionally skipping the most recent *skip_recent* bars to avoid
    short-term reversal contamination (the classic ``12-1`` formulation uses
    ``lookback=252, skip_recent=21``).

    Position size is scaled so that the expected portfolio volatility equals
    *vol_target*, subject to a hard *max_leverage* cap.

    Args:
        lookback: Look-back window in bars (default: 252 = 1 trading year).
        skip_recent: Bars to exclude from the return window (default: 0).
            Set to 21 to implement the 12-1 month TSMOM strategy.
        vol_target: Target annualised volatility as a decimal (default: 0.10).
        vol_lookback: Rolling window for realised-vol estimation (default: 63).
        max_leverage: Hard cap on the vol scalar (default: 2.0).
        rebal_freq: Rebalance every N bars (default: 1 = daily).  When greater
            than 1 the position is frozen between rebalance dates.

    Raises:
        ValueError: On invalid parameter values.

    Example::

        strategy = TimeSeriesMomentum(lookback=252, skip_recent=21, vol_target=0.10)
        positions = strategy.generate_signals(bars)
        # positions is a pl.Series of signed floats, same length as bars
    """

    def __init__(
        self,
        lookback: int = 252,
        skip_recent: int = 0,
        vol_target: float = 0.10,
        vol_lookback: int = 63,
        max_leverage: float = 2.0,
        rebal_freq: int = 1,
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be ≥ 1, got {lookback}")
        if skip_recent < 0:
            raise ValueError(f"skip_recent must be ≥ 0, got {skip_recent}")
        if vol_target <= 0:
            raise ValueError(f"vol_target must be positive, got {vol_target}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {max_leverage}")
        if rebal_freq < 1:
            raise ValueError(f"rebal_freq must be ≥ 1, got {rebal_freq}")

        self.lookback = lookback
        self.skip_recent = skip_recent
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage
        self.rebal_freq = rebal_freq

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        """Return signed target positions for each bar.

        Positions are in volatility-normalised units: a value of ``1.0`` means
        the strategy is long at full *vol_target* allocation, ``-1.5`` means
        short at 1.5× target volatility.

        Args:
            bars: Polars DataFrame with at minimum a ``close`` column.

        Returns:
            Float :class:`polars.Series` named ``"position"``, same length as
            *bars*.  Values are zero during the warmup period.
        """
        close = bars["close"]

        # ── Momentum return ────────────────────────────────────────────────────
        # Return from (t - skip_recent - lookback) to (t - skip_recent).
        recent_close = close.shift(self.skip_recent) if self.skip_recent > 0 else close

        past_close = recent_close.shift(self.lookback)
        mom_ret = recent_close / past_close - 1.0
        direction = mom_ret.sign()

        # ── Volatility-targeted sizing ─────────────────────────────────────────
        daily_ret = close.pct_change()
        rv = (daily_ret.rolling_std(self.vol_lookback) * (252**0.5)).clip(lower_bound=1e-6)
        vol_scalar = (self.vol_target / rv).clip(upper_bound=self.max_leverage)

        positions = direction * vol_scalar

        # ── Rebalancing frequency ──────────────────────────────────────────────
        # Between rebalance dates, carry forward the last position.
        if self.rebal_freq > 1:
            pos_arr = positions.to_numpy().copy()
            for i in range(1, len(pos_arr)):
                if i % self.rebal_freq != 0 and not np.isnan(pos_arr[i - 1]):
                    pos_arr[i] = pos_arr[i - 1]
            positions = pl.Series(pos_arr)

        return positions.fill_null(0.0).rename("position")
