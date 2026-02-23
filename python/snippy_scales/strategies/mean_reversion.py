"""Short-horizon mean-reversion strategy based on rolling z-scored returns.

Implements the simplest defensible mean-reversion signal:

.. math::

    z_t = \\frac{r_t - \\mu_L}{\\sigma_L}

where :math:`r_t` is the 1-bar return and :math:`\\mu_L, \\sigma_L` are the
rolling *L*-bar mean and standard deviation.

Position direction is **contrarian** to the z-score:

* :math:`z_t > +\\text{entry\\_z}` → short (expect downward reversion).
* :math:`z_t < -\\text{entry\\_z}` → long (expect upward reversion).
* :math:`|z_t| < \\text{exit\\_z}` → flat (reversion has occurred).
* :math:`\\text{exit\\_z} \\le |z_t| \\le \\text{entry\\_z}` → hold (hysteresis band).

Position size is vol-scaled via the same mechanism used across all strategies
in this codebase.

Why this strategy matters
-------------------------
Short-horizon mean reversion is the **worst case** for execution quality.
If this layer shows a bleed from paper to realistic P&L, your slippage and
commission modelling is wrong.  The strategy is deliberately designed so that:

* A naïve ``zero-fee`` backtest shows strong alpha.
* Realistic fees (≥ 5 bps) significantly reduce Sharpe.
* Unrealistic slippage breaks the strategy completely.

This makes it an excellent calibration target for transaction cost modelling.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from snippy_scales.research.features import zscore
from snippy_scales.strategies.trend import Strategy


class MeanReversion(Strategy):
    """Short-horizon mean-reversion strategy using rolling z-scored returns.

    The strategy enters a position when the daily return z-score exceeds an
    entry threshold and exits when the z-score reverts into an exit band.  A
    hysteresis band between *exit_z* and *entry_z* prevents excessive
    churning, which is critical for modelling realistic transaction costs.

    Args:
        lookback: Rolling window for z-score computation in bars (default: 20).
        entry_z: Absolute z-score at which a new position is opened
            (default: 2.0).
        exit_z: Absolute z-score at or below which a position is closed
            (default: 0.5).  Must be strictly less than *entry_z*.
        vol_target: Target annualised portfolio volatility as a decimal
            (default: 0.10 = 10%).
        vol_lookback: Rolling window for realised-vol estimation (default: 20).
        max_leverage: Hard cap on the vol scalar (default: 3.0).  Mean-
            reversion strategies often warrant slightly higher leverage because
            positions are held for shorter durations.

    Raises:
        ValueError: On invalid parameter combinations (e.g. ``exit_z ≥ entry_z``).

    Example::

        strategy = MeanReversion(lookback=20, entry_z=2.0, exit_z=0.5)
        positions = strategy.generate_signals(bars)
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        vol_target: float = 0.10,
        vol_lookback: int = 20,
        max_leverage: float = 3.0,
    ) -> None:
        if lookback < 2:  # noqa: PLR2004
            raise ValueError(f"lookback must be ≥ 2, got {lookback}")
        if entry_z <= 0:
            raise ValueError(f"entry_z must be positive, got {entry_z}")
        if exit_z < 0:
            raise ValueError(f"exit_z must be ≥ 0, got {exit_z}")
        if exit_z >= entry_z:
            raise ValueError(f"exit_z ({exit_z}) must be strictly less than entry_z ({entry_z})")
        if vol_target <= 0:
            raise ValueError(f"vol_target must be positive, got {vol_target}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {max_leverage}")

        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        """Return signed target positions for each bar.

        Implements a state-machine with three states (long / flat / short) and
        hysteresis between the entry and exit thresholds.  Transitions are
        detected bar-by-bar; the output is therefore path-dependent.

        Args:
            bars: Polars DataFrame with at minimum a ``close`` column.

        Returns:
            Float :class:`polars.Series` named ``"position"``, same length as
            *bars*.  Values are zero during the warmup period and in the flat
            state.
        """
        close = bars["close"]
        daily_ret = close.pct_change().fill_null(0.0)

        # Rolling z-score of the 1-bar return ─────────────────────────────────
        z_series = zscore(daily_ret, window=self.lookback)

        # Realised volatility for position sizing ──────────────────────────────
        rv = (daily_ret.rolling_std(self.vol_lookback) * (252**0.5)).clip(lower_bound=1e-6)

        # Convert to numpy for the state-machine loop ──────────────────────────
        z_arr = z_series.to_numpy()
        vol_arr = rv.to_numpy()

        pos = np.zeros(len(z_arr), dtype=np.float64)
        prev_pos = 0.0

        for i in range(len(z_arr)):
            zi = z_arr[i]
            vi = vol_arr[i] if not np.isnan(vol_arr[i]) else 1.0

            if np.isnan(zi):
                # Still in warmup period.
                pos[i] = 0.0
                prev_pos = 0.0
            elif abs(zi) > self.entry_z:
                # Entry: take contrarian position opposite to the z-score sign.
                vol_scalar = min(self.vol_target / vi, self.max_leverage)
                pos[i] = -np.sign(zi) * vol_scalar
                prev_pos = pos[i]
            elif prev_pos != 0.0 and abs(zi) >= self.exit_z:
                # Hysteresis band: maintain existing position to avoid churn.
                pos[i] = prev_pos
            else:
                # Exit: z-score has reverted inside the exit band.
                pos[i] = 0.0
                prev_pos = 0.0

        return pl.Series("position", pos)
