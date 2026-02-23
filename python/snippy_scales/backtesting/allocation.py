"""Capital allocation and volatility-targeting utilities.

Provides :class:`VolTargetAllocator`, which converts a target annualised
volatility into a static weight suitable for passing to raptorbt.  The weight
is computed once from the full price series (as a proxy for expected realised
vol) rather than dynamically, because raptorbt accepts only static per-leg
weights.
"""

from __future__ import annotations

import numpy as np

from snippy_scales.backtesting.domain import TRADING_DAYS_PER_YEAR


class VolTargetAllocator:
    """Compute static capital-allocation weights from a volatility target.

    raptorbt accepts a single ``weight`` float per instrument leg.  This class
    estimates average realised daily volatility (annualised) over the full
    close-price series and derives the weight needed to achieve *vol_target*,
    capped at *max_leverage*.

    For long/short strategies the result is halved because the same total
    capital is split across both books.

    Args:
        vol_target: Target annualised portfolio volatility (e.g. ``0.10``
            for 10 %).
        max_leverage: Hard cap on the derived weight before halving
            (e.g. ``2.0`` = 2× leverage).
        long_short: If ``True``, the computed weight is halved to account
            for symmetric long and short books (default ``True``).
    """

    def __init__(
        self,
        *,
        vol_target: float = 0.10,
        max_leverage: float = 2.0,
        long_short: bool = True,
    ) -> None:
        if vol_target <= 0.0:
            raise ValueError(f"vol_target must be positive, got {vol_target}")
        if max_leverage <= 0.0:
            raise ValueError(f"max_leverage must be positive, got {max_leverage}")
        self.vol_target = vol_target
        self.max_leverage = max_leverage
        self.long_short = long_short

    def weight(self, close: np.ndarray) -> float:
        """Return the allocation weight for a single instrument leg.

        Args:
            close: Close price array for the instrument.

        Returns:
            A positive float representing the per-leg weight.  For
            ``long_short=True`` strategies this is at most
            ``max_leverage / 2``.
        """
        if len(close) < 2:  # noqa: PLR2004
            raw = self.vol_target
        else:
            daily_ret = np.diff(close) / np.where(close[:-1] == 0.0, 1.0, close[:-1])
            avg_vol = float(np.nanstd(daily_ret)) * (TRADING_DAYS_PER_YEAR**0.5)
            raw = self.vol_target / max(avg_vol, 1e-6)

        capped = min(raw, self.max_leverage)
        return capped / 2.0 if self.long_short else capped
