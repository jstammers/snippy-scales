"""Cross-Sectional Momentum (CSMOM) strategy.

Implements the classic cross-sectional momentum factor:

1. Compute the *J*-month return for each asset (optionally skipping the most
   recent *S* bars to avoid the short-term reversal effect).
2. At each rebalancing date, rank assets by their momentum returns.
3. Go long assets in the top *k* quantile and short assets in the bottom *k*
   quantile.
4. Scale position sizes to target *vol_target* annualised volatility per asset.

This strategy is the primary test for **portfolio aggregation logic** and
**cross-sectional signal generation** in the backtesting infrastructure.  A
correct implementation should:

* Produce clean, rank-ordered signals with no look-ahead bias.
* Rebalance at the correct frequency with position carry-forward between
  rebalance dates.
* Produce a non-trivial long-short portfolio that exercises both books in
  :class:`~snippy_scales.backtesting.runner.BasketRunner`.

Interface
---------
:class:`CrossSectionalMomentum` extends :class:`MultiAssetStrategy`, which is
the multi-asset analogue of the single-asset
:class:`~snippy_scales.strategies.trend.Strategy` ABC.  The
:class:`~snippy_scales.backtesting.runner.BasketRunner` accepts either base
class and handles raptorbt wiring automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl


class MultiAssetStrategy(ABC):
    """Abstract base for strategies that operate on a universe of assets.

    Unlike the single-asset :class:`~snippy_scales.strategies.trend.Strategy`,
    a ``MultiAssetStrategy`` receives a mapping of symbol → bar DataFrame and
    returns a per-symbol position Series.  This enables cross-sectional signals
    that depend on the *relative* performance of assets.

    The :class:`~snippy_scales.backtesting.runner.BasketRunner` accepts any
    concrete implementation of this class.
    """

    @abstractmethod
    def generate_signals(self, multi_bars: dict[str, pl.DataFrame]) -> dict[str, pl.Series]:
        """Return signed target positions for every asset in the universe.

        Args:
            multi_bars: Mapping of ``symbol → bar DataFrame``.  Each DataFrame
                must include at minimum a ``close`` column and must have the
                same number of rows (i.e. they must be time-aligned).

        Returns:
            Mapping of ``symbol → position Series``.  Each Series must have the
            same length as the corresponding input DataFrame.  Positive values
            indicate long positions; negative values indicate short positions;
            zero indicates flat.
        """
        ...


class CrossSectionalMomentum(MultiAssetStrategy):
    """Cross-sectional momentum strategy with volatility-targeted sizing.

    At each rebalancing date, assets are ranked by their *J*-bar momentum
    return.  Assets in the top *top_quantile* are assigned long positions;
    assets in the bottom *bottom_quantile* are assigned short positions.
    All other assets are flat.  Position size is scaled to target
    *vol_target* annualised volatility per asset, subject to *max_leverage*.

    Args:
        lookback: Momentum return window in bars (default: 252 = 1 year).
        skip_recent: Bars to exclude from the end of the momentum window
            (default: 21 = 1 month).  Setting this to 21 gives the classic
            12-1 month momentum factor.
        top_quantile: Fraction of assets to buy (default: 0.25 = top 25%).
        bottom_quantile: Fraction of assets to short (default: 0.25 = bottom 25%).
        vol_target: Target annualised volatility per position as a decimal
            (default: 0.10 = 10%).
        vol_lookback: Rolling window for per-asset realised-vol estimation
            (default: 63 bars ≈ 3 months).
        max_leverage: Hard cap on the vol scalar per asset (default: 2.0).
        rebal_freq: Rebalance every N bars (default: 21 = monthly).
            Between rebalancing dates positions are carried forward.

    Raises:
        ValueError: On invalid parameter values.

    Example::

        strategy = CrossSectionalMomentum(lookback=252, top_quantile=0.25)
        runner = BasketRunner(initial_capital=1_000_000.0)
        result = runner.run(strategy, multi_bars)
    """

    def __init__(
        self,
        lookback: int = 252,
        skip_recent: int = 21,
        top_quantile: float = 0.25,
        bottom_quantile: float = 0.25,
        vol_target: float = 0.10,
        vol_lookback: int = 63,
        max_leverage: float = 2.0,
        rebal_freq: int = 21,
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be ≥ 1, got {lookback}")
        if skip_recent < 0:
            raise ValueError(f"skip_recent must be ≥ 0, got {skip_recent}")
        if not 0 < top_quantile < 1:
            raise ValueError(f"top_quantile must be in (0, 1), got {top_quantile}")
        if not 0 < bottom_quantile < 1:
            raise ValueError(f"bottom_quantile must be in (0, 1), got {bottom_quantile}")
        if top_quantile + bottom_quantile > 1:
            raise ValueError(
                f"top_quantile ({top_quantile}) + bottom_quantile ({bottom_quantile}) must be ≤ 1"
            )
        if vol_target <= 0:
            raise ValueError(f"vol_target must be positive, got {vol_target}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {max_leverage}")
        if rebal_freq < 1:
            raise ValueError(f"rebal_freq must be ≥ 1, got {rebal_freq}")

        self.lookback = lookback
        self.skip_recent = skip_recent
        self.top_quantile = top_quantile
        self.bottom_quantile = bottom_quantile
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage
        self.rebal_freq = rebal_freq

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, multi_bars: dict[str, pl.DataFrame]) -> dict[str, pl.Series]:
        """Compute cross-sectional momentum positions for all assets.

        Signals are computed in three stages:

        1. **Momentum returns** – the *lookback*-bar return ending *skip_recent*
           bars before the current date, computed for every asset.
        2. **Cross-sectional ranking** – at each rebalancing date, assets are
           ranked and assigned to long/short/flat based on their quantile.
        3. **Vol targeting** – position size = ``vol_target / realised_vol``
           capped at *max_leverage*.

        Args:
            multi_bars: ``symbol → bars DataFrame``.  All DataFrames must have
                the same length (time-aligned).

        Returns:
            ``symbol → position Series``.  Each Series has the same length as
            the corresponding input DataFrame.

        Raises:
            ValueError: If the DataFrames have different lengths or if fewer
                than 2 assets are provided.
        """
        symbols = list(multi_bars.keys())
        if len(symbols) < 2:  # noqa: PLR2004
            raise ValueError(f"CrossSectionalMomentum requires ≥ 2 assets, got {len(symbols)}")

        lengths = {sym: len(df) for sym, df in multi_bars.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"All bar DataFrames must have the same length.  Got: {lengths}")

        n = next(iter(lengths.values()))

        # ── Step 1: build close matrix and momentum returns ────────────────────
        close_matrix = np.column_stack(
            [multi_bars[sym]["close"].to_numpy().astype(np.float64) for sym in symbols]
        )  # shape (n, n_assets)

        mom_ret = self._compute_momentum_returns(close_matrix)  # (n, n_assets)

        # ── Step 2: compute rolling realised volatility per asset ──────────────
        vol_matrix = self._compute_vol_matrix(close_matrix)  # (n, n_assets)

        # ── Step 3: cross-sectional ranking and position assignment ────────────
        positions_matrix = self._assign_positions(n, mom_ret, vol_matrix)  # (n, n_assets)

        return {
            sym: pl.Series(positions_matrix[:, j]).rename("position")
            for j, sym in enumerate(symbols)
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_momentum_returns(self, close_matrix: np.ndarray) -> np.ndarray:
        """Compute the momentum return for every bar and asset.

        Args:
            close_matrix: Shape ``(n_bars, n_assets)`` of close prices.

        Returns:
            Float array of the same shape.  Entries within the warmup period
            are ``NaN``.
        """
        n, n_assets = close_matrix.shape
        warmup = self.lookback + self.skip_recent
        mom_ret = np.full((n, n_assets), np.nan)

        if n <= warmup:
            return mom_ret

        # close[t - skip_recent] / close[t - skip_recent - lookback] - 1
        for t in range(warmup, n):
            recent_idx = t - self.skip_recent if self.skip_recent > 0 else t
            past_idx = recent_idx - self.lookback
            with np.errstate(invalid="ignore", divide="ignore"):
                mom_ret[t] = close_matrix[recent_idx] / close_matrix[past_idx] - 1.0

        return mom_ret

    def _compute_vol_matrix(self, close_matrix: np.ndarray) -> np.ndarray:
        """Compute rolling annualised realised volatility for every asset.

        Uses the *vol_lookback*-bar standard deviation of daily log-returns,
        annualised by ``√252``.

        Args:
            close_matrix: Shape ``(n_bars, n_assets)`` of close prices.

        Returns:
            Float array of the same shape.  Entries in the warmup period are
            replaced by ``0.20`` (20% assumed vol) to avoid division by zero.
        """
        n, n_assets = close_matrix.shape
        vol_matrix = np.full((n, n_assets), 0.20)  # default 20% vol during warmup

        if n < self.vol_lookback + 1:
            return vol_matrix

        # Daily returns: shape (n-1, n_assets)
        with np.errstate(invalid="ignore", divide="ignore"):
            daily_ret = np.diff(close_matrix, axis=0) / close_matrix[:-1]

        for t in range(self.vol_lookback, n):
            window = daily_ret[t - self.vol_lookback : t]
            vol = np.nanstd(window, axis=0) * (252**0.5)
            vol_matrix[t] = np.maximum(vol, 1e-6)

        return vol_matrix

    def _assign_positions(
        self,
        n: int,
        mom_ret: np.ndarray,
        vol_matrix: np.ndarray,
    ) -> np.ndarray:
        """Assign long/short/flat positions at each rebalancing date.

        Args:
            n: Number of bars.
            mom_ret: Momentum returns array, shape ``(n, n_assets)``.
            vol_matrix: Realised vol array, shape ``(n, n_assets)``.

        Returns:
            Position matrix, shape ``(n, n_assets)``.  Positive = long,
            negative = short, zero = flat.
        """
        _n_assets = mom_ret.shape[1]
        positions = np.zeros((n, _n_assets), dtype=np.float64)
        warmup = self.lookback + self.skip_recent

        prev_rebal_pos = np.zeros(_n_assets)

        for t in range(warmup, n):
            is_rebal_date = (t - warmup) % self.rebal_freq == 0

            if not is_rebal_date:
                # Carry forward the last rebalanced position.
                positions[t] = prev_rebal_pos
                continue

            mom_at_t = mom_ret[t]
            valid_mask = ~np.isnan(mom_at_t)

            if valid_mask.sum() < 2:  # noqa: PLR2004
                positions[t] = prev_rebal_pos
                continue

            # Determine quantile thresholds from the cross-section at time t.
            valid_mom = mom_at_t[valid_mask]
            upper_threshold = float(np.percentile(valid_mom, (1.0 - self.top_quantile) * 100.0))
            lower_threshold = float(np.percentile(valid_mom, self.bottom_quantile * 100.0))

            new_pos = np.zeros(_n_assets)
            for j in range(_n_assets):
                if not valid_mask[j]:
                    continue
                vol_scalar = min(self.vol_target / vol_matrix[t, j], self.max_leverage)
                if mom_at_t[j] >= upper_threshold:
                    new_pos[j] = vol_scalar  # long
                elif mom_at_t[j] <= lower_threshold:
                    new_pos[j] = -vol_scalar  # short
                # else: flat (leave as 0.0)

            positions[t] = new_pos
            prev_rebal_pos = new_pos

        return positions
