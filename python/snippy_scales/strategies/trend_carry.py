"""Trend and carry on futures (``trend-carry``).

Two return sources that the futures literature finds to be nearly uncorrelated,
combined instrument by instrument:

* **Trend** (Moskowitz, Ooi & Pedersen, 2012): the sign of the past *lookback*-bar
  return.
* **Carry** (Koijen, Moskowitz, Pedersen & Vrugt, 2018): the return a future earns
  if its curve does not move, measured as the annualised roll yield between the
  front and second contract (the ``carry`` column of a rolled symbol's frame, see
  :func:`snippy_scales.data.futures.roll_yield`). The carry signal is **relative
  within an asset class**: long the contracts whose smoothed carry is above their
  class average and short those below, so it takes no view on the class as a whole
  (which trend already does) and its seasonal and interest-rate level cancels.

.. math::

    s_{i,t} = (1 - w)\\,\\operatorname{sign}(r_{i,t-L,t})
              + w\\,\\operatorname{sign}\\bigl(\\bar c_{i,t} - \\bar c_{g(i),t}\\bigr),
    \\qquad
    p_{i,t} = \\frac{1}{H}\\sum_{h=0}^{H-1} s_{i,t-h}
              \\min\\!\\left(\\frac{\\sigma^*}{\\hat\\sigma_{i,t-h}}, \\lambda\\right)

The position is the average of the last *H* = ``rebal_freq`` daily targets. That is
the book of *H* tranches, each rebalanced every *H* bars on a different day, so
the result does not depend on which day of the month the rebalance falls on.

With ``portfolio_vol_target`` set, the whole book is then scaled each bar so that its
**ex-ante** volatility (today's positions applied to the last ``cov_lookback`` bars
of returns, so correlations between positions count) equals the target. Per-instrument
sizing fixes each position's risk but not the book's: when trends line up across
markets the positions become correlated and the book's risk rises. The scale takes
that back out.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from snippy_scales._constants import TRADING_DAYS_PER_YEAR
from snippy_scales.strategies.momentum_cs import MultiAssetStrategy

#: Asset class of each futures root, the peer group of the relative carry signal.
ASSET_CLASSES: dict[str, str] = {
    **dict.fromkeys(
        ("ES", "NQ", "YM", "RTY", "NKD", "EMD", "Z", "MME", "MFS", "FESX", "FDAX"), "equity"
    ),
    **dict.fromkeys(
        ("ZT", "ZF", "ZN", "ZB", "UB", "TN", "R", "FGBS", "FGBM", "FGBL", "FGBX"), "bond"
    ),
    **dict.fromkeys(("6A", "6B", "6C", "6E", "6J", "6S", "6N", "6M"), "fx"),
    **dict.fromkeys(("CL", "HO", "RB", "NG", "BRN", "G"), "energy"),
    **dict.fromkeys(("GC", "SI", "HG", "PL", "PA"), "metal"),
    **dict.fromkeys(
        (
            *("ZC", "ZS", "ZW", "KE", "ZL", "ZM", "LE", "HE", "GF"),
            *("C", "RC", "W", "SB", "KC", "CC", "CT"),
        ),
        "agriculture",
    ),
    "VX": "volatility",
}

_MIN_GROUP = 2  # a relative signal needs a peer


def asset_class(symbol: str) -> str:
    """Asset class of a futures symbol (``BRN.IFEU.ROLL`` -> ``energy``).

    A root that is not in :data:`ASSET_CLASSES` is a class of its own.
    """
    root = symbol.split(".")[0]
    return ASSET_CLASSES.get(root, root)


def _rolling_nanmean(values: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Trailing mean over *window* rows per column, ignoring NaN.

    NaN where fewer than *min_count* of the rows are finite.
    """
    finite = np.isfinite(values)
    zero = np.zeros((1, values.shape[1]))
    total = np.vstack([zero, np.cumsum(np.where(finite, values, 0.0), axis=0)])
    count = np.vstack([zero, np.cumsum(finite, axis=0)])
    lo = np.maximum(np.arange(len(values)) + 1 - window, 0)
    hi = np.arange(len(values)) + 1
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_count, (total[hi] - total[lo]) / n, np.nan)


class TrendCarry(MultiAssetStrategy):
    """Time-series trend blended with asset-class-relative carry, volatility-targeted.

    Args:
        lookback: Trend return window in bars (default 252, twelve months).
        carry_weight: Weight of the carry signal, in ``[0, 1]``; trend gets the rest.
            ``0`` is time-series momentum, ``1`` is carry alone.
        carry_smooth: Bars the roll yield is averaged over (default 252: one year
            removes the seasonal shape of commodity curves).
        vol_target: Target annualised volatility per instrument (default 0.40, as in
            Moskowitz, Ooi & Pedersen).
        vol_lookback: Window of the realised-volatility estimate in bars.
        vol_floor: Lowest annualised volatility assumed, so a run of stale prices
            cannot produce extreme leverage.
        max_leverage: Cap on ``vol_target / volatility``. Bond futures need 10-25x at
            a 40% target, so the default (50) rarely binds.
        rebal_freq: Holding period in bars. The position is the average of the last
            ``rebal_freq`` daily targets (staggered tranches); ``1`` trades the daily
            target.
        groups: ``symbol -> peer group`` for the relative carry signal. Defaults to
            the asset class of each futures root (:func:`asset_class`).
        long_only: Peer groups (asset classes by default) whose trend signal is
            long-or-flat, comma-separated (``"equity"``, ``"equity,bond"``): a negative
            trend holds no position instead of a short one. Assets that pay a risk
            premium tend to lose on the short side of a trend rule. Empty (default)
            trades every group both ways. The carry signal is not affected.
        portfolio_vol_target: Annualised volatility the whole book is scaled to,
            assuming capital is split equally over the symbols (the engines'
            default). ``0`` (default) leaves the book unscaled.
        cov_lookback: Bars of returns the book's ex-ante volatility is measured on.
        max_portfolio_scale: Cap on the book-level scale, so a quiet, uncorrelated
            spell cannot lever the book without limit.

    Frames without a ``carry`` column (or with null carry) get a zero carry signal,
    as does a symbol with no peer that has carry on that date.

    Raises:
        ValueError: On invalid parameter values.
    """

    def __init__(  # noqa: PLR0913 — one argument per documented parameter
        self,
        lookback: int = 252,
        carry_weight: float = 0.5,
        carry_smooth: int = 252,
        vol_target: float = 0.40,
        vol_lookback: int = 63,
        vol_floor: float = 0.02,
        max_leverage: float = 50.0,
        rebal_freq: int = 21,
        groups: dict[str, str] | None = None,
        long_only: str = "",
        portfolio_vol_target: float = 0.0,
        cov_lookback: int = 126,
        max_portfolio_scale: float = 2.0,
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be ≥ 1, got {lookback}")
        if not 0.0 <= carry_weight <= 1.0:
            raise ValueError(f"carry_weight must be in [0, 1], got {carry_weight}")
        if carry_smooth < 1:
            raise ValueError(f"carry_smooth must be ≥ 1, got {carry_smooth}")
        if vol_target <= 0:
            raise ValueError(f"vol_target must be positive, got {vol_target}")
        if vol_lookback < 2:  # noqa: PLR2004 — a standard deviation needs two returns
            raise ValueError(f"vol_lookback must be ≥ 2, got {vol_lookback}")
        if vol_floor <= 0:
            raise ValueError(f"vol_floor must be positive, got {vol_floor}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {max_leverage}")
        if rebal_freq < 1:
            raise ValueError(f"rebal_freq must be ≥ 1, got {rebal_freq}")
        if portfolio_vol_target < 0:
            raise ValueError(f"portfolio_vol_target must be ≥ 0, got {portfolio_vol_target}")
        if cov_lookback < 2:  # noqa: PLR2004 — a variance needs two returns
            raise ValueError(f"cov_lookback must be ≥ 2, got {cov_lookback}")
        if max_portfolio_scale <= 0:
            raise ValueError(f"max_portfolio_scale must be positive, got {max_portfolio_scale}")

        self.lookback = lookback
        self.carry_weight = carry_weight
        self.carry_smooth = carry_smooth
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.vol_floor = vol_floor
        self.max_leverage = max_leverage
        self.rebal_freq = rebal_freq
        self.groups = groups
        self.long_only = long_only
        self.portfolio_vol_target = portfolio_vol_target
        self.cov_lookback = cov_lookback
        self.max_portfolio_scale = max_portfolio_scale

    def generate_signals(self, multi_bars: dict[str, pl.DataFrame]) -> dict[str, pl.Series]:
        """Target positions (units of equity) for every symbol.

        Args:
            multi_bars: ``symbol -> bars``, time-aligned and of equal length, with a
                ``close`` column (null where the symbol is not listed) and optionally
                a ``carry`` column (annualised roll yield).

        Returns:
            ``symbol -> position`` Series of the same length; zero while a signal
            warms up and where the symbol is not listed.

        Raises:
            ValueError: If the frames differ in length.
        """
        symbols = list(multi_bars)
        lengths = {len(frame) for frame in multi_bars.values()}
        if len(lengths) > 1:
            raise ValueError(f"all bar frames must have the same length, got {sorted(lengths)}")

        close = self._matrix(multi_bars, "close")
        listed = np.isfinite(close)
        trend = self._trend(close)
        flat_shorts = {g.strip() for g in self.long_only.split(",") if g.strip()}
        if flat_shorts:
            groups = self.groups or {}
            held_long = np.array([groups.get(s, asset_class(s)) in flat_shorts for s in symbols])
            trend = np.where(held_long[None, :], np.maximum(trend, 0.0), trend)
        signal = (1.0 - self.carry_weight) * trend
        if self.carry_weight > 0.0:
            carry = self._matrix(multi_bars, "carry")
            signal = signal + self.carry_weight * self._relative_carry(carry, listed, symbols)

        daily = np.where(listed, signal * self._leverage(close), 0.0)
        positions = _rolling_nanmean(daily, self.rebal_freq, 1)
        positions = np.where(listed, positions, 0.0)
        if self.portfolio_vol_target > 0.0 and symbols:
            positions = positions * self._portfolio_scale(positions, close)[:, None]
        return {s: pl.Series("position", positions[:, j]) for j, s in enumerate(symbols)}

    # ------------------------------------------------------------------
    # Signal components
    # ------------------------------------------------------------------

    @staticmethod
    def _matrix(multi_bars: dict[str, pl.DataFrame], column: str) -> np.ndarray:
        """``(bars, symbols)`` float matrix of *column*; NaN where null or absent."""
        n = len(next(iter(multi_bars.values()))) if multi_bars else 0
        columns = [
            frame[column].cast(pl.Float64).fill_null(np.nan).to_numpy()
            if column in frame.columns
            else np.full(n, np.nan)
            for frame in multi_bars.values()
        ]
        return np.column_stack(columns) if columns else np.empty((0, 0))

    def _trend(self, close: np.ndarray) -> np.ndarray:
        """Sign of the *lookback*-bar return; zero until it exists."""
        trend = np.zeros_like(close)
        if len(close) > self.lookback:
            with np.errstate(invalid="ignore", divide="ignore"):
                ret = close[self.lookback :] / close[: -self.lookback] - 1.0
            trend[self.lookback :] = np.where(np.isfinite(ret), np.sign(ret), 0.0)
        return trend

    def _relative_carry(
        self, carry: np.ndarray, listed: np.ndarray, symbols: list[str]
    ) -> np.ndarray:
        """Sign of smoothed carry minus its peer-group mean; zero without a peer."""
        smooth = _rolling_nanmean(carry, self.carry_smooth, max(1, self.carry_smooth // 4))
        smooth = np.where(listed, smooth, np.nan)
        groups = self.groups or {}
        labels = np.array([groups.get(s, asset_class(s)) for s in symbols])
        out = np.zeros_like(smooth)
        for label in np.unique(labels):
            cols = np.flatnonzero(labels == label)
            block = smooth[:, cols]
            valid = np.isfinite(block)
            count = valid.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(valid, block, 0.0).sum(axis=1, keepdims=True) / count
            relative = np.where(valid & (count >= _MIN_GROUP), block - mean, 0.0)
            out[:, cols] = np.sign(relative)
        return out

    def _portfolio_scale(self, positions: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Per-bar scale taking the book's ex-ante volatility to the target.

        The book at bar *t* is ``positions[t] / n_symbols`` (equal capital weights).
        Its volatility is that of the returns it would have earned over the last
        ``cov_lookback`` bars, which is ``sqrt(p' Σ p)`` for the sample covariance
        ``Σ`` of those bars. The scale is 1 until that many bars exist.
        """
        n, m = positions.shape
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.zeros((1, m)), np.where(np.isfinite(ret), ret, 0.0)])
        book = positions / m
        scale = np.ones(n)
        window = self.cov_lookback
        for t in range(window, n):
            hypothetical = ret[t - window + 1 : t + 1] @ book[t]
            vol = float(np.std(hypothetical, ddof=1)) * TRADING_DAYS_PER_YEAR**0.5
            if vol > 0.0:
                scale[t] = min(self.portfolio_vol_target / vol, self.max_portfolio_scale)
        return scale

    def _leverage(self, close: np.ndarray) -> np.ndarray:
        """``vol_target / realised volatility`` (floored, capped); zero while it warms up."""
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
        window = self.vol_lookback
        mean = _rolling_nanmean(ret, window, window)
        mean_sq = _rolling_nanmean(ret * ret, window, window)
        variance = np.maximum(mean_sq - mean * mean, 0.0) * window / (window - 1)
        vol = np.sqrt(variance * TRADING_DAYS_PER_YEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            leverage = np.minimum(
                self.vol_target / np.maximum(vol, self.vol_floor), self.max_leverage
            )
        return np.where(np.isfinite(vol), leverage, 0.0)
