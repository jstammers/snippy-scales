"""Strategy implementations for the snippy-scales backtesting layer.

All strategies implement either :class:`~snippy_scales.strategies.trend.Strategy`
(single-asset) or
:class:`~snippy_scales.strategies.momentum_cs.MultiAssetStrategy` (multi-asset)
and produce signed-float position series that are consumed by the
:mod:`snippy_scales.backtesting.runner` infrastructure.

Available strategies
--------------------

Phase 1 — Baseline, Robust, Low-Dimensional Alpha
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~snippy_scales.strategies.momentum_cs.CrossSectionalMomentum`
    Rank assets by their *J*-month return and go long (short) the top (bottom)
    quantile.  Uses :class:`~snippy_scales.backtesting.runner.BasketRunner`.

:class:`~snippy_scales.strategies.momentum_ts.TimeSeriesMomentum`
    Enter long (short) when the *L*-bar return is positive (negative).
    Classic trend-following signal with volatility-targeted sizing.

:class:`~snippy_scales.strategies.mean_reversion.MeanReversion`
    Enter contrarian positions when the rolling z-scored daily return exceeds
    a threshold.  Tests slippage and transaction-cost modelling.

:class:`~snippy_scales.strategies.trend.TrendFollowing`
    Moving-average crossover trend strategy (fast MA > slow MA → long).
"""

from snippy_scales.strategies.mean_reversion import MeanReversion
from snippy_scales.strategies.momentum_cs import CrossSectionalMomentum, MultiAssetStrategy
from snippy_scales.strategies.momentum_ts import TimeSeriesMomentum
from snippy_scales.strategies.trend import Strategy, TrendFollowing

__all__ = [
    "CrossSectionalMomentum",
    "MeanReversion",
    "MultiAssetStrategy",
    "Strategy",
    "TimeSeriesMomentum",
    "TrendFollowing",
]
