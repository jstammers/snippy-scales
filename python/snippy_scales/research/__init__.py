"""Research layer — feature engineering and stochastic-process diagnostics.

Four submodules:

* :mod:`.features` — general helpers (close-to-close realised vol, z-score,
  futures carry).
* :mod:`.volatility` — range-based volatility estimators (Parkinson,
  Garman–Klass, Rogers–Satchell, Yang–Zhang).  Substantially more efficient
  than close-to-close because they use the high and low the OHLC data already
  carries.
* :mod:`.roughness` — Hurst exponent, variogram, and jump diagnostics.  These
  answer whether a stochastic-volatility model is warranted *before* one is
  fitted.
* :mod:`.sde` — calibration and simulation for Ornstein–Uhlenbeck, Heston and
  rough Bergomi.  The classical baselines any neural model must beat.

Typical diagnostic workflow::

    from snippy_scales.research import yang_zhang_vol, hurst_exponent, jump_ratio

    rv = yang_zhang_vol(bars["high"], bars["low"], bars["open"], bars["close"])
    h = hurst_exponent(np.log(rv.drop_nulls().to_numpy() ** 2))
    jumps = jump_ratio(bars["close"].pct_change().drop_nulls().to_numpy())

Read ``h`` well below 0.5 as support for a rough-volatility model, ``h`` near
0.5 as support for a Markovian one, and a large ``jumps`` as a warning that no
continuous SDE will fit well.
"""

from __future__ import annotations

from snippy_scales.research.features import carry, realised_vol, zscore
from snippy_scales.research.roughness import (
    bipower_variation,
    default_lags,
    hurst_exponent,
    jump_ratio,
    realised_variance,
    variogram,
)
from snippy_scales.research.sde import (
    HestonParams,
    OUParams,
    RoughBergomiParams,
    fit_ou,
    fractional_gaussian_noise,
    simulate_heston,
    simulate_ou,
    simulate_rough_bergomi,
)
from snippy_scales.research.volatility import (
    garman_klass_vol,
    parkinson_vol,
    rogers_satchell_vol,
    yang_zhang_vol,
)

__all__ = [
    # features
    "realised_vol",
    "zscore",
    "carry",
    # volatility estimators
    "parkinson_vol",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "yang_zhang_vol",
    # roughness and jumps
    "hurst_exponent",
    "variogram",
    "default_lags",
    "realised_variance",
    "bipower_variation",
    "jump_ratio",
    # SDE models
    "OUParams",
    "fit_ou",
    "simulate_ou",
    "HestonParams",
    "simulate_heston",
    "RoughBergomiParams",
    "simulate_rough_bergomi",
    "fractional_gaussian_noise",
]
