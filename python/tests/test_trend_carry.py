"""TrendCarry: trend plus asset-class-relative carry in staggered tranches."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from snippy_scales.strategies import TrendCarry
from snippy_scales.strategies.trend_carry import asset_class

_N = 120


def _bars(seed: int, n: int = _N, *, drift: float = 0.0, carry: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    frame = pl.DataFrame({"close": close})
    if carry:
        frame = frame.with_columns(carry=pl.Series(rng.normal(0.0, 0.05, n)))
    return frame


def _universe(seed: int = 0) -> dict[str, pl.DataFrame]:
    # Two energy and two bond roots: each has a peer for the relative signal.
    return {
        s: _bars(seed + i) for i, s in enumerate(["CL.ROLL", "BRN.IFEU.ROLL", "ZN.ROLL", "ZB.ROLL"])
    }


def _fast(**kwargs: Any) -> TrendCarry:  # noqa: ANN401 — any TrendCarry argument
    params: dict[str, Any] = {
        "lookback": 20,
        "carry_smooth": 8,
        "vol_lookback": 10,
        "rebal_freq": 5,
    }
    return TrendCarry(**{**params, **kwargs})


def test_asset_class_of_rolled_symbols() -> None:
    assert asset_class("ES.ROLL") == "equity"
    assert asset_class("BRN.IFEU.ROLL") == "energy"
    assert asset_class("XYZ.ROLL") == "XYZ"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback": 0},
        {"carry_weight": -0.1},
        {"carry_weight": 1.1},
        {"carry_smooth": 0},
        {"vol_target": 0.0},
        {"vol_lookback": 1},
        {"vol_floor": 0.0},
        {"max_leverage": 0.0},
        {"rebal_freq": 0},
    ],
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        TrendCarry(**kwargs)  # type: ignore[arg-type]


def test_frames_must_share_a_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        TrendCarry().generate_signals({"CL.ROLL": _bars(0), "ZN.ROLL": _bars(1, 50)})


def test_pure_trend_is_the_sign_of_the_lookback_return_at_target_volatility() -> None:
    bars = {
        "CL.ROLL": _bars(0, drift=0.004, carry=False),
        "ZN.ROLL": _bars(1, drift=-0.004, carry=False),
    }
    out = _fast(carry_weight=0.0, rebal_freq=1, vol_target=0.10).generate_signals(bars)
    for symbol, sign in (("CL.ROLL", 1.0), ("ZN.ROLL", -1.0)):
        position = out[symbol].to_numpy()
        close = bars[symbol]["close"].to_numpy()
        assert np.all(position[:20] == 0.0)  # warm-up
        ret = close[1:] / close[:-1] - 1.0
        vol = np.std(ret[-10:], ddof=1) * np.sqrt(252)
        assert position[-1] == pytest.approx(sign * 0.10 / vol)


def test_carry_signal_is_relative_within_the_asset_class() -> None:
    n = 60
    flat = np.full(n, 100.0) * np.exp(np.cumsum(np.tile([0.01, -0.01], n // 2)))
    bars = {
        "CL.ROLL": pl.DataFrame({"close": flat, "carry": np.full(n, 0.10)}),
        "BRN.IFEU.ROLL": pl.DataFrame({"close": flat, "carry": np.full(n, 0.04)}),
        "ZN.ROLL": pl.DataFrame({"close": flat, "carry": np.full(n, 0.02)}),  # no bond peer
    }
    out = _fast(carry_weight=1.0, rebal_freq=1).generate_signals(bars)
    assert out["CL.ROLL"][-1] > 0  # above the energy average, though both are positive
    assert out["BRN.IFEU.ROLL"][-1] < 0
    assert out["CL.ROLL"][-1] == pytest.approx(-out["BRN.IFEU.ROLL"][-1])
    assert out["ZN.ROLL"][-1] == 0.0


def test_custom_groups_override_the_asset_class() -> None:
    bars = _universe()
    one = dict.fromkeys(bars, "all")
    default = _fast(carry_weight=1.0).generate_signals(bars)
    pooled = _fast(carry_weight=1.0, groups=one).generate_signals(bars)
    assert any(not np.allclose(default[s].to_numpy(), pooled[s].to_numpy()) for s in bars)


def test_missing_carry_column_leaves_trend_only() -> None:
    bars = {s: f.drop("carry") for s, f in _universe().items()}
    blended = _fast(carry_weight=0.5).generate_signals(bars)
    trend = _fast(carry_weight=0.0).generate_signals(bars)
    for symbol in bars:
        np.testing.assert_allclose(blended[symbol].to_numpy(), 0.5 * trend[symbol].to_numpy())


def test_position_is_the_average_of_the_last_daily_targets() -> None:
    bars = _universe()
    daily = _fast(rebal_freq=1).generate_signals(bars)
    tranched = _fast(rebal_freq=5).generate_signals(bars)
    for symbol in bars:
        expected = daily[symbol].rolling_mean(5, min_samples=1).to_numpy()
        np.testing.assert_allclose(tranched[symbol].to_numpy(), expected, atol=1e-12)


def test_unlisted_rows_get_no_position() -> None:
    bars = _universe()
    late = bars["ZB.ROLL"].with_columns(
        pl.when(pl.int_range(pl.len()) < 40).then(None).otherwise(pl.col("close")).alias("close")
    )
    out = _fast().generate_signals({**bars, "ZB.ROLL": late})
    position = out["ZB.ROLL"].to_numpy()
    assert np.all(position[:40] == 0.0)
    assert np.all(np.isfinite(position))
    assert np.any(position[70:] != 0.0)


def test_volatility_floor_caps_leverage_on_stale_prices() -> None:
    close = np.concatenate([100.0 * np.exp(np.cumsum(np.full(40, 0.001))), np.full(40, 104.0)])
    bars = {"CL.ROLL": pl.DataFrame({"close": close}), "ZN.ROLL": _bars(1, 80, carry=False)}
    out = _fast(carry_weight=0.0, vol_target=0.4, vol_floor=0.02).generate_signals(bars)
    assert np.abs(out["CL.ROLL"].to_numpy()).max() <= 0.4 / 0.02 + 1e-9


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(1, _N))
def test_causal_with_carry(seed: int, k: int) -> None:
    universe = _universe(seed)
    strategy = _fast()
    full = strategy.generate_signals(universe)
    prefix = strategy.generate_signals({s: b.head(k) for s, b in universe.items()})
    for symbol in universe:
        np.testing.assert_allclose(
            prefix[symbol].to_numpy(), full[symbol].head(k).to_numpy(), rtol=1e-12, atol=1e-12
        )
