"""execution.vector: the numba screening engine must reproduce Nautilus.

Every case runs the same strategy through ``execution.run_backtest`` (Nautilus)
and ``execution.vector.run_backtest`` and compares equity, the daily returns
series, return statistics, order counts and rolls. Synthetic bars carry ample
volume, so Nautilus fills at the close too (see the module docstring for what
the vector engine does not model).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from snippy_scales.evaluation import WalkForward, make_folds, run_study, summarize_study
from snippy_scales.execution import (
    ENGINES,
    SHARPE,
    BacktestRun,
    Costs,
    MarketData,
    run_backtest,
    vector,
)
from snippy_scales.strategies import (
    CrossSectionalMomentum,
    Strategy,
    TimeSeriesMomentum,
    TrendFollowing,
)
from tests.conftest import daily_bars, synthetic_data, synthetic_future
from tests.test_futures_rolls import _market


class _Fixed(Strategy):
    def __init__(self, value: float) -> None:
        self.value = value

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        return pl.Series("position", np.full(bars.height, self.value))


def _closes(n: int, seed: int, drift: float = 0.0003, vol: float = 0.012) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def _weekdays(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop bars opening on a Saturday or Sunday (gaps, as real daily data has)."""
    return frame.filter(pl.col("ts_event").dt.weekday() <= 5)


def _futures(multipliers: dict[str, int], n: int = 700, weekdays: bool = False) -> MarketData:
    frames = {}
    for i, symbol in enumerate(multipliers):
        frame = daily_bars(_closes(n, seed=10 + i, vol=0.008 + 0.004 * i))
        frames[symbol] = _weekdays(frame) if weekdays else frame
    instruments = {s: synthetic_future(s, multiplier=m) for s, m in multipliers.items()}
    return MarketData.from_frames(frames, instruments)


def _ragged(n: int = 600) -> MarketData:
    """Six symbols; B lists 150 bars late and C delists 200 bars early (an index universe)."""
    frames = {}
    for i, symbol in enumerate("ABCDEF"):
        frame = daily_bars(_closes(n, seed=30 + i))
        if symbol == "B":
            frame = frame.slice(150)
        elif symbol == "C":
            frame = frame.slice(0, n - 200)
        frames[symbol] = frame
    return MarketData.from_frames(frames, {s: synthetic_future(s) for s in frames})


def _rolled_es() -> MarketData:
    contracts, frames = _market(carry=0.001)
    return MarketData.from_outrights("ES.FUT", frames, contracts)


def assert_parity(ref: BacktestRun, got: BacktestRun) -> None:
    assert got.equity["ts"].equals(ref.equity["ts"])
    np.testing.assert_allclose(
        got.equity["equity"].to_numpy(), ref.equity["equity"].to_numpy(), rtol=1e-6
    )
    assert got.returns["ts"].equals(ref.returns["ts"])
    np.testing.assert_allclose(
        got.returns["return"].to_numpy(), ref.returns["return"].to_numpy(), atol=1e-7
    )
    assert set(got.stats_returns) == set(ref.stats_returns)
    for key, value in ref.stats_returns.items():
        assert got.stats_returns[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key
    assert got.sharpe == pytest.approx(ref.sharpe, abs=1e-4)
    assert got.total_return == pytest.approx(ref.total_return, abs=1e-6)
    assert got.total_orders == ref.total_orders
    assert got.rolls == ref.rolls
    assert ref.rejections == ()


_FEE = Costs(commission_rate=5e-4)

CASES: dict[str, tuple[Any, MarketData, dict[str, Any]]] = {
    "constant long": (_Fixed(1.0), synthetic_data({"X": _closes(400, 1)}), {}),
    "trend, rate fee": (TrendFollowing(), synthetic_data({"X": _closes(800, 2)}), {"costs": _FEE}),
    "trend, band, window": (
        TrendFollowing(fast_period=10, slow_period=40),
        synthetic_data({"X": _closes(800, 3)}),
        {"costs": _FEE, "rebalance_band": 0.02, "start": "2020-06-01", "end": "2021-09-30"},
    ),
    "weekday gaps": (
        TrendFollowing(fast_period=10, slow_period=40),
        _futures({"A": 1}, weekdays=True),
        {"costs": _FEE, "start": "2020-04-01"},
    ),
    "multipliers, leverage, shorts": (
        TimeSeriesMomentum(lookback=60, vol_lookback=20, vol_target=0.4, max_leverage=3.0),
        _futures({"ES": 50, "NQ": 20, "CL": 1000}, weekdays=True),
        {"costs": _FEE, "initial_capital": 1e8},
    ),
    "per-contract fee, band": (
        TimeSeriesMomentum(lookback=60, vol_lookback=20, vol_target=0.4),
        _futures({"ES": 50, "NQ": 20, "CL": 1000}),
        {"costs": Costs(commission_per_contract=2.25), "initial_capital": 1e8,
         "rebalance_band": 0.01},
    ),
    "custom weights": (
        TrendFollowing(fast_period=10, slow_period=40),
        _futures({"ES": 50, "NQ": 20}),
        {"costs": _FEE, "initial_capital": 1e8, "weights": {"ES": 0.7, "NQ": 0.6}},
    ),
    "cross-sectional": (
        CrossSectionalMomentum(lookback=60, skip_recent=5, rebal_freq=5),
        synthetic_data({s: _closes(600, 20 + i) for i, s in enumerate("ABCDEF")}),
        {"costs": _FEE, "start": "2020-05-01"},
    ),
    "ragged universe, cross-sectional": (
        CrossSectionalMomentum(lookback=60, skip_recent=5, rebal_freq=5),
        _ragged(),
        {"costs": _FEE, "start": "2020-04-01", "initial_capital": 1e8},
    ),
    "ragged universe, delisting exit": (
        TimeSeriesMomentum(lookback=60, vol_lookback=20, vol_target=0.4),
        _ragged(),
        {"costs": _FEE, "initial_capital": 1e8},
    ),
    "rolled ES, per-contract fee": (
        TrendFollowing(fast_period=10, slow_period=40),
        _rolled_es(),
        {"costs": Costs(commission_per_contract=2.25)},
    ),
    "rolled ES, band, window": (
        _Fixed(-0.5),
        _rolled_es(),
        {"costs": _FEE, "rebalance_band": 0.5, "start": "2020-05-01", "initial_capital": 1e8},
    ),
    # Nautilus slips every fill by one tick at probability 1, which is the expected cost.
    "slippage, probability 1": (
        TrendFollowing(fast_period=10, slow_period=40),
        _futures({"ES": 50}),
        {"costs": Costs(slippage_prob=1.0), "initial_capital": 1e8},
    ),
}  # fmt: skip


def test_large_universe_is_not_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    """More symbols than Nautilus's default 100 orders/second all trade on one close."""
    data = synthetic_data({f"S{i:03d}": _closes(40, 100 + i) for i in range(150)})
    run = run_backtest(_Fixed(1.0), data, initial_capital=1e8, costs=_FEE)
    assert run.rejections == ()
    assert_parity(run, vector.run_backtest(_Fixed(1.0), data, initial_capital=1e8, costs=_FEE))

    from snippy_scales.execution import backtest as backtest_module

    monkeypatch.setattr(backtest_module, "_MAX_ORDER_SUBMIT_RATE", "100/00:00:01")
    throttled = run_backtest(_Fixed(1.0), data, initial_capital=1e8, costs=_FEE)
    assert any("RATE_LIMIT_EXCEEDED" in r for r in throttled.rejections)


@pytest.mark.parametrize("case", list(CASES))
def test_vector_matches_nautilus(case: str) -> None:
    research, data, kwargs = CASES[case]
    assert_parity(
        run_backtest(research, data, **kwargs), vector.run_backtest(research, data, **kwargs)
    )


def test_non_positive_close_is_neither_marked_nor_traded() -> None:
    close = _closes(300, 5)
    close[150] = -1.0  # CL.c.0 closed at -2.67 on 2020-04-21
    data = synthetic_data({"X": close})
    ref = run_backtest(TrendFollowing(fast_period=5, slow_period=20), data)
    got = vector.run_backtest(TrendFollowing(fast_period=5, slow_period=20), data)
    assert_parity(ref, got)


def test_rolls_follow_the_held_contract() -> None:
    """A roll is recorded (and closes the old contract) only once the symbol has traded."""
    data = _rolled_es()
    schedule = data.schedules["ES.FUT"]
    late = schedule.rolls[2][0] + 3 * 86_400_000_000_000
    run = vector.run_backtest(_Fixed(1.0), data, start=late)
    assert [r[0] for r in run.rolls] == [r[0] for r in schedule.rolls if r[0] > late]
    fills = run.fills
    for ts, old, new in run.rolls:
        at_roll = fills[fills["ts_event"].astype("int64") == ts]
        assert set(at_roll["instrument_id"]) == {old, new}


def test_nautilus_returns_rule() -> None:
    day = 86_400_000_000_000
    ts = np.array([1, 2, 5], dtype=np.int64) * day  # a two-day gap
    post = np.array([99.0, 110.0, 121.0])
    out = vector.nautilus_returns(ts, post, capital=100.0)
    assert out["ts"].dt.epoch("ns").to_list() == [d * day for d in (1, 2, 3, 4)]
    # Day 1's base is the capital (its fees fall into day 1's return); gap days are flat.
    np.testing.assert_allclose(out["return"].to_numpy(), [0.10, 0.0, 0.0, 0.10])
    assert vector.nautilus_returns(ts[:0], post[:0], 100.0).height == 0


def test_engines_share_a_signature() -> None:
    assert set(ENGINES) == {"nautilus", "vector"}
    run = ENGINES["vector"](TrendFollowing(), synthetic_data({"X": _closes(300, 7)}))
    assert isinstance(run, BacktestRun)
    assert SHARPE in run.stats_returns
    assert run.stats_pnls == {} and run.positions.empty
    with pytest.raises(ValueError, match="rebalance_band"):
        vector.run_backtest(TrendFollowing(), synthetic_data({"X": _closes(300, 7)}),
                            rebalance_band=-0.1)  # fmt: skip


def test_price_matrix_is_cached_per_market_data() -> None:
    data = synthetic_data({"X": _closes(300, 8)})
    assert vector._prices(data) is vector._prices(data)
    assert vector._prices(synthetic_data({"X": _closes(300, 8)})) is not vector._prices(data)


def test_walk_forward_engines_agree(tmp_path: Any) -> None:  # noqa: ANN401
    data = synthetic_data({"X": _closes(700, 9, drift=0.0005)})
    folds = make_folds(data.closes(), n_splits=3, min_train=100)
    space = {"fast_period": [5, 10], "slow_period": [40, 80]}
    reports = {}
    for engine in ENGINES:
        wf = WalkForward(data, folds, costs=_FEE, engine=engine)
        study = run_study(
            TrendFollowing, wf, space=space, sampler="grid", storage=None,
            study_name=engine, returns_dir=tmp_path,
        )  # fmt: skip
        assert study.user_attrs["engine"] == engine
        reports[engine] = summarize_study(study)
    nautilus, vec = reports["nautilus"], reports["vector"]
    assert vec.best_params == nautilus.best_params
    assert vec.sharpe == pytest.approx(nautilus.sharpe, abs=1e-4)


def test_walk_forward_rejects_unknown_engine() -> None:
    data = synthetic_data({"X": _closes(300, 9)})
    with pytest.raises(ValueError, match="engine"):
        WalkForward(data, make_folds(data.closes(), n_splits=2, min_train=100), engine="gpu")
