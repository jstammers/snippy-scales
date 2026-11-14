"""Panel target generation must equal the per-symbol path it replaces.

``execution.panel_targets`` computes a whole universe at once: single-asset
strategies through ``Strategy.generate_signals_panel`` (``.over(symbol)``
window expressions), cross-sectional ones on a union-calendar grid built from
row indices. Each is checked here against the per-symbol computation, which
``tests/test_strategy_causality.py`` and ``tests/test_vector_parity.py`` build on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from snippy_scales.evaluation import WalkForward, make_folds
from snippy_scales.execution import (
    MarketData,
    compute_targets,
    panel_targets,
    stack_frames,
    vector,
)
from snippy_scales.execution.strategy import _align
from snippy_scales.strategies import (
    CrossSectionalMomentum,
    MeanReversion,
    MultiAssetStrategy,
    Strategy,
    TimeSeriesMomentum,
    TrendFollowing,
)
from snippy_scales.strategies.momentum_cs import _rolling_nanstd
from tests.conftest import daily_bars, synthetic_future

if TYPE_CHECKING:
    from collections.abc import Mapping

_SINGLE: list[Strategy] = [
    TrendFollowing(fast_period=5, slow_period=20, vol_lookback=10),
    TimeSeriesMomentum(lookback=20, skip_recent=3, vol_lookback=10),
    TimeSeriesMomentum(lookback=20, vol_lookback=10, rebal_freq=5),
    MeanReversion(lookback=15, entry_z=1.0, exit_z=0.3, vol_lookback=10),  # default panel path
]
_MULTI: list[MultiAssetStrategy] = [
    CrossSectionalMomentum(lookback=20, skip_recent=3, vol_lookback=10, rebal_freq=5),
    CrossSectionalMomentum(lookback=10, skip_recent=0, vol_lookback=5, rebal_freq=1),
]


def _closes(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))


def _universe(n: int = 300) -> dict[str, pl.DataFrame]:
    """Six symbols: one lists late, one delists early, one has gaps, one is short."""
    frames = {}
    for i, symbol in enumerate("ABCDEF"):
        frame = daily_bars(_closes(n, seed=50 + i))
        if symbol == "B":
            frame = frame.slice(60)
        elif symbol == "C":
            frame = frame.slice(0, n - 80)
        elif symbol == "D":
            frame = frame.filter(pl.int_range(pl.len()) % 7 != 3)
        elif symbol == "E":
            frame = frame.slice(100, 30)
        frames[symbol] = frame
    return frames


def _reference_align(
    frames: Mapping[str, pl.DataFrame],
) -> tuple[dict[str, pl.DataFrame], dict[str, np.ndarray]]:
    """The original alignment: one join per symbol on the union of bar closes."""
    calendar = (
        pl.concat([f.select(pl.col("ts_init")) for f in frames.values()]).unique().sort("ts_init")
    )
    aligned: dict[str, pl.DataFrame] = {}
    real: dict[str, np.ndarray] = {}
    for symbol, frame in frames.items():
        frame = frame.sort("ts_init")
        values = [c for c in frame.columns if c != "ts_init"]
        joined = calendar.join(
            frame.with_columns(pl.lit(True).alias("__bar")),  # noqa: FBT003
            on="ts_init",
            how="left",
        )
        has_bar = joined["__bar"].fill_null(False)  # noqa: FBT003
        if frame.height:
            first, last = frame["ts_init"][0], frame["ts_init"][-1]
            alive = pl.col("ts_init").is_between(first, last)
            fills = [
                pl.when(alive)
                .then(pl.lit(0.0) if c == "volume" else pl.col(c).forward_fill())
                .otherwise(None)
                .alias(c)
                for c in values
            ]
            joined = joined.with_columns(
                pl.when(pl.col("__bar")).then(pl.col(c)).otherwise(f).alias(c)
                for c, f in zip(values, fills, strict=True)
            )
        aligned[symbol] = joined.select(frame.columns)
        real[symbol] = has_bar.to_numpy()
    return aligned, real


def _reference_targets(
    research: Strategy | MultiAssetStrategy, frames: Mapping[str, pl.DataFrame]
) -> dict[str, dict[int, float]]:
    """Per-symbol targets as ``compute_targets`` produced them before panels."""

    def clean(values: pl.Series) -> np.ndarray:
        arr = values.cast(pl.Float64).fill_null(0.0).to_numpy()
        return np.where(np.isfinite(arr), arr, 0.0)

    def ns(frame: pl.DataFrame) -> list[int]:
        return frame["ts_init"].dt.epoch("ns").to_list()

    targets: dict[str, dict[int, float]] = {}
    if isinstance(research, MultiAssetStrategy):
        aligned, real = _reference_align(frames)
        signals = research.generate_signals(aligned)
        for s, frame in aligned.items():
            mask = real[s]
            ts = np.array(ns(frame))[mask].tolist()
            targets[s] = dict(zip(ts, clean(signals[s])[mask].tolist(), strict=True))
    else:
        for s, frame in frames.items():
            frame = frame.sort("ts_init")
            values = clean(research.generate_signals(frame)).tolist()
            targets[s] = dict(zip(ns(frame), values, strict=True))
    ends = {s: max(ns(f)) for s, f in frames.items() if f.height}
    if len(frames) >= 2:  # noqa: PLR2004
        horizon = max(ends.values())
        for s, end in ends.items():
            if end < horizon:
                targets[s][end] = 0.0
    return targets


@pytest.mark.parametrize("strategy", _SINGLE + _MULTI, ids=lambda s: type(s).__name__)
def test_panel_targets_equal_per_symbol_targets(strategy: Strategy | MultiAssetStrategy) -> None:
    frames = _universe()
    assert compute_targets(strategy, frames) == _reference_targets(strategy, frames)


@pytest.mark.parametrize("strategy", _SINGLE, ids=lambda s: type(s).__name__)
def test_panel_signals_equal_generate_signals(strategy: Strategy) -> None:
    frames = _universe()
    panel = stack_frames(frames)
    stacked = strategy.generate_signals_panel(panel.bars)
    for k, (symbol, frame) in enumerate(frames.items()):
        own = strategy.generate_signals(frame.sort("ts_init"))
        part = stacked.slice(int(panel.starts[k]), frame.height)
        np.testing.assert_array_equal(
            part.cast(pl.Float64).fill_null(np.nan).to_numpy(),
            own.cast(pl.Float64).fill_null(np.nan).to_numpy(),
            err_msg=symbol,
        )


def test_aligned_grid_equals_per_symbol_joins() -> None:
    frames = _universe()
    frames["G"] = frames["A"].head(0)  # a symbol with no bars at all
    grid, real, calendar = _align(stack_frames(frames))
    reference, reference_real = _reference_align(frames)
    t_n = calendar.size
    for k, symbol in enumerate(frames):
        part = grid.slice(k * t_n, t_n)
        assert part.equals(reference[symbol]), symbol
        np.testing.assert_array_equal(real[k * t_n : (k + 1) * t_n], reference_real[symbol])


def test_rolling_nanstd_equals_numpy() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(200, 7))
    values[rng.random(values.shape) < 0.2] = np.nan
    values[:50, 3] = np.nan  # an all-NaN stretch
    window = 12
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        expected = np.stack(
            [np.nanstd(values[k : k + window], axis=0) for k in range(200 - window + 1)]
        )
    np.testing.assert_allclose(_rolling_nanstd(values, window), expected, rtol=1e-12)


def test_panel_rejects_wrong_length() -> None:
    class Short(Strategy):
        def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
            return pl.Series(np.zeros(bars.height - 1))

    with pytest.raises(ValueError, match="returned"):
        compute_targets(Short(), _universe())


def _market(frames: dict[str, pl.DataFrame]) -> MarketData:
    return MarketData.from_frames(frames, {s: synthetic_future(s) for s in frames})


@pytest.mark.parametrize("strategy", _SINGLE, ids=lambda s: type(s).__name__)
def test_targets_shared_across_folds_equal_per_fold_targets(strategy: Strategy) -> None:
    """Single-asset targets computed once (causal per symbol) equal each fold's own."""
    data = _market(_universe())
    folds = make_folds(data.closes(union=True), n_splits=4, min_train=60)
    shared = WalkForward(data, folds, engine="vector").run_folds(strategy)
    for fold, run in zip(folds, shared, strict=True):
        own = vector.run_backtest(strategy, data, start=fold.test_start, end=fold.test_end)
        assert run.total_orders == own.total_orders
        assert run.equity.equals(own.equity)
        assert run.returns.equals(own.returns)


def test_precomputed_targets_refuse_cross_sectional_strategies() -> None:
    data = _market(_universe())
    research = _MULTI[0]
    assert vector.precompute_targets(research, data) is None
    targets = panel_targets(_SINGLE[0], stack_frames(data.frames))
    with pytest.raises(ValueError, match="single-asset"):
        vector.run_backtest(research, data, targets=targets)
