"""Shared harness for the vectorised-engine spike (see docs/research/vectorised_engines_spike.md).

Every engine is fed the same dense :class:`Panel`: one column per traded
instrument, one row per bar close, with the target already scaled to
``target * weight`` and roll bars marked. The panel reproduces what
``SignalStrategy._flush`` decides at each close, so the engines only differ in
how they simulate, not in what they are asked to trade.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from nautilus_trader.model import (
    AssetClass,
    Currency,
    FuturesContract,
    InstrumentId,
    Price,
    Quantity,
    Symbol,
)

from snippy_scales.evaluation.walk_forward import returns_statistics
from snippy_scales.execution import SHARPE, MarketData, compute_targets, run_backtest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from snippy_scales.execution import BacktestRun

DAY = 86_400_000_000_000


@dataclass(frozen=True)
class Prices:
    """Raw bar closes of every instrument in a :class:`MarketData`, built once."""

    ts: np.ndarray  # int64 [T], union of bar closes
    cols: list[str]  # instrument ids
    close: np.ndarray  # float64 [T, N], NaN where the instrument has no bar
    mult: np.ndarray  # float64 [N]
    incr: np.ndarray  # float64 [N] size increment
    col: dict[str, int]

    @classmethod
    def from_data(cls, data: MarketData) -> Prices:
        cols = list(data.instruments)
        per = {
            iid: (
                np.fromiter((b.ts_init for b in data.bars[iid]), np.int64),
                np.fromiter((float(b.close) for b in data.bars[iid]), np.float64),
            )
            for iid in cols
        }
        ts = np.unique(np.concatenate([t for t, _ in per.values()]))
        close = np.full((ts.size, len(cols)), np.nan)
        for j, iid in enumerate(cols):
            t, c = per[iid]
            close[np.searchsorted(ts, t), j] = c
        # Nautilus neither marks nor trades at a non-positive price (CL.c.0 closed
        # at -2.67 on 2020-04-21): the position keeps its last positive mark.
        close[close <= 0.0] = np.nan
        mult = np.array([float(data.instruments[i].multiplier) for i in cols])
        incr = np.array([float(data.instruments[i].size_increment) for i in cols])
        return cls(ts, cols, close, mult, incr, {c: j for j, c in enumerate(cols)})


@dataclass(frozen=True)
class Panel:
    """What to trade, row by row, for one run window."""

    ts: np.ndarray  # int64 [T]
    cols: list[str]
    close: np.ndarray  # [T, N] raw closes, NaN = no bar
    mult: np.ndarray
    incr: np.ndarray
    target: np.ndarray  # [T, N] target * weight in units of equity; NaN = no decision
    force: np.ndarray  # [T, N] bool: roll leg, ignore the band

    @property
    def ffill_close(self) -> np.ndarray:
        out = self.close.copy()
        for j in range(out.shape[1]):
            col = out[:, j]
            idx = np.where(np.isfinite(col), np.arange(col.size), 0)
            np.maximum.accumulate(idx, out=idx)
            out[:, j] = col[idx]
        return out


def target_matrix(
    prices: Prices,
    data: MarketData,
    targets: Mapping[str, Mapping[int, float]],
    weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(target, force)`` over the full ``prices`` span, as ``SignalStrategy`` decides them."""
    symbols = list(data.schedules)
    w = dict(weights) if weights is not None else dict.fromkeys(symbols, 1.0 / len(symbols))
    t_n = prices.ts.size
    target = np.full((t_n, len(prices.cols)), np.nan)
    force = np.zeros((t_n, len(prices.cols)), dtype=bool)
    has_bar = np.isfinite(prices.close)
    for s in symbols:
        sched = data.schedules[s]
        tmap = targets[s]
        if len(sched.instrument_ids) == 1:  # no rolls: vectorised lookup
            j = prices.col[sched.instrument_ids[0]]
            keys = np.fromiter(tmap.keys(), np.int64, len(tmap))
            vals = np.fromiter(tmap.values(), np.float64, len(tmap))
            pos = np.searchsorted(keys, prices.ts)
            pos = np.minimum(pos, keys.size - 1)
            hit = (keys[pos] == prices.ts) & has_bar[:, j]
            target[hit, j] = vals[pos[hit]] * w[s]
            continue
        prev: str | None = None
        for i, ts in enumerate(prices.ts.tolist()):
            iid = sched.active(ts)
            j = prices.col[iid]
            if not has_bar[i, j]:
                continue
            rolled = prev is not None and prev != iid
            if rolled:
                k = prices.col[prev]  # type: ignore[index]
                target[i, k] = 0.0
                force[i, k] = True
            prev = iid
            tgt = tmap.get(ts)
            if tgt is not None:
                target[i, j] = tgt * w[s]
                force[i, j] = rolled
    return target, force


def panel(
    research: Any,  # noqa: ANN401
    data: MarketData,
    prices: Prices,
    *,
    start: int | None = None,
    end: int | None = None,
    weights: Mapping[str, float] | None = None,
    targets: Mapping[str, Mapping[int, float]] | None = None,
) -> Panel:
    """The panel for ``run_backtest(research, data, start=start, end=end)``.

    *targets* may be passed in when already computed on the full history (they
    equal the windowed ones because every strategy is causal).
    """
    if targets is None:
        targets = compute_targets(research, data.window(None, end).frames)
    target, force = target_matrix(prices, data, targets, weights)
    lo = 0 if start is None else int(np.searchsorted(prices.ts, start, "left"))
    hi = prices.ts.size if end is None else int(np.searchsorted(prices.ts, end, "right"))
    sl = slice(lo, hi)
    return Panel(
        prices.ts[sl],
        prices.cols,
        prices.close[sl],
        prices.mult,
        prices.incr,
        target[sl],
        force[sl],
    )


# ── Results and comparison ────────────────────────────────────────────────────


@dataclass(frozen=True)
class EngineResult:
    """Equity at every bar close before that close's trades, plus the order count."""

    ts: np.ndarray
    equity: np.ndarray
    n_orders: int

    def returns(self) -> pl.DataFrame:
        r = self.equity[1:] / self.equity[:-1] - 1.0
        return pl.DataFrame({"ts": self.ts[1:], "return": r}).with_columns(
            pl.col("ts").cast(pl.Datetime("ns", "UTC"))
        )


def from_nautilus(run: BacktestRun) -> EngineResult:
    eq = run.equity
    return EngineResult(
        eq["ts"].dt.epoch("ns").to_numpy(), eq["equity"].to_numpy(), run.total_orders
    )


def nautilus(
    research: Any,  # noqa: ANN401
    data: MarketData,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[EngineResult, BacktestRun]:
    run = run_backtest(research, data, **kwargs)
    return from_nautilus(run), run


def sharpe(result: EngineResult) -> float:
    """Annualised Sharpe of the engine's mark-to-market returns, by Nautilus's analyzer."""
    value = returns_statistics(result.returns()).get(SHARPE, math.nan)
    return float(value)


def compare(ref: EngineResult, other: EngineResult) -> dict[str, float]:
    """Daily-return differences, final equity ratio and Sharpe of *other* vs *ref*."""
    common, ia, ib = np.intersect1d(ref.ts, other.ts, return_indices=True)
    if common.size != ref.ts.size:
        raise ValueError(f"timelines differ: {ref.ts.size} ref closes, {common.size} shared")
    ra = ref.equity[ia]
    rb = other.equity[ib]
    da = ra[1:] / ra[:-1] - 1.0
    db = rb[1:] / rb[:-1] - 1.0
    diff = db - da
    return {
        "max_abs_ret_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms_ret_diff": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "final_equity_ratio": float(rb[-1] / ra[-1]),
        "sharpe_ref": sharpe(ref),
        "sharpe": sharpe(other),
        "orders_ref": ref.n_orders,
        "orders": other.n_orders,
    }


def timeit(fn: Callable[[], Any], n: int = 5) -> tuple[float, Any]:
    """Median wall time of *n* warm runs (after one untimed call), and the last result."""
    out = fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


# ── Fixtures ──────────────────────────────────────────────────────────────────


def future(
    symbol: str, venue: str, multiplier: float, price_precision: int, tick: str
) -> FuturesContract:
    """A perpetual stand-in ``FuturesContract`` (no expiry), lot size 1."""
    return FuturesContract(
        InstrumentId.from_str(f"{symbol}.{venue}"),
        Symbol(symbol),
        AssetClass.INDEX,
        symbol,
        0,
        2**63 - 1,
        Currency.from_str("USD"),
        price_precision,
        Price.from_str(tick),
        Quantity.from_int(int(multiplier))
        if multiplier >= 1
        else Quantity.from_str(str(multiplier)),
        Quantity.from_int(1),
        0,
        0,
    )
