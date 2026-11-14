"""Vectorised screening engine: ``run_backtest`` on a numba kernel instead of Nautilus.

Same signature and the same :class:`~snippy_scales.execution.BacktestRun` as
:func:`snippy_scales.execution.run_backtest`, thousands of times faster, for
screening Optuna trials and multi-asset sweeps. Validate the trials you keep on
Nautilus (see ``docs/research/vectorised_engines_spike.md``).

The kernel replays :class:`~snippy_scales.execution.SignalStrategy`'s decisions:

* at each bar close every instrument with a bar is marked, then equity is read
  once and every live symbol is sized from it:
  ``contracts = round_half_even(target * weight * equity / (close * multiplier))``
  in size increments, filled at that close;
* a symbol whose active contract changed since its last order closes the old
  contract and opens the new one at that close, ignoring the rebalance band;
* trades below ``rebalance_band * equity`` are skipped;
* a non-positive close is neither marked nor traded (as Nautilus).

``returns`` and ``stats_returns`` are built exactly as Nautilus builds them
(daily post-fee equity over calendar days, the same 11 return statistics), so
fold Sharpes are comparable between engines. Parity with Nautilus is tested in
``tests/test_vector_parity.py``.

Not modelled — differences to expect against Nautilus:

* **Liquidity.** Nautilus treats a bar's volume as liquidity, so an order larger
  than a thin bar walks to worse prices. Here everything fills at the close.
* **Slippage.** ``slippage_prob`` is charged as its expected cost
  (``prob * tick * contracts * multiplier`` per fill), not drawn at random.
* **Margin.** There are no margin checks, so ``leverage`` is ignored and a run
  Nautilus would deny orders in reports no rejections.
* **Position statistics.** Nautilus computes ``stats_pnls`` (realized PnL, win
  rate, ...) and ``stats_general`` from closed positions, which the kernel does
  not track: both are empty, as is ``positions``. ``total_return`` (from the
  equity curve) is unaffected.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import nautilus_trader.analysis as analysis
import numpy as np
import pandas as pd
import polars as pl
from numba import njit

from snippy_scales.data import catalog as cat
from snippy_scales.execution.backtest import BacktestRun, Costs
from snippy_scales.execution.strategy import (
    Panel,
    Targets,
    panel_targets,
    stack_frames,
)
from snippy_scales.strategies.momentum_cs import MultiAssetStrategy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from snippy_scales.execution.backtest import MarketData, TimeBound
    from snippy_scales.strategies.trend import Strategy

_DAY = 86_400_000_000_000

#: The return statistics Nautilus's ``BacktestEngine`` registers (same keys and values).
RETURN_STATISTICS = (
    "ReturnsAverage",
    "ReturnsAverageLoss",
    "ReturnsAverageWin",
    "ReturnsKurtosis",
    "ReturnsSkewness",
    "ReturnsVolatility",
    "ProfitFactor",
    "RiskReturnRatio",
    "SharpeRatio",
    "SortinoRatio",
    "TailRatio",
)


# ── Dense price matrix, built once per MarketData ─────────────────────────────


@dataclass(frozen=True)
class _Prices:
    """Bar closes of every instrument, one row per bar close (union over instruments).

    A contract's bars outside its roll-schedule span are dropped, as
    :meth:`MarketData.window` drops them from replay.
    """

    ts: np.ndarray  # int64 [T]
    instruments: list[str]  # [N]
    close: np.ndarray  # float64 [T, N]; NaN = no bar
    multiplier: np.ndarray  # [N]
    increment: np.ndarray  # [N] size increment
    tick: np.ndarray  # [N] price increment
    column: dict[str, int]


_PRICES: dict[int, tuple[weakref.ref[MarketData], _Prices]] = {}


def _prices(data: MarketData) -> _Prices:
    """The price matrix of *data*, cached for as long as *data* is alive."""
    key = id(data)
    hit = _PRICES.get(key)
    if hit is not None and hit[0]() is data:
        return hit[1]
    spans = {
        iid: schedule.span(iid)
        for schedule in data.schedules.values()
        for iid in schedule.instrument_ids
    }
    instruments = list(spans)
    per: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for iid in instruments:
        bars = data.bars[iid]
        ts = np.fromiter((b.ts_init for b in bars), np.int64, len(bars))
        close = np.fromiter((float(b.close) for b in bars), np.float64, len(bars))
        first, last = spans[iid]
        keep = (ts >= first) & (ts <= last)
        per[iid] = (ts[keep], close[keep])
    all_ts = [t for t, _ in per.values()]
    ts = np.unique(np.concatenate(all_ts)) if all_ts else np.empty(0, np.int64)
    close = np.full((ts.size, len(instruments)), np.nan)
    for j, iid in enumerate(instruments):
        t, c = per[iid]
        close[np.searchsorted(ts, t), j] = c
    inst = [data.instruments[i] for i in instruments]
    prices = _Prices(
        ts=ts,
        instruments=instruments,
        close=close,
        multiplier=np.array([float(i.multiplier) for i in inst]),
        increment=np.array([float(i.size_increment) for i in inst]),
        tick=np.array([float(i.price_increment) for i in inst]),
        column={iid: j for j, iid in enumerate(instruments)},
    )
    _PRICES[key] = (weakref.ref(data, lambda _: _PRICES.pop(key, None)), prices)
    return prices


_PANELS: dict[int, tuple[weakref.ref[MarketData], Panel]] = {}


def _panel(data: MarketData) -> Panel:
    """*data*'s frames stacked into one :class:`Panel`, cached for as long as *data* is alive."""
    key = id(data)
    hit = _PANELS.get(key)
    if hit is not None and hit[0]() is data:
        return hit[1]
    panel = stack_frames(data.frames)
    _PANELS[key] = (weakref.ref(data, lambda _: _PANELS.pop(key, None)), panel)
    return panel


def _active_columns(prices: _Prices, data: MarketData) -> np.ndarray:
    """``[T, S]``: column of each symbol's active contract at every price row."""
    symbols = list(data.schedules)
    cols = np.empty((prices.ts.size, len(symbols)), np.int64)
    for s, symbol in enumerate(symbols):
        schedule = data.schedules[symbol]
        if len(schedule.instrument_ids) == 1:
            cols[:, s] = prices.column[schedule.instrument_ids[0]]
        else:
            cols[:, s] = [prices.column[schedule.active(int(t))] for t in prices.ts]
    return cols


@dataclass
class _ActiveCache:
    columns: np.ndarray  # [T, S] active contract column at every price row
    windows: dict[tuple[int, int], np.ndarray]  # (lo, hi) -> active, -1 where no bar


#: The active-contract panel of each window depends on the data alone, so every
#: trial of a study reuses it.
_ACTIVE: dict[int, tuple[weakref.ref[MarketData], _ActiveCache]] = {}


def _active(prices: _Prices, data: MarketData, lo: int, hi: int) -> np.ndarray:
    """``[T, S]`` over rows ``lo:hi``: symbol *s*'s active column where it has a bar, else -1."""
    key = id(data)
    hit = _ACTIVE.get(key)
    if hit is None or hit[0]() is not data:
        cache = _ActiveCache(_active_columns(prices, data), {})
        _ACTIVE[key] = (weakref.ref(data, lambda _: _ACTIVE.pop(key, None)), cache)
    else:
        cache = hit[1]
    if (lo, hi) not in cache.windows:
        cols = cache.columns[lo:hi]
        rows = np.arange(hi - lo)[:, None]
        live = np.isfinite(prices.close[lo:hi][rows, cols])
        cache.windows[(lo, hi)] = np.ascontiguousarray(np.where(live, cols, -1))
    return cache.windows[(lo, hi)]


@njit(cache=True)
def _scatter_targets(  # noqa: PLR0913
    ts: np.ndarray,
    active: np.ndarray,
    t_ts: np.ndarray,
    t_starts: np.ndarray,
    t_value: np.ndarray,
    column: np.ndarray,
    weight: np.ndarray,
    out: np.ndarray,
) -> None:
    """``out[i, column[b]] = value * weight[b]`` for every target row of block *b* whose
    bar close is ``ts[i]`` and whose symbol is live there (one merge pass per block)."""
    for b in range(t_starts.size - 1):
        s = column[b]
        j = 0
        for k in range(t_starts[b], t_starts[b + 1]):
            t = t_ts[k]
            while j < ts.size and ts[j] < t:
                j += 1
            if j == ts.size:
                break
            if ts[j] == t and active[j, s] >= 0:
                out[j, s] = t_value[k] * weight[b]


def _symbol_panel(
    prices: _Prices,
    data: MarketData,
    targets: Targets,
    weights: Mapping[str, float],
    lo: int,
    hi: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(active, target)``, both ``[T, S]`` over rows ``lo:hi``.

    ``active[i, s]`` is the column of symbol *s*'s active contract when it has a
    bar at row *i* (else -1); ``target[i, s]`` is ``target * weight`` (NaN = none).
    """
    symbols = list(data.schedules)
    active = _active(prices, data, lo, hi)
    target = np.full(active.shape, np.nan)
    index = {s: k for k, s in enumerate(symbols)}
    column = np.array([index[s] for s in targets.symbols], np.int64)
    weight = np.array([weights[s] for s in targets.symbols], np.float64)
    _scatter_targets(
        prices.ts[lo:hi], active, targets.ts, targets.starts, targets.target, column, weight, target
    )
    return active, target


# ── Kernel ────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _simulate(  # noqa: PLR0913, PLR0912, C901
    close: np.ndarray,
    multiplier: np.ndarray,
    increment: np.ndarray,
    tick: np.ndarray,
    active: np.ndarray,
    target: np.ndarray,
    capital: float,
    fee_rate: float,
    fee_per_contract: float,
    slippage_prob: float,
    band: float,
    eq_pre: np.ndarray,
    eq_post: np.ndarray,
    o_row: np.ndarray,
    o_col: np.ndarray,
    o_qty: np.ndarray,
    o_px: np.ndarray,
    o_fee: np.ndarray,
    r_row: np.ndarray,
    r_old: np.ndarray,
    r_new: np.ndarray,
) -> tuple[int, int]:
    t_n, n = close.shape
    s_n = active.shape[1]
    pos = np.zeros(n)
    last = np.full(n, np.nan)
    held = np.full(s_n, -1, np.int64)  # column last ordered per symbol, as SignalStrategy
    equity = capital
    n_orders = 0
    n_rolls = 0
    for i in range(t_n):
        for j in range(n):
            c = close[i, j]
            if np.isfinite(c) and c > 0.0:
                if pos[j] != 0.0 and np.isfinite(last[j]):
                    equity += pos[j] * multiplier[j] * (c - last[j])
                last[j] = c
        eq_pre[i] = equity
        e = equity
        for s in range(s_n):
            j = active[i, s]
            if j < 0:
                continue
            rolled = False
            h = held[s]
            if h >= 0 and h != j:
                rolled = True
                r_row[n_rolls] = i
                r_old[n_rolls] = h
                r_new[n_rolls] = j
                n_rolls += 1
                held[s] = -1
                if pos[h] != 0.0:
                    px = last[h]
                    qty = -pos[h]
                    fee = (
                        fee_rate * abs(qty) * px * multiplier[h]
                        + fee_per_contract * abs(qty)
                        + slippage_prob * tick[h] * abs(qty) * multiplier[h]
                    )
                    equity -= fee
                    pos[h] = 0.0
                    o_row[n_orders] = i
                    o_col[n_orders] = h
                    o_qty[n_orders] = qty
                    o_px[n_orders] = px
                    o_fee[n_orders] = fee
                    n_orders += 1
            t = target[i, s]
            c = close[i, j]
            if np.isnan(t) or not (c > 0.0) or e <= 0.0:
                continue
            lots = np.round(t * e / (c * multiplier[j]) / increment[j]) * increment[j]
            qty = lots - pos[j]
            if qty == 0.0:
                continue
            notional = abs(qty) * c * multiplier[j]
            if not rolled and notional < band * e:
                continue
            fee = (
                fee_rate * notional
                + fee_per_contract * abs(qty)
                + slippage_prob * tick[j] * abs(qty) * multiplier[j]
            )
            equity -= fee
            pos[j] = lots
            held[s] = j
            o_row[n_orders] = i
            o_col[n_orders] = j
            o_qty[n_orders] = qty
            o_px[n_orders] = c
            o_fee[n_orders] = fee
            n_orders += 1
        eq_post[i] = equity
    return n_orders, n_rolls


# ── Results ───────────────────────────────────────────────────────────────────


def nautilus_returns(ts: np.ndarray, post: np.ndarray, capital: float) -> pl.DataFrame:
    """Daily returns exactly as Nautilus's ``BacktestResult.returns_series`` builds them.

    Post-fee equity is sampled on every calendar day from the first bar close's
    day to the last one's, forward-filled over days without a bar (so weekends
    are zero returns); the first day's base is *capital*. A return is labelled
    with the day it starts on, so there is one fewer return than days. Verified
    against Nautilus on daily bars (``tests/test_vector_parity.py``).
    """
    schema = {"ts": pl.Datetime("ns", "UTC"), "return": pl.Float64}
    if ts.size == 0:
        return pl.DataFrame(schema=schema)
    first, last = ts[0] // _DAY, ts[-1] // _DAY
    days = np.arange(first, last + 1, dtype=np.int64)
    idx = np.searchsorted(ts // _DAY, days, "right") - 1
    values = post[idx].copy()
    values[0] = capital
    returns = values[1:] / values[:-1] - 1.0
    return pl.DataFrame({"ts": days[:-1] * _DAY, "return": returns}).with_columns(
        pl.col("ts").cast(pl.Datetime("ns", "UTC"))
    )


def return_statistics(returns: pl.DataFrame) -> dict[str, float]:
    """Nautilus's return statistics (as ``BacktestResult.stats_returns``) for *returns*."""
    analyzer = analysis.PortfolioAnalyzer()
    for name in RETURN_STATISTICS:
        analyzer.register_statistic(getattr(analysis, name)())
    for t, r in zip(
        returns["ts"].dt.epoch("ns").to_list(), returns["return"].to_list(), strict=True
    ):
        analyzer.add_return(int(t), float(r))
    return {k: float(v) for k, v in analyzer.get_performance_stats_returns().items()}


def _fills(prices: _Prices, ts: np.ndarray, rec: tuple[np.ndarray, ...], n: int) -> pd.DataFrame:
    row, col, qty, px, fee = (a[:n] for a in rec)
    return pd.DataFrame(
        {
            "instrument_id": np.array(prices.instruments, dtype=object)[col],
            "order_side": np.where(qty > 0, "BUY", "SELL"),
            "last_qty": np.abs(qty),
            "last_px": px,
            "commission": fee,
            "ts_event": pd.to_datetime(ts[row], utc=True),
        }
    )


def precompute_targets(research: Strategy | MultiAssetStrategy, data: MarketData) -> Targets | None:
    """*research*'s targets on all of *data*, to pass as ``run_backtest(targets=...)``.

    A single-asset strategy's target on a bar depends only on that symbol's
    bars up to it (``tests/test_strategy_causality.py``), so targets computed
    once on the whole history equal those computed on any window's history,
    and walk-forward folds can share them. A cross-sectional strategy's
    targets near a window's end depend on which symbols have bars there, so
    it returns ``None`` (recompute per window).
    """
    if isinstance(research, MultiAssetStrategy):
        return None
    return panel_targets(research, _panel(data), exits=False)


def run_backtest(
    research: Strategy | MultiAssetStrategy,
    data: MarketData,
    *,
    start: TimeBound = None,
    end: TimeBound = None,
    initial_capital: float = 1_000_000.0,
    costs: Costs | None = None,
    rebalance_band: float = 0.0,
    weights: Mapping[str, float] | None = None,
    leverage: float = 10.0,  # noqa: ARG001 — accepted for signature parity; margin not modelled
    targets: Targets | None = None,
) -> BacktestRun:
    """Backtest *research* on *data* with the vectorised kernel.

    Arguments are those of :func:`snippy_scales.execution.run_backtest`;
    *leverage* is accepted but not modelled (no margin checks).

    Args:
        targets: *research*'s targets on all of *data*, from
            :func:`precompute_targets`, to reuse across windows. Only valid for a
            single-asset strategy, which is causal per symbol.
    """
    costs = costs if costs is not None else Costs()
    if rebalance_band < 0.0:
        raise ValueError(f"rebalance_band must be non-negative, got {rebalance_band}")
    symbols = list(data.schedules)
    weights = dict(weights) if weights is not None else dict.fromkeys(symbols, 1.0 / len(symbols))
    prices = _prices(data)
    lo_ns, hi_ns = cat.to_ns(start), cat.to_ns(end)
    lo = 0 if lo_ns is None else int(np.searchsorted(prices.ts, lo_ns, "left"))
    hi = prices.ts.size if hi_ns is None else int(np.searchsorted(prices.ts, hi_ns, "right"))
    # Signals see every bar up to *end* (as MarketData.window).
    if targets is None:
        window = panel_targets(research, _panel(data).until(hi_ns))
    else:
        if isinstance(research, MultiAssetStrategy):
            raise ValueError("precomputed targets are only valid for single-asset strategies")
        window = targets.until(hi_ns).exit_delisted()
    active, target = _symbol_panel(prices, data, window, weights, lo, hi)

    ts = prices.ts[lo:hi]
    close = np.ascontiguousarray(prices.close[lo:hi])
    t_n = close.shape[0]
    cap = t_n * active.shape[1] * 2 + 1  # at most a roll close and a trade per symbol per row
    eq_pre, eq_post = np.empty(t_n), np.empty(t_n)
    rec = (
        np.empty(cap, np.int64),
        np.empty(cap, np.int64),
        np.empty(cap),
        np.empty(cap),
        np.empty(cap),
    )
    rolls = (np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap, np.int64))
    n_orders, n_rolls = _simulate(
        close,
        prices.multiplier,
        prices.increment,
        prices.tick,
        active,
        target,
        float(initial_capital),
        float(costs.commission_rate),
        float(costs.commission_per_contract),
        float(costs.slippage_prob),
        float(rebalance_band),
        eq_pre,
        eq_post,
        *rec,
        *rolls,
    )
    returns = nautilus_returns(ts, eq_post, initial_capital)
    return BacktestRun(
        stats_returns=return_statistics(returns),
        stats_pnls={},
        stats_general={},
        returns=returns,
        equity=pl.DataFrame({"ts": ts, "equity": eq_pre}).with_columns(
            pl.col("ts").cast(pl.Datetime("ns", "UTC"))
        ),
        fills=_fills(prices, ts, rec, n_orders),
        positions=pd.DataFrame(),
        rejections=(),
        total_orders=int(n_orders),
        rolls=tuple(
            (int(ts[r]), prices.instruments[o], prices.instruments[w])
            for r, o, w in zip(*(a[:n_rolls] for a in rolls), strict=True)
        ),
    )
