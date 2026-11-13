# ruff: noqa: ANN001, ANN202 — numba callbacks are untyped by design
"""Engine adapters: each simulates a :class:`~common.Panel` and returns an EngineResult.

* ``kernel``          — in-house numba loop (the "neither library" fallback).
* ``vbt_orders``      — vectorbt ``Portfolio.from_orders`` with ``targetpercent``.
* ``vbt_func``        — vectorbt ``Portfolio.from_order_func`` with a numba callback
                        that sizes exactly like ``SignalStrategy`` (band, rounding).
* ``rbt_session``     — raptorbt ``PortfolioSession`` driven bar by bar from Python.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import raptorbt as rb
import vectorbt as vbt
from common import EngineResult, Panel
from numba import njit, prange
from vectorbt.portfolio import nb as vnb
from vectorbt.portfolio.enums import Direction, SizeType

if TYPE_CHECKING:
    from snippy_scales.execution import Costs


# ── In-house numba kernel ─────────────────────────────────────────────────────


@njit(cache=True)
def _simulate(close, mult, incr, target, force, capital, fee_rate, fee_pc, band, eq_out):
    t_n, n = close.shape
    pos = np.zeros(n)
    last = np.full(n, np.nan)
    equity = capital
    orders = 0
    for i in range(t_n):
        for j in range(n):
            c = close[i, j]
            if np.isfinite(c):
                if pos[j] != 0.0 and np.isfinite(last[j]):
                    equity += pos[j] * mult[j] * (c - last[j])
                last[j] = c
        eq_out[i] = equity
        e = equity
        if e <= 0.0:
            continue
        for j in range(n):
            t = target[i, j]
            c = close[i, j]
            if np.isnan(t) or not np.isfinite(c) or c <= 0.0:
                continue
            lots = np.round(t * e / (c * mult[j]) / incr[j]) * incr[j]
            delta = lots - pos[j]
            if delta == 0.0:
                continue
            notional = abs(delta) * c * mult[j]
            if not force[i, j] and notional < band * e:
                continue
            equity -= fee_rate * notional + fee_pc * abs(delta)
            pos[j] = lots
            orders += 1
    return orders


@njit(parallel=True, cache=True)
def _simulate_batch(
    close, mult, incr, targets, force, capital, fee_rate, fee_pc, band, eq_out, n_out
):
    for k in prange(targets.shape[0]):
        n_out[k] = _simulate(
            close, mult, incr, targets[k], force, capital, fee_rate, fee_pc, band, eq_out[k]
        )


def _fees(costs: Costs | None) -> tuple[float, float]:
    if costs is None:
        return 0.0, 0.0
    return costs.commission_rate, costs.commission_per_contract


def kernel(p: Panel, capital: float, costs: Costs | None = None, band: float = 0.0) -> EngineResult:
    rate, pc = _fees(costs)
    eq = np.empty(p.ts.size)
    n = _simulate(p.close, p.mult, p.incr, p.target, p.force, capital, rate, pc, band, eq)
    return EngineResult(p.ts, eq, int(n))


def kernel_batch(
    p: Panel, targets: np.ndarray, capital: float, costs: Costs | None = None, band: float = 0.0
) -> np.ndarray:
    """Equity ``[K, T]`` for ``K`` target matrices on one panel (parameter sweep)."""
    rate, pc = _fees(costs)
    eq = np.empty((targets.shape[0], p.ts.size))
    n = np.empty(targets.shape[0], np.int64)
    _simulate_batch(p.close, p.mult, p.incr, targets, p.force, capital, rate, pc, band, eq, n)
    return eq


# ── vectorbt: from_orders ─────────────────────────────────────────────────────


def vbt_orders(
    p: Panel,
    capital: float,
    costs: Costs | None = None,
    *,
    granular: bool = True,
    engine: str | None = None,
) -> EngineResult:
    """``targetpercent`` orders at the close; one cash pool; sells before buys.

    Contract multipliers are folded into the price, so one vectorbt unit is one
    contract. A per-contract fee becomes a rate of ``fee / (close * multiplier)``.
    """
    idx = pd.RangeIndex(p.ts.size)
    px = pd.DataFrame(p.ffill_close * p.mult, index=idx, columns=p.cols)
    rate, pc = _fees(costs)
    fees: float | pd.DataFrame = rate if not pc else pc / px
    pf = vbt.Portfolio.from_orders(
        px,
        size=pd.DataFrame(p.target, index=idx, columns=p.cols),
        size_type="targetpercent",
        direction="both",
        price=px,
        fees=fees,
        size_granularity=p.incr if granular else np.nan,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        init_cash=capital,
        freq="1D",
        engine=engine,
    )
    value = pf.value().to_numpy()
    rec = pf.orders.values
    paid = np.bincount(rec["idx"], weights=rec["fees"], minlength=p.ts.size)
    return EngineResult(p.ts, value + paid, len(rec))


# ── vectorbt: from_order_func (exact sizing, band, leverage) ──────────────────

#: Phantom cash so vectorbt's cash check never binds; subtracted from value.
PHANTOM_MULT = 1000.0


@njit
def _pre_segment(c, px, incr, target, force, band, phantom, eq_out, cols, deltas, count):
    i = c.i
    value = c.last_cash[c.group]
    for col in range(c.from_col, c.to_col):
        if c.last_position[col] != 0.0:
            value += c.last_position[col] * px[i, col]
    e = value - phantom
    eq_out[i, c.group] = e
    count[0] = 0
    if e <= 0.0:
        return (cols, deltas, count)
    # sells first, then buys (cash check order, as call_seq="auto")
    for side in range(2):
        for col in range(c.from_col, c.to_col):
            t = target[i, col]
            price = px[i, col]
            if np.isnan(t) or not np.isfinite(price) or price <= 0.0:
                continue
            lots = np.round(t * e / price / incr[col]) * incr[col]
            d = lots - c.last_position[col]
            if d == 0.0 or (side == 0) != (d < 0.0):
                continue
            if not force[i, col] and abs(d) * price < band * e:
                continue
            cols[count[0]] = col
            deltas[col] = d
            count[0] += 1
    return (cols, deltas, count)


@njit
def _flex_order(c, cols, deltas, count, px, fee_rate, fee_pc):
    if c.call_idx >= count[0]:
        return -1, vnb.order_nothing_nb()
    col = cols[c.call_idx]
    d = deltas[col]
    return col, vnb.order_nb(
        size=d,
        price=px[c.i, col],
        size_type=SizeType.Amount,
        direction=Direction.Both,
        fees=fee_rate,
        fixed_fees=fee_pc * abs(d),
    )


def _vbt_func(
    px: np.ndarray,
    incr: np.ndarray,
    target: np.ndarray,
    force: np.ndarray,
    group_by: np.ndarray | bool,
    n_groups: int,
    capital: float,
    costs: Costs | None,
    band: float,
) -> tuple[np.ndarray, int]:
    rate, pc = _fees(costs)
    t_n, n = px.shape
    phantom = capital * PHANTOM_MULT
    eq = np.full((t_n, n_groups), np.nan)
    pf = vbt.Portfolio.from_order_func(
        pd.DataFrame(px),
        _flex_order,
        px,
        rate,
        pc,
        flexible=True,
        init_cash=capital + phantom,
        cash_sharing=True,
        group_by=group_by,
        pre_segment_func_nb=_pre_segment,
        pre_segment_args=(
            px,
            incr,
            target,
            force,
            band,
            phantom,
            eq,
            np.zeros(n, np.int64),
            np.zeros(n),
            np.zeros(1, np.int64),
        ),
        freq="1D",
    )
    return eq, len(pf.orders.values)


def vbt_func(
    p: Panel, capital: float, costs: Costs | None = None, band: float = 0.0
) -> EngineResult:
    px = p.ffill_close * p.mult
    eq, n = _vbt_func(px, p.incr, p.target, p.force, True, 1, capital, costs, band)
    return EngineResult(p.ts, eq[:, 0], n)


def vbt_func_batch(
    p: Panel, targets: np.ndarray, capital: float, costs: Costs | None = None, band: float = 0.0
) -> np.ndarray:
    """Equity ``[K, T]`` for ``K`` single-instrument target series, one group each."""
    k = targets.shape[0]
    px = np.repeat(p.ffill_close * p.mult, k, axis=1)
    tgt = np.ascontiguousarray(targets[:, :, 0].T)
    force = np.repeat(p.force, k, axis=1)
    incr = np.repeat(p.incr, k)
    eq, _ = _vbt_func(px, incr, tgt, force, np.arange(k), k, capital, costs, band)
    return eq.T


def vbt_orders_batch(
    p: Panel,
    targets: np.ndarray,
    capital: float,
    costs: Costs | None = None,
    engine: str | None = None,
) -> np.ndarray:
    """``from_orders`` over ``K`` single-instrument target series in one call (no cash sharing)."""
    k = targets.shape[0]
    px = np.repeat(p.ffill_close * p.mult, k, axis=1)
    rate, _ = _fees(costs)
    pf = vbt.Portfolio.from_orders(
        pd.DataFrame(px),
        size=pd.DataFrame(targets[:, :, 0].T),
        size_type="targetpercent",
        direction="both",
        price=pd.DataFrame(px),
        fees=rate,
        size_granularity=float(p.incr[0]),
        init_cash=capital,
        freq="1D",
        engine=engine,
    )
    return pf.value().to_numpy().T


# ── raptorbt: PortfolioSession ────────────────────────────────────────────────


def rbt_session(
    p: Panel,
    capital: float,
    costs: Costs | None = None,
    band: float = 0.0,
    leverage: float = 1000.0,
) -> EngineResult:
    """Drive raptorbt's shared-account session with our own sizing.

    raptorbt cannot resize a position: netting rejects an add (``position_open``)
    and a sell larger than the long only closes it; ``reduce_only`` is always
    rejected under hedging. The one mapping with exact P&L is a hedging account
    where every delta opens a new leg (net position = sum of legs). Legs never
    close, so locked margin grows with turnover: *leverage* is set high enough
    that margin never binds. A per-contract fee has no raptorbt equivalent and
    is approximated as a rate at the mean contract notional.
    """
    rate, pc = _fees(costs)
    if pc:
        notional = np.nanmean(p.close * p.mult)
        rate = pc / notional
    cfg = rb.BacktestConfig(
        initial_capital=capital, fees=rate, slippage=0.0, fill_timing="same_bar_close"
    )
    session = rb.PortfolioSession(config=cfg, account_type="margin", leverage=leverage)
    for j, iid in enumerate(p.cols):
        spec = rb.InstrumentSpec.futures_contract(
            iid,
            expiration_ns=2**62,
            lot_size=float(p.incr[j]),
            multiplier=float(p.mult[j]),
            price_increment=1e-9,
        )
        session.add_instrument(iid, direction=1, instrument=spec, oms_type="hedging")
    for j in range(len(p.cols)):
        ok = np.isfinite(p.close[:, j])
        c = np.ascontiguousarray(p.close[ok, j])
        session.set_bars(j, np.ascontiguousarray(p.ts[ok]), c, c, c, c, np.full(c.size, 1e15))
    session.seal()

    row = {int(t): i for i, t in enumerate(p.ts.tolist())}
    pos = np.zeros(len(p.cols))
    last = np.full(len(p.cols), np.nan)
    eq = np.full(p.ts.size, np.nan)
    cur_ts = -1
    e = capital
    orders = 0
    while (cur := session.current()) is not None:
        j, _local, ts, _o, _h, _l, close, _v = cur
        i = row[ts]
        if ts != cur_ts:
            cur_ts = ts
            # session.equity() marks each instrument at its last applied close; bring
            # every instrument with a bar at this close up to date first, so all
            # symbols are sized from one equity figure (as SignalStrategy does).
            e = session.equity()
            closes = p.close[i]
            live = np.isfinite(closes) & np.isfinite(last)
            e += float(np.sum(pos[live] * p.mult[live] * (closes[live] - last[live])))
            eq[i] = e
        t = p.target[i, j]
        if not math.isnan(t) and e > 0.0:
            lots = round(t * e / (close * p.mult[j]) / p.incr[j]) * p.incr[j]
            d = lots - pos[j]
            if d != 0.0 and (p.force[i, j] or abs(d) * close * p.mult[j] >= band * e):
                orders += 1
                session.submit_order(
                    j,
                    side="buy" if d > 0 else "sell",
                    kind="market",
                    submitted_idx=_local,
                    submitted_ts=ts,
                    client_id=f"o{orders}",
                    units=float(abs(d)),
                )
                pos[j] = lots
        last[j] = close
        session.apply_current()
    res = session.finish()
    if res.halted:
        raise RuntimeError(f"raptorbt halted at {res.halted_at}")
    return EngineResult(p.ts, eq, orders)
