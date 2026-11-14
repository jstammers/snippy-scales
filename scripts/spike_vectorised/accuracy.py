"""Accuracy matrix: every engine vs ``execution.run_backtest`` on the same data.

Run with the spike venv and ``SNIPPY_DATA_ROOT`` set::

    SNIPPY_DATA_ROOT=/path/to/data python scripts/spike_vectorised/accuracy.py [out.json]
"""

from __future__ import annotations

import json
import sys
from typing import Any

import common as C
import engines as E
import fixtures as F
import numpy as np
import polars as pl

from snippy_scales.execution import Costs
from snippy_scales.strategies import (
    CrossSectionalMomentum,
    Strategy,
    TimeSeriesMomentum,
    TrendFollowing,
)


class Fixed(Strategy):
    def __init__(self, value: float) -> None:
        self.value = value

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        return pl.Series("position", np.full(bars.height, self.value))


def cases() -> list[dict[str, Any]]:
    raw_aapl, raw_eqs, raw_fut = F.aapl(), F.equities(), F.glbx_futures()
    aapl, eqs, fut = (F.ample_volume(d) for d in (raw_aapl, raw_eqs, raw_fut))
    es = F.rolled_es()
    fee = Costs(commission_rate=5e-4)
    tsmom = TimeSeriesMomentum(vol_target=0.25)
    return [
        {"name": "AAPL const 1.0, no costs", "data": aapl, "research": Fixed(1.0)},
        {"name": "AAPL TrendFollowing, no costs", "data": aapl, "research": TrendFollowing()},
        {"name": "AAPL TrendFollowing, 5 bp", "data": aapl, "research": TrendFollowing(),
         "costs": fee},
        {"name": "19 futures TSMOM, no costs", "data": fut, "research": tsmom, "capital": 1e8},
        {"name": "19 futures TSMOM, 5 bp", "data": fut, "research": tsmom, "capital": 1e8,
         "costs": fee},
        {"name": "8 equities CSMOM, no costs", "data": eqs, "research": CrossSectionalMomentum()},
        {"name": "8 equities CSMOM, 5 bp", "data": eqs, "research": CrossSectionalMomentum(),
         "costs": fee},
        {"name": "Rolled ES TrendFollowing, no costs", "data": es,
         "research": TrendFollowing(fast_period=10, slow_period=40)},
        {"name": "Rolled ES TrendFollowing, $2.25/contract", "data": es,
         "research": TrendFollowing(fast_period=10, slow_period=40),
         "costs": Costs(commission_per_contract=2.25)},
        {"name": "AAPL TrendFollowing, 5 bp, band 2%", "data": aapl,
         "research": TrendFollowing(), "costs": fee, "band": 0.02},
        {"name": "19 futures TSMOM, 5 bp, band 0.5%", "data": fut, "research": tsmom,
         "capital": 1e8, "costs": fee, "band": 0.005},
        # As stored: Nautilus fills against each real bar's volume.
        {"name": "AAPL TrendFollowing, 5 bp [real volume]", "data": raw_aapl,
         "research": TrendFollowing(), "costs": fee},
        {"name": "19 futures TSMOM, no costs [real volume]", "data": raw_fut, "research": tsmom,
         "capital": 1e8},
        {"name": "8 equities CSMOM, 5 bp [real volume]", "data": raw_eqs,
         "research": CrossSectionalMomentum(), "costs": fee},
    ]  # fmt: skip


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    data, research = case["data"], case["research"]
    capital = case.get("capital", 1e6)
    costs = case.get("costs")
    band = case.get("band", 0.0)
    ref, run = C.nautilus(research, data, initial_capital=capital, costs=costs, rebalance_band=band)
    prices = C.Prices.from_data(data)
    p = C.panel(research, data, prices)
    gross = np.nansum(np.abs(np.nan_to_num(p.target)), axis=1)
    shorts = bool((np.nan_to_num(p.target) < 0).any())
    out: dict[str, Any] = {
        "case": case["name"],
        "bars": int(p.ts.size),
        "cols": len(p.cols),
        "max_gross": float(gross.max()),
        "shorts": shorts,
        "rejections": len(run.rejections),
        "rolls": len(run.rolls),
        "nautilus_sharpe_own": run.sharpe,
        "engines": {},
    }
    adapters = {
        "kernel": lambda: E.kernel(p, capital, costs, band),
        "vbt_func": lambda: E.vbt_func(p, capital, costs, band),
        "rbt_session": lambda: E.rbt_session(p, capital, costs, band),
    }
    if band == 0.0:  # from_orders cannot express an equity-relative band
        adapters["vbt_orders"] = lambda: E.vbt_orders(p, capital, costs)
        adapters["vbt_orders_frac"] = lambda: E.vbt_orders(p, capital, costs, granular=False)
    for name, fn in adapters.items():
        try:
            out["engines"][name] = C.compare(ref, fn())
        except Exception as exc:  # noqa: BLE001 — record and carry on
            out["engines"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def main() -> None:
    results = []
    for case in cases():
        res = run_case(case)
        results.append(res)
        print(f"\n## {res['case']}  (bars={res['bars']}, cols={res['cols']}, "
              f"max gross={res['max_gross']:.2f}, shorts={res['shorts']}, "
              f"rejections={res['rejections']}, rolls={res['rolls']})")  # fmt: skip
        for name, m in res["engines"].items():
            if "error" in m:
                print(f"  {name:16s} ERROR {m['error']}")
                continue
            print(
                f"  {name:16s} maxΔr={m['max_abs_ret_diff']:.2e} rmsΔr={m['rms_ret_diff']:.2e} "
                f"eq={m['final_equity_ratio']:.6f} SR {m['sharpe_ref']:.4f}/{m['sharpe']:.4f} "
                f"orders {m['orders_ref']}/{m['orders']}"
            )
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
