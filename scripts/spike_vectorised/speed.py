"""Speed: warm medians per engine on three workloads, plus first-call (JIT) cost.

Run in a fresh process with an empty numba cache so first calls include compilation::

    NUMBA_CACHE_DIR=$(mktemp -d) SNIPPY_DATA_ROOT=/path/to/data \\
        python scripts/spike_vectorised/speed.py [out.json]
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import common as C
import engines as E
import fixtures as F
import numpy as np

from snippy_scales.evaluation import make_folds
from snippy_scales.execution import Costs, compute_targets, run_backtest
from snippy_scales.strategies import TimeSeriesMomentum, TrendFollowing

FEE = Costs(commission_rate=5e-4)
RESULTS: dict[str, Any] = {"first_call": {}, "single": {}, "grid": {}}


def first(name: str, fn: Any) -> None:  # noqa: ANN401
    t0 = time.perf_counter()
    fn()
    RESULTS["first_call"][name] = time.perf_counter() - t0


def single(label: str, data: Any, research: Any, capital: float) -> None:  # noqa: ANN401
    prices = C.Prices.from_data(data)
    p = C.panel(research, data, prices)
    sims = {
        "kernel": lambda: E.kernel(p, capital, FEE),
        "vbt_orders[numba]": lambda: E.vbt_orders(p, capital, FEE, engine="numba"),
        "vbt_orders[rust]": lambda: E.vbt_orders(p, capital, FEE, engine="rust"),
        "vbt_func": lambda: E.vbt_func(p, capital, FEE),
        "rbt_session": lambda: E.rbt_session(p, capital, FEE),
    }
    for name, fn in sims.items():
        if name not in RESULTS["first_call"]:
            first(name, fn)
    out: dict[str, float] = {}
    out["nautilus"], _ = C.timeit(
        lambda: run_backtest(research, data, initial_capital=capital, costs=FEE)
    )
    out["signals+panel"], _ = C.timeit(lambda: C.panel(research, data, prices))
    for name, fn in sims.items():
        out[name], _ = C.timeit(fn)
    RESULTS["single"][label] = out
    print(f"\n## {label}")
    for k, v in out.items():
        print(f"  {k:20s} {v * 1e3:10.2f} ms")


def grid() -> None:
    data = F.ample_volume(F.aapl())
    prices = C.Prices.from_data(data)
    folds = make_folds(data.closes(), n_splits=5)
    bounds = [
        (int(f.test_start.timestamp() * 1e9), int(f.test_end.timestamp() * 1e9)) for f in folds
    ]
    trials = [
        TrendFollowing(fast_period=f, slow_period=s)
        for f in range(5, 55, 5)
        for s in range(60, 260, 20)
    ]
    out: dict[str, float] = {}

    # Signals once per trial on the full history (strategies are causal).
    t0 = time.perf_counter()
    full = [C.target_matrix(prices, data, compute_targets(r, data.frames))[0] for r in trials]
    out["signals (100 trials, once each)"] = time.perf_counter() - t0
    stacked = np.stack(full)  # [K, T, 1]

    def fold_panel(k: int, lo: int, hi: int) -> C.Panel:
        a = int(np.searchsorted(prices.ts, lo))
        b = int(np.searchsorted(prices.ts, hi, "right"))
        force = np.zeros((b - a, 1), dtype=bool)
        return C.Panel(
            prices.ts[a:b], prices.cols, prices.close[a:b], prices.mult, prices.incr,
            stacked[k, a:b], force,
        )  # fmt: skip

    looped = {
        "kernel": lambda p: E.kernel(p, 1e6, FEE),
        "vbt_orders[numba]": lambda p: E.vbt_orders(p, 1e6, FEE, engine="numba"),
        "vbt_orders[rust]": lambda p: E.vbt_orders(p, 1e6, FEE, engine="rust"),
        "vbt_func": lambda p: E.vbt_func(p, 1e6, FEE),
        "rbt_session": lambda p: E.rbt_session(p, 1e6, FEE),
    }
    for name, sim in looped.items():
        t0 = time.perf_counter()
        for k in range(len(trials)):
            for lo, hi in bounds:
                sim(fold_panel(k, lo, hi))
        out[f"{name} (500 runs, looped)"] = time.perf_counter() - t0

    def batched(fn: Any) -> float:  # noqa: ANN401
        t0 = time.perf_counter()
        for lo, hi in bounds:
            p = fold_panel(0, lo, hi)
            a = int(np.searchsorted(prices.ts, lo))
            fn(p, stacked[:, a : a + p.ts.size])
        return time.perf_counter() - t0

    batch = {
        "kernel_batch": lambda p, t: E.kernel_batch(p, t, 1e6, FEE),
        "vbt_orders_batch[numba]": lambda p, t: E.vbt_orders_batch(p, t, 1e6, FEE, "numba"),
        "vbt_orders_batch[rust]": lambda p, t: E.vbt_orders_batch(p, t, 1e6, FEE, "rust"),
        "vbt_func_batch": lambda p, t: E.vbt_func_batch(p, t, 1e6, FEE),
    }
    for name, fn in batch.items():
        batched(fn)  # warm (compiles the batch signatures)
        out[f"{name} (5 calls x 100 cols)"] = float(np.median([batched(fn) for _ in range(5)]))

    t0 = time.perf_counter()
    for r in trials:
        for f in folds:
            run_backtest(r, data, start=f.test_start, end=f.test_end, costs=FEE)
    out["nautilus (500 runs, as WalkForward)"] = time.perf_counter() - t0

    RESULTS["grid"] = out
    print("\n## 100-trial x 5-fold grid, AAPL TrendFollowing")
    for k, v in out.items():
        print(f"  {k:45s} {v:10.3f} s")


def main() -> None:
    single("AAPL TrendFollowing (1 run)", F.ample_volume(F.aapl()), TrendFollowing(), 1e6)
    single(
        "19 futures TSMOM (1 run)",
        F.ample_volume(F.glbx_futures()),
        TimeSeriesMomentum(vol_target=0.25),
        1e8,
    )
    grid()
    print("\n## first call (includes JIT compile with an empty numba cache)")
    for k, v in RESULTS["first_call"].items():
        print(f"  {k:20s} {v:8.2f} s")
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(RESULTS, fh, indent=1)


if __name__ == "__main__":
    main()
