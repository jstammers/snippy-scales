"""Target generation vs simulation: where a screening run's time goes.

Splits one run into the strategy's ``generate_signals`` (polars), the rest of
``compute_targets`` (per-symbol ``{ts: target}`` dicts), the panel build and
the kernel. Same environment as ``speed.py``.
"""

from __future__ import annotations

import time
from typing import Any

import common as C
import engines as E
import fixtures as F
import numpy as np

from snippy_scales.data import catalog as cat
from snippy_scales.execution import compute_targets, load_market_data
from snippy_scales.strategies import (
    CrossSectionalMomentum,
    MultiAssetStrategy,
    TimeSeriesMomentum,
    TrendFollowing,
)


def median_ms(fn: Any, n: int = 5) -> float:  # noqa: ANN401
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3


def report(label: str, data: Any, research: Any, capital: float) -> None:  # noqa: ANN401
    prices = C.Prices.from_data(data)
    targets = compute_targets(research, data.frames)
    # A multi-asset strategy needs compute_targets' alignment, so only the total is timed.
    maths = (
        float("nan")
        if isinstance(research, MultiAssetStrategy)
        else median_ms(lambda: [research.generate_signals(f) for f in data.frames.values()])
    )
    total = median_ms(lambda: compute_targets(research, data.frames))
    panel = median_ms(lambda: C.target_matrix(prices, data, targets))
    p = C.panel(research, data, prices, targets=targets)
    sim = median_ms(lambda: E.kernel(p, capital))
    print(
        f"| {label} | {maths:.2f} | {total:.2f} | {panel:.2f} | {sim:.2f} | "
        f"{(total + panel) / sim:.0f}x |"
    )


def main() -> None:
    print("| Workload | generate_signals | compute_targets | panel | kernel | targets/kernel |")
    print("|---|---|---|---|---|---|")
    report("AAPL TrendFollowing", F.aapl(), TrendFollowing(), 1e6)
    report("19 futures TSMOM", F.glbx_futures(), TimeSeriesMomentum(), 1e8)
    report("8 equities CSMOM", F.equities(), CrossSectionalMomentum(), 1e6)
    full = []
    for s, venue in cat.list_bar_series("ohlcv-1d"):
        if venue == "ALPACA" and cat.load_bars(s, "ohlcv-1d").height == 2690:
            full.append(s)
        if len(full) == 400:
            break
    for n in (100, 400):
        data = load_market_data(full[:n])
        report(f"{n} equities TrendFollowing", data, TrendFollowing(), 1e6)
        report(f"{n} equities CSMOM", data, CrossSectionalMomentum(), 1e6)


if __name__ == "__main__":
    main()
