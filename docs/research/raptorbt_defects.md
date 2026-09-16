# raptorbt 0.8.0 — two metric defects

Two narrow, reproducible defects found in `raptorbt==0.8.0` (verified 2026-08-15) while building
a second execution engine and cross-checking it against raptorbt's own output.

Both are **metric-reporting** defects. Equity curves, trade lists and P&L are correct throughout
— only derived statistics are affected. That is what makes them easy to miss: the backtest is
right, the number you read off it is not.

Two earlier defects present in `0.3.2.post1` were **already fixed** by 0.8.0 and are recorded at
the bottom for completeness.

Upstream: <https://github.com/alphabench/raptorbt/issues>

---

## 1. `calmar_ratio` ignores `periods_per_year`

`BacktestConfig(periods_per_year=...)` correctly rescales `sharpe_ratio` and `sortino_ratio`,
but `calmar_ratio` is unaffected by it and appears hard-wired to 365.

```python
import numpy as np, raptorbt

n = 300
rng = np.random.default_rng(0)
close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
ts = 1_577_836_800_000_000_000 + np.arange(n, dtype=np.int64) * 86_400_000_000_000
entries = np.zeros(n, bool); exits = np.zeros(n, bool)
for bar, arr in ((10, entries), (50, exits), (100, entries), (180, exits)):
    arr[bar] = True

def run(ppy):
    cfg = raptorbt.BacktestConfig(
        initial_capital=1e5, fees=0.0, slippage=0.0, periods_per_year=ppy
    )
    return raptorbt.run_single_backtest(
        timestamps=ts, open=close.copy(), high=close * 1.001, low=close * 0.999,
        close=close, volume=np.full(n, 1e6), entries=entries, exits=exits,
        direction=1, weight=1.0, symbol="X", config=cfg,
    ).metrics

for ppy in (252, 365, 1000):
    m = run(ppy)
    print(f"{ppy:>5}  sharpe={m.sharpe_ratio:.6f}  sortino={m.sortino_ratio:.6f}  "
          f"calmar={m.calmar_ratio:.6f}")
```

Observed:

```
  252  sharpe=0.342824  sortino=0.475868  calmar=0.486230
  365  sharpe=0.412588  sortino=0.572708  calmar=0.486230
 1000  sharpe=0.682921  sortino=0.947952  calmar=0.486230
```

`calmar_ratio` is byte-identical across a 4× change in the annualisation factor, while the other
two scale as expected.

**Confirmation that 365 is the value being used.** Calmar is annualised return over maximum
drawdown; computing it by hand from the same equity curve:

| Annualisation | Hand-computed calmar |
|---|---|
| 252 | 0.332414 |
| 365 | **0.484576** ← matches the reported 0.486230 |

**Impact.** On daily bars, `calmar_ratio` is overstated by roughly 1.46× — larger than the
`sqrt(365/252)` ≈ 1.20 error on Sharpe, because the factor enters the return annualisation
through an exponent rather than a square root. On intraday bars the error is far larger.

**Expected.** `calmar_ratio` should use `periods_per_year` for its annualisation, consistently
with `sharpe_ratio` and `sortino_ratio`.

---

## 2. Basket path returns zero for several trade-level metrics

`run_basket_backtest` returns `0` for `exposure_pct`, `omega_ratio` and `max_drawdown_duration`,
while `run_single_backtest` populates all three from a byte-identical equity curve.

```python
cfg = raptorbt.BacktestConfig(initial_capital=1e5, fees=0.0, slippage=0.0, periods_per_year=252)

single = raptorbt.run_single_backtest(
    timestamps=ts, open=close.copy(), high=close * 1.001, low=close * 0.999,
    close=close, volume=np.full(n, 1e6), entries=entries, exits=exits,
    direction=1, weight=1.0, symbol="X", config=cfg,
)
basket = raptorbt.run_basket_backtest(
    instruments=[(ts, close.copy(), close * 1.001, close * 0.999, close,
                  np.full(n, 1e6), entries, exits, 1, 1.0, "X")],
    config=cfg, sync_mode="any",
)

assert np.allclose(single.equity_curve(), basket.equity_curve())  # passes

for name in ("exposure_pct", "omega_ratio", "max_drawdown_duration"):
    print(f"{name:24s} single={getattr(single.metrics, name):>8.3f}  "
          f"basket={getattr(basket.metrics, name):>8.3f}")
```

Observed:

```
exposure_pct             single=  40.000  basket=   0.000
omega_ratio              single=   1.090  basket=   0.000
max_drawdown_duration    single= 155.000  basket=   0.000
```

Reproduces across every combination tested of `n ∈ {300, 600, 1500}` and seeds `{0, 1, 2}`.

**Impact.** Any wrapper that routes through the basket path — which is the natural choice for
multi-instrument work — silently records zero exposure and zero omega. A zero is
indistinguishable from a genuine measurement, so it is easy to aggregate without noticing.

**Expected.** The basket path should populate these three fields as the single-instrument path
does, or return `None` to signal "not computed" rather than a value that reads as a measurement.

---

## Already fixed in 0.8.0

Recorded because they were present in `0.3.2.post1`, which this project was pinned to until now.
Anyone else on an older pin should upgrade.

| Defect | Status in 0.8.0 |
|---|---|
| Basket path reported Sharpe **4–7× higher** than the single path on identical equity curves; the factor varied with the data, so results could not be rescaled after the fact | **Fixed** — ratio is now 1.000 across all tested configurations |
| Sharpe/Sortino annualised with 365 regardless of bar frequency, with no way to override | **Fixed** — `BacktestConfig(periods_per_year=...)` works correctly for both. The default remains 365; a 252 default would be a friendlier choice for daily-bar users, but it is at least now configurable |

---

## Environment

```
raptorbt   0.8.0
numpy      2.4.2
python     3.12
platform   linux
```

Regression guards for both open defects live in
`python/tests/test_continuous_engine.py`:
`test_raptorbt_basket_path_still_drops_trade_level_metrics` and the `_RATIO_FIELDS` exclusion in
`test_parity_metrics_match_raptorbt`. Both will start failing once upstream fixes land, which is
the intended signal to remove the workarounds.
