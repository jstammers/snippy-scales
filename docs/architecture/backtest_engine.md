# Backtesting Engine

## Overview

The backtesting system is the core component that simulates trading strategies over
historical market data.  SnippyScales provides two complementary backtesting
approaches:

1. **Vectorised backtesting** (Python + Rust) — fast, exploratory research on full
   historical datasets, powered by the `raptorbt` Rust extension.
2. **Event-driven backtesting** (Rust) — realistic, tick-by-tick simulation *(planned)*.

This flexibility lets you iterate quickly during research, then stress-test final
strategies with realistic fills and market microstructure.

---

## Module Responsibility Contract

| Module | Responsibility | Must NOT contain |
|---|---|---|
| `backtesting/` | Execute a backtest, return a `BacktestResult` | Fold/walk-forward logic, analysis aggregation |
| `evaluation/` | Parameter sweeps, fold analysis, persistence, tearsheets | Execution details |
| `research/` | Feature engineering helpers | Magic constants |
| `_constants.py` | Package-wide conventions (`TRADING_DAYS_PER_YEAR`) | — |

---

## Architecture

### Module layout

```
snippy_scales/
├── _constants.py          ← TRADING_DAYS_PER_YEAR = 252
├── backtesting/
│   ├── domain.py          ← Trade, BacktestMetrics, BacktestResult (engine-agnostic)
│   ├── engine.py          ← ExecutionEngine Protocol, RaptorExecutionEngine, make_config
│   ├── signals.py         ← SignalBundle, PositionInterpreter, SignFlipInterpreter
│   ├── allocation.py      ← VolTargetAllocator
│   ├── data.py            ← OhlcvArrays, normalize_ohlcv, to_numpy_ohlcv
│   ├── runners.py         ← BacktestRunner, BasketRunner (high-level helpers)
│   └── store.py           ← BacktestStore (DuckDB — single-run analytics)
└── evaluation/
    ├── results.py         ← FoldResult, SweepResult, EvaluationResult
    ├── split.py           ← WalkForwardSplit, SplitFold
    ├── sweep.py           ← ParameterGrid, RandomSearch, OptunaSearch
    ├── runner.py          ← EvaluationRunner (walk-forward + param search)
    ├── database.py        ← SQLiteStore (metadata), AnalyticsStore (DuckDB)
    └── tearsheet.py       ← TearsheetGenerator
```

---

### Vectorised Backtesting — ✅ Implemented

The research layer provides fast, **vectorised backtesting** using
[raptorbt](https://github.com/willybrauner/raptorbt) — a Rust extension that processes
entire OHLCV datasets as arrays without any Python loops.

**How it works:**

- Entire OHLCV dataset is loaded into NumPy arrays
- Entry/exit signals are computed in Python (vectorised)
- `raptorbt` processes signals as arrays — simulating fills bar-by-bar in Rust
- Results are converted to typed Python domain objects

**Key components:**

#### `ExecutionEngine` (Protocol)

Swappable execution backends — any class implementing `execute()` is valid.

```python
@runtime_checkable
class ExecutionEngine(Protocol):
    def execute(
        self,
        instruments: list[InstrumentSpec],
        *,
        config: Any | None = None,
        sync_mode: str = "any",
    ) -> BacktestResult: ...
```

#### `RaptorExecutionEngine`

Default implementation backed by `raptorbt`.

```python
engine = RaptorExecutionEngine()
result = engine.execute(instruments, config=config, sync_mode="any")
```

#### `SignalBundle`

Four boolean arrays produced by interpreting a signed-position time series:

```python
@dataclass(frozen=True)
class SignalBundle:
    long_entries:  np.ndarray   # True on bar where long position opens
    long_exits:    np.ndarray   # True on bar where long position closes
    short_entries: np.ndarray   # True on bar where short position opens
    short_exits:   np.ndarray   # True on bar where short position closes
```

#### `InstrumentSpec`

Immutable container for one instrument leg in a basket backtest:

```python
@dataclass(frozen=True)
class InstrumentSpec:
    symbol:     str
    timestamps: np.ndarray   # nanoseconds since epoch
    open:       np.ndarray
    high:       np.ndarray
    low:        np.ndarray
    close:      np.ndarray
    volume:     np.ndarray
    entries:    np.ndarray   # boolean — long or short entries
    exits:      np.ndarray   # boolean — position closes
    direction:  int          # 1 = long, -1 = short
    weight:     float        # fraction of capital allocated
```

#### `make_config()`

```python
config = make_config(
    initial_capital=100_000.0,   # NOT initial_cash
    fees=0.001,
    slippage=0.0005,
    upon_bar_close=True,
)
```

#### `BacktestResult`

```python
@dataclass
class BacktestResult:
    symbol:         str | list[str]
    metrics:        BacktestMetrics
    equity_curve:   np.ndarray   # float64
    drawdown_curve: np.ndarray   # float64
    returns:        np.ndarray   # float64 period returns
    trades:         list[Trade]
```

---

## BacktestMetrics — 33 fields

`BacktestMetrics` is a frozen dataclass holding all performance statistics returned
by `raptorbt`.  Fields are grouped below.

### Core performance

| Field | Type | Description |
|---|---|---|
| `total_return_pct` | `float` | Total strategy return as a percentage |
| `sharpe_ratio` | `float` | Annualised Sharpe (risk-free = 0) |
| `sortino_ratio` | `float` | Annualised Sortino ratio |
| `calmar_ratio` | `float` | Return / max-drawdown ratio |
| `omega_ratio` | `float` | Probability-weighted gains-to-losses ratio |

### Drawdown

| Field | Type | Description |
|---|---|---|
| `max_drawdown_pct` | `float` | Maximum peak-to-trough drawdown (%) |
| `max_drawdown_duration` | `int` | Longest drawdown in bars |

### Trade counts

| Field | Type | Description |
|---|---|---|
| `total_trades` | `int` | All trades executed |
| `total_closed_trades` | `int` | Closed positions |
| `total_open_trades` | `int` | Currently open positions |
| `winning_trades` | `int` | Profitable transactions |
| `losing_trades` | `int` | Unprofitable transactions |

### Trade performance

| Field | Type | Description |
|---|---|---|
| `win_rate_pct` | `float` | % of trades that were profitable |
| `profit_factor` | `float` | Gross profit / gross loss |
| `expectancy` | `float` | Average profit per trade (cash units) |
| `sqn` | `float` | System Quality Number |
| `avg_trade_return_pct` | `float` | Average return across all trades |
| `avg_win_pct` | `float` | Mean return of winning trades |
| `avg_loss_pct` | `float` | Mean return of losing trades |
| `best_trade_pct` | `float` | Maximum single-trade return |
| `worst_trade_pct` | `float` | Minimum single-trade return |
| `payoff_ratio` | `float` | Avg win return / avg loss return |
| `recovery_factor` | `float` | Net profit / max drawdown |

### Duration

| Field | Type | Description |
|---|---|---|
| `avg_holding_period` | `float` | Average trade duration (bars) |
| `avg_winning_duration` | `float` | Mean duration of winning trades |
| `avg_losing_duration` | `float` | Mean duration of losing trades |

### Streaks

| Field | Type | Description |
|---|---|---|
| `max_consecutive_wins` | `int` | Longest winning streak |
| `max_consecutive_losses` | `int` | Longest losing streak |

### Portfolio

| Field | Type | Description |
|---|---|---|
| `start_value` | `float` | Initial capital |
| `end_value` | `float` | Final portfolio value |
| `total_fees_paid` | `float` | Cumulative transaction costs |
| `open_trade_pnl` | `float` | Unrealised PnL from active positions |
| `exposure_pct` | `float` | % of time in market |

**Correct field name examples:**

```python
result.metrics.max_drawdown_pct         # NOT max_drawdown
result.metrics.total_return_pct         # NOT total_return
result.metrics.total_fees_paid          # NOT fees
```

---

## Evaluation Layer

The `evaluation/` module provides out-of-sample validation via walk-forward
cross-validation and parameter search.

### Result hierarchy

```
EvaluationResult
└── sweep_results: list[SweepResult]          ← one per parameter set
    └── folds: list[FoldResult]               ← one per (params, fold) combo
        ├── train_metrics: BacktestMetrics
        ├── test_metrics:  BacktestMetrics
        ├── train_equity_curve: np.ndarray
        └── test_equity_curve:  np.ndarray
```

### `SweepResult` — aggregate statistics

For each parameter set, `SweepResult` computes mean **and** population
std-dev across all test folds:

| Property | Description |
|---|---|
| `mean_test_sharpe` / `std_test_sharpe` | Sharpe ratio across folds |
| `mean_test_return` / `std_test_return` | Total return (%) across folds |
| `mean_test_max_dd` / `std_test_max_dd` | Max drawdown (%) across folds |
| `mean_test_sortino` / `std_test_sortino` | Sortino ratio across folds |

### `EvaluationResult` — top-level container

```python
result.best_params          # dict — highest mean OOS Sharpe
result.best_result          # SweepResult with best mean Sharpe
result.summary_df()         # Polars DataFrame, sorted by mean_test_sharpe DESC
result.oos_equity_curve()   # stitched OOS equity curve for best params (starts at 1.0)
```

`summary_df()` columns include all parameter names plus:
`mean_test_sharpe`, `std_test_sharpe`, `mean_test_return_pct`, `std_test_return_pct`,
`mean_test_max_dd_pct`, `std_test_max_dd_pct`, `mean_test_sortino`, `std_test_sortino`,
`n_folds`.

---

## Database Architecture

SnippyScales uses a **hybrid persistence model** separating analytics from metadata.

```
┌──────────────────────────────┐     ┌────────────────────────────────────────┐
│   SQLite — metadata.db       │     │   DuckDB — analytics.duckdb            │
│                              │     │                                        │
│  experiments                 │◄────│  sweep_results.experiment_id           │
│    id, name, strategy_class  │     │  sweep_results (mean+std, 4 metrics)   │
│    symbols, config, n_splits │     │  fold_results  (4 train + 33 test)     │
│    window_type, created_at   │     │  backtest_runs (all 33 metrics)        │
│                              │     │  [future: trades, pnl series, signals] │
└──────────────────────────────┘     └────────────────────────────────────────┘
```

**SQLite** stores lightweight relational metadata — experiment configuration and
run registry.  One row per evaluation run.

**DuckDB** stores wide columnar analytics data — per-fold metrics, per-param-set
aggregated statistics, single-pass backtest runs.

### `SQLiteStore` — metadata

```python
from snippy_scales.evaluation.database import SQLiteStore

store = SQLiteStore("data/metadata.db")
experiment_id = store.save_evaluation(eval_result, config={
    "initial_capital": 1_000_000,
    "n_splits": 5,
    "window": "expanding",
})
df = store.load_experiments()   # Polars DataFrame of all runs
```

`save_evaluation()` writes only to the `experiments` table and returns the
integer primary key.  Sweep/fold data belong in `AnalyticsStore`.

### `AnalyticsStore` — DuckDB analytics

```python
from snippy_scales.evaluation.database import AnalyticsStore

store = AnalyticsStore("data/analytics.duckdb")
store.save_evaluation_analytics(eval_result, experiment_id=1)

sweep_df = store.load_sweep_results(experiment_id=1)
fold_df   = store.load_fold_results(sweep_id="<uuid>")
```

Tables:
- **`sweep_results`** — one row per (experiment, parameter set) with mean and
  std for the 4 key test metrics.
- **`fold_results`** — one row per (sweep, fold) with 4 training-window metrics
  and all 33 test-window metrics.

### `BacktestStore` — DuckDB single-run analytics

```python
from snippy_scales.backtesting.store import BacktestStore

store = BacktestStore("data/analytics.duckdb")
run_id = store.save_run(
    result,
    strategy_name="TrendFollowing",
    initial_capital=100_000.0,
    fees=0.001,
    slippage=0.0005,
)
```

---

## Workflow

### 1. Vectorised single-pass backtest

```python
from snippy_scales.backtesting import (
    make_config,
    run_single,
    run_long_short,
    run_basket,
    InstrumentSpec,
)

# Single direction
result = run_single(
    symbol="ES.c.0",
    timestamps=bars["timestamp"].to_numpy(),
    open_prices=bars["open"].to_numpy(),
    high_prices=bars["high"].to_numpy(),
    low_prices=bars["low"].to_numpy(),
    close_prices=bars["close"].to_numpy(),
    volume=bars["volume"].to_numpy(),
    entries=entry_signals,
    exits=exit_signals,
    direction=1,            # 1 = long, -1 = short
    weight=1.0,
    config=make_config(initial_capital=1_000_000.0),   # NOT initial_cash
)

print(f"Sharpe:     {result.metrics.sharpe_ratio:.2f}")
print(f"Max DD:     {result.metrics.max_drawdown_pct:.2%}")
print(f"Return:     {result.metrics.total_return_pct:.2%}")
print(f"Win rate:   {result.metrics.win_rate_pct:.1f}%")
```

### 2. Long/short simultaneous strategy

```python
result = run_long_short(
    symbol="ES.c.0",
    timestamps=ts, open_prices=o, high_prices=h,
    low_prices=l, close_prices=c, volume=v,
    long_entries=le, long_exits=lx,    # NOT entries/exits/direction
    short_entries=se, short_exits=sx,
    long_weight=0.5, short_weight=0.5,
    config=make_config(initial_capital=1_000_000.0),
)
```

### 3. Walk-forward evaluation with parameter search

```python
from snippy_scales.evaluation import (
    EvaluationRunner,
    ParameterGrid,
    RandomSearch,
)
from snippy_scales.strategies.trend import TrendFollowing

runner = EvaluationRunner(
    n_splits=5,
    window="expanding",
    initial_capital=1_000_000.0,
    fees=0.001,
    slippage=0.0005,
    db_path="data/metadata.db",                      # SQLite — experiment registry
    analytics_db_path="data/analytics.duckdb",       # DuckDB — sweep/fold metrics
    tearsheet_dir="reports/",
)

grid = ParameterGrid({"fast_period": [10, 20, 40], "slow_period": [40, 60, 120]})
result = runner.evaluate(TrendFollowing, bars, params=grid, symbol="ES.c.0")

print(result.best_params)
print(result.summary_df())
```

### 4. VolTargetAllocator

```python
from snippy_scales.backtesting import VolTargetAllocator

allocator = VolTargetAllocator(vol_target=0.15, max_leverage=2.0)
weight = allocator.weight(close)      # NOT allocator.allocate(positions, returns)
```

---

## Event-Driven Backtesting — 🔄 Planned

A realistic, tick-by-tick Rust backtesting engine is planned for production
validation of final strategies.

**Planned characteristics:**
- Configurable fill models (slippage, market impact, partial fills)
- Order queuing and latency simulation
- Real-time signal callbacks from Python
- Risk checks (drawdown limits, max position size)
- Multi-asset correlation handling

---

## Performance

The vectorised engine is fast because:

- **Zero-copy data structures** — signals are passed as NumPy arrays directly to Rust
- **Compiled Rust core** — `raptorbt` processes fills in a tight loop without GIL
- **No Python loops** — bar-by-bar simulation runs entirely in Rust

Expect seconds-scale backtest times for years of daily data.

---

## Extending the Engine

### Custom strategy signals

```python
from snippy_scales.strategies.base import Strategy

class MyStrategy(Strategy):
    def generate_signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        # return bars with "position" column: +1 long, -1 short, 0 flat
        ...
```

### Custom position interpreter

```python
from snippy_scales.backtesting import PositionInterpreter, SignalBundle

class MyInterpreter:
    def interpret(self, positions: np.ndarray) -> SignalBundle:
        # convert signed positions to four boolean arrays
        return SignalBundle(
            long_entries=..., long_exits=...,
            short_entries=..., short_exits=...,
        )
```

### Custom execution backend

Any class implementing the `ExecutionEngine` Protocol can be substituted:

```python
from snippy_scales.backtesting import ExecutionEngine, InstrumentSpec, BacktestResult

class MyEngine:
    def execute(
        self,
        instruments: list[InstrumentSpec],
        *,
        config=None,
        sync_mode: str = "any",
    ) -> BacktestResult:
        ...
```

---

## Limitations

### Vectorised backtesting (current)

- Bar-close execution only (no intra-bar fills)
- No dynamic order routing or risk checks
- Single-threaded execution per run

### Event-driven backtesting (planned)

See the *Planned* section above.
