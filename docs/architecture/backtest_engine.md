# Backtesting Engine

## Overview

The backtesting system is the core component that simulates trading strategies over historical market data. SnippyScales provides two complementary backtesting approaches:

1. **Vectorised backtesting** (Python) — for fast, exploratory research on full historical datasets
2. **Event-driven backtesting** (Rust) — for realistic, tick-by-tick simulation (yet to be implemented)

This flexibility lets you iterate quickly during research, then stress-test final strategies with realistic fills and market microstructure.

## Architecture

SnippyScales backtesting consists of two implementations, each with different trade-offs:

### Approach 1: Vectorised Backtesting (Python) — ✅ Implemented

The research layer provides fast, **vectorised backtesting** using [raptorbt](https://github.com/willybrauner/raptorbt).

**How it works:**
- Entire OHLCV dataset is loaded into NumPy/Polars arrays
- Entry/exit signals are vectorised (broadcasted across all bars)
- Orders are filled using vectorised fill model
- Portfolio state is computed bar-by-bar
- Results are accumulated into metrics (Sharpe, max drawdown, etc.)

**Key components:**

- **ExecutionEngine** (Python Protocol) — swappable execution backends
  ```python
  @runtime_checkable
  class ExecutionEngine(Protocol):
      def run(self, instruments: List[InstrumentSpec]) -> BacktestResult: ...
  ```

- **RaptorExecutionEngine** — default implementation using raptorbt
  ```python
  class RaptorExecutionEngine(ExecutionEngine):
      def run(self, instruments: List[InstrumentSpec]) -> BacktestResult:
          return raptorbt.run_basket_backtest(...)
  ```

- **SignalBundle** — converts strategy output to entry/exit arrays
  ```python
  class SignalBundle:
      entries: np.ndarray      # boolean, True on entry bar
      exits: np.ndarray        # boolean, True on exit bar
      direction: int           # 1 for long, -1 for short
      weight: float            # position size fraction
  ```

- **BacktestResult** — typed results with metrics
  ```python
  @dataclass
  class BacktestResult:
      trades: List[Trade]      # all executed trades
      equity_curve: np.ndarray  # portfolio equity over time
      metrics: BacktestMetrics  # Sharpe, max DD, returns, etc.
  ```

**Characteristics:**
- ✅ Fast (sec-scale for years of data)
- ✅ Ideal for research, parameter sweeps, cross-validation
- ✅ Works with pre-computed signals
- ❌ Limited market microstructure simulation
- ❌ No tick-level fills or partial fills
- ❌ Bar-close execution only

**Use case:** Research layer — testing signal quality and strategy viability quickly.

---

### Approach 2: Event-Driven Backtesting (Rust) — 🔄 Planned

A **realistic, tick-by-tick backtesting engine** (yet to be implemented) in Rust for final validation.

**Planned architecture:**

```rust
pub struct BacktestEngine<F: FillModel> {
    pub portfolio: Portfolio,
    pub fill_model: F,
    pending_orders: VecDeque<Order>,
}

impl<F: FillModel> BacktestEngine<F> {
    pub fn new(initial_cash: f64, fill_model: F) -> Self { ... }
    pub fn submit_order(&mut self, order: Order) { ... }
    pub fn step(&mut self, event: &MarketEvent) -> Vec<Fill> { ... }
}
```

**Planned components:**

- **FillModel** — simulates realistic fills
  ```rust
  pub trait FillModel {
      fn simulate(&self, order: &Order, market_price: f64, slippage: f64) -> Fill;
  }
  ```
  Built-in implementations (planned):
  - **ImmediateFill** — instant fills at market (baseline)
  - **SlippageFill** — configurable BP slippage
  - **ProcessingFill** — multi-bar order processing delays

- **Portfolio** — tracks positions and P&L
  ```rust
  pub struct Portfolio {
      cash: f64,
      positions: HashMap<String, Position>,
      equity_curve: Vec<f64>,
  }
  ```

- **Event loop** — processes bars/ticks sequentially
  - Emit signals from Python
  - Submit orders
  - Match against fills
  - Update portfolio
  - Record results

**Planned characteristics:**
- ✅ Realistic — covers slippage, fills, delays
- ✅ Fast — compiled Rust, efficient data structures
- ✅ Deterministic — reproducible across runs
- ❌ More complex than vectorised approach
- ❌ Requires signal implementation in Rust (or callbacks)

**Use case:** Production validation — stress-test final strategies with realistic assumptions.

---

### Comparison

| Aspect | Vectorised (Python) | Event-Driven (Rust) |
|--------|---------------------|---------------------|
| **Status** | ✅ Ready | 🔄 Planned |
| **Speed** | Fast (vectorised) | Very fast (compiled) |
| **Use case** | Research, prototyping | Production validation |
| **Fills** | Simple (bar-close) | Realistic (configurable) |
| **Signals** | Pre-computed arrays | Real-time from Python callbacks |
| **Complexity** | Low | Medium |
| **Test coverage** | Broad | Planned |

## Workflow

This is the current **vectorised backtesting** workflow:

```
┌──────────────────────────────────────────┐
│ 1. Load historical OHLCV data            │
│    (from local data store or API)        │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│ 2. Generate signals (Python)             │
│    Strategy.generate_signals(bars)       │
│    → entry/exit points, position sizing  │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│ 3. Prepare instrument specs (vectorised) │
│    (OHLCV × signals → InstrumentSpec)    │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│ 4. Run vectorised backtest (raptorbt)    │
│    Entire dataset processed as arrays    │
│    → fills, equity curve, metrics        │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│ 5. Analyze results (Python)              │
│    Sharpe, max drawdown, returns         │
└──────────────────────────────────────────┘
```

**Future: Event-driven backtesting workflow**

```
┌──────────────────────────────────────────┐
│ 1. Load historical OHLCV data            │
│    (as nested array or feed)             │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│ 2. For each bar:                         │
│    a) Emit signal (Python callback)      │
│    b) Submit order to engine             │
│    c) Engine simulates fill (realistic)  │
│    d) Update portfolio                   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│ 3. Collect results                       │
│    Fills, equity curve, metrics          │
└──────────────────────────────────────────┘
```

## Running a Backtest

Currently, all backtests use the **vectorised approach** (fast, exploratory).

### From the CLI

```bash
just algo backtest run trend_following --symbol ES.c.0 --start 2020-01-01 --end 2024-12-31
```

This invokes the vectorised runner under the hood.

### Programmatically (Python)

```python
from snippy_scales.backtesting import BacktestRunner, run_basket
from snippy_scales.strategies.trend import TrendFollowing

# High-level API
runner = BacktestRunner(
    strategy=TrendFollowing(),
    instruments=["ES.c.0"],
    start_date="2020-01-01",
    end_date="2024-12-31",
    initial_capital=1_000_000.0,
)

result = runner.run()
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
print(f"Max DD: {result.metrics.max_drawdown:.2%}")
print(f"Total return: {result.metrics.total_return:.2%}")
```

Or use the low-level raptorbt API directly:

```python
from snippy_scales.backtesting import (
    InstrumentSpec,
    run_basket,
    make_config,
)

# Prepare instrument specs from pre-computed signals
specs = [
    InstrumentSpec(
        symbol="ES.c.0",
        timestamps=bars["timestamp"].to_numpy(),
        open=bars["open"].to_numpy(),
        high=bars["high"].to_numpy(),
        low=bars["low"].to_numpy(),
        close=bars["close"].to_numpy(),
        volume=bars["volume"].to_numpy(),
        entries=entry_signals,  # boolean array
        exits=exit_signals,      # boolean array
        direction=1,             # 1 = long, -1 = short
        weight=1.0,              # position size
    ),
]

config = make_config(initial_cash=1_000_000.0, fees=0.001)
result = run_basket(specs, config)
```

## Configuration

Backtest configuration is set via:
- Command-line flags (`--symbol`, `--start`, `--end`)
- Strategy hyperparameters (e.g., `fast_period`, `vol_target`)
- Fill model parameters (slippage, commissions)

## Performance

Because the core engine is implemented in Rust with:
- **Zero-copy data structures** (native arrays)
- **Event-driven processing** (no full-history copies)
- **Compiled, optimized code** (orders of magnitude faster than pure Python)

You can backtest years of minute-level data in seconds.

## Extending the Engine

### Vectorised Backtesting (Python)

Currently, you can customize:

1. **Strategy signals** — implement your own `Strategy` subclass
   ```python
   class MyStrategy(Strategy):
       def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
           # your logic here
           return positions.rename(\"position\")
   ```

2. **Position interpretation** — convert signals to entry/exit arrays
   ```python
   class MyPositionInterpreter(PositionInterpreter):
       def interpret(self, positions: pl.Series) -> Tuple[np.ndarray, np.ndarray]:
           # convert to entry/exit signals
           return entries, exits
   ```

3. **Allocation** — scale position sizes based on volatility or risk
   ```python
   allocator = VolTargetAllocator(vol_target=0.15)
   scaled_positions = allocator.allocate(positions, returns)
   ```

### Event-Driven Backtesting (Rust) — Future

When implemented, you'll be able to add custom fill models:

```rust
pub struct MyFillModel {
    slippage_bps: f64,
    market_impact: f64,
}

impl FillModel for MyFillModel {
    fn simulate(&self, order: &Order, market_price: f64, slippage: f64) -> Fill {
        // custom fill logic with market impact, execution delays, etc.
    }
}
```

Then rebuild with `just dev`.

## Limitations & Future Work

### Vectorised Backtesting (Current)

**Limitations:**
- Bar-close execution only (no intra-bar fills)
- Simplified fill model (no slippage, commissions, or partial fills)
- All entry/exit signals must be pre-computed
- No support for dynamic order routing or risk checks
- Single-threaded

**Future improvements:**
- Parameter sensitivity analysis (parallel sweeps)
- Walk-forward optimization
- Anchored rolling window backtests

### Event-Driven Backtesting (Planned)

**To implement:**
- ✅ Tick-level or sub-bar frequency processing
- ✅ Configurable fill models (slippage, commissions, market impact)
- ✅ Partial fills and order queuing
- ✅ Real-time signal generation (from Python callbacks)
- ✅ Risk checks (max position size, drawdown limits)
- ✅ Parallel backtest runs (multiple strategies/parameters)
- ❌ Margin and leverage support (future phase)
- ❌ Multi-asset correlations and hedging (future phase)
