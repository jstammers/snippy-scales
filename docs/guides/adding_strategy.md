# Adding a Strategy

## Overview

Strategies in SnippyScales are Python classes that inherit from the `Strategy` base class and implement a `generate_signals()` method. The method consumes a bar DataFrame and returns target positions.

## Strategy Interface

All strategies inherit from `Strategy`:

```python
from abc import ABC, abstractmethod
import polars as pl

class Strategy(ABC):
    """Abstract base for all strategies.

    A strategy consumes a bar DataFrame and returns a Series of target
    positions (float, signed; e.g. 1.0 = long 1 unit, -0.5 = short 0.5).
    """

    @abstractmethod
    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        """Return a Series of target positions, same length as bars."""
        ...
```

**Key inputs:**
- `bars`: Polars DataFrame with OHLCV columns (`open`, `high`, `low`, `close`, `volume`)

**Key output:**
- A Polars Series of signed floats representing target position sizes
  - Positive = long
  - Negative = short
  - 0 = flat

## Step-by-Step Example

### 1. Create a New Strategy File

Create `python/snippy_scales/strategies/my_strategy.py`:

```python
"""My custom mean-reversion strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from snippy_scales.strategies import Strategy

if TYPE_CHECKING:
    import polars as pl


class MyMeanReversion(Strategy):
    """Revert to the 50-day moving average.

    Long when price is below MA, short when above.
    """

    def __init__(self, ma_period: int = 50, position_size: float = 1.0):
        self.ma_period = ma_period
        self.position_size = position_size

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        close = bars["close"]
        ma = close.rolling_mean(self.ma_period)

        # Signal: revert to MA
        deviation = close - ma
        signal = (-deviation / ma).clip(min_value=-self.position_size, max_value=self.position_size)

        return signal.fill_null(0.0).rename("position")
```

### 2. Register in `__init__.py`

Update `python/snippy_scales/strategies/__init__.py`:

```python
from snippy_scales.strategies.my_strategy import MyMeanReversion

__all__ = [
    "Strategy",
    "TrendFollowing",
    "MyMeanReversion",
    # ... other strategies
]
```

### 3. Test Locally

```bash
# Run a quick test
just test-py
```

Or test manually:

```python
from snippy_scales.strategies.my_strategy import MyMeanReversion
import polars as pl

strat = MyMeanReversion(ma_period=50)

# Create dummy bars
bars = pl.DataFrame({
    "close": [100.0, 101.0, 99.0, 102.0, 98.0] * 20,  # 100 bars
})

signals = strat.generate_signals(bars)
print(signals)
```

### 4. Run a Backtest

```bash
just algo backtest run my_strategy --symbol ES.c.0 --start 2020-01-01 --end 2024-12-31
```

Or programmatically:

```python
from snippy_scales.backtesting.runner import BacktestRunner
from snippy_scales.strategies.my_strategy import MyMeanReversion

runner = BacktestRunner(
    strategy=MyMeanReversion(ma_period=50),
    instruments=["ES.c.0"],
    start_date="2020-01-01",
    end_date="2024-12-31",
)

result = runner.run()
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Max DD: {result.max_drawdown:.2%}")
print(f"Returns: {result.total_return:.2%}")
```

## Advanced Patterns

### 1. Multi-Timeframe Signals

Combine signals from multiple windows:

```python
def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
    close = bars["close"]

    # Long-term trend
    slow_ma = close.rolling_mean(200)
    trend = (close - slow_ma).sign()

    # Short-term reversion
    fast_ma = close.rolling_mean(20)
    reversion = -(close - fast_ma).sign()

    # Combined: only trade reversions when trend is favorable
    combined = trend * reversion * self.position_size

    return combined.fill_null(0.0).rename("position")
```

### 2. Volatility-Adjusted Sizing

Scale position size by realized volatility:

```python
def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
    close = bars["close"]
    returns = close.pct_change()

    # Realized vol (annualized)
    realized_vol = returns.rolling_std(20) * (252 ** 0.5)

    # Raw signal (e.g., from MA crossover)
    raw_signal = ...  # your logic here

    # Scale by inverse of vol (target vol = 15%)
    vol_scalar = (0.15 / realized_vol.clip(lower_bound=1e-6)).clip(upper_bound=5.0)

    positions = raw_signal * vol_scalar
    return positions.fill_null(0.0).rename("position")
```

### 3. Using Features from Research

Integrate engineered features:

```python
from snippy_scales.research.features import compute_features

class FeatureBasedStrategy(Strategy):
    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        # Compute rich features
        features = compute_features(bars)

        # Use features for signal generation
        signal = (
            features["momentum"] * 0.5 +
            features["mean_reversion"] * 0.3 +
            features["volatility_regime"] * 0.2
        )

        return signal.clip(min_value=-1.0, max_value=1.0).rename("position")
```

## Best Practices

1. **Start simple** — implement the core logic first, optimize later
2. **Handle NaN/null** — always fill null values in the return series
3. **Bound positions** — use `.clip()` to prevent extreme leverage
4. **Log hyperparameters** — store them as instance attributes for reproducibility
5. **Test edge cases** — verify behavior with missing data, gaps, and edge values
6. **Document assumptions** — explain what the strategy assumes about the market

## Common Pitfalls

❌ **Forward-looking bias**: Don't use future bars in the signal calculation

```python
# WRONG
signal = (close - close.shift(-1)).sign()  # uses future close!
```

✅ **Correct approach**: Only use current and past bars

```python
# RIGHT
signal = (close - close.shift(1)).sign()  # uses past close
```

❌ **Uninitialized indicators**: Don't assume indicators are available before the lookback period

```python
# WRONG
ma = close.rolling_mean(50)  # first 49 bars are null
target = close["position"]    # includes uninitialized values
```

✅ **Proper initialization**:

```python
ma = close.rolling_mean(50)
target = (close - ma).sign().fill_null(0.0)  # fills nulls with 0
```

## Debugging

To debug a strategy, inspect intermediate signals:

```python
def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
    close = bars["close"]
    ma = close.rolling_mean(self.ma_period)
    signal = (close - ma).sign()

    # Print diagnostics
    print(f"Close range: {close.min():.2f} - {close.max():.2f}")
    print(f"MA range: {ma.min():.2f} - {ma.max():.2f}")
    print(f"Signal value counts:\n{signal.value_counts()}")

    return signal.fill_null(0.0).rename("position")
```

## Testing

Add unit tests in `python/tests/test_strategies.py`:

```python
def test_my_mean_reversion():
    from snippy_scales.strategies.my_strategy import MyMeanReversion

    strat = MyMeanReversion(ma_period=5)

    bars = pl.DataFrame({
        "close": [100, 101, 99, 102, 98, 100, 101],
    })

    signals = strat.generate_signals(bars)

    assert signals.len() == bars.height
    assert signals.dtype == pl.Float64
    assert signals.is_null().sum() == 0
```

Run tests with:

```bash
just test-py
```
