# Backtesting and Evaluation

Backtests run on NautilusTrader's event-driven `BacktestEngine`. Our code is a thin
adapter (`execution/`) plus walk-forward studies on Optuna (`evaluation/`). The old
vectorised raptorbt layer (`backtesting/`), its 33-field `BacktestMetrics` and the DuckDB
stores were removed in [migration](nautilus_migration.md) phase 3.

## Module contract

| Module | Owns | Must not contain |
|---|---|---|
| `strategies/`, `research/` | `generate_signals(bars) -> target positions` (pure polars) | Any `nautilus_trader` import |
| `execution/` | `SignalStrategy`, `MarketData`, `run_backtest` | Folds, sweeps, aggregation |
| `evaluation/` | Folds, Optuna studies, DSR/PSR/PBO, tearsheets | Orders or engine details (it calls `run_backtest`) |

## From signals to fills

```
generate_signals(history)          once per run, on all bars up to `end`
        │  targets in units of equity, keyed by bar close (ts_init)
        ▼
SignalStrategy.on_bar              batches the bars closing at one timestamp
        │  equity read after all of them are marked
        │  contracts = target × weight × equity / (close × multiplier)
        ▼
market order for the delta         fills at that bar's close; earns the next bar
```

- **Units.** `1.0` is fully invested long, `-0.5` half short. Multi-symbol runs weight
  each symbol `1 / n` by default (`weights=` overrides).
- **Causality.** Precomputing signals is only valid if they never look ahead.
  `tests/test_strategy_causality.py` checks `generate_signals(bars[:k]) ==
  generate_signals(bars)[:k]` for every built-in strategy.
- **Warm-up.** `run_backtest(start=...)` computes signals on earlier bars too, so the
  first traded bar already has warm indicators.
- **Live path.** A bar past the precomputed history is appended and the targets are
  recomputed once per timestamp; backtest and live share one code path.
- **Rebalance band.** `rebalance_band=0.02` skips trades smaller than 2% of equity.
- **Liquidity.** With bar execution, an order larger than the bar's volume walks the
  simulated book and fills partly a tick worse.

## Costs

| `Costs` field | Nautilus model |
|---|---|
| `commission_per_contract` | `PerContractFeeModel` |
| `commission_rate` (fraction of notional) | `MakerTakerFeeModel` (taker rate) |
| `slippage_prob` | `DefaultFillModel` (one tick on bar data; seeded) |

The venue is a `MARGIN` account with `NETTING` positions and `leverage=10` by default.
Denied or rejected orders are collected in `BacktestRun.rejections`; a non-empty tuple
means the run did not trade as intended.

## Results

`BacktestRun` holds Nautilus' own outputs: `stats_returns` (Sharpe, Sortino, Calmar, max
drawdown, … on daily mark-to-market returns), `stats_pnls`, `stats_general`, `returns`
(daily), `equity` (per bar close), and the fills and positions reports. `stats` merges
them into one flat dict. New metrics are Nautilus `PortfolioStatistic`s, not new columns.

## Walk-forward studies

```python
from snippy_scales.evaluation import WalkForward, make_folds, run_study, summarize_study
from snippy_scales.execution import Costs, load_market_data
from snippy_scales.strategies import TrendFollowing

data = load_market_data("ES.c.0")
wf = WalkForward(data, make_folds(data.closes(), n_splits=5), costs=Costs(commission_rate=1e-4))
study = run_study(TrendFollowing, wf, space={"fast_period": [10, 20, 40]}, sampler="grid")
report = summarize_study(study)
```

- **Folds** (`make_folds`) are contiguous test windows at the end of the history,
  each `test_size` of the bars, after at least `min_train` warm-up bars.
- **Trials.** Each Optuna trial builds the strategy with one parameter set and runs
  `run_backtest` on every fold. Its value is the mean fold Sharpe. Per-fold stats are
  trial user attributes. The stitched out-of-sample daily returns go to
  `data/derived/experiments/<study>/trial_<n>.parquet`.
- **Search.** `sampler="grid" | "random" | "tpe"`. Space values are lists
  (categorical), `{"low", "high", "step"?, "log"?}` ranges, or fixed scalars.
- **Selection bias.** The built-in strategies fit nothing, so parameters are chosen on
  the out-of-sample folds themselves. `summarize_study` therefore reports, for the best
  trial's stitched returns: Sharpe, PSR, the **Deflated Sharpe Ratio** (benchmarked
  against the expected best of N skill-less trials), the minimum track record, and the
  **Probability of Backtest Overfitting** across all trials (CSCV).
- **Storage.** Studies are kept in `data/derived/experiments/optuna.db` (SQLite). Several
  processes can share one storage and study name to search in parallel;
  `optuna-dashboard sqlite:///data/derived/experiments/optuna.db` browses them.
- **Reports.** `write_tearsheet` renders Nautilus' `create_tearsheet_from_stats` on the
  stitched out-of-sample returns, optionally against a benchmark symbol.

CLI: `algo backtest run`, `algo eval run | sweep | run-all | list | report | tearsheet`.

## Performance

Measured on Alpaca daily bars (Apple M-series, single process): one AAPL backtest over
2018–2026 takes about 0.1–0.7 s. A 9-trial × 5-fold grid on AAPL takes 2.0 s. A 4-trial ×
5-fold cross-sectional momentum study on 8 equities takes 3.4 s. Order handling dominates,
so a rebalance band speeds runs up as well as cutting costs.
