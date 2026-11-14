# Backtesting and Evaluation

Backtests run on NautilusTrader's event-driven `BacktestEngine`. Our code is a thin
adapter (`execution/`) plus walk-forward studies on Optuna (`evaluation/`). The old
vectorised raptorbt layer (`backtesting/`), its 33-field `BacktestMetrics` and the DuckDB
stores were removed in [migration](nautilus_migration.md) phase 3.

## Module contract

| Module | Owns | Must not contain |
|---|---|---|
| `strategies/`, `research/` | `generate_signals(bars) -> target positions` (pure polars) | Any `nautilus_trader` import |
| `execution/` | `SignalStrategy`, `MarketData`, `run_backtest`, `vector.run_backtest` | Folds, sweeps, aggregation |
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

## Futures rolls

A symbol trades through a `RollSchedule` (`data/futures.py`) that names the active
instrument at each bar close. Equities and continuous series have one instrument.
A parent symbol (`ES.FUT`) rolls through real outrights:

```python
from snippy_scales.data.futures import RollRule
data = load_market_data("ES.FUT", roll=RollRule("volume"))   # or "calendar", days_before_expiry=5
run = run_backtest(TimeSeriesMomentum(), data, costs=Costs(commission_per_contract=2.25))
run.rolls   # (close ns, old contract, new contract) for each roll of a held position
```

- **Schedule.** With the volume rule, the strategy rolls on the first close a later
  contract trades more than the active one; it never rolls back. With the calendar
  rule, it rolls on the first close at least N days before expiry. `months="HMUZ"`
  restricts trading to a cycle. A roll at close `T` needs a bar in both contracts at `T`.
- **Signals** use `back_adjust`: before each roll at `T`, every bar is multiplied
  by `close_new(T) / close_old(T)`. Each return of the spliced series is then
  exactly the return of the contract held over that bar. A later roll rescales all
  earlier bars by one constant, so signals built from returns, volatility or
  moving-average crossovers are unchanged, and precomputing them stays causal.
  This is tested. Spread (additive) adjustment would not have this property, which
  is why ratio is the only mode.
- **Execution.** On a roll's close, `SignalStrategy` closes the old contract (that
  bar's close) and opens the new one, ignoring the rebalance band for that trade.
  Both fills pay costs.
- **Check.** With zero costs, a constant target on rolled outrights tracks
  `target × adjusted returns` to 1e-5, which is whole-contract rounding.
- Nautilus's own continuous futures (`continuous_future_transitions` subscription
  params) splice bars inside the engine at subscription time. They need internally
  aggregated bar types, and they aren't available as a polars history before the run,
  which precomputed signals need. That is why the splice is done here.

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

## Screening engine (`execution/vector.py`)

`vector.run_backtest` has `run_backtest`'s signature and returns the same `BacktestRun`.
It replays `SignalStrategy`'s decisions on a numba kernel:

- **Sizing.** All symbols are sized from one equity figure per close, rounded half-even
  to the size increment.
- **Fills** happen at the close.
- **Rolls** are triggered by the held contract and pay costs on both legs, ignoring the
  band.
- **Band.** Trades below the rebalance band are skipped.
- **Non-positive closes** are neither marked nor traded.

`returns` and `stats_returns` are built exactly as Nautilus builds them: daily post-fee
equity over **calendar** days, with the same 11 return statistics. Fold Sharpes are
therefore comparable across engines. `tests/test_vector_parity.py` checks equity,
returns, statistics, order counts and rolls against Nautilus. That covers leverage,
shorts, multipliers, both fee types, slippage, the band, windows and rolled outrights.

Choose it with `WalkForward(..., engine="vector")`, `execution.ENGINES["vector"]` or
`--engine vector` on `algo backtest run` and `algo eval run | sweep | run-all`. The
study records its engine as a user attribute.

| One run, warm | Nautilus | `vector` |
|---|---|---|
| AAPL TrendFollowing, 2,690 daily bars | ~187 ms | ~4 ms (kernel 0.03 ms) |

The remaining ~4 ms is target generation (`compute_targets`, ~0.9 ms), Nautilus's
statistics analyzer (~1.5 ms) and the fills frame. Making target generation
vectorisable is the next step (see the spike report).

What it does **not** model, so validate the trials you keep on Nautilus:

- **Liquidity.** Nautilus fills a thin bar's volume at worse prices; this engine fills
  everything at the close.
- **Slippage** is charged at its expected cost rather than drawn at random.
- **Margin.** There are no margin checks, so `leverage` is ignored and no rejections
  are reported.
- **Position statistics.** `stats_pnls`, `stats_general` and `positions` are empty.

Background and benchmarks: [vectorised engines spike](../research/vectorised_engines_spike.md).
