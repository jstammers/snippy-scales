# Migration Plan: NautilusTrader

**Status:** in progress (revision 2) — phases 0–4 done (phase 4 not yet run on real outright data); next is phase 5 (paper trading on IB) · **Date:** 2026-10-14 · **Target:** `nautilus_trader==2.0.0rc6`
(pinned exactly), moving to `2.0.x` stable when released

## Goal

Use NautilusTrader for **data storage, data download, backtest execution, analytics and live
trading**. Custom code must justify itself: it stays only where neither Nautilus nor an
established library does the job.

What survives that test is small:

| Kept | Why nothing replaces it |
|---|---|
| `research/`, `strategies/` | The actual research. |
| `SignalStrategy` adapter (~120 lines) | Bridges vectorised `generate_signals()` to Nautilus orders. |
| Catalog sync job + `load_bars()` (~200 lines) | Nautilus provides the download client and catalog, but no "keep my catalog up to date from a config" command. |
| Walk-forward fold generation + Optuna objective (~200 lines) | Nautilus has no walk-forward or parameter search. |
| `evaluation/statistics.py` (DSR, PSR, PBO, MinTRL) and `scoring.py` (CRPS, PIT) | No Nautilus or maintained-library equivalent. |
| `data/universe.py` | Survivorship-free S&P 500 membership; not market data. |

Everything else in `backtesting/`, `evaluation/`, `data/` and `rust/` is replaced or deleted. That
is roughly 8,000 of today's ~10,600 lines in those areas (details below).

---

## Corrections to revision 1

| Revision 1 said | Correction |
|---|---|
| Keep our Databento provider, Batch API client and DBN archive | **Delete them.** Nautilus's `DatabentoHistoricalClient.get_range_bars/trades/quotes/instruments` downloads straight to Nautilus objects for the catalog. `get_range_bars(..., timestamp_on_close=True)` also fixes the bar-open timestamp lookahead problem I had planned a custom converter for. The catalog stores data once, so Batch API's free re-download advantage disappears, and day-chunked requests stay small. |
| Keep `TargetPositionEngine` as a fast pre-screen for sweeps | **Delete after a one-time parity check.** Keeping two engines means maintaining parity forever. Measured daily-bar runs are fast enough (below). |
| Add `NautilusExecutionEngine` behind the existing `ExecutionEngine` Protocol so runners don't change | **Delete the Protocol and runners instead.** An abstraction over one engine is indirection. Call Nautilus directly. |
| Keep the 33-field `BacktestMetrics`, fed by `from_returns()` | **Delete.** Nautilus computes about 35 statistics. Custom ones register via `Portfolio.register_statistic()`. |
| Keep `AnalyticsStore` / `BacktestStore` (DuckDB) | **Replace with Optuna storage.** It records trials, params, values and user attributes, and comes with a dashboard. |
| Keep `tearsheet.py` / quantstats for walk-forward | **Delete.** `create_tearsheet_from_stats(returns=...)` renders stitched out-of-sample returns. |
| (not covered) | **`rust/` is an unused ~380-line stub** of a planned event-driven engine; only a `TODO` in `cli/backtest.py` references it. Delete it, and with it the maturin build, Rust toolchain and `just dev`. |
| Alpaca: no adapter (you said Nautilus supports Alpaca) | **Confirmed: no official adapter** in 2.0.0rc6 or on `develop` (`crates/adapters/`). [RFC #3374](https://github.com/nautechsystems/nautilus_trader/issues/3374) is open; in Jan 2026 the maintainer said "not in the near term". The community [`nautilus-alpaca`](https://github.com/gmatrunich/nautilus-alpaca) is alpha (v0.1.0, 0 stars) and pins `nautilus_trader>=1.226` (the v1 API). See decision D1. |

---

## Verified facts

All checked empirically on 2026-10-14 in a scratch venv (Linux x86_64, Python 3.12) with
synthetic data, unless marked *docs*.

### Packaging

- `2.0.0rc6`: `Requires-Python >=3.12,<3.15`, **no required dependencies** (pandas/plotly only
  under the `visualization` extra). A cp312 wheel installs and imports cleanly. Install needs
  pre-releases enabled (`uv --prerelease=allow`) until 2.0.0.
- v2 import paths are flat (`nautilus_trader.model.Bar`, `nautilus_trader.persistence.ParquetDataCatalog`).
  The v1 paths don't exist. License: LGPL-3.0-or-later.

### Catalog

| | v1 `1.231.0` | v2 `2.0.0rc6` |
|---|---|---|
| Bar directory | `data/bar/<bar_type>/` | `data/bars/<bar_type>/` |
| Price / volume columns | `fixed_size_binary[16]` (opaque) | `decimal128(38, 16)` (native in polars/DuckDB) |
| Timestamps | `uint64` ns | `timestamp[ns, UTC]` |

- One file per write, named `<start>_<end>.parquet`. **Overlapping writes raise** (`OSError` in v2);
  there is no upsert. Incremental sync writes only past `query_last_timestamp(...)`.
- `get_missing_intervals_for_request` counts the gaps *between* files as missing, so it can't
  drive incremental sync. Use `query_last_timestamp`. Run `consolidate_data` after appends.
- Casting float → decimal doesn't snap to the tick (`100.05` → `100.0499999999999972`). This
  only matters for our own converter (Alpaca and the one-off migration); the Databento client
  sets precision from instrument definitions.

| 1,000,000 one-minute bars, v2 | Time |
|---|---|
| Arrow batch → `Bar` (`BarDataWrangler.process_record_batch_bytes`) | 2.1 s |
| `write_bars` | 0.8 s |
| `query_data_arrow_bytes(NautilusDataType.Bar, display=True)` → float64 Arrow | 0.8 s |
| `pl.scan_parquet(...)` + cast decimal → float64 | 0.3 s |

### Backtest engine

Benchmark: 20 futures × 3,800 daily bars (15 years, 76,000 bars), target-position rebalancing
on every bar, about 11,000 market orders, `BacktestEngine` reused with `reset()`.

| Strategy implementation | Time per run |
|---|---|
| Replay only (strategy does nothing) | 0.25 s |
| **Signals precomputed** with polars, looked up in `on_bar` | **3.9–4.7 s** |
| Signals computed incrementally in `on_bar` (numpy ring buffer) | 7.5–7.9 s |
| Signals computed by calling polars `generate_signals()` on a 300-bar window every bar | 101–104 s |

Conclusions:

1. Replay costs about 3 µs per bar. Order handling dominates, at about 0.3 ms per order.
   Rebalance bands (trade only when the position is off target by more than a threshold) cut
   orders and cost in both backtest and live.
2. **Calling vectorised research code bar by bar is 25× slower.** The adapter precomputes signals
   once per run and looks them up. This is only valid if `generate_signals` is causal, which is
   enforced by a test (phase 2).
3. A 100-trial × 5-fold daily-bar walk-forward is about 500 runs of about 1 s each (fold-sized
   windows): roughly 8 minutes single-process, 1–2 minutes across 8 processes. **No vectorised
   engine is needed for daily or hourly strategies.**
4. The exception is intraday cross-sections. 500 symbols × 5 years of 1-minute bars ≈ 250M bars
   is about 13 minutes of replay per run before any orders, so wide sweeps there aren't
   feasible. If and when such a strategy exists, add [vectorbt](https://github.com/polakowo/vectorbt)
   (open source, supports `size_type="targetpercent"`) as a pre-screen, not a handwritten engine.

Also verified:
- `add_venue` **requires** an explicit `fee_model`.
- A Python `FeeModel` subclass works: a per-contract commission model was called once per fill.
  That covers `FuturesCostModel`'s commission part in about 10 lines.
- `DefaultFillModel(prob_slippage=...)` handles slippage. *Docs:* with bar data, slippage is
  one tick.
- `BacktestResult` exposes `returns_series` (timestamp → return), `stats_returns`, `stats_pnls`
  and `stats_general`.
- Built-in statistics include Sharpe, Sortino, Calmar, Omega, MaxDrawdown, CAGR, ProfitFactor,
  WinRate, Expectancy, VaR, ExpectedShortfall, TailRatio, UlcerIndex, Alpha/Beta, information
  ratio and capture ratios. Python `PortfolioStatistic` subclasses register via
  `Portfolio.register_statistic()`.

### Integrations

- Databento (*docs* + API surface): data only. `DatabentoHistoricalClient` has
  `get_range_bars(dataset, instrument_ids, aggregation, start, end, price_precision, timestamp_on_close=True)`
  plus `get_range_trades/quotes/instruments/order_book_deltas`. `DatabentoDataLoader` loads DBN files.
- Interactive Brokers (*docs*): futures and equities, data and execution. Continuous futures are
  historical-only.
- *Docs:* an order submitted in `on_bar` fills at that bar's close, matching our close-to-close
  convention. `ts_init` must be the bar close.

---

## Critical review of the backtesting code

The question for each component: *what would we lose by deleting it?*

| Component (lines) | Case for keeping it | Verdict |
|---|---|---|
| `backtesting/engine.py` (343): `ExecutionEngine` Protocol, `InstrumentSpec`, `RaptorExecutionEngine`, `make_config`, `run_single` | Engine swappability | **Delete.** One engine. Swappability is speculative. raptorbt has known metric defects and drops position magnitude. |
| `backtesting/continuous.py` (479): `TargetPositionEngine` | Fast vectorised sweeps | **Delete after the parity check.** Daily-bar Nautilus runs are about 1–4 s. Reintroduce a vectorised pre-screen (vectorbt) only when a measured sweep needs it. |
| `backtesting/costs.py` (220) | Per-contract commission + half-spread | **Delete.** A ~10-line `FeeModel` subclass plus `DefaultFillModel`. |
| `backtesting/allocation.py` (85), `signals.py` (138) | — | **Delete.** Workarounds for raptorbt's static weights and boolean signals. |
| `backtesting/domain.py` (544): `Trade`, `BacktestMetrics` (33 fields), `BacktestResult` | Stable metric schema | **Delete.** Use Nautilus stats dicts plus `returns_series`. A fixed 33-column schema forces a 7-step checklist for every new metric (CLAUDE.md). Store stats in long format (`stat`, `value`) instead. |
| `backtesting/store.py` (262), `evaluation/database.py` (319) | Queryable experiment history | **Delete.** Optuna storage + `optuna-dashboard`; `study.trials_dataframe()` → polars/DuckDB for ad-hoc SQL. |
| `backtesting/runner(s).py` (417), `data.py` (147) | Orchestration | **Delete.** Replaced by `run_backtest()` (~80 lines). |
| `evaluation/sweep.py` (232) `ParameterGrid`, `RandomSearch`, `OptunaSearch` | Search abstraction | **Delete.** Use Optuna directly (`GridSampler`, `RandomSampler`, `TPESampler`). |
| `evaluation/results.py` (334) `FoldResult`, `SweepResult` (mean/std) | Aggregation | **Delete.** Per-fold metrics go in trial user attributes; aggregate with polars. |
| `evaluation/split.py` (240) `WalkForwardSplit` | Fold boundaries | **Shrink to ~40 lines** producing `(train_start, train_end, test_start, test_end)` dates. Nautilus runs take a time range directly. |
| `evaluation/runner.py` (505) `EvaluationRunner` | Walk-forward orchestration | **Shrink to ~150 lines**: an Optuna objective that runs folds via `run_backtest()`. |
| `evaluation/tearsheet.py` (238) + `quantstats` | Reports | **Delete.** `create_tearsheet` (single run) and `create_tearsheet_from_stats(returns=stitched OOS)`. |
| `evaluation/statistics.py` (329), `scoring.py` (205) | DSR/PBO/CRPS | **Keep.** These consume `returns_series` and per-trial results. |
| `rust/` (381) + maturin build | Planned event-driven engine | **Delete.** Nautilus *is* the event-driven engine. The build becomes pure Python (`hatchling` or `uv_build`). |

**The one new adapter: `SignalStrategy`.** A Nautilus `Strategy` that:

1. On start, loads the bar history for its instruments and calls the wrapped
   `Strategy.generate_signals` / `MultiAssetStrategy.generate_signals` once.
2. In `on_bar`, looks up the target position (units of equity) and converts it to contracts
   (`target × equity / (price × multiplier)`, rounded to lot size).
3. Submits the delta when it is outside the rebalance band.
4. When a bar arrives beyond the precomputed range (live), recomputes on the trailing window.
   Live cost is negligible at daily or hourly frequency.

This leaves one code path, not separate "precomputed" and "online" modes.

---

## Target layout

```
python/snippy_scales/
├── research/            unchanged
├── strategies/          unchanged (pure polars, no Nautilus imports)
├── data/
│   ├── catalog.py       load_bars(), sync()            ← new, ~200 lines
│   ├── universe.py      unchanged
│   └── config.py        slimmed (sync configs + instrument specs)
├── execution/           ← new; the only package importing nautilus_trader
│   ├── strategy.py      SignalStrategy                 ~120
│   ├── instruments.py   symbol ↔ InstrumentId ↔ IB contract, from configs/instruments.yaml   ~60
│   ├── fees.py          PerContractFeeModel            ~15
│   ├── backtest.py      run_backtest(spec, start, end) -> BacktestResult   ~80
│   └── live.py          LiveNode config (phase 5)
├── evaluation/
│   ├── walk_forward.py  folds + Optuna objective       ~200
│   ├── statistics.py    unchanged
│   └── scoring.py       unchanged
└── cli/                 data sync, eval, backtest, live
```

Line budget (excluding `research/`, `strategies/`, tests):

| Area | Today | After |
|---|---|---|
| `backtesting/` → `execution/` | 2,659 | ~300 |
| `evaluation/` | 2,485 | ~750 |
| `data/` | 3,392 | ~500 (~800 if Alpaca code stays, D1) |
| `cli/` | 1,712 | ~400 |
| `rust/` | 381 | 0 |
| **Total** | **~10,600** | **~2,000–2,300** |

The module contract in CLAUDE.md becomes: `strategies/` and `research/` never import Nautilus;
`execution/` is the only package that does; `evaluation/` calls `execution.run_backtest` and
never touches orders.

---

## Phases

Each phase ends with a green `just check-py`. Phase 0's Rust removal and phase 1 can start
immediately and in parallel.

### Phase 0: Foundations — **done**

- `data/catalog/` added to `.gitignore`, and the CLAUDE.md "never delete `data/`" rule covers it.
- `rust/` deleted. The build backend is `hatchling` (pure Python wheel). `just dev`, the Rust
  CI/release/docs steps, the Rust prek hooks and `docs/architecture/pyo3_bridge.md` are gone.
- `nautilus_trader==2.0.0rc6` added as a core dependency. `raptorbt` and `quantstats` are
  removed in phases 2 and 3; `optuna` becomes core in phase 3.
- Catalog contract tests (`python/tests/test_nautilus_contract.py`) pin the verified facts:
  on-disk schema, overlap raises, disjoint appends, `query_last_timestamp` = last `ts_init`,
  float64 display queries, `BarDataWrangler` decimal input, instrument round trip.

`configs/instruments.yaml` moved to phase 1, where it is first used.

### Phase 1: Catalog is the datastore; Nautilus downloads — **done**

What landed:

- `data/catalog.py`: naming (`bar_type`, `instrument_id`), `load_bars` (same columns as
  before plus `ts_init`; `start`/`end` filter on the bar close), `frame_to_bars` (exact
  decimal prices, `ts_init` = open + interval), incremental `write_bars` (writes only before
  the first / after the last stored bar), `missing_ranges`, instrument helpers including
  `continuous_instrument` (synthetic `ES.c.0.GLBX` cloned from the latest outright).
- `data/databento.py`: `DatabentoSync` on Nautilus's `DatabentoHistoricalClient`
  (`get_range_bars(..., timestamp_on_close=True)`, `get_range_trades/quotes/...` one day per
  request, `get_range_instruments`). Cost via `databento.Historical().metadata.get_cost` for
  exactly the missing ranges. The client is built lazily, so `--dry-run` planning needs no key.
- `data/alpaca.py`: the single Alpaca loader (fetch with `alpaca-py`, shared rate limiter,
  concurrent fetches with serialised catalog writes, `Equity` instruments at venue `ALPACA`,
  4-decimal prices because adjusted history is off the tick grid).
- CLI: `algo data sync | coverage | consolidate | update-universe | backfill-sp500`;
  `sync --instruments-only` fetches definitions without data. `algo eval run-all` and
  `algo research roughness` read the catalog.
- `scripts/migrate_raw_to_catalog.py`: read-only on `data/raw/`; bars with inferred precision,
  ticks rebuilt from the DBN archive, per-series verification (row count + close checksum).
  It is idempotent, so it doesn't refuse a non-empty catalog.
- Deleted: `providers/`, `batch.py`, `ingest.py`, `schema.py`, `tick.py`, `ratelimit.py` (moved
  into `alpaca.py`), CLI `ingest`, `ingest-config`, `check-layout`, `migrate-layout`, and their
  tests.

Deviations from the plan:

- No `configs/instruments.yaml`: futures definitions come from Databento (`definition`
  schema), equities from `equity_instrument`. Per-contract commissions move to phase 2.
- Equity venue is `ALPACA` (where the data came from), not an IB venue; mapping to the
  execution venue belongs in phase 5's `instruments.py`.
- `ohlcv-eod` is no longer supported (no distinct Nautilus bar type); use `1d`.
- Tick migration supports `trades` from DBN files only; days without DBN are reported.

**Not yet verified end to end:** the Databento path was tested against a fake async client
with the real method signatures. No API key or network access to `hist.databento.com` was
available, so the first real `algo data sync configs/databento.yaml --dry-run` and a small
sync should be checked by hand. In particular, check that continuous symbols resolve in
`get_range_bars` and `get_range_instruments`.

**Exit (for you to run on real data):** `migrate_raw_to_catalog.py --verify-only` clean;
`algo eval run-all` gives the same results as before the migration.

### Phase 2: `SignalStrategy` and `run_backtest` — **done**

What landed:

- `execution/strategy.py`: `SignalStrategy` (a Nautilus `Strategy`) and `compute_targets`.
  Targets are precomputed once on the whole history and looked up by bar close. Bars closing
  at the same timestamp are batched, so every instrument is sized off one equity figure after
  all of them are marked:
  `contracts = target × weight × equity / (close × multiplier)`, rounded to the size increment.
  Weights default to `1 / n_symbols`, the old `BasketRunner` convention. `rebalance_band` skips
  trades below that fraction of equity. Bars beyond the history (live) are appended and the
  targets recomputed once per batch, so there is one code path. Denied and rejected orders
  are collected in `BacktestRun.rejections`.
- `execution/backtest.py`: `run_backtest(research, data, start, end, ...)` on `MarketData`,
  loaded from the catalog by `load_market_data` (bars replayed exactly as stored) or built
  from polars frames by `MarketData.from_frames` (synthetic data). *(Phase 3 split loading
  from running so folds reuse one load.)* Signals
  use all history up to `end`, so they are warm at `start`. `Costs` maps to Nautilus models:
  `commission_per_contract` → `PerContractFeeModel`, `commission_rate` → `MakerTakerFeeModel`,
  `slippage_prob` → `DefaultFillModel` (seeded). The venue is a `MARGIN`/`NETTING` account.
  `BacktestRun` holds the flat Nautilus stats dict, `returns_series`, per-bar equity, and the
  fills and positions reports.
- `tests/test_strategy_causality.py`: hypothesis property
  `generate_signals(bars[:k]) == generate_signals(bars)[:k]` for all four built-in strategies
  (TSMOM with and without `rebal_freq`, CSMOM with two configurations), plus a check that a
  peeking strategy fails it. All built-in strategies pass.
- `tests/test_execution_backtest.py`: sizing, fill timing, warm-up, rebalance band, multiplier,
  costs, catalog round trip, the live recompute path, and the parity check below (the
  parity test was deleted with the legacy engine in phase 3; its result is recorded here).
- Real-data smoke run (Alpaca daily, read-only): TrendFollowing on AAPL 2018–2026 took
  0.68 s for 695 orders; CSMOM on 8 equities took 0.70 s. No orders were rejected.

**Parity with `TargetPositionEngine` (recorded 2026-11-13).** Synthetic futures with
multiplier 1, 400 daily bars, `1e9` capital (whole-contract rounding is negligible), unlimited
bar volume, every-bar rebalancing. "Max rel. diff" compares the two equity curves bar by bar
with zero costs. "Cost drag" is the fall in final equity from adding a 10 bp commission.

| Strategy | Assets | Max rel. diff (zero cost) | Return | Legacy cost drag | Nautilus cost drag | Ratio |
|---|---|---|---|---|---|---|
| TrendFollowing(10, 40) | 1 | 9.3e-09 | +1.75% | 2.21% | 2.26% | 1.02 |
| TimeSeriesMomentum(60) | 1 | 5.1e-09 | +3.07% | 3.75% | 3.79% | 1.01 |
| MeanReversion(20, 1.0, 0.2) | 1 | 2.4e-09 | −5.58% | 9.44% | 9.48% | 1.01 |
| CrossSectionalMomentum(60, 5) | 4 | 1.1e-08 | +8.14% | 0.99% | 1.06% | 1.07 |

With zero costs the engines agree to rounding. With costs, Nautilus pays slightly more because
it also trades to re-size held positions as prices drift; the legacy engine treats a constant
target as free. That is the more realistic number. The test pins both properties: rtol 1e-7
with zero costs, and Nautilus drag between 1× and 2× the legacy drag.

Findings:

- **Bar volume is liquidity.** With bar execution, a market order larger than the bar's volume
  walks the simulated book and fills partly one tick worse. That was the only source of parity
  error before the test bars were given ample volume. It is realistic for thin bars, but means
  results depend on position size relative to volume.
- The Nautilus `Strategy` base is a PyO3 class whose `__new__` takes only `config`, so
  subclasses with their own constructor arguments must override `__new__`.
- `returns_series` is mark-to-market daily equity returns (not realised P&L only), so Nautilus
  Sharpe/Sortino are usable directly.

Deviations from the plan:

- **`backtesting/` is not deleted yet.** `evaluation/` (runner, results, database, tearsheet)
  and `cli/eval.py` are built on `BacktestRunner`, `BacktestMetrics` and `BacktestStore`.
  Replacing those *is* phase 3, so deleting `backtesting/` and `raptorbt` moves there. Nothing
  new imports `backtesting/`; only the parity test uses it, and it goes with it.
- No `execution/fees.py`: Nautilus 2.0 ships `PerContractFeeModel` and `MakerTakerFeeModel`.
- No `execution/instruments.py`: instruments come from the catalog (phase 1). The mapping to IB
  contracts belongs in phase 5.
- `BacktestEngine` with in-memory bars, not `BacktestNode`/`BacktestDataConfig`: signals need
  the polars history anyway, and phase 3 needs an engine it can reuse across trials.
- All instruments in one run must share one venue and currency.

**Exit:** parity recorded. "No imports from `backtesting/` remain" moves to phase 3's exit.

### Phase 3: Walk-forward on Optuna; delete `backtesting/` and the rest of `evaluation/` — **done**

What landed:

- `evaluation/walk_forward.py` (~500 lines with docs):
  - `make_folds` produces contiguous test windows at the end of the history (old
    `test_size` semantics, capped to leave `min_train` warm-up bars).
  - `WalkForward` holds the data, folds and backtest settings.
  - `run_study` is the Optuna study: grid, random or TPE samplers, and a search space of
    lists, ranges or fixed values. Each trial runs every fold through `run_backtest` and
    stores per-fold stats as user attributes. Its stitched OOS returns go to
    `data/derived/experiments/<study>/trial_<n>.parquet`. Study config (strategy, symbols,
    folds, costs) is saved as study attributes.
  - `summarize_study` computes, for the best trial, the stitched OOS Sharpe, return, max
    drawdown, PSR, **DSR** (N = completed trials, trial std from their stitched Sharpes),
    MinTRL and **PBO** (CSCV over all trials' aligned OOS returns). It also gives Nautilus
    return statistics via `PortfolioAnalyzer`.
  - `write_tearsheet` calls `create_tearsheet_from_stats` with an optional benchmark.
- Storage (D3): Optuna SQLite at `data/derived/experiments/optuna.db`; several processes can
  share a study to run in parallel.
- CLI: `algo eval run | sweep | run-all | list | report | tearsheet` rewritten around
  studies; `algo backtest run` now works (it was a stub). Multi-asset strategies are
  supported by `run` and `sweep`, which previously refused them.
- Deleted: all of `backtesting/` (11 modules), `evaluation/{runner,results,sweep,split,
  database,tearsheet}.py`, `raptorbt`, `quantstats`, the parity test, and the tests of
  the deleted code. Strategy integration tests now run through `run_backtest`
  (`tests/conftest.py`). Net: about 5,150 lines removed and 460 added in the package,
  before new tests. `optuna` and `nautilus_trader[visualization]` (plotly) are now core
  dependencies.
- Real-data smoke run (Alpaca daily, read-only, in-memory storage): a 9-trial × 5-fold
  TrendFollowing grid on AAPL took 2.0 s (best `fast=10, slow=200`: OOS Sharpe 0.68,
  DSR 0.94, PBO 0.29). A 4-trial CSMOM grid on 8 equities took 3.4 s (OOS Sharpe 0.12,
  DSR 0.55, PBO 0.76).

Deviations from the plan:

- **No train-window backtests, and no `window`/`gap` options.** None of the strategies fit
  anything, so a fold's "training" is only signal warm-up, which `run_backtest` gets from
  all earlier history. Rolling and expanding windows would give identical results.
  Parameters are chosen on the OOS folds, as the old runner did; DSR and PBO are the guard.
  A fitted model will need a per-fold fit step (fit on the train window, then backtest the
  test window) added to `WalkForward.run_folds`.
- Each fold starts flat with fresh capital, so the first bar of each test window only
  opens the position. Stitched OOS returns therefore have one day fewer per fold than
  test bars.
- Benchmarks come from a catalog symbol (`--benchmark NQ.c.0`), not a quantstats/Yahoo
  download.
- No process-pool parallelism inside `run_study`. Run several `algo eval sweep --study X`
  processes against the shared storage instead. Daily-bar studies are fast enough that
  this hasn't been needed.
- Old experiment databases (`metadata.db`, `analytics.duckdb`) are not migrated. Their
  raptorbt-era Sharpe values are not comparable (`docs/research/raptorbt_defects.md`).

**Exit:** met. `algo eval run-all` runs on Optuna + Nautilus, DSR/PBO are reported per
study, and nothing imports `backtesting/` (it no longer exists).

### Phase 4: Real futures contracts and rolls — **done** (not yet run on real outright data)

What landed:

- **Parent syncs.** `DatabentoSync` handles `ES.FUT` (parent symbology).
  - Definitions come first: outrights only, and on re-runs only those after the newest
    stored one.
  - Bars are stored per outright (`ESH5.GLBX-1-DAY-…`). Spreads and contracts without a
    definition are dropped.
  - Coverage is the union of the root's outright series, so re-runs fetch only the head
    or tail (`catalog.coverage_gaps`).
  - Parent symbols always use `stype_in=parent`.
  - `configs/databento_futures.yaml` is an example config.
- **`data/futures.py`.**
  - `outrights` lists a root's contracts from the catalog.
  - `RollRule`: volume (default) or calendar N days before expiry, with an optional
    `months` cycle.
  - `roll_schedule` produces a `RollSchedule`: the active instrument per bar close.
  - `back_adjust` splices the active contracts with **backward ratio** adjustment.
  - `tradable_symbols` collapses outright series into `<ROOT>.FUT` for discovery.
- **Execution.**
  - `MarketData` is now keyed by instrument, with a `RollSchedule` per symbol. It's a
    single instrument for anything but a parent.
  - `load_market_data("ES.FUT", roll=RollRule(...))` loads the outrights and builds the
    schedule and adjusted signal frame. `MarketData.from_outrights` does the same for
    synthetic data.
  - `SignalStrategy` sizes off the active contract. On a roll's close it closes the old
    contract and opens the new one (both fills pay costs; the rebalance band doesn't
    apply to the re-entry). `BacktestRun.rolls` lists the rolls.
- **CLI.** `--roll volume|calendar`, `--roll-days` and `--roll-months` on `algo backtest
  run` and `algo eval run | sweep | run-all`. `run-all` now:
  - evaluates one `<ROOT>.FUT` per root instead of each contract;
  - runs multi-asset strategies per venue (ALPACA equities and GLBX futures can't share
    one account);
  - skips symbols it can't load (e.g. no instrument definition) instead of aborting.
- **Tests** (`test_futures_rolls.py`, plus parent cases in `test_data_databento.py` and
  `test_cli_eval.py`): volume and calendar roll dates; the months filter; that adjusted
  returns equal held-contract returns (to 1e-10); that signals are unchanged by later
  rolls; that a constant target on rolled outrights tracks the adjusted series to 3e-5
  with zero costs; and that each roll bar fills both contracts and pays commission on both.

Deviations from the plan:

- **The splice is ours, not Nautilus's continuous futures.** Nautilus splices inside
  the engine at subscription time (`continuous_future_transitions` params). That needs
  internally aggregated bar types and produces no polars history before the run, which
  precomputed signals require. The splice itself is about 40 lines and is checked against
  the held-contract returns.
- **Ratio adjustment only.** With backward *spread* adjustment, every new roll shifts all
  earlier prices by a constant. That changes their percentage returns, so signals
  precomputed before a roll would differ from those computed after it, breaking the
  causality guarantee. Ratio adjustment rescales them instead, which return-based signals
  ignore. Strategies on rolled symbols must be scale-invariant; all four built-in ones are.
- **Live rolls are not handled yet.** A live session can extend a schedule's last
  contract, but it can't roll to a contract that wasn't in the loaded data (phase 5).

**Not yet verified end to end:**
- Like phase 1, the parent path was tested against a fake client with the real method
  signatures. In particular, it assumes `DatabentoHistoricalClient.get_range_bars` with
  `ES.FUT.GLBX` returns bars keyed per child instrument; check that on the first real
  `--dry-run` and sync.
- The catalog currently holds 19 continuous `*.c.0` daily series with no instrument
  definitions. They can't be backtested until `algo data sync configs/databento.yaml
  --instruments-only` is run.

**Benchmarks (2026-11-13, Apple Silicon, single process).** Same data for every engine.

| Workload | Old raptorbt | Old `TargetPositionEngine` | Nautilus (now) |
|---|---|---|---|
| 19 futures × 4,630 daily bars, TSMOM | 0.03 s | 0.04 s | 9.7 s (58,681 orders); 4.1 s with a 0.2% band (12,210 orders) |
| One AAPL run, 2,690 bars | 0.9 ms | 1.1 ms | 257 ms (2,612 orders) |
| AAPL 9-trial × 5-fold grid | 0.04 s (90 runs incl. train) | 0.05 s | 1.67 s (45 runs, 37 ms each) |

Against the plan's revision-2 measurements (Linux x86, synthetic data): order handling is
0.13 ms/order versus 0.3 ms. Bar handling is about 25 µs/bar versus 3 µs: the plan
measured a strategy that did nothing, whereas `SignalStrategy` batches, marks and sizes on
every bar. The plan's estimate that a 100-trial × 5-fold daily study takes "about 8 minutes"
holds: about 20 s on one equity, and about 7–16 min on the 19-future portfolio (2 s per fold
run, or 0.8 s with a rebalance band). The vectorised engines were 100–300× faster per run,
but they couldn't model orders, contracts or rolls, and raptorbt's metrics were wrong.

### Phase 5: Paper trading on Interactive Brokers

`algo live run --config configs/live/paper.yaml`:

- `LiveNode` with the Databento live data client and IB execution (IB Gateway, optionally
  `DockerizedIBGateway`, paper port 4002). The same `SignalStrategy`, warmed from the catalog.
- Turn on cache persistence and `shutdown_on_error`, check reconciliation on restart, and alert
  on disconnect.
- Run for several weeks, reconciling against a same-period backtest.

### Phase 6: Live

Preconditions: Nautilus `2.0.x` stable, phase 5 reconciled, `LiveRiskEngineConfig` limits set
(max order size, max notional), a kill switch, and small initial size.

---

## Risks and decisions

| ID | Item | Default / mitigation |
|---|---|---|
| R1 | v2 is a release candidate | Exact pin, contract tests, deliberate upgrades, no real capital before stable. |
| R2 | Catalog format changes again | Keep `data/raw/` until stable. Nautilus ships `nautilus catalog migrate-parquet` for its own changes. |
| R3 | Intraday cross-sectional sweeps too slow in an event-driven engine | Measure when such a strategy exists; add vectorbt as a pre-screen then, not before. |
| R4 | Lookahead | `timestamp_on_close=True` on download; `ts_init` = close in converters; causality test on every strategy. |
| R5 | Continuous-contract backtests overstate tradability | Phase 4 before live futures. |
| **D1** | **Alpaca equities data.** No official adapter; the community one targets v1 and is alpha. | (a) **Move equities data to Databento** (e.g. `EQUS.MINI` / `XNAS.ITCH`). This deletes all Alpaca code (`providers/alpaca.py`, `ratelimit.py`, `backfill.py`): one provider, one symbology, live data through the same adapter. It costs Databento credit. (b) Keep `AlpacaProvider` + rate limiter (~300 lines) writing to the catalog via `BarDataWrangler`. Free, but custom code. Recommended: (a) if the budget allows, else (b). Not recommended: the community adapter. |
| D2 | Execution venue for equities | Interactive Brokers (stable adapter). |
| D3 | Experiment store | Optuna storage (SQLite file under `data/`, or the journal file backend). |
| D4 | Daily-bar close time | `timestamp_on_close=True` from Databento; open + 1 day for converted legacy bars. |

---

## Appendix: reading the v2 catalog outside Nautilus

```python
import polars as pl

bars = (
    pl.scan_parquet("data/catalog/data/bars/ES.c.0.GLBX-1-DAY-LAST-EXTERNAL/*.parquet")
    .with_columns(pl.col("open", "high", "low", "close", "volume").cast(pl.Float64))
    .sort("ts_event")
    .collect()
)
```

```sql
SELECT ts_event, CAST(close AS DOUBLE) AS close
FROM read_parquet('data/catalog/data/bars/ES.c.0.GLBX-1-DAY-LAST-EXTERNAL/*.parquet')
ORDER BY ts_event;
```

Library code goes through `snippy_scales.data.catalog.load_bars`, so a layout change is fixed in
one place.

## References

- NautilusTrader: [README](https://github.com/nautechsystems/nautilus_trader) ·
  [PyPI](https://pypi.org/project/nautilus-trader/) ·
  [Alpaca RFC #3374](https://github.com/nautechsystems/nautilus_trader/issues/3374)
- Docs (`develop`): [catalog](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/data/catalog.md) ·
  [Databento](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/integrations/databento.md) ·
  [Interactive Brokers](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/integrations/interactive_brokers.md) ·
  [backtest APIs](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/backtesting/apis-and-runs.md) ·
  [bar execution](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/backtesting/bar-execution.md) ·
  [continuous futures](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/continuous_futures.md) ·
  [live](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/live.md) ·
  [reports](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/reports.md)
- Community Alpaca adapter: [gmatrunich/nautilus-alpaca](https://github.com/gmatrunich/nautilus-alpaca)
- Databento: [streaming vs batch download](https://databento.com/blog/streaming-vs-batch-download-historical-market-data)
