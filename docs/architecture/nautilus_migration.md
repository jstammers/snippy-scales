# Migration Plan: NautilusTrader

**Status:** proposed (revision 2) · **Date:** 2026-10-14 · **Target:** `nautilus_trader==2.0.0rc6`
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

### Phase 0: Foundations

- `data/catalog/` added to `.gitignore` (**done**). Add `data/derived/experiments/` usage for
  Optuna artifacts (already ignored via `data/derived/`).
- Delete `rust/`, switch the build backend from maturin to `hatchling`/`uv_build`, and remove
  `just dev`, the Rust CI jobs and `docs/architecture/pyo3_bridge.md`.
- Add `nautilus_trader==2.0.0rc6` and `optuna` as dependencies. Remove `raptorbt` and `quantstats`
  in phases 2 and 3.
- `configs/instruments.yaml`: per symbol, `venue`, `price_increment`, `multiplier`, `lot_size`,
  commission per contract, IB contract fields.
- Catalog contract tests (`tests/test_catalog_contract.py`) pinning the verified facts: on-disk
  schema, overlap raises, `ts_init` = bar close. They fail loudly if an upgrade changes behaviour.

### Phase 1: Catalog is the datastore; Nautilus downloads

**Sync job** (`algo data sync configs/databento.yaml [--dry-run]`):

- For each instrument in the config: `last = catalog.query_last_timestamp(...)`. Fetch from
  `last` with `DatabentoHistoricalClient.get_range_bars(..., timestamp_on_close=True)` (bars) or
  `get_range_trades/quotes` in **one-day chunks** (ticks). Write, then `consolidate_data`.
- Instruments via `get_range_instruments` (Databento `definition`), written before data.
- `--dry-run` prints the Databento cost from `databento.Historical().metadata.get_cost(...)`
  (~20 lines; kept deliberately as a guard against expensive typos).
- Days with no data (holidays) are cheap empty requests. No empty-marker bookkeeping.
- A one-off multi-GB tick backfill uses the Databento **portal** batch download plus
  `DatabentoDataLoader` → catalog. No Batch API code in the repo.

**`load_bars(symbol, schema, start=None, end=None) -> pl.DataFrame`**: same signature and columns
as today, reading via `query_data_arrow_bytes(..., display=True)`, so research and strategies
don't change.

**Migrating existing `data/raw/`** (one-off script in `scripts/`, not in the package; removed
after use):

- **Read-only on `data/raw/`; never deletes anything.** Writes only to `data/catalog/`, and
  refuses a non-empty target unless resuming. Deleting `data/raw/` stays a manual decision after
  verification and a backup, per CLAUDE.md.
- Tick data: `_dbn/*.dbn.zst` → `DatabentoDataLoader` → catalog (lossless; no re-purchase).
- Databento bars: run `sync --dry-run` first. If re-downloading is cheap (daily bars for 20
  futures usually are), re-download through Nautilus so precision and timestamps come from the
  official path. Otherwise convert from Parquet: quantise to tick, `ts_init` = open + interval.
- Alpaca minute bars: convert from Parquet (free data, but hours to re-download).
- Verify: per (symbol, schema), row count, first/last timestamp and checksum of
  `round(close / tick)` against `data/raw/`.

**Then delete:** `providers/databento.py`, `batch.py`, `ingest.py`, `schema.py`, `tick.py`,
`backfill.py` (unless Alpaca stays), CLI `ingest`, `ingest-config`, `coverage`, `check-layout`,
`migrate-layout`.

**Exit:** verification clean on your real data, and `algo eval run-all` gives the same results
on catalog data as on `data/raw/`.

### Phase 2: `SignalStrategy` and `run_backtest`; delete `backtesting/`

- `execution/strategy.py`, `instruments.py`, `fees.py`, `backtest.py` as above. Backtests use
  `BacktestNode` / `BacktestDataConfig` reading `data/catalog/` directly.
- **Causality test** for every strategy (hypothesis property test):
  `generate_signals(bars[:k]) == generate_signals(bars)[:k]` for random `k`. This makes
  precomputation safe and catches lookahead in research code generally.
- **One-time parity check** against `TargetPositionEngine` for the four built-in strategies, on
  a synthetic instrument with fine lot size, multiplier 1 and zero costs. Then add costs and
  check they are in proportion. Record the result in this doc.
- Then delete all of `backtesting/` and `raptorbt`, and update CLAUDE.md and
  `docs/architecture/backtest_engine.md`.

**Exit:** parity recorded; no imports from `backtesting/` remain.

### Phase 3: Walk-forward on Optuna; delete the rest of `evaluation/`

- `evaluation/walk_forward.py`: fold dates plus an Optuna objective. Each trial runs its folds
  through `run_backtest`, stores per-fold stats as trial user attributes, and saves its OOS
  returns to `data/derived/experiments/<study>/<trial>.parquet` (needed for PBO).
- DSR/PSR/PBO computed from the study: trial count, the trial Sharpe distribution and the OOS
  returns matrix.
- Reports: `create_tearsheet_from_stats` on stitched OOS returns.
- Parallelism: Optuna with one `BacktestEngine` per process (`n_jobs` via processes, or several
  workers sharing the study storage).
- Delete `sweep.py`, `results.py`, `database.py`, `tearsheet.py` and `quantstats`. Rewrite
  `cli/eval.py` around studies.

**Exit:** `algo eval run-all` runs on Optuna + Nautilus, and DSR/PBO are reported per study.

### Phase 4: Real futures contracts and rolls

Today every futures series is Databento's unadjusted continuous front month (`ES.c.0`). It can't
be traded, and its roll gaps distort returns.

- Sync parent symbology (`ES.FUT`) bars and definitions into the catalog.
- Build back-adjusted series for signals with Nautilus continuous-futures support (you supply
  the roll transitions). `SignalStrategy` trades the active outright and rolls on schedule, with
  roll costs in the backtest.

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
