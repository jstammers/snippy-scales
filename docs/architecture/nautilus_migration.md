# Migration Plan: NautilusTrader

**Status:** proposed · **Date:** 2026-10-14 · **Target:** `nautilus_trader==2.0.0rc6` (pinned exactly), moving to `2.0.x` stable when released

## Goal

Make NautilusTrader the platform for **data storage, backtest execution and live trading**,
and keep only the code that is genuinely ours: research, signal generation, walk-forward
evaluation and overfitting statistics.

Concretely:

1. Replace the custom Parquet datastore (`data/raw/<type>/<symbol>/<schema>.parquet`) with a
   Nautilus `ParquetDataCatalog`. One copy of the data, used by research, backtests and live.
2. Replace `RaptorExecutionEngine`, `TargetPositionEngine` and their support code with the
   Nautilus `BacktestEngine`.
3. Run the same strategy code on an Interactive Brokers paper account, then live.

Non-goals: replacing the research layer, the walk-forward / sweep framework, DSR/PBO, or
CRPS/PIT scoring. Nautilus has no equivalent for any of these.

---

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Nautilus version | **v2 (`2.0.0rc6`, exact pin)**, not v1 `1.231.0` | The v1 and v2 catalog formats differ on disk (see facts below), and v2 renames every import path. Starting on v1 means migrating the datastore twice. v2 also has native Arrow read/write paths that v1 lacks. Risk: release candidate. Mitigated by the exact pin, a catalog round-trip test suite, and not trading real capital until `2.0.x` is stable. |
| Datastore | Nautilus catalog at `data/catalog/` is the **single** store for market data | Your requirement. Backtests and live read it directly. |
| Research read path | `load_bars()` keeps its signature and return shape but reads the catalog | Strategies and `research/` don't change. |
| Ingestion | **Keep our providers** (Databento + Batch API + cost estimates, Alpaca + rate limiter); replace only the storage layer | Nautilus's Databento client has no Batch API or cost estimation, and there is no Nautilus Alpaca adapter. |
| Raw DBN archive | **Keep** `_dbn/*.dbn.zst` as the lossless source of truth for tick data | The catalog can be rebuilt from DBN without paying Databento again. |
| Execution venue | Interactive Brokers (stable adapter: futures + equities) | No Alpaca adapter exists. Databento is data-only. |
| Experiment store | Keep `AnalyticsStore` (DuckDB) and the 33-metric schema | Nautilus has no experiment tracking. Metrics come from `BacktestMetrics.from_returns()` on the Nautilus equity curve, so the schema survives the engine swap. |

---

## Verified facts

These were checked empirically on 2026-10-14 in a scratch venv (Linux x86_64, Python 3.12 / 3.13)
with synthetic data. Re-verify on every version bump.

**Packaging**
- `2.0.0rc6` declares `Requires-Python >=3.12,<3.15` with **no required dependencies**
  (pandas/plotly are optional under the `visualization` extra). A cp312 wheel installs and
  imports cleanly, so it is compatible with this repo's `>=3.12,<3.14` and CI's 3.12.
- Import paths are flat in v2: `nautilus_trader.model.Bar`,
  `nautilus_trader.persistence.ParquetDataCatalog`. The v1 paths
  (`nautilus_trader.persistence.catalog`, `nautilus_trader.model.data`) don't exist in v2.
- License: LGPL-3.0-or-later.

**Catalog on-disk format**

| | v1 `1.231.0` | v2 `2.0.0rc6` |
|---|---|---|
| Bar directory | `data/bar/<bar_type>/` | `data/bars/<bar_type>/` |
| Price / volume columns | `fixed_size_binary[16]` (opaque to polars/duckdb) | `decimal128(38, 16)` (native in polars/duckdb) |
| Timestamps | `uint64` ns | `timestamp[ns, UTC]` |
| Instruments | `data/<instrument_class>/<id>/` | `data/equity/<id>/`, `data/futures_contract/<id>/`, … |
| File name | `<start>_<end>.parquet` (ISO, `:`/`.` → `-`) | same |

**Write semantics (both versions)**
- One file per write call, named by its timestamp range.
- An overlapping write **raises** (`ValueError` in v1, `OSError` in v2). There is no upsert or
  dedup on write. Incremental ingestion must write only the tail past
  `query_last_timestamp(...)`, or `delete_data_range(...)` and then rewrite.
- `get_missing_intervals_for_request` works on file `ts_init` ranges. It reports the gaps
  *between* consecutive files as missing (for example, a 1-minute gap at each file boundary),
  so it **cannot** replace our bar high-water-mark logic. Use `query_last_timestamp` for bars.
- Many incremental appends leave many small files. Run `consolidate_data(...)` periodically.

**Throughput** (1,000,000 one-minute bars, single symbol, v2)

| Operation | Time |
|---|---|
| Arrow record batch → `Bar` list via `BarDataWrangler.process_record_batch_bytes` | 2.1 s |
| `write_bars` | 0.8 s |
| `query_bars` → `Bar` objects | 1.2–2.6 s |
| `query_data_arrow_bytes(NautilusDataType.Bar, display=True)` → float64 Arrow | 0.8 s |
| `pl.scan_parquet(<bar dir>/*.parquet)` + cast decimal → float64 | 0.3 s |

Extrapolated one-off migration of the S&P 500 1-minute set (~500 symbols × 5 years ≈ 250M bars):
roughly 10–15 minutes single-threaded, plus I/O. Parallelise by symbol.

**Gotchas found**
- The v2 `BarDataWrangler.process_record_batch_bytes` requires price/volume columns as
  `decimal128(38, 16)`. Float64 input is rejected.
- Casting a float to decimal does **not** snap to the instrument's tick. `100.05` was stored as
  `100.0499999999999972`. The converter must quantise to `price_increment` first.
- The v1 `BarDataWrangler.process` fails under pandas 3 (`buffer source array is read-only`).
  This is moot on v2, but another reason not to start on v1.
- pandas 3 `date_range` defaults to microsecond resolution, so `.asi8` is µs, not ns. Never
  derive nanosecond timestamps through pandas without `.as_unit("ns")`. The converter should
  work in polars/Arrow throughout.

**Docs-only facts (not probed)**
- Databento adapter: data only (no execution). Continuous `ES.c.0.GLBX`, parent `ES.FUT.GLBX`
  and raw `ESZ6.XCME` symbology. `DatabentoDataLoader.load_bars/trades/quotes/instruments`
  read DBN files directly.
- IB adapter: futures and equities, data and execution. Continuous futures (`CONTFUT`) are
  **historical only**, with no live subscription or orders.
- Bar execution: an order submitted in `on_bar` fills against that bar's close (no latency
  model). This matches our `execution_lag=1` close-to-close convention. `ts_init` **must** be
  the bar close, or the bar is visible before it formed.
- Live: one `LiveNode` per process. Startup reconciliation against the venue. Unbounded
  queues with no backpressure.

---

## Target architecture

```
                     ┌────────────────────────── ours ──────────────────────────┐
providers/  ──fetch──▶  data/catalog.py  ──write──▶  data/catalog/  (Nautilus ParquetDataCatalog)
(Databento, Alpaca)    (convert + upsert)                │
                                                         ├──▶ load_bars() ──▶ research/, strategies/  (polars, vectorised)
                                                         │                     │ generate_signals(bars) -> target positions
                                                         │                     ▼
                                                         ├──▶ nautilus/ adapter: SignalStrategy, instruments, roll manager
                                                         │        │
                                                         │        ├──▶ Nautilus BacktestEngine ──▶ returns ──▶ evaluation/ (walk-forward, DSR/PBO, AnalyticsStore)
                                                         │        └──▶ Nautilus LiveNode + IB execution, Databento live data
                     └───────────────────────────────────────────────────────────┘
```

**The contract between research and execution stays as it is today:**
`Strategy.generate_signals(bars: pl.DataFrame) -> pl.Series` of target positions in units of
equity, and `MultiAssetStrategy.generate_signals(dict[str, pl.DataFrame]) -> dict[str, pl.Series]`.
Nothing under `research/` or `strategies/` imports Nautilus.

New package `python/snippy_scales/nautilus/` (the only place Nautilus is imported, apart from
`data/catalog.py`):

| Module | Responsibility |
|---|---|
| `instruments.py` | Build `FuturesContract` / `Equity` from `configs/instruments.yaml` and Databento definitions; one symbology map (repo symbol ↔ catalog `InstrumentId` ↔ IB contract). |
| `strategy.py` | `SignalStrategy(Strategy)`: wraps any `Strategy`/`MultiAssetStrategy`. Two signal modes: **precomputed** (signals computed vectorised up front; fast backtests and sweeps) and **online** (calls `generate_signals` on a rolling window each bar; used live). Converts target position → order quantity (equity, price, multiplier, lot size) and submits the delta. |
| `engine.py` | `NautilusExecutionEngine`, which implements the existing `ExecutionEngine` Protocol, so `BacktestRunner`, `BasketRunner` and `evaluation/` don't change. Returns `BacktestResult` via `BacktestMetrics.from_returns`. |
| `rolls.py` | Phase 4: front-contract selection and roll schedule for trading real futures contracts. |
| `live.py` | Phase 5: `LiveNode` config (Databento data client, IB execution client), entry point `algo live run`. |

---

## Module disposition

**Replace** means the code is deleted once its replacement passes the exit criteria of its phase.
Each deletion is preceded by one release with a `DeprecationWarning`.

### `data/`

| Module | Fate | Phase | Notes |
|---|---|---|---|
| `providers/` (`databento.py`, `alpaca.py`, `base.py`) | **Keep** | — | Still fetch to polars. `BarProvider` unchanged. |
| `ratelimit.py`, `batch.py` | **Keep** | — | Alpaca rate limiting and Databento Batch API have no Nautilus equivalent. |
| `universe.py`, `backfill.py` (manifest) | **Keep** | — | Membership lists and per-symbol job state aren't market data. |
| `config.py` | **Keep, extend** | 1 | Add `venue` and instrument-spec fields. |
| `ingest.py` storage half (`_symbol_path`, `upsert_bars` write path, `_effective_start*`, `_earliest_covered*`, `is_range_cached`, `load_bars`, `find_cross_class_duplicates`) | **Replace** with `data/catalog.py` | 1 | Fetch orchestration and cost estimation stay. |
| `schema.py` (`BAR_SCHEMA_COLUMNS`, `conform_bars`) | **Replace** with the catalog converter | 1 | The Nautilus `Bar` schema becomes the shared schema. |
| `tick.py` Parquet day partitions | **Replace** with catalog `trades` / `quotes` / `order_book_deltas` | 1 | Keep `_dbn/` archive, `_empty/` markers, `missing_days` and cost estimation; DBN is loaded with `DatabentoDataLoader`. |
| `paths.py` | **Keep, extend** | 1 | Add `CATALOG_DIR = DATA_ROOT / "catalog"`. |
| CLI `check-layout`, `migrate-layout` | **Replace** with `migrate-to-catalog`, `catalog-verify` | 1 | |

### `backtesting/`

| Module | Fate | Phase | Notes |
|---|---|---|---|
| `engine.py` `ExecutionEngine` Protocol, `InstrumentSpec` | **Keep** | — | The seam that lets the engine be swapped. |
| `engine.py` `RaptorExecutionEngine`, `make_config`, `run_single` | **Replace** | 3 | raptorbt discards position magnitude and has known metric defects (`docs/research/raptorbt_defects.md`). |
| `continuous.py` `TargetPositionEngine` | **Keep until parity, then replace** | 2 → 3 | It is the reference for the parity test. Delete only once Nautilus matches it and sweep runtime is acceptable (see risk R3). |
| `costs.py` | **Replace** with Nautilus `fee_model` + fill model | 3 | Port `FuturesCostModel` parameters (commission per contract, half-spread in ticks) into venue config. |
| `allocation.py` `VolTargetAllocator` | **Delete** | 3 | It exists only because raptorbt takes static weights. Vol targeting already lives in the strategies. |
| `signals.py` `SignFlipInterpreter` | **Delete** | 3 | Boolean entry/exit was raptorbt's input format. |
| `domain.py` `BacktestMetrics.from_returns`, `BacktestResult` | **Keep** | — | Delete the `from_raptorbt` constructors in phase 3. |
| `store.py` `BacktestStore` | **Keep** | — | Fed by `from_returns`. |
| `data.py` (`normalize_ohlcv`, `to_numpy_ohlcv`) | **Shrink** | 3 | Keep only what `TargetPositionEngine` and research use. |
| `runners.py` | **Keep** | — | Default engine becomes `NautilusExecutionEngine`. |

### `evaluation/`, `research/`, `strategies/`

| Module | Fate | Notes |
|---|---|---|
| `split.py`, `sweep.py`, `runner.py`, `results.py` | **Keep** | Engine-agnostic through the Protocol. |
| `statistics.py` (DSR, PSR, PBO, MinTRL), `scoring.py` (CRPS, PIT) | **Keep** | No Nautilus equivalent. |
| `database.py` `AnalyticsStore` | **Keep** | |
| `tearsheet.py` | **Keep** for walk-forward (stitched OOS returns via quantstats); single runs may use Nautilus `create_tearsheet` | |
| `research/`, `strategies/` | **Keep unchanged** | |

Rough net effect: about 2,000–2,500 lines of storage and execution code removed
(`continuous.py`, raptorbt glue, `costs.py`, `allocation.py`, `signals.py`, storage half of
`ingest.py`, `schema.py`, tick Parquet writer). About 800–1,200 lines added in `data/catalog.py`
and `nautilus/`.

---

## Phases

Each phase ends with a green `just check-py` and its exit criteria met. Phases 1 and 2 are
independent and can run in parallel.

### Phase 0: Foundations

- Add `nautilus_trader==2.0.0rc6` as an optional extra (`[project.optional-dependencies] nautilus`),
  promoted to a core dependency in phase 3. Install needs `--prerelease=allow` (uv) until 2.0.0.
- Add `data/catalog/` to `.gitignore` (**done in this change**). Extend the CLAUDE.md
  "never delete `data/`" rule to cover it explicitly.
- Add `configs/instruments.yaml`: per root symbol, `venue`, `asset_class`, `currency`,
  `price_increment`, `multiplier`, `lot_size`, `ib` contract fields. Seed it from Databento
  `definition` records (cheap) for the futures in `configs/databento.yaml`.
- Write the catalog contract tests (`python/tests/test_catalog_contract.py`) asserting the
  facts above: on-disk schema, overlap raising, decimal precision, `ts_init` = bar close. These
  fail loudly when a Nautilus upgrade changes behaviour.

**Exit:** extra installs in CI; contract tests pass.

### Phase 1: Catalog becomes the datastore

**1a. Converter and catalog API (`data/catalog.py`)**

```python
def write_bars(catalog, df: pl.DataFrame, *, symbol: str, schema: str) -> int
def load_bars(symbol: str, schema: str, *, start=None, end=None) -> pl.DataFrame   # same shape as today
def load_ticks(symbol: str, schema: str, start, end) -> pl.DataFrame
def last_timestamp(symbol: str, schema: str) -> pd.Timestamp | None
```

- Symbol → `InstrumentId`. Futures: `ES.c.0` → `ES.c.0.GLBX` (Databento convention).
  Equities: `AAPL` → `AAPL.<venue>` from config. Schema → `BarType`: `ohlcv-1d` →
  `…-1-DAY-LAST-EXTERNAL`, `ohlcv-1h` → `1-HOUR`, `ohlcv-1m` → `1-MINUTE`, `ohlcv-1s` → `1-SECOND`.
- **Timestamps: Databento and Alpaca stamp bars at the interval *open*.** Set `ts_event` = open
  and `ts_init` = open + interval. For daily bars, use the session close from the instrument's
  calendar if available, else open + 1 day (conservative). This is the main lookahead trap of
  the migration; test it explicitly.
- Quantise OHLC to `price_increment` in polars before casting to `decimal128(38, 16)`, then
  `BarDataWrangler.process_record_batch_bytes` → `catalog.write_bars`.
- Write instruments before data.
- `load_bars` reads via `query_data_arrow_bytes(NautilusDataType.Bar, …, display=True)` (supported
  API, float64 output) and renames to today's columns (`ts_event`, `open` … `volume`, `symbol`).
  A direct `pl.scan_parquet` path (about 3× faster) may be added behind the same function,
  guarded by the contract tests.
- Columns that disappear: Databento `rtype`, `publisher_id`, `instrument_id` (these live in the
  instrument definition) and Alpaca `vwap`, `trade_count`. **Decision needed (D2):** drop them, or
  store them as a registered custom data type. Grep shows nothing outside `data/` reads them.

**1b. Upsert on top of the catalog**

`upsert_bars` keeps its public behaviour (download only the missing tail):
`last_timestamp()` → provider fetch from there → drop rows `<= last` → `write_bars`. For a
backfill *before* existing coverage, `delete_data_range` over the affected span, then rewrite.
Run `consolidate_data` at the end of `ingest-config` runs.

**1c. Ticks**

`upsert_ticks` keeps `_dbn/` and `_empty/` exactly as today and replaces the
`date=…/data.parquet` writer with `DatabentoDataLoader.load_trades(path, instrument_id)` →
`catalog.write_trade_ticks`. Coverage stays driven by our day files and markers; the catalog
can't represent "confirmed empty day".

**1d. One-off migration of existing data: `algo data migrate-to-catalog`**

Safety rules (from CLAUDE.md, non-negotiable):

- **Read-only on `data/raw/`.** The command never modifies or deletes anything under it.
- It writes only to `data/catalog/` (or `--catalog PATH`). If the target exists and is
  non-empty, it refuses unless `--resume`, and it never wipes a catalog.
- `data/raw/` stays in place after the cutover. Deleting it is a manual decision for you, after
  a verification period and a backup. No code path in this plan deletes it.

Procedure:

1. Walk `data/raw/<instrument_type>/<symbol>/`. For each bar file and tick partition, convert
   and write. Run in parallel by symbol; record progress in `data/catalog/_migration.csv` so it
   can resume.
2. Ticks: rebuild from `_dbn/*.dbn.zst` where present (lossless). Fall back to the Parquet
   partition only for days without DBN.
3. `algo data catalog-verify`: for every (symbol, schema), compare row count, first and last
   `ts_event`, and a checksum of `round(close / tick)` between raw and catalog. Print a table;
   exit non-zero on any mismatch.

**1e. Cutover**

`load_bars` reads the catalog by default. `SNIPPY_DATASTORE=legacy` falls back to `data/raw/`
for one release. Ingestion writes only to the catalog. Then remove the legacy readers and
writers, `schema.py` and the layout CLI.

**Exit:** `catalog-verify` is clean on your real data. The full test suite and `algo eval run-all`
produce identical metrics reading from the catalog as from `data/raw/`.

### Phase 2: Nautilus backtest adapter and parity

- `nautilus/instruments.py`, `nautilus/strategy.py` (`SignalStrategy`, precomputed mode first),
  `nautilus/engine.py` (`NautilusExecutionEngine`).
- Backtests use `BacktestNode` + `BacktestDataConfig` pointing at `data/catalog/` (no data
  copying). Sweeps share one `BacktestEngine` and call `reset()` between parameter sets.
- **Parity test** (`test_nautilus_parity.py`): for `TrendFollowing`, `TimeSeriesMomentum`,
  `MeanReversion` and `CrossSectionalMomentum` on fixed data, the Nautilus equity curve must match
  `TargetPositionEngine` within tolerance. To isolate engine differences from contract rounding,
  run parity on a synthetic instrument with fine `size_precision` and multiplier 1, and zero
  costs. Then add costs and compare against `ProportionalCost`.
- Lookahead test: shifting `ts_init` to the bar open must *improve* backtest results (proves the
  test can detect leakage), mirroring the existing `execution_lag=0` test.
- Benchmark: wall time of one walk-forward sweep (`algo eval sweep`) on both engines.

**Exit:** parity within tolerance for all four strategies, and a documented sweep runtime ratio.

### Phase 3: Switch evaluation to Nautilus; delete raptorbt

- `NautilusExecutionEngine` becomes the default in `BacktestRunner` / `BasketRunner` and the CLI.
- Port `FuturesCostModel` to the venue `fee_model`. Pick a fill model and document it.
- Remove `raptorbt` from dependencies. Delete `RaptorExecutionEngine`, `make_config`,
  `run_single`, `from_raptorbt` constructors, `allocation.py`, `signals.py` and
  `docs/research/raptorbt_defects.md` (or archive it).
- Decide the fate of `TargetPositionEngine` from the phase 2 benchmark (R3).
- Update CLAUDE.md (key patterns, module contract, "running a backtest") and
  `docs/architecture/backtest_engine.md`.

**Exit:** no raptorbt imports; `algo eval run-all` runs on Nautilus; docs updated.

### Phase 4: Realistic futures

Today every futures series is a Databento continuous contract (`ES.c.0`), which is unadjusted
front-month prices. It can't be traded, and its roll gaps distort returns.

- Ingest parent symbology (`ES.FUT`, `stype_in: parent`) plus `definition` records into the
  catalog, so every outright contract and its expiry exists.
- `nautilus/rolls.py`: roll schedule (for example N days before expiry or on open-interest
  crossover) that produces the transition list Nautilus's continuous-futures builder expects
  (back-adjusted series for *signals*) and the active contract for *orders*.
- `SignalStrategy` trades the active outright and rolls positions on schedule, with roll costs
  included in the backtest.

**Exit:** a backtest that trades real contracts with rolls, reconciled against the
continuous-contract result (differences explained by roll costs and adjustment).

### Phase 5: Paper trading on Interactive Brokers

- `nautilus/live.py` + `algo live run --config configs/live/paper.yaml`: `LiveNode` with the
  Databento live data client and IB execution client (IB Gateway, optionally via
  `DockerizedIBGateway`, paper port 4002).
- `SignalStrategy` in **online** mode: on startup, request history from the catalog / Databento
  to warm the rolling window, then call `generate_signals` per bar.
- Online/precomputed parity: replay a recorded day through online mode and assert identical
  target positions to precomputed mode on the same bars.
- Ops: cache persistence, `shutdown_on_error`, reconciliation on restart, alerting on
  disconnect, daily report export (`ReportProvider` fills/positions → DuckDB).

**Exit:** several weeks of paper trading with fills and P&L reconciled against a same-period
backtest.

### Phase 6: Live

Preconditions: Nautilus `2.0.x` stable (not an RC), phase 5 reconciliation within tolerance,
risk limits configured (`LiveRiskEngineConfig`: max order size, max notional, rate limits), a
kill switch, and small initial size.

---

## Risks and open decisions

| ID | Risk / decision | Mitigation / default |
|---|---|---|
| R1 | v2 is a release candidate; breaking changes between RCs | Exact pin; catalog contract tests; upgrade deliberately; no real capital before stable. |
| R2 | The catalog format changes again in a later release | Keep `data/raw/` and `_dbn/` until stable; `migrate-to-catalog` stays re-runnable; Nautilus ships `nautilus catalog migrate-parquet` for its own format changes. |
| R3 | An event-driven engine is much slower than vectorised `TargetPositionEngine` for large sweeps (especially the 1-minute S&P 500 cross-section) | Measure in phase 2. If too slow, keep `TargetPositionEngine` as a **pre-screen** for wide sweeps, with Nautilus as the engine of record for final candidates and all reported results. |
| R4 | Lookahead from bar timestamp semantics (`ts_event` = open) | Converter sets `ts_init` = close; dedicated test (phase 1) plus the leakage test (phase 2). |
| R5 | Continuous-contract backtests overstate tradability | Phase 4 before any live futures trading. |
| D1 | Equity venue IDs: Alpaca SIP bars are consolidated, but IB routes per exchange | Default: catalog IDs use the IB venue (`AAPL.NASDAQ` style) so backtest and live IDs match; map in `instruments.yaml`. |
| D2 | Keep Alpaca `vwap` / `trade_count` and Databento metadata columns? | Default: drop (nothing reads them). Alternative: custom data type. |
| D3 | Equities execution: IB vs a custom Alpaca adapter | Default: IB. A custom adapter is a large project; revisit only if Alpaca is a hard requirement. |
| D4 | Daily-bar close timestamp: session close vs open + 1 day | Default: open + 1 day for phase 1 (safe); session calendars in phase 4. |

---

## Appendix: reading the v2 catalog from polars / DuckDB

The v2 on-disk format is plain Arrow, so ad-hoc analytics need no Nautilus import:

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
-- DuckDB
SELECT ts_event, CAST(close AS DOUBLE) AS close
FROM read_parquet('data/catalog/data/bars/ES.c.0.GLBX-1-DAY-LAST-EXTERNAL/*.parquet')
ORDER BY ts_event;
```

Library code should go through `snippy_scales.data.catalog.load_bars`, so that a layout change is
fixed in one place.

## References

- NautilusTrader: [README](https://github.com/nautechsystems/nautilus_trader) ·
  [PyPI](https://pypi.org/project/nautilus-trader/)
- Docs (`develop` branch): [catalog](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/data/catalog.md) ·
  [Databento](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/integrations/databento.md) ·
  [Interactive Brokers](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/integrations/interactive_brokers.md) ·
  [backtest APIs and runs](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/backtesting/apis-and-runs.md) ·
  [bar execution](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/backtesting/bar-execution.md) ·
  [continuous futures](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/continuous_futures.md) ·
  [live](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/live.md) ·
  [reports](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/reports.md)
