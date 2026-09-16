# Data Ingestion

This guide covers how to pull bar data into the local Parquet store used by the
backtesting and research layers. Two providers are supported, selected per
config via `provider:` (default `"databento"`):

- **[Databento](https://databento.com)** — futures/CME Globex, paid, both bar
  and event-level (tick) schemas.
- **[Alpaca](https://alpaca.markets)** — US equities, free (Basic) historical
  tier, bar schemas only. See [Alpaca (Free-Tier Stock Bars)](#alpaca-free-tier-stock-bars).

Both providers write into the exact same layout and column schema (see
[Storage Layout](#storage-layout)), so [`load_bars`](#loading-data-in-python)
works identically regardless of which one fetched a given file.

---

## Prerequisites

1. **Databento account** — sign up at <https://databento.com> and note your API key.
2. **Set the API key** as an environment variable (the Databento client reads it
   automatically):
   ```bash
   export DATABENTO_API_KEY="db-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```
3. **For Alpaca**, sign up at <https://alpaca.markets> and set:
   ```bash
   export ALPACA_API_KEY="..."
   export ALPACA_SECRET_KEY="..."
   ```
4. **Install project dependencies** (includes `databento`, `alpaca-py`, and `pyyaml`):
   ```bash
   uv sync
   ```

---

## Storage Layout

All raw data is written under `data/raw/`.  Each symbol gets its own
sub-directory named after the symbol, and each schema variant gets its own
Parquet file:

```
data/
  raw/
    ES.c.0/
      ohlcv-1d.parquet      ← daily bars
      ohlcv-1h.parquet      ← hourly bars (if ingested)
    ZN.c.0/
      ohlcv-1d.parquet
    CL.c.0/
      ohlcv-1d.parquet
```

Event-level (tick) schemas are far too large for one file per symbol, so they
are stored **day-partitioned** under a directory named after the schema:

```
data/
  raw/
    ES.c.0/
      ohlcv-1d.parquet             ← bars: one file per schema
      trades/
        _dbn/2026-08-03.dbn.zst    ← lossless raw feed as returned by the API
        _empty/2026-08-02.empty    ← day confirmed to contain no records
        date=2026-08-03/
          data.parquet             ← hive-partitioned columnar
      mbo/
        date=2026-08-24/
          data.parquet
```

The raw `.dbn.zst` is kept alongside the Parquet so the columnar form can be
regenerated (different price type, different columns) without paying Databento
a second time.

!!! note "Gitignore"
    `data/raw/` is gitignored — never commit raw market data to version control.

---

## Upsert Semantics

The ingestion pipeline uses **upsert** (update + insert) semantics to avoid
re-downloading data you already have:

1. If a Parquet file for a symbol/schema pair already exists, the pipeline reads
   the latest `ts_event` timestamp stored locally.
2. Only data from **the next calendar day onwards** is requested from Databento.
3. The new rows are merged with the existing file, deduplicated on `ts_event`,
   sorted chronologically, and the file is overwritten.
4. If the file already covers the full requested date range, the download is
   skipped entirely.

This means each incremental run costs at most **one day of API credit** per
symbol, regardless of how much historical data is already on disk.

Note that this is a *high-water mark*: it only ever extends the tail.  Lowering
`start` will not backfill earlier history, and a hole in the middle of the range
is not detected.

### Tick coverage semantics

Tick schemas use a stronger rule.  A day is **covered** when either its Parquet
partition or its empty-day marker exists, and `missing_days()` is simply the
requested calendar range minus that covered set.  Consequently:

- Interior gaps *are* detected and filled.
- Backfilling below an existing start works.
- Re-running an identical command downloads nothing and costs nothing.

Each day is fetched in its own request and installed with an atomic rename, so
an interrupted run leaves no partial file that would be mistaken for a complete
day — just re-issue the same command to resume.

---

## Cost Estimation

Every download command queries the Databento metadata API first and asks you to
confirm the estimated spend.  Pass `--yes` to skip the prompt (or `--dry-run`
for `ingest-config`, which shows the plan and exits).  Symbols and days already
stored are excluded from the estimate, so a fully cached request reports zero
cost and makes no API call at all.

This matters most for `mbo`, which can be orders of magnitude larger than
`trades` for the same date range — read the billable size before confirming.

---

## Tick (Event-Level) Ingestion

`algo data ingest` routes automatically to the event-level (tick) store when
`--schema` names one of `trades`, `mbo`, `mbp-1`, `mbp-10`, or `tbbo` — there
is no separate command for ticks vs. bars.

!!! warning "The date range is half-open"
    `--start 2026-08-01 --end 2026-09-01` fetches all of August.  The end date
    is **exclusive**, unlike the bar commands.

```bash
# One month of ES trades
algo data ingest GLBX.MDP3 \
  --symbol ES.c.0 \
  --schema trades \
  --start 2026-08-01 \
  --end 2026-09-01

# One week of ES market-by-order
algo data ingest GLBX.MDP3 -s ES.c.0 --schema mbo --start 2026-08-24 --end 2026-08-31
```

Inspect what is stored, and whether anything is missing:

```bash
algo data coverage --symbol ES.c.0 --schema trades
```

Load it back in Python.  `load_ticks` returns a **`LazyFrame`** — an MBO store
is routinely larger than memory, so filter and aggregate before collecting:

```python
import polars as pl
from snippy_scales.data.tick import load_ticks

daily_volume = (
    load_ticks("ES.c.0", "trades", start="2026-08-01", end="2026-09-01")
    .group_by("date")
    .agg(pl.col("size").sum())
    .collect()
)
```

---

## Config-Based Batch Ingestion (Recommended)

The primary workflow is to define your universe in a YAML config file and run a
single CLI command.

### 1. Edit the config

The starter config lives at `configs/databento.yaml`:

```yaml
dataset: "GLBX.MDP3"
schemas: ["1d"]
start: "2018-01-01"

asset_classes:
  equity_index:
    symbols: [ES.c.0, NQ.c.0, YM.c.0, RTY.c.0]
  rates:
    symbols: [ZT.c.0, ZF.c.0, ZN.c.0, ZB.c.0]
  fx:
    symbols: [6E.c.0, 6J.c.0, 6B.c.0, 6A.c.0]
  commodities:
    symbols: [CL.c.0, NG.c.0, GC.c.0, SI.c.0, ZC.c.0, ZW.c.0, ZS.c.0]
```

| Field | Description |
|---|---|
| `provider` | `"databento"` (default) or `"alpaca"`. Alpaca configs may not list event-level schemas. |
| `dataset` | Databento dataset code. Use `GLBX.MDP3` for CME Globex. Ignored for Alpaca. |
| `schemas` | List of bar aliases and/or event-level schema names to ingest (see [Frequency Reference](#frequency-reference)). Each is ingested for every symbol below. |
| `start` | Earliest date to fetch (`YYYY-MM-DD`). |
| `end` | Latest date to fetch. Omit to default to today. |
| `stype_in` | Databento symbology type applied to every symbol in this config (e.g. `continuous`, `parent`). Ignored for Alpaca. |
| `alpaca` | Alpaca-specific options (`feed`, `adjustment`, `rate_limit_per_min`, `max_workers`) — see [Alpaca (Free-Tier Stock Bars)](#alpaca-free-tier-stock-bars). Only meaningful when `provider: alpaca`. |
| `asset_classes` | Mapping of human-readable labels to `symbols` (and/or `symbols_file` — a text file, one symbol per line, useful for large universes) — explicit identifiers per provider (Databento continuous/raw/parent notation, or plain Alpaca tickers; see [Symbol Format](#symbol-format)). |

### 2. Run the ingestion

```bash
# Full universe, every configured schema
algo data ingest-config configs/databento.yaml

# Restrict to one schema at the command line
algo data ingest-config configs/databento.yaml --schema 1h

# Preview what would be fetched without downloading anything
algo data ingest-config configs/databento.yaml --dry-run
```

The command prints a table of the ingestion plan — one row per
`(asset class, schema, symbol)` — then runs the upsert for each row in
sequence, logging progress and any failures.

### 3. Incremental refresh

Re-run the same command any time you want to extend the dataset to the current
date:

```bash
algo data ingest-config configs/databento.yaml
```

Only data more recent than what is already stored will be downloaded.

---

## Single-Symbol Ingestion

For ad-hoc or one-off pulls:

```bash
algo data ingest GLBX.MDP3 \
  --symbol ES.c.0 \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --schema 1d
```

This also uses upsert semantics — if `data/raw/ES.c.0/ohlcv-1d.parquet` already
exists, only the missing tail is fetched. The same command routes to the tick
store instead when `--schema` names an event-level schema (see
[Tick (Event-Level) Ingestion](#tick-event-level-ingestion)).

Add `--stype-in` when the symbol is not a raw contract code.  Continuous
notation such as `ES.c.0` requires `--stype-in continuous`:

```bash
algo data ingest GLBX.MDP3 -s ES.c.0 --start 2020-01-01 --end 2024-12-31 \
  --stype-in continuous
```

Databento's **parent** symbology (`--stype-in parent`) resolves a futures
root directly on their side — a symbol like `ES.FUT` returns every
individual outright contract active in the date range in a single request,
with no orchestration needed on our end:

```bash
algo data ingest GLBX.MDP3 -s ES.FUT --schema 1d --stype-in parent \
  --start 2020-01-01 --end 2024-12-31
```

The resulting file is written under `data/raw/ES.FUT/ohlcv-1d.parquet` and
distinguishes contracts via Databento's `instrument_id` column.

---

## Alpaca (Free-Tier Stock Bars)

Alpaca is a second bar provider, aimed at equities. Set `provider: alpaca` in
a config (`configs/alpaca.yaml` is a starting point) and run it exactly like a
Databento config:

```bash
algo data ingest-config configs/alpaca.yaml
algo data ingest-config configs/alpaca.yaml --dry-run
```

Since Alpaca's historical data has no per-request charge — only a rate limit —
`ingest-config` skips the Databento cost-estimation step entirely for an
Alpaca config: the plan table shows "free" instead of a dollar amount, and no
network call is made to build it.

Alpaca-specific behaviour, implemented in
`snippy_scales.data.providers.alpaca.AlpacaProvider`:

- **Rate limiting.** The `alpaca-py` SDK already paginates and retries on
  429/5xx, but only *reacts* to a 429 after it happens. A shared
  `snippy_scales.data.ratelimit.RateLimiter` sits in front of every HTTP call
  (including every page within one symbol's request) so a run stays under
  `alpaca.rate_limit_per_min` (default 190, free-tier cap is 200) in the
  first place. `alpaca.max_workers` controls how many symbols are fetched
  concurrently — throughput is bounded by the shared rate limiter, not by
  thread count, so raising it mainly reduces idle time between a symbol's
  pages.
- **Resume granularity.** Unlike Databento's day-based resume, Alpaca resumes
  from the exact timestamp after the last stored bar. Extended-hours bars can
  run past midnight UTC, so "the next calendar day" would silently skip the
  rest of a session.
- **Daily bars.** Normalised to `00:00 UTC` on the session date (Alpaca
  stamps them at session-open in US/Eastern) so they line up with Databento's
  convention, e.g. if a config or a query mixes both providers.
- **Adjustment.** Default `adjustment: all` (split + dividend) avoids fake
  price jumps at split dates across multi-year 1-minute history. Re-run with
  `--refresh` semantics if you need to re-baseline after a new split (not yet
  automatic — see the module docstring).
- **No event-level schemas.** `schemas: [trades]` etc. in an Alpaca config
  fails validation immediately; use a Databento config for tick data.

### S&P 500 1-minute backfill script

`scripts/pull_sp500_alpaca_1m.py` pulls N years (default 5) of 1-minute bars
for every ticker that was an S&P 500 constituent at *any point* in that
window — not just today's 500 — using
`snippy_scales.data.universe.sp500_ever_members` (sourced from Wikipedia) to
avoid survivorship bias:

```bash
# Preview: universe size, estimated request count and runtime — no download.
uv run python scripts/pull_sp500_alpaca_1m.py --dry-run

# Run it. Interruptible and resumable — re-running only fetches what's still
# missing (per-symbol upsert semantics), and a per-symbol CSV manifest
# (data/raw/_manifests/sp500_1m.csv by default) tracks pass/fail.
uv run python scripts/pull_sp500_alpaca_1m.py

# Retry only the symbols that failed last time.
uv run python scripts/pull_sp500_alpaca_1m.py --retry-failed
```

The resolved universe is cached to `data/universe/sp500_ever_members.txt` so
repeat runs (and `--retry-failed`) don't re-query Wikipedia and stay
reproducible; pass `--refresh-universe` to re-resolve it. Run
`--help` for every option (years, feed, adjustment, rate limit, worker count,
output/manifest/cache paths).

!!! note "Known limitation: pure ticker renames"
    A ticker rename with no index membership change (e.g. FB → META) doesn't
    appear as an addition/removal in Wikipedia's changes table, so history
    under the old ticker isn't picked up automatically. Check the script's
    manifest for symbols with zero rows if this matters for your research.

---

## Frequency Reference

The `--schema` flag (and `schemas` config field) accept the following bar
aliases, plus the event-level schema names `trades`, `mbo`, `mbp-1`,
`mbp-10`, and `tbbo`. Databento natively supports these bar schemas:

| Alias | Databento Schema | Notes |
|---|---|---|
| `1s`, `s` | `ohlcv-1s` | 1-second bars |
| `1m`, `1min`, `min` | `ohlcv-1m` | 1-minute bars |
| `1h`, `h`, `hourly` | `ohlcv-1h` | 1-hour bars |
| `1d`, `d`, `daily` | `ohlcv-1d` | Daily bars — **start here** |
| `eod` | `ohlcv-eod` | End-of-day bars |

!!! warning "5-minute bars"
    Databento does not natively provide 5-minute OHLCV bars.  To get 5-minute
    data, ingest `1m` and resample in Polars:
    ```python
    df.group_by_dynamic("ts_event", every="5m").agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ])
    ```

---

## Symbol Format

Databento uses a structured symbol notation for futures:

| Format | Meaning | Example |
|---|---|---|
| `<ROOT>.c.0` | Front-month continuous | `ES.c.0` |
| `<ROOT>.c.1` | Second-month continuous | `ES.c.1` |
| `<ROOT><MONTH><YEAR>` | Individual expiry | `ESZ2024` |
| `<ROOT>.FUT` (with `stype_in: parent`) | Every outright contract under the root | `ES.FUT` |

For systematic research, continuous contracts (`*.c.0`) are recommended as they
provide uninterrupted price series.  The raw data captures the roll events;
you will need to back-adjust separately (see [Continuous Contract Construction](#continuous-contract-construction)).

To verify a continuous series (`ES.c.0`) rolls correctly, ingest `ES.FUT`
with `--stype-in parent` (see [Single-Symbol Ingestion](#single-symbol-ingestion))
and compare it against the continuous series — Databento's `instrument_id`
column in the resulting file distinguishes which contract each row came from.

---

## Recommended Starting Universe

As described in the [architecture docs](../architecture/overview.md), start with
CME Globex daily bars across the four major asset classes:

| Asset Class | Symbols |
|---|---|
| Equity Index | ES, NQ, YM, RTY |
| Rates | ZT, ZF, ZN, ZB |
| FX | 6E, 6J, 6B, 6A |
| Commodities | CL, NG, GC, SI, ZC, ZW, ZS |

The minimal viable set for cross-asset strategy research is:

```yaml
asset_classes:
  core:
    symbols: [ES.c.0, ZN.c.0, CL.c.0, GC.c.0, 6E.c.0]
```

---

## Continuous Contract Construction

Databento provides **raw individual contracts**, not back-adjusted continuous
series.  After ingestion you should:

1. Pull front-month continuous symbols (`*.c.0`) from Databento.
2. Identify roll dates using volume or open interest crossover.
3. Back-adjust using:
   - **Difference method** for rates and equity index futures.
   - **Ratio method** for commodity futures.
4. Persist the adjusted series to `data/continuous/`.

This step is intentionally separate from raw ingestion so that raw data
remains immutable and can be re-adjusted with different methods.

---

## Loading Data in Python

```python
from pathlib import Path
from snippy_scales.data.ingest import load_bars

# Load daily bars for ES
df = load_bars("ES.c.0", "ohlcv-1d")
print(df.head())

# Custom data directory
df = load_bars("ES.c.0", "ohlcv-1d", output_dir=Path("data/raw"))
```

The returned :class:`polars.DataFrame` is sorted by `ts_event` and contains
the standard OHLCV columns.

---

## Programmatic Ingestion

You can call the ingestion functions directly from Python:

```python
from pathlib import Path
from snippy_scales.data.config import load_config
from snippy_scales.data.ingest import ingest_from_config, upsert_symbol

# Config-based batch ingestion (schemas × symbols, all expanded)
cfg = load_config(Path("configs/databento.yaml"))
rows = ingest_from_config(cfg)

# Single symbol upsert
path = upsert_symbol(
    dataset="GLBX.MDP3",
    symbol="ES.c.0",
    schema="ohlcv-1d",
    start="2020-01-01",
    end="2024-12-31",
)

# Every individual ES contract active in the range, via Databento parent symbology
path = upsert_symbol(
    dataset="GLBX.MDP3",
    symbol="ES.FUT",
    schema="ohlcv-1d",
    start="2020-01-01",
    end="2024-12-31",
    stype_in="parent",
)
```

Or against Alpaca directly, without a config file — `upsert_bars` is the
provider-agnostic entry point both `upsert_symbol` and the Alpaca path use
internally:

```python
from snippy_scales.data.ingest import upsert_bars
from snippy_scales.data.providers.alpaca import AlpacaProvider

# Reads ALPACA_API_KEY / ALPACA_SECRET_KEY from the environment.
provider = AlpacaProvider(rate_limit_per_min=190)
path = upsert_bars(
    provider=provider,
    symbol="AAPL",
    schema="ohlcv-1m",
    start="2024-01-01",
    end="2024-02-01",
)

# load_bars() works identically regardless of which provider wrote the file.
from snippy_scales.data.ingest import load_bars

df = load_bars("AAPL", "ohlcv-1m")
```

---

## API Reference

::: snippy_scales.data.config
    options:
      members:
        - frequency_to_schema
        - resolve_schema
        - is_tick_schema
        - AssetClassConfig
        - AlpacaConfig
        - IngestConfig
        - load_config

::: snippy_scales.data.ingest
    options:
      members:
        - upsert_bars
        - upsert_symbol
        - ingest_from_config
        - estimate_cost
        - is_range_cached
        - load_bars

::: snippy_scales.data.schema
    options:
      members:
        - conform_bars
        - BAR_SCHEMA_COLUMNS

::: snippy_scales.data.providers.base
    options:
      members:
        - BarProvider

::: snippy_scales.data.providers.alpaca
    options:
      members:
        - AlpacaProvider

::: snippy_scales.data.ratelimit
    options:
      members:
        - RateLimiter

::: snippy_scales.data.universe
    options:
      members:
        - sp500_ever_members
        - fetch_current_constituents
        - fetch_membership_changes

::: snippy_scales.data.tick
    options:
      members:
        - covered_days
        - missing_days
        - estimate_tick_cost
        - TickCostEstimate
        - upsert_ticks
        - load_ticks
