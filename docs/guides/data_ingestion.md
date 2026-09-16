# Data Ingestion

This guide covers how to pull futures bar data from [Databento](https://databento.com)
into the local Parquet store used by the backtesting and research layers.

---

## Prerequisites

1. **Databento account** — sign up at <https://databento.com> and note your API key.
2. **Set the API key** as an environment variable (the Databento client reads it
   automatically):
   ```bash
   export DATABENTO_API_KEY="db-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```
3. **Install project dependencies** (includes `databento` and `pyyaml`):
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
| `dataset` | Databento dataset code. Use `GLBX.MDP3` for CME Globex. |
| `schemas` | List of bar aliases and/or event-level schema names to ingest (see [Frequency Reference](#frequency-reference)). Each is ingested for every symbol below. |
| `start` | Earliest date to fetch (`YYYY-MM-DD`). |
| `end` | Latest date to fetch. Omit to default to today. |
| `stype_in` | Databento symbology type applied to every symbol in this config (e.g. `continuous`, `parent`). |
| `asset_classes` | Mapping of human-readable labels to `symbols` — explicit Databento identifiers (continuous, raw, or parent notation; see [Symbol Format](#symbol-format)). |

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

---

## API Reference

::: snippy_scales.data.config
    options:
      members:
        - frequency_to_schema
        - resolve_schema
        - is_tick_schema
        - AssetClassConfig
        - IngestConfig
        - load_config

::: snippy_scales.data.ingest
    options:
      members:
        - upsert_symbol
        - ingest_from_config
        - estimate_cost
        - load_bars

::: snippy_scales.data.tick
    options:
      members:
        - covered_days
        - missing_days
        - estimate_tick_cost
        - TickCostEstimate
        - upsert_ticks
        - load_ticks
