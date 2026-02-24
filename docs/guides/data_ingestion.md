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

---

## Config-Based Batch Ingestion (Recommended)

The primary workflow is to define your universe in a YAML config file and run a
single CLI command.

### 1. Edit the config

The starter config lives at `configs/databento.yaml`:

```yaml
dataset: "GLBX.MDP3"
tick_frequency: "1d"
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
| `tick_frequency` | Bar resolution (see [Frequency Reference](#frequency-reference)). |
| `start` | Earliest date to fetch (`YYYY-MM-DD`). |
| `end` | Latest date to fetch. Omit to default to today. |
| `asset_classes` | Mapping of human-readable labels to lists of symbols. |

### 2. Run the ingestion

```bash
# Full universe, daily bars (as configured)
algo data ingest-config configs/databento.yaml

# Override to hourly bars at the command line
algo data ingest-config configs/databento.yaml --frequency 1h

# Preview what would be fetched without downloading anything
algo data ingest-config configs/databento.yaml --dry-run
```

The command prints a table of the ingestion plan, then runs the upsert for
each symbol in sequence, logging progress and any failures.

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
  --frequency 1d
```

This also uses upsert semantics — if `data/raw/ES.c.0/ohlcv-1d.parquet` already
exists, only the missing tail is fetched.

---

## Frequency Reference

The `--frequency` flag (and `tick_frequency` config field) accept the following
aliases.  Databento natively supports these bar schemas:

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

For systematic research, continuous contracts (`*.c.0`) are recommended as they
provide uninterrupted price series.  The raw data captures the roll events;
you will need to back-adjust separately (see [Continuous Contract Construction](#continuous-contract-construction)).

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

# Config-based batch ingestion
cfg = load_config(Path("configs/databento.yaml"))
results = ingest_from_config(cfg)

# Single symbol upsert
path = upsert_symbol(
    dataset="GLBX.MDP3",
    symbol="ES.c.0",
    schema="ohlcv-1d",
    start="2020-01-01",
    end="2024-12-31",
)
```

---

## API Reference

::: snippy_scales.data.config
    options:
      members:
        - frequency_to_schema
        - AssetClassConfig
        - IngestConfig
        - load_config

::: snippy_scales.data.ingest
    options:
      members:
        - upsert_symbol
        - ingest_from_config
        - load_bars
