# Data Ingestion

All market data lives in one NautilusTrader `ParquetDataCatalog` at
`data/catalog/`. Research code reads it with
[`load_bars`](#loading-data-in-python); backtests and live trading read it
natively through Nautilus.

Two sources feed it, selected per config via `provider:`:

- **[Databento](https://databento.com)** (default) — futures and other
  exchange data, bars and event-level (tick) schemas. Downloaded through
  NautilusTrader's own `DatabentoHistoricalClient`.
- **[Alpaca](https://alpaca.markets)** — US equities, free historical tier,
  1-minute/1-hour/daily bars. NautilusTrader has no Alpaca adapter, so
  [`snippy_scales.data.alpaca`](#alpaca-free-tier-stock-bars) is a single
  loader that fetches with `alpaca-py` and writes to the catalog.

---

## Prerequisites

```bash
export DATABENTO_API_KEY="db-..."      # Databento
export ALPACA_API_KEY="..."            # Alpaca
export ALPACA_SECRET_KEY="..."
uv sync --all-extras
```

---

## Syncing

```bash
algo data sync configs/databento.yaml --dry-run   # what's missing, and its cost
algo data sync configs/databento.yaml             # fetch it (asks to confirm)
algo data sync configs/databento.yaml --schema 1h # one schema only
algo data sync configs/alpaca.yaml -s AAPL -s MSFT
algo data sync configs/databento.yaml --instruments-only
```

`sync` is incremental. For every symbol and schema it asks the catalog for the
first and last stored record and fetches only what is missing **before** the
first or **after** the last, within the config's `start`/`end`. Re-running
the same command fetches nothing; widening `start` fetches only the new head.
Interior gaps are not detected — to refill one, delete that range with
`ParquetDataCatalog.delete_data_range` and sync again.

Databento syncs also store instrument definitions (tick size, multiplier,
expiry). Each continuous symbol (`ES.c.0`) gets a synthetic instrument
`ES.c.0.GLBX` copied from its most recent outright, so backtests on the
continuous series have the right contract spec.

Incremental syncs leave many small files; merge them occasionally:

```bash
algo data consolidate
algo data coverage      # every bar series with its first and last bar close
```

### Config file

```yaml
dataset: "GLBX.MDP3"        # Databento only
# venue: "GLBX"             # default: dataset prefix (Databento) / ALPACA (Alpaca)
schemas: ["1d"]             # bar aliases and/or tick schemas
start: "2010-06-06"
# end: "2024-12-31"         # exclusive; default today
stype_in: "continuous"      # cost estimates only; inferred per symbol if omitted

asset_classes:              # free-form labels
  equity_index:
    symbols: [ES.c.0, NQ.c.0]
  sp500:
    symbols_file: ../data/universe/sp500_ever_members.txt   # one symbol per line
```

### Cost

`--dry-run` prints Databento's cost for exactly the ranges that would be
fetched (`metadata.get_cost`), without downloading. Already-stored data costs
nothing. Planning works without an API key; only the cost column needs one.

---

## Catalog Layout

Nautilus owns the layout; the parts you will see are:

```
data/catalog/data/
  bars/ES.c.0.GLBX-1-DAY-LAST-EXTERNAL/<start>_<end>.parquet
  bars/AAPL.ALPACA-1-MINUTE-LAST-EXTERNAL/...
  trades/ES.c.0.GLBX/...
  futures_contract/ES.c.0.GLBX/...
  equity/AAPL.ALPACA/...
```

- A bar series is `<symbol>.<venue>-<step>-LAST-EXTERNAL`.
- `ts_event` is the bar **open** and `ts_init` the bar **close**. Nautilus
  needs `ts_init` at the close so a bar is never visible before it formed.
- Prices are stored as `decimal128(38, 16)`, so polars and DuckDB read the
  files directly (see the [migration plan appendix](../architecture/nautilus_migration.md#appendix-reading-the-v2-catalog-outside-nautilus)).

---

## Tick (Event-Level) Data

Databento event-level schemas: `trades`, `tbbo`, `mbp-1`, `bbo-1s`, `bbo-1m`,
`mbo`, `mbp-10`. They are fetched one day per request and written as
Nautilus `TradeTick` / `QuoteTick` / `OrderBookDelta` / `OrderBookDepth10`
records. Keep tick configs separate from bar configs — a config's `schemas`
apply to every symbol in it (see `configs/databento_es_trades.yaml`).

For a multi-GB backfill, use a Databento **portal** batch download and load
the DBN files with `nautilus_trader.adapters.databento.DatabentoDataLoader`
(e.g. `load_trades(path, instrument_id)`) followed by
`catalog.write_trade_ticks(...)`.

---

## Alpaca (Free-Tier Stock Bars)

`provider: alpaca` configs use `AlpacaLoader`:

- Free (Basic) tier: 200 historical calls/minute, data since 2016, the latest
  15 minutes withheld. A shared rate limiter keeps concurrent fetches under
  `rate_limit_per_min`.
- `adjustment: all` (default) avoids fake jumps at split dates. Adjusted
  prices are off the 0.01 tick grid, so prices are stored with 4 decimals.
- Daily bars are stamped 00:00 UTC on the session date.
- Each symbol gets an `Equity` instrument at venue `ALPACA`.

```yaml
provider: "alpaca"
schemas: ["1d"]
start: "2016-01-01"
alpaca:
  feed: "sip"               # or "iex"
  adjustment: "all"
  rate_limit_per_min: 190
  max_workers: 4
asset_classes:
  mega_cap_tech:
    symbols: [AAPL, MSFT, GOOGL]
```

### S&P 500 universe and backfill

```bash
# Every ticker that was ever a constituent in the window (not survivorship-biased).
algo data update-universe --years 5

# Plan (universe size, request count, runtime) — no download.
algo data backfill-sp500 --dry-run

# Run it. Resumable: re-running fetches only what the catalog is missing.
# Per-symbol results go to data/derived/manifests/sp500_1m.csv.
algo data backfill-sp500
algo data backfill-sp500 --schema 1d
algo data backfill-sp500 --retry-failed
```

---

## Frequency Reference

| Alias | Schema | Nautilus bar step |
|---|---|---|
| `1s`, `s` | `ohlcv-1s` | `1-SECOND` (Databento only) |
| `1m`, `1min`, `min` | `ohlcv-1m` | `1-MINUTE` |
| `1h`, `h`, `hourly` | `ohlcv-1h` | `1-HOUR` |
| `1d`, `d`, `daily` | `ohlcv-1d` | `1-DAY` — **start here** |

Other intervals (e.g. 5 minutes) are aggregated from these — in polars with
`group_by_dynamic`, or natively in Nautilus with an `INTERNAL` bar type.

---

## Symbol Format

| Format | Meaning | Example |
|---|---|---|
| `<ROOT>.c.0` | Front-month continuous | `ES.c.0` |
| `<ROOT>.c.1` | Second-month continuous | `ES.c.1` |
| `<ROOT><MONTH><YEAR>` | Individual expiry | `ESZ4` |
| `<ROOT>.FUT` | Every outright contract under the root | `ES.FUT` |

Databento continuous series are **unadjusted** front-month prices: they jump
at every roll. Back-adjusting for signals and trading the real outright are
phase 4 of the [migration plan](../architecture/nautilus_migration.md).

---

## Recommended Starting Universe

CME Globex daily bars across four asset classes (`configs/databento.yaml`):

| Asset Class | Symbols |
|---|---|
| Equity Index | ES, NQ, YM, RTY |
| Rates | ZT, ZF, ZN, ZB |
| FX | 6E, 6J, 6B, 6A |
| Commodities | CL, NG, GC, SI, ZC, ZW, ZS |

---

## Loading Data in Python

```python
from snippy_scales.data.catalog import load_bars

bars = load_bars("ES.c.0", "ohlcv-1d")
bars = load_bars("AAPL", "ohlcv-1m", start="2024-01-01", end="2024-02-01")
bars = load_bars("ES.c.0", "ohlcv-1d", venue="GLBX", catalog_path=my_catalog)
```

Returns a polars DataFrame with `ts_event`, `ts_init`, `open`, `high`, `low`,
`close`, `volume`, `symbol`, sorted by `ts_event`. `start`/`end` filter on
the bar close.

Programmatic sync:

```python
from snippy_scales.data.alpaca import AlpacaLoader
from snippy_scales.data.catalog import open_catalog
from snippy_scales.data.databento import DatabentoSync

catalog = open_catalog()
DatabentoSync(catalog, dataset="GLBX.MDP3").sync("ES.c.0", "ohlcv-1d", start_ns, end_ns)
AlpacaLoader().sync(catalog, ["AAPL", "MSFT"], "ohlcv-1d", start_ns, end_ns)
```

---

## Migrating the Pre-Catalog Store

Data ingested before the migration lives in `data/raw/`. Copy it into the
catalog once:

```bash
uv run python scripts/migrate_raw_to_catalog.py --dry-run
uv run python scripts/migrate_raw_to_catalog.py
uv run python scripts/migrate_raw_to_catalog.py --verify-only
algo data sync configs/databento.yaml --instruments-only   # futures definitions
```

The script only reads `data/raw/`. Bars are copied with inferred precision;
tick data is rebuilt from the archived DBN files; the verification step
compares row counts and a close-price checksum per series. Deleting
`data/raw/` afterwards is a manual decision — keep it until you have checked
the catalog and have a backup.

---

## API Reference

See [Data API](../api/python/data.md).
