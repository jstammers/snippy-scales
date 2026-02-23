"""Databento data ingestion — downloads bar data and persists as Parquet.

Storage layout
--------------
Each symbol gets its own sub-directory under the raw data root so that
files remain small and individually addressable::

    data/
      raw/
        ES.c.0/
          ohlcv-1d.parquet
          ohlcv-1h.parquet
        ZN.c.0/
          ohlcv-1d.parquet

Upsert semantics
----------------
:func:`upsert_symbol` avoids re-downloading data that is already on disk.
On each call it:

1. Checks whether a Parquet file for the symbol/schema already exists.
2. If it does, reads the latest ``ts_event`` timestamp and sets the
   effective request start to the **next calendar day**, skipping any
   range already covered.
3. Downloads only the missing tail, concatenates it with the existing
   data, deduplicates on ``ts_event``, sorts chronologically, and
   overwrites the Parquet file.
4. If the existing file already covers the requested date range, the
   download is skipped entirely.

This keeps API costs low: each symbol incurs at most one day of overlap
on incremental runs.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

    from snippy_scales.data.config import IngestConfig

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _symbol_path(symbol: str, schema: str, output_dir: Path) -> Path:
    """Return the canonical Parquet path for a symbol/schema pair.

    Args:
        symbol: Databento symbol string (e.g. ``"ES.c.0"``).
        schema: Databento schema name (e.g. ``"ohlcv-1d"``).
        output_dir: Root directory for raw data.

    Returns:
        A :class:`~pathlib.Path` like ``<output_dir>/ES.c.0/ohlcv-1d.parquet``.
    """
    safe_symbol = symbol.replace("/", "_")
    return output_dir / safe_symbol / f"{schema}.parquet"


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert a Databento pandas DataFrame to a Polars DataFrame.

    Promotes the ``ts_event`` index to a regular column when Databento
    returns it as the DataFrame index.

    Args:
        df: pandas DataFrame returned by ``DBNStore.to_df()``.

    Returns:
        Equivalent :class:`polars.DataFrame` with ``ts_event`` as a column.
    """
    if df.index.name is not None and df.index.name != "index":
        df = df.reset_index()
    return pl.from_pandas(df)


def _effective_start(existing: pl.DataFrame, requested_start: str) -> str | None:
    """Compute the earliest date not yet covered by *existing* data.

    Args:
        existing: Polars DataFrame already on disk (must contain ``ts_event``).
        requested_start: The originally requested start date (``YYYY-MM-DD``).

    Returns:
        ISO date string for the next un-fetched day, or ``None`` if
        ``ts_event`` is absent or the column is empty.
    """
    if "ts_event" not in existing.columns:
        return requested_start

    max_ts = existing["ts_event"].max()
    if max_ts is None:
        return requested_start

    # Cast to Python datetime then extract date
    import pandas as pd  # noqa: PLC0415 — optional heavy dep, defer import

    last_date: date = pd.Timestamp(max_ts).date()
    next_day = last_date + timedelta(days=1)
    return next_day.isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upsert_symbol(
    *,
    dataset: str,
    symbol: str,
    schema: str,
    start: str,
    end: str,
    output_dir: Path = RAW_DIR,
) -> Path:
    """Download and upsert bar data for a single symbol.

    On the first call the full ``[start, end]`` range is fetched.  On
    subsequent calls only the portion after the latest locally stored bar is
    downloaded.  The result is merged with the existing file, deduplicated on
    ``ts_event``, sorted chronologically, and written back to Parquet.

    Args:
        dataset: Databento dataset code (e.g. ``"GLBX.MDP3"``).
        symbol: Instrument symbol (e.g. ``"ES.c.0"``).
        schema: Databento schema name (e.g. ``"ohlcv-1d"``).
        start: Earliest date to include (``YYYY-MM-DD``).
        end: Latest date to include (``YYYY-MM-DD``).
        output_dir: Root directory for raw Parquet files.

    Returns:
        Path to the written (or unchanged) Parquet file.

    Raises:
        RuntimeError: If the Databento API call fails.
    """
    import databento as db  # noqa: PLC0415 — optional dep

    out_path = _symbol_path(symbol, schema, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: pl.DataFrame | None = None
    fetch_start = start

    if out_path.exists():
        existing = pl.read_parquet(out_path)
        computed = _effective_start(existing, start)
        if computed is not None:
            fetch_start = computed

        if fetch_start >= end:
            logger.info(
                "[%s] Already up to date (coverage through %s). Skipping download.",
                symbol,
                fetch_start,
            )
            return out_path

    logger.info("[%s] Fetching %s  %s → %s", symbol, schema, fetch_start, end)

    client = db.Historical()
    store = client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema=schema,
        start=fetch_start,
        end=end,
    )

    new_df = _to_polars(store.to_df())

    if existing is not None and len(existing) > 0:
        combined = (
            pl.concat([existing, new_df], how="diagonal")
            .unique(subset=["ts_event"], keep="first")
            .sort("ts_event")
        )
        logger.info(
            "[%s] Appended %d new rows (total %d).",
            symbol,
            len(new_df),
            len(combined),
        )
    else:
        combined = new_df.sort("ts_event")
        logger.info("[%s] Wrote %d rows (initial load).", symbol, len(combined))

    combined.write_parquet(out_path)
    return out_path


def ingest_from_config(
    config: IngestConfig,
    *,
    frequency_override: str | None = None,
    output_dir: Path = RAW_DIR,
) -> dict[str, Path]:
    """Batch-ingest all symbols defined in a :class:`~snippy_scales.data.config.IngestConfig`.

    Iterates over every symbol in every asset class and calls
    :func:`upsert_symbol` for each one.  A ``frequency_override`` (from the
    ``--frequency`` CLI flag) takes precedence over the value in the config
    file.

    Args:
        config: Validated ingestion configuration.
        frequency_override: Optional frequency string (e.g. ``"1h"``) that
            overrides ``config.tick_frequency``.  Useful when the same config
            file is used at different resolutions from the command line.
        output_dir: Root directory for raw Parquet files.

    Returns:
        Mapping of symbol → path for every successfully ingested file.

    Raises:
        ValueError: If *frequency_override* is not a recognised frequency.
    """
    from snippy_scales.data.config import frequency_to_schema  # noqa: PLC0415

    if frequency_override is not None:
        schema = frequency_to_schema(frequency_override)
    else:
        schema = config.schema

    import datetime  # noqa: PLC0415

    end = config.end or datetime.date.today().isoformat()

    results: dict[str, Path] = {}
    symbols = config.all_symbols

    logger.info(
        "Starting ingestion of %d symbol(s) [%s / %s].",
        len(symbols),
        config.dataset,
        schema,
    )

    for symbol in symbols:
        try:
            path = upsert_symbol(
                dataset=config.dataset,
                symbol=symbol,
                schema=schema,
                start=config.start,
                end=end,
                output_dir=output_dir,
            )
            results[symbol] = path
        except Exception:
            logger.exception("Failed to ingest symbol %s — skipping.", symbol)

    logger.info("Ingestion complete. %d/%d symbols succeeded.", len(results), len(symbols))
    return results


def load_bars(symbol: str, schema: str, output_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Load a stored bar file into a Polars DataFrame.

    Args:
        symbol: Instrument symbol (e.g. ``"ES.c.0"``).
        schema: Databento schema name (e.g. ``"ohlcv-1d"``).
        output_dir: Root directory for raw Parquet files.

    Returns:
        A :class:`polars.DataFrame` sorted by ``ts_event``.

    Raises:
        FileNotFoundError: If no Parquet file exists for the given symbol and
            schema.
    """
    path = _symbol_path(symbol, schema, output_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No data found for symbol={symbol!r}, schema={schema!r} at {path}. "
            "Run `algo data ingest-config` first."
        )
    return pl.read_parquet(path)
