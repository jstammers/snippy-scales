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
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl

from snippy_scales.data.config import is_tick_schema, resolve_schema
from snippy_scales.data.paths import RAW_DIR

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from snippy_scales.data.config import VALID_STYPES, IngestConfig

logger = logging.getLogger(__name__)


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

    # Use Polars date casting to extract the max date — avoids any pandas dependency
    # and gives ty a well-typed result (datetime.date | None).
    last_date = existing.select(pl.col("ts_event").cast(pl.Date).max()).item()
    if last_date is None:
        return requested_start

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
    stype_in: VALID_STYPES = "raw_symbol",
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
        stype_in=stype_in,
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


@dataclass(frozen=True)
class IngestRow:
    """One row of a config-driven batch ingestion result.

    Attributes:
        asset_class: Asset-class label the item belongs to.
        schema: Resolved Databento schema name.
        symbol: Instrument symbol ingested, as listed in the config.
        succeeded: Whether ingestion completed without error.
        detail: Human-readable outcome (path written, day count, or error).
    """

    asset_class: str
    schema: str
    symbol: str
    succeeded: bool
    detail: str


@dataclass(frozen=True)
class CostRow:
    """One row of a config-driven batch cost estimate.

    Attributes:
        asset_class: Asset-class label the item belongs to.
        schema: Resolved Databento schema name.
        symbol: Instrument symbol the estimate applies to.
        cost_usd: Estimated cost in US dollars, or ``NaN`` if estimation failed.
    """

    asset_class: str
    schema: str
    symbol: str
    cost_usd: float


def ingest_from_config(
    config: IngestConfig,
    *,
    schema_override: str | None = None,
    output_dir: Path = RAW_DIR,
) -> list[IngestRow]:
    """Batch-ingest every symbol across every requested schema in a config.

    Each asset class's ``symbols`` are ingested as-is — including Databento
    parent symbology (e.g. ``"ES.FUT"`` with ``stype_in: parent``), which
    Databento itself expands to every individual outright contract in a
    single request. Each resolved schema is routed to the bar
    (:func:`upsert_symbol`) or event-level
    (:func:`snippy_scales.data.tick.upsert_ticks`) ingestion path via
    :func:`~snippy_scales.data.config.is_tick_schema`.

    Args:
        config: Validated ingestion configuration.
        schema_override: Optional schema string (bar alias or tick schema
            name) that restricts ingestion to a single schema instead of the
            full ``config.schemas`` list.
        output_dir: Root directory for raw data.

    Returns:
        One :class:`IngestRow` per ``(asset class, schema, symbol)`` triple.

    Raises:
        ValueError: If *schema_override* is not a recognised schema.
    """
    import datetime  # noqa: PLC0415

    from snippy_scales.data.tick import upsert_ticks  # noqa: PLC0415

    schemas = [resolve_schema(schema_override)] if schema_override else config.resolved_schemas
    end = config.end or datetime.date.today().isoformat()
    stype_in = config.stype_in or "raw_symbol"

    rows: list[IngestRow] = []

    for class_name, ac in config.asset_classes.items():
        for schema in schemas:
            for symbol in ac.symbols:
                try:
                    if is_tick_schema(schema):
                        written = upsert_ticks(
                            dataset=config.dataset,
                            symbol=symbol,
                            schema=schema,
                            start=config.start,
                            end=end,
                            output_dir=output_dir,
                            stype_in=stype_in,
                        )
                        detail = f"{len(written)} day(s) written"
                    else:
                        path = upsert_symbol(
                            dataset=config.dataset,
                            symbol=symbol,
                            schema=schema,
                            start=config.start,
                            end=end,
                            output_dir=output_dir,
                            stype_in=stype_in,
                        )
                        detail = str(path)
                    rows.append(IngestRow(class_name, schema, symbol, True, detail))
                except Exception as exc:
                    logger.exception("Failed to ingest %s (%s) — skipping.", symbol, schema)
                    rows.append(IngestRow(class_name, schema, symbol, False, str(exc)))

    succeeded = sum(1 for row in rows if row.succeeded)
    logger.info("Ingestion complete. %d/%d item(s) succeeded.", succeeded, len(rows))
    return rows


def estimate_cost(
    *,
    dataset: str,
    symbol: str,
    schema: str,
    start: str,
    end: str,
    output_dir: Path = RAW_DIR,
    stype_in: VALID_STYPES = "raw_symbol",
) -> float:
    """Estimate the Databento API cost in USD for a single symbol download.

    Applies the same upsert logic as :func:`upsert_symbol`: if local data
    already covers the full requested range the symbol is up to date and
    ``0.0`` is returned immediately without making any API call.

    Args:
        dataset: Databento dataset code (e.g. ``"GLBX.MDP3"``).
        symbol: Instrument symbol (e.g. ``"ES.c.0"``).
        schema: Databento schema name (e.g. ``"ohlcv-1d"``).
        start: Requested start date (``YYYY-MM-DD``).
        end: Requested end date (``YYYY-MM-DD``).
        output_dir: Root directory for raw Parquet files.
        stype_in: Optional Databento stype string (e.g. ``"continuous"``) to pass to the API.

    Returns:
        Estimated cost in US dollars.  Returns ``0.0`` when no download is
        needed (symbol already up to date).

    Raises:
        RuntimeError: If the Databento metadata API call fails.
    """
    import databento as db  # noqa: PLC0415 — optional dep

    out_path = _symbol_path(symbol, schema, output_dir)
    fetch_start = start

    if out_path.exists():
        existing = pl.read_parquet(out_path)
        computed = _effective_start(existing, start)
        if computed is not None:
            fetch_start = computed
        if fetch_start >= end:
            return 0.0
    client = db.Historical()
    cost: float = client.metadata.get_cost(
        dataset=dataset,
        start=fetch_start,
        end=end,
        symbols=[symbol],
        schema=schema,
        stype_in=stype_in,
    )
    return cost


def estimate_costs_from_config(
    config: IngestConfig,
    *,
    schema_override: str | None = None,
    output_dir: Path = RAW_DIR,
) -> list[CostRow]:
    """Estimate download costs for every symbol across every requested schema.

    Mirrors :func:`ingest_from_config`'s bar/tick routing, but calls
    :func:`estimate_cost` / :func:`snippy_scales.data.tick.estimate_tick_cost`
    instead of downloading.

    Per-item costs respect upsert state: items whose local data already
    covers the requested range contribute ``0.0``. Items where the Databento
    metadata call fails contribute ``float('nan')`` so the caller can surface
    a warning without aborting the entire estimate.

    Args:
        config: Validated ingestion configuration.
        schema_override: Optional schema string that restricts estimation to
            a single schema instead of the full ``config.schemas`` list.
        output_dir: Root directory for raw data.

    Returns:
        One :class:`CostRow` per ``(asset class, schema, symbol)`` triple.
    """
    import datetime  # noqa: PLC0415

    from snippy_scales.data.tick import estimate_tick_cost  # noqa: PLC0415

    schemas = [resolve_schema(schema_override)] if schema_override else config.resolved_schemas
    end = config.end or datetime.date.today().isoformat()
    stype_in = config.stype_in or "raw_symbol"

    rows: list[CostRow] = []

    for class_name, ac in config.asset_classes.items():
        for schema in schemas:
            for symbol in ac.symbols:
                try:
                    if is_tick_schema(schema):
                        cost = estimate_tick_cost(
                            dataset=config.dataset,
                            symbol=symbol,
                            schema=schema,
                            start=config.start,
                            end=end,
                            output_dir=output_dir,
                            stype_in=stype_in,
                        ).cost_usd
                    else:
                        cost = estimate_cost(
                            dataset=config.dataset,
                            symbol=symbol,
                            schema=schema,
                            start=config.start,
                            end=end,
                            output_dir=output_dir,
                            stype_in=stype_in,
                        )
                    rows.append(CostRow(class_name, schema, symbol, cost))
                except Exception:
                    logger.warning(
                        "Could not estimate cost for %s (%s) — recorded as NaN.", symbol, schema
                    )
                    rows.append(CostRow(class_name, schema, symbol, float("nan")))

    return rows


def load_bars(symbol: str, schema: str, output_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Load a stored bar file into a Polars DataFrame.

    Args:
        symbol: Instrument symbol (e.g. ``"ES.c.0"``, or ``"ES.FUT"`` if
            ingested via Databento parent symbology).
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
