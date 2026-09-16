"""Bar/tick data ingestion — downloads OHLCV bar data and persists as Parquet.

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

This layout — and the Parquet column schema within it — is shared by every
:class:`~snippy_scales.data.providers.base.BarProvider`
(:mod:`snippy_scales.data.providers`), so :func:`load_bars` returns an
identical shape whether the file came from Databento or Alpaca.

Upsert semantics
----------------
:func:`upsert_bars` avoids re-downloading data that is already on disk.
On each call it:

1. Checks whether a Parquet file for the symbol/schema already exists.
2. If it does, computes the effective request start from the latest stored
   bar — the day after it for providers that bill/resume per calendar day
   (Databento), or the exact timestamp after it for providers whose bars can
   cross a UTC day boundary (Alpaca's extended-hours sessions) — see
   :attr:`~snippy_scales.data.providers.base.BarProvider.resume_granularity`.
3. Downloads only the missing tail, concatenates it with the existing
   data, deduplicates on ``ts_event``, sorts chronologically, and
   overwrites the Parquet file.
4. If the existing file already covers the requested date range, the
   download is skipped entirely.

This keeps API costs (or, for a free provider, rate-limit spend) low: each
symbol incurs at most one day/bar of overlap on incremental runs.

:func:`upsert_symbol` is Databento's thin wrapper around :func:`upsert_bars`,
kept for backward compatibility with callers that pass ``dataset``/``stype_in``
directly instead of constructing a
:class:`~snippy_scales.data.providers.databento.DatabentoProvider`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl

from snippy_scales.data.config import is_tick_schema, resolve_schema
from snippy_scales.data.paths import RAW_DIR
from snippy_scales.data.providers.databento import _to_polars  # noqa: F401 — re-exported
from snippy_scales.data.schema import conform_bars

if TYPE_CHECKING:
    from pathlib import Path

    from snippy_scales.data.config import VALID_STYPES, IngestConfig
    from snippy_scales.data.providers.base import BarProvider, ResumeGranularity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _symbol_path(symbol: str, schema: str, output_dir: Path) -> Path:
    """Return the canonical Parquet path for a symbol/schema pair.

    Args:
        symbol: Instrument symbol string (e.g. ``"ES.c.0"``, ``"AAPL"``).
        schema: Bar schema name (e.g. ``"ohlcv-1d"``).
        output_dir: Root directory for raw data.

    Returns:
        A :class:`~pathlib.Path` like ``<output_dir>/ES.c.0/ohlcv-1d.parquet``.
    """
    safe_symbol = symbol.replace("/", "_")
    return output_dir / safe_symbol / f"{schema}.parquet"


def _effective_start(existing: pl.DataFrame, requested_start: str) -> str | None:
    """Compute the earliest date not yet covered by *existing* data.

    Used for providers with ``resume_granularity == "day"``.

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


def _effective_start_ts(existing: pl.DataFrame, requested_start: str) -> str | None:
    """Compute the earliest timestamp not yet covered by *existing* data.

    Used for providers with ``resume_granularity == "timestamp"``: resuming
    from "the day after the last bar" would silently skip the remainder of a
    session whose bars cross a UTC day boundary (e.g. Alpaca's extended-hours
    bars run past midnight UTC in winter).

    Args:
        existing: Polars DataFrame already on disk (must contain ``ts_event``).
        requested_start: The originally requested start (``YYYY-MM-DD`` or
            an RFC-3339 timestamp).

    Returns:
        An RFC-3339 timestamp one nanosecond after the last stored bar, or
        ``None`` if ``ts_event`` is absent or the column is empty.
    """
    if "ts_event" not in existing.columns:
        return requested_start

    last_ts = existing.select(pl.col("ts_event").max()).item()
    if last_ts is None:
        return requested_start

    next_ts = last_ts + timedelta(microseconds=1)
    return next_ts.isoformat()


def _effective_start_for(
    resume_granularity: ResumeGranularity, existing: pl.DataFrame, requested_start: str
) -> str | None:
    """Dispatch to :func:`_effective_start` or :func:`_effective_start_ts` by granularity."""
    if resume_granularity == "timestamp":
        return _effective_start_ts(existing, requested_start)
    return _effective_start(existing, requested_start)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_range_cached(
    *,
    resume_granularity: ResumeGranularity,
    symbol: str,
    schema: str,
    start: str,
    end: str,
    output_dir: Path = RAW_DIR,
) -> bool:
    """Return whether ``[start, end)`` is already fully covered on disk.

    A local-only check — makes no network call — used by cost/plan estimation
    to report "cached" for a provider (like Alpaca) whose cost is always
    ``0.0`` and so can't use ``cost_usd`` itself as the cached signal.

    Args:
        resume_granularity: ``"day"`` or ``"timestamp"`` — see
            :data:`~snippy_scales.data.providers.base.ResumeGranularity`.
        symbol: Instrument symbol.
        schema: Bar schema name.
        start: Requested start date/timestamp.
        end: Requested (exclusive) end date/timestamp.
        output_dir: Root directory for raw Parquet files.

    Returns:
        ``True`` if no download would be needed for this range.
    """
    out_path = _symbol_path(symbol, schema, output_dir)
    if not out_path.exists():
        return False
    existing = pl.read_parquet(out_path)
    fetch_start = _effective_start_for(resume_granularity, existing, start) or start
    return fetch_start >= end


def upsert_bars(
    *,
    provider: BarProvider,
    symbol: str,
    schema: str,
    start: str,
    end: str,
    output_dir: Path = RAW_DIR,
) -> Path:
    """Download and upsert bar data for a single symbol from any provider.

    See the module docstring for upsert semantics.

    Args:
        provider: The :class:`~snippy_scales.data.providers.base.BarProvider`
            to fetch from (e.g. ``DatabentoProvider(...)``, ``AlpacaProvider(...)``).
        symbol: Instrument symbol, in the convention *provider* expects.
        schema: Bar schema name (e.g. ``"ohlcv-1d"``).
        start: Earliest date/timestamp to include.
        end: Exclusive end date/timestamp.
        output_dir: Root directory for raw Parquet files.

    Returns:
        Path to the written (or unchanged) Parquet file.
    """
    out_path = _symbol_path(symbol, schema, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: pl.DataFrame | None = None
    fetch_start = start

    if out_path.exists():
        existing = pl.read_parquet(out_path)
        if "ts_event" in existing.columns:
            # Re-conform whatever's on disk to the exact dtypes/column order a
            # fresh fetch gets below, so a file written before a column had a
            # fixed dtype (or by a provider with different native dtypes)
            # still concatenates cleanly instead of raising a SchemaError.
            existing = conform_bars(existing, symbol=symbol, schema=schema)
        computed = _effective_start_for(provider.resume_granularity, existing, start)
        if computed is not None:
            fetch_start = computed

        if fetch_start >= end:
            logger.info(
                "[%s] Already up to date (coverage through %s). Skipping download.",
                symbol,
                fetch_start,
            )
            return out_path

    logger.info("[%s/%s] Fetching %s  %s → %s", provider.name, symbol, schema, fetch_start, end)

    new_df = provider.fetch_bars(symbol=symbol, schema=schema, start=fetch_start, end=end)

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
    """Download and upsert Databento bar data for a single symbol.

    Thin Databento-specific wrapper around :func:`upsert_bars` — kept for
    backward compatibility with callers that pass ``dataset``/``stype_in``
    directly. Non-Databento providers call :func:`upsert_bars` directly with
    their own :class:`~snippy_scales.data.providers.base.BarProvider`.

    Args:
        dataset: Databento dataset code (e.g. ``"GLBX.MDP3"``).
        symbol: Instrument symbol (e.g. ``"ES.c.0"``).
        schema: Databento schema name (e.g. ``"ohlcv-1d"``).
        start: Earliest date to include (``YYYY-MM-DD``).
        end: Latest date to include (``YYYY-MM-DD``).
        output_dir: Root directory for raw Parquet files.
        stype_in: Databento symbology type for the request.

    Returns:
        Path to the written (or unchanged) Parquet file.

    Raises:
        RuntimeError: If the Databento API call fails.
    """
    from snippy_scales.data.providers.databento import DatabentoProvider  # noqa: PLC0415

    provider = DatabentoProvider(dataset=dataset, stype_in=stype_in)
    return upsert_bars(
        provider=provider,
        symbol=symbol,
        schema=schema,
        start=start,
        end=end,
        output_dir=output_dir,
    )


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
            Always ``0.0`` for a free provider (Alpaca) — use :attr:`cached`,
            not ``cost_usd == 0.0``, to tell "nothing to download" apart from
            "download is free".
        cached: Whether the requested range is already fully covered on disk
            (no download needed), regardless of whether the provider bills.
    """

    asset_class: str
    schema: str
    symbol: str
    cost_usd: float
    cached: bool = False


def _ingest_one_databento(
    *,
    config: IngestConfig,
    class_name: str,
    schema: str,
    symbol: str,
    end: str,
    stype_in: VALID_STYPES,
    output_dir: Path,
) -> IngestRow:
    """Ingest one ``(asset class, schema, symbol)`` triple via Databento."""
    from snippy_scales.data.tick import upsert_ticks  # noqa: PLC0415

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
        return IngestRow(class_name, schema, symbol, True, detail)
    except Exception as exc:
        logger.exception("Failed to ingest %s (%s) — skipping.", symbol, schema)
        return IngestRow(class_name, schema, symbol, False, str(exc))


def _ingest_one_alpaca(
    *,
    provider: BarProvider,
    class_name: str,
    schema: str,
    symbol: str,
    start: str,
    end: str,
    output_dir: Path,
) -> IngestRow:
    """Ingest one ``(asset class, schema, symbol)`` triple via *provider* (Alpaca)."""
    try:
        path = upsert_bars(
            provider=provider,
            symbol=symbol,
            schema=schema,
            start=start,
            end=end,
            output_dir=output_dir,
        )
        return IngestRow(class_name, schema, symbol, True, str(path))
    except Exception as exc:
        logger.exception("Failed to ingest %s (%s) — skipping.", symbol, schema)
        return IngestRow(class_name, schema, symbol, False, str(exc))


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
    single request. For ``provider == "databento"``, each resolved schema is
    routed to the bar (:func:`upsert_symbol`) or event-level
    (:func:`snippy_scales.data.tick.upsert_ticks`) ingestion path via
    :func:`~snippy_scales.data.config.is_tick_schema`, run sequentially (each
    request is separately billed, so there's no throughput reason to
    parallelise). For ``provider == "alpaca"`` (bar schemas only — enforced
    when the config is validated), symbols are fetched concurrently across
    ``config.alpaca_options.max_workers`` threads sharing one rate-limited
    :class:`~snippy_scales.data.providers.alpaca.AlpacaProvider`.

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

    schemas = [resolve_schema(schema_override)] if schema_override else config.resolved_schemas
    end = config.end or datetime.date.today().isoformat()

    tasks = [
        (class_name, schema, symbol)
        for class_name, ac in config.asset_classes.items()
        for schema in schemas
        for symbol in ac.symbols
    ]

    if config.provider == "alpaca":
        from snippy_scales.data.providers.alpaca import AlpacaProvider  # noqa: PLC0415

        options = config.alpaca_options
        provider = AlpacaProvider(
            feed=options.feed,
            adjustment=options.adjustment,
            rate_limit_per_min=options.rate_limit_per_min,
        )
        with ThreadPoolExecutor(max_workers=options.max_workers) as pool:
            rows = list(
                pool.map(
                    lambda task: _ingest_one_alpaca(
                        provider=provider,
                        class_name=task[0],
                        schema=task[1],
                        symbol=task[2],
                        start=config.start,
                        end=end,
                        output_dir=output_dir,
                    ),
                    tasks,
                )
            )
    else:
        stype_in = config.stype_in or "raw_symbol"
        rows = [
            _ingest_one_databento(
                config=config,
                class_name=class_name,
                schema=schema,
                symbol=symbol,
                end=end,
                stype_in=stype_in,
                output_dir=output_dir,
            )
            for class_name, schema, symbol in tasks
        ]

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
        Every row is ``0.0`` when ``config.provider == "alpaca"`` — Alpaca's
        historical data has no per-request charge, only a rate limit (see
        :class:`~snippy_scales.data.config.AlpacaConfig`), so no API call is
        made to produce these rows.
    """
    import datetime  # noqa: PLC0415

    schemas = [resolve_schema(schema_override)] if schema_override else config.resolved_schemas
    end = config.end or datetime.date.today().isoformat()

    if config.provider == "alpaca":
        return [
            CostRow(
                class_name,
                schema,
                symbol,
                0.0,
                cached=is_range_cached(
                    resume_granularity="timestamp",
                    symbol=symbol,
                    schema=schema,
                    start=config.start,
                    end=end,
                    output_dir=output_dir,
                ),
            )
            for class_name, ac in config.asset_classes.items()
            for schema in schemas
            for symbol in ac.symbols
        ]

    from snippy_scales.data.tick import estimate_tick_cost  # noqa: PLC0415

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
                    rows.append(CostRow(class_name, schema, symbol, cost, cached=cost == 0.0))
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
