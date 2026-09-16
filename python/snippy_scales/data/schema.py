"""Shared bar-data column schema, used to make every :class:`BarProvider`'s
output indistinguishable from Databento's once written to Parquet.

Databento's OHLCV response already carries ``ts_event``, ``rtype``,
``publisher_id``, ``instrument_id``, ``open``, ``high``, ``low``, ``close``,
``volume``, and ``symbol`` columns. Other providers (Alpaca, ...) only have
the OHLCV core — :func:`conform_bars` fills in the Databento-shaped metadata
columns they lack (with provider-neutral sentinel values) and leaves any
columns already present untouched, so :func:`~snippy_scales.data.ingest.load_bars`
returns the same shape regardless of which provider wrote the file.
"""

from __future__ import annotations

import polars as pl

#: Column order every stored bar file conforms to (before any provider-specific
#: extras such as Alpaca's ``vwap``/``trade_count``, which are appended after).
BAR_SCHEMA_COLUMNS: tuple[str, ...] = (
    "ts_event",
    "rtype",
    "publisher_id",
    "instrument_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
)

#: Columns every provider must supply — the rest are synthesised if absent.
_REQUIRED_COLUMNS: tuple[str, ...] = ("ts_event", "open", "high", "low", "close", "volume")

#: Databento's ``RType`` enum value for each OHLCV schema, used to populate the
#: ``rtype`` column for providers (e.g. Alpaca) that don't supply one natively.
#: See https://databento.com/docs/schemas-and-conventions.
RTYPE_BY_SCHEMA: dict[str, int] = {
    "ohlcv-1s": 32,
    "ohlcv-1m": 33,
    "ohlcv-1h": 34,
    "ohlcv-1d": 35,
    "ohlcv-eod": 36,
}


def conform_bars(df: pl.DataFrame, *, symbol: str, schema: str) -> pl.DataFrame:
    """Cast/order *df* to the shared bar schema, synthesising missing metadata columns.

    Only columns that are **absent** are synthesised — a real Databento frame
    that already carries ``rtype``/``publisher_id``/``instrument_id``/``symbol``
    passes through with just a dtype-normalising cast and column reorder.

    Args:
        df: Raw provider output. Must contain ``ts_event`` and the OHLCV
            columns (``open``, ``high``, ``low``, ``close``, ``volume``).
        symbol: Instrument symbol, used to fill a missing ``symbol`` column.
        schema: Bar schema name (e.g. ``"ohlcv-1m"``), used to fill a missing
            ``rtype`` column via :data:`RTYPE_BY_SCHEMA`.

    Returns:
        A new DataFrame with :data:`BAR_SCHEMA_COLUMNS` first (in order),
        followed by any provider-specific extra columns.

    Raises:
        ValueError: If *df* is missing one of the required OHLCV columns.
    """
    missing_required = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Bar frame is missing required column(s) {missing_required} — got {df.columns}."
        )

    out = df

    if "rtype" not in out.columns:
        rtype = RTYPE_BY_SCHEMA.get(schema, 0)
        out = out.with_columns(pl.lit(rtype, dtype=pl.UInt8).alias("rtype"))
    if "publisher_id" not in out.columns:
        out = out.with_columns(pl.lit(0, dtype=pl.UInt16).alias("publisher_id"))
    if "instrument_id" not in out.columns:
        out = out.with_columns(pl.lit(0, dtype=pl.UInt32).alias("instrument_id"))
    if "symbol" not in out.columns:
        out = out.with_columns(pl.lit(symbol, dtype=pl.Utf8).alias("symbol"))

    out = out.with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
        pl.col("rtype").cast(pl.UInt8),
        pl.col("publisher_id").cast(pl.UInt16),
        pl.col("instrument_id").cast(pl.UInt32),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.UInt64),
        pl.col("symbol").cast(pl.Utf8),
    )

    extra = [c for c in out.columns if c not in BAR_SCHEMA_COLUMNS]
    return out.select([*BAR_SCHEMA_COLUMNS, *extra])


def empty_bar_frame(*, symbol: str, schema: str) -> pl.DataFrame:
    """Return a zero-row, correctly-typed bar frame for *symbol*/*schema*.

    Used by providers when a request legitimately returns no rows (e.g. the
    requested range is entirely non-trading days), so callers can merge the
    result with :func:`~snippy_scales.data.storage.merge_and_write` uniformly
    without special-casing "no data".

    Args:
        symbol: Instrument symbol to stamp in the ``symbol`` column.
        schema: Bar schema name, used for the ``rtype`` column.

    Returns:
        An empty :class:`polars.DataFrame` with :data:`BAR_SCHEMA_COLUMNS`.
    """
    empty = pl.DataFrame(
        {
            "ts_event": pl.Series([], dtype=pl.Datetime("ns", "UTC")),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.UInt64),
        }
    )
    return conform_bars(empty, symbol=symbol, schema=schema)
