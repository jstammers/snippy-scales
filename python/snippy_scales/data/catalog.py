"""Market-data storage on a NautilusTrader ``ParquetDataCatalog``.

The catalog at :data:`~snippy_scales.data.paths.CATALOG_DIR` is the single store
for bars, ticks and instrument definitions. Research code reads it through
:func:`load_bars` (a polars DataFrame, same shape as before the migration);
backtests and live trading read it natively through Nautilus.

Naming
------
A bar series is identified by a Nautilus ``BarType`` string built from the
repo symbol, a venue and the Databento-style schema name::

    ES.c.0 + GLBX + ohlcv-1d  ->  ES.c.0.GLBX-1-DAY-LAST-EXTERNAL
    AAPL   + ALPACA + ohlcv-1m ->  AAPL.ALPACA-1-MINUTE-LAST-EXTERNAL

Timestamps
----------
``ts_event`` is the bar open and ``ts_init`` the bar close. Nautilus requires
``ts_init`` to be the close so a bar is never visible before it has formed;
:func:`frame_to_bars` enforces this for provider data stamped at the open.

Writes
------
The catalog rejects overlapping writes, so :func:`write_bars` is incremental: it
writes only bars before the first or after the last stored ``ts_init`` and skips
anything inside existing coverage. Filling an interior gap needs
``catalog.delete_data_range`` over the span, then a rewrite.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyarrow as pa
from nautilus_trader.model import (
    Bar,
    Currency,
    Equity,
    FuturesContract,
    InstrumentId,
    NautilusDataType,
    Price,
    Symbol,
)
from nautilus_trader.persistence import BarDataWrangler, ParquetDataCatalog

from snippy_scales.data.paths import CATALOG_DIR

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: Databento OHLCV schema name -> (Nautilus bar step-aggregation, bar length in ns).
BAR_SPECS: dict[str, tuple[str, int]] = {
    "ohlcv-1s": ("1-SECOND", 1_000_000_000),
    "ohlcv-1m": ("1-MINUTE", 60_000_000_000),
    "ohlcv-1h": ("1-HOUR", 3_600_000_000_000),
    "ohlcv-1d": ("1-DAY", 86_400_000_000_000),
}

#: Columns :func:`load_bars` returns, in order.
BAR_COLUMNS: tuple[str, ...] = (
    "ts_event",
    "ts_init",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
)

#: Arrow type the Nautilus v2 ``BarDataWrangler`` requires for price/volume columns.
_DECIMAL = pa.decimal128(38, 16)
#: Highest price precision :func:`infer_price_precision` will report.
_MAX_INFERRED_PRECISION = 9


@dataclass(frozen=True)
class SyncResult:
    """Outcome of syncing one symbol/schema into the catalog."""

    symbol: str
    schema: str
    written: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _bar_spec(schema: str) -> tuple[str, int]:
    try:
        return BAR_SPECS[schema]
    except KeyError:
        raise ValueError(
            f"Unsupported bar schema {schema!r}. Supported: {sorted(BAR_SPECS)}."
        ) from None


def instrument_id(symbol: str, venue: str) -> str:
    """Return the Nautilus ``InstrumentId`` string for *symbol* at *venue*."""
    return f"{symbol}.{venue}"


def bar_type(symbol: str, venue: str, schema: str) -> str:
    """Return the Nautilus ``BarType`` string for a provider-aggregated bar series."""
    step, _ = _bar_spec(schema)
    return f"{instrument_id(symbol, venue)}-{step}-LAST-EXTERNAL"


def bar_interval_ns(schema: str) -> int:
    """Return the bar length in nanoseconds for *schema*."""
    return _bar_spec(schema)[1]


def _parse_bar_dir(name: str, schema: str) -> tuple[str, str] | None:
    """Split a catalog bar directory name into ``(symbol, venue)`` for *schema*, else None."""
    suffix = f"-{_bar_spec(schema)[0]}-LAST-EXTERNAL"
    if not name.endswith(suffix):
        return None
    instrument = name[: -len(suffix)]
    symbol, sep, venue = instrument.rpartition(".")
    if not sep or not symbol or not venue:
        return None
    return symbol, venue


# ---------------------------------------------------------------------------
# Catalog access
# ---------------------------------------------------------------------------


def open_catalog(path: Path | None = None) -> ParquetDataCatalog:
    """Open (creating the directory if needed) the catalog at *path*.

    Args:
        path: Catalog root. Defaults to :data:`~snippy_scales.data.paths.CATALOG_DIR`.
    """
    root = path if path is not None else CATALOG_DIR
    root.mkdir(parents=True, exist_ok=True)
    return ParquetDataCatalog(str(root))


def list_bar_series(schema: str, catalog_path: Path | None = None) -> list[tuple[str, str]]:
    """Return every ``(symbol, venue)`` that has *schema* bars in the catalog, sorted."""
    bars_dir = (catalog_path if catalog_path is not None else CATALOG_DIR) / "data" / "bars"
    if not bars_dir.is_dir():
        return []
    found = (_parse_bar_dir(p.name, schema) for p in bars_dir.iterdir() if p.is_dir())
    return sorted(s for s in found if s is not None)


def resolve_bar_type(
    symbol: str, schema: str, *, venue: str | None = None, catalog_path: Path | None = None
) -> str:
    """Find the catalog ``BarType`` for *symbol*/*schema*.

    Args:
        symbol: Repo symbol (e.g. ``"ES.c.0"``, ``"AAPL"``).
        schema: Bar schema (e.g. ``"ohlcv-1d"``).
        venue: If given, used directly. Otherwise the catalog is searched.
        catalog_path: Catalog root. Defaults to ``CATALOG_DIR``.

    Raises:
        FileNotFoundError: If no series exists for *symbol*/*schema*.
        ValueError: If *venue* is omitted and the symbol exists at several venues.
    """
    if venue is not None:
        return bar_type(symbol, venue, schema)
    venues = [v for s, v in list_bar_series(schema, catalog_path) if s == symbol]
    if not venues:
        raise FileNotFoundError(
            f"No {schema} bars for {symbol!r} in the catalog. Run `algo data sync` first."
        )
    if len(venues) > 1:
        raise ValueError(
            f"{symbol!r} has {schema} bars at several venues ({', '.join(venues)}); "
            "pass venue= to choose one."
        )
    return bar_type(symbol, venues[0], schema)


def _to_ns(value: str | dt.date | dt.datetime | int | None) -> int | None:
    """Convert a date, datetime, ISO string or ns integer to UNIX nanoseconds (UTC)."""
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        value = dt.datetime.fromisoformat(text) if "T" in text else dt.date.fromisoformat(text)
    if not isinstance(value, dt.datetime):
        value = dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return int(value.timestamp()) * 1_000_000_000 + value.microsecond * 1_000


def load_bars(
    symbol: str,
    schema: str,
    *,
    venue: str | None = None,
    start: str | dt.date | dt.datetime | int | None = None,
    end: str | dt.date | dt.datetime | int | None = None,
    catalog_path: Path | None = None,
) -> pl.DataFrame:
    """Load a bar series from the catalog as a polars DataFrame.

    Args:
        symbol: Repo symbol (e.g. ``"ES.c.0"``).
        schema: Bar schema (e.g. ``"ohlcv-1d"``).
        venue: Venue, if the symbol is stored at more than one.
        start: Optional inclusive lower bound (on ``ts_init``, i.e. bar close).
        end: Optional inclusive upper bound (on ``ts_init``).
        catalog_path: Catalog root. Defaults to ``CATALOG_DIR``.

    Returns:
        Columns :data:`BAR_COLUMNS`, sorted by ``ts_event``. Prices and volume are
        ``Float64``; ``ts_event`` (bar open) and ``ts_init`` (bar close) are
        ``Datetime("ns", "UTC")``.

    Raises:
        FileNotFoundError: If the catalog or the series does not exist.
    """
    root = catalog_path if catalog_path is not None else CATALOG_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"No catalog at {root}. Run `algo data sync` first.")
    bt = resolve_bar_type(symbol, schema, venue=venue, catalog_path=root)
    payload = ParquetDataCatalog(str(root)).query_data_arrow_bytes(
        NautilusDataType.Bar, [bt], start=_to_ns(start), end=_to_ns(end), display=True
    )
    df = pl.from_arrow(pa.ipc.open_stream(payload).read_all())
    assert isinstance(df, pl.DataFrame)  # noqa: S101 — from_arrow(Table) is always a DataFrame
    return (
        df.with_columns(pl.lit(symbol, dtype=pl.Utf8).alias("symbol"))
        .select(BAR_COLUMNS)
        .sort("ts_event")
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def infer_price_precision(
    df: pl.DataFrame, columns: Sequence[str] = ("open", "high", "low", "close")
) -> int:
    """Smallest number of decimals that represents every price in *df* exactly.

    Used when converting provider frames that carry no instrument definition.
    Prices are floats, so "exact" means within ``1e-6`` of an integer once scaled.
    Capped at 9 decimals.
    """
    values = (
        np.concatenate([df[c].drop_nulls().to_numpy() for c in columns])
        if df.height
        else np.array([])
    )
    for precision in range(_MAX_INFERRED_PRECISION + 1):
        scaled = values * 10**precision
        if np.all(np.abs(scaled - np.round(scaled)) < 1e-6):
            return precision
    return _MAX_INFERRED_PRECISION


def _exact_decimals(values: np.ndarray, precision: int) -> pa.Array:
    """Round floats to *precision* decimals and encode them exactly as ``decimal128(38, 16)``.

    A direct float -> decimal cast keeps binary error (``100.05`` becomes
    ``100.0499999999999972``); going through scaled integers does not.
    """
    scaled = np.round(values * 10**precision).astype(np.int64)
    return pa.array([Decimal(int(v)).scaleb(-precision) for v in scaled], type=_DECIMAL)


def frame_to_bars(
    df: pl.DataFrame,
    *,
    bar_type: str,
    price_precision: int,
    interval_ns: int,
    size_precision: int = 0,
) -> list[Bar]:
    """Convert a provider bar frame stamped at the bar **open** into Nautilus bars.

    Args:
        df: Columns ``ts_event`` (bar open, datetime or ns integers) and
            ``open``, ``high``, ``low``, ``close``, ``volume``.
        bar_type: Target ``BarType`` string.
        price_precision: Decimals to round prices to.
        interval_ns: Bar length; ``ts_init = ts_event + interval_ns`` (bar close).
        size_precision: Decimals to round volume to.

    Returns:
        Bars sorted by ``ts_init``.
    """
    if df.height == 0:
        return []
    df = df.sort("ts_event")
    ts_event = df["ts_event"]
    if ts_event.dtype != pl.Int64:
        ts_event = ts_event.dt.cast_time_unit("ns").dt.epoch("ns")
    ts_open = ts_event.to_numpy().astype(np.uint64)

    columns = {
        col: _exact_decimals(df[col].cast(pl.Float64).to_numpy(), price_precision)
        for col in ("open", "high", "low", "close")
    }
    columns["volume"] = _exact_decimals(df["volume"].cast(pl.Float64).to_numpy(), size_precision)
    columns["ts_event"] = pa.array(ts_open, pa.uint64())
    columns["ts_init"] = pa.array(ts_open + np.uint64(interval_ns), pa.uint64())
    batch = pa.RecordBatch.from_pydict(columns)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    wrangler = BarDataWrangler(bar_type, price_precision, size_precision)
    return wrangler.process_record_batch_bytes(sink.getvalue().to_pybytes())


def missing_ranges(
    catalog: ParquetDataCatalog,
    data_type: NautilusDataType,
    identifier: str,
    start: int,
    end: int,
    *,
    interval_ns: int = 0,
) -> list[tuple[int, int]]:
    """Half-open ``[from, to)`` ns ranges of ``[start, end)`` still to fetch.

    Coverage is treated as one contiguous block from the first to the last stored
    ``ts_init`` (interior gaps are not detected). For bars, pass *interval_ns* so
    ranges are expressed in bar-open time: the head range stops at the first
    stored bar's open and the tail range starts at the last stored bar's close.
    For ticks (``interval_ns=0``) the tail starts one nanosecond after the last.

    Args:
        catalog: Catalog to inspect.
        data_type: Stored family (e.g. ``NautilusDataType.Bar``).
        identifier: Bar type or instrument ID string.
        start: Requested inclusive start, UNIX ns.
        end: Requested exclusive end, UNIX ns.
        interval_ns: Bar length, or 0 for tick data.
    """
    first = catalog.query_first_timestamp(data_type, identifier)
    last = catalog.query_last_timestamp(data_type, identifier)
    if first is None or last is None:
        return [(start, end)] if start < end else []
    ranges = []
    head_end = first - interval_ns
    if start < head_end:
        ranges.append((start, min(head_end, end)))
    tail_start = last if interval_ns else last + 1
    if max(tail_start, start) < end:
        ranges.append((max(tail_start, start), end))
    return ranges


def chunk_range(start: int, end: int, step_ns: int) -> list[tuple[int, int]]:
    """Split ``[start, end)`` into consecutive windows of at most *step_ns*."""
    return [(lo, min(lo + step_ns, end)) for lo in range(start, end, step_ns)]


def write_bars(catalog: ParquetDataCatalog, bars: Sequence[Bar]) -> int:
    """Write the part of *bars* not already covered by the catalog.

    Bars strictly before the first stored ``ts_init`` and strictly after the last
    one are written (as separate files, so each write is disjoint). Bars inside
    the stored range are skipped.

    Args:
        catalog: Target catalog.
        bars: Bars of a **single** bar type, in any order.

    Returns:
        Number of bars written.
    """
    if not bars:
        return 0
    types = {str(b.bar_type) for b in bars}
    if len(types) != 1:
        raise ValueError(f"write_bars needs bars of one bar type, got {sorted(types)}.")
    (bt,) = types
    ordered = sorted(bars, key=lambda b: b.ts_init)

    first = catalog.query_first_timestamp(NautilusDataType.Bar, bt)
    last = catalog.query_last_timestamp(NautilusDataType.Bar, bt)
    if first is None or last is None:
        chunks = [ordered]
    else:
        chunks = [
            [b for b in ordered if b.ts_init < first],
            [b for b in ordered if b.ts_init > last],
        ]
    written = 0
    for chunk in chunks:
        if chunk:
            catalog.write_bars(chunk)
            written += len(chunk)
    return written


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


def equity_instrument(symbol: str, venue: str, *, price_precision: int = 4) -> Equity:
    """A USD equity definition for bar storage and backtesting.

    The default precision is 4 decimals rather than the 0.01 trading tick
    because split/dividend-adjusted historical prices are not on the tick grid.
    """
    increment = Decimal(1).scaleb(-price_precision)
    return Equity(
        InstrumentId.from_str(instrument_id(symbol, venue)),
        Symbol(symbol),
        Currency.from_str("USD"),
        price_precision,
        Price.from_str(f"{increment:f}"),
        0,
        0,
    )


def continuous_instrument(contract: FuturesContract, symbol: str) -> FuturesContract:
    """Clone an outright *contract* as the instrument for a continuous series.

    Databento continuous symbols (``ES.c.0``) map to a different outright over
    time, so they have no definition of their own. Backtests on the continuous
    bar series still need an instrument with the right tick size and
    multiplier; this copies them from a real contract under the continuous ID,
    with no activation or expiry.
    """
    spec = contract.to_dict()
    venue = str(contract.id.venue)
    spec.update(
        id=instrument_id(symbol, venue),
        raw_symbol=symbol,
        activation_ns=0,
        expiration_ns=2**63 - 1,
    )
    return FuturesContract.from_dict(spec)


def write_instruments(catalog: ParquetDataCatalog, instruments: Sequence[object]) -> None:
    """Write instrument definitions, skipping IDs the catalog already holds."""
    existing = {str(i.id) for i in catalog.instruments()}
    new = [i for i in instruments if str(i.id) not in existing]  # type: ignore[attr-defined]
    if new:
        catalog.write_instruments(new)
