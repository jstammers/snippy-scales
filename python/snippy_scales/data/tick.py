"""Databento tick ingestion — event-level data persisted day-by-day.

Why this is separate from :mod:`snippy_scales.data.ingest`
----------------------------------------------------------
Bar data is small: one Parquet file per ``(symbol, schema)`` holds a decade of
daily bars, and "what do I already have?" can be answered by reading the file
and taking ``max(ts_event)``.  Event-level schemas (``trades``, ``mbo``, ...)
break both assumptions — a single ES session can be hundreds of millions of
messages — so this module uses a different storage model.

Storage layout
--------------
::

    data/raw/futures/ES.c.0/trades/
      _dbn/2026-08-03.dbn.zst        ← lossless raw feed as returned by the API
      _empty/2026-08-02.empty        ← day confirmed to contain no records
      date=2026-08-03/data.parquet   ← hive-partitioned columnar

The ``futures/`` segment is the symbol's
:data:`~snippy_scales.data.config.InstrumentType` classification — see
:mod:`snippy_scales.data.ingest` for why it's kept separate from the
free-form ``asset_classes`` config grouping.

The raw ``.dbn.zst`` is kept so the Parquet can be regenerated (different price
type, different columns) without ever paying Databento a second time.

Coverage semantics
------------------
A day is **covered** when either its Parquet partition or its empty-marker
exists.  :func:`missing_days` is the difference between the requested calendar
range and the covered set, which means — unlike the high-water-mark logic used
for bars — interior gaps are detected and backfilling below an existing start
works.  Downloads only ever touch missing days, so re-running an identical
request costs nothing and downloads nothing.

Both the DBN and the Parquet are written to temporary siblings and atomically
renamed into place, so an interrupted run leaves no partial file that would be
mistaken for a covered day.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from snippy_scales.data.batch import run_batch_job
from snippy_scales.data.paths import RAW_DIR

if TYPE_CHECKING:
    import databento as db

    from snippy_scales.data.config import VALID_STYPES, DownloadMethod

logger = logging.getLogger(__name__)

#: Sub-directory holding the raw DBN payload for each downloaded day.
_DBN_DIR = "_dbn"

#: Sub-directory holding zero-byte markers for days confirmed to have no records.
_EMPTY_DIR = "_empty"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _tick_root(symbol: str, schema: str, output_dir: Path, instrument_type: str) -> Path:
    """Return the root directory for a symbol/schema tick store.

    Args:
        symbol: Databento symbol string (e.g. ``"ES.c.0"``).
        schema: Event-level schema name (e.g. ``"trades"``).
        output_dir: Root directory for raw data.
        instrument_type: Closed top-level storage classification — see
            :data:`~snippy_scales.data.config.InstrumentType`.

    Returns:
        A path like ``<output_dir>/futures/ES.c.0/trades``.
    """
    safe_symbol = symbol.replace("/", "_")
    return output_dir / instrument_type / safe_symbol / schema


def _day_parquet(root: Path, day: dt.date) -> Path:
    """Return the hive-partitioned Parquet path for *day*."""
    return root / f"date={day.isoformat()}" / "data.parquet"


def _day_dbn(root: Path, day: dt.date) -> Path:
    """Return the raw DBN path for *day*."""
    return root / _DBN_DIR / f"{day.isoformat()}.dbn.zst"


def _day_empty_marker(root: Path, day: dt.date) -> Path:
    """Return the empty-day marker path for *day*."""
    return root / _EMPTY_DIR / f"{day.isoformat()}.empty"


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _parse_day(value: str | dt.date) -> dt.date:
    """Coerce an ISO date string or :class:`datetime.date` to a ``date``."""
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)


def covered_days(
    symbol: str, schema: str, output_dir: Path = RAW_DIR, *, instrument_type: str
) -> set[dt.date]:
    """Return the set of days already present in the local tick store.

    Coverage is derived from the filesystem rather than a manifest, so there is
    no separate index that can drift out of sync with the data.

    Args:
        symbol: Instrument symbol (e.g. ``"ES.c.0"``).
        schema: Event-level schema name (e.g. ``"trades"``).
        output_dir: Root directory for raw data.
        instrument_type: Closed top-level storage classification — see
            :data:`~snippy_scales.data.config.InstrumentType`.

    Returns:
        Set of dates for which data (or a confirmed-empty marker) exists.
    """
    root = _tick_root(symbol, schema, output_dir, instrument_type)
    if not root.exists():
        return set()

    days: set[dt.date] = set()

    for partition in root.glob("date=*"):
        if not (partition / "data.parquet").exists():
            continue  # partial/interrupted download — not covered
        try:
            days.add(dt.date.fromisoformat(partition.name.removeprefix("date=")))
        except ValueError:
            logger.warning("Ignoring unparseable partition directory: %s", partition)

    for marker in root.glob(f"{_EMPTY_DIR}/*.empty"):
        try:
            days.add(dt.date.fromisoformat(marker.stem))
        except ValueError:
            logger.warning("Ignoring unparseable empty marker: %s", marker)

    return days


def missing_days(
    symbol: str,
    schema: str,
    start: str | dt.date,
    end: str | dt.date,
    output_dir: Path = RAW_DIR,
    *,
    instrument_type: str,
) -> list[dt.date]:
    """Return the sorted days in ``[start, end)`` not yet present locally.

    This is the primitive that guarantees historical data is never
    re-downloaded: every billed request in this module is driven by its result.

    Args:
        symbol: Instrument symbol (e.g. ``"ES.c.0"``).
        schema: Event-level schema name (e.g. ``"trades"``).
        start: Inclusive start date (``YYYY-MM-DD`` or ``date``).
        end: **Exclusive** end date (``YYYY-MM-DD`` or ``date``).
        output_dir: Root directory for raw data.
        instrument_type: Closed top-level storage classification — see
            :data:`~snippy_scales.data.config.InstrumentType`.

    Returns:
        Chronologically sorted list of uncovered dates.  Empty if the range is
        already fully covered or if ``end <= start``.
    """
    start_date = _parse_day(start)
    end_date = _parse_day(end)
    covered = covered_days(symbol, schema, output_dir, instrument_type=instrument_type)

    days: list[dt.date] = []
    day = start_date
    while day < end_date:
        if day not in covered:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def contiguous_ranges(days: list[dt.date]) -> list[tuple[dt.date, dt.date]]:
    """Group consecutive dates into half-open ``(start, end)`` ranges.

    Used so cost estimation issues one metadata call per run of missing days
    rather than one per day.

    Args:
        days: Chronologically sorted dates (as returned by :func:`missing_days`).

    Returns:
        List of ``(start_inclusive, end_exclusive)`` tuples.

    Examples:
        >>> import datetime as dt
        >>> contiguous_ranges([dt.date(2026, 1, 1), dt.date(2026, 1, 2), dt.date(2026, 1, 5)])
        [(datetime.date(2026, 1, 1), datetime.date(2026, 1, 3)), \
(datetime.date(2026, 1, 5), datetime.date(2026, 1, 6))]
    """
    if not days:
        return []

    ranges: list[tuple[dt.date, dt.date]] = []
    run_start = days[0]
    previous = days[0]

    for day in days[1:]:
        if day == previous + dt.timedelta(days=1):
            previous = day
            continue
        ranges.append((run_start, previous + dt.timedelta(days=1)))
        run_start = day
        previous = day

    ranges.append((run_start, previous + dt.timedelta(days=1)))
    return ranges


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickCostEstimate:
    """Estimated cost and size of a tick download.

    Attributes:
        symbol: Instrument symbol the estimate applies to.
        schema: Event-level schema name.
        cost_usd: Estimated Databento charge in US dollars.
        billable_bytes: Estimated billable size of the response, in bytes.
        missing_days: Days that would actually be downloaded.
        cached_days: Days in the requested range already held locally.
    """

    symbol: str
    schema: str
    cost_usd: float
    billable_bytes: int
    missing_days: list[dt.date]
    cached_days: int

    @property
    def billable_gb(self) -> float:
        """Estimated billable size in gigabytes (1 GB = 1e9 bytes)."""
        return self.billable_bytes / 1e9


def estimate_tick_cost(
    *,
    dataset: str,
    symbol: str,
    schema: str,
    start: str | dt.date,
    end: str | dt.date,
    instrument_type: str,
    output_dir: Path = RAW_DIR,
    stype_in: VALID_STYPES = "raw_symbol",
) -> TickCostEstimate:
    """Estimate the cost of downloading a tick range, excluding cached days.

    Makes no API call at all when the requested range is already fully covered.

    Args:
        dataset: Databento dataset code (e.g. ``"GLBX.MDP3"``).
        symbol: Instrument symbol (e.g. ``"ES.c.0"``).
        schema: Event-level schema name (e.g. ``"trades"``).
        start: Inclusive start date (``YYYY-MM-DD``).
        end: **Exclusive** end date (``YYYY-MM-DD``).
        instrument_type: Closed top-level storage classification — see
            :data:`~snippy_scales.data.config.InstrumentType`.
        output_dir: Root directory for raw data.
        stype_in: Databento symbology type for the request.

    Returns:
        A :class:`TickCostEstimate` covering only the missing days.

    Raises:
        RuntimeError: If the Databento metadata API call fails.
    """
    import databento as db  # noqa: PLC0415 — optional dep

    start_date = _parse_day(start)
    end_date = _parse_day(end)
    total_days = max((end_date - start_date).days, 0)

    missing = missing_days(
        symbol, schema, start_date, end_date, output_dir, instrument_type=instrument_type
    )
    if not missing:
        return TickCostEstimate(
            symbol=symbol,
            schema=schema,
            cost_usd=0.0,
            billable_bytes=0,
            missing_days=[],
            cached_days=total_days,
        )

    client = db.Historical()
    cost = 0.0
    size = 0

    for range_start, range_end in contiguous_ranges(missing):
        cost += client.metadata.get_cost(
            dataset=dataset,
            start=range_start.isoformat(),
            end=range_end.isoformat(),
            symbols=[symbol],
            schema=schema,
            stype_in=stype_in,
        )
        size += client.metadata.get_billable_size(
            dataset=dataset,
            start=range_start.isoformat(),
            end=range_end.isoformat(),
            symbols=[symbol],
            schema=schema,
            stype_in=stype_in,
        )

    return TickCostEstimate(
        symbol=symbol,
        schema=schema,
        cost_usd=cost,
        billable_bytes=size,
        missing_days=missing,
        cached_days=total_days - len(missing),
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _mark_day_empty(root: Path, day: dt.date, *, symbol: str, schema: str) -> None:
    """Write the empty-day marker for *day* (no records found)."""
    marker = _day_empty_marker(root, day)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    logger.info("[%s] %s %s — no records (marked empty).", symbol, schema, day)


def _install_day(
    store: db.DBNStore,
    *,
    symbol: str,
    schema: str,
    day: dt.date,
    root: Path,
) -> Path | None:
    """Convert an already-open :class:`DBNStore` to *day*'s Parquet partition.

    Shared by the streaming and batch download paths so a day's Parquet file
    is byte-for-byte schema-identical no matter which delivery mechanism
    fetched it — both always go through ``DBNStore.to_parquet()``, never
    pandas/polars, which is what keeps every partition under
    ``date=*/data.parquet`` concatenable via :func:`load_ticks`.

    Args:
        store: A :class:`databento.DBNStore` opened from a single day's DBN.
        symbol: Instrument symbol (for logging only).
        schema: Event-level schema name.
        day: The calendar day *store* covers.
        root: Tick-store root for this symbol/schema.

    Returns:
        Path to the written Parquet partition, or ``None`` when the day
        contained no records (an empty marker is written instead).
    """
    parquet_path = _day_parquet(root, day)
    tmp_parquet = parquet_path.parent.with_name(f".tmp-{day.isoformat()}.parquet")
    tmp_parquet.unlink(missing_ok=True)
    store.to_parquet(tmp_parquet, schema=schema)

    # ``to_parquet`` writes nothing at all when the store holds no records, so a
    # missing temp file is how we detect a non-trading day.
    if not tmp_parquet.exists():
        _mark_day_empty(root, day, symbol=symbol, schema=schema)
        return None

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_parquet, parquet_path)
    return parquet_path


def _download_day(
    client: db.Historical,
    *,
    dataset: str,
    symbol: str,
    schema: str,
    day: dt.date,
    root: Path,
    stype_in: VALID_STYPES,
) -> Path | None:
    """Download, convert, and atomically install a single day of tick data.

    Uses the Historical Streaming API (one billed call per day). See
    :func:`_download_batch_range` for the batch equivalent.

    Args:
        client: A ``databento.Historical`` client.
        dataset: Databento dataset code.
        symbol: Instrument symbol.
        schema: Event-level schema name.
        day: The calendar day to fetch (requested as ``[day, day+1)``).
        root: Tick-store root for this symbol/schema.
        stype_in: Databento symbology type.

    Returns:
        Path to the written Parquet partition, or ``None`` when the day
        contained no records (an empty marker is written instead).
    """
    import databento as db  # noqa: PLC0415 — optional dep

    next_day = day + dt.timedelta(days=1)

    dbn_path = _day_dbn(root, day)
    dbn_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Stream the raw feed straight to disk (never held in memory) ---
    tmp_dbn = dbn_path.with_name(f".tmp-{dbn_path.name}")
    tmp_dbn.unlink(missing_ok=True)
    client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema=schema,
        start=day.isoformat(),
        end=next_day.isoformat(),
        stype_in=stype_in,
        path=str(tmp_dbn),
    )
    os.replace(tmp_dbn, dbn_path)

    store = db.DBNStore.from_file(dbn_path)
    return _install_day(store, symbol=symbol, schema=schema, day=day, root=root)


def _download_batch_range(
    client: db.Historical,
    *,
    dataset: str,
    symbol: str,
    schema: str,
    range_start: dt.date,
    range_end: dt.date,
    root: Path,
    stype_in: VALID_STYPES,
) -> list[Path]:
    """Download and install every day in ``[range_start, range_end)`` via one batch job.

    Submits a single ``split_duration="day"`` batch job for the whole
    contiguous range (one billed job instead of one billed call per day),
    then installs each returned per-day file exactly like
    :func:`_download_day` does (via :func:`_install_day`), and writes an
    empty marker for any requested day the job returned no file for.

    Args:
        client: A ``databento.Historical`` client.
        dataset: Databento dataset code.
        symbol: Instrument symbol.
        schema: Event-level schema name.
        range_start: Inclusive start of the contiguous missing range.
        range_end: Exclusive end of the contiguous missing range.
        root: Tick-store root for this symbol/schema.
        stype_in: Databento symbology type.

    Returns:
        Paths of the Parquet partitions written by this call. Days that
        turned out to be empty are not included.

    Raises:
        RuntimeError: If a returned file unexpectedly spans more than one
            UTC day (violates the ``split_duration="day"`` contract this
            function relies on) — raised rather than guessing how to
            re-partition it.
    """
    import databento as db  # noqa: PLC0415 — optional dep

    written: list[Path] = []
    covered: set[dt.date] = set()

    with tempfile.TemporaryDirectory(prefix="databento-batch-") as tmp_dir:
        files = run_batch_job(
            client,
            dataset=dataset,
            symbols=[symbol],
            schema=schema,
            start=range_start.isoformat(),
            end=range_end.isoformat(),
            stype_in=stype_in,
            split_duration="day",
            output_dir=Path(tmp_dir),
        )

        for downloaded in files:
            store = db.DBNStore.from_file(downloaded)
            day = store.start.date()
            if store.end is not None and store.end - store.start > dt.timedelta(days=1):
                raise RuntimeError(
                    f"Batch job file {downloaded.name} for {symbol}/{schema} spans "
                    f"{store.start} → {store.end}, more than one UTC day despite "
                    "split_duration='day' — refusing to guess how to re-partition it."
                )

            dbn_path = _day_dbn(root, day)
            dbn_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_dbn = dbn_path.with_name(f".tmp-{dbn_path.name}")
            tmp_dbn.unlink(missing_ok=True)
            shutil.copy2(downloaded, tmp_dbn)  # source is a throwaway temp download
            os.replace(tmp_dbn, dbn_path)

            installed = _install_day(
                db.DBNStore.from_file(dbn_path), symbol=symbol, schema=schema, day=day, root=root
            )
            covered.add(day)
            if installed is not None:
                written.append(installed)

    day = range_start
    while day < range_end:
        if day not in covered:
            _mark_day_empty(root, day, symbol=symbol, schema=schema)
        day += dt.timedelta(days=1)

    return written


def upsert_ticks(
    *,
    dataset: str,
    symbol: str,
    schema: str,
    start: str | dt.date,
    end: str | dt.date,
    instrument_type: str,
    output_dir: Path = RAW_DIR,
    stype_in: VALID_STYPES = "raw_symbol",
    download_method: DownloadMethod = "batch",
) -> list[Path]:
    """Download every day in ``[start, end)`` that is not already stored locally.

    Days already present are skipped without any API call.

    With ``download_method="streaming"``, each missing day is fetched in its
    own Historical Streaming API call and installed atomically, so an
    interrupted run can be resumed by re-issuing the identical command —
    already-completed days are not re-billed.

    With ``download_method="batch"`` (default), each *contiguous* run of
    missing days (see :func:`contiguous_ranges`) is fetched as a single
    Databento batch job — see :func:`_download_batch_range`.

    Args:
        dataset: Databento dataset code (e.g. ``"GLBX.MDP3"``).
        symbol: Instrument symbol (e.g. ``"ES.c.0"``).
        schema: Event-level schema name (e.g. ``"trades"``).
        start: Inclusive start date (``YYYY-MM-DD``).
        end: **Exclusive** end date (``YYYY-MM-DD``).
        instrument_type: Closed top-level storage classification — see
            :data:`~snippy_scales.data.config.InstrumentType`.
        output_dir: Root directory for raw data.
        stype_in: Databento symbology type for the request.
        download_method: ``"batch"`` (default) or ``"streaming"`` — see
            :data:`~snippy_scales.data.config.DownloadMethod`.

    Returns:
        Paths of the Parquet partitions written by this call.  Days that turned
        out to be empty, and days that were already cached, are not included.
    """
    import databento as db  # noqa: PLC0415 — optional dep

    root = _tick_root(symbol, schema, output_dir, instrument_type)
    missing = missing_days(symbol, schema, start, end, output_dir, instrument_type=instrument_type)

    if not missing:
        logger.info("[%s] %s already covers %s → %s. Nothing to do.", symbol, schema, start, end)
        return []

    logger.info("[%s] Fetching %s for %d missing day(s).", symbol, schema, len(missing))

    client = db.Historical()
    written: list[Path] = []

    if download_method == "streaming":
        for day in missing:
            try:
                path = _download_day(
                    client,
                    dataset=dataset,
                    symbol=symbol,
                    schema=schema,
                    day=day,
                    root=root,
                    stype_in=stype_in,
                )
            except Exception:
                logger.exception("[%s] Failed to ingest %s for %s — skipping.", symbol, schema, day)
                continue
            if path is not None:
                written.append(path)
    else:
        for range_start, range_end in contiguous_ranges(missing):
            try:
                written.extend(
                    _download_batch_range(
                        client,
                        dataset=dataset,
                        symbol=symbol,
                        schema=schema,
                        range_start=range_start,
                        range_end=range_end,
                        root=root,
                        stype_in=stype_in,
                    )
                )
            except Exception:
                logger.exception(
                    "[%s] Failed to ingest %s for %s → %s — skipping.",
                    symbol,
                    schema,
                    range_start,
                    range_end,
                )
                continue

    logger.info("[%s] %s: wrote %d day(s) of data.", symbol, schema, len(written))
    return written


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_ticks(
    symbol: str,
    schema: str,
    *,
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
    output_dir: Path = RAW_DIR,
    instrument_type: str | None = None,
) -> pl.LazyFrame:
    """Lazily scan a stored tick dataset.

    Returns a :class:`polars.LazyFrame` rather than a materialised frame — an
    MBO store is routinely larger than memory, so callers are expected to
    filter/aggregate before collecting.

    Args:
        symbol: Instrument symbol (e.g. ``"ES.c.0"``, or ``"ES.FUT"`` if
            ingested via Databento parent symbology).
        schema: Event-level schema name (e.g. ``"trades"``).
        start: Optional inclusive start date filter.
        end: Optional **exclusive** end date filter.
        output_dir: Root directory for raw data.
        instrument_type: Closed top-level storage classification — see
            :data:`~snippy_scales.data.config.InstrumentType`. When omitted,
            resolved by globbing ``output_dir/*/<symbol>/<schema>``; raises
            if the symbol is missing or exists under more than one
            classification.

    Returns:
        A lazy frame over the day partitions, with a ``date`` column supplied by
        the hive partitioning.

    Raises:
        FileNotFoundError: If no data exists for the given symbol and schema.
        ValueError: If *instrument_type* is omitted and the symbol/schema
            pair exists under more than one classification.
    """
    if instrument_type is not None:
        root = _tick_root(symbol, schema, output_dir, instrument_type)
    else:
        safe_symbol = symbol.replace("/", "_")
        matches = [
            p
            for p in output_dir.glob(f"*/{safe_symbol}/{schema}")
            if any(p.glob("date=*/data.parquet"))
        ]
        if len(matches) > 1:
            classes = ", ".join(m.parent.parent.name for m in matches)
            raise ValueError(
                f"symbol={symbol!r}, schema={schema!r} exists under more than one "
                f"instrument_type ({classes}) — pass instrument_type explicitly to "
                "disambiguate."
            )
        root = matches[0] if matches else _tick_root(symbol, schema, output_dir, "unclassified")
    if not root.exists() or not any(root.glob("date=*/data.parquet")):
        raise FileNotFoundError(
            f"No tick data found for symbol={symbol!r}, schema={schema!r} at {root}. "
            "Run `algo data ingest-config` first."
        )

    frame = pl.scan_parquet(root / "date=*" / "data.parquet", hive_partitioning=True)

    if start is not None:
        frame = frame.filter(pl.col("date") >= _parse_day(start))
    if end is not None:
        frame = frame.filter(pl.col("date") < _parse_day(end))

    return frame
