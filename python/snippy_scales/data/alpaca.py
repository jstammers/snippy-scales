"""Alpaca equities -> catalog loader.

NautilusTrader has no official Alpaca adapter (see
docs/architecture/nautilus_migration.md, decision D1), so this module is the one
place Alpaca is handled: it fetches bars with the official ``alpaca-py`` SDK,
throttles requests to the free tier's per-minute quota, converts them to
Nautilus bars and writes them to the catalog incrementally.

The SDK paginates and retries on 429/5xx itself, but it only reacts to a 429
after the fact. :class:`RateLimiter` wraps every HTTP GET so a multi-threaded
backfill stays under quota in the first place.

Storage conventions:

* Instruments are ``Equity`` at venue :data:`DEFAULT_VENUE` with 4-decimal
  prices (adjusted history is off the 0.01 tick grid).
* Alpaca stamps bars at the open; daily bars are normalised to 00:00 UTC on the
  session date. ``ts_init`` is set to the bar close (open + bar length).

Adjusted prices
---------------
With ``adjustment`` other than ``raw``, Alpaca re-bases the *whole* history
whenever a split or dividend occurs. Bars stored before that event are then on
a different basis from bars fetched after it, and appending would put a false
jump at the seam. So every incremental fetch overlaps the stored series by one
bar and compares it: if the stored and fresh closes disagree, the symbol fails
with :class:`AdjustmentDriftError` and nothing is appended. Setting
``rebuild_adjusted=True`` instead re-downloads the full series and replaces the
stored one — the old bars are removed only after the new download is in hand
and covers at least the stored range.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import polars as pl
from nautilus_trader.model import NautilusDataType

from snippy_scales.data import catalog as cat

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from alpaca.data.historical.stock import StockHistoricalDataClient
    from nautilus_trader.persistence import ParquetDataCatalog

logger = logging.getLogger(__name__)

#: Venue used in instrument IDs for Alpaca bars (consolidated SIP/IEX tape).
DEFAULT_VENUE = "ALPACA"

#: Price decimals stored for Alpaca bars.
PRICE_PRECISION = 4

#: Free-tier historical data withholds the most recent 15 minutes.
_RECENT_DATA_EMBARGO = dt.timedelta(minutes=16)

#: Earliest date Alpaca's historical stock data covers.
HISTORY_START = dt.date(2016, 1, 1)

#: A session's bars are final once its UTC day and the extended-hours session
#: (until 01:00 UTC in winter) have ended, plus the free-tier embargo.
_SESSION_COMPLETE_LAG = dt.timedelta(hours=1) + _RECENT_DATA_EMBARGO

#: Bar schema -> alpaca-py TimeFrame(amount, unit) args. No 1-second bars on Alpaca.
_TIMEFRAME_BY_SCHEMA: dict[str, tuple[int, str]] = {
    "ohlcv-1m": (1, "Minute"),
    "ohlcv-1h": (1, "Hour"),
    "ohlcv-1d": (1, "Day"),
}

#: Schemas :class:`AlpacaLoader` supports.
SUPPORTED_BAR_SCHEMAS: frozenset[str] = frozenset(_TIMEFRAME_BY_SCHEMA)

_EMPTY_SCHEMA = {
    "ts_event": pl.Datetime("ns", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def weekday_sessions(first: dt.date, last: dt.date) -> list[dt.date]:
    """Mon–Fri dates in ``[first, last]`` — an offline stand-in for the NYSE calendar."""
    days = (first + dt.timedelta(days=i) for i in range((last - first).days + 1))
    return [d for d in days if d.weekday() < 5]  # noqa: PLR2004


def resolve_window(
    start: int | None,
    end: int | None,
    sessions: Callable[[dt.date, dt.date], list[dt.date]],
    *,
    now: dt.datetime | None = None,
) -> tuple[int, int]:
    """The ``[start, end)`` to sync, in UNIX ns at 00:00 UTC.

    Starts on the first session on or after *start* (default and floor:
    :data:`HISTORY_START`) and ends the day after the last **completed** session,
    never later than *end*. A session is complete once
    :data:`_SESSION_COMPLETE_LAG` past the end of its UTC day. Returns an empty
    window (start == end) when no completed session falls inside.

    Args:
        start: Requested start (UNIX ns), or None.
        end: Requested exclusive end (UNIX ns), or None.
        sessions: ``(first, last) -> session dates`` in that inclusive range.
        now: Current time (injectable for tests).
    """
    first = max(
        _ns_to_datetime(start).date() if start is not None else HISTORY_START, HISTORY_START
    )
    cutoff = ((now or dt.datetime.now(dt.UTC)) - _SESSION_COMPLETE_LAG).date()
    if end is not None:
        cutoff = min(cutoff, _ns_to_datetime(end).date())
    last = cutoff - dt.timedelta(days=1)
    days = sessions(first, last) if first <= last else []
    if not days:
        return _date_ns(first), _date_ns(first)
    return _date_ns(days[0]), _date_ns(days[-1] + dt.timedelta(days=1))


class AlpacaCredentialsError(RuntimeError):
    """Raised when Alpaca API credentials are not available."""


class AdjustmentDriftError(RuntimeError):
    """Stored adjusted bars no longer match Alpaca's current adjustment basis."""


class RateLimiter:
    """Spaces :meth:`acquire` calls evenly across a per-minute budget.

    Each call reserves the next slot under a lock, then sleeps outside it, so
    threads queue without serialising on the sleep while no more than
    ``calls_per_minute`` calls start in any 60-second window.

    Args:
        calls_per_minute: Maximum calls allowed to start per 60 seconds.
        clock: Monotonic time source (injectable for tests).
        sleep: Sleep function (injectable for tests).
    """

    def __init__(
        self,
        calls_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if calls_per_minute <= 0:
            raise ValueError(f"calls_per_minute must be positive, got {calls_per_minute}.")
        self._interval = 60.0 / calls_per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_available = clock()

    def acquire(self) -> None:
        """Block the calling thread until its reserved slot arrives."""
        with self._lock:
            now = self._clock()
            start_at = max(self._next_available, now)
            self._next_available = start_at + self._interval
        wait = start_at - now
        if wait > 0:
            self._sleep(wait)


def _make_rate_limited_client(
    *,
    api_key: str,
    secret_key: str,
    rate_limiter: RateLimiter,
    retry_attempts: int,
    retry_wait_seconds: int,
) -> StockHistoricalDataClient:
    """Build a ``StockHistoricalDataClient`` whose every HTTP GET is rate-limited.

    ``alpaca-py`` sends every page of every request through ``RESTClient.get()``,
    so overriding it throttles pagination as well as calls across threads.
    """
    from alpaca.data.historical.stock import StockHistoricalDataClient  # noqa: PLC0415

    class _RateLimitedStockClient(StockHistoricalDataClient):
        def get(self, path: str, data: dict | str | None = None, **kwargs: object) -> object:
            rate_limiter.acquire()
            return super().get(path, data=data, **kwargs)

    client = _RateLimitedStockClient(api_key=api_key, secret_key=secret_key)
    # StockHistoricalDataClient.__init__ doesn't forward retry settings to RESTClient.
    client._retry = retry_attempts  # noqa: SLF001
    client._retry_wait = retry_wait_seconds  # noqa: SLF001
    client._retry_codes = [429, 500, 502, 503, 504]  # noqa: SLF001
    return client


def _ns_to_datetime(ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.UTC)


def _date_ns(day: dt.date) -> int:
    """00:00 UTC on *day*, in UNIX ns."""
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()) * 10**9


@dataclass
class AlpacaLoader:
    """Fetch Alpaca stock bars and keep them up to date in the catalog.

    Attributes:
        feed: ``"sip"`` (all US exchanges; free historical tier) or ``"iex"``.
        adjustment: Corporate-action adjustment: ``raw``, ``split``,
            ``dividend`` or ``all``.
        rate_limit_per_min: Historical API calls per minute (free tier: 200).
        max_workers: Symbols fetched concurrently. The rate limiter, not the
            thread count, bounds request throughput.
        venue: Venue in instrument IDs.
        retry_attempts: Passed to the SDK's built-in 429/5xx retry.
        retry_wait_seconds: Wait between SDK retries.
        api_key: Falls back to ``ALPACA_API_KEY``.
        secret_key: Falls back to ``ALPACA_SECRET_KEY``.
        rebuild_adjusted: On adjustment drift, re-download and replace the
            series instead of failing the symbol (see module docstring).

    Raises:
        AlpacaCredentialsError: If no key/secret is available.
    """

    feed: Literal["iex", "sip"] = "sip"
    adjustment: Literal["raw", "split", "dividend", "all"] = "all"
    rate_limit_per_min: int = 190
    max_workers: int = 4
    venue: str = DEFAULT_VENUE
    retry_attempts: int = 5
    retry_wait_seconds: int = 5
    api_key: str | None = None
    secret_key: str | None = None
    rebuild_adjusted: bool = False
    #: ``(first, last) -> NYSE session dates`` in that inclusive range. Defaults
    #: to Alpaca's market calendar (free); injectable for tests.
    calendar: Callable[[dt.date, dt.date], list[dt.date]] | None = None
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("ALPACA_API_KEY")
        secret = self.secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise AlpacaCredentialsError(
                "Alpaca requires an API key/secret — set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY, or pass api_key=/secret_key= explicitly."
            )
        self._rate_limiter = RateLimiter(self.rate_limit_per_min)
        self._client = _make_rate_limited_client(
            api_key=key,
            secret_key=secret,
            rate_limiter=self._rate_limiter,
            retry_attempts=self.retry_attempts,
            retry_wait_seconds=self.retry_wait_seconds,
        )

    # -- window ----------------------------------------------------------

    def _sessions(self, first: dt.date, last: dt.date) -> list[dt.date]:
        if self.calendar is not None:
            return self.calendar(first, last)
        from alpaca.trading.client import TradingClient  # noqa: PLC0415
        from alpaca.trading.requests import GetCalendarRequest  # noqa: PLC0415

        key = self.api_key or os.environ["ALPACA_API_KEY"]
        secret = self.secret_key or os.environ["ALPACA_SECRET_KEY"]
        days = TradingClient(key, secret, paper=True).get_calendar(
            GetCalendarRequest(start=first, end=last)
        )
        return [d.date for d in days]  # type: ignore[union-attr]

    def resolve_window(
        self, start: int | None, end: int | None, *, now: dt.datetime | None = None
    ) -> tuple[int, int]:
        """:func:`resolve_window` using Alpaca's market calendar."""
        return resolve_window(start, end, self._sessions, now=now)

    # -- fetching ---------------------------------------------------------

    def fetch_bars(
        self, symbol: str, schema: str, start: dt.datetime, end: dt.datetime
    ) -> pl.DataFrame:
        """Fetch bars for *symbol* over ``[start, end)``.

        Returns:
            Columns ``ts_event`` (bar open, ``Datetime("ns", "UTC")``), ``open``,
            ``high``, ``low``, ``close``, ``volume``; sorted. Empty when the
            embargo-capped range is empty.

        Raises:
            ValueError: If *schema* has no Alpaca timeframe (e.g. ``"ohlcv-1s"``).
        """
        if schema not in _TIMEFRAME_BY_SCHEMA:
            raise ValueError(
                f"Alpaca does not support schema {schema!r}. "
                f"Supported: {sorted(_TIMEFRAME_BY_SCHEMA)}."
            )
        end = min(end, dt.datetime.now(dt.UTC) - _RECENT_DATA_EMBARGO)
        if start >= end:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)
        return self._request_bars(symbol, schema, start, end)

    def _request_bars(
        self, symbol: str, schema: str, start: dt.datetime, end: dt.datetime
    ) -> pl.DataFrame:
        from alpaca.common.enums import Sort  # noqa: PLC0415
        from alpaca.data.enums import Adjustment, DataFeed  # noqa: PLC0415
        from alpaca.data.models.bars import BarSet  # noqa: PLC0415
        from alpaca.data.requests import StockBarsRequest  # noqa: PLC0415
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: PLC0415

        amount, unit_name = _TIMEFRAME_BY_SCHEMA[schema]
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(amount, TimeFrameUnit[unit_name]),
            start=start,
            end=end,
            feed=DataFeed(self.feed),
            adjustment=Adjustment(self.adjustment),
            sort=Sort.ASC,
        )
        bar_set = self._client.get_stock_bars(request)
        assert isinstance(bar_set, BarSet)  # noqa: S101 — raw_data is never set
        pdf = bar_set.df
        if pdf.empty:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        df = pl.from_pandas(pdf.reset_index().rename(columns={"timestamp": "ts_event"}))
        ts = pl.col("ts_event").dt.convert_time_zone("UTC").dt.cast_time_unit("ns")
        if schema == "ohlcv-1d":
            ts = ts.dt.date().cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
        return (
            df.with_columns(ts.alias("ts_event"))
            .select(
                "ts_event",
                *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")),
            )
            .sort("ts_event")
        )

    # -- syncing ----------------------------------------------------------

    def sync_symbol(
        self, catalog: ParquetDataCatalog, symbol: str, schema: str, start: int, end: int
    ) -> int:
        """Fetch and store the bars missing for *symbol* in ``[start, end)`` (UNIX ns).

        Raises:
            AdjustmentDriftError: If the stored series is on a stale adjustment
                basis and ``rebuild_adjusted`` is False.
        """
        bt = cat.bar_type(symbol, self.venue, schema)
        interval = cat.bar_interval_ns(schema)
        with self._write_lock:
            ranges = cat.missing_ranges(
                catalog, NautilusDataType.Bar, bt, start, end, interval_ns=interval
            )
            first = catalog.query_first_timestamp(NautilusDataType.Bar, bt)
            last = catalog.query_last_timestamp(NautilusDataType.Bar, bt)
        check_seams = self.adjustment != "raw" and first is not None and last is not None

        frames = []
        for lo, hi in ranges:
            seam = None
            if check_seams:
                # Overlap the stored series by one bar so its basis can be compared.
                is_head = hi + interval <= first  # type: ignore[operator]
                seam = first if is_head else last
                lo, hi = (lo, hi + interval) if is_head else (lo - interval, hi)
            frame = self.fetch_bars(symbol, schema, _ns_to_datetime(lo), _ns_to_datetime(hi))
            if seam is not None and self._has_drifted(catalog, bt, frame, seam, interval):
                if not self.rebuild_adjusted:
                    raise AdjustmentDriftError(
                        f"{symbol} {schema}: stored bars are on an older adjustment basis "
                        "(a split or dividend since the last sync). Nothing appended; "
                        "re-run with --rebuild-adjusted to re-download the full series."
                    )
                return self._rebuild(catalog, symbol, schema, start, end)
            frames.append(frame)

        written = 0
        for frame in frames:
            bars = cat.frame_to_bars(
                frame, bar_type=bt, price_precision=PRICE_PRECISION, interval_ns=interval
            )
            with self._write_lock:
                written += cat.write_bars(catalog, bars)
        return written

    @staticmethod
    def _has_drifted(
        catalog: ParquetDataCatalog, bt: str, frame: pl.DataFrame, seam: int, interval: int
    ) -> bool:
        """Whether the fresh bar closing at *seam* disagrees with the stored one."""
        with_open = frame.filter(pl.col("ts_event").dt.epoch("ns") == seam - interval)
        stored = cat.stored_bar(catalog, bt, seam)
        if with_open.height == 0 or stored is None:
            logger.warning("%s: no overlapping bar at the seam; adjustment basis unchecked", bt)
            return False
        tolerance = 1.5 * 10.0**-PRICE_PRECISION
        return abs(with_open["close"][0] - stored.close.as_double()) > tolerance

    def _rebuild(
        self, catalog: ParquetDataCatalog, symbol: str, schema: str, start: int, end: int
    ) -> int:
        """Re-download the whole series and replace the stored one."""
        bt = cat.bar_type(symbol, self.venue, schema)
        interval = cat.bar_interval_ns(schema)
        with self._write_lock:
            first = catalog.query_first_timestamp(NautilusDataType.Bar, bt)
            last = catalog.query_last_timestamp(NautilusDataType.Bar, bt)
        assert first is not None and last is not None  # noqa: S101 — only called on drift
        lo = min(start, first - interval)
        frame = self.fetch_bars(symbol, schema, _ns_to_datetime(lo), _ns_to_datetime(end))
        bars = cat.frame_to_bars(
            frame, bar_type=bt, price_precision=PRICE_PRECISION, interval_ns=interval
        )
        # Never drop stored history the fresh download doesn't cover.
        if not bars or bars[0].ts_init > first or bars[-1].ts_init < last:
            raise AdjustmentDriftError(
                f"{symbol} {schema}: rebuild download does not cover the stored range; "
                "stored series left untouched."
            )
        with self._write_lock:
            catalog.delete_data_range(NautilusDataType.Bar, bt, first, last)
            catalog.write_bars(bars)
        logger.info("%s: rebuilt %d bars on the current adjustment basis", bt, len(bars))
        return len(bars)

    def sync(
        self,
        catalog: ParquetDataCatalog,
        symbols: Sequence[str],
        schema: str,
        start: int,
        end: int,
    ) -> list[cat.SyncResult]:
        """Sync every symbol concurrently; one failure doesn't stop the rest.

        Also writes an ``Equity`` instrument per symbol. Catalog writes are
        serialised; only the HTTP fetches run in parallel.
        """
        with self._write_lock:
            cat.write_instruments(
                catalog,
                [
                    cat.equity_instrument(s, self.venue, price_precision=PRICE_PRECISION)
                    for s in symbols
                ],
            )

        def one(symbol: str) -> cat.SyncResult:
            try:
                written = self.sync_symbol(catalog, symbol, schema, start, end)
            except Exception as exc:  # noqa: BLE001 — recorded per symbol in the result
                logger.warning("Alpaca sync failed for %s: %s", symbol, exc)
                return cat.SyncResult(symbol, schema, error=f"{type(exc).__name__}: {exc}")
            return cat.SyncResult(symbol, schema, written=written)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(one, symbols))
