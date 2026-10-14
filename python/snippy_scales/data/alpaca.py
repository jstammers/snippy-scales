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


class AlpacaCredentialsError(RuntimeError):
    """Raised when Alpaca API credentials are not available."""


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
        """Fetch and store the bars missing for *symbol* in ``[start, end)`` (UNIX ns)."""
        bt = cat.bar_type(symbol, self.venue, schema)
        interval = cat.bar_interval_ns(schema)
        with self._write_lock:
            ranges = cat.missing_ranges(
                catalog, NautilusDataType.Bar, bt, start, end, interval_ns=interval
            )
        written = 0
        for lo, hi in ranges:
            frame = self.fetch_bars(symbol, schema, _ns_to_datetime(lo), _ns_to_datetime(hi))
            bars = cat.frame_to_bars(
                frame, bar_type=bt, price_precision=PRICE_PRECISION, interval_ns=interval
            )
            with self._write_lock:
                written += cat.write_bars(catalog, bars)
        return written

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
