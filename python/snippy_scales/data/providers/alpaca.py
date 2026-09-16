"""Alpaca :class:`~snippy_scales.data.providers.base.BarProvider` implementation.

Built on the official ``alpaca-py`` SDK rather than a hand-rolled HTTP client:
the SDK already paginates (`page_token`) and retries on 429/5xx internally
(``alpaca.common.rest.RESTClient._get_marketdata`` /
``RESTClient._request``). What it does *not* do is throttle proactively — it
only reacts to a 429 after the fact — so this module wraps the low-level
``get()`` call with a :class:`~snippy_scales.data.ratelimit.RateLimiter` to
stay under the free tier's per-minute quota in the first place.

Bars are renamed/cast/normalised into the same column layout Databento bars
are stored in (see :mod:`snippy_scales.data.schema`), so
:func:`snippy_scales.data.ingest.load_bars` returns an identical shape
regardless of which provider fetched the data.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import polars as pl

from snippy_scales.data.ratelimit import RateLimiter
from snippy_scales.data.schema import conform_bars, empty_bar_frame

if TYPE_CHECKING:
    from alpaca.data.historical.stock import StockHistoricalDataClient

#: Free-tier historical data withholds the most recent 15 minutes. Cap every
#: request's effective end a little further back to avoid a 422 for
#: requesting not-yet-available data.
_RECENT_DATA_EMBARGO = dt.timedelta(minutes=16)

#: Bar schema name -> alpaca-py TimeFrame(amount, unit) constructor args.
#: ``ohlcv-1s`` has no Alpaca equivalent (finest granularity is 1 minute).
_TIMEFRAME_BY_SCHEMA: dict[str, tuple[int, str]] = {
    "ohlcv-1m": (1, "Minute"),
    "ohlcv-1h": (1, "Hour"),
    "ohlcv-1d": (1, "Day"),
    "ohlcv-eod": (1, "Day"),
}

#: Public view of :data:`_TIMEFRAME_BY_SCHEMA`'s keys, for callers (e.g. the
#: CLI) that need to validate a schema up front rather than discovering it's
#: unsupported from the :exc:`ValueError` :meth:`AlpacaProvider.fetch_bars` raises.
SUPPORTED_BAR_SCHEMAS: frozenset[str] = frozenset(_TIMEFRAME_BY_SCHEMA)

#: Schemas whose bars are stamped once per session and must be normalised to
#: 00:00 UTC on the session date to line up with Databento's convention.
_DAILY_SCHEMAS = frozenset({"ohlcv-1d", "ohlcv-eod"})


class AlpacaCredentialsError(RuntimeError):
    """Raised when Alpaca API credentials are not available."""


def _parse_when(value: str) -> dt.datetime:
    """Parse a ``YYYY-MM-DD`` date or an RFC-3339 timestamp into a UTC datetime.

    Args:
        value: A date-only string (interpreted as 00:00 UTC) or a full
            ISO-8601/RFC-3339 timestamp (``Z`` suffix accepted).

    Returns:
        A timezone-aware :class:`datetime.datetime` in UTC.
    """
    normalised = value.strip()
    if normalised.endswith("Z"):
        normalised = normalised[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(normalised)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _make_rate_limited_client(
    *,
    api_key: str,
    secret_key: str,
    rate_limiter: RateLimiter,
    retry_attempts: int,
    retry_wait_seconds: int,
) -> StockHistoricalDataClient:
    """Build a ``StockHistoricalDataClient`` that rate-limits every HTTP GET.

    ``alpaca-py`` funnels every page of every request through
    ``RESTClient.get()`` (see ``_get_marketdata``), so overriding it here
    throttles pagination within a single symbol's request as well as calls
    across symbols/threads sharing the same *rate_limiter*.
    """
    from alpaca.data.historical.stock import StockHistoricalDataClient  # noqa: PLC0415

    class _RateLimitedStockClient(StockHistoricalDataClient):
        def get(self, path: str, data: dict | str | None = None, **kwargs: object) -> object:
            rate_limiter.acquire()
            return super().get(path, data=data, **kwargs)

    client = _RateLimitedStockClient(api_key=api_key, secret_key=secret_key)
    # StockHistoricalDataClient's __init__ doesn't forward retry_* kwargs to its
    # RESTClient base, so set the (plain instance attributes) it would otherwise
    # default from DEFAULT_RETRY_ATTEMPTS/DEFAULT_RETRY_WAIT_SECONDS/DEFAULT_RETRY_EXCEPTION_CODES.
    client._retry = retry_attempts  # noqa: SLF001
    client._retry_wait = retry_wait_seconds  # noqa: SLF001
    client._retry_codes = [429, 500, 502, 503, 504]  # noqa: SLF001
    return client


@dataclass
class AlpacaProvider:
    """Fetches OHLCV bars from Alpaca's Market Data API.

    Attributes:
        feed: Data feed to request — ``"sip"`` (all US exchanges) or
            ``"iex"``. The free tier includes full historical SIP data
            (real-time SIP is not included).
        adjustment: Corporate-action adjustment applied by Alpaca —
            ``"raw"``, ``"split"``, ``"dividend"``, or ``"all"``.
        rate_limit_per_min: Historical API calls allowed per minute. The free
            (Basic) tier allows 200; default leaves headroom.
        retry_attempts: Passed through to the SDK's built-in 429/5xx retry.
        retry_wait_seconds: Fixed wait between SDK-level retries.
        api_key: Alpaca API key ID. Falls back to ``ALPACA_API_KEY``.
        secret_key: Alpaca API secret. Falls back to ``ALPACA_SECRET_KEY``.

    Raises:
        AlpacaCredentialsError: If no API key/secret is available from either
            the constructor arguments or the environment.
    """

    feed: Literal["iex", "sip"] = "sip"
    adjustment: Literal["raw", "split", "dividend", "all"] = "all"
    rate_limit_per_min: int = 190
    retry_attempts: int = 5
    retry_wait_seconds: int = 5
    api_key: str | None = None
    secret_key: str | None = None
    name: str = field(default="alpaca", init=False)
    resume_granularity: Literal["day", "timestamp"] = field(default="timestamp", init=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("ALPACA_API_KEY")
        secret = self.secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise AlpacaCredentialsError(
                "Alpaca provider requires an API key/secret — set ALPACA_API_KEY and "
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

    def fetch_bars(self, *, symbol: str, schema: str, start: str, end: str) -> pl.DataFrame:
        """Fetch bars for *symbol* over ``[start, end)`` and conform them.

        Args:
            symbol: Alpaca ticker symbol (e.g. ``"AAPL"``, ``"BRK.B"``).
            schema: Bar schema name — one of :data:`_TIMEFRAME_BY_SCHEMA`'s keys.
            start: Inclusive start, ``YYYY-MM-DD`` or an RFC-3339 timestamp.
            end: Exclusive end, ``YYYY-MM-DD`` or an RFC-3339 timestamp.

        Returns:
            Bars conformed to the shared schema, sorted by ``ts_event``. Empty
            when the (possibly embargo-capped) range contains no bars.

        Raises:
            ValueError: If *schema* has no Alpaca timeframe equivalent (e.g.
                ``"ohlcv-1s"``) or the range is entirely embargoed.
        """
        if schema not in _TIMEFRAME_BY_SCHEMA:
            raise ValueError(
                f"Alpaca provider does not support schema {schema!r}. "
                f"Supported: {sorted(_TIMEFRAME_BY_SCHEMA)}."
            )

        start_dt = _parse_when(start)
        end_dt = min(_parse_when(end), dt.datetime.now(dt.UTC) - _RECENT_DATA_EMBARGO)
        if start_dt >= end_dt:
            return empty_bar_frame(symbol=symbol, schema=schema)

        bars = self._request_bars(symbol=symbol, schema=schema, start=start_dt, end=end_dt)
        return conform_bars(bars, symbol=symbol, schema=schema).sort("ts_event")

    def _request_bars(
        self, *, symbol: str, schema: str, start: dt.datetime, end: dt.datetime
    ) -> pl.DataFrame:
        """Issue the Alpaca request and return a raw (pre-conform) OHLCV frame."""
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
        # get_stock_bars() is typed as BarSet | dict since raw_data=True would
        # return a dict — this client never sets that, so it's always a BarSet.
        assert isinstance(bar_set, BarSet)  # noqa: S101
        pdf = bar_set.df

        if pdf.empty:
            # BaseDataSet.df returns a plain empty DataFrame (no columns at all)
            # when `data` is empty — return the shared-schema shape directly
            # rather than feeding an empty, unindexed frame through conform_bars.
            return empty_bar_frame(symbol=symbol, schema=schema)

        pdf = pdf.reset_index()  # "symbol", "timestamp" from the MultiIndex -> columns
        pdf = pdf.rename(columns={"timestamp": "ts_event"})
        df = pl.from_pandas(pdf)

        if schema in _DAILY_SCHEMAS:
            df = df.with_columns(
                pl.col("ts_event")
                .dt.date()
                .cast(pl.Datetime("ns"))
                .dt.replace_time_zone("UTC")
                .alias("ts_event")
            )

        return df
