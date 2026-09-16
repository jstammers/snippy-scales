"""The provider interface every bar-data source implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import polars as pl

#: How a provider's "already have data up to here" resume point is computed.
#: ``"day"`` — resume from the calendar day after the last stored bar (safe
#: when a provider bills per day and bars never cross a UTC day boundary,
#: e.g. Databento). ``"timestamp"`` — resume from immediately after the last
#: stored bar's exact timestamp (required for providers whose bars can span a
#: UTC day boundary, e.g. Alpaca's extended-hours session).
ResumeGranularity = Literal["day", "timestamp"]


class BarProvider(Protocol):
    """A source of OHLCV bar data for a single symbol/schema/date-range.

    Implementations fetch data and return it already conformed to
    :data:`snippy_scales.data.schema.BAR_SCHEMA_COLUMNS` via
    :func:`snippy_scales.data.schema.conform_bars` — callers
    (:func:`snippy_scales.data.ingest.upsert_bars`) never need to know which
    provider produced a given frame.
    """

    name: str
    resume_granularity: ResumeGranularity

    def fetch_bars(self, *, symbol: str, schema: str, start: str, end: str) -> pl.DataFrame:
        """Fetch bars for one symbol.

        Args:
            symbol: Instrument symbol, in whatever convention the provider
                expects (e.g. ``"ES.c.0"`` for Databento, ``"AAPL"`` for
                Alpaca).
            schema: Bar schema name (e.g. ``"ohlcv-1d"``, ``"ohlcv-1m"``).
            start: Inclusive start, ``YYYY-MM-DD`` or an RFC-3339 timestamp.
            end: Exclusive end, ``YYYY-MM-DD`` or an RFC-3339 timestamp.

        Returns:
            A :class:`polars.DataFrame` conformed to the shared bar schema,
            sorted by ``ts_event``. Empty (not ``None``) when the range
            contains no bars.
        """
        ...
