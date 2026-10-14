"""Databento -> catalog sync, using NautilusTrader's own Databento client.

:class:`DatabentoSync` keeps the catalog up to date for a list of symbols: for
each series it asks the catalog what is already stored
(:func:`~snippy_scales.data.catalog.missing_ranges`), downloads only the missing
head and tail through ``DatabentoHistoricalClient``, and writes Nautilus objects
straight into the catalog. Nothing else is stored: no DBN archive, no custom
Parquet. Data is downloaded once, so Databento's batch API (whose advantage is
free re-downloads) isn't needed; requests are chunked to keep them small.

Bars are requested with ``timestamp_on_close=True`` so ``ts_init`` is the bar
close. Instrument definitions come from Databento's ``definition`` schema; a
continuous symbol (``ES.c.0``) additionally gets a synthetic instrument cloned
from its most recent outright, so backtests on the continuous series have the
right tick size and multiplier.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nautilus_trader.model import (
    Bar,
    BarAggregation,
    BarType,
    FuturesContract,
    InstrumentId,
    NautilusDataType,
)

from snippy_scales.data import catalog as cat
from snippy_scales.data.paths import CACHE_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

    from nautilus_trader.persistence import ParquetDataCatalog

logger = logging.getLogger(__name__)

_DAY_NS = 86_400_000_000_000

#: Bar schema -> (Nautilus aggregation, request chunk length).
_BAR_AGGREGATION: dict[str, tuple[BarAggregation, int]] = {
    "ohlcv-1s": (BarAggregation.SECOND, _DAY_NS),
    "ohlcv-1m": (BarAggregation.MINUTE, 31 * _DAY_NS),
    "ohlcv-1h": (BarAggregation.HOUR, 366 * _DAY_NS),
    "ohlcv-1d": (BarAggregation.DAY, 20 * 366 * _DAY_NS),
}


@dataclass(frozen=True)
class _TickSpec:
    fetch: str  # DatabentoHistoricalClient method
    write: str  # ParquetDataCatalog method
    data_type: NautilusDataType
    kwargs: tuple[tuple[str, Any], ...] = ()


#: Event-level schema -> how to fetch and store it. Requests are one day each.
TICK_SPECS: dict[str, _TickSpec] = {
    "trades": _TickSpec("get_range_trades", "write_trade_ticks", NautilusDataType.TradeTick),
    "tbbo": _TickSpec(
        "get_range_trades", "write_trade_ticks", NautilusDataType.TradeTick, (("schema", "tbbo"),)
    ),
    "mbp-1": _TickSpec(
        "get_range_quotes", "write_quote_ticks", NautilusDataType.QuoteTick, (("schema", "mbp-1"),)
    ),
    "bbo-1s": _TickSpec(
        "get_range_quotes", "write_quote_ticks", NautilusDataType.QuoteTick, (("schema", "bbo-1s"),)
    ),
    "bbo-1m": _TickSpec(
        "get_range_quotes", "write_quote_ticks", NautilusDataType.QuoteTick, (("schema", "bbo-1m"),)
    ),
    "mbo": _TickSpec(
        "get_range_order_book_deltas", "write_order_book_deltas", NautilusDataType.OrderBookDelta
    ),
    "mbp-10": _TickSpec(
        "get_order_book_depth", "write_order_book_depths", NautilusDataType.OrderBookDepth
    ),
}

DEFAULT_PUBLISHERS_PATH = CACHE_DIR / "databento_publishers.json"


class DatabentoCredentialsError(RuntimeError):
    """Raised when no Databento API key is available."""


def default_venue(dataset: str) -> str:
    """Venue Nautilus assigns to a dataset's instruments: ``GLBX.MDP3`` -> ``GLBX``."""
    return dataset.split(".", 1)[0]


def _api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise DatabentoCredentialsError("Set DATABENTO_API_KEY or pass api_key=.")
    return key


def ensure_publishers_file(
    path: Path = DEFAULT_PUBLISHERS_PATH, *, api_key: str | None = None
) -> Path:
    """Download Databento's publisher list (required by the Nautilus client) if absent."""
    if path.exists():
        return path
    import databento as db  # noqa: PLC0415

    publishers = db.Historical(_api_key(api_key)).metadata.list_publishers()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(publishers, indent=2))
    return path


def _iso(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.UTC).isoformat()


def _call(fn: Callable[[], Coroutine[Any, Any, Any]]) -> Any:  # noqa: ANN401
    """Run one async client call. The coroutine must be created inside the loop."""

    async def runner() -> Any:  # noqa: ANN401
        return await fn()

    return asyncio.run(runner())


def _normalise_bar(bar: Bar, bar_type: BarType, interval_ns: int) -> Bar:
    """Re-key *bar* to *bar_type* with ``ts_event`` = open and ``ts_init`` = close."""
    ts_init = bar.ts_init
    ts_event = bar.ts_event if bar.ts_event < ts_init else ts_init - interval_ns
    if str(bar.bar_type) == str(bar_type) and ts_event == bar.ts_event:
        return bar
    return Bar(bar_type, bar.open, bar.high, bar.low, bar.close, bar.volume, ts_event, ts_init)


class DatabentoSync:
    """Incrementally sync Databento data into a catalog.

    Args:
        catalog: Target catalog.
        dataset: Databento dataset (e.g. ``"GLBX.MDP3"``).
        venue: Venue for instrument IDs. Defaults to :func:`default_venue`.
        client: A ``DatabentoHistoricalClient`` (or test double). Built lazily
            from *api_key* / ``DATABENTO_API_KEY`` when omitted.
        api_key: Databento API key.
        publishers_path: Publisher list file for the client.
    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        *,
        dataset: str,
        venue: str | None = None,
        client: Any = None,  # noqa: ANN401
        api_key: str | None = None,
        publishers_path: Path = DEFAULT_PUBLISHERS_PATH,
    ) -> None:
        self.catalog = catalog
        self.dataset = dataset
        self.venue = venue or default_venue(dataset)
        self._client = client
        self._api_key = api_key
        self._publishers_path = publishers_path

    @property
    def client(self) -> Any:  # noqa: ANN401
        """The Databento client, built on first use (planning needs no API key)."""
        if self._client is None:
            from nautilus_trader.adapters.databento import (
                DatabentoHistoricalClient,  # noqa: PLC0415
            )

            key = _api_key(self._api_key)
            publishers = ensure_publishers_file(self._publishers_path, api_key=key)
            self._client = DatabentoHistoricalClient(key, str(publishers), False)
        return self._client

    def _instrument_id(self, symbol: str) -> InstrumentId:
        return InstrumentId.from_str(cat.instrument_id(symbol, self.venue))

    # -- planning ---------------------------------------------------------

    def plan(self, symbol: str, schema: str, start: int, end: int) -> list[tuple[int, int]]:
        """Ranges still missing for *symbol*/*schema* within ``[start, end)``."""
        if schema in _BAR_AGGREGATION:
            return cat.missing_ranges(
                self.catalog,
                NautilusDataType.Bar,
                cat.bar_type(symbol, self.venue, schema),
                start,
                end,
                interval_ns=cat.bar_interval_ns(schema),
            )
        spec = _tick_spec(schema)
        return cat.missing_ranges(
            self.catalog, spec.data_type, str(self._instrument_id(symbol)), start, end
        )

    def estimate_cost(
        self, symbol: str, schema: str, start: int, end: int, *, stype_in: str | None = None
    ) -> float:
        """Databento cost in USD of fetching what :meth:`plan` reports missing."""
        ranges = self.plan(symbol, schema, start, end)
        if not ranges:
            return 0.0
        import databento as db  # noqa: PLC0415

        client = db.Historical(_api_key(self._api_key))
        return sum(
            client.metadata.get_cost(
                dataset=self.dataset,
                symbols=[symbol],
                schema=schema,
                start=_iso(lo),
                end=_iso(hi),
                stype_in=stype_in or _stype_for(symbol),
            )
            for lo, hi in ranges
        )

    # -- syncing ----------------------------------------------------------

    def sync_bars(self, symbol: str, schema: str, start: int, end: int) -> int:
        """Fetch and store missing *schema* bars for *symbol*. Returns bars written."""
        aggregation, chunk_ns = _BAR_AGGREGATION[schema]
        interval_ns = cat.bar_interval_ns(schema)
        bar_type = BarType.from_str(cat.bar_type(symbol, self.venue, schema))
        iid = self._instrument_id(symbol)
        written = 0
        for lo, hi in self.plan(symbol, schema, start, end):
            for c_lo, c_hi in cat.chunk_range(lo, hi, chunk_ns):
                bars = _call(
                    lambda c_lo=c_lo, c_hi=c_hi: self.client.get_range_bars(
                        self.dataset, [iid], aggregation, c_lo, c_hi, timestamp_on_close=True
                    )
                )
                normalised = [_normalise_bar(b, bar_type, interval_ns) for b in bars]
                written += cat.write_bars(self.catalog, normalised)
        return written

    def sync_ticks(self, symbol: str, schema: str, start: int, end: int) -> int:
        """Fetch and store missing event-level *schema* data, one day per request."""
        spec = _tick_spec(schema)
        iid = self._instrument_id(symbol)
        write = getattr(self.catalog, spec.write)
        written = 0
        for lo, hi in self.plan(symbol, schema, start, end):
            for c_lo, c_hi in cat.chunk_range(lo, hi, _DAY_NS):
                data = _call(
                    lambda c_lo=c_lo, c_hi=c_hi: getattr(self.client, spec.fetch)(
                        self.dataset, [iid], c_lo, c_hi, **dict(spec.kwargs)
                    )
                )
                if data:
                    write(sorted(data, key=lambda d: d.ts_init))
                    written += len(data)
        return written

    def sync(self, symbol: str, schema: str, start: int, end: int) -> int:
        """Sync bars or ticks depending on *schema*."""
        if schema in _BAR_AGGREGATION:
            return self.sync_bars(symbol, schema, start, end)
        return self.sync_ticks(symbol, schema, start, end)

    def sync_instruments(self, symbols: list[str], start: int, end: int) -> int:
        """Store definitions for *symbols*, plus a synthetic instrument per continuous symbol.

        Returns:
            Number of instruments passed to the catalog (existing IDs are skipped).
        """
        iids = [self._instrument_id(s) for s in symbols]
        defs = _call(lambda: self.client.get_range_instruments(self.dataset, iids, start, end))
        instruments: list[Any] = list(defs)
        for symbol in symbols:
            if ".c." not in symbol:
                continue
            root = symbol.split(".", 1)[0]
            contracts = [d for d in defs if isinstance(d, FuturesContract) and d.underlying == root]
            if contracts:
                latest = max(contracts, key=lambda c: c.activation_ns)
                instruments.append(cat.continuous_instrument(latest, symbol))
            else:
                logger.warning("No outright definition found for continuous symbol %s", symbol)
        cat.write_instruments(self.catalog, instruments)
        return len(instruments)


def _tick_spec(schema: str) -> _TickSpec:
    try:
        return TICK_SPECS[schema]
    except KeyError:
        supported = sorted([*_BAR_AGGREGATION, *TICK_SPECS])
        raise ValueError(
            f"Unsupported Databento schema {schema!r}. Supported: {supported}."
        ) from None


def _stype_for(symbol: str) -> str:
    """Databento ``stype_in`` implied by a symbol's form."""
    if ".c." in symbol or ".n." in symbol or ".v." in symbol:
        return "continuous"
    if symbol.endswith((".FUT", ".OPT")):
        return "parent"
    return "raw_symbol"


SUPPORTED_SCHEMAS: frozenset[str] = frozenset([*_BAR_AGGREGATION, *TICK_SPECS])
