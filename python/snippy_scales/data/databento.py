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
import time
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

#: How far back from the window end to look for a continuous symbol's latest definition.
_DEFINITION_LOOKBACK_NS = 7 * _DAY_NS

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


def _floor_day(ns: int) -> int:
    """*ns* rounded down to 00:00 UTC."""
    return ns - ns % _DAY_NS


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
        stype_in: Symbology for metadata calls; inferred per symbol if omitted.
        historical: A ``databento.Historical`` (or test double) for the free
            metadata endpoints. Built from the API key on first use. With
            neither (offline planning), no metadata checks are made.

    Window and pruning (when metadata is available):

    * :meth:`resolve_window` starts at the dataset's first day and ends at the
      last **completed** UTC day the dataset has. Databento bills a bar whose
      period hasn't finished like the intraday data behind it (a still-forming
      daily ES bar costs about $0.40, a finished one about $0.00001).
    * :meth:`plan` drops head gaps of up to 5 days outright (weekends,
      holidays), then any other missing range of an already-stored series that
      ``metadata.get_record_count`` (free) says holds no records — e.g. the
      years before a late listing, or a weekend at the tail. A series' first
      sync is not pruned, so a mistyped symbol still fails loudly instead of
      silently fetching nothing; if the count call fails the range is kept.
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
        stype_in: str | None = None,
        historical: Any = None,  # noqa: ANN401
    ) -> None:
        self.catalog = catalog
        self.dataset = dataset
        self.venue = venue or default_venue(dataset)
        self.stype_in = stype_in
        self._client = client
        self._api_key = api_key
        self._publishers_path = publishers_path
        self._historical = historical
        self._dataset_range: tuple[int, int] | None = None
        self._schema_ends: dict[str, int] = {}
        self._record_counts: dict[tuple[str, str, int, int], int] = {}

    # -- metadata (free endpoints) -----------------------------------------

    def _metadata(self) -> Any:  # noqa: ANN401
        """``databento.Historical().metadata``, or None when offline."""
        if self._historical is None:
            key = self._api_key or os.environ.get("DATABENTO_API_KEY")
            if not key:
                return None
            import databento as db  # noqa: PLC0415

            self._historical = db.Historical(key)
        return self._historical.metadata

    def dataset_range(self) -> tuple[int, int] | None:
        """``(first, last)`` timestamps the dataset holds (UNIX ns), or None offline."""
        if self._dataset_range is None:
            meta = self._metadata()
            if meta is None:
                return None
            import pandas as pd  # noqa: PLC0415

            info = meta.get_dataset_range(dataset=self.dataset)
            self._dataset_range = (
                int(pd.Timestamp(info["start"]).value),
                int(pd.Timestamp(info["end"]).value),
            )
            self._schema_ends = {
                name: int(pd.Timestamp(r["end"]).value)
                for name, r in (info.get("schema") or {}).items()
            }
        return self._dataset_range

    def resolve_window(
        self, start: int | None, end: int | None, *, now_ns: int | None = None
    ) -> tuple[int, int]:
        """The ``[start, end)`` to sync, in UNIX ns at 00:00 UTC.

        *start* defaults to the dataset's first day and is never earlier than
        it. *end* defaults to today and is capped at the last completed day the
        dataset has (today's bar is fetched tomorrow).

        Raises:
            ValueError: If *start* is None and the dataset range is unknown.
        """
        today = _floor_day(now_ns if now_ns is not None else time.time_ns())
        rng = self.dataset_range()
        if start is None:
            if rng is None:
                raise ValueError("No start date: set `start:` or DATABENTO_API_KEY.")
            start = rng[0]
        hi = min(end if end is not None else today, today)
        if rng is not None:
            start = max(start, _floor_day(rng[0]))
            hi = min(hi, _floor_day(rng[1]))
        return _floor_day(start), hi

    def _record_count(self, symbol: str, schema: str, lo: int, hi: int) -> int | None:
        """Records in ``[lo, hi)`` (free); 0 if nothing resolves there; None if unknown.

        Unknown (offline, or the endpoint failed) means "don't prune": the
        range is fetched as before.
        """
        meta = self._metadata()
        if meta is None:
            return None
        key = (symbol, schema, lo, hi)
        if key not in self._record_counts:
            import databento as db  # noqa: PLC0415

            try:
                n = meta.get_record_count(
                    dataset=self.dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=_iso(lo),
                    end=_iso(hi),
                    stype_in=self.stype_in or _stype_for(symbol),
                )
            except db.BentoClientError as exc:
                if getattr(exc, "http_status", None) != 422:  # noqa: PLR2004
                    logger.warning("Record count failed for %s %s: %s", symbol, schema, exc)
                    return None
                n = 0  # symbology_invalid_request: the symbol maps to nothing here
            except db.BentoServerError as exc:  # e.g. 504 gateway timeout: keep the range
                logger.warning("Record count failed for %s %s: %s", symbol, schema, exc)
                return None
            self._record_counts[key] = int(n)
        return self._record_counts[key]

    def _cap_end(self, end: int, interval_ns: int, schema: str | None = None) -> int:
        """*end* capped at the schema's (else the dataset's) available end, floored to a bar."""
        rng = self.dataset_range()
        if rng is None:
            return end
        avail = self._schema_ends.get(schema or "", rng[1])
        if interval_ns:
            avail -= avail % interval_ns
        return min(end, avail)

    def _requests(self, lo: int, hi: int, chunk_ns: int) -> list[tuple[int, int]]:
        """Split ``[lo, hi)`` into the requests actually sent (and priced).

        Chunks of at most *chunk_ns*, with the newest day always a request of
        its own when the range reaches the dataset's newest day boundary:
        Databento bills a multi-day request that includes the newest day far
        above the sum of its days (measured: daily NQ bars 2026-11-09→11-11 as
        one request $0.40, as two one-day requests $0.00002; 1h bars $0.016 vs
        $0.0005).
        """
        chunks = cat.chunk_range(lo, hi, chunk_ns)
        rng = self.dataset_range()
        if not chunks or rng is None:
            return chunks
        newest_day = _floor_day(rng[1]) - _DAY_NS
        c_lo, c_hi = chunks[-1]
        if c_hi > newest_day and c_lo < newest_day:
            chunks[-1:] = [(c_lo, newest_day), (newest_day, c_hi)]
        return chunks

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
        """Ranges still missing for *symbol*/*schema* within ``[start, end)``.

        *end* is capped at the dataset's available end. For a series that is
        already stored, ranges with no records are dropped (see class docstring).
        """
        if schema in _BAR_AGGREGATION:
            interval = cat.bar_interval_ns(schema)
            data_type, identifier = NautilusDataType.Bar, cat.bar_type(symbol, self.venue, schema)
        else:
            interval = 0
            data_type = _tick_spec(schema).data_type
            identifier = str(self._instrument_id(symbol))
        # Short head gaps are dropped without asking (cheap, and Databento is slow
        # to reject an empty weekend window); record counts handle the rest.
        ranges = cat.missing_ranges(
            self.catalog,
            data_type,
            identifier,
            start,
            self._cap_end(end, interval, schema),
            interval_ns=interval,
        )
        stored = self.catalog.query_last_timestamp(data_type, identifier) is not None
        if not ranges or not stored or self._metadata() is None:
            return ranges
        return [r for r in ranges if self._record_count(symbol, schema, *r) != 0]

    def estimate_cost(
        self, symbol: str, schema: str, start: int, end: int, *, stype_in: str | None = None
    ) -> float:
        """Databento cost in USD of the requests :meth:`sync` would send.

        Priced per request (see :meth:`_requests`), because that is how
        Databento bills them.
        """
        ranges = self.plan(symbol, schema, start, end)
        if not ranges:
            return 0.0
        chunk_ns = _BAR_AGGREGATION[schema][1] if schema in _BAR_AGGREGATION else _DAY_NS
        requests = [r for lo, hi in ranges for r in self._requests(lo, hi, chunk_ns)]
        import databento as db  # noqa: PLC0415

        meta = self._metadata()
        if meta is None:
            raise DatabentoCredentialsError("Set DATABENTO_API_KEY or pass api_key=.")
        total = 0.0
        for lo, hi in requests:
            try:
                total += meta.get_cost(
                    dataset=self.dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=_iso(lo),
                    end=_iso(hi),
                    stype_in=stype_in or self.stype_in or _stype_for(symbol),
                )
            except db.BentoClientError as exc:
                # 422 symbology_invalid_request on a short window means nothing
                # trades in it (e.g. a weekend at the tail), so it costs nothing.
                # On a long window it more likely means a bad symbol: re-raise.
                short = hi - lo <= cat.HEAD_GAP_TOLERANCE_NS
                if not short or getattr(exc, "http_status", None) != 422:  # noqa: PLR2004
                    raise
        return total

    # -- syncing ----------------------------------------------------------

    def price_precision(self, symbol: str, schema: str | None = None) -> int | None:
        """Precision to decode *symbol*'s prices at, from what the catalog already holds.

        The stored *schema* bar series wins (so appends match it), then a stored
        instrument definition. The Nautilus client otherwise only knows the
        precision of instruments it fetched itself, and a continuous symbol
        (``ES.c.0``) never has a definition of its own.
        """
        if schema in _BAR_AGGREGATION:
            stored = cat.stored_precision(self.catalog, cat.bar_type(symbol, self.venue, schema))
            if stored is not None:
                return stored[0]
        iid = str(self._instrument_id(symbol))
        for inst in self.catalog.instruments(instrument_ids=[iid]):
            if str(inst.id) == iid:
                return inst.price_precision
        return None

    def sync_bars(self, symbol: str, schema: str, start: int, end: int) -> int:
        """Fetch and store missing *schema* bars for *symbol*. Returns bars written."""
        aggregation, chunk_ns = _BAR_AGGREGATION[schema]
        interval_ns = cat.bar_interval_ns(schema)
        bar_type = BarType.from_str(cat.bar_type(symbol, self.venue, schema))
        iid = self._instrument_id(symbol)
        ranges = self.plan(symbol, schema, start, end)
        precision = self.price_precision(symbol, schema) if ranges else None
        if ranges and precision is None:
            raise ValueError(
                f"No price precision known for {iid}: sync instrument definitions first "
                "(drop --no-instruments, or run with --instruments-only)."
            )
        written = 0
        for lo, hi in ranges:
            for c_lo, c_hi in self._requests(lo, hi, chunk_ns):
                bars = _call(
                    lambda c_lo=c_lo, c_hi=c_hi: self.client.get_range_bars(
                        self.dataset,
                        [iid],
                        aggregation,
                        c_lo,
                        c_hi,
                        price_precision=precision,
                        timestamp_on_close=True,
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
        ranges = self.plan(symbol, schema, start, end)
        kwargs = dict(spec.kwargs)
        precision = self.price_precision(symbol) if ranges else None
        if precision is not None:
            kwargs["price_precision"] = precision
        written = 0
        for lo, hi in ranges:
            for c_lo, c_hi in self._requests(lo, hi, _DAY_NS):
                data = _call(
                    lambda c_lo=c_lo, c_hi=c_hi: getattr(self.client, spec.fetch)(
                        self.dataset, [iid], c_lo, c_hi, **kwargs
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
        """Store one definition per instrument ID, with a synthetic one per continuous symbol.

        The Nautilus client returns a continuous symbol's definitions already
        keyed by the continuous ID (``ES.c.0.GLBX``), one version per day of the
        window, each carrying the outright's expiry. Those are replaced by a
        single synthetic instrument (no activation/expiry) cloned from the most
        recent one. Other IDs keep their most recent version. Instruments are
        written in ``ts_init`` order, which the catalog requires.

        Only symbols whose instrument ID the catalog doesn't hold yet are
        requested, so re-runs cost nothing. A continuous symbol only needs its
        latest definition, so it is asked for over the last
        :data:`_DEFINITION_LOOKBACK_NS` of the window (the full window if that
        returns nothing) rather than all of it.

        Returns:
            Number of new instruments written.
        """
        stored = {str(i.id) for i in self.catalog.instruments()}
        todo = [s for s in symbols if str(self._instrument_id(s)) not in stored]
        if not todo:
            return 0
        defs = self._fetch_definitions(todo, start, end)
        latest: dict[str, Any] = {}
        for d in sorted(defs, key=lambda d: d.ts_init):
            latest[str(d.id)] = d
        for symbol in todo:
            if ".c." not in symbol:
                continue
            cid = str(self._instrument_id(symbol))
            root = symbol.split(".", 1)[0]
            source = latest.get(cid)
            if not isinstance(source, FuturesContract):
                outrights = [
                    d for d in defs if isinstance(d, FuturesContract) and d.underlying == root
                ]
                source = max(outrights, key=lambda c: c.ts_init) if outrights else None
            if source is None:
                logger.warning("No outright definition found for continuous symbol %s", symbol)
                latest.pop(cid, None)
                continue
            latest[cid] = cat.continuous_instrument(source, symbol)
        instruments = sorted(
            (i for i in latest.values() if str(i.id) not in stored), key=lambda i: i.ts_init
        )
        cat.write_instruments(self.catalog, instruments)
        return len(instruments)

    def _fetch_definitions(self, symbols: list[str], start: int, end: int) -> list[Any]:
        continuous = [s for s in symbols if ".c." in s]
        others = [s for s in symbols if ".c." not in s]
        defs: list[Any] = []

        def fetch(syms: list[str], lo: int) -> list[Any]:
            iids = [self._instrument_id(s) for s in syms]
            return list(
                _call(lambda: self.client.get_range_instruments(self.dataset, iids, lo, end))
            )

        if continuous:
            recent = fetch(continuous, max(start, end - _DEFINITION_LOOKBACK_NS))
            found = {str(d.id) for d in recent} | {getattr(d, "underlying", "") for d in recent}
            missing = [
                s
                for s in continuous
                if str(self._instrument_id(s)) not in found and s.split(".", 1)[0] not in found
            ]
            defs += recent + (fetch(missing, start) if missing else [])
        if others:
            defs += fetch(others, start)
        return defs


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
