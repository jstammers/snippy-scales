"""Tests for snippy_scales.data.databento — incremental sync against a fake client.

The real ``DatabentoHistoricalClient`` needs an API key and network access, so
these tests use an async fake with the same method signatures and assert on
which ranges get requested and what lands in the catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from nautilus_trader.model import (
    AggressorSide,
    AssetClass,
    Bar,
    BarAggregation,
    BarType,
    Currency,
    FuturesContract,
    InstrumentId,
    Price,
    Quantity,
    Symbol,
    TradeId,
    TradeTick,
)

from snippy_scales.data import catalog as cat
from snippy_scales.data.databento import DatabentoSync, _stype_for, default_venue

if TYPE_CHECKING:
    from pathlib import Path

    from nautilus_trader.persistence import ParquetDataCatalog

_DAY = 86_400_000_000_000
_T0 = 1_704_067_200_000_000_000  # 2024-01-01T00:00Z


def _contract(symbol: str, activation: int) -> FuturesContract:
    return FuturesContract(
        InstrumentId.from_str(f"{symbol}.GLBX"),
        Symbol(symbol),
        AssetClass.INDEX,
        "ES",
        activation,
        activation + 90 * _DAY,
        Currency.from_str("USD"),
        2,
        Price.from_str("0.25"),
        Quantity.from_int(50),
        Quantity.from_int(1),
        0,
        0,
    )


class FakeClient:
    """Serves daily bars / trades for any requested window and records the calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.precisions: list[int] = []

    async def get_range_bars(
        self,
        dataset: str,
        instrument_ids: list[InstrumentId],
        aggregation: BarAggregation,
        start: int,
        end: int,
        price_precision: int | None = None,
        timestamp_on_close: bool = True,
    ) -> list[Bar]:
        assert timestamp_on_close
        assert price_precision is not None  # the real client can't resolve ES.c.0 itself
        self.precisions.append(price_precision)
        self.calls.append(("bars", start, end))
        bar_type = BarType.from_str(f"{instrument_ids[0]}-1-DAY-LAST-EXTERNAL")
        out = []
        for open_ns in range(start, end, _DAY):
            px = Price(4500 + (open_ns - _T0) // _DAY * 0.25, price_precision)
            # Like Databento with timestamp_on_close: both stamps at the bar close.
            close_ns = open_ns + _DAY
            out.append(Bar(bar_type, px, px, px, px, Quantity(10, 0), close_ns, close_ns))
        return out

    async def get_range_trades(
        self, dataset: str, instrument_ids: list[InstrumentId], start: int, end: int, **_: Any
    ) -> list[TradeTick]:
        self.calls.append(("trades", start, end))
        return [
            TradeTick(
                instrument_ids[0],
                Price(4500.25, 2),
                Quantity(1, 0),
                AggressorSide.BUY,
                TradeId(str(ts)),
                ts,
                ts,
            )
            for ts in range(start, end, _DAY // 4)
        ]

    async def get_range_instruments(
        self, dataset: str, instrument_ids: list[InstrumentId], start: int, end: int
    ) -> list[FuturesContract]:
        self.calls.append(("instruments", start, end))
        return [_contract("ESH4", _T0 - 90 * _DAY), _contract("ESM4", _T0)]


class FakeHistorical:
    """Stands in for ``databento.Historical``: the free metadata endpoints only."""

    def __init__(self, *, end: int, start: int = _T0 - 365 * _DAY, records: Any = None) -> None:
        self.metadata = self
        self._start, self._end = start, end
        # (lo_ns, hi_ns) -> record count; default: one record per day.
        self._records = records or (lambda lo, hi: (hi - lo) // _DAY)
        self.count_calls: list[tuple[int, int]] = []

    @staticmethod
    def _ns(iso: str) -> int:
        import pandas as pd

        return int(pd.Timestamp(iso).value)

    def get_dataset_range(self, **_: Any) -> dict[str, str]:
        import pandas as pd

        return {
            "start": pd.Timestamp(self._start, tz="UTC").isoformat(),
            "end": pd.Timestamp(self._end, tz="UTC").isoformat(),
        }

    def get_record_count(self, *, start: str, end: str, **_: Any) -> int:
        lo, hi = self._ns(start), self._ns(end)
        self.count_calls.append((lo, hi))
        return self._records(lo, hi)

    def get_cost(self, *, start: str, end: str, **_: Any) -> float:
        return 0.001 * self.get_record_count(start=start, end=end)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real key, so DatabentoSync never looks up the dataset range over the network."""
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)


@pytest.fixture
def catalog(tmp_path: Path) -> ParquetDataCatalog:
    return cat.open_catalog(tmp_path / "catalog")


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


def _sync(catalog: ParquetDataCatalog, client: FakeClient) -> DatabentoSync:
    """A syncer over a catalog that already holds the ES.c.0 instrument (2 decimals)."""
    cat.write_instruments(catalog, [cat.continuous_instrument(_contract("ESM4", _T0), "ES.c.0")])
    return DatabentoSync(catalog, dataset="GLBX.MDP3", client=client)


class TestBars:
    def test_initial_sync_then_noop(
        self, catalog: ParquetDataCatalog, client: FakeClient, tmp_path: Path
    ) -> None:
        sync = _sync(catalog, client)
        assert sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 10 * _DAY) == 10
        assert sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 10 * _DAY) == 0
        assert [c[0] for c in client.calls] == ["bars"]

        df = cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=tmp_path / "catalog")
        assert df.height == 10
        # ts_event normalised back to the bar open.
        assert (df["ts_init"] - df["ts_event"]).dt.total_nanoseconds().unique().to_list() == [_DAY]

    def test_extending_requests_only_head_and_tail(
        self, catalog: ParquetDataCatalog, client: FakeClient, tmp_path: Path
    ) -> None:
        sync = _sync(catalog, client)
        sync.sync("ES.c.0", "ohlcv-1d", _T0 + 10 * _DAY, _T0 + 15 * _DAY)
        client.calls.clear()

        assert sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 20 * _DAY) == 15
        assert client.calls == [
            ("bars", _T0, _T0 + 10 * _DAY),
            ("bars", _T0 + 15 * _DAY, _T0 + 20 * _DAY),
        ]
        df = cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=tmp_path / "catalog")
        assert df.height == 20
        assert df["ts_event"].n_unique() == 20

    def test_requests_stop_at_last_complete_bar(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        # Dataset has data up to 23:31 on day 3: the day-3 bar hasn't finished.
        avail = _T0 + 3 * _DAY - 29 * 60 * 10**9
        sync = _sync(catalog, client)
        sync._historical = FakeHistorical(end=avail)  # noqa: SLF001
        assert sync.plan("ES.c.0", "ohlcv-1d", _T0, _T0 + 10 * _DAY) == [(_T0, _T0 + 2 * _DAY)]
        assert sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 10 * _DAY) == 2

    def test_short_head_gap_is_not_requested(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        # e.g. config start on a Sunday, first bar on the Monday.
        sync = _sync(catalog, client)
        sync.sync("ES.c.0", "ohlcv-1d", _T0 + _DAY, _T0 + 5 * _DAY)
        assert sync.plan("ES.c.0", "ohlcv-1d", _T0, _T0 + 5 * _DAY) == []

    def test_appends_conform_to_stored_precision(
        self, catalog: ParquetDataCatalog, client: FakeClient, tmp_path: Path
    ) -> None:
        # A migrated series stored at 4 decimals; Databento delivers 2.
        root = tmp_path / "catalog"
        bt = cat.bar_type("ES.c.0", "GLBX", "ohlcv-1d")
        stored = cat.frame_to_bars(
            pl.DataFrame(
                {
                    "ts_event": [_T0 - _DAY],
                    "open": [4499.75],
                    "high": [4499.75],
                    "low": [4499.75],
                    "close": [4499.75],
                    "volume": [10.0],
                }
            ),
            bar_type=bt,
            price_precision=4,
            interval_ns=_DAY,
        )
        cat.write_bars(catalog, stored)
        assert _sync(catalog, client).sync("ES.c.0", "ohlcv-1d", _T0 - _DAY, _T0 + 3 * _DAY) == 3
        # Decoded at the stored series' precision (4), not the instrument's (2).
        assert client.precisions == [4]
        assert cat.stored_precision(catalog, bt) == (4, 0)
        catalog.consolidate_catalog()  # would fail on mixed-precision files
        assert cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=root).height == 4

    def test_unknown_precision_fails_before_any_request(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        sync = DatabentoSync(catalog, dataset="GLBX.MDP3", client=client)  # no instruments
        with pytest.raises(ValueError, match="sync instrument definitions first"):
            sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 3 * _DAY)
        assert client.calls == []

    def test_unsupported_schema(self, catalog: ParquetDataCatalog, client: FakeClient) -> None:
        with pytest.raises(ValueError, match="Unsupported Databento schema"):
            _sync(catalog, client).sync("ES.c.0", "ohlcv-eod", _T0, _T0 + _DAY)


class TestTicks:
    def test_day_chunks_and_resume(self, catalog: ParquetDataCatalog, client: FakeClient) -> None:
        sync = _sync(catalog, client)
        assert sync.sync("ES.c.0", "trades", _T0, _T0 + 2 * _DAY) == 8
        assert client.calls == [("trades", _T0, _T0 + _DAY), ("trades", _T0 + _DAY, _T0 + 2 * _DAY)]

        client.calls.clear()
        assert sync.sync("ES.c.0", "trades", _T0, _T0 + 3 * _DAY) > 0
        # Resumes one ns after the last stored trade, not from the start.
        assert client.calls[0][1] > _T0 + _DAY


class TestInstruments:
    def test_continuous_symbol_gets_synthetic_instrument(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        sync = DatabentoSync(catalog, dataset="GLBX.MDP3", client=client)
        assert sync.sync_instruments(["ES.c.0"], _T0, _T0 + 30 * _DAY) == 3
        ids = sorted(str(i.id) for i in catalog.instruments())
        assert ids == ["ES.c.0.GLBX", "ESH4.GLBX", "ESM4.GLBX"]
        cont = next(i for i in catalog.instruments() if str(i.id) == "ES.c.0.GLBX")
        assert str(cont.multiplier) == "50"
        # Continuous symbols only need the latest definition: last 7 days of the window.
        assert client.calls == [("instruments", _T0 + 23 * _DAY, _T0 + 30 * _DAY)]

    def test_stored_instruments_are_not_refetched(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        sync = DatabentoSync(catalog, dataset="GLBX.MDP3", client=client)
        sync.sync_instruments(["ES.c.0"], _T0, _T0 + 30 * _DAY)
        client.calls.clear()
        assert sync.sync_instruments(["ES.c.0"], _T0, _T0 + 30 * _DAY) == 0
        assert client.calls == []


class TestWindowAndPruning:
    def test_window_defaults_to_dataset_start_and_last_completed_day(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        avail = _T0 + 10 * _DAY + 23 * 3_600 * 10**9  # data up to 23:00 on day 10
        sync = DatabentoSync(
            catalog,
            dataset="GLBX.MDP3",
            client=client,
            historical=FakeHistorical(start=_T0 - 5 * _DAY + 3_600 * 10**9, end=avail),
        )
        now = _T0 + 11 * _DAY + 3_600 * 10**9
        assert sync.resolve_window(None, None, now_ns=now) == (_T0 - 5 * _DAY, _T0 + 10 * _DAY)
        # An explicit range is clipped to the same bounds.
        assert sync.resolve_window(_T0 - 30 * _DAY, _T0 + 60 * _DAY, now_ns=now) == (
            _T0 - 5 * _DAY,
            _T0 + 10 * _DAY,
        )

    def test_ranges_without_records_are_dropped_for_stored_series(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        sync = _sync(catalog, client)
        sync.sync("ES.c.0", "ohlcv-1d", _T0 + 100 * _DAY, _T0 + 110 * _DAY)
        # Nothing before day 100 (a late listing) and nothing on day 110 (a holiday).
        sync._historical = FakeHistorical(  # noqa: SLF001
            end=_T0 + 400 * _DAY,
            records=lambda lo, hi: 0 if hi <= _T0 + 111 * _DAY else (hi - lo) // _DAY,
        )
        assert sync.plan("ES.c.0", "ohlcv-1d", _T0, _T0 + 111 * _DAY) == []
        assert sync.plan("ES.c.0", "ohlcv-1d", _T0, _T0 + 115 * _DAY) == [
            (_T0 + 110 * _DAY, _T0 + 115 * _DAY)
        ]

    def test_newest_day_is_requested_and_priced_separately(
        self, catalog: ParquetDataCatalog, client: FakeClient, tmp_path: Path
    ) -> None:
        hist = FakeHistorical(end=_T0 + 10 * _DAY + 3_600 * 10**9)  # newest day: day 9
        sync = DatabentoSync(catalog, dataset="GLBX.MDP3", client=client, historical=hist)
        cat.write_instruments(
            catalog, [cat.continuous_instrument(_contract("ESM4", _T0), "ES.c.0")]
        )
        assert sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 10 * _DAY) == 10
        assert client.calls == [
            ("bars", _T0, _T0 + 9 * _DAY),
            ("bars", _T0 + 9 * _DAY, _T0 + 10 * _DAY),
        ]
        hist.count_calls.clear()
        catalog2 = cat.open_catalog(tmp_path / "empty")
        sync2 = DatabentoSync(catalog2, dataset="GLBX.MDP3", client=client, historical=hist)
        sync2.estimate_cost("ES.c.0", "ohlcv-1d", _T0, _T0 + 10 * _DAY)
        assert hist.count_calls == [(_T0, _T0 + 9 * _DAY), (_T0 + 9 * _DAY, _T0 + 10 * _DAY)]

    def test_count_failure_keeps_the_range(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        import databento as db

        def timeout(lo: int, hi: int) -> int:
            raise db.BentoServerError(504, message="The remote gateway timed out.")

        sync = _sync(catalog, client)
        sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 5 * _DAY)
        sync._historical = FakeHistorical(end=_T0 + 400 * _DAY, records=timeout)  # noqa: SLF001
        assert sync.plan("ES.c.0", "ohlcv-1d", _T0, _T0 + 8 * _DAY) == [
            (_T0 + 5 * _DAY, _T0 + 8 * _DAY)
        ]

    def test_first_sync_is_not_pruned(
        self, catalog: ParquetDataCatalog, client: FakeClient
    ) -> None:
        # A mistyped symbol resolves to nothing; that must surface, not be skipped.
        hist = FakeHistorical(end=_T0 + 400 * _DAY, records=lambda lo, hi: 0)
        sync = _sync(catalog, client)
        sync._historical = hist  # noqa: SLF001
        assert sync.plan("ES.c.0", "ohlcv-1d", _T0, _T0 + 5 * _DAY) == [(_T0, _T0 + 5 * _DAY)]
        assert hist.count_calls == []


class _ContinuousKeyedClient(FakeClient):
    """Like the real client: continuous defs come back keyed by ES.c.0.GLBX, one per day."""

    async def get_range_instruments(
        self, dataset: str, instrument_ids: list[InstrumentId], start: int, end: int
    ) -> list[FuturesContract]:
        out = []
        for day in range(3):
            spec = _contract("ESZ4", _T0 - 90 * _DAY).to_dict()
            ts = _T0 + day * _DAY
            spec.update(id="ES.c.0.GLBX", raw_symbol="ES.c.0", ts_event=ts, ts_init=ts)
            out.append(FuturesContract.from_dict(spec))
        return out


def test_continuous_keyed_definitions_store_one_synthetic(catalog: ParquetDataCatalog) -> None:
    sync = DatabentoSync(catalog, dataset="GLBX.MDP3", client=_ContinuousKeyedClient())
    assert sync.sync_instruments(["ES.c.0"], _T0, _T0 + 3 * _DAY) == 1
    (inst,) = catalog.instruments()
    assert str(inst.id) == "ES.c.0.GLBX"
    assert inst.expiration_ns == 2**63 - 1  # synthetic: never expires
    assert inst.ts_init == _T0 + 2 * _DAY  # cloned from the most recent version


class TestCost:
    def test_nothing_missing_costs_nothing_without_api_call(
        self, catalog: ParquetDataCatalog, client: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        sync = _sync(catalog, client)
        sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 5 * _DAY)
        assert sync.estimate_cost("ES.c.0", "ohlcv-1d", _T0, _T0 + 5 * _DAY) == 0.0

    def _fake_cost(self, monkeypatch: pytest.MonkeyPatch, cost: Any) -> None:  # noqa: ANN401
        import databento as db

        class _Meta:
            def get_cost(self, *, start: str, end: str, **_: Any) -> float:
                return cost(start, end)

            def get_dataset_range(self, **_: Any) -> dict[str, str]:
                return {"start": "2010-06-06T00:00:00Z", "end": "2100-01-01T00:00:00Z"}

            def get_record_count(self, *, start: str, end: str, **_: Any) -> int:
                cost(start, end)  # same failure mode as get_cost
                return 1

        class _Historical:
            def __init__(self, key: str) -> None:
                self.metadata = _Meta()

        monkeypatch.setattr(db, "Historical", _Historical)
        monkeypatch.setenv("DATABENTO_API_KEY", "test")

    def test_unresolvable_short_window_costs_nothing(
        self, catalog: ParquetDataCatalog, client: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import databento as db

        sync = _sync(catalog, client)
        sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 5 * _DAY)

        def cost(start: str, end: str) -> float:
            raise db.BentoClientError(422, message="symbology_invalid_request")

        self._fake_cost(monkeypatch, cost)
        # Only a 1-day tail is missing (e.g. a weekend): no data there, so $0.
        assert sync.estimate_cost("ES.c.0", "ohlcv-1d", _T0, _T0 + 6 * _DAY) == 0.0

    def test_unresolvable_long_window_still_raises(
        self, catalog: ParquetDataCatalog, client: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import databento as db

        def cost(start: str, end: str) -> float:
            raise db.BentoClientError(422, message="symbology_invalid_request")

        self._fake_cost(monkeypatch, cost)
        with pytest.raises(db.BentoClientError):
            _sync(catalog, client).estimate_cost("ESX.c.0", "ohlcv-1d", _T0, _T0 + 30 * _DAY)


def test_config_tick_schemas_match_sync_specs() -> None:
    from snippy_scales.data.config import TICK_SCHEMAS
    from snippy_scales.data.databento import TICK_SPECS

    assert frozenset(TICK_SPECS) == TICK_SCHEMAS


def test_helpers() -> None:
    assert default_venue("GLBX.MDP3") == "GLBX"
    assert default_venue("XNAS.ITCH") == "XNAS"
    assert _stype_for("ES.c.0") == "continuous"
    assert _stype_for("ES.FUT") == "parent"
    assert _stype_for("ESZ4") == "raw_symbol"


class ParentClient(FakeClient):
    """Parent requests (``ES.FUT``) return bars for two outrights and a calendar spread."""

    async def get_range_bars(
        self,
        dataset: str,
        instrument_ids: list[InstrumentId],
        aggregation: BarAggregation,
        start: int,
        end: int,
        price_precision: int | None = None,
        timestamp_on_close: bool = True,
    ) -> list[Bar]:
        assert str(instrument_ids[0]) == "ES.FUT.GLBX"
        assert price_precision == 2
        self.calls.append(("bars", start, end))
        out = []
        for raw, base in (("ESH4", 4500.0), ("ESM4", 4550.0), ("ESH4-ESM4", -50.0)):
            bar_type = BarType.from_str(f"{raw}.GLBX-1-DAY-LAST-EXTERNAL")
            for open_ns in range(start, end, _DAY):
                px = Price(base + (open_ns - _T0) // _DAY * 0.25, 2)
                close_ns = open_ns + _DAY
                out.append(Bar(bar_type, px, px, px, px, Quantity(10, 0), close_ns, close_ns))
        return out


class TestParentSymbols:
    def test_stype_is_parent_even_if_config_says_continuous(
        self, catalog: ParquetDataCatalog
    ) -> None:
        sync = DatabentoSync(
            catalog, dataset="GLBX.MDP3", stype_in="continuous", client=ParentClient()
        )
        assert sync._stype("ES.FUT") == "parent"  # noqa: SLF001
        assert sync._stype("ES.c.0") == "continuous"  # noqa: SLF001

    def test_instruments_then_bars_per_outright(
        self, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        client = ParentClient()
        sync = DatabentoSync(catalog, dataset="GLBX.MDP3", client=client)
        with pytest.raises(ValueError, match="sync instrument definitions first"):
            sync.sync("ES.FUT", "ohlcv-1d", _T0, _T0 + 5 * _DAY)

        assert sync.sync_instruments(["ES.FUT"], _T0, _T0 + 5 * _DAY) == 2
        assert sorted(str(i.id) for i in catalog.instruments()) == ["ESH4.GLBX", "ESM4.GLBX"]

        # Spread bars are dropped; each outright gets its own series.
        assert sync.sync("ES.FUT", "ohlcv-1d", _T0, _T0 + 5 * _DAY) == 10
        root = tmp_path / "catalog"
        assert sorted(s for s, _ in cat.list_bar_series("ohlcv-1d", root)) == ["ESH4", "ESM4"]
        h4 = cat.load_bars("ESH4", "ohlcv-1d", catalog_path=root)
        assert h4["close"].to_list()[0] == 4500.0

        # Covered now: nothing to plan or fetch; extending the window fetches the tail.
        assert sync.plan("ES.FUT", "ohlcv-1d", _T0, _T0 + 5 * _DAY) == []
        client.calls.clear()
        assert sync.sync("ES.FUT", "ohlcv-1d", _T0, _T0 + 8 * _DAY) == 6
        assert client.calls == [("bars", _T0 + 5 * _DAY, _T0 + 8 * _DAY)]

    def test_instrument_resync_requests_only_after_newest_definition(
        self, catalog: ParquetDataCatalog
    ) -> None:
        client = ParentClient()
        sync = DatabentoSync(catalog, dataset="GLBX.MDP3", client=client)
        sync.sync_instruments(["ES.FUT"], _T0 - 200 * _DAY, _T0 + 5 * _DAY)
        client.calls.clear()
        assert sync.sync_instruments(["ES.FUT"], _T0 - 200 * _DAY, _T0 + 5 * _DAY) == 0
        newest = max(i.ts_init for i in catalog.instruments())
        assert client.calls == [("instruments", max(_T0 - 200 * _DAY, newest), _T0 + 5 * _DAY)]

    def test_parent_ticks_unsupported(self, catalog: ParquetDataCatalog) -> None:
        sync = DatabentoSync(catalog, dataset="GLBX.MDP3", client=ParentClient())
        with pytest.raises(ValueError, match="bar schemas only"):
            sync.plan("ES.FUT", "trades", _T0, _T0 + _DAY)
