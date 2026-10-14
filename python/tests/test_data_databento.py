"""Tests for snippy_scales.data.databento — incremental sync against a fake client.

The real ``DatabentoHistoricalClient`` needs an API key and network access, so
these tests use an async fake with the same method signatures and assert on
which ranges get requested and what lands in the catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    async def get_range_bars(
        self,
        dataset: str,
        instrument_ids: list[InstrumentId],
        aggregation: BarAggregation,
        start: int,
        end: int,
        timestamp_on_close: bool = True,
    ) -> list[Bar]:
        assert timestamp_on_close
        self.calls.append(("bars", start, end))
        bar_type = BarType.from_str(f"{instrument_ids[0]}-1-DAY-LAST-EXTERNAL")
        out = []
        for open_ns in range(start, end, _DAY):
            px = Price(4500 + (open_ns - _T0) // _DAY * 0.25, 2)
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


@pytest.fixture
def catalog(tmp_path: Path) -> ParquetDataCatalog:
    return cat.open_catalog(tmp_path / "catalog")


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


def _sync(catalog: ParquetDataCatalog, client: FakeClient) -> DatabentoSync:
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
        sync.sync("ES.c.0", "ohlcv-1d", _T0 + 5 * _DAY, _T0 + 10 * _DAY)
        client.calls.clear()

        assert sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 15 * _DAY) == 10
        assert client.calls == [
            ("bars", _T0, _T0 + 5 * _DAY),
            ("bars", _T0 + 10 * _DAY, _T0 + 15 * _DAY),
        ]
        df = cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=tmp_path / "catalog")
        assert df.height == 15
        assert df["ts_event"].n_unique() == 15

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
        _sync(catalog, client).sync_instruments(["ES.c.0"], _T0, _T0 + _DAY)
        ids = sorted(str(i.id) for i in catalog.instruments())
        assert ids == ["ES.c.0.GLBX", "ESH4.GLBX", "ESM4.GLBX"]
        cont = next(i for i in catalog.instruments() if str(i.id) == "ES.c.0.GLBX")
        assert str(cont.multiplier) == "50"


class TestCost:
    def test_nothing_missing_costs_nothing_without_api_call(
        self, catalog: ParquetDataCatalog, client: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        sync = _sync(catalog, client)
        sync.sync("ES.c.0", "ohlcv-1d", _T0, _T0 + 5 * _DAY)
        assert sync.estimate_cost("ES.c.0", "ohlcv-1d", _T0, _T0 + 5 * _DAY) == 0.0


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
