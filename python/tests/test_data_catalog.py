"""Tests for snippy_scales.data.catalog — naming, conversion, incremental writes, loading."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl
import pytest
from nautilus_trader.model import (
    AssetClass,
    Currency,
    FuturesContract,
    InstrumentId,
    Price,
    Quantity,
    Symbol,
)

from snippy_scales.data import catalog as cat

if TYPE_CHECKING:
    from pathlib import Path

    from nautilus_trader.persistence import ParquetDataCatalog

_DAY = 86_400_000_000_000


def _daily_frame(start: dt.date, n: int, base: float = 4500.25) -> pl.DataFrame:
    days = [start + dt.timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "ts_event": pl.Series(days).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"),
            "open": [base + i for i in range(n)],
            "high": [base + i + 1.5 for i in range(n)],
            "low": [base + i - 1.25 for i in range(n)],
            "close": [base + i + 0.25 for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        }
    )


def _es_bars(start: dt.date, n: int) -> list:
    return cat.frame_to_bars(
        _daily_frame(start, n),
        bar_type=cat.bar_type("ES.c.0", "GLBX", "ohlcv-1d"),
        price_precision=2,
        interval_ns=_DAY,
    )


@pytest.fixture
def catalog(tmp_path: Path) -> ParquetDataCatalog:
    return cat.open_catalog(tmp_path / "catalog")


class TestNaming:
    def test_bar_type(self) -> None:
        assert cat.bar_type("ES.c.0", "GLBX", "ohlcv-1d") == "ES.c.0.GLBX-1-DAY-LAST-EXTERNAL"
        assert cat.bar_type("BRK.B", "ALPACA", "ohlcv-1m") == "BRK.B.ALPACA-1-MINUTE-LAST-EXTERNAL"

    def test_unsupported_schema(self) -> None:
        with pytest.raises(ValueError, match="Unsupported bar schema"):
            cat.bar_type("ES.c.0", "GLBX", "ohlcv-eod")

    def test_list_and_resolve(self, catalog: ParquetDataCatalog, tmp_path: Path) -> None:
        root = tmp_path / "catalog"
        cat.write_bars(catalog, _es_bars(dt.date(2024, 1, 1), 3))
        assert cat.list_bar_series("ohlcv-1d", root) == [("ES.c.0", "GLBX")]
        assert cat.list_bar_series("ohlcv-1m", root) == []
        assert cat.resolve_bar_type("ES.c.0", "ohlcv-1d", catalog_path=root).startswith(
            "ES.c.0.GLBX-"
        )
        with pytest.raises(FileNotFoundError):
            cat.resolve_bar_type("ES", "ohlcv-1d", catalog_path=root)

    def test_ambiguous_venue(self, catalog: ParquetDataCatalog, tmp_path: Path) -> None:
        for venue in ("GLBX", "XCME"):
            bars = cat.frame_to_bars(
                _daily_frame(dt.date(2024, 1, 1), 2),
                bar_type=cat.bar_type("ES.c.0", venue, "ohlcv-1d"),
                price_precision=2,
                interval_ns=_DAY,
            )
            cat.write_bars(catalog, bars)
        with pytest.raises(ValueError, match="several venues"):
            cat.resolve_bar_type("ES.c.0", "ohlcv-1d", catalog_path=tmp_path / "catalog")


class TestFrameToBars:
    def test_ts_init_is_bar_close(self) -> None:
        bars = _es_bars(dt.date(2024, 1, 1), 2)
        assert bars[0].ts_init - bars[0].ts_event == _DAY

    def test_prices_are_exact_at_precision(
        self, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        frame = _daily_frame(dt.date(2024, 1, 1), 1).with_columns(pl.lit(100.05).alias("high"))
        bars = cat.frame_to_bars(
            frame,
            bar_type=cat.bar_type("X", "SIM", "ohlcv-1d"),
            price_precision=2,
            interval_ns=_DAY,
        )
        catalog.write_bars(bars)
        files = list((tmp_path / "catalog").rglob("data/bars/*/*.parquet"))
        stored = pl.read_parquet(files[0])["high"][0]
        assert str(stored).rstrip("0") == "100.05"

    def test_empty_frame(self) -> None:
        empty = _daily_frame(dt.date(2024, 1, 1), 0)
        assert (
            cat.frame_to_bars(
                empty, bar_type="X.SIM-1-DAY-LAST-EXTERNAL", price_precision=2, interval_ns=_DAY
            )
            == []
        )


class TestWriteBars:
    def test_incremental_append_and_prepend(
        self, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        root = tmp_path / "catalog"
        assert cat.write_bars(catalog, _es_bars(dt.date(2024, 1, 10), 10)) == 10
        # Overlapping tail: only the 5 new days after the last stored bar are written.
        assert cat.write_bars(catalog, _es_bars(dt.date(2024, 1, 15), 10)) == 5
        # Backfill before coverage: the 9 earlier days are written, the overlap skipped.
        assert cat.write_bars(catalog, _es_bars(dt.date(2024, 1, 1), 12)) == 9
        # Fully covered: nothing written.
        assert cat.write_bars(catalog, _es_bars(dt.date(2024, 1, 5), 5)) == 0
        df = cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=root)
        assert df.height == 24
        assert df["ts_event"].is_sorted()
        assert df["ts_event"].n_unique() == 24

    def test_rejects_mixed_bar_types(self, catalog: ParquetDataCatalog) -> None:
        mixed = _es_bars(dt.date(2024, 1, 1), 1) + cat.frame_to_bars(
            _daily_frame(dt.date(2024, 1, 1), 1),
            bar_type=cat.bar_type("NQ.c.0", "GLBX", "ohlcv-1d"),
            price_precision=2,
            interval_ns=_DAY,
        )
        with pytest.raises(ValueError, match="one bar type"):
            cat.write_bars(catalog, mixed)


class TestLoadBars:
    def test_shape_and_values(self, catalog: ParquetDataCatalog, tmp_path: Path) -> None:
        cat.write_bars(catalog, _es_bars(dt.date(2024, 1, 1), 3))
        df = cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=tmp_path / "catalog")
        assert tuple(df.columns) == cat.BAR_COLUMNS
        assert df.schema["close"] == pl.Float64
        assert df.schema["ts_event"] == pl.Datetime("ns", "UTC")
        assert df["close"].to_list() == [4500.5, 4501.5, 4502.5]
        assert df["symbol"].unique().to_list() == ["ES.c.0"]
        assert df["ts_event"][0] == dt.datetime(2024, 1, 1, tzinfo=dt.UTC)

    def test_start_end_filter_on_bar_close(
        self, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        cat.write_bars(catalog, _es_bars(dt.date(2024, 1, 1), 10))
        df = cat.load_bars(
            "ES.c.0",
            "ohlcv-1d",
            start="2024-01-04",
            end=dt.date(2024, 1, 6),
            catalog_path=tmp_path / "catalog",
        )
        # Bars closing on Jan 4, 5 and 6 are the ones opening Jan 3, 4 and 5.
        assert [d.day for d in df["ts_event"].to_list()] == [3, 4, 5]

    def test_missing_catalog(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No catalog"):
            cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=tmp_path / "nope")


class TestPrecision:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([4500.0, 4501.0], 0),
            ([4500.25, 4500.5], 2),
            ([110.015625, 110.03125], 6),
            ([0.0067235, 0.006724], 7),
        ],
    )
    def test_infer_price_precision(self, values: list[float], expected: int) -> None:
        df = pl.DataFrame({c: values for c in ("open", "high", "low", "close")})
        assert cat.infer_price_precision(df) == expected


class TestInstruments:
    def test_equity_instrument(self) -> None:
        eq = cat.equity_instrument("BRK.B", "ALPACA")
        assert str(eq.id) == "BRK.B.ALPACA"
        assert eq.price_precision == 4
        assert str(eq.price_increment) == "0.0001"

    def test_continuous_instrument_copies_contract_spec(self) -> None:
        contract = FuturesContract(
            InstrumentId.from_str("ESZ6.GLBX"),
            Symbol("ESZ6"),
            AssetClass.INDEX,
            "ES",
            1,
            2,
            Currency.from_str("USD"),
            2,
            Price.from_str("0.25"),
            Quantity.from_int(50),
            Quantity.from_int(1),
            0,
            0,
        )
        cont = cat.continuous_instrument(contract, "ES.c.0")
        assert str(cont.id) == "ES.c.0.GLBX"
        assert str(cont.multiplier) == "50"
        assert str(cont.price_increment) == "0.25"
        assert cont.expiration_ns > contract.expiration_ns

    def test_write_instruments_skips_existing(self, catalog: ParquetDataCatalog) -> None:
        cat.write_instruments(catalog, [cat.equity_instrument("AAPL", "ALPACA")])
        cat.write_instruments(
            catalog,
            [cat.equity_instrument("AAPL", "ALPACA"), cat.equity_instrument("MSFT", "ALPACA")],
        )
        assert sorted(str(i.id) for i in catalog.instruments()) == ["AAPL.ALPACA", "MSFT.ALPACA"]
