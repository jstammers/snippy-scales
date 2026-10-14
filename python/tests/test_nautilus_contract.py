"""Contract tests pinning the NautilusTrader catalog behaviour this repo relies on.

``nautilus_trader`` is pinned to a release candidate whose catalog API and on-disk
format have changed between releases. These tests assert the specific behaviour
``snippy_scales.data`` depends on, so an upgrade that changes any of it fails
here first instead of silently corrupting the datastore.

See docs/architecture/nautilus_migration.md ("Verified facts").
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nautilus_trader.model import (
    Bar,
    BarType,
    Currency,
    Equity,
    InstrumentId,
    NautilusDataType,
    Price,
    Quantity,
    Symbol,
)
from nautilus_trader.persistence import BarDataWrangler, ParquetDataCatalog

_MINUTE_NS = 60_000_000_000
_BAR_TYPE = "AAPL.XNAS-1-MINUTE-LAST-EXTERNAL"


def _bars(start: int, n: int, close: float = 100.0) -> list[Bar]:
    """*n* one-minute bars; bar *i* opens at minute ``start + i`` and closes a minute later."""
    bar_type = BarType.from_str(_BAR_TYPE)
    out = []
    for i in range(start, start + n):
        px = Price(close + i * 0.01, 2)
        out.append(
            Bar(bar_type, px, px, px, px, Quantity(10, 0), i * _MINUTE_NS, (i + 1) * _MINUTE_NS)
        )
    return out


@pytest.fixture
def catalog(tmp_path: Path) -> ParquetDataCatalog:
    return ParquetDataCatalog(str(tmp_path))


def test_bars_are_stored_as_plain_arrow_decimal_and_timestamp(
    catalog: ParquetDataCatalog, tmp_path: Path
) -> None:
    catalog.write_bars(_bars(1, 5))

    files = sorted((tmp_path / "data" / "bars" / _BAR_TYPE).glob("*.parquet"))
    assert len(files) == 1
    schema = pq.read_schema(files[0])
    for col in ("open", "high", "low", "close", "volume"):
        assert schema.field(col).type == pa.decimal128(38, 16), col
    for col in ("ts_event", "ts_init"):
        assert schema.field(col).type == pa.timestamp("ns", tz="UTC"), col


def test_overlapping_write_raises(catalog: ParquetDataCatalog) -> None:
    catalog.write_bars(_bars(1, 10))
    with pytest.raises(OSError, match="non-disjoint"):
        catalog.write_bars(_bars(5, 10))


def test_disjoint_append_is_accepted(catalog: ParquetDataCatalog) -> None:
    catalog.write_bars(_bars(1, 10))
    catalog.write_bars(_bars(11, 10))
    assert len(catalog.query_bars([_BAR_TYPE])) == 20


def test_query_last_timestamp_is_last_ts_init(catalog: ParquetDataCatalog) -> None:
    catalog.write_bars(_bars(1, 3))
    # Last bar opens at minute 3, so its ts_init (bar close) is minute 4.
    assert catalog.query_last_timestamp(NautilusDataType.Bar, _BAR_TYPE) == 4 * _MINUTE_NS


def test_query_last_timestamp_is_none_for_unknown_series(catalog: ParquetDataCatalog) -> None:
    assert catalog.query_last_timestamp(NautilusDataType.Bar, _BAR_TYPE) is None


def test_display_arrow_query_returns_float64(catalog: ParquetDataCatalog) -> None:
    catalog.write_bars(_bars(1, 3))
    payload = catalog.query_data_arrow_bytes(NautilusDataType.Bar, [_BAR_TYPE], display=True)
    table = pa.ipc.open_stream(payload).read_all()

    for col in ("open", "high", "low", "close", "volume"):
        assert table.schema.field(col).type == pa.float64(), col
    assert table.schema.field("ts_event").type == pa.timestamp("ns", tz="UTC")
    assert table.column("close").to_pylist() == pytest.approx([100.01, 100.02, 100.03])


def test_bar_wrangler_requires_decimal_price_columns() -> None:
    batch = pa.RecordBatch.from_pydict(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
            "ts_event": pa.array([0], pa.uint64()),
            "ts_init": pa.array([_MINUTE_NS], pa.uint64()),
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)

    wrangler = BarDataWrangler(_BAR_TYPE, 2, 0)
    with pytest.raises(ValueError, match="Decimal128"):
        wrangler.process_record_batch_bytes(sink.getvalue().to_pybytes())


def test_instruments_round_trip(catalog: ParquetDataCatalog) -> None:
    equity = Equity(
        InstrumentId.from_str("AAPL.XNAS"),
        Symbol("AAPL"),
        Currency.from_str("USD"),
        2,
        Price.from_str("0.01"),
        0,
        0,
    )
    catalog.write_instruments([equity])
    assert [str(i.id) for i in catalog.instruments()] == ["AAPL.XNAS"]
