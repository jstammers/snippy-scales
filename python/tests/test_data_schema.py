"""Unit tests for the shared bar-data column schema (snippy_scales.data.schema)."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from snippy_scales.data.schema import BAR_SCHEMA_COLUMNS, conform_bars, empty_bar_frame


def _minimal_ohlcv(n: int = 2) -> pl.DataFrame:
    """Build a bare-minimum OHLCV frame with no metadata columns."""
    return pl.DataFrame(
        {
            "ts_event": [datetime(2024, 1, i + 1, tzinfo=UTC) for i in range(n)],
            "open": [100.0] * n,
            "high": [110.0] * n,
            "low": [90.0] * n,
            "close": [105.0] * n,
            "volume": [1_000] * n,
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
        pl.col("volume").cast(pl.Int64),
    )


class TestConformBars:
    def test_synthesises_missing_metadata_columns(self) -> None:
        out = conform_bars(_minimal_ohlcv(), symbol="AAPL", schema="ohlcv-1m")

        assert list(out.columns) == list(BAR_SCHEMA_COLUMNS)
        assert out["symbol"].to_list() == ["AAPL", "AAPL"]
        assert out["publisher_id"].to_list() == [0, 0]
        assert out["instrument_id"].to_list() == [0, 0]

    def test_rtype_derived_from_schema(self) -> None:
        out = conform_bars(_minimal_ohlcv(n=1), symbol="ES.c.0", schema="ohlcv-1d")
        assert out["rtype"].to_list() == [35]

    def test_unknown_schema_falls_back_to_zero_rtype(self) -> None:
        out = conform_bars(_minimal_ohlcv(n=1), symbol="ES.c.0", schema="ohlcv-eod")
        assert out["rtype"].to_list() == [36]

    def test_existing_metadata_columns_are_preserved_not_overwritten(self) -> None:
        df = _minimal_ohlcv(n=1).with_columns(
            pl.lit(7, dtype=pl.UInt16).alias("publisher_id"),
            pl.lit("REAL_SYMBOL").alias("symbol"),
        )
        out = conform_bars(df, symbol="IGNORED", schema="ohlcv-1d")
        assert out["publisher_id"].to_list() == [7]
        assert out["symbol"].to_list() == ["REAL_SYMBOL"]

    def test_extra_columns_kept_after_core_schema(self) -> None:
        df = _minimal_ohlcv(n=1).with_columns(pl.lit(1.5).alias("vwap"))
        out = conform_bars(df, symbol="AAPL", schema="ohlcv-1m")
        assert out.columns == [*BAR_SCHEMA_COLUMNS, "vwap"]

    def test_dtypes_normalised(self) -> None:
        out = conform_bars(_minimal_ohlcv(n=1), symbol="AAPL", schema="ohlcv-1m")
        assert out.schema["volume"] == pl.UInt64
        assert out.schema["ts_event"] == pl.Datetime("ns", "UTC")
        assert out.schema["rtype"] == pl.UInt8

    def test_missing_required_column_raises(self) -> None:
        df = _minimal_ohlcv(n=1).drop("close")
        with pytest.raises(ValueError, match="missing required column"):
            conform_bars(df, symbol="AAPL", schema="ohlcv-1m")


class TestEmptyBarFrame:
    def test_has_zero_rows_and_shared_schema(self) -> None:
        out = empty_bar_frame(symbol="AAPL", schema="ohlcv-1m")
        assert len(out) == 0
        assert list(out.columns) == list(BAR_SCHEMA_COLUMNS)

    def test_conforms_cleanly_with_conform_bars_output(self) -> None:
        empty = empty_bar_frame(symbol="AAPL", schema="ohlcv-1m")
        nonempty = conform_bars(_minimal_ohlcv(n=1), symbol="AAPL", schema="ohlcv-1m")
        combined = pl.concat([empty, nonempty], how="diagonal")
        assert len(combined) == 1
