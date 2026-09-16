"""Unit tests for the Alpaca dispatch paths in snippy_scales.data.ingest:
`upsert_bars` timestamp-granularity resume, and `ingest_from_config` /
`estimate_costs_from_config` routing when `provider == "alpaca"`.

A fake in-memory `BarProvider` stands in for `AlpacaProvider` throughout —
these tests exercise the generic dispatch/resume logic, not the Alpaca SDK
integration itself (covered by test_data_alpaca_provider.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import polars as pl

from snippy_scales.data.config import AssetClassConfig, IngestConfig
from snippy_scales.data.ingest import (
    estimate_costs_from_config,
    ingest_from_config,
    is_range_cached,
    upsert_bars,
)
from snippy_scales.data.schema import conform_bars

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


@dataclass
class _FakeTimestampProvider:
    """Records every fetch_bars() call and returns one bar per call."""

    name: str = field(default="fake", init=False)
    resume_granularity: Literal["day", "timestamp"] = field(default="timestamp", init=False)
    calls: list[tuple[str, str, str, str]] = field(default_factory=list)

    def fetch_bars(self, *, symbol: str, schema: str, start: str, end: str) -> pl.DataFrame:
        self.calls.append((symbol, schema, start, end))
        df = pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 1, 20, 45, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [10],
            }
        )
        return conform_bars(df, symbol=symbol, schema=schema)


class TestUpsertBarsTimestampResume:
    def test_resume_uses_exact_timestamp_not_next_calendar_day(self, tmp_path: Path) -> None:
        """A bar at 20:45 UTC (crossing into the next UTC day for extended-hours
        sessions in some timezones) must resume from just after that exact
        timestamp, not "the next calendar day" — which would skip data.
        """
        provider = _FakeTimestampProvider()
        upsert_bars(
            provider=provider,
            symbol="AAPL",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-01-03",
            output_dir=tmp_path,
            instrument_type="equities",
        )
        provider.fetch_bars(symbol="AAPL", schema="ohlcv-1m", start="ignored", end="ignored")
        # Second upsert call should ask for data starting just after 20:45:00,
        # same calendar day — not 2024-01-02.
        upsert_bars(
            provider=provider,
            symbol="AAPL",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-01-03",
            output_dir=tmp_path,
            instrument_type="equities",
        )
        second_call_start = provider.calls[-1][2]
        assert second_call_start.startswith("2024-01-01T20:45:00.000001")

    def test_no_refetch_when_already_covered(self, tmp_path: Path) -> None:
        provider = _FakeTimestampProvider()
        upsert_bars(
            provider=provider,
            symbol="AAPL",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-01-02",
            output_dir=tmp_path,
            instrument_type="equities",
        )
        assert len(provider.calls) == 1

        upsert_bars(
            provider=provider,
            symbol="AAPL",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-01-01T20:45:00",  # already covered by the single stored bar
            output_dir=tmp_path,
            instrument_type="equities",
        )
        assert len(provider.calls) == 1  # no second fetch


class TestIsRangeCached:
    def test_false_when_no_file(self, tmp_path: Path) -> None:
        assert (
            is_range_cached(
                resume_granularity="timestamp",
                symbol="AAPL",
                schema="ohlcv-1m",
                start="2024-01-01",
                end="2024-01-02",
                output_dir=tmp_path,
                instrument_type="equities",
            )
            is False
        )

    def test_true_when_fully_covered(self, tmp_path: Path) -> None:
        provider = _FakeTimestampProvider()
        upsert_bars(
            provider=provider,
            symbol="AAPL",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-01-02",
            output_dir=tmp_path,
            instrument_type="equities",
        )
        assert (
            is_range_cached(
                resume_granularity="timestamp",
                symbol="AAPL",
                schema="ohlcv-1m",
                start="2024-01-01",
                end="2024-01-01T20:45:00.000001",
                output_dir=tmp_path,
                instrument_type="equities",
            )
            is True
        )


class TestIngestFromConfigAlpacaDispatch:
    def _config(self, symbols: list[str]) -> IngestConfig:
        return IngestConfig(
            provider="alpaca",
            schemas=["1m"],
            start="2024-01-01",
            end="2024-01-02",
            asset_classes={"eq": AssetClassConfig(symbols=symbols)},
            instrument_type="equities",
        )

    def test_routes_through_alpaca_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

        calls: list[str] = []

        class _StubAlpacaProvider:
            name = "alpaca"
            resume_granularity: Literal["timestamp"] = "timestamp"

            def __init__(self, **_kwargs: object) -> None:
                pass

            def fetch_bars(self, *, symbol: str, schema: str, start: str, end: str) -> pl.DataFrame:
                calls.append(symbol)
                df = pl.DataFrame(
                    {
                        "ts_event": [datetime(2024, 1, 1, 15, 0, tzinfo=UTC)],
                        "open": [1.0],
                        "high": [1.0],
                        "low": [1.0],
                        "close": [1.0],
                        "volume": [1],
                    }
                )
                return conform_bars(df, symbol=symbol, schema=schema)

        monkeypatch.setattr(
            "snippy_scales.data.providers.alpaca.AlpacaProvider", _StubAlpacaProvider
        )

        rows = ingest_from_config(self._config(["AAPL", "MSFT"]), output_dir=tmp_path)

        assert {row.symbol for row in rows} == {"AAPL", "MSFT"}
        assert all(row.succeeded for row in rows)
        assert set(calls) == {"AAPL", "MSFT"}

    def test_estimate_costs_are_free_and_make_no_network_call(self, tmp_path: Path) -> None:
        rows = estimate_costs_from_config(self._config(["AAPL", "MSFT"]), output_dir=tmp_path)
        assert all(row.cost_usd == 0.0 for row in rows)
        assert all(row.cached is False for row in rows)  # nothing on disk yet

    def test_estimate_costs_reports_cached_when_already_downloaded(self, tmp_path: Path) -> None:
        provider = _FakeTimestampProvider()
        upsert_bars(
            provider=provider,
            symbol="AAPL",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-01-02",
            output_dir=tmp_path,
            instrument_type="equities",
        )

        cfg = IngestConfig(
            provider="alpaca",
            schemas=["1m"],
            start="2024-01-01",
            end="2024-01-01T20:45:00.000001",
            asset_classes={"eq": AssetClassConfig(symbols=["AAPL"])},
            instrument_type="equities",
        )
        rows = estimate_costs_from_config(cfg, output_dir=tmp_path)
        assert rows[0].cached is True
        assert rows[0].cost_usd == 0.0
