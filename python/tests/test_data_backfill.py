"""Unit tests for snippy_scales.data.backfill — window resolution, plan
estimation, symbols cache, and manifest merge logic behind
scripts/pull_sp500_alpaca_1m.py.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from snippy_scales.data.backfill import (
    default_manifest_path,
    estimate_backfill_plan,
    load_symbols_cache,
    merge_manifest,
    read_manifest,
    resolve_backfill_window,
    save_symbols_cache,
)
from snippy_scales.data.ingest import IngestRow

if TYPE_CHECKING:
    from pathlib import Path


class TestResolveBackfillWindow:
    def test_defaults_end_to_today(self) -> None:
        start, end = resolve_backfill_window(5, None)
        assert end == dt.date.today()
        assert start == end.replace(year=end.year - 5)

    def test_explicit_end_string(self) -> None:
        start, end = resolve_backfill_window(5, "2026-09-15")
        assert end == dt.date(2026, 9, 15)
        assert start == dt.date(2021, 9, 15)

    def test_explicit_end_date(self) -> None:
        start, end = resolve_backfill_window(2, dt.date(2024, 1, 1))
        assert (start, end) == (dt.date(2022, 1, 1), dt.date(2024, 1, 1))

    def test_leap_day_end_falls_back_to_feb_28(self) -> None:
        # 2020-02-29 minus 1 year -> 2019 has no Feb 29.
        start, _end = resolve_backfill_window(1, "2020-02-29")
        assert start == dt.date(2019, 2, 28)


class TestEstimateBackfillPlan:
    def test_zero_symbols_gives_zero_requests(self) -> None:
        estimate = estimate_backfill_plan(
            num_symbols=0,
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 1, 2),
            rate_limit_per_min=190,
        )
        assert estimate["total_requests"] == 0

    def test_scales_linearly_with_symbol_count(self) -> None:
        start, end = dt.date(2020, 1, 1), dt.date(2025, 1, 1)
        one = estimate_backfill_plan(num_symbols=1, start=start, end=end, rate_limit_per_min=190)
        ten = estimate_backfill_plan(num_symbols=10, start=start, end=end, rate_limit_per_min=190)
        assert ten["total_requests"] == one["total_requests"] * 10

    def test_end_before_start_gives_zero_trading_days(self) -> None:
        estimate = estimate_backfill_plan(
            num_symbols=5,
            start=dt.date(2024, 6, 1),
            end=dt.date(2024, 1, 1),
            rate_limit_per_min=190,
        )
        assert estimate["trading_days"] == 0
        # Still at least 1 page/symbol (a partial page is still a request).
        assert estimate["total_requests"] == 5

    def test_estimated_minutes_respects_rate_limit(self) -> None:
        start, end = dt.date(2020, 1, 1), dt.date(2025, 1, 1)
        slow = estimate_backfill_plan(num_symbols=100, start=start, end=end, rate_limit_per_min=50)
        fast = estimate_backfill_plan(num_symbols=100, start=start, end=end, rate_limit_per_min=200)
        assert slow["estimated_minutes"] > fast["estimated_minutes"]

    def test_defaults_to_1m_bars_per_trading_day(self) -> None:
        start, end = dt.date(2024, 1, 1), dt.date(2025, 1, 1)
        default = estimate_backfill_plan(
            num_symbols=1, start=start, end=end, rate_limit_per_min=190
        )
        explicit = estimate_backfill_plan(
            num_symbols=1, start=start, end=end, rate_limit_per_min=190, schema="ohlcv-1m"
        )
        assert default == explicit

    def test_daily_schema_needs_far_fewer_requests_than_1m(self) -> None:
        start, end = dt.date(2010, 1, 1), dt.date(2025, 1, 1)
        one_minute = estimate_backfill_plan(
            num_symbols=1, start=start, end=end, rate_limit_per_min=190, schema="ohlcv-1m"
        )
        daily = estimate_backfill_plan(
            num_symbols=1, start=start, end=end, rate_limit_per_min=190, schema="ohlcv-1d"
        )
        assert daily["total_requests"] < one_minute["total_requests"]
        assert daily["total_requests"] == 1  # ~15y of daily bars fits in a single page

    def test_unrecognised_schema_falls_back_to_1m_figure(self) -> None:
        start, end = dt.date(2024, 1, 1), dt.date(2025, 1, 1)
        one_minute = estimate_backfill_plan(
            num_symbols=1, start=start, end=end, rate_limit_per_min=190, schema="ohlcv-1m"
        )
        unknown = estimate_backfill_plan(
            num_symbols=1, start=start, end=end, rate_limit_per_min=190, schema="trades"
        )
        assert unknown == one_minute


class TestDefaultManifestPath:
    def test_path_is_schema_specific(self) -> None:
        assert default_manifest_path("1m").name == "sp500_1m.csv"
        assert default_manifest_path("1d").name == "sp500_1d.csv"
        assert default_manifest_path("1m") != default_manifest_path("1d")


class TestSymbolsCache:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_symbols_cache(tmp_path / "nope.txt") is None

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "universe" / "symbols.txt"
        save_symbols_cache(path, ["AAPL", "MSFT", "BRK.B"])
        assert load_symbols_cache(path) == ["AAPL", "MSFT", "BRK.B"]

    def test_comment_header_is_ignored_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "symbols.txt"
        path.write_text("# a comment\n\nAAPL\n# another\nMSFT\n")
        assert load_symbols_cache(path) == ["AAPL", "MSFT"]


class TestManifest:
    def test_read_missing_manifest_returns_empty_dict(self, tmp_path: Path) -> None:
        assert read_manifest(tmp_path / "manifest.csv") == {}

    def test_merge_writes_new_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        rows = [
            IngestRow("sp500", "ohlcv-1m", "AAPL", True, "data/raw/AAPL/ohlcv-1m.parquet"),
            IngestRow("sp500", "ohlcv-1m", "BADCO", False, "AlpacaCredentialsError"),
        ]
        merged = merge_manifest(path, rows)

        assert set(merged) == {"AAPL", "BADCO"}
        assert merged["AAPL"]["succeeded"] == "True"
        assert merged["BADCO"]["succeeded"] == "False"

        reloaded = read_manifest(path)
        assert reloaded == merged

    def test_merge_preserves_untouched_symbols(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        merge_manifest(
            path,
            [
                IngestRow("sp500", "ohlcv-1m", "AAPL", True, "ok"),
                IngestRow("sp500", "ohlcv-1m", "BADCO", False, "boom"),
            ],
        )

        # A retry-failed run only re-ingests BADCO.
        merge_manifest(path, [IngestRow("sp500", "ohlcv-1m", "BADCO", True, "ok now")])

        final = read_manifest(path)
        assert final["AAPL"]["succeeded"] == "True"
        assert final["BADCO"]["succeeded"] == "True"
        assert final["BADCO"]["detail"] == "ok now"

    def test_merge_updates_existing_symbol_row(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        merge_manifest(path, [IngestRow("sp500", "ohlcv-1m", "AAPL", False, "first try")])
        merge_manifest(path, [IngestRow("sp500", "ohlcv-1m", "AAPL", True, "second try")])

        final = read_manifest(path)
        assert len(final) == 1
        assert final["AAPL"]["succeeded"] == "True"
        assert final["AAPL"]["detail"] == "second try"
