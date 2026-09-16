"""Unit tests for event-level (tick) ingestion — coverage, cost, and upsert logic.

All tests are fully offline: Databento API calls are mocked so no credentials
or network access are required.
"""

from __future__ import annotations

import datetime as dt
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pandas as pd
import polars as pl
import pytest
from typer.testing import CliRunner

from snippy_scales.data.config import (
    TICK_SCHEMAS,
    frequency_to_schema,
    is_tick_schema,
    resolve_schema,
)
from snippy_scales.data.tick import (
    TickCostEstimate,
    _day_dbn,
    _day_empty_marker,
    _day_parquet,
    _tick_root,
    contiguous_ranges,
    covered_days,
    estimate_tick_cost,
    load_ticks,
    missing_days,
    upsert_ticks,
)

D = dt.date


# ===========================================================================
# Helpers
# ===========================================================================


@contextmanager
def _mock_databento(client_mock: MagicMock) -> Generator[MagicMock, None, None]:
    """Inject a fake ``databento`` module into sys.modules.

    Mirrors the idiom in ``test_data_ingest.py``: ``tick.py`` uses
    ``import databento as db`` *inside* function bodies, so patching the
    ``databento`` key in ``sys.modules`` intercepts the import at call time and
    no real package, credentials, or network are needed.
    """
    fake_db = MagicMock()
    fake_db.Historical.return_value = client_mock
    with patch.dict(sys.modules, {"databento": fake_db}):
        yield fake_db


def _seed_day(root: Path, day: str, *, rows: int = 3) -> None:
    """Write a fake Parquet partition so *day* counts as covered."""
    path = _day_parquet(root, D.fromisoformat(day))
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_event": list(range(rows)), "price": [100.0] * rows}).write_parquet(path)


def _seed_empty_day(root: Path, day: str) -> None:
    """Write an empty-day marker so *day* counts as covered with no data."""
    marker = _day_empty_marker(root, D.fromisoformat(day))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


# ===========================================================================
# Schema resolution
# ===========================================================================


class TestResolveSchema:
    def test_tick_schemas_pass_through(self) -> None:
        for schema in TICK_SCHEMAS:
            assert resolve_schema(schema) == schema

    def test_bar_aliases_still_work(self) -> None:
        assert resolve_schema("daily") == "ohlcv-1d"
        assert resolve_schema("1h") == "ohlcv-1h"
        assert resolve_schema("ohlcv-1m") == "ohlcv-1m"

    def test_case_and_whitespace_insensitive(self) -> None:
        assert resolve_schema("  TRADES ") == "trades"

    def test_unknown_raises_listing_both_families(self) -> None:
        with pytest.raises(ValueError, match="Unknown schema"):
            resolve_schema("orderbook")

    def test_frequency_to_schema_still_rejects_tick_names(self) -> None:
        # resolve_schema is the superset; the bar-only helper is unchanged.
        with pytest.raises(ValueError, match="Unknown frequency"):
            frequency_to_schema("trades")


class TestIsTickSchema:
    def test_true_for_event_schemas(self) -> None:
        assert is_tick_schema("trades")
        assert is_tick_schema("mbo")

    def test_false_for_bar_schemas(self) -> None:
        assert not is_tick_schema("ohlcv-1d")


# ===========================================================================
# Coverage
# ===========================================================================


class TestCoveredDays:
    def test_empty_store(self, tmp_path: Path) -> None:
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == set()

    def test_counts_parquet_partitions(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-03")
        _seed_day(root, "2026-08-04")
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == {
            D(2026, 8, 3),
            D(2026, 8, 4),
        }

    def test_counts_empty_markers(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_empty_day(root, "2026-08-01")
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == {
            D(2026, 8, 1)
        }

    def test_partition_without_parquet_is_not_covered(self, tmp_path: Path) -> None:
        # Simulates an interrupted download: the directory exists but the
        # atomic rename never happened.
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        (root / "date=2026-08-03").mkdir(parents=True)
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == set()

    def test_tmp_file_is_not_covered(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        root.mkdir(parents=True)
        (root / ".tmp-2026-08-03.parquet").write_bytes(b"partial")
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == set()

    def test_schemas_are_isolated(self, tmp_path: Path) -> None:
        _seed_day(_tick_root("ES.c.0", "trades", tmp_path, "futures"), "2026-08-03")
        assert covered_days("ES.c.0", "mbo", tmp_path, instrument_type="futures") == set()


class TestMissingDays:
    def test_all_missing_when_store_empty(self, tmp_path: Path) -> None:
        result = missing_days(
            "ES.c.0", "trades", "2026-08-01", "2026-08-04", tmp_path, instrument_type="futures"
        )
        assert result == [D(2026, 8, 1), D(2026, 8, 2), D(2026, 8, 3)]

    def test_end_is_exclusive(self, tmp_path: Path) -> None:
        result = missing_days(
            "ES.c.0", "trades", "2026-08-01", "2026-08-02", tmp_path, instrument_type="futures"
        )
        assert result == [D(2026, 8, 1)]

    def test_none_missing_when_fully_covered(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        for day in ("2026-08-01", "2026-08-02", "2026-08-03"):
            _seed_day(root, day)
        assert (
            missing_days(
                "ES.c.0", "trades", "2026-08-01", "2026-08-04", tmp_path, instrument_type="futures"
            )
            == []
        )

    def test_detects_interior_gap(self, tmp_path: Path) -> None:
        # The bar high-water-mark logic cannot see this; coverage-set logic can.
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-01")
        _seed_day(root, "2026-08-04")
        result = missing_days(
            "ES.c.0", "trades", "2026-08-01", "2026-08-05", tmp_path, instrument_type="futures"
        )
        assert result == [D(2026, 8, 2), D(2026, 8, 3)]

    def test_backfills_below_existing_start(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-05")
        result = missing_days(
            "ES.c.0", "trades", "2026-08-03", "2026-08-06", tmp_path, instrument_type="futures"
        )
        assert result == [D(2026, 8, 3), D(2026, 8, 4)]

    def test_inverted_range_yields_nothing(self, tmp_path: Path) -> None:
        assert (
            missing_days(
                "ES.c.0", "trades", "2026-08-05", "2026-08-01", tmp_path, instrument_type="futures"
            )
            == []
        )

    def test_accepts_date_objects(self, tmp_path: Path) -> None:
        result = missing_days(
            "ES.c.0", "trades", D(2026, 8, 1), D(2026, 8, 3), tmp_path, instrument_type="futures"
        )
        assert result == [D(2026, 8, 1), D(2026, 8, 2)]


class TestContiguousRanges:
    def test_empty(self) -> None:
        assert contiguous_ranges([]) == []

    def test_single_day(self) -> None:
        assert contiguous_ranges([D(2026, 8, 1)]) == [(D(2026, 8, 1), D(2026, 8, 2))]

    def test_one_run(self) -> None:
        days = [D(2026, 8, 1), D(2026, 8, 2), D(2026, 8, 3)]
        assert contiguous_ranges(days) == [(D(2026, 8, 1), D(2026, 8, 4))]

    def test_split_runs(self) -> None:
        days = [D(2026, 8, 1), D(2026, 8, 2), D(2026, 8, 5)]
        assert contiguous_ranges(days) == [
            (D(2026, 8, 1), D(2026, 8, 3)),
            (D(2026, 8, 5), D(2026, 8, 6)),
        ]

    def test_crosses_month_boundary(self) -> None:
        days = [D(2026, 8, 31), D(2026, 9, 1)]
        assert contiguous_ranges(days) == [(D(2026, 8, 31), D(2026, 9, 2))]


# ===========================================================================
# estimate_tick_cost
# ===========================================================================


class TestEstimateTickCost:
    def test_no_api_call_when_fully_cached(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        for day in ("2026-08-01", "2026-08-02"):
            _seed_day(root, day)

        client = MagicMock()
        with _mock_databento(client):
            estimate = estimate_tick_cost(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-03",
                output_dir=tmp_path,
                instrument_type="futures",
            )

        client.metadata.get_cost.assert_not_called()
        client.metadata.get_billable_size.assert_not_called()
        assert estimate.cost_usd == 0.0
        assert estimate.billable_bytes == 0
        assert estimate.missing_days == []
        assert estimate.cached_days == 2

    def test_sums_across_contiguous_ranges(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-03")  # splits the request into two runs

        client = MagicMock()
        client.metadata.get_cost.return_value = 1.5
        client.metadata.get_billable_size.return_value = 2_000_000_000

        with _mock_databento(client):
            estimate = estimate_tick_cost(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-06",
                output_dir=tmp_path,
                instrument_type="futures",
            )

        assert client.metadata.get_cost.call_count == 2
        assert estimate.cost_usd == 3.0
        assert estimate.billable_bytes == 4_000_000_000
        assert estimate.billable_gb == pytest.approx(4.0)
        assert estimate.missing_days == [D(2026, 8, 1), D(2026, 8, 2), D(2026, 8, 4), D(2026, 8, 5)]
        assert estimate.cached_days == 1

    def test_excludes_cached_days_from_the_request(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-01")

        client = MagicMock()
        client.metadata.get_cost.return_value = 1.0
        client.metadata.get_billable_size.return_value = 1

        with _mock_databento(client):
            estimate_tick_cost(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-04",
                output_dir=tmp_path,
                instrument_type="futures",
            )

        kwargs = client.metadata.get_cost.call_args.kwargs
        assert kwargs["start"] == "2026-08-02"
        assert kwargs["end"] == "2026-08-04"

    def test_forwards_stype_in(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.metadata.get_cost.return_value = 1.0
        client.metadata.get_billable_size.return_value = 1

        with _mock_databento(client):
            estimate_tick_cost(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="mbo",
                start="2026-08-01",
                end="2026-08-02",
                output_dir=tmp_path,
                stype_in="continuous",
                instrument_type="futures",
            )

        assert client.metadata.get_cost.call_args.kwargs["stype_in"] == "continuous"
        assert client.metadata.get_cost.call_args.kwargs["schema"] == "mbo"


# ===========================================================================
# upsert_ticks
# ===========================================================================


def _install_fake_download(fake_db: MagicMock, *, rows: int = 2) -> None:
    """Make the mocked DBNStore.to_parquet write a real Parquet temp file."""

    def _to_parquet(path: object, **_: object) -> None:
        pl.DataFrame({"ts_event": list(range(rows)), "price": [100.0] * rows}).write_parquet(
            str(path)
        )

    store = MagicMock()
    store.to_parquet.side_effect = _to_parquet
    fake_db.DBNStore.from_file.return_value = store


class TestUpsertTicks:
    def test_one_request_per_missing_day(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.timeseries.get_range.side_effect = lambda **kw: Path(kw["path"]).touch()

        with _mock_databento(client) as fake_db:
            _install_fake_download(fake_db)
            written = upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-04",
                output_dir=tmp_path,
                instrument_type="futures",
                download_method="streaming",
            )

        assert client.timeseries.get_range.call_count == 3
        assert len(written) == 3
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == {
            D(2026, 8, 1),
            D(2026, 8, 2),
            D(2026, 8, 3),
        }

    def test_requests_each_day_as_half_open_range(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.timeseries.get_range.side_effect = lambda **kw: Path(kw["path"]).touch()

        with _mock_databento(client) as fake_db:
            _install_fake_download(fake_db)
            upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-02",
                output_dir=tmp_path,
                instrument_type="futures",
                download_method="streaming",
            )

        kwargs = client.timeseries.get_range.call_args.kwargs
        assert kwargs["start"] == "2026-08-01"
        assert kwargs["end"] == "2026-08-02"
        assert kwargs["schema"] == "trades"
        assert kwargs["path"].endswith(".dbn.zst")

    def test_no_request_when_already_covered(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        for day in ("2026-08-01", "2026-08-02"):
            _seed_day(root, day)

        client = MagicMock()
        with _mock_databento(client):
            written = upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-03",
                output_dir=tmp_path,
                instrument_type="futures",
                download_method="streaming",
            )

        client.timeseries.get_range.assert_not_called()
        assert written == []

    def test_only_missing_days_are_fetched(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-02")

        client = MagicMock()
        client.timeseries.get_range.side_effect = lambda **kw: Path(kw["path"]).touch()

        with _mock_databento(client) as fake_db:
            _install_fake_download(fake_db)
            upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-04",
                output_dir=tmp_path,
                instrument_type="futures",
                download_method="streaming",
            )

        fetched = {kw.kwargs["start"] for kw in client.timeseries.get_range.call_args_list}
        assert fetched == {"2026-08-01", "2026-08-03"}

    def test_empty_day_gets_a_marker_and_is_not_refetched(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.timeseries.get_range.side_effect = lambda **kw: Path(kw["path"]).touch()

        # to_parquet writes nothing when the store holds no records.
        store = MagicMock()
        store.to_parquet.side_effect = lambda *_, **__: None

        with _mock_databento(client) as fake_db:
            fake_db.DBNStore.from_file.return_value = store
            written = upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-02",
                output_dir=tmp_path,
                instrument_type="futures",
                download_method="streaming",
            )

        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        assert written == []
        assert _day_empty_marker(root, D(2026, 8, 1)).exists()
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == {
            D(2026, 8, 1)
        }
        assert (
            missing_days(
                "ES.c.0", "trades", "2026-08-01", "2026-08-02", tmp_path, instrument_type="futures"
            )
            == []
        )

    def test_keeps_the_raw_dbn(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.timeseries.get_range.side_effect = lambda **kw: Path(kw["path"]).touch()

        with _mock_databento(client) as fake_db:
            _install_fake_download(fake_db)
            upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-02",
                output_dir=tmp_path,
                instrument_type="futures",
                download_method="streaming",
            )

        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        assert _day_dbn(root, D(2026, 8, 1)).exists()

    def test_failed_day_does_not_abort_the_run(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def _get_range(**kw: object) -> None:
            start = str(kw["start"])
            calls.append(start)
            if start == "2026-08-02":
                raise RuntimeError("boom")
            Path(str(kw["path"])).touch()

        client = MagicMock()
        client.timeseries.get_range.side_effect = _get_range

        with _mock_databento(client) as fake_db:
            _install_fake_download(fake_db)
            written = upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-04",
                output_dir=tmp_path,
                instrument_type="futures",
                download_method="streaming",
            )

        assert calls == ["2026-08-01", "2026-08-02", "2026-08-03"]
        assert len(written) == 2
        # The failed day stays missing, so a re-run picks it up.
        assert missing_days(
            "ES.c.0", "trades", "2026-08-01", "2026-08-04", tmp_path, instrument_type="futures"
        ) == [D(2026, 8, 2)]

    def test_rerun_is_a_no_op(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.timeseries.get_range.side_effect = lambda **kw: Path(kw["path"]).touch()

        with _mock_databento(client) as fake_db:
            _install_fake_download(fake_db)
            for _ in range(2):
                upsert_ticks(
                    dataset="GLBX.MDP3",
                    symbol="ES.c.0",
                    schema="trades",
                    start="2026-08-01",
                    end="2026-08-03",
                    output_dir=tmp_path,
                    instrument_type="futures",
                    download_method="streaming",
                )

        assert client.timeseries.get_range.call_count == 2  # not 4


# ===========================================================================
# upsert_ticks — batch download path (the default)
# ===========================================================================


def _fake_tick_store(day: str, *, rows: int = 2) -> MagicMock:
    """A DBNStore stand-in for one calendar day's worth of a batch download."""
    start = pd.Timestamp(f"{day}T00:00:00", tz="UTC")
    store = MagicMock()
    store.start = start
    store.end = start + pd.Timedelta(days=1)

    def _to_parquet(path: object, **_: object) -> None:
        pl.DataFrame({"ts_event": list(range(rows)), "price": [100.0] * rows}).write_parquet(
            str(path)
        )

    store.to_parquet.side_effect = _to_parquet
    return store


class TestUpsertTicksBatch:
    def test_submits_one_job_per_contiguous_range(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.side_effect = [{"id": "job-1"}, {"id": "job-2"}]
        client.batch.list_jobs.side_effect = [
            [],  # reuse-check for job-1
            [{"id": "job-1", "state": "done"}],
            [],  # reuse-check for job-2
            [{"id": "job-2", "state": "done"}],
        ]

        def _download(job_id: str, output_dir: Path) -> list[Path]:
            day = "2026-08-01" if job_id == "job-1" else "2026-08-04"
            out = Path(output_dir) / f"{day}.dbn.zst"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake")
            return [out]

        client.batch.download.side_effect = _download

        # Two contiguous ranges: [08-01, 08-02) and [08-04, 08-05) — a gap at 08-02/08-03.
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-02")
        _seed_day(root, "2026-08-03")

        with _mock_databento(client) as fake_db:
            fake_db.DBNStore.from_file.side_effect = [
                _fake_tick_store("2026-08-01"),
                _fake_tick_store("2026-08-01"),
                _fake_tick_store("2026-08-04"),
                _fake_tick_store("2026-08-04"),
            ]
            written = upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-05",
                output_dir=tmp_path,
                instrument_type="futures",
            )  # download_method defaults to "batch"

        assert client.batch.submit_job.call_count == 2
        assert len(written) == 2
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == {
            D(2026, 8, 1),
            D(2026, 8, 2),
            D(2026, 8, 3),
            D(2026, 8, 4),
        }

    def test_uncovered_day_in_range_gets_empty_marker(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-1"}
        client.batch.list_jobs.return_value = [{"id": "job-1", "state": "done"}]

        def _download(_job_id: str, output_dir: Path) -> list[Path]:
            # Only 08-01 has data; 08-02 is a non-trading day with no returned file.
            out = Path(output_dir) / "2026-08-01.dbn.zst"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake")
            return [out]

        client.batch.download.side_effect = _download

        with _mock_databento(client) as fake_db:
            fake_db.DBNStore.from_file.side_effect = [
                _fake_tick_store("2026-08-01"),
                _fake_tick_store("2026-08-01"),
            ]
            written = upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-03",
                output_dir=tmp_path,
                instrument_type="futures",
            )

        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        assert len(written) == 1
        assert _day_empty_marker(root, D(2026, 8, 2)).exists()
        assert covered_days("ES.c.0", "trades", tmp_path, instrument_type="futures") == {
            D(2026, 8, 1),
            D(2026, 8, 2),
        }

    def test_multi_day_file_raises_and_is_skipped_not_installed(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-1"}
        client.batch.list_jobs.return_value = [{"id": "job-1", "state": "done"}]

        def _download(_job_id: str, output_dir: Path) -> list[Path]:
            out = Path(output_dir) / "spans-two-days.dbn.zst"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake")
            return [out]

        client.batch.download.side_effect = _download

        bad_store = MagicMock()
        bad_store.start = pd.Timestamp("2026-08-01T00:00:00", tz="UTC")
        bad_store.end = pd.Timestamp("2026-08-03T00:00:00", tz="UTC")  # spans 2 days

        with _mock_databento(client) as fake_db:
            fake_db.DBNStore.from_file.return_value = bad_store
            written = upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-03",
                output_dir=tmp_path,
                instrument_type="futures",
            )

        # The bad range's exception is caught and logged, not raised — nothing installed.
        assert written == []
        assert missing_days(
            "ES.c.0", "trades", "2026-08-01", "2026-08-03", tmp_path, instrument_type="futures"
        ) == [D(2026, 8, 1), D(2026, 8, 2)]

    def test_rerun_is_a_no_op(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-1"}
        client.batch.list_jobs.return_value = [{"id": "job-1", "state": "done"}]

        def _download(_job_id: str, output_dir: Path) -> list[Path]:
            out = Path(output_dir) / "2026-08-01.dbn.zst"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake")
            return [out]

        client.batch.download.side_effect = _download

        with _mock_databento(client) as fake_db:
            fake_db.DBNStore.from_file.side_effect = [
                _fake_tick_store("2026-08-01"),
                _fake_tick_store("2026-08-01"),
            ]
            upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-02",
                output_dir=tmp_path,
                instrument_type="futures",
            )
            written_again = upsert_ticks(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="trades",
                start="2026-08-01",
                end="2026-08-02",
                output_dir=tmp_path,
                instrument_type="futures",
            )

        assert client.batch.submit_job.call_count == 1  # not 2
        assert written_again == []


# ===========================================================================
# load_ticks
# ===========================================================================


class TestLoadTicks:
    def test_missing_store_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ingest-config"):
            load_ticks("ES.c.0", "trades", output_dir=tmp_path, instrument_type="futures")

    def test_empty_markers_only_raises(self, tmp_path: Path) -> None:
        _seed_empty_day(_tick_root("ES.c.0", "trades", tmp_path, "futures"), "2026-08-01")
        with pytest.raises(FileNotFoundError):
            load_ticks("ES.c.0", "trades", output_dir=tmp_path, instrument_type="futures")

    def test_reads_all_partitions(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-01", rows=2)
        _seed_day(root, "2026-08-02", rows=3)

        frame = load_ticks(
            "ES.c.0", "trades", output_dir=tmp_path, instrument_type="futures"
        ).collect()
        assert isinstance(frame, pl.DataFrame)
        assert frame.height == 5
        assert "date" in frame.columns

    def test_date_filters(self, tmp_path: Path) -> None:
        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-01", rows=2)
        _seed_day(root, "2026-08-02", rows=3)
        _seed_day(root, "2026-08-03", rows=4)

        frame = load_ticks(
            "ES.c.0",
            "trades",
            start="2026-08-02",
            end="2026-08-03",
            output_dir=tmp_path,
            instrument_type="futures",
        ).collect()
        assert isinstance(frame, pl.DataFrame)
        assert frame.height == 3


# ===========================================================================
# CLI
# ===========================================================================


class TestCli:
    def test_ingest_forwards_stype_in_to_the_download(self, tmp_path: Path) -> None:
        """Regression: --stype-in previously affected only the cost estimate."""
        from snippy_scales.cli.data import app

        with (
            patch("snippy_scales.data.ingest.estimate_cost", return_value=1.0),
            patch("snippy_scales.data.ingest.upsert_symbol") as upsert,
        ):
            upsert.return_value = tmp_path / "out.parquet"
            result = CliRunner().invoke(
                app,
                [
                    "ingest",
                    "GLBX.MDP3",
                    "-s",
                    "ES.c.0",
                    "--start",
                    "2026-08-01",
                    "--end",
                    "2026-08-02",
                    "--stype-in",
                    "continuous",
                    "--instrument-class",
                    "futures",
                    "--yes",
                ],
            )

        assert result.exit_code == 0, result.output
        assert upsert.call_args.kwargs["stype_in"] == "continuous"

    def test_ingest_defaults_download_method_to_batch(self, tmp_path: Path) -> None:
        from snippy_scales.cli.data import app

        with (
            patch("snippy_scales.data.ingest.estimate_cost", return_value=1.0),
            patch("snippy_scales.data.ingest.upsert_symbol") as upsert,
        ):
            upsert.return_value = tmp_path / "out.parquet"
            result = CliRunner().invoke(
                app,
                [
                    "ingest",
                    "GLBX.MDP3",
                    "-s",
                    "ES.c.0",
                    "--start",
                    "2026-08-01",
                    "--end",
                    "2026-08-02",
                    "--instrument-class",
                    "futures",
                    "--yes",
                ],
            )

        assert result.exit_code == 0, result.output
        assert upsert.call_args.kwargs["download_method"] == "batch"

    def test_ingest_forwards_streaming_download_method(self, tmp_path: Path) -> None:
        from snippy_scales.cli.data import app

        with (
            patch("snippy_scales.data.ingest.estimate_cost", return_value=1.0),
            patch("snippy_scales.data.ingest.upsert_symbol") as upsert,
        ):
            upsert.return_value = tmp_path / "out.parquet"
            result = CliRunner().invoke(
                app,
                [
                    "ingest",
                    "GLBX.MDP3",
                    "-s",
                    "ES.c.0",
                    "--start",
                    "2026-08-01",
                    "--end",
                    "2026-08-02",
                    "--instrument-class",
                    "futures",
                    "--download-method",
                    "streaming",
                    "--yes",
                ],
            )

        assert result.exit_code == 0, result.output
        assert upsert.call_args.kwargs["download_method"] == "streaming"

    def test_ingest_routes_trades_schema_to_the_tick_path(self, tmp_path: Path) -> None:
        """--schema trades should call tick functions, not the bar upsert."""
        from snippy_scales.cli.data import app

        with (
            patch("snippy_scales.data.ingest.estimate_cost") as bar_estimate,
            patch("snippy_scales.data.tick.estimate_tick_cost") as tick_estimate,
            patch("snippy_scales.data.tick.upsert_ticks") as upsert,
        ):
            tick_estimate.return_value = TickCostEstimate(
                symbol="ES.c.0",
                schema="trades",
                cost_usd=1.0,
                billable_bytes=0,
                missing_days=[dt.date(2026, 8, 1)],
                cached_days=0,
            )
            result = CliRunner().invoke(
                app,
                [
                    "ingest",
                    "GLBX.MDP3",
                    "-s",
                    "ES.c.0",
                    "--schema",
                    "trades",
                    "--start",
                    "2026-08-01",
                    "--end",
                    "2026-08-02",
                    "--output-dir",
                    str(tmp_path),
                    "--instrument-class",
                    "futures",
                    "--yes",
                ],
            )

        assert result.exit_code == 0, result.output
        bar_estimate.assert_not_called()
        upsert.assert_called_once()

    def test_ingest_skips_tick_download_when_cached(self, tmp_path: Path) -> None:
        from snippy_scales.cli.data import app

        _seed_day(_tick_root("ES.c.0", "trades", tmp_path, "futures"), "2026-08-01")

        with patch("snippy_scales.data.tick.upsert_ticks") as upsert:
            result = CliRunner().invoke(
                app,
                [
                    "ingest",
                    "GLBX.MDP3",
                    "-s",
                    "ES.c.0",
                    "--schema",
                    "trades",
                    "--start",
                    "2026-08-01",
                    "--end",
                    "2026-08-02",
                    "--output-dir",
                    str(tmp_path),
                    "--instrument-class",
                    "futures",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "already stored" in result.output
        upsert.assert_not_called()

    def test_ingest_bar_forwards_parent_stype_in(self, tmp_path: Path) -> None:
        """A symbol like 'ES.FUT' with --stype-in parent goes through the normal bar path."""
        from snippy_scales.cli.data import app

        with (
            patch("snippy_scales.data.ingest.estimate_cost", return_value=1.0),
            patch("snippy_scales.data.ingest.upsert_symbol") as upsert,
        ):
            upsert.return_value = tmp_path / "out.parquet"
            result = CliRunner().invoke(
                app,
                [
                    "ingest",
                    "GLBX.MDP3",
                    "-s",
                    "ES.FUT",
                    "--schema",
                    "1d",
                    "--start",
                    "2026-08-01",
                    "--end",
                    "2026-08-02",
                    "--stype-in",
                    "parent",
                    "--instrument-class",
                    "futures",
                    "--yes",
                ],
            )

        assert result.exit_code == 0, result.output
        assert upsert.call_args.kwargs["stype_in"] == "parent"

    def test_coverage_reports_gaps(self, tmp_path: Path) -> None:
        from snippy_scales.cli.data import app

        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-01")
        _seed_day(root, "2026-08-04")

        result = CliRunner().invoke(
            app,
            ["coverage", "-s", "ES.c.0", "--schema", "trades", "--output-dir", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "Gaps:" in result.output
        assert "2026-08-02" in result.output

    def test_coverage_reports_no_gaps(self, tmp_path: Path) -> None:
        from snippy_scales.cli.data import app

        root = _tick_root("ES.c.0", "trades", tmp_path, "futures")
        _seed_day(root, "2026-08-01")
        _seed_day(root, "2026-08-02")

        result = CliRunner().invoke(
            app,
            ["coverage", "-s", "ES.c.0", "--schema", "trades", "--output-dir", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "No gaps." in result.output
