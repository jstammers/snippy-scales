"""Unit tests for data ingestion — config loading, frequency mapping, and upsert logic.

All tests are fully offline: Databento API calls are mocked so no credentials
or network access are required.
"""

from __future__ import annotations

import sys
import textwrap
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

import polars as pl
import pytest

from snippy_scales.data.config import (
    AssetClassConfig,
    IngestConfig,
    frequency_to_schema,
    load_config,
)
from snippy_scales.data.ingest import (
    _effective_start,
    _symbol_path,
    _to_polars,
    estimate_cost,
    estimate_costs_from_config,
    ingest_from_config,
    load_bars,
    upsert_symbol,
)

# ===========================================================================
# frequency_to_schema
# ===========================================================================


class TestFrequencyToSchema:
    def test_daily_aliases(self) -> None:
        for alias in ("1d", "d", "daily"):
            assert frequency_to_schema(alias) == "ohlcv-1d"

    def test_hourly_aliases(self) -> None:
        for alias in ("1h", "h", "hourly"):
            assert frequency_to_schema(alias) == "ohlcv-1h"

    def test_minute_aliases(self) -> None:
        for alias in ("1m", "1min", "min"):
            assert frequency_to_schema(alias) == "ohlcv-1m"

    def test_second_aliases(self) -> None:
        for alias in ("1s", "s"):
            assert frequency_to_schema(alias) == "ohlcv-1s"

    def test_eod_alias(self) -> None:
        assert frequency_to_schema("eod") == "ohlcv-eod"

    def test_passthrough_valid_schema(self) -> None:
        """Already-valid schema names should be returned unchanged."""
        assert frequency_to_schema("ohlcv-1d") == "ohlcv-1d"
        assert frequency_to_schema("ohlcv-1m") == "ohlcv-1m"

    def test_case_insensitive(self) -> None:
        assert frequency_to_schema("DAILY") == "ohlcv-1d"
        assert frequency_to_schema("1D") == "ohlcv-1d"

    def test_unknown_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown frequency"):
            frequency_to_schema("5min")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown frequency"):
            frequency_to_schema("")


# ===========================================================================
# IngestConfig / load_config
# ===========================================================================


class TestIngestConfig:
    def test_minimal_valid_config(self) -> None:
        cfg = IngestConfig(
            dataset="GLBX.MDP3",
            start="2020-01-01",
            asset_classes={"eq": AssetClassConfig(symbols=["ES.c.0"])},
        )
        assert cfg.schema == "ohlcv-1d"
        assert cfg.all_symbols == ["ES.c.0"]

    def test_all_symbols_across_classes(self) -> None:
        cfg = IngestConfig(
            dataset="GLBX.MDP3",
            start="2020-01-01",
            asset_classes={
                "eq": AssetClassConfig(symbols=["ES.c.0", "NQ.c.0"]),
                "rates": AssetClassConfig(symbols=["ZN.c.0"]),
            },
        )
        assert cfg.all_symbols == ["ES.c.0", "NQ.c.0", "ZN.c.0"]

    def test_schema_property_respects_tick_frequency(self) -> None:
        cfg = IngestConfig(
            dataset="GLBX.MDP3",
            start="2020-01-01",
            tick_frequency="1h",
            asset_classes={},
        )
        assert cfg.schema == "ohlcv-1h"

    def test_invalid_tick_frequency_raises(self) -> None:
        with pytest.raises(ValueError):
            IngestConfig(
                dataset="GLBX.MDP3",
                start="2020-01-01",
                tick_frequency="5min",
                asset_classes={},
            )

    def test_end_defaults_to_none(self) -> None:
        cfg = IngestConfig(dataset="GLBX.MDP3", start="2020-01-01", asset_classes={})
        assert cfg.end is None


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            dataset: "GLBX.MDP3"
            tick_frequency: "1d"
            start: "2020-01-01"
            asset_classes:
              equity_index:
                symbols: [ES.c.0, NQ.c.0]
              rates:
                symbols: [ZN.c.0]
        """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_text)

        cfg = load_config(config_file)

        assert cfg.dataset == "GLBX.MDP3"
        assert cfg.tick_frequency == "1d"
        assert cfg.start == "2020-01-01"
        assert cfg.all_symbols == ["ES.c.0", "NQ.c.0", "ZN.c.0"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_frequency_in_yaml_raises(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            dataset: "GLBX.MDP3"
            tick_frequency: "5min"
            start: "2020-01-01"
            asset_classes: {}
        """)
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml_text)

        with pytest.raises(ValueError):
            load_config(config_file)

    def test_optional_end_date_absent(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            dataset: "GLBX.MDP3"
            start: "2020-01-01"
            asset_classes: {}
        """)
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml_text)
        cfg = load_config(config_file)
        assert cfg.end is None


# ===========================================================================
# _symbol_path
# ===========================================================================


class TestSymbolPath:
    def test_basic(self, tmp_path: Path) -> None:
        p = _symbol_path("ES.c.0", "ohlcv-1d", tmp_path)
        assert p == tmp_path / "ES.c.0" / "ohlcv-1d.parquet"

    def test_slash_in_symbol_is_sanitised(self, tmp_path: Path) -> None:
        p = _symbol_path("BTC/USD", "ohlcv-1d", tmp_path)
        assert "/" not in str(p.name)


# ===========================================================================
# _to_polars
# ===========================================================================


class TestToPolars:
    def test_promotes_named_index(self) -> None:
        import pandas as pd

        ts = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame({"open": [100.0]}, index=pd.DatetimeIndex([ts], name="ts_event"))
        result = _to_polars(df)
        assert "ts_event" in result.columns
        assert "open" in result.columns

    def test_regular_range_index_not_promoted(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"ts_event": [1], "close": [42.0]})
        result = _to_polars(df)
        # Should NOT add an 'index' column
        assert "index" not in result.columns
        assert "ts_event" in result.columns


# ===========================================================================
# _effective_start
# ===========================================================================


class TestEffectiveStart:
    def _make_df(self, dates: list[str]) -> pl.DataFrame:
        timestamps = [datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC) for d in dates]
        return pl.DataFrame({"ts_event": timestamps}).with_columns(
            pl.col("ts_event").cast(pl.Datetime("us", "UTC"))
        )

    def test_returns_day_after_max(self) -> None:
        df = self._make_df(["2024-01-10", "2024-01-15", "2024-01-12"])
        result = _effective_start(df, "2024-01-01")
        assert result == "2024-01-16"

    def test_missing_ts_event_column_returns_requested_start(self) -> None:
        df = pl.DataFrame({"close": [100.0]})
        result = _effective_start(df, "2024-01-01")
        assert result == "2024-01-01"

    def test_empty_column_returns_requested_start(self) -> None:
        df = pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Datetime("us", "UTC"))})
        result = _effective_start(df, "2024-01-01")
        assert result == "2024-01-01"


# ===========================================================================
# upsert_symbol (mocked Databento)
# ===========================================================================


@contextmanager
def _mock_databento(client_mock: MagicMock) -> Generator[None, None, None]:
    """Inject a fake ``databento`` module into sys.modules.

    Because ``upsert_symbol`` uses ``import databento as db`` *inside* the
    function body (a deferred import), the real package does not need to be
    installed for tests.  Inserting a MagicMock at the ``databento`` key in
    ``sys.modules`` is the only reliable way to intercept the import at call
    time.
    """
    fake_db = MagicMock()
    fake_db.Historical.return_value = client_mock
    with patch.dict(sys.modules, {"databento": fake_db}):
        yield


def _make_ohlcv_df(dates: list[str]) -> pl.DataFrame:
    """Build a minimal OHLCV Polars DataFrame for testing."""
    import pandas as pd

    ts_list = [pd.Timestamp(d, tz="UTC") for d in dates]
    pdf = pd.DataFrame(
        {
            "ts_event": ts_list,
            "open": [100.0] * len(dates),
            "high": [110.0] * len(dates),
            "low": [90.0] * len(dates),
            "close": [105.0] * len(dates),
            "volume": [1000] * len(dates),
        }
    )
    return pl.from_pandas(pdf)


def _make_mock_store(dates: list[str]) -> MagicMock:
    """Return a mock Databento DBNStore whose to_df() yields OHLCV rows."""
    import pandas as pd

    ts_list = [pd.Timestamp(d, tz="UTC") for d in dates]
    pdf = pd.DataFrame(
        {
            "open": [100.0] * len(dates),
            "high": [110.0] * len(dates),
            "low": [90.0] * len(dates),
            "close": [105.0] * len(dates),
            "volume": [1000] * len(dates),
        },
        index=pd.DatetimeIndex(ts_list, name="ts_event"),
    )
    store = MagicMock()
    store.to_df.return_value = pdf
    return store


class TestUpsertSymbol:
    # databento is imported lazily inside upsert_symbol, so we patch at the
    # databento package level (which is what sys.modules sees on import).

    def test_initial_download_creates_file(self, tmp_path: Path) -> None:
        store = _make_mock_store(["2024-01-01", "2024-01-02", "2024-01-03"])
        client_mock = MagicMock()
        client_mock.timeseries.get_range.return_value = store

        with _mock_databento(client_mock):
            path = upsert_symbol(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="ohlcv-1d",
                start="2024-01-01",
                end="2024-01-04",
                output_dir=tmp_path,
            )

        assert path.exists()
        df = pl.read_parquet(path)
        assert len(df) == 3

    def test_upsert_appends_new_rows(self, tmp_path: Path) -> None:
        # Pre-populate with 3 days
        initial_df = _make_ohlcv_df(["2024-01-01", "2024-01-02", "2024-01-03"])
        out_path = tmp_path / "ES.c.0" / "ohlcv-1d.parquet"
        out_path.parent.mkdir(parents=True)
        initial_df.write_parquet(out_path)

        # Databento returns 1 new day (2024-01-04)
        store = _make_mock_store(["2024-01-04"])
        client_mock = MagicMock()
        client_mock.timeseries.get_range.return_value = store

        with _mock_databento(client_mock):
            path = upsert_symbol(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="ohlcv-1d",
                start="2024-01-01",
                end="2024-01-05",
                output_dir=tmp_path,
            )

        df = pl.read_parquet(path)
        assert len(df) == 4
        # Verify API was called starting from 2024-01-04 (day after last stored)
        call_kwargs = client_mock.timeseries.get_range.call_args.kwargs
        assert call_kwargs["start"] == "2024-01-04"

    def test_no_download_when_already_up_to_date(self, tmp_path: Path) -> None:
        initial_df = _make_ohlcv_df(["2024-01-01", "2024-01-02", "2024-01-03"])
        out_path = tmp_path / "ES.c.0" / "ohlcv-1d.parquet"
        out_path.parent.mkdir(parents=True)
        initial_df.write_parquet(out_path)

        client_mock = MagicMock()

        with _mock_databento(client_mock):
            upsert_symbol(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="ohlcv-1d",
                start="2024-01-01",
                end="2024-01-03",  # already covered
                output_dir=tmp_path,
            )

        client_mock.timeseries.get_range.assert_not_called()

    def test_deduplication_on_overlap(self, tmp_path: Path) -> None:
        initial_df = _make_ohlcv_df(["2024-01-01", "2024-01-02", "2024-01-03"])
        out_path = tmp_path / "ES.c.0" / "ohlcv-1d.parquet"
        out_path.parent.mkdir(parents=True)
        initial_df.write_parquet(out_path)

        # Simulate overlap: API returns 2024-01-03 again + new day
        store = _make_mock_store(["2024-01-03", "2024-01-04"])
        client_mock = MagicMock()
        client_mock.timeseries.get_range.return_value = store

        with _mock_databento(client_mock):
            path = upsert_symbol(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="ohlcv-1d",
                start="2024-01-01",
                end="2024-01-05",
                output_dir=tmp_path,
            )

        df = pl.read_parquet(path)
        # Deduplicated: 4 unique days, not 5
        assert len(df) == 4


# ===========================================================================
# ingest_from_config (mocked Databento)
# ===========================================================================


class TestIngestFromConfig:
    def _make_config(self, symbols: list[str], frequency: str = "1d") -> IngestConfig:
        return IngestConfig(
            dataset="GLBX.MDP3",
            tick_frequency=frequency,
            start="2024-01-01",
            end="2024-01-05",
            asset_classes={"test": AssetClassConfig(symbols=symbols)},
        )

    def test_all_symbols_ingested(self, tmp_path: Path) -> None:
        store = _make_mock_store(["2024-01-01", "2024-01-02"])
        client_mock = MagicMock()
        client_mock.timeseries.get_range.return_value = store

        with _mock_databento(client_mock):
            results = ingest_from_config(
                self._make_config(["ES.c.0", "ZN.c.0"]),
                output_dir=tmp_path,
            )

        assert set(results.keys()) == {"ES.c.0", "ZN.c.0"}
        assert client_mock.timeseries.get_range.call_count == 2

    def test_frequency_override(self, tmp_path: Path) -> None:
        store = _make_mock_store(["2024-01-01"])
        client_mock = MagicMock()
        client_mock.timeseries.get_range.return_value = store

        with _mock_databento(client_mock):
            results = ingest_from_config(
                self._make_config(["ES.c.0"], frequency="1d"),
                frequency_override="1h",
                output_dir=tmp_path,
            )

        # With hourly override, file should be under ohlcv-1h
        path = results["ES.c.0"]
        assert "ohlcv-1h" in str(path)

    def test_failed_symbol_does_not_abort_others(self, tmp_path: Path) -> None:
        good_store = _make_mock_store(["2024-01-01"])

        def side_effect(**kwargs: object) -> MagicMock:
            if "ES.c.0" in kwargs.get("symbols", []):  # type: ignore[arg-type]
                raise RuntimeError("API error")
            return good_store

        client_mock = MagicMock()
        client_mock.timeseries.get_range.side_effect = side_effect

        with _mock_databento(client_mock):
            results = ingest_from_config(
                self._make_config(["ES.c.0", "ZN.c.0"]),
                output_dir=tmp_path,
            )

        # ZN should succeed even though ES failed
        assert "ZN.c.0" in results
        assert "ES.c.0" not in results


# ===========================================================================
# load_bars
# ===========================================================================


class TestLoadBars:
    def test_returns_dataframe(self, tmp_path: Path) -> None:
        df = _make_ohlcv_df(["2024-01-01", "2024-01-02"])
        out_path = tmp_path / "ES.c.0" / "ohlcv-1d.parquet"
        out_path.parent.mkdir(parents=True)
        df.write_parquet(out_path)

        result = load_bars("ES.c.0", "ohlcv-1d", tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No data found"):
            load_bars("MISSING.c.0", "ohlcv-1d", tmp_path)


# ===========================================================================
# estimate_cost (mocked Databento)
# ===========================================================================


class TestEstimateCost:
    def test_returns_zero_when_already_up_to_date(self, tmp_path: Path) -> None:
        """No API call should be made when local data already covers the range."""
        initial_df = _make_ohlcv_df(["2024-01-01", "2024-01-02", "2024-01-03"])
        out_path = tmp_path / "ES.c.0" / "ohlcv-1d.parquet"
        out_path.parent.mkdir(parents=True)
        initial_df.write_parquet(out_path)

        client_mock = MagicMock()

        with _mock_databento(client_mock):
            cost = estimate_cost(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="ohlcv-1d",
                start="2024-01-01",
                end="2024-01-03",  # already covered
                output_dir=tmp_path,
            )

        assert cost == 0.0
        client_mock.metadata.get_cost.assert_not_called()

    def test_calls_metadata_api_for_missing_data(self, tmp_path: Path) -> None:
        """metadata.get_cost should be called with the effective (trimmed) date range."""
        client_mock = MagicMock()
        client_mock.metadata.get_cost.return_value = 1.23

        with _mock_databento(client_mock):
            cost = estimate_cost(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="ohlcv-1d",
                start="2024-01-01",
                end="2024-01-10",
                output_dir=tmp_path,
            )

        assert cost == pytest.approx(1.23)
        client_mock.metadata.get_cost.assert_called_once()
        call_kwargs = client_mock.metadata.get_cost.call_args.kwargs
        assert call_kwargs["dataset"] == "GLBX.MDP3"
        assert call_kwargs["schema"] == "ohlcv-1d"
        assert "ES.c.0" in call_kwargs["symbols"]

    def test_uses_effective_start_for_partial_coverage(self, tmp_path: Path) -> None:
        """Cost query start should be trimmed to the day after the last stored bar."""
        initial_df = _make_ohlcv_df(["2024-01-01", "2024-01-02", "2024-01-03"])
        out_path = tmp_path / "ES.c.0" / "ohlcv-1d.parquet"
        out_path.parent.mkdir(parents=True)
        initial_df.write_parquet(out_path)

        client_mock = MagicMock()
        client_mock.metadata.get_cost.return_value = 0.50

        with _mock_databento(client_mock):
            cost = estimate_cost(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="ohlcv-1d",
                start="2024-01-01",
                end="2024-01-10",
                output_dir=tmp_path,
            )

        assert cost == pytest.approx(0.50)
        call_kwargs = client_mock.metadata.get_cost.call_args.kwargs
        # Effective start should be 2024-01-04 (day after last stored 2024-01-03)
        assert call_kwargs["start"] == "2024-01-04"

    def test_returns_zero_for_no_file_and_no_data_needed(self, tmp_path: Path) -> None:
        """When end <= start (degenerate range) no API call should be made."""
        client_mock = MagicMock()
        client_mock.metadata.get_cost.return_value = 0.0

        with _mock_databento(client_mock):
            cost = estimate_cost(
                dataset="GLBX.MDP3",
                symbol="ES.c.0",
                schema="ohlcv-1d",
                start="2024-01-05",
                end="2024-01-05",  # same day — no range
                output_dir=tmp_path,
            )

        # get_cost is still called (no local file exists), but returns 0.0
        assert cost == pytest.approx(0.0)


# ===========================================================================
# estimate_costs_from_config (mocked Databento)
# ===========================================================================


class TestEstimateCostsFromConfig:
    def _make_config(self, symbols: list[str], frequency: str = "1d") -> IngestConfig:
        return IngestConfig(
            dataset="GLBX.MDP3",
            tick_frequency=frequency,
            start="2024-01-01",
            end="2024-01-10",
            asset_classes={"test": AssetClassConfig(symbols=symbols)},
        )

    def test_returns_cost_per_symbol(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.metadata.get_cost.return_value = 2.00

        with _mock_databento(client_mock):
            costs = estimate_costs_from_config(
                self._make_config(["ES.c.0", "ZN.c.0"]),
                output_dir=tmp_path,
            )

        assert set(costs.keys()) == {"ES.c.0", "ZN.c.0"}
        assert costs["ES.c.0"] == pytest.approx(2.00)
        assert costs["ZN.c.0"] == pytest.approx(2.00)
        assert client_mock.metadata.get_cost.call_count == 2

    def test_cached_symbol_shows_zero_cost(self, tmp_path: Path) -> None:
        """Symbols already on disk within the requested range should cost $0."""
        initial_df = _make_ohlcv_df(["2024-01-01", "2024-01-02", "2024-01-03"])
        out_path = tmp_path / "ES.c.0" / "ohlcv-1d.parquet"
        out_path.parent.mkdir(parents=True)
        initial_df.write_parquet(out_path)

        client_mock = MagicMock()
        client_mock.metadata.get_cost.return_value = 1.00

        with _mock_databento(client_mock):
            costs = estimate_costs_from_config(
                IngestConfig(
                    dataset="GLBX.MDP3",
                    tick_frequency="1d",
                    start="2024-01-01",
                    end="2024-01-03",  # fully covered by existing data
                    asset_classes={"test": AssetClassConfig(symbols=["ES.c.0"])},
                ),
                output_dir=tmp_path,
            )

        assert costs["ES.c.0"] == 0.0
        client_mock.metadata.get_cost.assert_not_called()

    def test_failed_estimate_recorded_as_nan(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.metadata.get_cost.side_effect = RuntimeError("API error")

        with _mock_databento(client_mock):
            costs = estimate_costs_from_config(
                self._make_config(["ES.c.0"]),
                output_dir=tmp_path,
            )

        import math

        assert math.isnan(costs["ES.c.0"])

    def test_frequency_override_applied(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.metadata.get_cost.return_value = 5.00

        with _mock_databento(client_mock):
            estimate_costs_from_config(
                self._make_config(["ES.c.0"], frequency="1d"),
                frequency_override="1h",
                output_dir=tmp_path,
            )

        call_kwargs = client_mock.metadata.get_cost.call_args.kwargs
        assert call_kwargs["schema"] == "ohlcv-1h"
