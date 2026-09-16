"""Unit tests for the Databento BarProvider (snippy_scales.data.providers.databento).

All tests are fully offline: ``databento`` is injected into ``sys.modules`` as
a MagicMock (the module uses ``import databento as db`` inside function
bodies), and ``run_batch_job``'s polling never really sleeps (mocked
``client.batch.list_jobs`` returns "done" immediately).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

import pandas as pd

from snippy_scales.data.providers.databento import DatabentoProvider


@contextmanager
def _mock_databento(client_mock: MagicMock) -> Generator[MagicMock, None, None]:
    """Inject a fake ``databento`` module into sys.modules (mirrors test_data_ingest.py)."""
    fake_db = MagicMock()
    fake_db.Historical.return_value = client_mock
    with patch.dict(sys.modules, {"databento": fake_db}):
        yield fake_db


def _mock_store(dates: list[str]) -> MagicMock:
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


class TestDownloadMethodDefault:
    def test_defaults_to_batch(self) -> None:
        provider = DatabentoProvider(dataset="GLBX.MDP3")
        assert provider.download_method == "batch"


class TestStreamingFetch:
    def test_calls_get_range_and_conforms(self) -> None:
        client_mock = MagicMock()
        client_mock.timeseries.get_range.return_value = _mock_store(["2024-01-01", "2024-01-02"])

        provider = DatabentoProvider(dataset="GLBX.MDP3", download_method="streaming")
        with _mock_databento(client_mock):
            df = provider.fetch_bars(
                symbol="ES.c.0", schema="ohlcv-1d", start="2024-01-01", end="2024-01-03"
            )

        assert len(df) == 2
        client_mock.timeseries.get_range.assert_called_once()
        client_mock.batch.submit_job.assert_not_called()


class TestBatchFetch:
    def test_submits_job_downloads_and_conforms(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.batch.submit_job.return_value = {"id": "job-bar-1"}
        client_mock.batch.list_jobs.return_value = [{"id": "job-bar-1", "state": "done"}]
        fake_file = tmp_path / "bar.dbn.zst"
        fake_file.write_bytes(b"fake")
        client_mock.batch.download.return_value = [fake_file]

        provider = DatabentoProvider(dataset="GLBX.MDP3")  # download_method defaults to "batch"
        with _mock_databento(client_mock) as fake_db:
            fake_db.DBNStore.from_file.return_value = _mock_store(["2024-01-01", "2024-01-02"])
            df = provider.fetch_bars(
                symbol="ES.c.0", schema="ohlcv-1d", start="2024-01-01", end="2024-01-03"
            )

        assert len(df) == 2
        client_mock.batch.submit_job.assert_called_once()
        submit_kwargs = client_mock.batch.submit_job.call_args.kwargs
        assert submit_kwargs["split_duration"] == "none"
        client_mock.timeseries.get_range.assert_not_called()

    def test_concatenates_multiple_returned_files(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.batch.submit_job.return_value = {"id": "job-bar-2"}
        client_mock.batch.list_jobs.return_value = [{"id": "job-bar-2", "state": "done"}]
        file_a = tmp_path / "a.dbn.zst"
        file_b = tmp_path / "b.dbn.zst"
        file_a.write_bytes(b"fake")
        file_b.write_bytes(b"fake")
        client_mock.batch.download.return_value = [file_a, file_b]

        provider = DatabentoProvider(dataset="GLBX.MDP3")
        with _mock_databento(client_mock) as fake_db:
            fake_db.DBNStore.from_file.side_effect = [
                _mock_store(["2024-01-01"]),
                _mock_store(["2024-01-02"]),
            ]
            df = provider.fetch_bars(
                symbol="ES.c.0", schema="ohlcv-1d", start="2024-01-01", end="2024-01-03"
            )

        assert len(df) == 2
