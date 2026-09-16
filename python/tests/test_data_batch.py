"""Unit tests for snippy_scales.data.batch.run_batch_job — the shared Databento
batch-job submit/poll/download primitive.

All tests are fully offline: the ``databento`` client is a MagicMock, and
``sleep``/``clock`` are injected fakes so no test ever really waits.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from snippy_scales.data.batch import run_batch_job


class _FakeClock:
    """A monotonic clock that advances only when told to."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _base_kwargs(client: MagicMock, tmp_path: Path, **overrides: object) -> dict:
    kwargs = {
        "client": client,
        "dataset": "GLBX.MDP3",
        "symbols": ["ES.c.0"],
        "schema": "ohlcv-1d",
        "start": "2024-01-01",
        "end": "2024-01-05",
        "stype_in": "raw_symbol",
        "split_duration": "none",
        "output_dir": tmp_path,
        "sleep": lambda _seconds: None,
        "clock": _FakeClock(),
    }
    kwargs.update(overrides)
    return kwargs


def _matching_job(job_id: str, *, state: str, symbols: object = "ES.c.0") -> dict:
    """A list_jobs() entry that matches _base_kwargs()'s default request.

    Field shapes here mirror a real ``list_jobs()`` response (verified
    against the live API): ``start``/``end`` come back as tz-aware
    nanosecond-precision UTC strings, not the bare dates we submitted, and
    ``split_duration="none"`` is echoed back as ``None``, not the string
    ``"none"``. A fixture using the same literal strings we submit would not
    catch either mismatch.
    """
    return {
        "id": job_id,
        "state": state,
        "dataset": "GLBX.MDP3",
        "symbols": symbols,
        "schema": "ohlcv-1d",
        "start": "2024-01-01T00:00:00.000000000Z",
        "end": "2024-01-05T00:00:00.000000000Z",
        "stype_in": "raw_symbol",
        "split_duration": None,
    }


class TestJobReuse:
    def test_reuses_matching_done_job_without_submitting(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.list_jobs.return_value = [_matching_job("old-job", state="done")]
        client.batch.download.return_value = [tmp_path / "old-job" / "file.dbn.zst"]

        files = run_batch_job(**_base_kwargs(client, tmp_path))

        assert files == [tmp_path / "old-job" / "file.dbn.zst"]
        client.batch.submit_job.assert_not_called()
        client.batch.download.assert_called_once_with("old-job", output_dir=tmp_path)

    def test_reuses_matching_job_still_processing(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.list_jobs.side_effect = [
            [_matching_job("old-job", state="processing")],  # reuse-check
            [{"id": "old-job", "state": "processing"}],  # poll #1
            [{"id": "old-job", "state": "done"}],  # poll #2
        ]
        client.batch.download.return_value = []

        run_batch_job(**_base_kwargs(client, tmp_path, sleep=lambda _s: None))

        client.batch.submit_job.assert_not_called()
        client.batch.download.assert_called_once_with("old-job", output_dir=tmp_path)

    def test_symbols_as_comma_joined_string_still_matches(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.list_jobs.return_value = [
            _matching_job("old-job", state="done", symbols="ES.c.0,NQ.c.0")
        ]
        client.batch.download.return_value = []

        run_batch_job(**_base_kwargs(client, tmp_path, symbols=["NQ.c.0", "ES.c.0"]))

        client.batch.submit_job.assert_not_called()

    def test_does_not_reuse_job_with_different_symbols(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.list_jobs.side_effect = [
            [_matching_job("old-job", state="done", symbols="NQ.c.0")],  # reuse-check
            [{"id": "new-job", "state": "done"}],  # poll
        ]
        client.batch.submit_job.return_value = {"id": "new-job"}
        client.batch.download.return_value = []

        run_batch_job(**_base_kwargs(client, tmp_path))

        client.batch.submit_job.assert_called_once()

    def test_does_not_reuse_job_with_different_date_range(self, tmp_path: Path) -> None:
        client = MagicMock()
        stale = _matching_job("old-job", state="done")
        stale["end"] = "2023-01-01"
        client.batch.list_jobs.side_effect = [
            [stale],  # reuse-check
            [{"id": "new-job", "state": "done"}],  # poll
        ]
        client.batch.submit_job.return_value = {"id": "new-job"}
        client.batch.download.return_value = []

        run_batch_job(**_base_kwargs(client, tmp_path))

        client.batch.submit_job.assert_called_once()

    def test_reuse_check_queries_non_expired_states_only(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.list_jobs.side_effect = [
            [],  # reuse-check
            [{"id": "new-job", "state": "done"}],  # poll
        ]
        client.batch.submit_job.return_value = {"id": "new-job"}
        client.batch.download.return_value = []

        run_batch_job(**_base_kwargs(client, tmp_path))

        first_call_kwargs = client.batch.list_jobs.call_args_list[0].kwargs
        assert "expired" not in first_call_kwargs["states"]

    def test_list_jobs_failure_falls_back_to_submitting(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.list_jobs.side_effect = [
            RuntimeError("network error"),
            [{"id": "new-job", "state": "done"}],
        ]
        client.batch.submit_job.return_value = {"id": "new-job"}
        client.batch.download.return_value = []

        run_batch_job(**_base_kwargs(client, tmp_path))

        client.batch.submit_job.assert_called_once()


class TestRunBatchJob:
    def test_submits_and_downloads_once_job_is_done(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-1"}
        client.batch.list_jobs.return_value = [{"id": "job-1", "state": "done"}]
        client.batch.download.return_value = [tmp_path / "job-1" / "file.dbn.zst"]

        files = run_batch_job(**_base_kwargs(client, tmp_path))

        assert files == [tmp_path / "job-1" / "file.dbn.zst"]
        client.batch.submit_job.assert_called_once()
        submit_kwargs = client.batch.submit_job.call_args.kwargs
        assert submit_kwargs["dataset"] == "GLBX.MDP3"
        assert submit_kwargs["symbols"] == ["ES.c.0"]
        assert submit_kwargs["split_duration"] == "none"
        assert submit_kwargs["encoding"] == "dbn"
        client.batch.download.assert_called_once_with("job-1", output_dir=tmp_path)

    def test_filters_out_non_data_files(self, tmp_path: Path) -> None:
        """Regression: every batch job also includes condition.json/manifest.json/etc.

        alongside the real .dbn(.zst) files — client.batch.download() returns
        all of them, and handing a .json file to DBNStore.from_file() blows up
        with 'Could not determine compression format'.
        """
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-7"}
        client.batch.list_jobs.return_value = [{"id": "job-7", "state": "done"}]
        job_dir = tmp_path / "job-7"
        client.batch.download.return_value = [
            job_dir / "condition.json",
            job_dir / "manifest.json",
            job_dir / "metadata.json",
            job_dir / "symbology.json",
            job_dir / "glbx-mdp3-20240101.ohlcv-1d.dbn.zst",
        ]

        files = run_batch_job(**_base_kwargs(client, tmp_path))

        assert files == [job_dir / "glbx-mdp3-20240101.ohlcv-1d.dbn.zst"]

    def test_polls_through_queued_and_processing_before_done(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-2"}
        client.batch.list_jobs.side_effect = [
            [],  # reuse-check: no existing job matches
            [{"id": "job-2", "state": "queued"}],
            [{"id": "job-2", "state": "processing"}],
            [{"id": "job-2", "state": "done"}],
        ]
        client.batch.download.return_value = []
        sleeps: list[float] = []

        run_batch_job(**_base_kwargs(client, tmp_path, sleep=sleeps.append))

        assert client.batch.list_jobs.call_count == 4
        assert len(sleeps) == 2  # slept after "queued" and after "processing"

    def test_backoff_caps_at_max_poll_interval(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-3"}
        client.batch.list_jobs.side_effect = [
            [],  # reuse-check: no existing job matches
            *([{"id": "job-3", "state": "processing"}] for _ in range(4)),
            [{"id": "job-3", "state": "done"}],
        ]
        client.batch.download.return_value = []
        sleeps: list[float] = []

        run_batch_job(
            **_base_kwargs(
                client,
                tmp_path,
                sleep=sleeps.append,
                poll_interval=1.0,
                poll_backoff=3.0,
                max_poll_interval=5.0,
            )
        )

        assert sleeps == [1.0, 3.0, 5.0, 5.0]

    def test_expired_job_raises(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-4"}
        client.batch.list_jobs.return_value = [{"id": "job-4", "state": "expired"}]

        with pytest.raises(RuntimeError, match="expired"):
            run_batch_job(**_base_kwargs(client, tmp_path))

    def test_missing_job_raises(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-5"}
        client.batch.list_jobs.return_value = [{"id": "some-other-job", "state": "done"}]

        with pytest.raises(RuntimeError, match="disappeared"):
            run_batch_job(**_base_kwargs(client, tmp_path))

    def test_timeout_raises(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.batch.submit_job.return_value = {"id": "job-6"}
        client.batch.list_jobs.return_value = [{"id": "job-6", "state": "processing"}]

        clock = _FakeClock()

        def _advancing_sleep(seconds: float) -> None:
            clock.now += seconds

        with pytest.raises(TimeoutError, match="job-6"):
            run_batch_job(
                **_base_kwargs(
                    client,
                    tmp_path,
                    clock=clock,
                    sleep=_advancing_sleep,
                    timeout=10.0,
                    poll_interval=4.0,
                    poll_backoff=1.0,
                )
            )
