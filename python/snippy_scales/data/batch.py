"""Shared Databento Batch API primitive.

Both the bar path (:class:`~snippy_scales.data.providers.databento.DatabentoProvider`)
and the tick path (:mod:`snippy_scales.data.tick`) can fetch data via the
Historical Streaming API (``client.timeseries.get_range``, billed per call
with no server-side retention) or the Batch API (``client.batch.submit_job``
→ poll → ``client.batch.download``, billed identically per byte but with
completed job output kept downloadable free of charge for a retention
window). :func:`run_batch_job` is the one place that submits a job, polls it
to completion, and downloads its files — both callers just hand it request
parameters and get back a list of downloaded file paths.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    import databento as db

logger = logging.getLogger(__name__)

#: States a batch job can report from ``client.batch.list_jobs()``.
_POLL_STATES = "queued,processing,done,expired"
#: States worth reusing — "expired" jobs are no longer downloadable for free.
_REUSABLE_STATES = "queued,processing,done"

#: Suffixes of the actual data files in a batch job's output. Every job also
#: includes non-data files (``condition.json``, ``manifest.json``, ``metadata.json``,
#: ``symbology.json``, ...) that ``client.batch.download()`` returns alongside
#: them — those must never be handed to ``DBNStore.from_file``.
_DATA_FILE_SUFFIXES = (".dbn", ".dbn.zst")


def _to_utc_timestamp(value: str) -> pd.Timestamp:
    """Parse *value* and normalize it to a tz-aware UTC timestamp.

    ``list_jobs()`` echoes back tz-aware UTC timestamps (e.g.
    ``"2010-06-06T00:00:00.000000000Z"``); the ``start``/``end`` strings we
    submit are typically bare dates (e.g. ``"2010-06-06"``), which
    ``pd.Timestamp`` parses as tz-*naive*. Comparing a naive and an aware
    timestamp with ``==``/``!=`` silently evaluates as "not equal" rather
    than raising, so without this normalization every comparison would
    incorrectly report a mismatch.

    Raises:
        ValueError: If *value* doesn't parse to a real timestamp (including NaT).
    """
    ts = pd.Timestamp(value)
    if not isinstance(ts, pd.Timestamp):
        raise ValueError(f"{value!r} did not parse to a valid timestamp")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _find_reusable_job(
    client: db.Historical,
    *,
    dataset: str,
    symbols: list[str],
    schema: str,
    start: str,
    end: str,
    stype_in: str,
    split_duration: str,
) -> str | None:
    """Look for an already-submitted, non-expired job matching this exact request.

    Best-effort: matching is done against whatever fields
    ``client.batch.list_jobs()`` happens to echo back for each job. If a
    field this function expects is missing or doesn't parse, that job is
    treated as a non-match rather than risking a false positive — the worst
    case is falling back to submitting a (billed) duplicate job, never
    silently reusing the wrong one.

    Returns:
        The matching job's id, or ``None`` if no reusable job was found.
    """
    try:
        jobs = client.batch.list_jobs(states=_REUSABLE_STATES)
    except Exception:
        logger.warning("Could not list existing Databento batch jobs — submitting a new one.")
        return None

    requested_symbols = set(symbols)
    try:
        requested_start = _to_utc_timestamp(start)
        requested_end = _to_utc_timestamp(end)
    except (ValueError, TypeError):
        return None

    for job in jobs:
        if _job_matches_request(
            job,
            dataset=dataset,
            requested_symbols=requested_symbols,
            schema=schema,
            requested_start=requested_start,
            requested_end=requested_end,
            stype_in=stype_in,
            split_duration=split_duration,
        ):
            return job.get("id")

    return None


def _job_matches_request(
    job: dict[str, Any],
    *,
    dataset: str,
    requested_symbols: set[str],
    schema: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    stype_in: str,
    split_duration: str,
) -> bool:
    if job.get("dataset") != dataset:
        return False
    if job.get("schema") != schema:
        return False
    if job.get("stype_in") != stype_in:
        return False
    # Databento echoes split_duration="none" back as null/None, not "none".
    if (job.get("split_duration") or "none") != split_duration:
        return False

    job_symbols = job.get("symbols")
    if isinstance(job_symbols, str):
        job_symbols = job_symbols.split(",")
    if job_symbols is None or set(job_symbols) != requested_symbols:
        return False

    try:
        job_start = job.get("start")
        if job_start is None or _to_utc_timestamp(job_start) != requested_start:
            return False

        job_end = job.get("end")
        if job_end is None or _to_utc_timestamp(job_end) != requested_end:
            return False
    except (TypeError, ValueError):
        return False

    return True


def run_batch_job(
    client: db.Historical,
    *,
    dataset: str,
    symbols: Iterable[str],
    schema: str,
    start: str,
    end: str,
    stype_in: str,
    split_duration: Literal["day", "none"],
    output_dir: Path,
    poll_interval: float = 2.0,
    poll_backoff: float = 1.5,
    max_poll_interval: float = 20.0,
    timeout: float = 1800.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[Path]:
    """Submit a Databento batch job (or reuse a matching existing one), and download it.

    Before submitting, checks ``client.batch.list_jobs()`` for a
    non-expired job that already covers this exact
    ``(dataset, symbols, schema, start, end, stype_in, split_duration)`` —
    e.g. one submitted by an earlier run that failed after the job
    completed. If found, that job is reused (no new billing) instead of
    submitting a duplicate.

    Args:
        client: A ``databento.Historical`` client.
        dataset: Databento dataset code (e.g. ``"GLBX.MDP3"``).
        symbols: Instrument symbols to include in the job.
        schema: Databento schema name (bar or event-level).
        start: Inclusive start date/timestamp.
        end: Exclusive end date/timestamp.
        stype_in: Databento symbology type for the request.
        split_duration: ``"day"`` to have Databento split output into one
            file per UTC day (used by the tick path, which is stored
            day-partitioned), or ``"none"`` for a single file (used by the
            bar path, where a whole symbol/schema/range is one small file).
        output_dir: Directory the job's files are downloaded into (Databento
            nests them under ``output_dir/<job_id>/``).
        poll_interval: Initial delay between polls, in seconds.
        poll_backoff: Multiplier applied to the poll interval after each
            poll that isn't yet done (capped at *max_poll_interval*).
        max_poll_interval: Upper bound on the poll interval.
        timeout: Maximum total time to wait for the job to complete, in
            seconds.
        sleep: Sleep function, injectable for deterministic tests.
        clock: Monotonic time source, injectable for deterministic tests.

    Returns:
        Paths to the downloaded (and, if applicable, zip-extracted) files.

    Raises:
        RuntimeError: If the job disappears from ``list_jobs`` or reaches
            the ``"expired"`` state before completing.
        TimeoutError: If the job hasn't reached ``"done"`` within *timeout*.
    """
    symbols = list(symbols)

    job_id = _find_reusable_job(
        client,
        dataset=dataset,
        symbols=symbols,
        schema=schema,
        start=start,
        end=end,
        stype_in=stype_in,
        split_duration=split_duration,
    )
    if job_id is not None:
        logger.info(
            "Found an existing Databento batch job %s covering this exact request — "
            "reusing it instead of submitting a new one.",
            job_id,
        )
    else:
        job = client.batch.submit_job(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
            stype_in=stype_in,
            encoding="dbn",
            compression="zstd",
            split_duration=split_duration,
        )
        job_id = job["id"]
        logger.info("Submitted Databento batch job %s (%s, %s → %s).", job_id, schema, start, end)

    deadline = clock() + timeout
    interval = poll_interval
    while True:
        jobs = client.batch.list_jobs(states=_POLL_STATES)
        matches = [j for j in jobs if j.get("id") == job_id]
        if not matches:
            raise RuntimeError(f"Databento batch job {job_id} disappeared while polling.")
        state = matches[0].get("state")

        if state == "done":
            logger.info("Databento batch job %s is done — downloading.", job_id)
            break
        if state == "expired":
            raise RuntimeError(f"Databento batch job {job_id} expired before completing.")

        if clock() >= deadline:
            raise TimeoutError(
                f"Databento batch job {job_id} did not complete within {timeout:.0f}s "
                f"(last state: {state!r})."
            )

        logger.info(
            "Databento batch job %s still %s — polling again in %.1fs.", job_id, state, interval
        )
        sleep(interval)
        interval = min(interval * poll_backoff, max_poll_interval)

    files: list[Path] = client.batch.download(job_id, output_dir=output_dir)
    data_files = [f for f in files if f.name.endswith(_DATA_FILE_SUFFIXES)]
    skipped = len(files) - len(data_files)
    if skipped:
        logger.info(
            "Databento batch job %s: ignoring %d non-data file(s) (condition.json, "
            "manifest.json, ...).",
            job_id,
            skipped,
        )
    return sorted(data_files)
