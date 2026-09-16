"""Reusable helpers for large, resumable multi-symbol backfills.

Factored out of ``scripts/pull_sp500_alpaca_1m.py`` so the estimation,
symbols-cache, and manifest logic is unit-testable without importing a
standalone script (which isn't part of the installed package). The script
itself stays a thin Typer wrapper around these functions plus
:func:`~snippy_scales.data.ingest.ingest_from_config`.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snippy_scales.data.ingest import IngestRow

#: Rough bars/trading-day used only for the printed time/volume estimate —
#: actual counts vary (half days, extended-hours coverage per ticker, etc.).
ESTIMATED_BARS_PER_TRADING_DAY = 410  # ~390 regular-session minutes + a margin
TRADING_DAYS_PER_CALENDAR_YEAR = 252
MAX_BARS_PER_REQUEST = 10_000

#: Shared default paths for `algo data backfill-sp500` / `algo data update-universe`
#: and scripts/pull_sp500_alpaca_1m.py, so both refer to the same locations.
DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_MANIFEST = Path("data/raw/_manifests/sp500_1m.csv")
DEFAULT_SYMBOLS_CACHE = Path("data/universe/sp500_ever_members.txt")

_MANIFEST_FIELDNAMES = ["symbol", "asset_class", "schema", "succeeded", "detail", "updated_at"]


def resolve_backfill_window(years: int, end: str | dt.date | None) -> tuple[dt.date, dt.date]:
    """Compute the ``[start, end]`` calendar-date window for a *years*-long backfill.

    Args:
        years: How many years back from *end* the window should span.
        end: End date (``YYYY-MM-DD`` or a :class:`~datetime.date`). Defaults
            to today.

    Returns:
        ``(start_date, end_date)``.
    """
    if isinstance(end, dt.date):
        end_date = end
    else:
        end_date = dt.date.fromisoformat(end) if end else dt.date.today()
    try:
        start_date = end_date.replace(year=end_date.year - years)
    except ValueError:
        # end_date was Feb 29 and start year isn't a leap year.
        start_date = end_date.replace(year=end_date.year - years, day=28)
    return start_date, end_date


def estimate_backfill_plan(
    *, num_symbols: int, start: dt.date, end: dt.date, rate_limit_per_min: int
) -> dict[str, float]:
    """Rough upper-bound estimate of request volume and wall-clock time.

    Deliberately conservative (assumes every symbol needs the full range —
    upsert semantics mean a resumed run needs far less), so this is a safe
    upper bound, not a promise.

    Args:
        num_symbols: Symbol count in the planned run.
        start: Inclusive start date.
        end: Inclusive end date.
        rate_limit_per_min: Provider request budget per minute.

    Returns:
        A dict with ``trading_days``, ``bars_per_symbol``, ``total_requests``,
        and ``estimated_minutes``.
    """
    calendar_days = max((end - start).days, 0)
    trading_days = round(calendar_days * (TRADING_DAYS_PER_CALENDAR_YEAR / 365))
    bars_per_symbol = trading_days * ESTIMATED_BARS_PER_TRADING_DAY
    pages_per_symbol = max(math.ceil(bars_per_symbol / MAX_BARS_PER_REQUEST), 1)
    total_requests = num_symbols * pages_per_symbol
    return {
        "trading_days": trading_days,
        "bars_per_symbol": bars_per_symbol,
        "total_requests": total_requests,
        "estimated_minutes": total_requests / rate_limit_per_min,
    }


def load_symbols_cache(path: Path) -> list[str] | None:
    """Read a newline-delimited symbols cache file (``#`` comments, blank lines ignored).

    Args:
        path: Path to the cache file.

    Returns:
        The symbol list, or ``None`` if *path* doesn't exist.
    """
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def save_symbols_cache(path: Path, symbols: list[str]) -> None:
    """Write *symbols* to *path*, one per line, with a provenance comment header.

    Args:
        path: Destination file. Parent directories are created if missing.
        symbols: Ticker symbols to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    header = f"# S&P 500 ever-members, resolved {today} — {len(symbols)} tickers\n"
    path.write_text(header + "\n".join(symbols) + "\n")


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load an ingestion manifest CSV as ``{symbol: row_dict}``.

    Args:
        path: Manifest CSV path (see :func:`merge_manifest` for the schema).

    Returns:
        Empty dict if *path* doesn't exist yet.
    """
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["symbol"]: row for row in csv.DictReader(f)}


def merge_manifest(path: Path, new_rows: list[IngestRow]) -> dict[str, dict[str, str]]:
    """Merge freshly-ingested rows into any existing manifest and write it back.

    Existing rows for symbols not touched by this run (e.g. a
    ``--retry-failed`` run that only covers a subset) are preserved.

    Args:
        path: Manifest CSV path. Parent directories are created if missing.
        new_rows: Results from this run, one per ``(asset class, schema,
            symbol)`` triple, as returned by
            :func:`~snippy_scales.data.ingest.ingest_from_config`.

    Returns:
        The full merged manifest as ``{symbol: row_dict}``.
    """
    existing = read_manifest(path)
    now = dt.datetime.now(dt.UTC).isoformat()
    for row in new_rows:
        existing[row.symbol] = {
            "symbol": row.symbol,
            "asset_class": row.asset_class,
            "schema": row.schema,
            "succeeded": str(row.succeeded),
            "detail": row.detail,
            "updated_at": now,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDNAMES)
        writer.writeheader()
        for symbol in sorted(existing):
            writer.writerow(existing[symbol])

    return existing
