"""Daily bars of options on futures, stored outside the Nautilus catalog.

An option root has thousands of contracts (every strike of every expiry), far too
many for one catalog series each, and research on them wants the whole chain of a
day at once. They are therefore kept as one Parquet file per root,
``data/options/<ROOT>/ohlcv-1d.parquet``, written once by :func:`sync_option_bars`
(a paid Databento request for the parent symbol ``<ROOT>.OPT``) and read with
:func:`load_option_bars`.

Strike, right and underlying future are read from the raw symbol
(``ESH4 P4490``), so no instrument definitions are needed.
"""

from __future__ import annotations

import datetime as dt
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import polars as pl

from snippy_scales.data.futures import MONTH_CODES
from snippy_scales.data.paths import DATA_ROOT

if TYPE_CHECKING:
    from pathlib import Path

OPTIONS_DIR = DATA_ROOT / "options"

_SYMBOL = re.compile(rf"^([A-Z0-9]+[{MONTH_CODES}]\d{{1,2}}) ([CP])(\d+)$")
_UNDERLYING = re.compile(rf"^([A-Z0-9]+)([{MONTH_CODES}])(\d{{1,2}})$")
_FRIDAY = 4


def option_path(root: str, schema: str = "ohlcv-1d", options_dir: Path | None = None) -> Path:
    """Where the bars of *root*'s options are stored."""
    return (options_dir if options_dir is not None else OPTIONS_DIR) / root / f"{schema}.parquet"


def parse_option_symbol(raw: str) -> tuple[str, str, float] | None:
    """``ESH4 P4490`` -> ``("ESH4", "P", 4490.0)``: underlying future, right, strike.

    ``None`` for anything that is not an outright option (spreads and user-defined
    strategies such as ``UD:1V: SG 2719478``).
    """
    match = _SYMBOL.match(raw)
    if match is None:
        return None
    return match.group(1), match.group(2), float(match.group(3))


def third_friday(year: int, month: int) -> dt.date:
    """The third Friday of a month: when quarterly equity-index options expire."""
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(_FRIDAY - first.weekday()) % 7 + 14)


def expiry(underlying: str, on: dt.date) -> dt.date:
    """Expiry (third Friday of the contract month) of an option on *underlying*.

    A raw symbol carries one year digit (``ESH4``), so the year is the first one
    ending in that digit whose third Friday is not before *on*, the date the
    option was seen trading.

    Raises:
        ValueError: If *underlying* is not a futures symbol.
    """
    match = _UNDERLYING.match(underlying)
    if match is None:
        raise ValueError(f"{underlying!r} is not a futures contract symbol")
    month = MONTH_CODES.index(match.group(2)) + 1
    digits = match.group(3)
    modulus = 10 ** len(digits)
    year = on.year - on.year % modulus + int(digits)
    while third_friday(year, month) < on:
        year += modulus
    return third_friday(year, month)


def tidy(frame: pl.DataFrame) -> pl.DataFrame:
    """Raw bars (``ts_event``, ``symbol``, OHLCV) as one row per option and day.

    Keeps outright options with a positive close and adds ``date`` (the session's
    UTC date), ``underlying``, ``right``, ``strike``, ``expiry`` and
    ``days_to_expiry`` (calendar days).
    """
    parsed = [parse_option_symbol(s) for s in frame["symbol"].to_list()]
    keep = pl.Series([p is not None for p in parsed])
    out = frame.filter(keep).with_columns(
        pl.Series("underlying", [p[0] for p in parsed if p is not None]),
        pl.Series("right", [p[1] for p in parsed if p is not None]),
        pl.Series("strike", [p[2] for p in parsed if p is not None]),
        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")).dt.date().alias("date"),
    )
    out = out.filter(pl.col("close") > 0)
    # One expiry per (underlying, year the bar was seen): cheap to compute, exact.
    keys = out.select("underlying", pl.col("date").dt.year().alias("year")).unique()
    lookup = {
        (u, y): (expiry(u, dt.date(y, 1, 1)), expiry(u, dt.date(y, 12, 31)))
        for u, y in keys.iter_rows()
    }
    first = pl.Series(
        "__first",
        [lookup[(u, d.year)][0] for u, d in zip(out["underlying"], out["date"], strict=True)],
        dtype=pl.Date,
    )
    last = pl.Series(
        "__last",
        [lookup[(u, d.year)][1] for u, d in zip(out["underlying"], out["date"], strict=True)],
        dtype=pl.Date,
    )
    out = out.with_columns(first, last).with_columns(
        expiry=pl.when(pl.col("date") <= pl.col("__first"))
        .then(pl.col("__first"))
        .otherwise(pl.col("__last"))
    )
    return (
        out.with_columns(days_to_expiry=(pl.col("expiry") - pl.col("date")).dt.total_days())
        .select(
            "date",
            "symbol",
            "underlying",
            "right",
            "strike",
            "expiry",
            "days_to_expiry",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
        .sort("date", "expiry", "strike", "right")
    )


def _month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """``[start, end)`` split at calendar-month boundaries, as ISO date pairs."""
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    chunks = []
    while lo < hi:
        first_of_next = (lo.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        nxt = min(first_of_next, hi)
        chunks.append((lo.isoformat(), nxt.isoformat()))
        lo = nxt
    return chunks


_ATTEMPTS = 4


def sync_option_bars(  # noqa: PLR0913 — one argument per request field
    root: str,
    start: str,
    end: str,
    *,
    dataset: str = "GLBX.MDP3",
    api_key: str | None = None,
    options_dir: Path | None = None,
    workers: int = 4,
) -> Path:
    """Download daily bars of every option on *root* and store them (a paid request).

    Price the request first with Databento's ``metadata.get_cost``. The range is
    fetched one calendar month at a time (Databento's gateway times out on a whole
    chain over a year), *workers* months at once, and each month is kept in
    ``parts/``, so a failed run resumes without paying again for the months it has.
    The combined file is written once: an existing one is never overwritten.

    Raises:
        FileExistsError: If the bars are already stored.
    """
    import databento as db  # noqa: PLC0415 — only this function needs the client

    from snippy_scales.data.databento import _api_key  # noqa: PLC0415

    path = option_path(root, options_dir=options_dir)
    if path.exists():
        raise FileExistsError(f"{path} exists: option bars are written once, never replaced")
    parts_dir = path.parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    key = _api_key(api_key)

    def fetch(chunk: tuple[str, str]) -> Path:
        lo, hi = chunk
        part = parts_dir / f"{path.stem}_{lo}_{hi}.parquet"
        client = db.Historical(key)
        for attempt in range(_ATTEMPTS):
            if part.exists():
                break
            try:
                store = client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[f"{root}.OPT"],
                    stype_in="parent",
                    schema="ohlcv-1d",
                    start=lo,
                    end=hi,
                )
            except db.BentoServerError:  # gateway timeouts are transient
                if attempt == _ATTEMPTS - 1:
                    raise
                continue
            frame = pl.from_pandas(store.to_df().reset_index()).select(
                "ts_event", "symbol", "open", "high", "low", "close", "volume"
            )
            # Write under another name first, so a part is never seen half-written.
            scratch = part.with_suffix(".partial")
            frame.write_parquet(scratch)
            scratch.rename(part)
        return part

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(fetch, _month_chunks(start, end)))
    pl.concat([pl.read_parquet(p) for p in parts]).sort("ts_event", "symbol").write_parquet(path)
    return path


def load_option_bars(root: str, options_dir: Path | None = None) -> pl.DataFrame:
    """One row per option and day for *root* (see :func:`tidy`).

    Raises:
        FileNotFoundError: If the bars were never synced.
    """
    path = option_path(root, options_dir=options_dir)
    if not path.exists():
        raise FileNotFoundError(f"No option bars at {path}: run sync_option_bars({root!r}, ...)")
    return tidy(pl.read_parquet(path))
