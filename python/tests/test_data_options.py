"""Option bars: symbol parsing, expiries and the tidy chain frame."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from snippy_scales.data import options as o


def test_parse_option_symbol() -> None:
    assert o.parse_option_symbol("ESH4 P4490") == ("ESH4", "P", 4490.0)
    assert o.parse_option_symbol("ESZ6 C4900") == ("ESZ6", "C", 4900.0)
    assert o.parse_option_symbol("UD:1V: SG 2719478") is None  # a user-defined spread
    assert o.parse_option_symbol("ESH4") is None


def test_expiry_is_the_third_friday_of_the_next_matching_year() -> None:
    assert o.third_friday(2024, 3) == dt.date(2024, 3, 15)
    assert o.expiry("ESH4", dt.date(2023, 12, 1)) == dt.date(2024, 3, 15)
    assert o.expiry("ESZ9", dt.date(2018, 6, 1)) == dt.date(2019, 12, 20)
    assert o.expiry("ESH0", dt.date(2019, 12, 20)) == dt.date(2020, 3, 20)  # decade wraps
    assert o.expiry("ESH4", dt.date(2024, 3, 15)) == dt.date(2024, 3, 15)  # on expiry day
    with pytest.raises(ValueError, match="not a futures"):
        o.expiry("AAPL", dt.date(2024, 1, 1))


def test_month_chunks_cover_the_range_without_gaps() -> None:
    chunks = o._month_chunks("2018-01-15", "2018-04-10")  # noqa: SLF001
    assert chunks == [
        ("2018-01-15", "2018-02-01"),
        ("2018-02-01", "2018-03-01"),
        ("2018-03-01", "2018-04-01"),
        ("2018-04-01", "2018-04-10"),
    ]


def _raw() -> pl.DataFrame:
    day = dt.datetime(2023, 12, 1, tzinfo=dt.UTC)
    return pl.DataFrame(
        {
            "ts_event": [day, day, day, day + dt.timedelta(days=3)],
            "symbol": ["ESH4 P4490", "UD:1V: SG 1", "ESH4 C4700", "ESH4 C4700"],
            "open": [50.0, 1.0, 40.0, 0.0],
            "high": [55.0, 1.0, 45.0, 0.0],
            "low": [48.0, 1.0, 38.0, 0.0],
            "close": [52.0, 1.0, 41.0, 0.0],
            "volume": [10.0, 1.0, 20.0, 0.0],
        }
    )


def test_tidy_keeps_priced_outright_options_and_dates_their_expiry() -> None:
    out = o.tidy(_raw())
    assert out["symbol"].to_list() == ["ESH4 P4490", "ESH4 C4700"]  # spread, zero close gone
    assert out["right"].to_list() == ["P", "C"] and out["strike"].to_list() == [4490.0, 4700.0]
    assert out["expiry"].to_list() == [dt.date(2024, 3, 15)] * 2
    assert out["days_to_expiry"].to_list() == [105, 105]


def test_load_reads_the_stored_file_and_sync_never_overwrites(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sync_option_bars"):
        o.load_option_bars("ES", options_dir=tmp_path)
    path = o.option_path("ES", options_dir=tmp_path)
    path.parent.mkdir(parents=True)
    _raw().write_parquet(path)
    assert o.load_option_bars("ES", options_dir=tmp_path).height == 2
    with pytest.raises(FileExistsError, match="never replaced"):  # before any request is made
        o.sync_option_bars("ES", "2018-01-01", "2018-02-01", options_dir=tmp_path)
