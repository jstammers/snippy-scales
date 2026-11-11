"""Unit tests for snippy_scales.data.alpaca — fetching and syncing into the catalog.

Fully offline: the only network boundary in ``alpaca-py`` —
``RESTClient._request`` — is mocked, so pagination, rate limiting and timestamp
handling run through real SDK code.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from snippy_scales.data import catalog as cat
from snippy_scales.data.alpaca import (
    AlpacaCredentialsError,
    AlpacaLoader,
    resolve_window,
    weekday_sessions,
)

if TYPE_CHECKING:
    from pathlib import Path

    from nautilus_trader.persistence import ParquetDataCatalog

_REQUEST_PATH = "alpaca.common.rest.RESTClient._request"
_JAN1 = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
_JAN3 = dt.datetime(2024, 1, 3, tzinfo=dt.UTC)


def _ns(when: dt.datetime) -> int:
    return int(when.timestamp()) * 1_000_000_000


@pytest.fixture(autouse=True)
def _alpaca_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")


@pytest.fixture
def loader() -> AlpacaLoader:
    return AlpacaLoader(rate_limit_per_min=6_000)


@pytest.fixture
def catalog(tmp_path: Path) -> ParquetDataCatalog:
    return cat.open_catalog(tmp_path / "catalog")


def _bar(t: str, c: float = 105.0) -> dict:
    return {"t": t, "o": 100.0, "h": 110.0, "l": 90.0, "c": c, "v": 1_000, "n": 10, "vw": 101.0}


def _response(symbol: str, bars: list[dict], *, next_page_token: str | None = None) -> dict:
    return {"bars": {symbol: bars}, "next_page_token": next_page_token}


class TestCredentials:
    def test_missing_credentials_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(AlpacaCredentialsError):
            AlpacaLoader()

    def test_explicit_credentials_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        assert AlpacaLoader(api_key="k", secret_key="s").venue == "ALPACA"


class TestFetchBars:
    def test_unsupported_schema_raises(self, loader: AlpacaLoader) -> None:
        with pytest.raises(ValueError, match="does not support schema"):
            loader.fetch_bars("AAPL", "ohlcv-1s", _JAN1, _JAN3)

    def test_single_page(self, loader: AlpacaLoader) -> None:
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [_bar("2024-01-02T14:30:00Z")])):
            df = loader.fetch_bars("AAPL", "ohlcv-1m", _JAN1, _JAN3)
        assert df.columns == ["ts_event", "open", "high", "low", "close", "volume"]
        assert df["close"].to_list() == [105.0]

    def test_pagination_and_rate_limit_per_http_call(self, loader: AlpacaLoader) -> None:
        page1 = _response("AAPL", [_bar("2024-01-02T14:30:00Z")], next_page_token="tok-2")
        page2 = _response("AAPL", [_bar("2024-01-02T14:31:00Z")])
        with (
            patch(_REQUEST_PATH, side_effect=[page1, page2]) as request,
            patch.object(loader._rate_limiter, "acquire") as acquire,  # noqa: SLF001
        ):
            df = loader.fetch_bars("AAPL", "ohlcv-1m", _JAN1, _JAN3)
        assert request.call_count == 2
        assert acquire.call_count == 2
        assert df.height == 2

    def test_empty_response(self, loader: AlpacaLoader) -> None:
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [])):
            assert loader.fetch_bars("AAPL", "ohlcv-1m", _JAN1, _JAN3).height == 0

    def test_start_past_end_makes_no_request(self, loader: AlpacaLoader) -> None:
        with patch(_REQUEST_PATH) as request:
            assert loader.fetch_bars("AAPL", "ohlcv-1m", _JAN3, _JAN1).height == 0
        request.assert_not_called()

    def test_daily_bars_normalised_to_midnight_utc(self, loader: AlpacaLoader) -> None:
        # Alpaca stamps daily bars at midnight US/Eastern; winter EST is UTC-5.
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [_bar("2024-01-02T05:00:00Z")])):
            ts = loader.fetch_bars("AAPL", "ohlcv-1d", _JAN1, _JAN3)["ts_event"][0]
        assert ts == dt.datetime(2024, 1, 2, tzinfo=dt.UTC)

    def test_minute_bars_keep_intraday_timestamp(self, loader: AlpacaLoader) -> None:
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [_bar("2024-01-02T14:31:00Z")])):
            ts = loader.fetch_bars("AAPL", "ohlcv-1m", _JAN1, _JAN3)["ts_event"][0]
        assert (ts.hour, ts.minute) == (14, 31)


class TestSync:
    def _requests(self, request: Any) -> list[tuple[str, str]]:
        """(start, end) query params of every mocked HTTP call."""
        return [(c.args[2]["start"], c.args[2]["end"]) for c in request.call_args_list]

    def test_writes_bars_and_instrument(
        self, loader: AlpacaLoader, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        bars = [_bar("2024-01-02T14:30:00Z", 105.0), _bar("2024-01-02T14:31:00Z", 105.1234)]
        with patch(_REQUEST_PATH, return_value=_response("AAPL", bars)):
            results = loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))

        assert results == [cat.SyncResult("AAPL", "ohlcv-1m", written=2)]
        assert [str(i.id) for i in catalog.instruments()] == ["AAPL.ALPACA"]
        df = cat.load_bars("AAPL", "ohlcv-1m", catalog_path=tmp_path / "catalog")
        assert df["close"].to_list() == [105.0, 105.1234]
        assert (df["ts_init"] - df["ts_event"]).dt.total_seconds().to_list() == [60, 60]

    def test_resume_requests_only_after_last_bar(
        self, loader: AlpacaLoader, catalog: ParquetDataCatalog
    ) -> None:
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [_bar("2024-01-02T14:30:00Z")])):
            loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [])) as request:
            loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        starts = [start for start, _ in self._requests(request)]
        # The 1.6-day head gap is non-trading time, so only the tail is requested,
        # starting one bar early (14:30) to compare the adjustment basis at the seam.
        assert starts == ["2024-01-02T14:30:00+00:00"]

    def test_raw_adjustment_skips_seam_overlap(self, catalog: ParquetDataCatalog) -> None:
        loader = AlpacaLoader(rate_limit_per_min=6_000, adjustment="raw")
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [_bar("2024-01-02T14:30:00Z")])):
            loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [])) as request:
            loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        assert [s for s, _ in self._requests(request)] == ["2024-01-02T14:31:00+00:00"]

    def test_matching_seam_appends(
        self, loader: AlpacaLoader, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [_bar("2024-01-02T14:30:00Z")])):
            loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        tail = [_bar("2024-01-02T14:30:00Z"), _bar("2024-01-02T14:31:00Z", 106.0)]
        with patch(_REQUEST_PATH, return_value=_response("AAPL", tail)):
            (result,) = loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        assert result == cat.SyncResult("AAPL", "ohlcv-1m", written=1)
        df = cat.load_bars("AAPL", "ohlcv-1m", catalog_path=tmp_path / "catalog")
        assert df["close"].to_list() == [105.0, 106.0]

    def test_adjustment_drift_fails_without_appending(
        self, loader: AlpacaLoader, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        with patch(_REQUEST_PATH, return_value=_response("AAPL", [_bar("2024-01-02T14:30:00Z")])):
            loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        # A dividend re-based history: the overlapping bar now closes 0.25% lower.
        tail = [_bar("2024-01-02T14:30:00Z", 104.7375), _bar("2024-01-02T14:31:00Z", 106.0)]
        with patch(_REQUEST_PATH, return_value=_response("AAPL", tail)):
            (result,) = loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        assert not result.succeeded
        assert "AdjustmentDriftError" in (result.error or "")
        df = cat.load_bars("AAPL", "ohlcv-1m", catalog_path=tmp_path / "catalog")
        assert df["close"].to_list() == [105.0]

    def test_rebuild_adjusted_replaces_series(
        self, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        loader = AlpacaLoader(rate_limit_per_min=6_000, rebuild_adjusted=True)
        stored = [_bar("2024-01-02T14:30:00Z"), _bar("2024-01-02T14:31:00Z", 105.5)]
        with patch(_REQUEST_PATH, return_value=_response("AAPL", stored)):
            loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        rebased = [
            _bar("2024-01-02T14:30:00Z", 104.7375),
            _bar("2024-01-02T14:31:00Z", 105.2363),
            _bar("2024-01-02T14:32:00Z", 106.0),
        ]
        with patch(_REQUEST_PATH, return_value=_response("AAPL", rebased)):
            (result,) = loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        assert result == cat.SyncResult("AAPL", "ohlcv-1m", written=3)
        df = cat.load_bars("AAPL", "ohlcv-1m", catalog_path=tmp_path / "catalog")
        assert df["close"].to_list() == [104.7375, 105.2363, 106.0]

    def test_rebuild_refuses_partial_download(
        self, catalog: ParquetDataCatalog, tmp_path: Path
    ) -> None:
        loader = AlpacaLoader(rate_limit_per_min=6_000, rebuild_adjusted=True)
        stored = [_bar("2024-01-02T14:30:00Z"), _bar("2024-01-02T14:31:00Z", 105.5)]
        with patch(_REQUEST_PATH, return_value=_response("AAPL", stored)):
            loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        # Seam bar drifted, but the "full" re-download is missing the first stored bar.
        partial = [_bar("2024-01-02T14:31:00Z", 104.0), _bar("2024-01-02T14:32:00Z", 106.0)]
        with patch(_REQUEST_PATH, return_value=_response("AAPL", partial)):
            (result,) = loader.sync(catalog, ["AAPL"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))
        assert not result.succeeded
        df = cat.load_bars("AAPL", "ohlcv-1m", catalog_path=tmp_path / "catalog")
        assert df["close"].to_list() == [105.0, 105.5]  # untouched

    def test_failure_is_recorded_per_symbol(
        self, loader: AlpacaLoader, catalog: ParquetDataCatalog
    ) -> None:
        def respond(method: str, path: str, data: dict, *a: Any, **k: Any) -> dict:
            if "MSFT" in path or data.get("symbols") == "MSFT":
                raise RuntimeError("boom")
            return _response("AAPL", [_bar("2024-01-02T14:30:00Z")])

        with patch(_REQUEST_PATH, side_effect=respond):
            results = loader.sync(catalog, ["AAPL", "MSFT"], "ohlcv-1m", _ns(_JAN1), _ns(_JAN3))

        by_symbol = {r.symbol: r for r in results}
        assert by_symbol["AAPL"].succeeded
        assert not by_symbol["MSFT"].succeeded
        assert "boom" in (by_symbol["MSFT"].error or "")


class TestWindow:
    """Window resolution: first session on/after start, last *completed* session."""

    @staticmethod
    def _cal(first: dt.date, last: dt.date) -> list[dt.date]:
        # Weekdays minus 2024-01-01 (New Year) and 2024-01-15 (MLK day).
        holidays = {dt.date(2024, 1, 1), dt.date(2024, 1, 15)}
        days = (first + dt.timedelta(days=i) for i in range((last - first).days + 1))
        return [d for d in days if d.weekday() < 5 and d not in holidays]

    def _window(self, start: dt.date | None, end: dt.date | None, now: dt.datetime) -> tuple:
        lo, hi = resolve_window(
            _ns(dt.datetime.combine(start, dt.time(), dt.UTC)) if start else None,
            _ns(dt.datetime.combine(end, dt.time(), dt.UTC)) if end else None,
            self._cal,
            now=now,
        )
        to_date = lambda ns: dt.datetime.fromtimestamp(ns / 1e9, dt.UTC).date()  # noqa: E731
        return to_date(lo), to_date(hi)

    def test_start_snaps_past_holiday_and_end_is_last_completed_session(self) -> None:
        # Wednesday 2024-01-17 20:00 UTC: the Wednesday session isn't complete yet.
        now = dt.datetime(2024, 1, 17, 20, tzinfo=dt.UTC)
        assert self._window(dt.date(2023, 12, 30), None, now) == (
            dt.date(2024, 1, 2),  # Sat 30th, Sun 31st, holiday 1st skipped
            dt.date(2024, 1, 17),  # exclusive: last completed session is Tue 16th
        )

    def test_session_completes_after_extended_hours_and_embargo(self) -> None:
        before = dt.datetime(2024, 1, 18, 1, 15, tzinfo=dt.UTC)
        after = dt.datetime(2024, 1, 18, 1, 17, tzinfo=dt.UTC)
        assert self._window(dt.date(2024, 1, 2), None, before)[1] == dt.date(2024, 1, 17)
        assert self._window(dt.date(2024, 1, 2), None, after)[1] == dt.date(2024, 1, 18)

    def test_default_start_and_explicit_end(self) -> None:
        now = dt.datetime(2024, 2, 1, tzinfo=dt.UTC)
        lo, hi = self._window(None, dt.date(2024, 1, 13), now)  # end on a Saturday
        assert lo == dt.date(2016, 1, 1)  # HISTORY_START (a Friday in this calendar)
        assert hi == dt.date(2024, 1, 13)  # day after Friday the 12th

    def test_no_completed_session_gives_empty_window(self) -> None:
        now = dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC)  # first session still running
        lo, hi = self._window(dt.date(2024, 1, 1), None, now)
        assert lo == hi

    def test_weekday_fallback(self) -> None:
        days = weekday_sessions(dt.date(2024, 1, 5), dt.date(2024, 1, 8))
        assert days == [dt.date(2024, 1, 5), dt.date(2024, 1, 8)]
