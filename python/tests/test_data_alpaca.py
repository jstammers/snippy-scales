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
from snippy_scales.data.alpaca import AlpacaCredentialsError, AlpacaLoader

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
        # Head (before the stored bar) and tail (from its close at 14:31) only.
        assert starts == ["2024-01-01T00:00:00+00:00", "2024-01-02T14:31:00+00:00"]

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
