"""Unit tests for the Alpaca BarProvider (snippy_scales.data.providers.alpaca).

Fully offline: the only network boundary in the ``alpaca-py`` SDK —
``RESTClient._request`` — is mocked, so pagination, rate-limiting, schema
mapping, and the shared-schema conversion all run through real SDK code
without ever hitting the network.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from snippy_scales.data.providers.alpaca import AlpacaCredentialsError, AlpacaProvider
from snippy_scales.data.schema import BAR_SCHEMA_COLUMNS

_REQUEST_PATH = "alpaca.common.rest.RESTClient._request"


@pytest.fixture(autouse=True)
def _alpaca_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")


def _bar(
    t: str,
    *,
    o: float = 100.0,
    h: float = 110.0,
    low: float = 90.0,
    c: float = 105.0,
    v: int = 1_000,
    n: int = 10,
    vw: float = 101.0,
) -> dict:
    return {"t": t, "o": o, "h": h, "l": low, "c": c, "v": v, "n": n, "vw": vw}


def _bars_response(symbol: str, bars: list[dict], *, next_page_token: str | None = None) -> dict:
    return {"bars": {symbol: bars}, "next_page_token": next_page_token}


class TestAlpacaProviderCredentials:
    def test_missing_credentials_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(AlpacaCredentialsError):
            AlpacaProvider()

    def test_explicit_credentials_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        provider = AlpacaProvider(api_key="k", secret_key="s")
        assert provider.name == "alpaca"

    def test_resume_granularity_is_timestamp(self) -> None:
        assert AlpacaProvider().resume_granularity == "timestamp"


class TestAlpacaProviderFetchBars:
    def test_unsupported_schema_raises(self) -> None:
        provider = AlpacaProvider()
        with pytest.raises(ValueError, match="does not support schema"):
            provider.fetch_bars(
                symbol="AAPL", schema="ohlcv-1s", start="2024-01-01", end="2024-01-02"
            )

    def test_single_page(self) -> None:
        provider = AlpacaProvider(rate_limit_per_min=6_000)
        response = _bars_response("AAPL", [_bar("2024-01-02T14:30:00Z")])

        with patch(_REQUEST_PATH, return_value=response) as mock_request:
            df = provider.fetch_bars(
                symbol="AAPL", schema="ohlcv-1m", start="2024-01-01", end="2024-01-03"
            )

        assert mock_request.call_count == 1
        assert len(df) == 1
        assert list(df.columns) == [*BAR_SCHEMA_COLUMNS, "trade_count", "vwap"]
        assert df["close"].to_list() == [105.0]
        assert df["symbol"].to_list() == ["AAPL"]
        assert df["rtype"].to_list() == [33]  # ohlcv-1m

    def test_pagination_follows_next_page_token(self) -> None:
        provider = AlpacaProvider(rate_limit_per_min=6_000)
        page1 = _bars_response("AAPL", [_bar("2024-01-02T14:30:00Z")], next_page_token="tok-2")
        page2 = _bars_response("AAPL", [_bar("2024-01-02T14:31:00Z")])

        with patch(_REQUEST_PATH, side_effect=[page1, page2]) as mock_request:
            df = provider.fetch_bars(
                symbol="AAPL", schema="ohlcv-1m", start="2024-01-01", end="2024-01-03"
            )

        assert mock_request.call_count == 2
        assert len(df) == 2

    def test_rate_limiter_acquired_once_per_http_call(self) -> None:
        provider = AlpacaProvider(rate_limit_per_min=6_000)
        page1 = _bars_response("AAPL", [_bar("2024-01-02T14:30:00Z")], next_page_token="tok-2")
        page2 = _bars_response("AAPL", [_bar("2024-01-02T14:31:00Z")])

        with (
            patch(_REQUEST_PATH, side_effect=[page1, page2]),
            patch.object(provider._rate_limiter, "acquire") as mock_acquire,  # noqa: SLF001
        ):
            provider.fetch_bars(
                symbol="AAPL", schema="ohlcv-1m", start="2024-01-01", end="2024-01-03"
            )

        assert mock_acquire.call_count == 2

    def test_empty_response_returns_empty_conformed_frame(self) -> None:
        provider = AlpacaProvider(rate_limit_per_min=6_000)
        response = _bars_response("AAPL", [])

        with patch(_REQUEST_PATH, return_value=response):
            df = provider.fetch_bars(
                symbol="AAPL", schema="ohlcv-1m", start="2024-01-01", end="2024-01-03"
            )

        assert len(df) == 0
        assert list(df.columns) == list(BAR_SCHEMA_COLUMNS)

    def test_start_past_end_returns_empty_without_any_request(self) -> None:
        provider = AlpacaProvider(rate_limit_per_min=6_000)

        with patch(_REQUEST_PATH) as mock_request:
            df = provider.fetch_bars(
                symbol="AAPL", schema="ohlcv-1m", start="2024-01-05", end="2024-01-01"
            )

        mock_request.assert_not_called()
        assert len(df) == 0

    def test_daily_bars_normalised_to_midnight_utc(self) -> None:
        provider = AlpacaProvider(rate_limit_per_min=6_000)
        # Alpaca stamps daily bars at session-open in US/Eastern; winter EST is UTC-5.
        response = _bars_response("AAPL", [_bar("2024-01-02T05:00:00Z")])

        with patch(_REQUEST_PATH, return_value=response):
            df = provider.fetch_bars(
                symbol="AAPL", schema="ohlcv-1d", start="2024-01-01", end="2024-01-03"
            )

        ts = df["ts_event"][0]
        assert (ts.hour, ts.minute, ts.second) == (0, 0, 0)
        assert ts.date().isoformat() == "2024-01-02"

    def test_minute_bars_keep_intraday_timestamp(self) -> None:
        provider = AlpacaProvider(rate_limit_per_min=6_000)
        response = _bars_response("AAPL", [_bar("2024-01-02T14:31:00Z")])

        with patch(_REQUEST_PATH, return_value=response):
            df = provider.fetch_bars(
                symbol="AAPL", schema="ohlcv-1m", start="2024-01-01", end="2024-01-03"
            )

        ts = df["ts_event"][0]
        assert (ts.hour, ts.minute) == (14, 31)
