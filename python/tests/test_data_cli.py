"""Tests for the `algo data` CLI: sync planning, coverage, consolidate."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl
from typer.testing import CliRunner

from snippy_scales.cli.data import app
from snippy_scales.data import catalog as cat

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_runner = CliRunner()
_WIDE = {"COLUMNS": "300"}


def _seed_catalog(root: Path) -> None:
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(5)]
    frame = pl.DataFrame(
        {
            "ts_event": pl.Series(days).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"),
            "open": [4500.0] * 5,
            "high": [4501.0] * 5,
            "low": [4499.0] * 5,
            "close": [4500.25] * 5,
            "volume": [10.0] * 5,
        }
    )
    cat.write_bars(
        cat.open_catalog(root),
        cat.frame_to_bars(
            frame,
            bar_type=cat.bar_type("ES.c.0", "GLBX", "ohlcv-1d"),
            price_precision=2,
            interval_ns=cat.bar_interval_ns("ohlcv-1d"),
        ),
    )


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    return path


def test_databento_dry_run_shows_missing_ranges_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    root = tmp_path / "catalog"
    _seed_catalog(root)
    cfg = _config(
        tmp_path,
        "dataset: GLBX.MDP3\nschemas: ['1d']\nstart: '2024-01-01'\nend: '2024-01-10'\n"
        "asset_classes:\n  idx:\n    symbols: [ES.c.0, NQ.c.0]\n",
    )
    result = _runner.invoke(app, ["sync", str(cfg), "--catalog", str(root), "--dry-run"], env=_WIDE)

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    # ES.c.0 has Jan 1-5 stored, so only the tail from the last close is missing.
    assert "2024-01-06 00:00 → 2024-01-10 00:00" in result.output
    # NQ.c.0 has nothing stored: the whole range is missing.
    assert "2024-01-01 00:00 → 2024-01-10 00:00" in result.output


def test_alpaca_dry_run_needs_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    cfg = _config(
        tmp_path,
        "provider: alpaca\nschemas: ['1d']\nstart: '2024-01-01'\nend: '2024-01-10'\n"
        "asset_classes:\n  eq:\n    symbols: [AAPL, MSFT]\n",
    )
    result = _runner.invoke(
        app, ["sync", str(cfg), "--catalog", str(tmp_path / "catalog"), "--dry-run"], env=_WIDE
    )
    assert result.exit_code == 0, result.output
    assert "2 of 2" in result.output


def test_instruments_only_rejects_alpaca(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        "provider: alpaca\nstart: '2024-01-01'\nasset_classes:\n  eq:\n    symbols: [AAPL]\n",
    )
    result = _runner.invoke(
        app, ["sync", str(cfg), "--catalog", str(tmp_path / "c"), "--instruments-only"]
    )
    assert result.exit_code == 1


def test_coverage_lists_series(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    _seed_catalog(root)
    result = _runner.invoke(app, ["coverage", "--catalog", str(root)], env=_WIDE)
    assert result.exit_code == 0, result.output
    assert "ES.c.0" in result.output
    assert "2024-01-06 00:00" in result.output  # last bar close


def test_coverage_empty_catalog(tmp_path: Path) -> None:
    result = _runner.invoke(app, ["coverage", "--catalog", str(tmp_path / "catalog")])
    assert result.exit_code == 0
    assert "No bar data" in result.output


def test_consolidate_keeps_data(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    _seed_catalog(root)
    result = _runner.invoke(app, ["consolidate", "--catalog", str(root)])
    assert result.exit_code == 0, result.output
    assert cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=root).height == 5
