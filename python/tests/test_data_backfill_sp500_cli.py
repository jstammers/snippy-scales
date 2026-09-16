"""Unit tests for the `algo data backfill-sp500` CLI's `--schema` option.

All tests use `--dry-run` (no download, no confirmation prompt) and a
pre-populated `--symbols-cache` file so the universe resolver never makes a
network call to Wikipedia.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from snippy_scales.cli.data import app

if TYPE_CHECKING:
    from pathlib import Path

_runner = CliRunner()

#: Rich truncates long table cells (e.g. an absolute tmp_path manifest path)
#: to a default 80-column width when there's no real terminal — widen it so
#: assertions on full path strings in the rendered output are reliable.
_WIDE_TERMINAL = {"COLUMNS": "300"}


def _symbols_cache(tmp_path: Path) -> Path:
    path = tmp_path / "universe.txt"
    path.write_text("AAPL\nMSFT\n")
    return path


class TestBackfillSp500Schema:
    def test_defaults_to_1m(self, tmp_path: Path) -> None:
        result = _runner.invoke(
            app,
            [
                "backfill-sp500",
                "--dry-run",
                "--symbols-cache",
                str(_symbols_cache(tmp_path)),
                "--output-dir",
                str(tmp_path),
            ],
            env=_WIDE_TERMINAL,
        )

        assert result.exit_code == 0, result.output
        assert "ohlcv-1m" in result.output
        assert "sp500_1m.csv" in result.output

    def test_daily_schema_resolves_and_uses_its_own_manifest(self, tmp_path: Path) -> None:
        result = _runner.invoke(
            app,
            [
                "backfill-sp500",
                "--schema",
                "1d",
                "--dry-run",
                "--symbols-cache",
                str(_symbols_cache(tmp_path)),
                "--output-dir",
                str(tmp_path),
            ],
            env=_WIDE_TERMINAL,
        )

        assert result.exit_code == 0, result.output
        assert "ohlcv-1d" in result.output
        assert "sp500_1d.csv" in result.output
        assert "sp500_1m.csv" not in result.output

    def test_tick_schema_is_rejected(self, tmp_path: Path) -> None:
        result = _runner.invoke(
            app,
            [
                "backfill-sp500",
                "--schema",
                "trades",
                "--dry-run",
                "--symbols-cache",
                str(_symbols_cache(tmp_path)),
                "--output-dir",
                str(tmp_path),
            ],
            env=_WIDE_TERMINAL,
        )

        assert result.exit_code == 1
        assert "does not support" in result.output

    def test_unknown_schema_is_rejected(self, tmp_path: Path) -> None:
        result = _runner.invoke(
            app,
            [
                "backfill-sp500",
                "--schema",
                "bogus",
                "--dry-run",
                "--symbols-cache",
                str(_symbols_cache(tmp_path)),
                "--output-dir",
                str(tmp_path),
            ],
            env=_WIDE_TERMINAL,
        )

        assert result.exit_code == 1
        assert "Unknown schema" in result.output

    def test_explicit_manifest_path_overrides_schema_default(self, tmp_path: Path) -> None:
        custom_manifest = tmp_path / "custom.csv"
        result = _runner.invoke(
            app,
            [
                "backfill-sp500",
                "--schema",
                "1d",
                "--dry-run",
                "--symbols-cache",
                str(_symbols_cache(tmp_path)),
                "--output-dir",
                str(tmp_path),
                "--manifest-path",
                str(custom_manifest),
            ],
            env=_WIDE_TERMINAL,
        )

        assert result.exit_code == 0, result.output
        assert "custom.csv" in result.output
        assert "sp500_1d.csv" not in result.output
