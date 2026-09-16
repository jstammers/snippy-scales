"""Unit tests for the `algo data check-layout` and `algo data migrate-layout` CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from snippy_scales.cli.data import app

if TYPE_CHECKING:
    from pathlib import Path

_runner = CliRunner()


class TestCheckLayout:
    def test_reports_clean_when_no_duplicates(self, tmp_path: Path) -> None:
        (tmp_path / "equities" / "AAPL").mkdir(parents=True)
        (tmp_path / "futures" / "ES.c.0").mkdir(parents=True)

        result = _runner.invoke(app, ["check-layout", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "No cross-class duplicates" in result.output

    def test_fails_and_reports_duplicate(self, tmp_path: Path) -> None:
        (tmp_path / "equities" / "AAPL").mkdir(parents=True)
        (tmp_path / "futures" / "AAPL").mkdir(parents=True)

        result = _runner.invoke(app, ["check-layout", "--output-dir", str(tmp_path)])

        assert result.exit_code == 1
        assert "AAPL" in result.output
        assert "equities" in result.output
        assert "futures" in result.output

    def test_empty_output_dir_is_clean(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        result = _runner.invoke(app, ["check-layout", "--output-dir", str(missing)])
        assert result.exit_code == 0, result.output


class TestMigrateLayout:
    def test_dry_run_does_not_move_anything(self, tmp_path: Path) -> None:
        sym_dir = tmp_path / "AAPL"
        sym_dir.mkdir()
        (sym_dir / "ohlcv-1d.parquet").write_bytes(b"fake")

        result = _runner.invoke(
            app,
            ["migrate-layout", "--instrument-class", "equities", "--output-dir", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "Dry-run" in result.output
        # Nothing moved — original path still exists, destination does not.
        assert (tmp_path / "AAPL" / "ohlcv-1d.parquet").exists()
        assert not (tmp_path / "equities").exists()

    def test_yes_moves_flat_symbol_dirs_under_instrument_class(self, tmp_path: Path) -> None:
        for sym in ["AAPL", "MSFT"]:
            sym_dir = tmp_path / sym
            sym_dir.mkdir()
            (sym_dir / "ohlcv-1d.parquet").write_bytes(b"fake")

        result = _runner.invoke(
            app,
            [
                "migrate-layout",
                "--instrument-class",
                "equities",
                "--output-dir",
                str(tmp_path),
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.output
        assert not (tmp_path / "AAPL").exists()
        assert not (tmp_path / "MSFT").exists()
        assert (tmp_path / "equities" / "AAPL" / "ohlcv-1d.parquet").exists()
        assert (tmp_path / "equities" / "MSFT" / "ohlcv-1d.parquet").exists()

    def test_already_classified_dirs_are_left_alone(self, tmp_path: Path) -> None:
        (tmp_path / "futures" / "ES.c.0").mkdir(parents=True)
        (tmp_path / "futures" / "ES.c.0" / "ohlcv-1d.parquet").write_bytes(b"fake")

        result = _runner.invoke(
            app,
            [
                "migrate-layout",
                "--instrument-class",
                "equities",
                "--output-dir",
                str(tmp_path),
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Nothing to migrate" in result.output
        assert (tmp_path / "futures" / "ES.c.0" / "ohlcv-1d.parquet").exists()

    def test_refuses_when_destination_already_exists_and_is_non_empty(self, tmp_path: Path) -> None:
        sym_dir = tmp_path / "AAPL"
        sym_dir.mkdir()
        (sym_dir / "ohlcv-1d.parquet").write_bytes(b"fake-flat")

        dest_dir = tmp_path / "equities" / "AAPL"
        dest_dir.mkdir(parents=True)
        (dest_dir / "ohlcv-1d.parquet").write_bytes(b"fake-existing")

        result = _runner.invoke(
            app,
            [
                "migrate-layout",
                "--instrument-class",
                "equities",
                "--output-dir",
                str(tmp_path),
                "--yes",
            ],
        )

        assert result.exit_code == 1
        assert "Refusing" in result.output
        # Neither side was touched.
        assert (tmp_path / "AAPL" / "ohlcv-1d.parquet").read_bytes() == b"fake-flat"
        assert (dest_dir / "ohlcv-1d.parquet").read_bytes() == b"fake-existing"
