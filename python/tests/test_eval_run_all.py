"""Tests for the ``algo eval run-all`` command and ``_align_bars`` helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
from typer.testing import CliRunner

from snippy_scales.cli.eval import _align_bars, app

# ── helpers ───────────────────────────────────────────────────────────────────

_BASE_NS = 1_577_836_800_000_000_000  # 2020-01-01 UTC in nanoseconds
_DAY_NS = 86_400_000_000_000


def _make_bars(n: int = 400, seed: int = 0, base_ns: int = _BASE_NS) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = rng.uniform(1_000, 5_000, n)
    timestamps = base_ns + np.arange(n, dtype=np.int64) * _DAY_NS
    return pl.DataFrame(
        {
            "ts_event": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_fake_result(symbols: list[str] | None = None) -> Any:
    """Minimal EvaluationResult usable as a mock return value."""
    from snippy_scales.backtesting.domain import BacktestMetrics
    from snippy_scales.evaluation.results import EvaluationResult, FoldResult, SweepResult

    def _metrics() -> BacktestMetrics:
        return BacktestMetrics(
            total_return_pct=5.0,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            calmar_ratio=0.5,
            omega_ratio=1.2,
            max_drawdown_pct=-10.0,
            max_drawdown_duration=30,
            total_trades=20,
            total_closed_trades=20,
            total_open_trades=0,
            winning_trades=12,
            losing_trades=8,
            win_rate_pct=60.0,
            profit_factor=1.3,
            expectancy=25.0,
            sqn=0.0,
            avg_trade_return_pct=0.25,
            avg_win_pct=0.5,
            avg_loss_pct=-0.3,
            best_trade_pct=2.0,
            worst_trade_pct=-1.5,
            payoff_ratio=1.7,
            recovery_factor=0.5,
            avg_holding_period=5.0,
            avg_winning_duration=6.0,
            avg_losing_duration=4.0,
            max_consecutive_wins=4,
            max_consecutive_losses=3,
            start_value=1_000_000.0,
            end_value=1_050_000.0,
            total_fees_paid=200.0,
            open_trade_pnl=0.0,
            exposure_pct=50.0,
        )

    fold = FoldResult(
        fold_idx=0,
        params={},
        train_metrics=_metrics(),
        test_metrics=_metrics(),
        train_equity_curve=np.ones(200),
        test_equity_curve=np.ones(80),
        train_start="2020-01-01",
        train_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-12-31",
    )
    sweep = SweepResult(params={}, folds=[fold])
    return EvaluationResult(
        experiment_name="test",
        strategy_class="TrendFollowing",
        symbols=symbols or ["SYM"],
        sweep_results=[sweep],
    )


# ── _align_bars ───────────────────────────────────────────────────────────────


class TestAlignBars:
    def test_common_timestamps_are_retained(self) -> None:
        bars_a = _make_bars(300, seed=0)
        bars_b = _make_bars(300, seed=1)  # same timestamps
        aligned = _align_bars({"A": bars_a, "B": bars_b})
        assert len(aligned) == 2
        assert len(aligned["A"]) == 300
        assert len(aligned["B"]) == 300

    def test_partial_overlap_trims_to_intersection(self) -> None:
        # A: days 0-199, B: days 100-299 → overlap is days 100-199 (100 rows)
        bars_a = _make_bars(200, seed=0, base_ns=_BASE_NS)
        bars_b = _make_bars(200, seed=1, base_ns=_BASE_NS + 100 * _DAY_NS)
        aligned = _align_bars({"A": bars_a, "B": bars_b})
        assert len(aligned) == 2
        assert len(aligned["A"]) == 100
        assert len(aligned["B"]) == 100

    def test_no_overlap_returns_empty(self) -> None:
        # A: days 0-99, B: days 200-299 → no overlap
        bars_a = _make_bars(100, seed=0, base_ns=_BASE_NS)
        bars_b = _make_bars(100, seed=1, base_ns=_BASE_NS + 200 * _DAY_NS)
        aligned = _align_bars({"A": bars_a, "B": bars_b})
        assert aligned == {}

    def test_missing_ts_event_column_excluded(self) -> None:
        bars_ok = _make_bars(200, seed=0)
        bars_bad = bars_ok.drop("ts_event")  # no ts_event column
        aligned = _align_bars({"OK": bars_ok, "BAD": bars_bad})
        # BAD is excluded because it has no ts_event; OK is retained
        assert "BAD" not in aligned
        assert "OK" in aligned
        assert len(aligned["OK"]) == 200

    def test_single_entry_skipped(self) -> None:
        bars_a = _make_bars(200, seed=0)
        # Just one symbol is valid — intersection with a single ts_set is itself
        # but the caller is responsible for checking len >= 2.
        aligned = _align_bars({"A": bars_a})
        assert "A" in aligned
        assert len(aligned["A"]) == 200

    def test_output_is_sorted_by_ts_event(self) -> None:
        bars_a = _make_bars(200, seed=0)
        bars_b = _make_bars(200, seed=1)
        aligned = _align_bars({"A": bars_a, "B": bars_b})
        ts_a = aligned["A"]["ts_event"].to_list()
        assert ts_a == sorted(ts_a)

    def test_empty_input_returns_empty(self) -> None:
        assert _align_bars({}) == {}


# ── run-all CLI ───────────────────────────────────────────────────────────────

_runner = CliRunner()


class TestRunAllCommand:
    def test_no_data_exits_nonzero(self, tmp_path: Path) -> None:
        result = _runner.invoke(app, ["run-all", "--data-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "No" in result.output

    def test_runs_single_asset_strategies_per_symbol(self, tmp_path: Path) -> None:
        """Single symbol: single-asset strategies run; multi-asset is skipped."""
        sym_dir = tmp_path / "ES.c.0"
        sym_dir.mkdir()
        _make_bars(500).write_parquet(sym_dir / "ohlcv-1d.parquet")

        fake_result = _make_fake_result(["ES.c.0"])

        with patch("snippy_scales.evaluation.EvaluationRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.evaluate.return_value = fake_result
            instance.evaluate_multi_asset.return_value = fake_result

            result = _runner.invoke(
                app, ["run-all", "--data-dir", str(tmp_path), "--benchmark", "none"]
            )

        # Should succeed
        assert result.exit_code == 0, result.output
        # Multi-asset skipped because only 1 symbol
        assert "multi-asset" not in result.output.lower() or "Skipping" in result.output
        # Summary table rendered
        assert "Summary" in result.output

    def test_runs_multi_asset_strategy_with_multiple_symbols(self, tmp_path: Path) -> None:
        """Two symbols: all strategies including multi-asset run."""
        for sym in ["ES.c.0", "NQ.c.0"]:
            sym_dir = tmp_path / sym
            sym_dir.mkdir()
            _make_bars(500).write_parquet(sym_dir / "ohlcv-1d.parquet")

        fake_single = _make_fake_result(["ES.c.0"])
        fake_multi = _make_fake_result(["ES.c.0", "NQ.c.0"])

        with patch("snippy_scales.evaluation.EvaluationRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.evaluate.return_value = fake_single
            instance.evaluate_multi_asset.return_value = fake_multi

            result = _runner.invoke(
                app, ["run-all", "--data-dir", str(tmp_path), "--benchmark", "none"]
            )

        assert result.exit_code == 0, result.output
        # Both single-asset and multi-asset runs happened
        assert instance.evaluate.called
        assert instance.evaluate_multi_asset.called
        assert "Summary" in result.output

    def test_error_in_one_strategy_does_not_abort(self, tmp_path: Path) -> None:
        """An error in one strategy is caught; remaining strategies still run."""
        sym_dir = tmp_path / "ES.c.0"
        sym_dir.mkdir()
        _make_bars(500).write_parquet(sym_dir / "ohlcv-1d.parquet")

        fake_result = _make_fake_result(["ES.c.0"])

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated backtest failure")
            return fake_result

        with patch("snippy_scales.evaluation.EvaluationRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.evaluate.side_effect = side_effect
            instance.evaluate_multi_asset.return_value = fake_result

            result = _runner.invoke(
                app, ["run-all", "--data-dir", str(tmp_path), "--benchmark", "none"]
            )

        assert result.exit_code == 0, result.output
        assert "error" in result.output.lower()

    def test_respects_schema_option(self, tmp_path: Path) -> None:
        """--schema filters which parquet files are discovered."""
        sym_dir = tmp_path / "ES.c.0"
        sym_dir.mkdir()
        # Write ohlcv-1h.parquet but NOT ohlcv-1d.parquet
        _make_bars(400).write_parquet(sym_dir / "ohlcv-1h.parquet")

        # Default schema is ohlcv-1d → no files found
        result = _runner.invoke(app, ["run-all", "--data-dir", str(tmp_path)])
        assert result.exit_code == 1

        fake_result = _make_fake_result(["ES.c.0"])
        with patch("snippy_scales.evaluation.EvaluationRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.evaluate.return_value = fake_result
            instance.evaluate_multi_asset.return_value = fake_result

            result = _runner.invoke(
                app,
                [
                    "run-all",
                    "--data-dir",
                    str(tmp_path),
                    "--schema",
                    "ohlcv-1h",
                    "--benchmark",
                    "none",
                ],
            )
        assert result.exit_code == 0, result.output
