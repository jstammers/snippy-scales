"""Tests for BacktestStore — DuckDB single-run persistence layer.

Covers:
* Schema creation: backtest_runs table exists after init
* save_run: row is written with correct metric values
* save_run: returns the correct run_id (auto and custom)
* save_run: idempotent with same id (INSERT OR REPLACE)
* save_run: metadata fields are stored correctly
* Context-manager protocol
"""

from __future__ import annotations

import numpy as np
import pytest

from snippy_scales.backtesting.domain import (
    BacktestMetrics,
    BacktestResult,
)
from snippy_scales.backtesting.store import BacktestStore

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_metrics(**overrides: float | int) -> BacktestMetrics:
    defaults: dict[str, float | int] = {
        "total_return_pct": 10.0,
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.5,
        "calmar_ratio": 0.8,
        "omega_ratio": 1.4,
        "max_drawdown_pct": -12.0,
        "max_drawdown_duration": 30,
        "total_trades": 50,
        "total_closed_trades": 48,
        "total_open_trades": 2,
        "winning_trades": 28,
        "losing_trades": 20,
        "win_rate_pct": 56.0,
        "profit_factor": 1.6,
        "expectancy": 250.0,
        "sqn": 2.1,
        "avg_trade_return_pct": 0.5,
        "avg_win_pct": 1.2,
        "avg_loss_pct": -0.8,
        "best_trade_pct": 4.5,
        "worst_trade_pct": -3.0,
        "payoff_ratio": 1.5,
        "recovery_factor": 3.2,
        "avg_holding_period": 5.0,
        "avg_winning_duration": 6.0,
        "avg_losing_duration": 4.0,
        "max_consecutive_wins": 7,
        "max_consecutive_losses": 5,
        "start_value": 100_000.0,
        "end_value": 110_000.0,
        "total_fees_paid": 500.0,
        "open_trade_pnl": 0.0,
        "exposure_pct": 60.0,
    }
    defaults.update(overrides)
    return BacktestMetrics(**defaults)  # type: ignore[arg-type]


def _make_result(**metric_overrides: float | int) -> BacktestResult:
    n = 50
    rng = np.random.default_rng(42)
    equity = 100_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, n))
    return BacktestResult(
        symbol="SIM",
        metrics=_make_metrics(**metric_overrides),
        equity_curve=equity,
        drawdown_curve=np.zeros(n),
        returns=np.diff(equity) / equity[:-1],
    )


@pytest.fixture
def store() -> BacktestStore:
    """In-memory BacktestStore; closed automatically after each test."""
    s = BacktestStore(":memory:")
    yield s
    s.close()


# ── Schema ────────────────────────────────────────────────────────────────────


def test_schema_creates_backtest_runs_table(store: BacktestStore) -> None:
    tables = {row[0] for row in store.query("SELECT table_name FROM information_schema.tables")}
    assert "backtest_runs" in tables


def test_schema_does_not_create_walk_forward_tables(store: BacktestStore) -> None:
    tables = {row[0] for row in store.query("SELECT table_name FROM information_schema.tables")}
    assert "walk_forward_runs" not in tables
    assert "walk_forward_folds" not in tables
    assert "walk_forward_summary" not in tables


def test_create_schema_is_idempotent(store: BacktestStore) -> None:
    """Calling create_schema() a second time should not raise."""
    store.create_schema()
    store.create_schema()


# ── save_run ──────────────────────────────────────────────────────────────────


def test_save_run_returns_string_id(store: BacktestStore) -> None:
    run_id = store.save_run(_make_result(), strategy_name="Test")
    assert isinstance(run_id, str) and len(run_id) > 0


def test_save_run_custom_id_is_used(store: BacktestStore) -> None:
    run_id = store.save_run(_make_result(), run_id="my-fixed-id")
    assert run_id == "my-fixed-id"
    rows = store.query("SELECT id FROM backtest_runs WHERE id = ?", ["my-fixed-id"])
    assert len(rows) == 1


def test_save_run_metric_roundtrip(store: BacktestStore) -> None:
    result = _make_result(sharpe_ratio=2.34, total_return_pct=42.0)
    store.save_run(result, run_id="rt-test")
    rows = store.query(
        "SELECT sharpe_ratio, total_return_pct FROM backtest_runs WHERE id = ?",
        ["rt-test"],
    )
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(2.34)
    assert rows[0][1] == pytest.approx(42.0)


def test_save_run_all_33_metrics_stored(store: BacktestStore) -> None:
    """backtest_runs table should have exactly 33 metric columns + 7 metadata."""
    store.save_run(_make_result(), run_id="col-test")
    cols = {
        row[0]
        for row in store.query(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'backtest_runs'"
        )
    }
    for metric in (
        "total_return_pct",
        "sharpe_ratio",
        "omega_ratio",
        "max_drawdown_duration",
        "sqn",
        "recovery_factor",
        "exposure_pct",
        "total_fees_paid",
        "open_trade_pnl",
    ):
        assert metric in cols, f"Column {metric!r} missing from backtest_runs"


def test_save_run_idempotent_with_same_id(store: BacktestStore) -> None:
    store.save_run(_make_result(), run_id="dup")
    store.save_run(_make_result(sharpe_ratio=9.9), run_id="dup")
    count = store.query("SELECT COUNT(*) FROM backtest_runs WHERE id = 'dup'")[0][0]
    assert count == 1
    # Latest value wins
    sr = store.query("SELECT sharpe_ratio FROM backtest_runs WHERE id = 'dup'")[0][0]
    assert sr == pytest.approx(9.9)


def test_save_run_strategy_metadata_stored(store: BacktestStore) -> None:
    store.save_run(
        _make_result(),
        run_id="meta-test",
        strategy_name="MeanReversion",
        initial_capital=50_000.0,
        fees=0.0005,
        slippage=0.0002,
    )
    row = store.query(
        "SELECT strategy_name, initial_capital, fees, slippage "
        "FROM backtest_runs WHERE id = 'meta-test'"
    )[0]
    assert row[0] == "MeanReversion"
    assert row[1] == pytest.approx(50_000.0)
    assert row[2] == pytest.approx(0.0005)
    assert row[3] == pytest.approx(0.0002)


# ── Context manager ───────────────────────────────────────────────────────────


def test_store_context_manager() -> None:
    with BacktestStore(":memory:") as s:
        run_id = s.save_run(_make_result(), run_id="ctx-test")
        count = s.query("SELECT COUNT(*) FROM backtest_runs")[0][0]
    assert count == 1
    assert run_id == "ctx-test"
