"""Tests for BacktestStore — DuckDB persistence layer.

Covers:
* Schema creation: all four tables exist after init
* save_run: row is written with correct metric values
* save_run: returns the correct run_id (auto and custom)
* save_walk_forward: rows written to all three WF tables
* save_walk_forward: fold metrics match source data
* save_walk_forward: summary mean/std match domain computation
* Context-manager protocol
* Repeated save_run with same id uses INSERT OR REPLACE (idempotent)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from snippy_scales.backtesting.domain import (
    BacktestMetrics,
    BacktestResult,
    FoldResult,
    WalkForwardResult,
)
from snippy_scales.backtesting.store import BacktestStore

# ── Helpers (shared with test_backtesting_domain) ─────────────────────────────

_BASE_NS = 1_577_836_800_000_000_000
_DAY_NS = 86_400_000_000_000


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


def _make_fold(index: int, **metric_overrides: float | int) -> FoldResult:
    oos_start = _BASE_NS + index * 90 * _DAY_NS
    oos_end = oos_start + 90 * _DAY_NS
    return FoldResult(
        fold_index=index,
        oos_start=oos_start,
        oos_end=oos_end,
        result=_make_result(**metric_overrides),
    )


@pytest.fixture
def store() -> BacktestStore:
    """In-memory BacktestStore; closed automatically after each test."""
    s = BacktestStore(":memory:")
    yield s
    s.close()


# ── Schema ────────────────────────────────────────────────────────────────────


def test_schema_creates_all_four_tables(store: BacktestStore) -> None:
    tables = {row[0] for row in store.query("SELECT table_name FROM information_schema.tables")}
    assert "backtest_runs" in tables
    assert "walk_forward_runs" in tables
    assert "walk_forward_folds" in tables
    assert "walk_forward_summary" in tables


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


# ── save_walk_forward ─────────────────────────────────────────────────────────


def test_save_walk_forward_returns_string_id(store: BacktestStore) -> None:
    folds = [_make_fold(i) for i in range(3)]
    wf = WalkForwardResult.from_fold_results("SIM", folds)
    wf_id = store.save_walk_forward(wf)
    assert isinstance(wf_id, str) and len(wf_id) > 0


def test_save_walk_forward_run_row_written(store: BacktestStore) -> None:
    folds = [_make_fold(i) for i in range(4)]
    wf = WalkForwardResult.from_fold_results("ES.c.0", folds)
    store.save_walk_forward(wf, run_id="wf-main", strategy_name="Trend")

    rows = store.query(
        "SELECT symbol, strategy_name, n_folds FROM walk_forward_runs WHERE id = ?",
        ["wf-main"],
    )
    assert len(rows) == 1
    assert rows[0][0] == "ES.c.0"
    assert rows[0][1] == "Trend"
    assert rows[0][2] == 4


def test_save_walk_forward_fold_count_matches(store: BacktestStore) -> None:
    n = 5
    folds = [_make_fold(i) for i in range(n)]
    wf = WalkForwardResult.from_fold_results("SIM", folds)
    store.save_walk_forward(wf, run_id="wf-folds")
    count = store.query(
        "SELECT COUNT(*) FROM walk_forward_folds WHERE walk_forward_run_id = ?",
        ["wf-folds"],
    )[0][0]
    assert count == n


def test_save_walk_forward_fold_metrics_roundtrip(store: BacktestStore) -> None:
    folds = [_make_fold(0, sharpe_ratio=3.14, win_rate_pct=66.0)]
    wf = WalkForwardResult.from_fold_results("SIM", folds)
    store.save_walk_forward(wf, run_id="wf-metric")
    rows = store.query(
        "SELECT sharpe_ratio, win_rate_pct FROM walk_forward_folds WHERE walk_forward_run_id = ?",
        ["wf-metric"],
    )
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(3.14)
    assert rows[0][1] == pytest.approx(66.0)


def test_save_walk_forward_fold_window_stored(store: BacktestStore) -> None:
    fold = _make_fold(0)
    wf = WalkForwardResult.from_fold_results("SIM", [fold])
    store.save_walk_forward(wf, run_id="wf-window")
    rows = store.query(
        "SELECT fold_index, oos_start, oos_end FROM walk_forward_folds "
        "WHERE walk_forward_run_id = ?",
        ["wf-window"],
    )
    assert rows[0][0] == 0
    assert rows[0][1] == fold.oos_start
    assert rows[0][2] == fold.oos_end


def test_save_walk_forward_summary_mean_roundtrip(store: BacktestStore) -> None:
    folds = [
        _make_fold(0, total_return_pct=5.0),
        _make_fold(1, total_return_pct=15.0),
    ]
    wf = WalkForwardResult.from_fold_results("SIM", folds)
    store.save_walk_forward(wf, run_id="wf-summary")
    rows = store.query(
        "SELECT total_return_pct_mean, total_return_pct_std "
        "FROM walk_forward_summary WHERE walk_forward_run_id = ?",
        ["wf-summary"],
    )
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(10.0)
    assert rows[0][1] == pytest.approx(5.0)


def test_save_walk_forward_summary_all_std_columns_present(store: BacktestStore) -> None:
    folds = [_make_fold(i) for i in range(2)]
    wf = WalkForwardResult.from_fold_results("SIM", folds)
    store.save_walk_forward(wf, run_id="wf-cols")
    cols = {
        row[0]
        for row in store.query(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'walk_forward_summary'"
        )
    }
    for metric in ("sharpe_ratio", "sortino_ratio", "sqn", "recovery_factor", "exposure_pct"):
        assert f"{metric}_mean" in cols
        assert f"{metric}_std" in cols


def test_save_walk_forward_summary_std_finite(store: BacktestStore) -> None:
    folds = [_make_fold(i) for i in range(3)]
    wf = WalkForwardResult.from_fold_results("SIM", folds)
    store.save_walk_forward(wf, run_id="wf-finite")
    rows = store.query(
        "SELECT * FROM walk_forward_summary WHERE walk_forward_run_id = ?",
        ["wf-finite"],
    )
    assert len(rows) == 1
    for val in rows[0][1:]:  # skip walk_forward_run_id
        assert val is not None and math.isfinite(float(val))


# ── Context manager ───────────────────────────────────────────────────────────


def test_store_context_manager() -> None:
    with BacktestStore(":memory:") as s:
        run_id = s.save_run(_make_result(), run_id="ctx-test")
        count = s.query("SELECT COUNT(*) FROM backtest_runs")[0][0]
    assert count == 1
    assert run_id == "ctx-test"
