"""Tests for the evaluation framework.

Coverage:
* WalkForwardSplit — expanding and rolling windows, no data leakage
* ParameterGrid — exhaustive Cartesian product
* RandomSearch — reproducible sampling
* SweepResult — mean and population std-dev properties
* EvaluationRunner — single-asset and multi-asset end-to-end
* SQLiteStore — experiment metadata round-trip (metadata only)
* AnalyticsStore — DuckDB sweep + fold analytics round-trip
* TearsheetGenerator — file generation
* EvaluationResult — best_params selection, summary_df, oos_equity_curve
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from snippy_scales.evaluation import (
    AnalyticsStore,
    EvaluationResult,
    EvaluationRunner,
    FoldResult,
    ParameterGrid,
    RandomSearch,
    SplitFold,
    SQLiteStore,
    SweepResult,
    TearsheetGenerator,
    WalkForwardSplit,
)
from snippy_scales.strategies.momentum_cs import CrossSectionalMomentum
from snippy_scales.strategies.trend import TrendFollowing

# ── Fixtures ──────────────────────────────────────────────────────────────────

_BASE_NS = 1_577_836_800_000_000_000  # 2020-01-01 UTC in nanoseconds
_DAY_NS = 86_400_000_000_000


def _make_bars(n: int = 600, seed: int = 0, trend: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1000.0 * np.cumprod(1.0 + rng.normal(trend, 0.01, n))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = rng.uniform(1_000, 10_000, n)
    timestamps = _BASE_NS + np.arange(n, dtype=np.int64) * _DAY_NS
    return pl.DataFrame(
        {
            "ts": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_multi_bars(n: int = 600, symbols: list[str] | None = None) -> dict[str, pl.DataFrame]:
    symbols = symbols or ["A", "B", "C", "D"]
    return {sym: _make_bars(n, seed=i) for i, sym in enumerate(symbols)}


def _make_dummy_metrics() -> Any:
    from snippy_scales.backtesting.domain import BacktestMetrics

    return BacktestMetrics(
        total_return_pct=10.0,
        sharpe_ratio=1.0,
        sortino_ratio=1.2,
        calmar_ratio=0.8,
        omega_ratio=1.3,
        max_drawdown_pct=-8.0,
        max_drawdown_duration=0,
        total_trades=30,
        total_closed_trades=30,
        total_open_trades=0,
        winning_trades=17,
        losing_trades=13,
        win_rate_pct=55.0,
        profit_factor=1.4,
        expectancy=50.0,
        sqn=0.0,
        avg_trade_return_pct=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        best_trade_pct=0.0,
        worst_trade_pct=0.0,
        payoff_ratio=0.0,
        recovery_factor=0.0,
        avg_holding_period=0.0,
        avg_winning_duration=0.0,
        avg_losing_duration=0.0,
        max_consecutive_wins=0,
        max_consecutive_losses=0,
        start_value=100_000.0,
        end_value=110_000.0,
        total_fees_paid=0.0,
        open_trade_pnl=0.0,
        exposure_pct=0.0,
    )


# ── WalkForwardSplit ──────────────────────────────────────────────────────────


class TestWalkForwardSplit:
    def test_expanding_fold_count(self) -> None:
        bars = _make_bars(500)
        splitter = WalkForwardSplit(n_splits=5, test_size=0.15, window="expanding")
        folds = splitter.split(bars)
        assert len(folds) == 5

    def test_expanding_fold_indices_return_splitfold(self) -> None:
        bars = _make_bars(400)
        folds = WalkForwardSplit(n_splits=3).split(bars)
        for fold in folds:
            assert isinstance(fold, SplitFold)
            assert len(fold.train) > 0
            assert len(fold.test) > 0

    def test_no_data_leakage_expanding(self) -> None:
        """Test window ends are strictly before test window starts."""
        bars = _make_bars(500)
        folds = WalkForwardSplit(n_splits=4, window="expanding").split(bars)
        # Last row of train must be before first row of test
        for fold in folds:
            last_train_ts = fold.train["ts"][-1]
            first_test_ts = fold.test["ts"][0]
            assert last_train_ts < first_test_ts

    def test_rolling_window_fixed_train_size(self) -> None:
        n = 500
        n_splits = 4
        test_frac = 0.1
        bars = _make_bars(n)
        folds = WalkForwardSplit(
            n_splits=n_splits,
            test_size=test_frac,
            window="rolling",
        ).split(bars)
        # All training windows should have the same size
        if len(folds) >= 2:  # noqa: PLR2004
            sizes = {len(f.train) for f in folds}
            assert len(sizes) == 1, f"Rolling folds have different train sizes: {sizes}"

    def test_no_data_leakage_rolling(self) -> None:
        bars = _make_bars(400)
        folds = WalkForwardSplit(n_splits=3, window="rolling").split(bars)
        for fold in folds:
            last_train_ts = fold.train["ts"][-1]
            first_test_ts = fold.test["ts"][0]
            assert last_train_ts < first_test_ts

    def test_min_train_size_filters_folds(self) -> None:
        bars = _make_bars(600)
        # min_train_size larger than what the early folds would have
        folds_normal = WalkForwardSplit(n_splits=5).split(bars)
        folds_filtered = WalkForwardSplit(n_splits=5, min_train_size=10_000).split(bars)
        assert len(folds_filtered) < len(folds_normal)

    def test_split_multi_aligned_slices(self) -> None:
        multi = _make_multi_bars(400, symbols=["X", "Y"])
        splitter = WalkForwardSplit(n_splits=3)
        multi_folds = splitter.split_multi(multi)
        assert len(multi_folds) == 3
        for train_bars, test_bars in multi_folds:
            # All symbols must have the same slice length
            train_lens = {len(v) for v in train_bars.values()}
            test_lens = {len(v) for v in test_bars.values()}
            assert len(train_lens) == 1
            assert len(test_lens) == 1

    def test_date_strings_populated(self) -> None:
        folds = WalkForwardSplit(n_splits=2).split(_make_bars(300))
        for fold in folds:
            assert fold.train_start != ""
            assert fold.test_end != ""


# ── ParameterGrid ─────────────────────────────────────────────────────────────


class TestParameterGrid:
    def test_cartesian_product_length(self) -> None:
        grid = ParameterGrid({"a": [1, 2, 3], "b": [10, 20]})
        combos = list(grid)
        assert len(combos) == 6  # 3 * 2
        assert len(grid) == 6

    def test_all_combos_unique(self) -> None:
        grid = ParameterGrid({"x": [1, 2], "y": [3, 4, 5]})
        combos = list(grid)
        tuples = [tuple(sorted(d.items())) for d in combos]
        assert len(set(tuples)) == len(tuples)

    def test_single_param(self) -> None:
        grid = ParameterGrid({"p": [10, 20, 30]})
        assert list(grid) == [{"p": 10}, {"p": 20}, {"p": 30}]

    def test_empty_values_raises(self) -> None:
        with pytest.raises(ValueError):
            ParameterGrid({"a": []})

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError):
            ParameterGrid({})


# ── RandomSearch ──────────────────────────────────────────────────────────────


class TestRandomSearch:
    def test_length(self) -> None:
        rs = RandomSearch({"a": [1, 2, 3]}, n_iter=10)
        assert len(rs) == 10
        assert len(list(rs)) == 10

    def test_reproducible(self) -> None:
        params = {"a": [1, 2, 3], "b": [10, 20]}
        seq1 = list(RandomSearch(params, n_iter=5, seed=42))
        seq2 = list(RandomSearch(params, n_iter=5, seed=42))
        assert seq1 == seq2

    def test_different_seeds_differ(self) -> None:
        params = {"a": list(range(20))}
        seq1 = list(RandomSearch(params, n_iter=5, seed=0))
        seq2 = list(RandomSearch(params, n_iter=5, seed=1))
        # With 20 options the chance of identical draw is negligible
        assert seq1 != seq2

    def test_values_from_distribution(self) -> None:
        allowed = [100, 200, 300]
        for params in RandomSearch({"v": allowed}, n_iter=50, seed=0):
            assert params["v"] in allowed


# ── SweepResult std-dev properties ────────────────────────────────────────────


def _make_fold_with_sharpe(fold_idx: int, sharpe: float) -> FoldResult:
    from snippy_scales.backtesting.domain import BacktestMetrics

    base = _make_dummy_metrics()
    m = BacktestMetrics(
        total_return_pct=base.total_return_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=base.sortino_ratio,
        calmar_ratio=base.calmar_ratio,
        omega_ratio=base.omega_ratio,
        max_drawdown_pct=base.max_drawdown_pct,
        max_drawdown_duration=base.max_drawdown_duration,
        total_trades=base.total_trades,
        total_closed_trades=base.total_closed_trades,
        total_open_trades=base.total_open_trades,
        winning_trades=base.winning_trades,
        losing_trades=base.losing_trades,
        win_rate_pct=base.win_rate_pct,
        profit_factor=base.profit_factor,
        expectancy=base.expectancy,
        sqn=base.sqn,
        avg_trade_return_pct=base.avg_trade_return_pct,
        avg_win_pct=base.avg_win_pct,
        avg_loss_pct=base.avg_loss_pct,
        best_trade_pct=base.best_trade_pct,
        worst_trade_pct=base.worst_trade_pct,
        payoff_ratio=base.payoff_ratio,
        recovery_factor=base.recovery_factor,
        avg_holding_period=base.avg_holding_period,
        avg_winning_duration=base.avg_winning_duration,
        avg_losing_duration=base.avg_losing_duration,
        max_consecutive_wins=base.max_consecutive_wins,
        max_consecutive_losses=base.max_consecutive_losses,
        start_value=base.start_value,
        end_value=base.end_value,
        total_fees_paid=base.total_fees_paid,
        open_trade_pnl=base.open_trade_pnl,
        exposure_pct=base.exposure_pct,
    )
    return FoldResult(
        fold_idx=fold_idx,
        params={"x": 1},
        train_metrics=m,
        test_metrics=m,
        train_equity_curve=np.linspace(1.0, 1.05, 50),
        test_equity_curve=np.linspace(1.05, 1.08, 25),
        train_start="2020-01-01",
        train_end="2021-01-01",
        test_start="2021-01-01",
        test_end="2021-06-01",
    )


def test_sweep_result_std_test_sharpe_population_std() -> None:
    """Population std of [1.0, 3.0] = 1.0."""
    folds = [_make_fold_with_sharpe(0, 1.0), _make_fold_with_sharpe(1, 3.0)]
    sweep = SweepResult(params={}, folds=folds)
    assert sweep.mean_test_sharpe == pytest.approx(2.0)
    assert sweep.std_test_sharpe == pytest.approx(1.0)


def test_sweep_result_std_single_fold_is_zero() -> None:
    """Population std of a single value is 0.0, not NaN."""
    sweep = SweepResult(params={}, folds=[_make_fold_with_sharpe(0, 1.5)])
    assert sweep.std_test_sharpe == pytest.approx(0.0)


def test_sweep_result_std_empty_folds_is_nan() -> None:
    import math

    sweep = SweepResult(params={}, folds=[])
    assert math.isnan(sweep.std_test_sharpe)
    assert math.isnan(sweep.std_test_return)
    assert math.isnan(sweep.std_test_max_dd)
    assert math.isnan(sweep.std_test_sortino)


def test_sweep_result_to_dict_includes_std() -> None:
    folds = [_make_fold_with_sharpe(0, 1.0), _make_fold_with_sharpe(1, 3.0)]
    sweep = SweepResult(params={"a": 1}, folds=folds)
    d = sweep.to_dict()
    assert "std_test_sharpe" in d
    assert "std_test_return" in d
    assert "std_test_max_dd" in d
    assert "std_test_sortino" in d


# ── EvaluationResult ──────────────────────────────────────────────────────────


class TestEvaluationResult:
    def _make_sweep(self, sharpe: float, n_folds: int = 2) -> SweepResult:
        from snippy_scales.backtesting.domain import BacktestMetrics

        metrics = BacktestMetrics(
            total_return_pct=5.0,
            sharpe_ratio=sharpe,
            sortino_ratio=sharpe * 1.1,
            calmar_ratio=0.5,
            omega_ratio=1.2,
            max_drawdown_pct=-5.0,
            max_drawdown_duration=0,
            total_trades=20,
            total_closed_trades=20,
            total_open_trades=0,
            winning_trades=11,
            losing_trades=9,
            win_rate_pct=55.0,
            profit_factor=1.3,
            expectancy=30.0,
            sqn=0.0,
            avg_trade_return_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            payoff_ratio=0.0,
            recovery_factor=0.0,
            avg_holding_period=0.0,
            avg_winning_duration=0.0,
            avg_losing_duration=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            start_value=100_000.0,
            end_value=105_000.0,
            total_fees_paid=0.0,
            open_trade_pnl=0.0,
            exposure_pct=0.0,
        )
        folds = [
            FoldResult(
                fold_idx=i,
                params={"x": 1},
                train_metrics=metrics,
                test_metrics=metrics,
                train_equity_curve=np.linspace(1.0, 1.05, 50),
                test_equity_curve=np.linspace(1.05, 1.08, 25),
                train_start="2020-01-01",
                train_end="2021-01-01",
                test_start="2021-01-01",
                test_end="2021-06-01",
            )
            for i in range(n_folds)
        ]
        return SweepResult(params={"x": 1}, folds=folds)

    def test_best_params_selects_highest_sharpe(self) -> None:
        low = self._make_sweep(sharpe=0.5)
        low.params = {"fast_period": 10}
        high = self._make_sweep(sharpe=2.0)
        high.params = {"fast_period": 40}

        result = EvaluationResult(
            experiment_name="test",
            strategy_class="TrendFollowing",
            symbols=["ES"],
            sweep_results=[low, high],
        )
        assert result.best_params["fast_period"] == 40

    def test_summary_df_sorted(self) -> None:
        sweeps = [self._make_sweep(s) for s in [1.0, 2.5, 0.5]]
        for i, s in enumerate(sweeps):
            s.params = {"idx": i}
        result = EvaluationResult("x", "Strat", ["S"], sweeps)
        df = result.summary_df()
        # First row should have the highest Sharpe
        assert df["mean_test_sharpe"][0] >= df["mean_test_sharpe"][-1]

    def test_summary_df_includes_std_columns(self) -> None:
        """summary_df should contain std columns alongside mean columns."""
        result = EvaluationResult("x", "Strat", ["S"], [self._make_sweep(1.5)])
        df = result.summary_df()
        assert "std_test_sharpe" in df.columns
        assert "std_test_return_pct" in df.columns
        assert "std_test_max_dd_pct" in df.columns
        assert "std_test_sortino" in df.columns

    def test_oos_equity_curve_non_empty(self) -> None:
        result = EvaluationResult("x", "Strat", ["S"], [self._make_sweep(1.5)])
        oos = result.oos_equity_curve()
        assert len(oos) > 0

    def test_no_sweep_results_raises(self) -> None:
        result = EvaluationResult("x", "Strat", ["S"], [])
        with pytest.raises(ValueError):
            _ = result.best_params


# ── EvaluationRunner (end-to-end) ─────────────────────────────────────────────


class TestEvaluationRunner:
    def test_single_asset_default_params(self) -> None:
        bars = _make_bars(600, seed=1, trend=0.001)
        runner = EvaluationRunner(n_splits=3, initial_capital=100_000.0)
        result = runner.evaluate(TrendFollowing, bars, symbol="SIM")

        assert isinstance(result, EvaluationResult)
        assert len(result.sweep_results) == 1  # single default param set
        assert result.sweep_results[0].n_folds == 3

    def test_single_asset_grid_search(self) -> None:
        bars = _make_bars(600, seed=2)
        grid = ParameterGrid({"fast_period": [10, 20], "slow_period": [40, 60]})
        runner = EvaluationRunner(n_splits=2, initial_capital=100_000.0)
        result = runner.evaluate(TrendFollowing, bars, params=grid, symbol="SIM")

        assert len(result.sweep_results) == 4  # 2 * 2
        assert isinstance(result.best_params, dict)
        assert "fast_period" in result.best_params

    def test_single_asset_random_search(self) -> None:
        bars = _make_bars(600, seed=3)
        rs = RandomSearch({"fast_period": [5, 10, 20], "slow_period": [30, 60]}, n_iter=3)
        runner = EvaluationRunner(n_splits=2, initial_capital=100_000.0)
        result = runner.evaluate(TrendFollowing, bars, params=rs, symbol="SIM")
        assert len(result.sweep_results) == 3

    def test_multi_asset(self) -> None:
        multi = _make_multi_bars(600, symbols=["A", "B", "C", "D"])
        runner = EvaluationRunner(n_splits=2, initial_capital=100_000.0)
        result = runner.evaluate_multi_asset(
            CrossSectionalMomentum, multi, experiment_name="CSMOM_test"
        )
        assert isinstance(result, EvaluationResult)
        assert len(result.symbols) == 4

    def test_sqlite_metadata_persistence(self, tmp_path: Path) -> None:
        bars = _make_bars(500, seed=4)
        db = tmp_path / "test_eval.db"
        runner = EvaluationRunner(n_splits=2, db_path=db)
        runner.evaluate(TrendFollowing, bars, symbol="SIM", experiment_name="db_test")

        store = SQLiteStore(db)
        exps = store.load_experiments()
        assert len(exps) == 1
        assert exps["name"][0] == "db_test"

    def test_analytics_persistence(self, tmp_path: Path) -> None:
        bars = _make_bars(500, seed=4)
        analytics_db = tmp_path / "analytics.duckdb"
        runner = EvaluationRunner(n_splits=2, analytics_db_path=analytics_db)
        runner.evaluate(TrendFollowing, bars, symbol="SIM", experiment_name="analytics_test")

        store = AnalyticsStore(str(analytics_db))
        # experiment_id=0 when no SQLite db is provided
        df = store.load_sweep_results(0)
        assert len(df) >= 1
        store.close()

    def test_tearsheet_generation(self, tmp_path: Path) -> None:
        bars = _make_bars(500, seed=5)
        runner = EvaluationRunner(
            n_splits=2,
            tearsheet_dir=tmp_path / "sheets",
            benchmark=None,  # avoid network calls in tests
        )
        runner.evaluate(TrendFollowing, bars, symbol="SIM", experiment_name="sheet_test")

        htmls = list((tmp_path / "sheets").glob("*.html"))
        assert len(htmls) >= 1
        assert htmls[0].stat().st_size > 0

    def test_experiment_name_defaults(self) -> None:
        bars = _make_bars(400)
        runner = EvaluationRunner(n_splits=2)
        result = runner.evaluate(TrendFollowing, bars, symbol="X")
        assert "TrendFollowing" in result.experiment_name
        assert "X" in result.experiment_name


# ── SQLiteStore — metadata only ────────────────────────────────────────────────


class TestSQLiteStore:
    def _make_eval_result(self) -> EvaluationResult:
        metrics = _make_dummy_metrics()
        fold = FoldResult(
            fold_idx=0,
            params={"fast_period": 20, "slow_period": 60},
            train_metrics=metrics,
            test_metrics=metrics,
            train_equity_curve=np.linspace(1.0, 1.07, 100),
            test_equity_curve=np.linspace(1.07, 1.10, 50),
            train_start="2020-01-01",
            train_end="2021-01-01",
            test_start="2021-01-02",
            test_end="2021-06-30",
        )
        sweep = SweepResult(params={"fast_period": 20, "slow_period": 60}, folds=[fold])
        return EvaluationResult(
            experiment_name="store_test",
            strategy_class="snippy_scales.strategies.trend.TrendFollowing",
            symbols=["ES.c.0"],
            sweep_results=[sweep],
        )

    def test_roundtrip_experiment_metadata(self, tmp_path: Path) -> None:
        store = SQLiteStore(tmp_path / "rt.db")
        result = self._make_eval_result()
        exp_id = store.save_evaluation(result)
        assert exp_id > 0

        exps = store.load_experiments()
        assert len(exps) == 1
        assert exps["name"][0] == "store_test"
        assert exps["strategy_class"][0] == "snippy_scales.strategies.trend.TrendFollowing"

    def test_only_experiments_table_written(self, tmp_path: Path) -> None:
        """SQLiteStore is metadata-only — no sweep or fold rows are written."""
        store = SQLiteStore(tmp_path / "meta.db")
        result = self._make_eval_result()
        store.save_evaluation(result)

        # Only experiments table should exist and have data
        exps = store.load_experiments()
        assert len(exps) == 1

        # sweep_results and fold_results tables should not exist in SQLite
        tables = store.query("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = {row["name"] for row in tables.to_dicts()}
        assert "sweep_results" not in table_names
        assert "fold_results" not in table_names

    def test_multiple_experiments(self, tmp_path: Path) -> None:
        store = SQLiteStore(tmp_path / "multi.db")
        result = self._make_eval_result()
        id1 = store.save_evaluation(result)
        result2 = EvaluationResult(
            experiment_name="second_run",
            strategy_class=result.strategy_class,
            symbols=result.symbols,
            sweep_results=result.sweep_results,
        )
        id2 = store.save_evaluation(result2)
        assert id1 != id2

        exps = store.load_experiments()
        assert len(exps) == 2

    def test_arbitrary_query(self, tmp_path: Path) -> None:
        store = SQLiteStore(tmp_path / "q.db")
        result = self._make_eval_result()
        store.save_evaluation(result)

        df = store.query("SELECT name, strategy_class FROM experiments")
        assert len(df) == 1
        assert df["name"][0] == "store_test"

    def test_db_created_automatically(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "nested" / "eval.db"
        store = SQLiteStore(db_path)
        assert db_path.exists()
        # Should be usable
        exps = store.load_experiments()
        assert len(exps) == 0


# ── AnalyticsStore — DuckDB ────────────────────────────────────────────────────


class TestAnalyticsStore:
    def _make_eval_result(self) -> EvaluationResult:
        metrics = _make_dummy_metrics()
        fold = FoldResult(
            fold_idx=0,
            params={"fast_period": 20, "slow_period": 60},
            train_metrics=metrics,
            test_metrics=metrics,
            train_equity_curve=np.linspace(1.0, 1.07, 100),
            test_equity_curve=np.linspace(1.07, 1.10, 50),
            train_start="2020-01-01",
            train_end="2021-01-01",
            test_start="2021-01-02",
            test_end="2021-06-30",
        )
        sweep = SweepResult(params={"fast_period": 20, "slow_period": 60}, folds=[fold])
        return EvaluationResult(
            experiment_name="analytics_test",
            strategy_class="snippy_scales.strategies.trend.TrendFollowing",
            symbols=["ES.c.0"],
            sweep_results=[sweep],
        )

    def test_schema_creates_tables(self) -> None:
        store = AnalyticsStore(":memory:")
        tables = {row[0] for row in store.query("SELECT table_name FROM information_schema.tables")}
        assert "sweep_results" in tables
        assert "fold_results" in tables
        store.close()

    def test_save_evaluation_analytics_sweep_rows(self) -> None:
        store = AnalyticsStore(":memory:")
        result = self._make_eval_result()
        store.save_evaluation_analytics(result, experiment_id=42)

        rows = store.query(
            "SELECT experiment_id, n_folds FROM sweep_results WHERE experiment_id = ?", [42]
        )
        assert len(rows) == 1
        assert rows[0][0] == 42
        assert rows[0][1] == 1  # one fold
        store.close()

    def test_save_evaluation_analytics_fold_rows(self) -> None:
        store = AnalyticsStore(":memory:")
        result = self._make_eval_result()
        store.save_evaluation_analytics(result, experiment_id=1)

        sweep_rows = store.query("SELECT id FROM sweep_results")
        assert len(sweep_rows) == 1
        sweep_id = sweep_rows[0][0]

        fold_rows = store.query("SELECT fold_idx FROM fold_results WHERE sweep_id = ?", [sweep_id])
        assert len(fold_rows) == 1
        assert fold_rows[0][0] == 0
        store.close()

    def test_fold_metrics_roundtrip(self) -> None:
        """All test metrics are stored and retrieved correctly."""
        store = AnalyticsStore(":memory:")
        result = self._make_eval_result()
        store.save_evaluation_analytics(result, experiment_id=1)

        rows = store.query("SELECT sharpe_ratio, total_return_pct FROM fold_results")
        assert len(rows) == 1
        assert rows[0][0] == pytest.approx(1.0)  # sharpe_ratio default
        assert rows[0][1] == pytest.approx(10.0)  # total_return_pct default
        store.close()

    def test_mean_std_stored_in_sweep(self) -> None:
        """mean_test_sharpe and std_test_sharpe are persisted in sweep_results."""
        store = AnalyticsStore(":memory:")
        result = self._make_eval_result()
        store.save_evaluation_analytics(result, experiment_id=1)

        rows = store.query("SELECT mean_test_sharpe, std_test_sharpe FROM sweep_results")
        assert len(rows) == 1
        assert rows[0][0] == pytest.approx(1.0)  # mean of single fold's sharpe
        assert rows[0][1] == pytest.approx(0.0)  # std of single fold is 0
        store.close()

    def test_load_sweep_results_returns_dataframe(self) -> None:
        store = AnalyticsStore(":memory:")
        result = self._make_eval_result()
        store.save_evaluation_analytics(result, experiment_id=7)
        df = store.load_sweep_results(7)
        assert len(df) == 1
        assert "mean_test_sharpe" in df.columns
        assert "std_test_sharpe" in df.columns
        store.close()

    def test_context_manager(self) -> None:
        with AnalyticsStore(":memory:") as store:
            result = self._make_eval_result()
            store.save_evaluation_analytics(result, experiment_id=1)
            rows = store.query("SELECT COUNT(*) FROM sweep_results")
        assert rows[0][0] == 1


# ── TearsheetGenerator ────────────────────────────────────────────────────────


class TestTearsheetGenerator:
    def _make_backtest_result(self) -> Any:
        """Create a minimal BacktestResult-like object."""
        from snippy_scales.backtesting.domain import BacktestMetrics, BacktestResult

        metrics = BacktestMetrics(
            total_return_pct=15.0,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=1.0,
            omega_ratio=1.5,
            max_drawdown_pct=-10.0,
            max_drawdown_duration=0,
            total_trades=50,
            total_closed_trades=50,
            total_open_trades=0,
            winning_trades=30,
            losing_trades=20,
            win_rate_pct=60.0,
            profit_factor=1.6,
            expectancy=100.0,
            sqn=0.0,
            avg_trade_return_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            payoff_ratio=0.0,
            recovery_factor=0.0,
            avg_holding_period=0.0,
            avg_winning_duration=0.0,
            avg_losing_duration=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            start_value=100_000.0,
            end_value=115_000.0,
            total_fees_paid=0.0,
            open_trade_pnl=0.0,
            exposure_pct=0.0,
        )
        equity = np.cumprod(1.0 + np.random.default_rng(0).normal(0.001, 0.01, 200))
        equity = equity / equity[0] * 100_000.0
        drawdown = np.zeros_like(equity)
        returns = np.diff(equity) / equity[:-1]
        return BacktestResult(
            symbol="SIM",
            metrics=metrics,
            equity_curve=equity,
            drawdown_curve=drawdown,
            returns=returns,
        )

    def test_generate_html(self, tmp_path: Path) -> None:
        gen = TearsheetGenerator(benchmark=None)  # no network in tests
        result = self._make_backtest_result()
        path = gen.generate(
            result, title="Test", output_path=tmp_path / "ts", start_date="2020-01-01"
        )
        assert path.suffix == ".html"
        assert path.stat().st_size > 0

    def test_generate_walk_forward(self, tmp_path: Path) -> None:
        bars = _make_bars(400)
        runner = EvaluationRunner(n_splits=2, initial_capital=100_000.0)
        eval_result = runner.evaluate(TrendFollowing, bars, symbol="SIM")

        gen = TearsheetGenerator(benchmark=None)  # no network in tests
        path = gen.generate_walk_forward(eval_result, output_path=tmp_path / "wf_ts")
        assert path.suffix == ".html"
        assert path.stat().st_size > 0
