"""Tests for the evaluation framework.

Coverage:
* WalkForwardSplit — expanding and rolling windows, no data leakage
* ParameterGrid — exhaustive Cartesian product
* RandomSearch — reproducible sampling
* EvaluationRunner — single-asset and multi-asset end-to-end
* SQLiteStore — round-trip persistence
* TearsheetGenerator — file generation
* EvaluationResult — best_params selection, summary_df, oos_equity_curve
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from snippy_scales.evaluation import (
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
        max_drawdown_pct=-8.0,
        win_rate_pct=55.0,
        profit_factor=1.4,
        total_trades=30,
        expectancy=50.0,
    )


from typing import Any  # noqa: E402

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


# ── EvaluationResult ──────────────────────────────────────────────────────────


class TestEvaluationResult:
    def _make_sweep(self, sharpe: float, n_folds: int = 2) -> SweepResult:
        from snippy_scales.backtesting.domain import BacktestMetrics

        metrics = BacktestMetrics(
            total_return_pct=5.0,
            sharpe_ratio=sharpe,
            sortino_ratio=sharpe * 1.1,
            calmar_ratio=0.5,
            max_drawdown_pct=-5.0,
            win_rate_pct=55.0,
            profit_factor=1.3,
            total_trades=20,
            expectancy=30.0,
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

    def test_db_persistence(self, tmp_path: Path) -> None:
        bars = _make_bars(500, seed=4)
        db = tmp_path / "test_eval.db"
        runner = EvaluationRunner(n_splits=2, db_path=db)
        runner.evaluate(TrendFollowing, bars, symbol="SIM", experiment_name="db_test")

        store = SQLiteStore(db)
        exps = store.load_experiments()
        assert len(exps) == 1
        assert exps["name"][0] == "db_test"

        exp_id = int(exps["id"][0])
        sweeps = store.load_sweep_results(exp_id)
        assert len(sweeps) >= 1

    def test_tearsheet_generation(self, tmp_path: Path) -> None:
        bars = _make_bars(500, seed=5)
        runner = EvaluationRunner(
            n_splits=2,
            tearsheet_dir=tmp_path / "sheets",
            tearsheet_fmt="png",
        )
        runner.evaluate(TrendFollowing, bars, symbol="SIM", experiment_name="sheet_test")

        pngs = list((tmp_path / "sheets").glob("*.png"))
        assert len(pngs) >= 1
        # File should be non-empty
        assert pngs[0].stat().st_size > 0

    def test_experiment_name_defaults(self) -> None:
        bars = _make_bars(400)
        runner = EvaluationRunner(n_splits=2)
        result = runner.evaluate(TrendFollowing, bars, symbol="X")
        assert "TrendFollowing" in result.experiment_name
        assert "X" in result.experiment_name


# ── SQLiteStore ───────────────────────────────────────────────────────────────


class TestSQLiteStore:
    def _make_eval_result(self) -> EvaluationResult:
        from snippy_scales.backtesting.domain import BacktestMetrics

        metrics = BacktestMetrics(
            total_return_pct=7.5,
            sharpe_ratio=1.3,
            sortino_ratio=1.5,
            calmar_ratio=0.9,
            max_drawdown_pct=-6.0,
            win_rate_pct=58.0,
            profit_factor=1.5,
            total_trades=40,
            expectancy=60.0,
        )
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

    def test_roundtrip(self, tmp_path: Path) -> None:
        store = SQLiteStore(tmp_path / "rt.db")
        result = self._make_eval_result()
        exp_id = store.save_evaluation(result)
        assert exp_id > 0

        exps = store.load_experiments()
        assert len(exps) == 1
        assert exps["name"][0] == "store_test"

    def test_sweep_results_persisted(self, tmp_path: Path) -> None:
        store = SQLiteStore(tmp_path / "sr.db")
        result = self._make_eval_result()
        exp_id = store.save_evaluation(result)

        df = store.load_sweep_results(exp_id)
        assert len(df) == 1
        assert abs(float(df["mean_test_sharpe"][0]) - 1.3) < 0.01

    def test_fold_results_persisted(self, tmp_path: Path) -> None:
        store = SQLiteStore(tmp_path / "fr.db")
        result = self._make_eval_result()
        exp_id = store.save_evaluation(result)

        sweep_df = store.load_sweep_results(exp_id)
        sweep_id = int(sweep_df["id"][0])
        folds_df = store.load_fold_results(sweep_id)
        assert len(folds_df) == 1
        assert folds_df["fold_idx"][0] == 0

    def test_arbitrary_query(self, tmp_path: Path) -> None:
        store = SQLiteStore(tmp_path / "q.db")
        result = self._make_eval_result()
        store.save_evaluation(result)

        df = store.query(
            "SELECT e.name, s.mean_test_sharpe "
            "FROM experiments e JOIN sweep_results s ON s.experiment_id = e.id"
        )
        assert len(df) == 1
        assert df["name"][0] == "store_test"

    def test_db_created_automatically(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "nested" / "eval.db"
        store = SQLiteStore(db_path)
        assert db_path.exists()
        # Should be usable
        exps = store.load_experiments()
        assert len(exps) == 0


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
            max_drawdown_pct=-10.0,
            win_rate_pct=60.0,
            profit_factor=1.6,
            total_trades=50,
            expectancy=100.0,
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

    def test_generate_png(self, tmp_path: Path) -> None:
        gen = TearsheetGenerator()
        result = self._make_backtest_result()
        paths = gen.generate(result, title="Test", output_path=tmp_path / "ts.png", fmt="png")
        assert len(paths) == 1
        assert paths[0].suffix == ".png"
        assert paths[0].stat().st_size > 0

    def test_generate_both(self, tmp_path: Path) -> None:
        gen = TearsheetGenerator()
        result = self._make_backtest_result()
        paths = gen.generate(result, title="Test", output_path=tmp_path / "ts", fmt="both")
        assert len(paths) == 2
        suffixes = {p.suffix for p in paths}
        assert ".png" in suffixes
        assert ".pdf" in suffixes

    def test_generate_walk_forward(self, tmp_path: Path) -> None:
        bars = _make_bars(400)
        runner = EvaluationRunner(n_splits=2, initial_capital=100_000.0)
        eval_result = runner.evaluate(TrendFollowing, bars, symbol="SIM")

        gen = TearsheetGenerator()
        paths = gen.generate_walk_forward(eval_result, output_path=tmp_path / "wf_ts", fmt="png")
        assert len(paths) >= 1
        assert paths[0].stat().st_size > 0
