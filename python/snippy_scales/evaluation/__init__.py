"""Evaluation framework — walk-forward backtesting with parameter search.

Provides out-of-sample validation, hyperparameter optimisation, a hybrid
persistence layer (SQLite metadata + DuckDB analytics), and tearsheet
generation for all strategies in the platform.

Typical usage::

    from snippy_scales.evaluation import (
        EvaluationRunner,
        ParameterGrid,
        RandomSearch,
        WalkForwardSplit,
        SQLiteStore,
        AnalyticsStore,
        TearsheetGenerator,
    )
    from snippy_scales.strategies.trend import TrendFollowing

    # Walk-forward with grid search — persist to both metadata and analytics stores
    runner = EvaluationRunner(
        n_splits=5,
        db_path="data/metadata.db",
        analytics_db_path="data/analytics.duckdb",
        tearsheet_dir="reports/",
    )
    grid = ParameterGrid({"fast_period": [10, 20, 40], "slow_period": [40, 60, 120]})
    result = runner.evaluate(TrendFollowing, bars, params=grid, symbol="ES.c.0")

    print(result.best_params)
    print(result.summary_df())
"""

from __future__ import annotations

from snippy_scales.evaluation.database import AnalyticsStore, SQLiteStore
from snippy_scales.evaluation.results import EvaluationResult, FoldResult, SweepResult
from snippy_scales.evaluation.runner import EvaluationRunner
from snippy_scales.evaluation.split import SplitFold, WalkForwardSplit
from snippy_scales.evaluation.sweep import OptunaSearch, ParameterGrid, RandomSearch
from snippy_scales.evaluation.tearsheet import TearsheetGenerator

__all__ = [
    # split
    "SplitFold",
    "WalkForwardSplit",
    # sweep
    "ParameterGrid",
    "RandomSearch",
    "OptunaSearch",
    # results
    "FoldResult",
    "SweepResult",
    "EvaluationResult",
    # database
    "SQLiteStore",
    "AnalyticsStore",
    # tearsheet
    "TearsheetGenerator",
    # runner
    "EvaluationRunner",
]
