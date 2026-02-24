"""Evaluation framework — walk-forward backtesting with parameter search.

Provides out-of-sample validation, hyperparameter optimisation, SQLite result
persistence, and tearsheet generation for all strategies in the platform.

Typical usage::

    from snippy_scales.evaluation import (
        EvaluationRunner,
        ParameterGrid,
        RandomSearch,
        WalkForwardSplit,
        SQLiteStore,
        TearsheetGenerator,
    )
    from snippy_scales.strategies.trend import TrendFollowing

    # Walk-forward with grid search
    runner = EvaluationRunner(
        n_splits=5,
        db_path="results.db",
        tearsheet_dir="reports/",
    )
    grid = ParameterGrid({"fast_period": [10, 20, 40], "slow_period": [40, 60, 120]})
    result = runner.evaluate(TrendFollowing, bars, params=grid, symbol="ES.c.0")

    print(result.best_params)
    print(result.summary_df())
"""

from __future__ import annotations

from snippy_scales.evaluation.database import SQLiteStore
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
    # tearsheet
    "TearsheetGenerator",
    # runner
    "EvaluationRunner",
]
