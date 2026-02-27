"""Evaluation framework — walk-forward backtesting with parameter search.

Provides out-of-sample validation, hyperparameter optimisation, a single
DuckDB persistence layer, and tearsheet generation for all strategies.

Typical usage::

    from snippy_scales.evaluation import (
        EvaluationRunner,
        ParameterGrid,
        RandomSearch,
        WalkForwardSplit,
        AnalyticsStore,
        TearsheetGenerator,
    )
    from snippy_scales.strategies.trend import TrendFollowing

    runner = EvaluationRunner(
        n_splits=5,
        db_path="data/analytics.duckdb",
        tearsheet_dir="reports/",
    )
    grid = ParameterGrid({"fast_period": [10, 20, 40], "slow_period": [40, 60, 120]})
    result = runner.evaluate(TrendFollowing, bars, params=grid, symbol="ES.c.0")

    print(result.best_params)
    print(result.summary_df())
"""

from __future__ import annotations

from snippy_scales.evaluation.database import AnalyticsStore
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
    "AnalyticsStore",
    # tearsheet
    "TearsheetGenerator",
    # runner
    "EvaluationRunner",
]
