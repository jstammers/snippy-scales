"""Evaluation — walk-forward studies on Optuna and Nautilus, plus statistics
that correct for selection bias and score probabilistic forecasts.

Typical usage::

    from snippy_scales.evaluation import (
        WalkForward, make_folds, run_study, summarize_study, write_tearsheet,
    )
    from snippy_scales.execution import Costs, load_market_data
    from snippy_scales.strategies import TrendFollowing

    data = load_market_data("ES.c.0")
    wf = WalkForward(data, make_folds(data.closes(), n_splits=5), costs=Costs(commission_rate=1e-4))
    study = run_study(
        TrendFollowing, wf,
        space={"fast_period": [10, 20, 40], "slow_period": [60, 120]}, sampler="grid",
    )
    report = summarize_study(study)
    print(report.best_params, report.sharpe, report.dsr, report.pbo)

``evaluation/`` never touches orders; it calls ``execution.run_backtest``.
"""

from __future__ import annotations

from snippy_scales.evaluation.scoring import (
    calibration_error,
    coverage,
    crps_ensemble,
    crps_gaussian,
    pit_values,
)
from snippy_scales.evaluation.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from snippy_scales.evaluation.walk_forward import (
    DEFAULT_STORAGE,
    Fold,
    StudyReport,
    WalkForward,
    load_strategy_class,
    make_folds,
    returns_statistics,
    run_study,
    stitch_returns,
    suggest,
    summarize_study,
    write_tearsheet,
)

__all__ = [
    # walk-forward
    "DEFAULT_STORAGE",
    "Fold",
    "StudyReport",
    "WalkForward",
    "load_strategy_class",
    "make_folds",
    "returns_statistics",
    "run_study",
    "stitch_returns",
    "suggest",
    "summarize_study",
    "write_tearsheet",
    # statistics — selection-bias corrections
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "min_track_record_length",
    # scoring — probabilistic forecast evaluation
    "crps_gaussian",
    "crps_ensemble",
    "pit_values",
    "calibration_error",
    "coverage",
]
