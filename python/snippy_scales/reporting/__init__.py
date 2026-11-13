"""Strategy research reports: a tracked marimo template, untracked instances.

Each report is ``reports/<name>/`` holding ``config.yaml`` (what to test),
``report.py`` (a copy of ``reports/templates/strategy_report.py``),
``notes.md`` (the analyst's conclusions), the Optuna studies, and the exported
``report.html`` / ``tearsheet.html``. Only the template, example configs and
theory write-ups (``docs/strategies/``) are version-controlled.

Typical use::

    algo report new tsmom-sp500 --from reports/templates/configs/tsmom_sp500.yaml
    algo report build tsmom-sp500          # → reports/tsmom-sp500/report.html

The notebook's computations live in :mod:`snippy_scales.reporting.pipeline`
(screening on the vector engine, validation on Nautilus, the full backtest).
"""

from snippy_scales.reporting.config import (
    REPO_ROOT,
    DataConfig,
    EvaluationConfig,
    ReportConfig,
    ScreeningConfig,
    ValidationConfig,
    load_report_config,
)
from snippy_scales.reporting.pipeline import REPORTS_DIR, ReportPaths, StageResult
from snippy_scales.reporting.scaffold import TEMPLATE, export_html, list_reports, new_report

__all__ = [
    "REPORTS_DIR",
    "REPO_ROOT",
    "TEMPLATE",
    "DataConfig",
    "EvaluationConfig",
    "ReportConfig",
    "ReportPaths",
    "ScreeningConfig",
    "StageResult",
    "ValidationConfig",
    "export_html",
    "list_reports",
    "load_report_config",
    "new_report",
]
