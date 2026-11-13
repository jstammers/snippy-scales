"""Tearsheet generation using quantstats.

Generates interactive HTML performance reports with benchmark comparison.

Two entry points are provided:

* :meth:`TearsheetGenerator.generate` — single backtest result.
* :meth:`TearsheetGenerator.generate_walk_forward` — walk-forward evaluation
  result; stitches test-fold equity curves end-to-end into an out-of-sample
  returns series using real fold dates where available.

Usage::

    gen = TearsheetGenerator(benchmark="SPY")
    gen.generate(result, title="TrendFollowing ES.c.0",
                 output_path="reports/tf.html")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from snippy_scales.backtesting.domain import BacktestResult
    from snippy_scales.evaluation.results import EvaluationResult


# ── TearsheetGenerator ────────────────────────────────────────────────────────


class TearsheetGenerator:
    """Generate HTML tearsheets using quantstats.

    Args:
        benchmark: Ticker string (e.g. ``"SPY"``) or a ``pd.Series`` of
            benchmark returns to overlay on the report.  Pass ``None`` to
            generate the report without a benchmark.  Defaults to ``"SPY"``.
            When a ticker is supplied, quantstats downloads price data via
            yfinance; if the download fails (no network, bad ticker) the
            report is generated without a benchmark rather than raising.
        rf: Risk-free rate used in Sharpe / Sortino calculations
            (annualised decimal, e.g. ``0.05`` = 5 %).  Default ``0.0``.
        periods_per_year: Annualisation factor.  Use ``252`` for daily bars
            (default) or ``12`` for monthly bars.

    Example::

        gen = TearsheetGenerator(benchmark="SPY")
        gen.generate(result, title="TrendFollowing ES.c.0",
                     output_path="reports/tf.html")
    """

    def __init__(
        self,
        benchmark: str | pd.Series | None = "SPY",
        rf: float = 0.0,
        periods_per_year: int = 252,
    ) -> None:
        self._benchmark = benchmark
        self._rf = rf
        self._periods_per_year = periods_per_year

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        result: BacktestResult,
        *,
        title: str = "Strategy Tearsheet",
        output_path: Path | str,
        start_date: str | None = None,
    ) -> Path:
        """Generate an HTML tearsheet for a single
        :class:`~snippy_scales.backtesting.domain.BacktestResult`.

        Args:
            result: Backtest result containing the equity curve.
            title: Report heading shown at the top of the HTML page.
            output_path: Destination file path.  The ``.html`` extension
                is applied automatically regardless of what is supplied.
            start_date: ISO date string (e.g. ``"2020-01-01"``) for the
                first bar.  When omitted, synthetic business-day dates
                ending at today are used.  Providing real dates improves
                benchmark alignment.

        Returns:
            :class:`Path` of the generated HTML file.
        """
        equity = np.asarray(result.equity_curve, dtype=np.float64)
        returns = _equity_to_returns(equity, start_date=start_date)

        out = Path(output_path).with_suffix(".html")
        out.parent.mkdir(parents=True, exist_ok=True)
        self._run_report(returns, title=title, output=out)
        return out

    def generate_walk_forward(
        self,
        eval_result: EvaluationResult,
        *,
        output_path: Path | str,
    ) -> Path:
        """Generate an HTML tearsheet for a walk-forward
        :class:`~snippy_scales.evaluation.results.EvaluationResult`.

        The test-fold equity curves of the best parameter set are stitched
        end-to-end.  Each fold's ``test_start`` date is used to build a real
        :class:`pandas.DatetimeIndex` so that benchmark comparison and the
        monthly returns heatmap align correctly.

        Args:
            eval_result: Completed evaluation result.
            output_path: Destination file path.  The ``.html`` extension
                is applied automatically.

        Returns:
            :class:`Path` of the generated HTML file.
        """
        returns = _oos_returns(eval_result.best_result)
        title = (
            f"{eval_result.experiment_name} — Walk-Forward OOS "
            f"({eval_result.strategy_class.split('.')[-1]})"
        )
        out = Path(output_path).with_suffix(".html")
        out.parent.mkdir(parents=True, exist_ok=True)
        self._run_report(returns, title=title, output=out)
        return out

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_report(
        self,
        returns: pd.Series,
        *,
        title: str,
        output: Path,
    ) -> None:
        """Call ``qs.reports.html``; fall back to no benchmark on download errors."""
        import quantstats as qs

        kwargs: dict[str, Any] = dict(
            rf=self._rf,
            title=title,
            output=str(output),
            periods_per_year=self._periods_per_year,
        )
        try:
            qs.reports.html(returns, benchmark=self._benchmark, **kwargs)
        except Exception:
            if self._benchmark is not None:
                # Benchmark download likely failed — retry without it.
                qs.reports.html(returns, benchmark=None, **kwargs)
            else:
                raise


# ── Module-level helpers ──────────────────────────────────────────────────────


def _equity_to_returns(
    equity: np.ndarray,
    start_date: str | None = None,
) -> pd.Series:
    """Convert an equity-curve array to a ``pd.Series`` with a DatetimeIndex.

    When *start_date* is ``None`` synthetic business-day dates ending today
    are used so that quantstats can still produce all its charts.
    """
    import pandas as pd

    if len(equity) < 2:  # noqa: PLR2004
        equity = np.array([1.0, 1.0])

    returns_arr = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    n = len(returns_arr)

    if start_date is not None:
        dates = pd.bdate_range(start=pd.Timestamp(start_date), periods=n)
    else:
        end = pd.Timestamp.today().normalize()
        dates = pd.bdate_range(end=end, periods=n)

    return pd.Series(returns_arr, index=dates, name="strategy", dtype="float64")


def _oos_returns(best: Any) -> pd.Series:
    """Stitch walk-forward test-fold equity curves into a single returns Series.

    Real fold ``test_start`` dates are used to build the DatetimeIndex.
    Consecutive folds are joined end-to-end with no overlap.
    """
    import pandas as pd

    folds = sorted(best.folds, key=lambda f: f.fold_idx)
    all_returns: list[float] = []
    all_dates: list[pd.Timestamp] = []
    last_date: pd.Timestamp | None = None

    for fold in folds:
        equity = np.asarray(fold.test_equity_curve, dtype=np.float64)
        if len(equity) < 2:  # noqa: PLR2004
            continue
        rets = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
        n = len(rets)

        if fold.test_start:
            start = pd.Timestamp(fold.test_start)
        elif last_date is not None:
            start = last_date + pd.offsets.BDay(1)
        else:
            start = pd.Timestamp("2020-01-01")

        dates = pd.bdate_range(start=start, periods=n)
        all_returns.extend(rets.tolist())
        all_dates.extend(dates.tolist())
        if len(dates):
            last_date = dates[-1]

    if not all_returns:
        end = pd.Timestamp.today().normalize()
        return pd.Series(
            [0.0],
            index=pd.bdate_range(end=end, periods=1),
            name="strategy",
            dtype="float64",
        )

    return pd.Series(
        all_returns,
        index=pd.DatetimeIndex(all_dates),
        name="strategy",
        dtype="float64",
    )
