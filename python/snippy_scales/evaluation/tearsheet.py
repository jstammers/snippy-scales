"""Tearsheet generation using matplotlib.

Generates multi-panel performance reports in PNG, PDF, or both formats.

Two entry points are provided:

* :meth:`TearsheetGenerator.generate` — single backtest result.
* :meth:`TearsheetGenerator.generate_walk_forward` — walk-forward evaluation
  result; stitches test-fold equity curves end-to-end to form an OOS equity
  curve and overlays train/test shading on the equity chart.

Layout (16 × 12 inches, 4 rows):

    ┌──────────────────────────────────────────────┐
    │           Title + summary header             │
    │              Equity curve                    │
    ├────────────────────┬─────────────────────────┤
    │  Underwater DD     │  Rolling 63-bar Sharpe  │
    ├────────────────────┼─────────────────────────┤
    │  Monthly returns   │  Metrics table          │
    │     heatmap        │                         │
    └────────────────────┴─────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from snippy_scales.backtesting.domain import BacktestResult
    from snippy_scales.evaluation.results import EvaluationResult


# ── TearsheetGenerator ────────────────────────────────────────────────────────


class TearsheetGenerator:
    """Generate multi-panel performance tearsheets from backtest results.

    Args:
        figsize: Matplotlib figure size in inches (width, height).
        dpi: Output resolution (default 150).
        style: Matplotlib style string (default ``"seaborn-v0_8-darkgrid"``).

    Example::

        gen = TearsheetGenerator()
        gen.generate(result, title="TrendFollowing ES.c.0", output_path="reports/tf.png")
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (16, 12),
        dpi: int = 150,
        style: str = "seaborn-v0_8-darkgrid",
    ) -> None:
        self._figsize = figsize
        self._dpi = dpi
        self._style = style

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        result: BacktestResult,
        *,
        title: str = "Strategy Tearsheet",
        output_path: Path | str,
        fmt: str = "both",
    ) -> list[Path]:
        """Generate a tearsheet for a single
        :class:`~snippy_scales.backtesting.domain.BacktestResult`.

        Args:
            result: Backtest result containing equity curve and metrics.
            title: Report title shown at the top of the figure.
            output_path: Destination file path.  The extension is replaced
                by the format(s) requested.
            fmt: ``"png"``, ``"pdf"``, or ``"both"`` (default).

        Returns:
            List of :class:`Path` objects for the generated files.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        equity = np.asarray(result.equity_curve, dtype=np.float64)
        returns = (
            np.diff(equity) / np.maximum(equity[:-1], 1e-10) if len(equity) > 1 else np.array([])
        )

        fig = self._build_figure(
            title=title,
            equity=equity,
            returns=returns,
            metrics=result.metrics,
        )
        paths = self._save(fig, Path(output_path), fmt)
        plt.close(fig)
        return paths

    def generate_walk_forward(
        self,
        eval_result: EvaluationResult,
        *,
        output_path: Path | str,
        fmt: str = "both",
    ) -> list[Path]:
        """Generate a tearsheet for a walk-forward
        :class:`~snippy_scales.evaluation.results.EvaluationResult`.

        The test-fold equity curves of the best parameter set are stitched
        end-to-end to form an out-of-sample equity curve.  Train and test
        windows are shaded on the equity chart.

        Args:
            eval_result: Completed evaluation result.
            output_path: Destination file path.
            fmt: ``"png"``, ``"pdf"``, or ``"both"`` (default).

        Returns:
            List of generated file paths.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        oos_equity = eval_result.oos_equity_curve()
        returns = (
            np.diff(oos_equity) / np.maximum(oos_equity[:-1], 1e-10)
            if len(oos_equity) > 1
            else np.array([])
        )

        best = eval_result.best_result
        title = (
            f"{eval_result.experiment_name} — Walk-Forward OOS "
            f"({eval_result.strategy_class.split('.')[-1]})"
        )
        # Aggregate test metrics from all folds of the best param set
        agg_metrics = _aggregate_metrics(best)

        fig = self._build_figure(
            title=title,
            equity=oos_equity,
            returns=returns,
            metrics=agg_metrics,
            fold_boundaries=self._fold_boundaries(best, len(oos_equity)),
        )
        paths = self._save(fig, Path(output_path), fmt)
        plt.close(fig)
        return paths

    # ── Figure construction ───────────────────────────────────────────────────

    def _build_figure(
        self,
        *,
        title: str,
        equity: np.ndarray,
        returns: np.ndarray,
        metrics: Any,
        fold_boundaries: list[tuple[float, float]] | None = None,
    ) -> Any:
        """Compose the 4-panel matplotlib figure."""
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        try:
            plt.style.use(self._style)
        except OSError:
            plt.style.use("default")

        fig = plt.figure(figsize=self._figsize)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(
            3,
            2,
            figure=fig,
            top=0.93,
            bottom=0.06,
            hspace=0.45,
            wspace=0.35,
        )

        ax_equity = fig.add_subplot(gs[0, :])
        ax_dd = fig.add_subplot(gs[1, 0])
        ax_sharpe = fig.add_subplot(gs[1, 1])
        ax_monthly = fig.add_subplot(gs[2, 0])
        ax_table = fig.add_subplot(gs[2, 1])

        self._plot_equity(ax_equity, equity, fold_boundaries)
        self._plot_drawdown(ax_dd, equity)
        self._plot_rolling_sharpe(ax_sharpe, returns)
        self._plot_monthly_heatmap(ax_monthly, returns)
        self._plot_metrics_table(ax_table, metrics)

        return fig

    # ── Individual panels ─────────────────────────────────────────────────────

    @staticmethod
    def _plot_equity(
        ax: Any,
        equity: np.ndarray,
        fold_boundaries: list[tuple[float, float]] | None = None,
    ) -> None:
        """Equity curve with optional train/test shading."""
        x = np.arange(len(equity))
        ax.plot(x, equity, linewidth=1.4, color="#2196F3", label="Equity")
        ax.set_ylabel("Portfolio Value", fontsize=9)
        ax.set_title("Equity Curve", fontsize=10, fontweight="semibold")
        ax.yaxis.set_major_formatter(
            __import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(
                lambda v, _: f"{v:,.0f}"
            )
        )

        if fold_boundaries:
            colors = ["#E3F2FD", "#BBDEFB"]
            for i, (start, end) in enumerate(fold_boundaries):
                ax.axvspan(
                    start, end, alpha=0.25, color=colors[i % 2], label="" if i else "Test fold"
                )
            ax.legend(fontsize=8, loc="upper left")

    @staticmethod
    def _plot_drawdown(ax: Any, equity: np.ndarray) -> None:
        """Underwater drawdown chart."""
        if len(equity) < 2:  # noqa: PLR2004
            ax.set_title("Drawdown", fontsize=10)
            return
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.maximum(peak, 1e-10) * 100.0
        x = np.arange(len(dd))
        ax.fill_between(x, dd, 0, color="#F44336", alpha=0.6)
        ax.plot(x, dd, color="#D32F2F", linewidth=0.8)
        ax.set_ylabel("Drawdown (%)", fontsize=9)
        ax.set_title("Underwater Drawdown", fontsize=10, fontweight="semibold")
        ax.set_ylim(top=0)

    @staticmethod
    def _plot_rolling_sharpe(ax: Any, returns: np.ndarray, window: int = 63) -> None:
        """Rolling Sharpe ratio (annualised)."""
        ax.set_title(f"Rolling {window}-bar Sharpe", fontsize=10, fontweight="semibold")
        ax.set_ylabel("Sharpe Ratio", fontsize=9)
        if len(returns) < window:
            return
        sharpes = []
        for i in range(window, len(returns) + 1):
            w = returns[i - window : i]
            std = np.std(w)
            sharpes.append(np.mean(w) / std * (252**0.5) if std > 1e-10 else 0.0)
        x = np.arange(window, len(returns) + 1)
        ax.plot(x, sharpes, linewidth=1.0, color="#4CAF50")
        ax.axhline(0, color="#9E9E9E", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="#8BC34A", linewidth=0.6, linestyle=":")

    @staticmethod
    def _plot_monthly_heatmap(ax: Any, returns: np.ndarray) -> None:
        """Calendar monthly returns heatmap."""
        ax.set_title("Monthly Returns (%)", fontsize=10, fontweight="semibold")

        if len(returns) < 20:  # noqa: PLR2004
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.axis("off")
            return

        # Aggregate to approximate months (21 bars/month)
        bars_per_month = 21
        n_months = len(returns) // bars_per_month
        monthly_rets = []
        for i in range(n_months):
            chunk = returns[i * bars_per_month : (i + 1) * bars_per_month]
            monthly_rets.append(float(np.prod(1.0 + chunk) - 1.0) * 100.0)

        # Arrange into a grid (up to 12 columns = months per year)
        cols = 12
        rows_needed = max(1, (n_months + cols - 1) // cols)
        # Pad to full grid
        padded = monthly_rets + [float("nan")] * (rows_needed * cols - len(monthly_rets))
        grid = np.array(padded, dtype=np.float64).reshape(rows_needed, cols)

        vmax = max(3.0, float(np.nanmax(np.abs(grid))))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(cols))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            fontsize=7,
        )
        ax.set_yticks(range(rows_needed))
        ax.set_yticklabels([f"Y{i + 1}" for i in range(rows_needed)], fontsize=7)

        # Annotate cells
        for r in range(rows_needed):
            for c in range(cols):
                val = grid[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=6, color="black")

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        ax.get_figure().colorbar(im, cax=cax)

    @staticmethod
    def _plot_metrics_table(ax: Any, metrics: Any) -> None:
        """Metrics summary table."""
        ax.set_title("Performance Summary", fontsize=10, fontweight="semibold")
        ax.axis("off")

        def _fmt(val: float | None, pct: bool = False, dp: int = 2) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            if pct:
                return f"{val:.{dp}f}%"
            return f"{val:.{dp}f}"

        rows = [
            ("Total Return", _fmt(getattr(metrics, "total_return_pct", None), pct=True)),
            ("Sharpe Ratio", _fmt(getattr(metrics, "sharpe_ratio", None))),
            ("Sortino Ratio", _fmt(getattr(metrics, "sortino_ratio", None))),
            ("Calmar Ratio", _fmt(getattr(metrics, "calmar_ratio", None))),
            ("Max Drawdown", _fmt(getattr(metrics, "max_drawdown_pct", None), pct=True)),
            ("Win Rate", _fmt(getattr(metrics, "win_rate_pct", None), pct=True)),
            ("Profit Factor", _fmt(getattr(metrics, "profit_factor", None))),
            ("Total Trades", str(int(getattr(metrics, "total_trades", 0) or 0))),
            ("Expectancy", _fmt(getattr(metrics, "expectancy", None))),
        ]

        table = ax.table(
            cellText=[[r[1]] for r in rows],
            rowLabels=[r[0] for r in rows],
            colLabels=["Value"],
            loc="center",
            cellLoc="right",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.5)

        # Colour-code Sharpe and Return rows
        for (row_idx, _col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#37474F")
                cell.set_text_props(color="white", fontweight="bold")
            elif row_idx % 2 == 0:
                cell.set_facecolor("#ECEFF1")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fold_boundaries(sweep_result: Any, total_bars: int) -> list[tuple[float, float]]:
        """Compute approximate bar-index boundaries for test folds."""
        folds = sorted(sweep_result.folds, key=lambda f: f.fold_idx)
        if not folds:
            return []
        # Each fold's equity curve contributes len(curve) bars
        boundaries = []
        cursor = 0
        for fold in folds:
            n = len(fold.test_equity_curve)
            boundaries.append((float(cursor), float(cursor + n - 1)))
            cursor += n
        return boundaries

    @staticmethod
    def _save(fig: Any, path: Path, fmt: str) -> list[Path]:
        """Save the figure in the requested format(s)."""
        stem = path.with_suffix("")
        paths: list[Path] = []
        formats = ["png", "pdf"] if fmt == "both" else [fmt]
        for f in formats:
            out = stem.with_suffix(f".{f}")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, format=f, dpi=fig.get_dpi(), bbox_inches="tight")
            paths.append(out)
        return paths


# ── Helpers ───────────────────────────────────────────────────────────────────


class _AggregatedMetrics:
    """Simple namespace for aggregated walk-forward metrics."""

    def __init__(self, **kwargs: float | int) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _aggregate_metrics(sweep_result: Any) -> _AggregatedMetrics:
    """Build aggregate metrics from a SweepResult's test folds."""
    folds = sweep_result.folds
    if not folds:
        return _AggregatedMetrics(
            total_return_pct=float("nan"),
            sharpe_ratio=float("nan"),
            sortino_ratio=float("nan"),
            calmar_ratio=float("nan"),
            max_drawdown_pct=float("nan"),
            win_rate_pct=float("nan"),
            profit_factor=float("nan"),
            total_trades=0,
            expectancy=float("nan"),
        )

    def _mean(attr: str) -> float:
        vals = [getattr(f.test_metrics, attr, float("nan")) for f in folds]
        clean = [v for v in vals if not np.isnan(v)]
        return float(np.mean(clean)) if clean else float("nan")

    return _AggregatedMetrics(
        total_return_pct=_mean("total_return_pct"),
        sharpe_ratio=_mean("sharpe_ratio"),
        sortino_ratio=_mean("sortino_ratio"),
        calmar_ratio=_mean("calmar_ratio"),
        max_drawdown_pct=_mean("max_drawdown_pct"),
        win_rate_pct=_mean("win_rate_pct"),
        profit_factor=_mean("profit_factor"),
        total_trades=int(sum(f.test_metrics.total_trades for f in folds)),
        expectancy=_mean("expectancy"),
    )
