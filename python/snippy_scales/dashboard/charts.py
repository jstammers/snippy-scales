"""Plotly figure factory for the trading dashboard.

All chart functions return a ``plotly.graph_objects.Figure`` with consistent
styling.  The dashboard pages call these rather than building charts inline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_EQUITY_COLOUR = "#2196F3"
_DRAWDOWN_COLOUR = "#F44336"
_TRAIN_COLOUR = "#4CAF50"
_TEST_COLOUR = "#FF9800"
_HIST_COLOUR = "#7E57C2"

_LAYOUT_DEFAULTS: dict[str, Any] = dict(
    template="plotly_dark",
    margin=dict(l=40, r=20, t=40, b=30),
    height=400,
)


# ---------------------------------------------------------------------------
# Equity & drawdown
# ---------------------------------------------------------------------------


def equity_curve_fig(
    equity: np.ndarray,
    *,
    title: str = "Equity Curve",
) -> go.Figure:
    """Single-line equity curve."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=equity,
            mode="lines",
            name="Equity",
            line=dict(color=_EQUITY_COLOUR, width=1.5),
        )
    )
    fig.update_layout(title=title, yaxis_title="Portfolio Value", **_LAYOUT_DEFAULTS)
    return fig


def equity_with_drawdown_fig(
    equity: np.ndarray,
    drawdown: np.ndarray,
    *,
    title: str = "Equity & Drawdown",
) -> go.Figure:
    """Two-panel chart: equity on top, drawdown below."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )
    fig.add_trace(
        go.Scatter(
            y=equity, mode="lines", name="Equity", line=dict(color=_EQUITY_COLOUR, width=1.5)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color=_DRAWDOWN_COLOUR, width=1),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(title=title, **_LAYOUT_DEFAULTS, height=500)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Sweep analysis
# ---------------------------------------------------------------------------


def sweep_heatmap_fig(
    sweep_df: Any,
    param_x: str,
    param_y: str,
    metric: str = "mean_test_sharpe",
) -> go.Figure:
    """2D heatmap of two parameters against a metric value."""
    pivot = sweep_df.pivot_table(index=param_y, columns=param_x, values=metric)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            colorscale="RdYlGn",
            colorbar_title=metric,
        )
    )
    fig.update_layout(
        title=f"{metric} by {param_x} × {param_y}",
        xaxis_title=param_x,
        yaxis_title=param_y,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Fold comparison
# ---------------------------------------------------------------------------


def fold_comparison_fig(
    fold_indices: list[int],
    train_values: list[float],
    test_values: list[float],
    *,
    metric_name: str = "Sharpe Ratio",
) -> go.Figure:
    """Grouped bar chart of train vs test metric per fold."""
    fig = go.Figure()
    x_labels = [f"Fold {i}" for i in fold_indices]
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=train_values,
            name="Train",
            marker_color=_TRAIN_COLOUR,
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=test_values,
            name="Test",
            marker_color=_TEST_COLOUR,
        )
    )
    fig.update_layout(
        title=f"Train vs Test — {metric_name}",
        barmode="group",
        yaxis_title=metric_name,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Trade analysis
# ---------------------------------------------------------------------------


def pnl_histogram_fig(
    pnl_values: np.ndarray,
    *,
    title: str = "P&L Distribution",
) -> go.Figure:
    """Histogram of trade P&L with a mean line."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=pnl_values,
            nbinsx=40,
            marker_color=_HIST_COLOUR,
            name="P&L",
        )
    )
    mean_val = float(np.mean(pnl_values)) if len(pnl_values) > 0 else 0
    fig.add_vline(
        x=mean_val, line_dash="dash", line_color="white", annotation_text=f"Mean: {mean_val:.2f}"
    )
    fig.update_layout(title=title, xaxis_title="P&L", yaxis_title="Count", **_LAYOUT_DEFAULTS)
    return fig


def trade_returns_fig(
    returns: np.ndarray,
    *,
    title: str = "Trade Returns Distribution",
) -> go.Figure:
    """Histogram of trade returns (%)."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=40,
            marker_color=_TEST_COLOUR,
            name="Return %",
        )
    )
    fig.update_layout(title=title, xaxis_title="Return %", yaxis_title="Count", **_LAYOUT_DEFAULTS)
    return fig


def cumulative_pnl_fig(
    pnl_values: np.ndarray,
    *,
    title: str = "Cumulative P&L",
) -> go.Figure:
    """Running cumulative P&L curve."""
    cum_pnl = np.cumsum(pnl_values)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=cum_pnl,
            mode="lines",
            name="Cumulative P&L",
            line=dict(color=_EQUITY_COLOUR, width=1.5),
        )
    )
    fig.update_layout(title=title, yaxis_title="Cumulative P&L", **_LAYOUT_DEFAULTS)
    return fig


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------


def rolling_metric_fig(
    values: np.ndarray,
    window: int = 20,
    *,
    label: str = "Rolling Sharpe",
) -> go.Figure:
    """Rolling average of a metric (e.g. rolling Sharpe from returns)."""
    import pandas as pd

    series = pd.Series(values)
    rolling = series.rolling(window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=rolling.values,
            mode="lines",
            name=label,
            line=dict(color=_EQUITY_COLOUR, width=1.5),
        )
    )
    fig.update_layout(title=label, yaxis_title=label, **_LAYOUT_DEFAULTS)
    return fig
