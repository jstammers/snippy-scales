"""Fold breakdown page — train vs test comparison and overfitting detection."""

from __future__ import annotations

import streamlit as st

from snippy_scales.dashboard.charts import fold_comparison_fig
from snippy_scales.dashboard.data_access import (
    db_path_from_state,
    get_analytics_store,
    load_fold_results,
)

st.header("Fold Breakdown")

db_path = db_path_from_state()
store = get_analytics_store(db_path)

# Need a sweep ID — check session state or let user input
sweep_id = st.text_input(
    "Sweep ID",
    value=st.session_state.get("selected_sweep_id", ""),
    help="Paste a sweep ID or select one from the Sweep Analysis page.",
)

if not sweep_id:
    st.info("Enter a sweep ID above, or select one from the **Sweep Analysis** page.")
    st.stop()

fold_df = load_fold_results(store, sweep_id)
if fold_df.empty:
    st.warning(f"No fold results found for sweep `{sweep_id}`.")
    st.stop()

# ── Train vs Test comparison ──────────────────────────────────────────────
st.subheader("Train vs Test Metrics")

fold_indices = fold_df["fold_idx"].tolist()

col1, col2 = st.columns(2)

with col1:
    fig_sharpe = fold_comparison_fig(
        fold_indices,
        fold_df["train_sharpe"].tolist(),
        fold_df["sharpe_ratio"].tolist(),
        metric_name="Sharpe Ratio",
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

with col2:
    fig_return = fold_comparison_fig(
        fold_indices,
        fold_df["train_return_pct"].tolist(),
        fold_df["total_return_pct"].tolist(),
        metric_name="Return %",
    )
    st.plotly_chart(fig_return, use_container_width=True)

# Drawdown comparison
fig_dd = fold_comparison_fig(
    fold_indices,
    fold_df["train_max_dd"].tolist(),
    fold_df["max_drawdown_pct"].tolist(),
    metric_name="Max Drawdown %",
)
st.plotly_chart(fig_dd, use_container_width=True)

# ── Overfitting indicator ────────────────────────────────────────────────
mean_train_sharpe = fold_df["train_sharpe"].mean()
mean_test_sharpe = fold_df["sharpe_ratio"].mean()

if mean_test_sharpe > 0 and mean_train_sharpe > 2 * mean_test_sharpe:
    st.error(
        f"Possible overfitting: mean train Sharpe ({mean_train_sharpe:.3f}) "
        f"is more than 2x mean test Sharpe ({mean_test_sharpe:.3f})."
    )
elif mean_test_sharpe > 0 and mean_train_sharpe > 1.5 * mean_test_sharpe:
    st.warning(
        f"Moderate overfit risk: mean train Sharpe ({mean_train_sharpe:.3f}) "
        f"vs mean test Sharpe ({mean_test_sharpe:.3f})."
    )

# ── Fold equity curves ───────────────────────────────────────────────────
st.subheader("Fold Equity Curves")

if "test_equity_curve" in fold_df.columns:
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    for _, row in fold_df.iterrows():
        eq = row.get("test_equity_curve")
        if eq is not None and len(eq) > 0:
            fig.add_trace(
                go.Scatter(
                    y=np.asarray(eq, dtype=np.float64),
                    mode="lines",
                    name=f"Fold {row['fold_idx']}",
                )
            )
    fig.update_layout(
        title="Test Equity Curves by Fold",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("No equity curve data stored for these folds.")

# ── Date ranges ──────────────────────────────────────────────────────────
st.subheader("Fold Date Ranges")
date_cols = ["fold_idx", "train_start", "train_end", "test_start", "test_end"]
available = [c for c in date_cols if c in fold_df.columns]
st.dataframe(fold_df[available], use_container_width=True, hide_index=True)

# ── Full metrics detail ──────────────────────────────────────────────────
with st.expander("Full test metrics per fold"):
    st.dataframe(fold_df, use_container_width=True, hide_index=True)
