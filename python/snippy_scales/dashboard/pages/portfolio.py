"""Portfolio overview page — aggregate strategy performance."""

from __future__ import annotations

import streamlit as st

from snippy_scales.dashboard.data_access import (
    db_path_from_state,
    get_analytics_store,
    get_backtest_store,
    load_backtest_runs,
    load_experiments,
)

st.header("Portfolio Overview")

db_path = db_path_from_state()

# ── Counts ────────────────────────────────────────────────────────────────
analytics_store = get_analytics_store(db_path)
backtest_store = get_backtest_store(db_path)

experiments_df = load_experiments(analytics_store)
runs_df = load_backtest_runs(backtest_store)

col1, col2 = st.columns(2)
col1.metric("Total Experiments", len(experiments_df))
col2.metric("Total Backtest Runs", len(runs_df))

if runs_df.empty:
    st.info("No backtest runs to analyse.")
    st.stop()

# ── Strategy summary ─────────────────────────────────────────────────────
st.subheader("Strategy Summary")

summary = (
    runs_df.groupby("strategy_name")
    .agg(
        runs=("id", "count"),
        avg_sharpe=("sharpe_ratio", "mean"),
        avg_return=("total_return_pct", "mean"),
        avg_max_dd=("max_drawdown_pct", "mean"),
        avg_win_rate=("win_rate_pct", "mean"),
    )
    .reset_index()
    .sort_values("avg_sharpe", ascending=False)
)

st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Comparison charts ────────────────────────────────────────────────────
if len(summary) > 1:
    import plotly.graph_objects as go

    st.subheader("Strategy Comparison")

    strategies = summary["strategy_name"].tolist()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=strategies, y=summary["avg_sharpe"].tolist(), name="Avg Sharpe"))
        fig.update_layout(
            title="Average Sharpe Ratio by Strategy",
            template="plotly_dark",
            height=350,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=summary["avg_return"].tolist(),
                name="Avg Return %",
                marker_color="#4CAF50",
            )
        )
        fig.update_layout(
            title="Average Return % by Strategy",
            template="plotly_dark",
            height=350,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Symbol breakdown ─────────────────────────────────────────────────────
st.subheader("Symbol Breakdown")

symbol_summary = (
    runs_df.groupby("symbol")
    .agg(
        runs=("id", "count"),
        avg_sharpe=("sharpe_ratio", "mean"),
        avg_return=("total_return_pct", "mean"),
    )
    .reset_index()
    .sort_values("avg_sharpe", ascending=False)
)

st.dataframe(symbol_summary, use_container_width=True, hide_index=True)
