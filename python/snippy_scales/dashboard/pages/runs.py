"""Backtest runs browser — view and filter single-pass backtest results."""

from __future__ import annotations

import streamlit as st

from snippy_scales.dashboard.data_access import (
    db_path_from_state,
    get_backtest_store,
    load_backtest_runs,
)

st.header("Backtest Runs")

db_path = db_path_from_state()
store = get_backtest_store(db_path)
df = load_backtest_runs(store)

if df.empty:
    st.info("No backtest runs found. Run a backtest first.")
    st.stop()

# Filters
col1, col2 = st.columns(2)
with col1:
    strategies = sorted(df["strategy_name"].dropna().unique())
    selected_strategy = st.selectbox("Strategy", ["All"] + strategies)
with col2:
    symbols = sorted(df["symbol"].dropna().unique())
    selected_symbol = st.selectbox("Symbol", ["All"] + symbols)

filtered = df.copy()
if selected_strategy != "All":
    filtered = filtered[filtered["strategy_name"] == selected_strategy]
if selected_symbol != "All":
    filtered = filtered[filtered["symbol"] == selected_symbol]

# Key metrics table
key_cols = [
    "id",
    "symbol",
    "strategy_name",
    "run_at",
    "sharpe_ratio",
    "total_return_pct",
    "max_drawdown_pct",
    "win_rate_pct",
    "profit_factor",
    "total_trades",
]
available_cols = [c for c in key_cols if c in filtered.columns]
st.dataframe(
    filtered[available_cols],
    use_container_width=True,
    hide_index=True,
)

# Run selector for drill-down
run_ids = filtered["id"].tolist()
if run_ids:
    selected_run = st.selectbox("Select run for detail", options=run_ids)
    st.session_state["selected_run_id"] = selected_run

    # Detail expander
    with st.expander(f"Full metrics for run {selected_run}"):
        run_row = filtered[filtered["id"] == selected_run]
        st.dataframe(run_row.T, use_container_width=True)

    st.info(
        f"Run **{selected_run}** selected. "
        "Go to **Equity Curves** or **Trade Analysis** to explore."
    )
