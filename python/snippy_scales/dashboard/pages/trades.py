"""Trade analysis page — P&L, returns, and duration analysis."""

from __future__ import annotations

import streamlit as st

from snippy_scales.dashboard.charts import (
    cumulative_pnl_fig,
    pnl_histogram_fig,
    trade_returns_fig,
)
from snippy_scales.dashboard.data_access import (
    db_path_from_state,
    get_backtest_store,
    load_backtest_runs,
    load_trades,
)

st.header("Trade Analysis")

db_path = db_path_from_state()
store = get_backtest_store(db_path)

# Run selector
runs_df = load_backtest_runs(store)
if runs_df.empty:
    st.info("No backtest runs found.")
    st.stop()

run_ids = runs_df["id"].tolist()
default_idx = 0
if "selected_run_id" in st.session_state:
    target = st.session_state["selected_run_id"]
    if target in run_ids:
        default_idx = run_ids.index(target)

selected_run = st.selectbox(
    "Run ID",
    options=run_ids,
    index=default_idx,
    format_func=lambda x: (
        f"{x[:8]}… — "
        f"{runs_df[runs_df['id'] == x]['strategy_name'].iloc[0]} / "
        f"{runs_df[runs_df['id'] == x]['symbol'].iloc[0]}"
    ),
)

trades_df = load_trades(store, selected_run)

if trades_df.empty:
    st.info("No trades stored for this run.")
    st.stop()

# ── Trade list ────────────────────────────────────────────────────────────
st.subheader(f"{len(trades_df)} Trades")

display_cols = [
    "symbol",
    "direction",
    "entry_price",
    "exit_price",
    "pnl",
    "return_pct",
]
available = [c for c in display_cols if c in trades_df.columns]
st.dataframe(trades_df[available], use_container_width=True, hide_index=True)

# ── Summary metrics ──────────────────────────────────────────────────────
pnl = trades_df["pnl"].values
returns = trades_df["return_pct"].values
wins = (pnl > 0).sum()
losses = (pnl < 0).sum()
total = len(pnl)
win_rate = wins / total * 100 if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades", total)
col2.metric("Win Rate", f"{win_rate:.1f}%")
col3.metric("Total P&L", f"{pnl.sum():,.2f}")
col4.metric("Avg P&L", f"{pnl.mean():,.2f}")

# ── Charts ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(pnl_histogram_fig(pnl), use_container_width=True)
with col2:
    st.plotly_chart(trade_returns_fig(returns), use_container_width=True)

# Cumulative P&L
st.plotly_chart(cumulative_pnl_fig(pnl), use_container_width=True)

# ── Long vs Short breakdown ──────────────────────────────────────────────
if "direction" in trades_df.columns:
    st.subheader("Long vs Short")
    long_trades = trades_df[trades_df["direction"] == 1]
    short_trades = trades_df[trades_df["direction"] == -1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Long trades", len(long_trades))
        if len(long_trades) > 0:
            st.metric("Long avg P&L", f"{long_trades['pnl'].mean():,.2f}")
    with col2:
        st.metric("Short trades", len(short_trades))
        if len(short_trades) > 0:
            st.metric("Short avg P&L", f"{short_trades['pnl'].mean():,.2f}")

# ── Holding period ───────────────────────────────────────────────────────
if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
    durations = trades_df["exit_time"] - trades_df["entry_time"]
    durations_hours = durations / 1e9 / 3600  # ns -> hours
    if durations_hours.mean() > 0:
        st.subheader("Holding Period")
        st.metric("Avg holding (hours)", f"{durations_hours.mean():.1f}")
