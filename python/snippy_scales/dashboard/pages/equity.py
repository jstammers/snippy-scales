"""Equity curves page — interactive equity and drawdown charts."""

from __future__ import annotations

import numpy as np
import streamlit as st

from snippy_scales.dashboard.charts import (
    equity_curve_fig,
    equity_with_drawdown_fig,
    rolling_metric_fig,
    trade_returns_fig,
)
from snippy_scales.dashboard.data_access import (
    db_path_from_state,
    get_backtest_store,
    load_backtest_runs,
    load_equity_series,
)

st.header("Equity Curves")

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

series = load_equity_series(store, selected_run)

if series is None:
    st.warning("No equity series data stored for this run.")
    st.stop()

equity = series["equity_curve"]
drawdown = series["drawdown_curve"]
returns = series["returns"]

# ── Equity + Drawdown ────────────────────────────────────────────────────
fig = equity_with_drawdown_fig(equity, drawdown) if len(drawdown) > 0 else equity_curve_fig(equity)
st.plotly_chart(fig, use_container_width=True)

# ── Returns distribution ─────────────────────────────────────────────────
if len(returns) > 0:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            trade_returns_fig(returns, title="Period Returns Distribution"),
            use_container_width=True,
        )
    with col2:
        window = st.slider("Rolling window", min_value=5, max_value=100, value=20)
        st.plotly_chart(
            rolling_metric_fig(returns, window=window, label="Rolling Mean Return"),
            use_container_width=True,
        )

# ── Summary stats ────────────────────────────────────────────────────────
if len(returns) > 0:
    from snippy_scales._constants import TRADING_DAYS_PER_YEAR

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    ann_sharpe = (mean_ret / std_ret * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_ret > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Daily Return", f"{mean_ret:.4%}")
    col2.metric("Daily Volatility", f"{std_ret:.4%}")
    col3.metric("Ann. Sharpe (from returns)", f"{ann_sharpe:.3f}")
