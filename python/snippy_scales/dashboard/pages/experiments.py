"""Experiments overview page — browse and filter evaluation runs."""

from __future__ import annotations

import streamlit as st

from snippy_scales.dashboard.data_access import (
    db_path_from_state,
    get_analytics_store,
    load_experiments,
)

st.header("Experiments")

db_path = db_path_from_state()
store = get_analytics_store(db_path)
df = load_experiments(store)

if df.empty:
    st.info("No experiments found. Run `algo eval run` or `algo eval sweep` first.")
    st.stop()

# Strategy filter
strategies = sorted(df["strategy_class"].dropna().unique())
selected_strategy = st.selectbox(
    "Filter by strategy",
    options=["All"] + strategies,
)

filtered = df if selected_strategy == "All" else df[df["strategy_class"] == selected_strategy]

# Display table
st.dataframe(
    filtered[["id", "name", "strategy_class", "n_splits", "window_type", "created_at"]],
    use_container_width=True,
    hide_index=True,
)

# Drill-down selector
exp_ids = filtered["id"].tolist()
if exp_ids:
    selected_id = st.selectbox("Select experiment to explore sweeps", options=exp_ids)
    st.session_state["selected_experiment_id"] = selected_id
    st.info(f"Experiment **{selected_id}** selected. Go to **Sweep Analysis** to explore.")
