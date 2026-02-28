"""Sweep analysis page — parameter sweep heatmaps and ranking."""

from __future__ import annotations

import json

import streamlit as st

from snippy_scales.dashboard.charts import sweep_heatmap_fig
from snippy_scales.dashboard.data_access import (
    db_path_from_state,
    get_analytics_store,
    load_experiments,
    load_sweep_results,
)

st.header("Sweep Analysis")

db_path = db_path_from_state()
store = get_analytics_store(db_path)

# Experiment selector
experiments_df = load_experiments(store)
if experiments_df.empty:
    st.info("No experiments found.")
    st.stop()

default_idx = 0
if "selected_experiment_id" in st.session_state:
    target = st.session_state["selected_experiment_id"]
    ids = experiments_df["id"].tolist()
    if target in ids:
        default_idx = ids.index(target)

selected_exp = st.selectbox(
    "Experiment",
    options=experiments_df["id"].tolist(),
    index=default_idx,
    format_func=lambda x: f"{x} — {experiments_df[experiments_df['id'] == x]['name'].iloc[0]}",
)

sweep_df = load_sweep_results(store, int(selected_exp))
if sweep_df.empty:
    st.info("No sweep results for this experiment.")
    st.stop()

# Parse params_json into columns
params_parsed = sweep_df["params_json"].apply(json.loads)
param_keys = sorted(params_parsed.iloc[0].keys()) if len(params_parsed) > 0 else []
for key in param_keys:
    sweep_df[key] = params_parsed.apply(lambda p, k=key: p.get(k))

# Metric selector
metric_options = [
    "mean_test_sharpe",
    "mean_test_return",
    "mean_test_max_dd",
    "mean_test_sortino",
]
sort_metric = st.selectbox("Sort / colour by", options=metric_options)

# Ranked table
st.subheader("Parameter Sets (ranked)")
display_cols = param_keys + [
    "mean_test_sharpe",
    "std_test_sharpe",
    "mean_test_return",
    "std_test_return",
    "mean_test_max_dd",
    "std_test_max_dd",
    "n_folds",
]
available_cols = [c for c in display_cols if c in sweep_df.columns]
st.dataframe(
    sweep_df[available_cols].sort_values(sort_metric, ascending=False),
    use_container_width=True,
    hide_index=True,
)

# 2D heatmap for 2-param sweeps
if len(param_keys) == 2:
    st.subheader("Parameter Heatmap")
    fig = sweep_heatmap_fig(sweep_df, param_keys[0], param_keys[1], sort_metric)
    st.plotly_chart(fig, use_container_width=True)
elif len(param_keys) > 2:
    st.caption(
        f"Heatmap available for 2-parameter sweeps. "
        f"This sweep has {len(param_keys)} parameters: {', '.join(param_keys)}"
    )

# Sweep selector for drill-down
sweep_ids = sweep_df["id"].tolist()
if sweep_ids:
    selected_sweep = st.selectbox("Select sweep to explore folds", options=sweep_ids)
    st.session_state["selected_sweep_id"] = selected_sweep
    st.info(f"Sweep **{selected_sweep}** selected. Go to **Fold Breakdown** to explore.")
