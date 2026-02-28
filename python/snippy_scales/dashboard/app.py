"""Streamlit multi-page application entry point.

Launch via::

    algo dashboard launch --db data/analytics.duckdb

Or directly::

    streamlit run python/snippy_scales/dashboard/app.py -- --db data/analytics.duckdb
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SnippyScales Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — database path selector
# ---------------------------------------------------------------------------

default_db = os.environ.get("SNIPPY_DB_PATH", "data/analytics.duckdb")

with st.sidebar:
    st.title("SnippyScales")
    db_path = st.text_input("DuckDB path", value=default_db)
    st.session_state["db_path"] = db_path

    db_exists = Path(db_path).exists() if db_path != ":memory:" else True
    if not db_exists:
        st.warning(f"Database not found: {db_path}")

    if st.button("Refresh data"):
        st.cache_data.clear()

# ---------------------------------------------------------------------------
# Page registration
# ---------------------------------------------------------------------------

pages_dir = Path(__file__).parent / "pages"

pages = [
    st.Page(str(pages_dir / "experiments.py"), title="Experiments", icon=":test_tube:"),
    st.Page(str(pages_dir / "sweep.py"), title="Sweep Analysis", icon=":bar_chart:"),
    st.Page(str(pages_dir / "folds.py"), title="Fold Breakdown", icon=":clipboard:"),
    st.Page(str(pages_dir / "runs.py"), title="Backtest Runs", icon=":rocket:"),
    st.Page(str(pages_dir / "equity.py"), title="Equity Curves", icon=":chart_with_upwards_trend:"),
    st.Page(str(pages_dir / "trades.py"), title="Trade Analysis", icon=":money_with_wings:"),
    st.Page(str(pages_dir / "portfolio.py"), title="Portfolio", icon=":briefcase:"),
]

pg = st.navigation(pages)
pg.run()
