"""Cached data access layer for the Streamlit dashboard.

Wraps :class:`~snippy_scales.evaluation.database.AnalyticsStore` and
:class:`~snippy_scales.backtesting.store.BacktestStore` with Streamlit
caching decorators for efficient re-renders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

from snippy_scales.backtesting.store import BacktestStore
from snippy_scales.evaluation.database import AnalyticsStore

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Resource singletons — one connection per db_path for the session
# ---------------------------------------------------------------------------


@st.cache_resource
def get_analytics_store(db_path: str) -> AnalyticsStore:
    """Return a shared AnalyticsStore connection."""
    return AnalyticsStore(db_path)


@st.cache_resource
def get_backtest_store(db_path: str) -> BacktestStore:
    """Return a shared BacktestStore connection."""
    return BacktestStore(db_path)


# ---------------------------------------------------------------------------
# Cached data queries — TTL keeps data fresh on re-runs
# ---------------------------------------------------------------------------

_TTL = 30  # seconds


@st.cache_data(ttl=_TTL)
def load_experiments(_store: AnalyticsStore) -> Any:
    """Load all experiments as a Pandas DataFrame."""
    df = _store.load_experiments()
    return df.to_pandas() if len(df) > 0 else df.to_pandas()


@st.cache_data(ttl=_TTL)
def load_sweep_results(_store: AnalyticsStore, experiment_id: int) -> Any:
    """Load sweep results for an experiment as a Pandas DataFrame."""
    df = _store.load_sweep_results(experiment_id)
    return df.to_pandas() if len(df) > 0 else df.to_pandas()


@st.cache_data(ttl=_TTL)
def load_fold_results(_store: AnalyticsStore, sweep_id: str) -> Any:
    """Load fold results for a sweep as a Pandas DataFrame."""
    df = _store.load_fold_results(sweep_id)
    return df.to_pandas() if len(df) > 0 else df.to_pandas()


@st.cache_data(ttl=_TTL)
def load_backtest_runs(_store: BacktestStore) -> Any:
    """Load all backtest runs as a Pandas DataFrame."""
    df = _store.load_runs()
    return df.to_pandas() if len(df) > 0 else df.to_pandas()


@st.cache_data(ttl=_TTL)
def load_trades(_store: BacktestStore, run_id: str) -> Any:
    """Load trades for a run as a Pandas DataFrame."""
    df = _store.load_trades(run_id)
    return df.to_pandas() if len(df) > 0 else df.to_pandas()


@st.cache_data(ttl=_TTL)
def load_fold_trades(_store: AnalyticsStore, fold_id: str) -> Any:
    """Load trades for a fold as a Pandas DataFrame."""
    df = _store.load_fold_trades(fold_id)
    return df.to_pandas() if len(df) > 0 else df.to_pandas()


@st.cache_data(ttl=_TTL)
def load_equity_series(_store: BacktestStore, run_id: str) -> dict[str, np.ndarray] | None:
    """Load equity curve, drawdown, and returns for a run."""
    return _store.load_equity_series(run_id)


def db_path_from_state() -> str:
    """Get the DuckDB path from session state or environment."""
    import os

    return st.session_state.get(
        "db_path",
        os.environ.get("SNIPPY_DB_PATH", "data/analytics.duckdb"),
    )
