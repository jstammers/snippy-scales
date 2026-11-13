"""DuckDB persistence layer for single-pass backtest results.

Schema
------
One table is maintained in a single DuckDB database file:

``backtest_runs``
    One row per single-pass backtest.  Contains all 33
    :class:`~snippy_scales.backtesting.domain.BacktestMetrics` fields plus
    run metadata (id, symbol, strategy name, timestamps, config parameters).

For evaluation (walk-forward) analytics, see
:class:`~snippy_scales.evaluation.database.AnalyticsStore`.

Typical usage::

    import duckdb
    from snippy_scales.backtesting.store import BacktestStore

    store = BacktestStore("data/analytics.duckdb")
    run_id = store.save_run(result, strategy_name="TrendFollowing")
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb as _duckdb

    from snippy_scales.backtesting.domain import BacktestResult

# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------

# All 33 BacktestMetrics columns — defined once and reused by BacktestStore
# and evaluation.AnalyticsStore (fold_results test metrics).
METRICS_COLS = """
    -- Core performance ratios
    total_return_pct          DOUBLE,
    sharpe_ratio              DOUBLE,
    sortino_ratio             DOUBLE,
    calmar_ratio              DOUBLE,
    omega_ratio               DOUBLE,
    -- Drawdown
    max_drawdown_pct          DOUBLE,
    max_drawdown_duration     INTEGER,
    -- Trade counts
    total_trades              INTEGER,
    total_closed_trades       INTEGER,
    total_open_trades         INTEGER,
    winning_trades            INTEGER,
    losing_trades             INTEGER,
    -- Trade performance
    win_rate_pct              DOUBLE,
    profit_factor             DOUBLE,
    expectancy                DOUBLE,
    sqn                       DOUBLE,
    avg_trade_return_pct      DOUBLE,
    avg_win_pct               DOUBLE,
    avg_loss_pct              DOUBLE,
    best_trade_pct            DOUBLE,
    worst_trade_pct           DOUBLE,
    payoff_ratio              DOUBLE,
    recovery_factor           DOUBLE,
    -- Duration (bars)
    avg_holding_period        DOUBLE,
    avg_winning_duration      DOUBLE,
    avg_losing_duration       DOUBLE,
    -- Streaks
    max_consecutive_wins      INTEGER,
    max_consecutive_losses    INTEGER,
    -- Portfolio
    start_value               DOUBLE,
    end_value                 DOUBLE,
    total_fees_paid           DOUBLE,
    open_trade_pnl            DOUBLE,
    exposure_pct              DOUBLE"""

_DDL_BACKTEST_RUNS = f"""
CREATE TABLE IF NOT EXISTS backtest_runs (
    id               VARCHAR PRIMARY KEY,
    symbol           VARCHAR  NOT NULL,
    strategy_name    VARCHAR,
    run_at           TIMESTAMP NOT NULL,
    initial_capital  DOUBLE,
    fees             DOUBLE,
    slippage         DOUBLE,
    {METRICS_COLS.strip()}
)"""

# Flat ordered list of the 33 metric attribute names — used to build INSERT
# parameter lists without repeating the names in multiple places.
METRIC_NAMES: tuple[str, ...] = (
    "total_return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "max_drawdown_pct",
    "max_drawdown_duration",
    "total_trades",
    "total_closed_trades",
    "total_open_trades",
    "winning_trades",
    "losing_trades",
    "win_rate_pct",
    "profit_factor",
    "expectancy",
    "sqn",
    "avg_trade_return_pct",
    "avg_win_pct",
    "avg_loss_pct",
    "best_trade_pct",
    "worst_trade_pct",
    "payoff_ratio",
    "recovery_factor",
    "avg_holding_period",
    "avg_winning_duration",
    "avg_losing_duration",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "start_value",
    "end_value",
    "total_fees_paid",
    "open_trade_pnl",
    "exposure_pct",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def metrics_values(metrics: object) -> list[object]:
    """Extract all 33 metric values from a BacktestMetrics instance."""
    return [getattr(metrics, name) for name in METRIC_NAMES]


def _symbol_str(symbol: str | list[str]) -> str:
    """Normalise symbol to a comma-separated string for storage."""
    if isinstance(symbol, str):
        return symbol
    return ",".join(symbol)


def _now() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# BacktestStore
# ---------------------------------------------------------------------------


class BacktestStore:
    """Thin DuckDB wrapper that persists single-pass backtest results.

    For walk-forward / evaluation analytics, use
    :class:`~snippy_scales.evaluation.database.AnalyticsStore`.

    Args:
        db_path: Path to the DuckDB file.  Pass ``\":memory:\"`` for an
            in-process, in-memory database (useful for testing).

    Example::

        store = BacktestStore("data/analytics.duckdb")
        run_id = store.save_run(result, strategy_name="TrendFollowing",
                                initial_capital=100_000, fees=0.001)
    """

    def __init__(self, db_path: str = "data/analytics.duckdb") -> None:
        import duckdb  # noqa: PLC0415 — optional dep, imported lazily

        self._conn: _duckdb.DuckDBPyConnection = duckdb.connect(db_path)
        self.create_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create the ``backtest_runs`` table if it does not already exist."""
        self._conn.execute(_DDL_BACKTEST_RUNS)

    # ------------------------------------------------------------------
    # Single-pass backtest persistence
    # ------------------------------------------------------------------

    def save_run(
        self,
        result: BacktestResult,
        *,
        strategy_name: str | None = None,
        initial_capital: float | None = None,
        fees: float | None = None,
        slippage: float | None = None,
        run_id: str | None = None,
    ) -> str:
        """Persist a single :class:`BacktestResult` to ``backtest_runs``.

        Args:
            result: The result to persist.
            strategy_name: Optional human-readable strategy identifier.
            initial_capital: Starting capital used for this run.
            fees: Per-trade commission used (fraction of trade value).
            slippage: Round-trip slippage used (fraction of trade value).
            run_id: Override the auto-generated UUID.  Useful for idempotent
                writes or linking results to external run tracking.

        Returns:
            The ``id`` string written to the database.
        """
        rid = run_id or str(uuid.uuid4())
        meta = [
            rid,
            _symbol_str(result.symbol),
            strategy_name,
            _now(),
            initial_capital,
            fees,
            slippage,
        ]
        metric_vals = metrics_values(result.metrics)
        placeholders = ", ".join(["?"] * (len(meta) + len(metric_vals)))
        self._conn.execute(
            f"INSERT OR REPLACE INTO backtest_runs VALUES ({placeholders})",
            meta + metric_vals,
        )
        return rid

    # ------------------------------------------------------------------
    # Querying convenience methods
    # ------------------------------------------------------------------

    def query(self, sql: str, params: list[object] | None = None) -> list[tuple[object, ...]]:
        """Execute a raw SQL query and return all rows.

        Args:
            sql: SQL statement.
            params: Optional positional parameters.

        Returns:
            List of result tuples.
        """
        if params:
            return self._conn.execute(sql, params).fetchall()
        return self._conn.execute(sql).fetchall()

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> BacktestStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
