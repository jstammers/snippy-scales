"""DuckDB persistence layer for backtest results.

Schema
------
Four tables are maintained in a single DuckDB database file:

``backtest_runs``
    One row per single-pass backtest.  Contains all 33
    :class:`~snippy_scales.backtesting.domain.BacktestMetrics` fields plus
    run metadata (id, symbol, strategy name, timestamps, config parameters).

``walk_forward_runs``
    One row per walk-forward analysis — a container that groups multiple
    out-of-sample folds together.

``walk_forward_folds``
    One row per fold within a walk-forward run.  Stores the fold's OOS window
    boundaries and all 33 metrics for that fold's out-of-sample period.

``walk_forward_summary``
    One row per walk-forward run.  Stores the mean **and** standard deviation
    for the 14 key metrics aggregated across all folds, making it easy to
    assess both average performance and fold-to-fold consistency.

Typical usage::

    import duckdb
    from snippy_scales.backtesting.store import BacktestStore

    store = BacktestStore("data/results.duckdb")
    run_id = store.save_run(result, strategy_name="TrendFollowing")
    wf_id  = store.save_walk_forward(wf_result, strategy_name="TrendFollowing")
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb as _duckdb

    from snippy_scales.backtesting.domain import BacktestResult, WalkForwardResult

# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------

# All 33 BacktestMetrics columns, shared across backtest_runs and
# walk_forward_folds.  Defined once and interpolated into both CREATE TABLE
# statements to keep the schema in sync automatically.
_METRICS_COLS = """
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

# Mean + std_dev columns for the 14 key metrics stored in walk_forward_summary.
_SUMMARY_COLS = """
    total_return_pct_mean     DOUBLE,
    total_return_pct_std      DOUBLE,
    sharpe_ratio_mean         DOUBLE,
    sharpe_ratio_std          DOUBLE,
    sortino_ratio_mean        DOUBLE,
    sortino_ratio_std         DOUBLE,
    calmar_ratio_mean         DOUBLE,
    calmar_ratio_std          DOUBLE,
    omega_ratio_mean          DOUBLE,
    omega_ratio_std           DOUBLE,
    max_drawdown_pct_mean     DOUBLE,
    max_drawdown_pct_std      DOUBLE,
    win_rate_pct_mean         DOUBLE,
    win_rate_pct_std          DOUBLE,
    profit_factor_mean        DOUBLE,
    profit_factor_std         DOUBLE,
    expectancy_mean           DOUBLE,
    expectancy_std            DOUBLE,
    sqn_mean                  DOUBLE,
    sqn_std                   DOUBLE,
    recovery_factor_mean      DOUBLE,
    recovery_factor_std       DOUBLE,
    payoff_ratio_mean         DOUBLE,
    payoff_ratio_std          DOUBLE,
    avg_trade_return_pct_mean DOUBLE,
    avg_trade_return_pct_std  DOUBLE,
    exposure_pct_mean         DOUBLE,
    exposure_pct_std          DOUBLE"""

_DDL_BACKTEST_RUNS = f"""
CREATE TABLE IF NOT EXISTS backtest_runs (
    id               VARCHAR PRIMARY KEY,
    symbol           VARCHAR  NOT NULL,
    strategy_name    VARCHAR,
    run_at           TIMESTAMP NOT NULL,
    initial_capital  DOUBLE,
    fees             DOUBLE,
    slippage         DOUBLE,
    {_METRICS_COLS.strip()}
)"""

_DDL_WALK_FORWARD_RUNS = """
CREATE TABLE IF NOT EXISTS walk_forward_runs (
    id               VARCHAR PRIMARY KEY,
    symbol           VARCHAR  NOT NULL,
    strategy_name    VARCHAR,
    run_at           TIMESTAMP NOT NULL,
    n_folds          INTEGER   NOT NULL,
    initial_capital  DOUBLE,
    fees             DOUBLE,
    slippage         DOUBLE
)"""

_DDL_WALK_FORWARD_FOLDS = f"""
CREATE TABLE IF NOT EXISTS walk_forward_folds (
    id                    VARCHAR PRIMARY KEY,
    walk_forward_run_id   VARCHAR  NOT NULL REFERENCES walk_forward_runs(id),
    fold_index            INTEGER  NOT NULL,
    oos_start             BIGINT,
    oos_end               BIGINT,
    {_METRICS_COLS.strip()}
)"""

_DDL_WALK_FORWARD_SUMMARY = f"""
CREATE TABLE IF NOT EXISTS walk_forward_summary (
    walk_forward_run_id   VARCHAR PRIMARY KEY REFERENCES walk_forward_runs(id),
    {_SUMMARY_COLS.strip()}
)"""

# Flat ordered list of the 33 metric attribute names — used to build INSERT
# parameter lists without repeating the names in multiple places.
_METRIC_NAMES: tuple[str, ...] = (
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

_SUMMARY_METRIC_NAMES: tuple[str, ...] = (
    "total_return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "max_drawdown_pct",
    "win_rate_pct",
    "profit_factor",
    "expectancy",
    "sqn",
    "recovery_factor",
    "payoff_ratio",
    "avg_trade_return_pct",
    "exposure_pct",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _metrics_values(metrics: object) -> list[object]:
    """Extract all 33 metric values from a BacktestMetrics instance."""
    return [getattr(metrics, name) for name in _METRIC_NAMES]


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
    """Thin DuckDB wrapper that persists backtest and walk-forward results.

    Args:
        db_path: Path to the DuckDB file.  Pass ``":memory:"`` for an
            in-process, in-memory database (useful for testing).

    Example::

        store = BacktestStore("data/results.duckdb")
        run_id = store.save_run(result, strategy_name="TrendFollowing",
                                initial_capital=100_000, fees=0.001)
    """

    def __init__(self, db_path: str = "data/results.duckdb") -> None:
        import duckdb  # noqa: PLC0415 — optional dep, imported lazily

        self._conn: _duckdb.DuckDBPyConnection = duckdb.connect(db_path)
        self.create_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create all four tables if they do not already exist."""
        for ddl in (
            _DDL_BACKTEST_RUNS,
            _DDL_WALK_FORWARD_RUNS,
            _DDL_WALK_FORWARD_FOLDS,
            _DDL_WALK_FORWARD_SUMMARY,
        ):
            self._conn.execute(ddl)

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
        metric_vals = _metrics_values(result.metrics)
        placeholders = ", ".join(["?"] * (len(meta) + len(metric_vals)))
        self._conn.execute(
            f"INSERT OR REPLACE INTO backtest_runs VALUES ({placeholders})",
            meta + metric_vals,
        )
        return rid

    # ------------------------------------------------------------------
    # Walk-forward persistence
    # ------------------------------------------------------------------

    def save_walk_forward(
        self,
        result: WalkForwardResult,
        *,
        strategy_name: str | None = None,
        initial_capital: float | None = None,
        fees: float | None = None,
        slippage: float | None = None,
        run_id: str | None = None,
    ) -> str:
        """Persist a :class:`WalkForwardResult` across all three WF tables.

        Writes:
        * one row to ``walk_forward_runs``
        * one row per fold to ``walk_forward_folds``
        * one row to ``walk_forward_summary``

        All three writes are wrapped in a single transaction.

        Args:
            result: The walk-forward result to persist.
            strategy_name: Optional strategy identifier.
            initial_capital: Starting capital per fold.
            fees: Per-trade commission used.
            slippage: Round-trip slippage used.
            run_id: Override the auto-generated UUID.

        Returns:
            The ``id`` of the ``walk_forward_runs`` row.
        """
        wf_id = run_id or str(uuid.uuid4())
        sym = _symbol_str(result.symbol)
        now = _now()

        self._conn.begin()
        try:
            # walk_forward_runs
            self._conn.execute(
                "INSERT OR REPLACE INTO walk_forward_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    wf_id,
                    sym,
                    strategy_name,
                    now,
                    len(result.folds),
                    initial_capital,
                    fees,
                    slippage,
                ],
            )

            # walk_forward_folds — one row per fold
            fold_placeholders = ", ".join(["?"] * (5 + len(_METRIC_NAMES)))
            for fold in result.folds:
                fold_id = str(uuid.uuid4())
                fold_meta = [
                    fold_id,
                    wf_id,
                    fold.fold_index,
                    fold.oos_start,
                    fold.oos_end,
                ]
                self._conn.execute(
                    f"INSERT OR REPLACE INTO walk_forward_folds VALUES ({fold_placeholders})",
                    fold_meta + _metrics_values(fold.result.metrics),
                )

            # walk_forward_summary
            s = result.summary
            summary_vals: list[object] = [wf_id]
            for metric in _SUMMARY_METRIC_NAMES:
                summary_vals.append(getattr(s, f"{metric}_mean"))
                summary_vals.append(getattr(s, f"{metric}_std"))
            summary_placeholders = ", ".join(["?"] * len(summary_vals))
            self._conn.execute(
                f"INSERT OR REPLACE INTO walk_forward_summary VALUES ({summary_placeholders})",
                summary_vals,
            )

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        return wf_id

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
