"""Persistence layer for evaluation results.

Two complementary stores implement the hybrid database architecture:

``SQLiteStore`` — metadata store
    Lightweight SQLite database holding experiment configuration and the run
    registry.  One row per evaluation run in the ``experiments`` table.

``AnalyticsStore`` — analytics store
    DuckDB database holding wide columnar data: per-parameter-set aggregated
    metrics (``sweep_results``) and per-fold detailed metrics (``fold_results``
    with all 33 :class:`~snippy_scales.backtesting.domain.BacktestMetrics`
    fields for the test window).

Typical usage::

    from snippy_scales.evaluation.database import SQLiteStore, AnalyticsStore

    sqlite = SQLiteStore("data/metadata.db")
    duckdb = AnalyticsStore("data/analytics.duckdb")

    experiment_id = sqlite.save_evaluation(eval_result, config={...})
    duckdb.save_evaluation_analytics(eval_result, experiment_id)

    # Query analytics
    df = duckdb.load_sweep_results(experiment_id)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from snippy_scales.backtesting.store import METRIC_NAMES, METRICS_COLS, metrics_values

if TYPE_CHECKING:
    import duckdb as _duckdb
    import polars as pl

    from snippy_scales.evaluation.results import EvaluationResult

# ---------------------------------------------------------------------------
# SQLiteStore — metadata
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    strategy_class  TEXT    NOT NULL,
    symbols_json    TEXT    NOT NULL,
    n_splits        INTEGER NOT NULL,
    window_type     TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    config_json     TEXT    NOT NULL
);
"""


class SQLiteStore:
    """Persist and retrieve experiment metadata in a local SQLite database.

    Stores only the lightweight registry and configuration for each evaluation
    run.  Analytics data (per-fold and per-sweep metrics) are stored in the
    :class:`AnalyticsStore` (DuckDB).

    The database file is created automatically if it does not exist.

    Args:
        db_path: Path to the SQLite file (e.g. ``"data/metadata.db"`` or
            ``Path("data/metadata.db")``).

    Example::

        store = SQLiteStore("data/metadata.db")
        experiment_id = store.save_evaluation(eval_result, config={...})
        df = store.load_experiments()
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SQLITE_SCHEMA)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_evaluation(
        self,
        result: EvaluationResult,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Register an evaluation run in the ``experiments`` table.

        Args:
            result: The evaluation result to register.
            config: Optional dict of runner configuration to store alongside
                the experiment (fees, slippage, initial_capital, etc.).

        Returns:
            The ``experiment_id`` (primary key) assigned to this run.
        """
        config = config or {}
        created_at = datetime.now(tz=UTC).isoformat()

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments
                    (name, strategy_class, symbols_json, n_splits, window_type,
                     created_at, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.experiment_name,
                    result.strategy_class,
                    json.dumps(result.symbols),
                    config.get("n_splits", 0),
                    config.get("window", ""),
                    created_at,
                    json.dumps(config),
                ),
            )
            exp_id = cur.lastrowid

        if exp_id is None:  # pragma: no cover — INSERT always sets lastrowid
            raise RuntimeError("INSERT did not return a lastrowid")
        return exp_id

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_experiments(self) -> pl.DataFrame:
        """Return all experiments as a Polars DataFrame."""
        import polars as pl

        rows = self._fetchall("SELECT * FROM experiments ORDER BY id DESC")
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> pl.DataFrame:
        """Execute arbitrary read-only SQL and return results as a Polars DataFrame.

        Args:
            sql: SQL query string.
            params: Optional bind parameters.

        Returns:
            Query results as a Polars DataFrame.
        """
        import polars as pl

        rows = self._fetchall(sql, params)
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def __repr__(self) -> str:
        return f"SQLiteStore({str(self._path)!r})"


# ---------------------------------------------------------------------------
# AnalyticsStore — DuckDB
# ---------------------------------------------------------------------------

_DDL_SWEEP_RESULTS = """
CREATE TABLE IF NOT EXISTS sweep_results (
    id                  VARCHAR PRIMARY KEY,
    experiment_id       INTEGER NOT NULL,
    params_json         TEXT    NOT NULL,
    n_folds             INTEGER NOT NULL,
    mean_test_sharpe    DOUBLE,
    std_test_sharpe     DOUBLE,
    mean_test_return    DOUBLE,
    std_test_return     DOUBLE,
    mean_test_max_dd    DOUBLE,
    std_test_max_dd     DOUBLE,
    mean_test_sortino   DOUBLE,
    std_test_sortino    DOUBLE
)"""

_DDL_FOLD_RESULTS = f"""
CREATE TABLE IF NOT EXISTS fold_results (
    id              VARCHAR PRIMARY KEY,
    sweep_id        VARCHAR NOT NULL,
    fold_idx        INTEGER NOT NULL,
    train_start     TEXT,
    train_end       TEXT,
    test_start      TEXT,
    test_end        TEXT,
    -- 4 key training-window metrics
    train_return_pct    DOUBLE,
    train_sharpe        DOUBLE,
    train_max_dd        DOUBLE,
    train_trades        INTEGER,
    -- all 33 test-window metrics
    {METRICS_COLS.strip()}
)"""


class AnalyticsStore:
    """DuckDB analytics store for evaluation sweep and fold results.

    Stores wide columnar data for offline analysis:

    * ``sweep_results`` — one row per (experiment, parameter set) with
      mean and population std-dev for 4 key test metrics.
    * ``fold_results`` — one row per (sweep, fold) with 4 training-window
      metrics and all 33 test-window metrics.

    Args:
        db_path: Path to the DuckDB file.  Pass ``\":memory:\"`` for an
            in-process, in-memory database (useful for testing).

    Example::

        store = AnalyticsStore("data/analytics.duckdb")
        store.save_evaluation_analytics(eval_result, experiment_id=1)
        df = store.load_sweep_results(experiment_id=1)
    """

    def __init__(self, db_path: Path | str = "data/analytics.duckdb") -> None:
        import duckdb  # noqa: PLC0415 — optional dep, imported lazily

        self._conn: _duckdb.DuckDBPyConnection = duckdb.connect(str(db_path))
        self.create_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create ``sweep_results`` and ``fold_results`` tables if absent."""
        for ddl in (_DDL_SWEEP_RESULTS, _DDL_FOLD_RESULTS):
            self._conn.execute(ddl)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_evaluation_analytics(
        self,
        result: EvaluationResult,
        experiment_id: int,
    ) -> None:
        """Persist sweep and fold analytics for an evaluation result.

        Writes one row to ``sweep_results`` per :class:`SweepResult` and one
        row to ``fold_results`` per :class:`FoldResult`.  All writes are
        wrapped in a single transaction.

        Args:
            result: The evaluation result to persist.
            experiment_id: Primary key from the ``SQLiteStore.experiments``
                table (use ``0`` if not linked to a SQLite metadata store).
        """
        self._conn.begin()
        try:
            sweep_placeholders = ", ".join(["?"] * 12)
            fold_placeholders = ", ".join(["?"] * (11 + len(METRIC_NAMES)))

            for sweep in result.sweep_results:
                sweep_id = str(uuid.uuid4())
                self._conn.execute(
                    f"INSERT OR REPLACE INTO sweep_results VALUES ({sweep_placeholders})",
                    [
                        sweep_id,
                        experiment_id,
                        json.dumps(sweep.params),
                        sweep.n_folds,
                        sweep.mean_test_sharpe,
                        sweep.std_test_sharpe,
                        sweep.mean_test_return,
                        sweep.std_test_return,
                        sweep.mean_test_max_dd,
                        sweep.std_test_max_dd,
                        sweep.mean_test_sortino,
                        sweep.std_test_sortino,
                    ],
                )

                for fold in sweep.folds:
                    fold_id = str(uuid.uuid4())
                    fold_meta = [
                        fold_id,
                        sweep_id,
                        fold.fold_idx,
                        fold.train_start,
                        fold.train_end,
                        fold.test_start,
                        fold.test_end,
                        # 4 key training metrics
                        fold.train_metrics.total_return_pct,
                        fold.train_metrics.sharpe_ratio,
                        fold.train_metrics.max_drawdown_pct,
                        fold.train_metrics.total_trades,
                    ]
                    self._conn.execute(
                        f"INSERT OR REPLACE INTO fold_results VALUES ({fold_placeholders})",
                        fold_meta + metrics_values(fold.test_metrics),
                    )

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_sweep_results(self, experiment_id: int) -> pl.DataFrame:
        """Return all sweep results for an experiment as a Polars DataFrame.

        Args:
            experiment_id: Primary key from the ``experiments`` table.

        Returns:
            DataFrame with one row per parameter set, sorted by
            ``mean_test_sharpe`` descending.
        """
        import polars as pl

        rows = self._conn.execute(
            "SELECT * FROM sweep_results WHERE experiment_id = ? ORDER BY mean_test_sharpe DESC",
            [experiment_id],
        ).fetchall()
        if not rows:
            return pl.DataFrame()
        cols = [d[0] for d in self._conn.description]
        return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])

    def load_fold_results(self, sweep_id: str) -> pl.DataFrame:
        """Return all fold results for a sweep as a Polars DataFrame."""
        import polars as pl

        rows = self._conn.execute(
            "SELECT * FROM fold_results WHERE sweep_id = ? ORDER BY fold_idx",
            [sweep_id],
        ).fetchall()
        if not rows:
            return pl.DataFrame()
        cols = [d[0] for d in self._conn.description]
        return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])

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

    def __enter__(self) -> AnalyticsStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"AnalyticsStore({self._conn!r})"
