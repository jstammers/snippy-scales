"""Persistence layer for evaluation results.

All data is stored in a single DuckDB database via :class:`AnalyticsStore`:

* ``experiments`` — one row per evaluation run (name, strategy, config).
* ``sweep_results`` — one row per (experiment, parameter set) with mean and
  population std-dev for 4 key test metrics.
* ``fold_results`` — one row per (sweep, fold) with 4 training-window metrics
  and all 33 test-window
  :class:`~snippy_scales.backtesting.domain.BacktestMetrics` fields.

Typical usage::

    from snippy_scales.evaluation.database import AnalyticsStore

    store = AnalyticsStore("data/analytics.duckdb")
    experiment_id = store.save_evaluation(eval_result, config={...})
    store.save_evaluation_analytics(eval_result, experiment_id)

    # Query
    df = store.load_sweep_results(experiment_id)
"""

from __future__ import annotations

import json
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
# DDL
# ---------------------------------------------------------------------------

_DDL_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS experiments (
    id              BIGINT PRIMARY KEY,
    name            TEXT   NOT NULL,
    strategy_class  TEXT   NOT NULL,
    symbols_json    TEXT   NOT NULL,
    n_splits        INTEGER NOT NULL,
    window_type     TEXT   NOT NULL,
    created_at      TEXT   NOT NULL,
    config_json     TEXT   NOT NULL
)"""

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
    """DuckDB store for all evaluation data: experiments, sweeps, and folds.

    Stores all persistent evaluation data in a single DuckDB file:

    * ``experiments`` — lightweight registry of evaluation runs.
    * ``sweep_results`` — one row per (experiment, parameter set) with
      mean and population std-dev for 4 key test metrics.
    * ``fold_results`` — one row per (sweep, fold) with 4 training-window
      metrics and all 33 test-window metrics.

    Args:
        db_path: Path to the DuckDB file.  Pass ``\":memory:\"`` for an
            in-process, in-memory database (useful for testing).

    Example::

        store = AnalyticsStore("data/analytics.duckdb")
        exp_id = store.save_evaluation(eval_result, config={...})
        store.save_evaluation_analytics(eval_result, experiment_id=exp_id)
        df = store.load_sweep_results(experiment_id=exp_id)
    """

    def __init__(self, db_path: Path | str = "data/analytics.duckdb") -> None:
        import duckdb  # noqa: PLC0415 — optional dep, imported lazily

        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: _duckdb.DuckDBPyConnection = duckdb.connect(path)
        self.create_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create all tables if absent."""
        for ddl in (_DDL_EXPERIMENTS, _DDL_SWEEP_RESULTS, _DDL_FOLD_RESULTS):
            self._conn.execute(ddl)

    # ------------------------------------------------------------------
    # Write — experiments
    # ------------------------------------------------------------------

    def save_evaluation(
        self,
        result: EvaluationResult,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Register an evaluation run in the ``experiments`` table.

        Args:
            result: The evaluation result to register.
            config: Optional dict of runner configuration (fees, slippage, etc.).

        Returns:
            The ``experiment_id`` (primary key) assigned to this run.
        """
        config = config or {}
        created_at = datetime.now(tz=UTC).isoformat()
        next_id = int(
            self._conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM experiments").fetchone()[0]  # type: ignore[index]
        )
        self._conn.execute(
            """
            INSERT INTO experiments
                (id, name, strategy_class, symbols_json, n_splits, window_type,
                 created_at, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                next_id,
                result.experiment_name,
                result.strategy_class,
                json.dumps(result.symbols),
                config.get("n_splits", 0),
                config.get("window", ""),
                created_at,
                json.dumps(config),
            ],
        )
        return next_id

    # ------------------------------------------------------------------
    # Write — sweep + fold analytics
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
            experiment_id: Primary key from the ``experiments`` table
                (use ``0`` if not linked to an experiment row).
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

    def load_experiments(self) -> pl.DataFrame:
        """Return all experiments as a Polars DataFrame, newest first."""
        import polars as pl

        rows = self._conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
        if not rows:
            return pl.DataFrame()
        cols = [d[0] for d in self._conn.description]
        return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])

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
