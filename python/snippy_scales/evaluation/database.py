"""SQLite persistence layer for evaluation results.

All evaluation runs, sweep parameter sets, and per-fold metrics are stored in
a single SQLite database file.  The schema uses three tables with foreign keys:

    experiments → sweep_results → fold_results

Usage::

    store = SQLiteStore("results.db")
    exp_id = store.save_evaluation(eval_result)

    # Offline analysis
    df = store.load_sweep_results(exp_id)
    print(df.sort("mean_test_sharpe", descending=True))

    # Raw SQL
    df = store.query("SELECT * FROM fold_results WHERE test_sharpe > 1.5")
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

    from snippy_scales.evaluation.results import EvaluationResult

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
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

CREATE TABLE IF NOT EXISTS sweep_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       INTEGER NOT NULL REFERENCES experiments(id),
    params_json         TEXT    NOT NULL,
    mean_test_sharpe    REAL,
    mean_test_return    REAL,
    mean_test_max_dd    REAL,
    mean_test_sortino   REAL,
    n_folds             INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS fold_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id            INTEGER NOT NULL REFERENCES sweep_results(id),
    fold_idx            INTEGER NOT NULL,
    train_start         TEXT,
    train_end           TEXT,
    test_start          TEXT,
    test_end            TEXT,
    -- training window metrics
    train_return_pct    REAL,
    train_sharpe        REAL,
    train_max_dd        REAL,
    train_trades        INTEGER,
    -- test window metrics
    test_return_pct     REAL,
    test_sharpe         REAL,
    test_sortino        REAL,
    test_calmar         REAL,
    test_max_dd         REAL,
    test_win_rate       REAL,
    test_profit_factor  REAL,
    test_trades         INTEGER,
    test_expectancy     REAL,
    -- equity curves (JSON arrays)
    train_equity_json   TEXT,
    test_equity_json    TEXT
);
"""


# ── SQLiteStore ───────────────────────────────────────────────────────────────


class SQLiteStore:
    """Persist and retrieve evaluation results in a local SQLite database.

    The database file is created automatically if it does not exist.

    Args:
        db_path: Path to the SQLite file (e.g. ``"results.db"`` or
            ``Path("data/eval.db")``).

    Example::

        store = SQLiteStore("results.db")
        experiment_id = store.save_evaluation(eval_result)
        df = store.load_sweep_results(experiment_id)
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_evaluation(
        self,
        result: EvaluationResult,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Persist a complete :class:`~snippy_scales.evaluation.results.EvaluationResult`.

        Args:
            result: The evaluation result to store.
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

            for sweep in result.sweep_results:
                cur = conn.execute(
                    """
                    INSERT INTO sweep_results
                        (experiment_id, params_json, mean_test_sharpe, mean_test_return,
                         mean_test_max_dd, mean_test_sortino, n_folds)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exp_id,
                        json.dumps(sweep.params),
                        sweep.mean_test_sharpe,
                        sweep.mean_test_return,
                        sweep.mean_test_max_dd,
                        sweep.mean_test_sortino,
                        sweep.n_folds,
                    ),
                )
                sweep_id = cur.lastrowid

                for fold in sweep.folds:
                    conn.execute(
                        """
                        INSERT INTO fold_results (
                            sweep_id, fold_idx,
                            train_start, train_end, test_start, test_end,
                            train_return_pct, train_sharpe, train_max_dd, train_trades,
                            test_return_pct, test_sharpe, test_sortino, test_calmar,
                            test_max_dd, test_win_rate, test_profit_factor,
                            test_trades, test_expectancy,
                            train_equity_json, test_equity_json
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?,
                            ?, ?,
                            ?, ?
                        )
                        """,
                        (
                            sweep_id,
                            fold.fold_idx,
                            fold.train_start,
                            fold.train_end,
                            fold.test_start,
                            fold.test_end,
                            # train metrics
                            fold.train_metrics.total_return_pct,
                            fold.train_metrics.sharpe_ratio,
                            fold.train_metrics.max_drawdown_pct,
                            fold.train_metrics.total_trades,
                            # test metrics
                            fold.test_metrics.total_return_pct,
                            fold.test_metrics.sharpe_ratio,
                            fold.test_metrics.sortino_ratio,
                            fold.test_metrics.calmar_ratio,
                            fold.test_metrics.max_drawdown_pct,
                            fold.test_metrics.win_rate_pct,
                            fold.test_metrics.profit_factor,
                            fold.test_metrics.total_trades,
                            fold.test_metrics.expectancy,
                            # equity curves
                            json.dumps(fold.train_equity_curve.tolist()),
                            json.dumps(fold.test_equity_curve.tolist()),
                        ),
                    )

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

    def load_sweep_results(self, experiment_id: int) -> pl.DataFrame:
        """Return all sweep results for an experiment as a Polars DataFrame.

        Args:
            experiment_id: Primary key from the ``experiments`` table.

        Returns:
            DataFrame with one row per parameter set, sorted by
            ``mean_test_sharpe`` descending.
        """
        import polars as pl

        rows = self._fetchall(
            "SELECT * FROM sweep_results WHERE experiment_id = ? ORDER BY mean_test_sharpe DESC",
            (experiment_id,),
        )
        if not rows:
            return pl.DataFrame()
        df = pl.DataFrame(rows)
        # Parse params_json into individual columns
        params_dicts = [json.loads(r) for r in df["params_json"].to_list()]
        if params_dicts:
            params_df = pl.DataFrame(params_dicts)
            df = pl.concat([df, params_df], how="horizontal")
        return df

    def load_fold_results(self, sweep_id: int) -> pl.DataFrame:
        """Return all fold results for a sweep as a Polars DataFrame."""
        import polars as pl

        rows = self._fetchall(
            "SELECT * FROM fold_results WHERE sweep_id = ? ORDER BY fold_idx",
            (sweep_id,),
        )
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

        Example::

            df = store.query(
                "SELECT e.name, s.params_json, s.mean_test_sharpe "
                "FROM experiments e JOIN sweep_results s ON s.experiment_id = e.id "
                "WHERE s.mean_test_sharpe > 1.0"
            )
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
