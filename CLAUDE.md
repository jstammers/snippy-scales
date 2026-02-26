# CLAUDE.md — SnippyScales Agent & Contributor Guide

Quick-start reference for AI agents and human contributors.

---

## Repository layout

```
.
├── python/
│   ├── snippy_scales/          ← Python package (importable as snippy_scales)
│   │   ├── _constants.py       ← TRADING_DAYS_PER_YEAR = 252 (single source of truth)
│   │   ├── backtesting/        ← Execution layer — run a backtest, return a result
│   │   │   ├── domain.py       ← Trade, BacktestMetrics, BacktestResult
│   │   │   ├── engine.py       ← ExecutionEngine Protocol, RaptorExecutionEngine
│   │   │   ├── signals.py      ← SignalBundle, PositionInterpreter, SignFlipInterpreter
│   │   │   ├── allocation.py   ← VolTargetAllocator
│   │   │   ├── data.py         ← OhlcvArrays, normalize_ohlcv, to_numpy_ohlcv
│   │   │   ├── runners.py      ← BacktestRunner, BasketRunner (high-level helpers)
│   │   │   └── store.py        ← BacktestStore (DuckDB — single-run analytics)
│   │   ├── evaluation/         ← Analysis layer — walk-forward, param search, persistence
│   │   │   ├── results.py      ← FoldResult, SweepResult, EvaluationResult
│   │   │   ├── split.py        ← WalkForwardSplit, SplitFold
│   │   │   ├── sweep.py        ← ParameterGrid, RandomSearch, OptunaSearch
│   │   │   ├── runner.py       ← EvaluationRunner
│   │   │   ├── database.py     ← SQLiteStore (metadata), AnalyticsStore (DuckDB)
│   │   │   └── tearsheet.py    ← TearsheetGenerator
│   │   ├── research/           ← Feature engineering helpers
│   │   ├── strategies/         ← Concrete strategy implementations
│   │   ├── data/               ← Data ingestion (Databento)
│   │   └── cli/                ← Typer CLI (entry point: algo)
│   └── tests/
├── rust/                       ← Rust extension (algo-pyo3 → _algo_core)
├── docs/                       ← MkDocs documentation
├── justfile                    ← Dev workflow commands (see below)
└── pyproject.toml
```

---

## Module responsibility contract

**Never add walk-forward types, fold analysis, or sweep aggregation to `backtesting/`.**
**Never add execution logic or raptorbt imports to `evaluation/`.**

| Module | Owns | Must NOT contain |
|---|---|---|
| `backtesting/` | Execute a backtest; return `BacktestResult` | Fold/sweep logic, analysis aggregation |
| `evaluation/` | Walk-forward splits, param sweeps, persistence, tearsheets | Execution details |
| `research/` | Feature engineering helpers | Magic constants |
| `_constants.py` | `TRADING_DAYS_PER_YEAR = 252` | — |

---

## Dev commands

All commands use `just` (a Makefile replacement). Run `just` with no args to list all recipes.

```bash
# Setup
just setup          # uv sync + maturin develop + prek install
just dev            # rebuild Rust extension (run after any Rust change)

# Python
just test-py        # uv run pytest
just lint-py        # uv run ruff check . --fix
just fmt-py         # uv run ruff format .
just type-check     # uv run ty check

# Combined
just check-py       # lint + fmt + type-check + test (Python only)
just check          # everything (format, lint, type-check, test, Rust)

# Docs
just docs-serve     # mkdocs serve (live preview)
```

**Raw equivalents** (if `just` is unavailable):

```bash
uv run pytest                    # tests
uv run ruff check . --fix        # lint with auto-fix
uv run ruff format .             # format
uv run ty check                  # type-check
uv run maturin develop           # rebuild Rust extension
```

### Git hooks — MANDATORY

This repo uses **prek** to enforce git hooks. **Always install hooks before making any commit:**

```bash
just setup          # installs hooks automatically (prek install is part of setup)
# or directly:
prek install        # installs pre-commit and commit-msg hooks
```

**NEVER bypass hooks** with `--no-verify`, `--no-gpg-sign`, or any other flag.
If a hook fails, fix the underlying issue — do not skip it.

#### Conventional Commits

Every commit message **must** follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <short description>
```

Allowed types (enforced by `commit-msg` hook):

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code restructure (no behaviour change) |
| `revert` | Reverts a previous commit |
| `docs` | Documentation only |
| `style` | Formatting / whitespace |
| `test` | Adding or fixing tests |
| `build` | Build system / dependency changes |
| `ci` | CI/CD pipeline changes |
| `chore` | Maintenance tasks (not in changelog) |

Examples of **valid** commit messages:
```
feat(cli): add algo eval run-all command
fix(evaluation): handle empty fold list in SweepResult
test(backtesting): add edge-case tests for BacktestMetrics
```

### CI gates

All PRs must pass: `ruff check`, `ruff format --check`, `ty check`, `pytest`.

---

## Key patterns

### Running a backtest

```python
from snippy_scales.backtesting import make_config, run_single, InstrumentSpec

result = run_single(
    symbol="ES.c.0",
    timestamps=ts, open_prices=o, high_prices=h,
    low_prices=l, close_prices=c, volume=v,
    entries=entries, exits=exits,
    direction=1,        # 1 = long, -1 = short
    weight=1.0,
    config=make_config(initial_capital=1_000_000.0),   # NOT initial_cash
)
# Access metrics
result.metrics.sharpe_ratio
result.metrics.max_drawdown_pct    # NOT max_drawdown
result.metrics.total_return_pct    # NOT total_return
result.metrics.total_fees_paid     # NOT fees
```

### BacktestMetrics — 33 fields

`BacktestMetrics` is a frozen dataclass; **all 33 fields are required** when constructing
directly. Fields use `_pct` suffixes and full names:

```python
# Correct field names
metrics.max_drawdown_pct         metrics.total_return_pct
metrics.max_drawdown_duration    metrics.total_fees_paid
metrics.win_rate_pct             metrics.avg_trade_return_pct
```

See `backtesting/domain.py` or `docs/architecture/backtest_engine.md` for the full list.

### Shared DDL — avoid duplicating the 33 metrics columns

`backtesting/store.py` exports three public symbols for reuse:

```python
from snippy_scales.backtesting.store import METRICS_COLS, METRIC_NAMES, metrics_values
# METRICS_COLS  — SQL DDL fragment for all 33 columns
# METRIC_NAMES  — tuple of 33 attribute names (matches BacktestMetrics field order)
# metrics_values(metrics) → list[object]  — extract values in METRIC_NAMES order
```

`evaluation/database.py` imports these to build `fold_results` test metric columns.
Do not duplicate the DDL.

### Walk-forward evaluation

```python
from snippy_scales.evaluation import EvaluationRunner, ParameterGrid

runner = EvaluationRunner(
    n_splits=5,
    window="expanding",
    initial_capital=1_000_000.0,
    fees=0.001,
    db_path="data/metadata.db",               # SQLite — experiment registry
    analytics_db_path="data/analytics.duckdb", # DuckDB — sweep/fold metrics
)
grid = ParameterGrid({"fast": [10, 20], "slow": [40, 80]})
result = runner.evaluate(MyStrategy, bars, params=grid, symbol="ES.c.0")

result.best_params        # highest mean OOS Sharpe
result.summary_df()       # Polars DataFrame sorted by mean_test_sharpe DESC
```

### SweepResult std-dev

`SweepResult` exposes **both** mean and population std-dev (ddof=0) for four key metrics:

```python
sweep.mean_test_sharpe   sweep.std_test_sharpe
sweep.mean_test_return   sweep.std_test_return
sweep.mean_test_max_dd   sweep.std_test_max_dd
sweep.mean_test_sortino  sweep.std_test_sortino
```

---

## Database architecture

```
SQLite (metadata.db)            DuckDB (analytics.duckdb)
─────────────────────           ──────────────────────────────────────
experiments                     backtest_runs  (all 33 metrics per run)
  id, name, strategy_class  ←── sweep_results.experiment_id
  symbols, config, n_splits      sweep_results (mean+std, 4 key metrics)
  window_type, created_at        fold_results  (4 train + 33 test metrics)
                                 [future: trades, pnl series, signals]
```

**SQLite** = lightweight relational metadata. One row per experiment run.
**DuckDB** = wide columnar analytics. Use for sweeps, folds, backtest history.

```python
from snippy_scales.evaluation.database import SQLiteStore, AnalyticsStore
from snippy_scales.backtesting.store import BacktestStore

# Metadata
sqlite = SQLiteStore("data/metadata.db")          # Path | str
exp_id = sqlite.save_evaluation(result, config={...})

# Evaluation analytics
analytics = AnalyticsStore("data/analytics.duckdb")  # Path | str
analytics.save_evaluation_analytics(result, experiment_id=exp_id)
sweep_df = analytics.load_sweep_results(experiment_id=exp_id)
fold_df  = analytics.load_fold_results(sweep_id="<uuid>")

# Single-run backtests
store = BacktestStore("data/analytics.duckdb")   # can share the same DuckDB file
run_id = store.save_run(result, strategy_name="MyStrategy")
```

---

## Shared constant

```python
from snippy_scales._constants import TRADING_DAYS_PER_YEAR  # = 252
```

Import this whenever you need the annualisation factor. Do not hardcode `252`.
It is also re-exported from `backtesting.domain` and `backtesting.runner`.

---

## Fixtures & test conventions

- Tests live in `python/tests/`.
- In-memory DuckDB: `BacktestStore(":memory:")`, `AnalyticsStore(":memory:")`.
- In-memory SQLite: `SQLiteStore(":memory:")`.
- Generator fixtures must be typed `Generator[T, None, None]`, not `T`.
- `BacktestMetrics` requires all 33 fields — use a `_make_metrics(**overrides)` factory
  in test files to keep tests concise.

---

## Adding a new metric to BacktestMetrics

1. Add the field to `BacktestMetrics` in `backtesting/domain.py`.
2. Map it in `BacktestMetrics.from_raptorbt()`.
3. Add the SQL column to `METRICS_COLS` in `backtesting/store.py`.
4. Add the name to `METRIC_NAMES` in the same file (order must match `METRICS_COLS`).
5. `evaluation/database.py` inherits the new column automatically via `METRICS_COLS` / `metrics_values`.
6. Update `cli/eval.py`'s `_test_metrics()` helper to map the new column.
7. Run `just check-py`.
