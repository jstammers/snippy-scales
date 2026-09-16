# CLAUDE.md — SnippyScales Agent & Contributor Guide

Quick-start reference for AI agents and human contributors.

---

## ⚠️ NEVER delete anything under `data/`

`data/` (`data/raw/`, `data/derived/`, `data/cache/`, `data/universe/`, and
their contents) is **git-ignored** — nothing under it is tracked, so nothing
under it can be recovered with `git checkout`/`git reflog`/etc. once it's
gone. It routinely holds locally-ingested market data that cost real money
(Databento credit) or real wall-clock time (an hours-long Alpaca backfill) to
build, and once deleted it is gone unless the user has an independent backup.

**Never run `rm -rf data/`, `rm -rf data/raw/`, `shutil.rmtree(...)` on any
`data/` subpath, or any other recursive delete under `data/` — not even to
"clean up" a test artifact, not even if it looks empty, not even inside
`/tmp`-adjacent throwaway logic that got a real path by mistake.** If you
create a scratch file under `data/` for a test or a smoke check, delete that
**exact file**, never the directory. If a whole directory genuinely needs
clearing, list its contents first, confirm with the user what you're about
to remove, and only delete the specific paths you created yourself.

This rule exists because an agent working in this repo once ran `rm -rf
data/` mid-session — while another process was live-writing a real Alpaca
backfill into the same path — and destroyed pre-existing Databento futures
history in the process. Treat every path under `data/` as irreplaceable
unless you personally created it in the same command that's about to delete
it.

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
│   │   ├── data/               ← Data ingestion (Databento futures + Alpaca equities)
│   │   │   ├── config.py       ← IngestConfig, AlpacaConfig, AssetClassConfig, load_config
│   │   │   ├── ingest.py       ← upsert_bars (provider-agnostic), load_bars, ingest_from_config
│   │   │   ├── schema.py       ← BAR_SCHEMA_COLUMNS, conform_bars — shared Parquet layout
│   │   │   ├── providers/      ← BarProvider Protocol + DatabentoProvider, AlpacaProvider
│   │   │   ├── ratelimit.py    ← RateLimiter (used by AlpacaProvider)
│   │   │   ├── batch.py        ← run_batch_job — shared Databento Batch API submit/poll/download
│   │   │   ├── universe.py     ← sp500_ever_members (Wikipedia-sourced membership history)
│   │   │   ├── backfill.py     ← helpers behind `algo data backfill-sp500`
│   │   │   └── tick.py         ← Event-level (Databento-only) ingestion
│   │   └── cli/                ← Typer CLI (entry point: algo) — see `algo data --help`
│   │       └── data.py         ← ingest, ingest-config, coverage, update-universe, backfill-sp500
│   └── tests/
├── rust/                       ← Rust extension (algo-pyo3 → _algo_core)
├── docs/                       ← MkDocs documentation
├── configs/                    ← Ingestion configs (databento.yaml, databento_es_trades.yaml,
│                                  alpaca.yaml, alpaca_sp500_1m.yaml)
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

<!-- code-graph-mcp:begin v2 -->
## Code Graph (repo-wide AST index)

AST + FTS + vector index of the whole repo — prefer over multi-round Grep/Read for
structural queries (LSP only sees open files; this sees everything). Fastest path = Bash CLI:

| Intent | Command |
|--------|---------|
| Who calls X / what X calls | `code-graph-mcp callgraph X` |
| Impact before editing a fn | `code-graph-mcp impact X` |
| Unfamiliar dir / module | `code-graph-mcp overview <dir>` |
| Symbol source / signature | `code-graph-mcp show X` |
| Concept search (no exact name) | `code-graph-mcp search "…"` (vector: MCP `semantic_code_search`) |
| grep + AST context | `code-graph-mcp grep "pat" [paths] [-t lang] [-g glob] [-c]` |

Not on PATH? A plugin-only install keeps its own copy — same commands, run
`~/.cache/code-graph/bin/code-graph-mcp` (or `npm i -g @sdsrs/code-graph` once).

Still use Grep for literal strings/regex in non-code files; still Read files you'll edit.
Full command + MCP-tool table: `.claude/plugin_code_graph_mcp.md`
<!-- code-graph-mcp:end -->
