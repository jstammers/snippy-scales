# CLAUDE.md — SnippyScales Agent & Contributor Guide

Quick-start reference for AI agents and human contributors.

---

## ⚠️ NEVER delete anything under `data/`

`data/` (`data/raw/`, `data/catalog/`, `data/derived/`, `data/cache/`,
`data/universe/`, and their contents) is **git-ignored** — nothing under it is tracked, so nothing
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
│   │   ├── execution/          ← NautilusTrader execution — the only trading user of nautilus_trader
│   │   │   ├── strategy.py     ← SignalStrategy, compute_targets (research targets → orders)
│   │   │   ├── backtest.py     ← MarketData, load_market_data, run_backtest, Costs, BacktestRun
│   │   │   └── vector.py       ← numba screening engine: same run_backtest signature, Nautilus-parity tested
│   │   ├── evaluation/         ← Walk-forward studies and selection-bias statistics
│   │   │   ├── walk_forward.py ← make_folds, WalkForward, run_study (Optuna), summarize_study, write_tearsheet
│   │   │   ├── statistics.py   ← DSR, PSR, PBO, MinTRL
│   │   │   └── scoring.py      ← CRPS, PIT, calibration (probabilistic forecasts)
│   │   ├── research/           ← Feature engineering helpers
│   │   ├── strategies/         ← Concrete strategy implementations
│   │   ├── data/               ← Market data in the NautilusTrader catalog (data/catalog/)
│   │   │   ├── catalog.py      ← load_bars, write_bars, frame_to_bars, missing_ranges, instruments
│   │   │   ├── databento.py    ← DatabentoSync — incremental sync via Nautilus's Databento client (incl. ES.FUT parents)
│   │   │   ├── futures.py      ← outrights, RollRule, RollSchedule, roll_schedule, back_adjust (ratio)
│   │   │   ├── alpaca.py       ← AlpacaLoader + RateLimiter — the one Alpaca module
│   │   │   ├── config.py       ← IngestConfig, AlpacaConfig, AssetClassConfig, load_config
│   │   │   ├── universe.py     ← sp500_ever_members (Wikipedia-sourced membership history)
│   │   │   ├── backfill.py     ← helpers behind `algo data backfill-sp500`
│   │   │   └── paths.py        ← CATALOG_DIR, RAW_DIR (legacy, read-only), MANIFEST_DIR
│   │   ├── reporting/          ← Strategy research reports (marimo template → reports/<name>/)
│   │   │   ├── config.py       ← ReportConfig (config.yaml), fingerprint
│   │   │   ├── pipeline.py     ← screening (vector) → Nautilus validation → full backtest, checks
│   │   │   └── scaffold.py     ← new_report, export_html (marimo export)
│   │   └── cli/                ← Typer CLI (entry point: algo) — see `algo data --help`
│   │       ├── data.py         ← sync, coverage, consolidate, update-universe, backfill-sp500
│   │       └── report.py       ← algo report new | build | edit | list
│   └── tests/
├── scripts/
│   └── migrate_raw_to_catalog.py ← one-off copy of data/raw/ into the catalog (read-only on raw)
├── reports/                    ← Report instances: GIT-IGNORED (proprietary results)
│   └── templates/              ← tracked: strategy_report.py (marimo) + configs/*.yaml
├── docs/                       ← MkDocs documentation (docs/strategies/<name>.md = report theory)
├── configs/                    ← Sync configs (databento.yaml, databento_es_trades.yaml,
│                                  alpaca.yaml, alpaca_sp500_1m.yaml)
├── justfile                    ← Dev workflow commands (see below)
└── pyproject.toml
```

---

## Module responsibility contract

**Never add walk-forward types, fold analysis, or sweep aggregation to `execution/`.**
**Never add order or engine logic to `evaluation/`** — it calls `execution.run_backtest`.
**`strategies/` and `research/` never import `nautilus_trader`.**

| Module | Owns | Must NOT contain |
|---|---|---|
| `strategies/` | `generate_signals(bars) → target positions` (units of equity, causal) | Nautilus imports |
| `execution/` | Nautilus strategy adapter, `MarketData`, `run_backtest` | Fold/sweep logic, analysis aggregation |
| `evaluation/` | Folds, Optuna studies, DSR/PSR/PBO, tearsheets | Execution details |
| `research/` | Feature engineering helpers | Magic constants |
| `reporting/` | Report configs, the screen → validate → backtest pipeline, scaffolding | Strategy or engine logic (it calls `evaluation/` and `execution/`) |
| `_constants.py` | `TRADING_DAYS_PER_YEAR = 252` | — |

---

## Dev commands

All commands use `just` (a Makefile replacement). Run `just` with no args to list all recipes.

```bash
# Setup
just setup          # uv sync --all-extras + prek install

# Python
just test-py        # uv run pytest
just lint-py        # uv run ruff check . --fix
just fmt-py         # uv run ruff format .
just type-check     # uv run ty check

# Combined
just check-py       # lint + fmt + type-check + test (Python only)
just check          # everything (format, lint, type-check, test)

# Docs
just docs-serve     # mkdocs serve (live preview)
```

**Raw equivalents** (if `just` is unavailable):

```bash
uv run pytest                    # tests
uv run ruff check . --fix        # lint with auto-fix
uv run ruff format .             # format
uv run ty check                  # type-check
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
fix(evaluation): handle a study with no completed trials
test(execution): cover rebalance-band edge cases
```

### CI gates

All PRs must pass: `ruff check`, `ruff format --check`, `ty check`, `pytest`.

---

## Key patterns

### Market data (NautilusTrader catalog)

```python
from snippy_scales.data.catalog import load_bars, open_catalog

bars = load_bars("ES.c.0", "ohlcv-1d")   # polars: ts_event (open), ts_init (close), OHLCV, symbol
catalog = open_catalog()                 # nautilus_trader ParquetDataCatalog at data/catalog/
```

Write to the catalog only through `data.catalog.write_bars` / `DatabentoSync` /
`AlpacaLoader`: the catalog rejects overlapping writes, and these helpers write
only data outside existing coverage. `ts_init` must be the bar **close**.
Fetch data with `algo data sync <config>`.

### Running a backtest

```python
from snippy_scales.execution import Costs, load_market_data, run_backtest
from snippy_scales.strategies import TrendFollowing

data = load_market_data("AAPL", "ohlcv-1d")   # once; reuse across runs
run = run_backtest(
    TrendFollowing(), data,
    start="2018-01-01",                 # signals warm up on earlier bars
    costs=Costs(commission_rate=0.0005),
    rebalance_band=0.02,                # skip trades < 2% of equity
)
run.sharpe                              # annualised, daily mark-to-market returns
run.stats                               # all Nautilus statistics, flat dict
run.equity                              # polars: ts, equity (per bar close)
run.rejections                          # non-empty → orders were denied/rejected
```

Futures: `-s ES.FUT` / `load_market_data("ES.FUT", roll=RollRule("volume"))` trades real
outrights through a `RollSchedule`; signals use a backward-**ratio**-adjusted splice, so
strategies on rolled symbols must be scale-invariant (returns, vol, MA signs).
`ES.c.0` (Databento continuous) is unadjusted and untradable — prefer `ES.FUT`.

Targets are in units of equity; `SignalStrategy` holds
`target × weight × equity / (close × multiplier)` contracts, filled at the bar close.
Every strategy must pass `tests/test_strategy_causality.py` (add new ones to its lists)
and be registered in `cli/_common.py:registry()` to be usable from the CLI.
Targets for a whole universe come from `execution.panel_targets` (array `Targets`, no
per-symbol dicts). For large universes, override `generate_signals_panel` with
`.over("symbol")` window expressions and add the strategy to
`tests/test_strategy_panel.py`. Load universes with one `load_market_data([...])` call:
it reads every symbol in one catalog query.
Synthetic data for tests: `tests/conftest.py` (`synthetic_data`, the `backtest` fixture).

### Metrics

There is no fixed metrics schema. Statistics are Nautilus's (`BacktestRun.stats_returns`,
`stats_pnls`, `stats_general`); add a new one as a Nautilus `PortfolioStatistic`, not a
column. The Sharpe key is `execution.SHARPE` (`"Sharpe Ratio (252 days)"`).

### Walk-forward evaluation

```python
from snippy_scales.evaluation import WalkForward, make_folds, run_study, summarize_study

wf = WalkForward(data, make_folds(data.closes(), n_splits=5), costs=Costs(commission_rate=1e-4))
study = run_study(
    TrendFollowing, wf,
    space={"fast_period": [10, 20, 40], "slow_period": {"low": 60, "high": 200}},
    sampler="tpe", n_trials=50,         # or sampler="grid" with list-only spaces
)
report = summarize_study(study)
report.best_params, report.sharpe, report.dsr, report.pbo
```

Each trial's value is its mean fold Sharpe; the report is computed on the best trial's
stitched out-of-sample returns and deflated for the number of trials.
`WalkForward(..., engine="vector")` (CLI `--engine vector`) screens on the numba kernel in
`execution/vector.py` (~50× faster end to end); validate kept trials on Nautilus.
CLI: `algo eval run | sweep | run-all | list | report | tearsheet`, `algo backtest run`.

### Strategy reports

```bash
algo report new tsmom-sp500 --from reports/templates/configs/tsmom_sp500.yaml
algo report build tsmom-sp500     # marimo export → reports/tsmom-sp500/report.html
```

One tracked marimo template (`reports/templates/strategy_report.py`); instances in
`reports/<name>/` are git-ignored, never commit them. Keep computations in
`reporting/pipeline.py` (tested) and only layout in the notebook; run
`uv run marimo check reports/templates/strategy_report.py` after editing it. A new
strategy needs `docs/strategies/<registry-name>.md` (rendered as the report's theory).
Guide: `docs/guides/strategy_reports.md`.

Multi-asset strategies see frames aligned on the **union** of dates (null before a
symbol's first bar / after its last), so ragged index universes work. A symbol whose
data ends early is flattened on its last bar (`compute_targets`).

## Experiment storage

```
data/derived/experiments/
├── optuna.db                      Optuna studies (SQLite): params, values, fold stats
│                                  as trial user attrs, study config as study user attrs
└── <study>/trial_<n>.parquet      each trial's stitched OOS daily returns (ts, return)
```

Query with `optuna.load_study(...).trials_dataframe()` (→ polars/DuckDB for ad-hoc SQL)
or `optuna-dashboard sqlite:///data/derived/experiments/optuna.db`. Tests pass
`storage=None` (in-memory) and a `tmp_path` `returns_dir`.

## Shared constant

```python
from snippy_scales._constants import TRADING_DAYS_PER_YEAR  # = 252
```

Import this whenever you need the annualisation factor. Do not hardcode `252`.

---

## Fixtures & test conventions

- Tests live in `python/tests/`; shared helpers in `tests/conftest.py`
  (`synthetic_future`, `daily_bars`, `synthetic_data`, the `backtest` fixture).
- Generator fixtures must be typed `Generator[T, None, None]`, not `T`.
- Never write studies or returns under the real `data/` in tests: use `storage=None`
  and `tmp_path` (CLI tests monkeypatch `cli.eval.EXPERIMENTS_DIR`).

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
