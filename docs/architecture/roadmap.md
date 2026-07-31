# Status Review & Development Roadmap

A review of the repository as it stands, and a phased plan for extending the
**research** and **execution** layers.

Baseline at time of writing: commit `55da044`, 184 tests passing in ~21 s,
~9 000 LOC across Python and Rust, CI green on `ruff` / `ty` / `pytest`.

---

## 1. Where the repo stands

### What is solid

| Area | State |
|---|---|
| Evaluation scaffolding | `WalkForwardSplit`, `ParameterGrid` / `RandomSearch` / `OptunaSearch`, `EvaluationRunner`, DuckDB persistence, quantstats tearsheets — all working and tested |
| Domain types | `BacktestMetrics` (33 fields), `Trade`, `BacktestResult` are frozen, typed, and engine-agnostic |
| Data ingestion | Databento config-driven ingest with genuine incremental upsert semantics |
| Tooling | `just` recipes, prek hooks, conventional commits, git-cliff changelog, 4 CI workflows, mkdocs |
| Module discipline | `CLAUDE.md` states a clear contract between `backtesting/` and `evaluation/`, and the code honours it |

### The central structural fact

**The Rust layer is dead code.** `algo-core` contains `BacktestEngine`,
`Portfolio`, `FillModel`, `Order`, `RiskCheck` and market types — and nothing
in Python imports any of it. The only reference is a TODO:

```python
# python/snippy_scales/cli/backtest.py:22
# TODO: load strategy, call Rust engine via _algo_core
```

`algo-pyo3` exposes exactly two things (`run_vectorised`, some stats helpers),
and Python never calls either. All actual execution runs through the external
`raptorbt` package.

This matters because `README.md` and `docs/architecture/overview.md` both
present Rust as the backtest and execution engine:

> | Backtesting Engine | Rust (`algo-core`) | Event-driven, fast, realistic |
> | Execution Engine | Rust (`algo-core`) | Order routing, risk checks |

Neither is true today. The first decision this roadmap has to force is
**which of the two engines is the real one** — see §4.

---

## 2. Execution layer — findings

### 2.1 Position magnitude is silently discarded (highest impact)

Every strategy in the repo computes a vol-targeted *magnitude*:

```python
# strategies/trend.py
vol_scalar = (self.vol_target / realised_vol.clip(lower_bound=1e-6)).clip(upper_bound=5.0)
positions = raw_signal * vol_scalar        # e.g. 0.83, -1.4, 0.0
```

`SignFlipInterpreter` then reduces that to four boolean arrays:

```python
is_long = positions > 0.0                  # magnitude gone
```

Sizing collapses to a single static `weight` float per leg. So the vol
targeting in `TrendFollowing`, `TimeSeriesMomentum`, `MeanReversion` and
`CrossSectionalMomentum` has **no effect on the backtest whatsoever** — it only
changes the sign pattern via the clip at zero. A strategy tuned on `vol_target`
is tuning a parameter that cannot move the result except through numerical
edge cases. `ParameterGrid({"vol_target": [...]})` sweeps are near-noise.

This is the single biggest gap in the execution layer, and it caps what the
research layer can honestly measure.

### 2.2 Sizing uses lookahead

`VolTargetAllocator.weight()` estimates volatility from the **entire** close
array it is given:

```python
daily_ret = np.diff(close) / ...
avg_vol = float(np.nanstd(daily_ret)) * (TRADING_DAYS_PER_YEAR**0.5)
```

Inside a walk-forward test fold, that array *is* the test window. The position
size for bar 0 is therefore derived from bars 0..N of the same window. Sizing
is fitted on the future it is being evaluated against.

### 2.3 Same-bar fills

Positions at bar `t` are computed from `close[t]`, and `make_config` defaults
to `upon_bar_close=True`, so the fill also happens at `close[t]`. There is no
signal lag anywhere in the pipeline. Any strategy that uses the current bar's
close in its signal is trading on information it would not have had.

### 2.4 The `ExecutionEngine` abstraction is half-wired

`BacktestRunner` accepts an `engine` argument, stores it, and never uses it:

```python
self._engine: ExecutionEngine = engine if engine is not None else _DEFAULT_ENGINE
...
return run_long_short(...)      # bypasses self._engine entirely
```

Passing a custom engine to `BacktestRunner` silently does nothing.
`BasketRunner` does honour `self._engine`. Also, `BasketRunner` ignores
`VolTargetAllocator` completely and hardcodes `weight = 1.0 / n_assets`.

### 2.5 No instrument model

`InstrumentSpec` carries OHLCV, signals, direction and weight — no contract
multiplier, tick size, currency, margin requirement, or trading calendar.
Combined with continuous-contract ingestion (`ES.c.0`), this means:

- No roll handling and no back-adjustment — the price series has roll gaps in
  it, and those gaps are being counted as PnL.
- PnL is in abstract "price units × weight", not contract PnL.
- Costs are a flat fraction of notional. No bid/ask spread, no market impact,
  no per-contract commission, no financing or borrow.

### 2.6 No execution path beyond backtest

There is no paper-trading or live order path. `RiskCheck` in Rust is the only
risk code and it is both unused and incomplete — it takes a `_portfolio`
argument it never reads, and `RiskViolation::PositionLimitBreached` is never
constructed.

### 2.7 Persistence gaps

Only aggregate metrics are stored. Trades, per-bar positions, and equity curves
are computed and then dropped. `CLAUDE.md` already marks these as
`[future: trades, pnl series, signals]`. Without them there is no turnover
analysis, no attribution, and no way to audit a historical run.

---

## 3. Research layer — findings

### 3.1 `research/` is 32 lines

Three functions: `realised_vol`, `zscore`, `carry`. `carry` is not used by any
strategy and has no test. There is no feature store, no cross-sectional
utilities, no data-quality validation, and no point-in-time universe handling.

### 3.2 Parameters are selected on the test set

```python
# evaluation/results.py
@property
def best_result(self) -> SweepResult:
    return max(self.sweep_results, key=lambda r: r.mean_test_sharpe)
```

`best_params` is the argmax over the parameter grid of **out-of-sample**
Sharpe. `train_metrics` is computed for every fold, persisted, and never used
for selection. The headline "OOS Sharpe" is therefore the maximum of a grid
evaluated on the test set — an in-sample statistic wearing an out-of-sample
label, and biased upward by roughly the width of the grid search.

Proper anchored walk-forward selects per fold on train, then evaluates *that*
choice on that fold's test window. The data structures already support this
(`FoldResult` holds both), only the selection logic is missing.

### 3.3 Test folds start cold

`WalkForwardSplit` slices train and test as disjoint index ranges. A test fold
gets no warmup prefix, so `CrossSectionalMomentum(lookback=252, skip_recent=21)`
on a 200-bar test window returns all zeros — `n <= warmup` short-circuits.
The `gap` parameter opens a hole between train and test but does not supply
warmup; the docstring's claim that it avoids "lookahead from features that
require a warmup period" describes the wrong mechanism.

### 3.4 Failures are swallowed

Both fold loops do:

```python
except Exception:
    logger.warning("Fold %d failed for params %s", ..., exc_info=True)
```

A parameter set whose folds all fail yields `SweepResult(folds=[])`, whose
aggregate properties return `nan`. Those `nan`s then flow into
`max(..., key=lambda r: r.mean_test_sharpe)`, where comparison behaviour
depends on iteration order. A sweep can report a "best" parameter set that
never successfully ran, and nothing surfaces it.

### 3.5 No statistical rigour on results

No deflated Sharpe or multiple-testing correction, despite the framework's
whole purpose being to search parameter grids. No purging or embargo around
fold boundaries. No fold-level confidence intervals. No turnover or capacity
statistics. No benchmark-relative metrics beyond what quantstats renders.

### 3.6 Doc drift

`CLAUDE.md` and `docs/architecture/backtest_engine.md` both document a
`SQLiteStore` class and a two-database architecture. Commit `b34b138`
consolidated everything into DuckDB and deleted `SQLiteStore`; the docs still
instruct readers to import it. Separately, `CLAUDE.md` mandates importing
`TRADING_DAYS_PER_YEAR` and never hardcoding `252`, but `252**0.5` appears
literally in four strategy files.

---

## 4. The decision that gates everything

Before building further, pick one:

**Option A — raptorbt is the engine; delete or archive `algo-core`.**
Fastest path to research throughput. Correct the README and architecture docs
to describe reality. Rust stays only for hot loops that profiling justifies.
Accepts raptorbt's model of the world (static per-leg weights, boolean signals),
which caps §2.1 and §2.5 permanently.

**Option B — `algo-core` becomes the engine; raptorbt becomes the reference.**
Much more work, but it is the only route to continuous position sizing,
per-instrument contract specs, realistic fills, and a shared backtest/live code
path. The `ExecutionEngine` Protocol already exists as the seam, and
`algo-core` already has `Portfolio`, `Order`, `Fill` and `BacktestEngine`
skeletons to build on.

The plan below assumes **Option B**, staged so that raptorbt keeps working
throughout and the Rust engine is validated against it before anything
switches over. Phases 0, 1, 4 and 6 are worth doing under either option.

---

## 5. Plan

### Phase 0 — Correctness and honesty (small, do first)

Nothing here is a feature; all of it is currently producing wrong numbers or
wrong documentation.

| # | Change | Files |
|---|---|---|
| 0.1 | Make `BacktestRunner` actually use `self._engine` | `backtesting/runners.py` |
| 0.2 | Add explicit signal lag (`shift(1)` by default, configurable) so signals computed at bar `t` fill at `t+1` | `backtesting/runners.py`, `backtesting/signals.py` |
| 0.3 | Make `VolTargetAllocator` causal — expanding or rolling window, never full-series | `backtesting/allocation.py` |
| 0.4 | Replace hardcoded `252**0.5` with `TRADING_DAYS_PER_YEAR` | 4 files in `strategies/` |
| 0.5 | Surface fold failures: count them on `SweepResult`, exclude all-failed sets from `best_result`, raise if every fold of every set failed | `evaluation/results.py`, `evaluation/runner.py` |
| 0.6 | Fix `SQLiteStore` references; correct README/overview to describe raptorbt as the current engine | `CLAUDE.md`, `docs/`, `README.md` |

**Acceptance:** existing 184 tests still pass; new regression tests for lag,
causal sizing, and failed-fold accounting.

### Phase 1 — Honest walk-forward

The evaluation framework's headline number is currently biased. Fix that before
using it to make research decisions.

- **1.1 Per-fold selection.** Add `selection_metric` and a
  `select_on="train" | "test"` mode to `EvaluationRunner`, defaulting to
  `"train"`. For each fold: rank parameter sets by train metric, evaluate the
  winner on that fold's test window, and report the stitched result. Add
  `EvaluationResult.walk_forward_result()` distinct from today's
  `best_result`.
- **1.2 Warmup prefix.** Give `WalkForwardSplit` a `warmup` parameter that
  prepends `warmup` bars to each test slice for signal computation, and mark
  those bars as non-tradeable so they contribute signals but not PnL. Requires
  a `tradeable_mask` on `InstrumentSpec` or equivalent slicing of the result.
- **1.3 Purge and embargo.** Rename/extend `gap` into proper purging
  (drop train bars whose label window overlaps test) plus an embargo after the
  test window.
- **1.4 Multiple-testing correction.** Deflated Sharpe ratio and
  probabilistic Sharpe on `SweepResult`, parameterised by the number of trials
  actually run. This is the number that should be quoted, not `mean_test_sharpe`.

**Acceptance:** a synthetic random-signal strategy swept over a 50-point grid
reports a deflated Sharpe near zero, while `mean_test_sharpe` of the argmax is
visibly positive — demonstrating the bias the correction removes.

### Phase 2 — Continuous position sizing (the §2.1 fix)

- **2.1 Widen the signal contract.** Introduce a `TargetPositions` type
  carrying the signed float series alongside the boolean bundle.
  `PositionInterpreter` gains an implementation that preserves magnitude.
- **2.2 Engine support.** raptorbt takes static weights only, so this is where
  `algo-core` earns its place: extend `BacktestEngine` to consume a per-bar
  target-position series and emit the implied rebalancing orders through
  `FillModel`.
- **2.3 Expose through PyO3.** Replace the toy `run_vectorised` with a real
  `run_target_positions(...)` returning equity, returns, drawdown and trades in
  the shape `BacktestResult.from_*` expects.
- **2.4 New engine implementation.** `AlgoCoreExecutionEngine(ExecutionEngine)`
  in `backtesting/engine.py`, selectable but not yet default.

**Acceptance:** for sign-only signals, `AlgoCoreExecutionEngine` and
`RaptorExecutionEngine` agree on equity curve to within fees/slippage
tolerance — a differential test that pins the new engine to the old one.

### Phase 3 — Instrument and cost model

- **3.1 `InstrumentSpec` gains** `multiplier`, `tick_size`, `currency`,
  `margin_rate`, `commission_per_contract`.
- **3.2 Futures roll handling** in `data/`: roll calendar, back-adjusted price
  series, and roll-cost accounting so roll gaps stop being booked as PnL.
- **3.3 Cost models** as a swappable Protocol: fixed-fraction (today's
  behaviour), spread-based, and a square-root market-impact model driven by the
  `volume` column already carried on every spec.
- **3.4 Turnover and capacity** metrics added to `BacktestMetrics` — follow the
  6-step procedure in `CLAUDE.md` §"Adding a new metric".

**Acceptance:** a backtest on a rolled continuous contract shows PnL
attributable to rolls separately from strategy PnL.

### Phase 4 — Research layer build-out

`research/` needs to become a real feature library, because right now every
strategy re-implements its own vol estimate inline.

- **4.1 Feature primitives.** Momentum, carry, value, seasonality, skew,
  breakout, term-structure slope. All causal by construction, all returning
  Polars expressions where possible so they compose in lazy pipelines.
- **4.2 Cross-sectional toolkit.** Ranking, winsorisation, sector/asset-class
  neutralisation, z-scoring across the panel. Extract the ranking logic
  currently buried in `CrossSectionalMomentum._assign_positions` so it is
  testable and reusable.
- **4.3 Panel data structure.** A typed multi-asset panel replacing
  `dict[str, pl.DataFrame]`, enforcing time alignment once instead of
  re-validating it in every strategy.
- **4.4 Feature evaluation.** Information coefficient, IC decay by horizon,
  quantile-bucket returns, feature autocorrelation and turnover. This lets
  features be judged *before* they are wrapped in a strategy and pushed through
  a full backtest.
- **4.5 Data quality gates.** Gap detection, stale-price and outlier flags,
  calendar validation — run at ingest and surfaced in the CLI.
- **4.6 Signal combination.** A framework for blending multiple features into
  one position series (equal-weight, IC-weighted, risk-parity), which is the
  natural next strategy layer above the four single-signal strategies.

**Acceptance:** `CrossSectionalMomentum` is rewritten on top of the new
primitives and reproduces its current test expectations.

### Phase 5 — Portfolio construction

Currently sizing is a scalar weight per leg. Between signal and execution there
should be a real portfolio step:

- **5.1 `PortfolioConstructor` Protocol** — signal panel in, target weights out.
- **5.2 Implementations:** equal weight (today's behaviour), inverse-vol,
  risk parity, mean-variance with shrinkage covariance.
- **5.3 Constraints:** gross/net exposure caps, per-asset and per-sector limits,
  turnover penalty.
- **5.4 Portfolio-level vol targeting**, replacing the per-leg approximation in
  `VolTargetAllocator`.

This is where `algo-core`'s `RiskCheck` finally gets used — finish its
position-limit branch and call it from the constructor.

### Phase 6 — Persistence and analysis

- **6.1 Store trades, per-bar positions, and equity curves** in DuckDB, closing
  the `[future: trades, pnl series, signals]` gap.
- **6.2 Run comparison** — diff two experiments, track metric drift across runs.
- **6.3 Attribution** — decompose PnL by asset, by signal, by holding period.
- **6.4 CLI surface** for all of the above; also replace the `algo backtest run`
  stub, which still prints "Not yet implemented".

### Phase 7 — Live / paper execution

Only worth starting once Phase 2 has one engine driving both paths.

- **7.1 `OrderRouter` Protocol** with a paper implementation over live data.
- **7.2 Shared position-target logic** between backtest and live, so the two
  cannot drift.
- **7.3 Pre-trade risk checks** reusing the Phase 5 constraint code.
- **7.4 Reconciliation** — expected vs actual fills, slippage attribution.

---

## 6. Sequencing

```
Phase 0 ──┬── Phase 1 ──────────────┐
          │                          ├── Phase 5 ── Phase 7
          ├── Phase 2 ── Phase 3 ────┤
          │                          │
          └── Phase 4 ───────────────┘
                                     └── Phase 6 (any time after 2)
```

Phases 0 and 1 are prerequisites for trusting any research output and should
land first — they are also small. Phase 4 is independent of the engine
decision and can run in parallel with Phase 2 if there is capacity. Phase 7
should not start until Phase 2 has consolidated on one engine.

Suggested first slice: **Phase 0 in full**, then **1.1 and 1.2**. That is a
few days of work and it converts the existing framework from "produces numbers"
to "produces numbers you can act on".

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Phase 2 is a large rewrite of the execution core | Differential-test `algo-core` against raptorbt before switching the default; keep both engines selectable |
| Fixing sizing and lag will make historical results look worse | Expected — current results are inflated by same-bar fills and OOS selection. Record before/after explicitly rather than quietly rebasing |
| `algo-core` has no test coverage to speak of | Build coverage as part of Phase 2, not after |
| Scope creep across seven phases | Phases 0 and 1 are independently valuable; treat everything past Phase 3 as re-plannable |
