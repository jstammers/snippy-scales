# Spike: a vectorised screening engine (vectorbt vs raptorbt)

**Status:** done (2026-11-13). Findings and recommendation: [Vectorised engines spike](../research/vectorised_engines_spike.md).
**Deliverable:** a findings report and a recommendation, not production code.
**Base branch:** `claude/futures-rolls-phase-4` (PR #15, stacked on PR #14).

## Why

Since migration phases 2–4, every backtest runs on NautilusTrader's event-driven
engine (`execution.run_backtest`). That engine is correct but slow for searching:

| Workload | Old vectorised engines | Nautilus |
|---|---|---|
| One AAPL daily run, 2,690 bars, ~2,600 orders | ~1 ms | ~180 ms |
| 19 futures × ~4,630 daily bars, time-series momentum | 0.03–0.04 s | 9.7 s (4.1 s with a 0.2% band) |

Profiling (after removing our own per-bar overhead) puts ~70% of a run inside
Nautilus' engine itself. Nautilus has no vectorised mode, so walk-forward
studies with hundreds of trials across many assets are 100–300× slower than they
need to be.

The intended design has two tiers. A **vectorised engine screens** Optuna trials
and multi-asset sweeps. **Nautilus validates** the top trials, produces the final
report and runs live. Before building anything, this spike decides whether an
existing, tested library can be the screening engine, and which one.

## Questions to answer

1. **Accuracy.** Given the same target positions, how closely does each library's
   equity curve track Nautilus's? Test both without costs and with costs.
2. **Speed.** How long does each library take on the workloads below, using its
   fastest entry point that still does what our model does?
3. **Fit.** Which parts of our model can each library express, which need
   workarounds, and which are impossible?

Nautilus is the **reference**. The libraries are compared with Nautilus, not
with each other.

## The model to reproduce

The source is `python/snippy_scales/execution/strategy.py` (`compute_targets`,
`SignalStrategy._flush` / `_rebalance`) and `execution/backtest.py`
(`run_backtest`, `Costs`, `MarketData`).

- **Targets.** `compute_targets(research, frames)` returns `symbol -> {ts_init ns:
  target}`. Targets are positions in **units of equity**, so `1.0` is fully long and
  `-0.5` is half short. Null or NaN means flat. Targets are precomputed once over
  the whole history; causality is tested separately.
- **Sizing.** `contracts = round_half_even(target × weight × equity / (close ×
  multiplier), size_increment)`. The default weight is `1 / n_symbols`. Equity is
  read **after** every bar closing at that timestamp has been marked, so all
  symbols are sized from one equity figure.
- **Fills.** A market order for the difference fills at **that bar's close** and
  earns the next bar's return. In Nautilus, a bar's volume acts as liquidity, so
  give synthetic data ample volume.
- **Account.** One shared cash pool per venue: a `MARGIN` account with `NETTING`
  positions and `leverage=10` by default. Gross exposure above 1× and shorts are
  normal, especially for volatility-targeted futures.
- **Costs** (`Costs`):
  - `commission_rate` is a fraction of notional.
  - `commission_per_contract` is a fixed amount per contract. Only one of the two
    may be set.
  - `slippage_prob` is the probability of filling one tick worse; it is seeded.
- **Rebalance band.** Skip a trade whose notional is below `rebalance_band ×
  equity`. Whether a trade is skipped depends on equity along the way, so it
  can't be precomputed.
- **Futures rolls** (`data/futures.py`):
  - A `RollSchedule` names the active outright at each close. On a roll bar,
    close the old contract and open the new one at that bar's close. Both fills
    pay costs, and the band is ignored.
  - Signals use a backward **ratio**-adjusted series (`back_adjust`), whose
    returns are exactly the returns of the contract held.
  - `MarketData.from_outrights` builds all of this. `tests/test_futures_rolls.py`
    has a synthetic ES fixture.

## Candidates

Both are to be **installed into a scratch virtualenv, never the project**.
Pin exact versions. Python 3.12 is needed; see the memory note about raptorbt's
PyO3 build on 3.13.

### vectorbt `==1.2.0` (polakowo)
- **Licence:** Apache 2.0 + Commons Clause. Free to use, but you can't sell it.
- **Engine:** Numba, with an optional Rust engine (`vectorbt[rust]`, `engine=`).
- **Release:** 1.2.1 came out on 2026-11-11 and is too new; use 1.2.0.
- **How our model maps:**
  `Portfolio.from_orders(close, size=targets, size_type="targetpercent",
  price=close, fees=rate, slippage=..., cash_sharing=True, group_by=True,
  call_seq="auto", size_granularity=1, init_cash=capital, freq="1D")`.
  - **Per-contract fee:** pass `fees = per_contract / (close × multiplier)`, which
    gives the same total cost.
  - **Multiplier:** scale `price` and `close` by the multiplier, so that one unit is
    one contract.
  - **Rolls:** each outright is its own column. On the roll bar, the old contract's
    target goes to 0 and the new one takes the target.
  - **Band:** this needs `from_order_func` with a Numba callback. Measure that
    path separately.
- **Must verify:** whether `targetpercent` above 1 and shorts are filled in full
  or cut back by the cash limit. The free docs don't cover leverage, and PRO
  advertises both leverage and multipliers as PRO features. If positions are cut
  back, find out whether a workaround (`lock_cash=False`, a larger `init_cash`, or
  scaled targets) gives exactly the same returns.

### raptorbt `==0.13.4` (Alphabench, MIT)
- **What's new since we removed it:** margin accounts with leverage
  (`account_type="margin"`, `leverage=N`), `InstrumentSpec` contract multipliers,
  and portfolio runs that share one cash pool (`run_portfolio_strategy`,
  `run_portfolio_backtest`).
- **Why we removed it** (`docs/research/sde_market_dynamics.md`,
  `docs/research/raptorbt_defects.md`):
  - Metric bugs: annualising with 365 days, Sharpe 4–7× too high on the basket
    path, and a broken `calmar_ratio`. This spike avoids them, because statistics
    are computed by Nautilus from raptorbt's equity curve.
  - Our wrapper fed it boolean entries/exits, so **position size was discarded**.
    That is a simulation flaw, and it must not recur.
- **Must verify:**
  - Which entry point accepts **per-bar target sizes**. Candidates are
    `batch_single_backtest` (`position_sizes`, undocumented) and the class-based
    `run_portfolio_strategy` (`size_frac`).
  - Whether that entry point is still vectorised, or calls into Python on every
    bar.
  - If no entry point can express per-bar targets with shared cash, record that
    as the finding. Don't work around it with entries/exits.

## Method

Keep spike code outside the package, e.g. in a scratch directory or
`scripts/spike_vectorised/`, and don't commit it unless asked.

1. **Fixtures.**
   - Synthetic: reuse `tests/conftest.py` (`synthetic_data`, `synthetic_future`)
     and the ES outright fixture in `tests/test_futures_rolls.py`.
   - Real data, **read-only**: set
     `SNIPPY_DATA_ROOT=/Users/jimmy/Code/algotrading/data`. Use AAPL `ohlcv-1d`
     (ALPACA) and the 19 GLBX `*.c.0` daily series, with stand-in instruments as
     in the earlier benchmark. Those series have no instrument definitions in the
     catalog, so build `FuturesContract`s with sensible multipliers and ticks.
2. **One adapter per library** with the signature
   `run(research, data: MarketData, *, start, end, initial_capital, costs,
   rebalance_band, weights, leverage) -> (returns: pl.DataFrame[ts, return],
   equity: pl.DataFrame[ts, equity], n_orders: int)`.
   Build targets with `execution.strategy.compute_targets` so that every engine
   sees identical targets.
3. **Accuracy matrix.** Compare each adapter with `execution.run_backtest` on the
   same data:

   | Case | Purpose |
   |---|---|
   | Constant target 1.0, no costs, 1 equity | Basic accounting |
   | `TrendFollowing`, no costs / `commission_rate=5e-4` | Changing targets and fees |
   | `TimeSeriesMomentum` on 19 futures, targets that reach >1× gross and go short | Leverage, shorts, multipliers, shared cash |
   | `CrossSectionalMomentum` on ~8 equities | Multi-asset, sells before buys |
   | Rolled ES outrights, `RollRule("volume")`, `commission_per_contract=2.25` | Rolls, per-contract fees |
   | `rebalance_band=0.02` | Path-dependent skipping |

   For each case, report:
   - the maximum and the RMS daily difference in returns;
   - the final equity ratio;
   - the Sharpe of each engine, **both computed with Nautilus's
     `PortfolioAnalyzer`** on that engine's returns;
   - the order count.

   **Pass:** final equity within 0.1% and Sharpe within 0.02 of Nautilus when
   costs are zero. With costs, explain any gap.
4. **Speed.**
   - Use warm timings, with the median of at least 5 runs. Exclude Numba/JIT
     compilation, but report it separately.
   - Workloads:
     - one AAPL run;
     - the 19-future TSMOM portfolio;
     - a 100-trial × 5-fold grid on AAPL, running each engine in a loop of
       `make_folds` windows, as `evaluation.WalkForward` does.
   - Also report a batched or broadcast variant where the library supports one
     (e.g. vectorbt over many parameter columns at once).
5. **Fit table.** For each feature of the model above, mark it *native*,
   *workaround* (say which) or *impossible*.

## Constraints (from `CLAUDE.md` and the user)

- **Never delete anything under `data/`.** Real data is read-only. Delete only
  exact files you created yourself.
- Never spend Databento credit or run `algo data sync` without the user's
  approval.
- Don't add either library to `pyproject.toml`; this is a spike.
- Commits, if asked for: Conventional Commits, and hooks must not be bypassed.

## Deliverable

Write `docs/research/vectorised_engines_spike.md` containing:
- the versions and machine used;
- the accuracy matrix;
- the speed table, with Nautilus as a column;
- the fit table;
- a recommendation: vectorbt, raptorbt, or neither (in which case, a small
  in-house numpy kernel tested for parity with Nautilus).

Then stop and report back. Implementing the chosen engine is a separate task.

## The engine that follows if the spike succeeds

Not part of this spike, but it constrains the adapter design.

- **New module.** `execution/vector.py` would expose a `run_backtest` with
  **exactly** the current signature, returning the same `BacktestRun`.
  - `stats_*` would come from Nautilus's `PortfolioAnalyzer` on the vectorised
    returns, so metric definitions stay identical.
  - `fills` and `positions` would come from the library's order records.
  - `rolls` would come from the schedules.
- **Only one existing change.** `evaluation.WalkForward` calls
  `run_backtest` in one place and would gain `engine="nautilus" | "vector"`.
- **Parity test.** A parity test with Nautilus, at the tolerances above, would
  run in CI.
