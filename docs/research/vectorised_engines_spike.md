# Spike: a vectorised screening engine (vectorbt vs raptorbt)

**Date:** 2026-11-13. **Plan:** [Spike: Vectorised Screening](../architecture/vectorised_screening_spike.md).
**Code:** `scripts/spike_vectorised/` (scratch virtualenv only; see its README).

## Recommendation

**Neither library. Build the screening engine as a small in-house numba kernel.**

- **Accuracy.** The kernel is about 50 lines. It reproduces Nautilus to within **1e-9 to
  1e-8 per day** on every case below, and every order count matches. The cases cover
  leverage, shorts, multipliers, the shared cash pool, rolls, both fee types and the
  rebalance band.
- **Speed.** One AAPL run takes **0.03 ms**, against 187 ms on Nautilus (about 6,000×).
  The 19-future portfolio takes 0.6 ms against 3.9 s. The 100-trial × 5-fold grid takes
  0.1 s including signals, against 11.5 s.
- **vectorbt** can match Nautilus exactly, but only through `from_order_func` with our own
  numba callbacks. That is the same sizing code as the kernel, plus vectorbt's per-call
  overhead (1.8 ms against 0.03 ms), a Commons Clause licence and a phantom-cash
  workaround for leverage. Its native `from_orders` path is off by 1–2% of equity, about
  0.01 Sharpe, whenever gross exposure exceeds 1×. It also can't express the band.
- **raptorbt** has no entry point that resizes a position. It can be made exact only by
  opening a new hedging leg for every trade. That makes it *slower than Nautilus*: 0.98 s
  for one AAPL run and 28 s for the futures portfolio. Its per-contract fees are
  approximate.

The kernel in `scripts/spike_vectorised/engines.py` (`_simulate`) is the starting point
for `execution/vector.py`.

## Setup

| | |
|---|---|
| Machine | Apple M4 Pro, macOS (Darwin 25.6) |
| Python | 3.12.15 |
| Reference | `nautilus_trader==2.0.0rc6` via `execution.run_backtest` |
| vectorbt | `vectorbt==1.2.0`, `vectorbt-rust==1.2.0` (`vectorbt[rust]`) |
| raptorbt | `raptorbt==0.13.4` |
| Other | numba 0.68.1, numpy 2.5.4, pandas 3.1.1 |

Every engine gets the same **panel**. It has one column per traded instrument and one
row per bar close, and holds the raw closes and `target × weight`. Roll bars are marked:
the old contract's target is 0, and both legs ignore the band. The panel is built from
`compute_targets` and replays `SignalStrategy._flush` decision by decision. The engines
therefore differ only in how they simulate. Each adapter returns equity at every close,
before that close's trades, which is what Nautilus's `BacktestRun.equity` records.

Returns are daily mark-to-market returns of that equity. Sharpe is computed for every
engine, Nautilus included, by Nautilus's `PortfolioAnalyzer`
(`evaluation.walk_forward.returns_statistics`).

### Two Nautilus behaviours the reference depends on

Both were found because all three exact engines agreed with each other and disagreed
with Nautilus.

1. **Bar volume is liquidity.** On the real 19-future data, 1,394 of 43,144 Nautilus
   fills (3.2%) happen away from the bar close. An order larger than the liquidity Nautilus derives
   from a thin bar's volume walks to the next price level. For example, a 1-lot ZW sell filled at
   722.00 against a 731.50 close on 2011-05-09. The vectorised engines fill everything at
   the close.
   - The accuracy matrix therefore uses `ample_volume` copies of the real data (each bar's
     volume set to 1e9, as the synthetic fixtures already do), so it measures engine
     accuracy.
   - The rows marked *[real volume]* show the effect on the stored data. It is
     negligible for AAPL and the 8 equities. For the futures portfolio it is −1.0% of
     final equity and −0.0075 Sharpe.
   - A screening engine without a liquidity model overstates thinly traded futures
     slightly. This is one more reason to validate the top trials on Nautilus.
2. **Non-positive prices aren't marked.** `CL.c.0` closed at −2.67 on 2020-04-21.
   Nautilus keeps marking the position at its last positive close, and `_rebalance`
   doesn't trade at a price ≤ 0. The panel treats a non-positive close as "no bar", which
   brings the futures case from a 1.4% one-day error to an exact match.
   - The same happened to real CL outrights in April 2020, so `execution/vector.py` must
     copy the rule.
   - Whether Nautilus *should* mark a negative price is a separate question.

## Accuracy

Each engine cell shows **max |Δ daily return| / final equity ÷ Nautilus / Δ Sharpe**.

- **Pass:** final equity within 0.1% and Sharpe within 0.02 of Nautilus.
- **Data:** real cases use ample volume unless marked.
- **Capital:** $1M; $100M for the futures, so integer contract rounding is not the
  dominant effect.
- **Fees:** 5 bp is `commission_rate=5e-4`.

| Case | Max gross | kernel | vectorbt `from_order_func` | raptorbt session | vectorbt `from_orders` | Nautilus Sharpe | Nautilus orders |
|---|---|---|---|---|---|---|---|
| AAPL const 1.0, no costs | 1.00 | 4e-16 / 1.00000 / 0 | 1e-13 / 1.00000 / 0 | 7e-16 / 1.00000 / 0 | 2e-06 / 1.00000 / 0 | 0.9997 | 1 |
| AAPL TrendFollowing, no costs | 1.31 | 1e-08 / 1.00000 / 0 | 1e-08 / 1.00000 / 0 | 1e-08 / 1.00000 / 0 | **1e-02 / 0.98517 / −0.0097** | 0.4427 | 2,612 |
| AAPL TrendFollowing, 5 bp | 1.31 | 2e-08 / 1.00000 / 0 | 2e-08 / 1.00000 / 0 | 2e-08 / 1.00000 / 0 | **1e-02 / 0.98621 / −0.0091** | 0.4027 | 2,613 |
| 19 futures TSMOM (vol target 25%), no costs | 1.81 | 6e-10 / 1.00000 / 0 | 6e-10 / 1.00000 / 0 | 6e-10 / 1.00000 / 0 | **1e-02 / 0.98348 / −0.0122** | 0.2275 | 42,155 |
| 19 futures TSMOM, 5 bp | 1.81 | 8e-10 / 1.00000 / 0 | 8e-10 / 1.00000 / 0 | 8e-10 / 1.00000 / 0 | **1e-02 / 1.00583 / +0.0044** | 0.0542 | 39,908 |
| 8 equities CSMOM, no costs | 0.33 | 1e-05 / 1.00000 / 0 | 1e-05 / 1.00000 / 0 | 1e-05 / 1.00000 / 0 | 7e-05 / 0.99962 / −0.0015 | 0.1014 | 8,953 |
| 8 equities CSMOM, 5 bp | 0.33 | 3e-08 / 1.00000 / 0 | 3e-08 / 1.00000 / 0 | 3e-08 / 1.00000 / 0 | 7e-05 / 0.99988 / −0.0004 | 0.0635 | 8,989 |
| Rolled ES (synthetic, volume roll, 7 rolls), no costs | 1.00 | 1e-08 / 1.00000 / 0 | 1e-08 / 1.00000 / 0 | 1e-08 / 1.00000 / 0 | 2e-04 / 1.00116 / +0.0040 | −0.8852 | 563 |
| Rolled ES, $2.25/contract | 1.00 | 1e-08 / 1.00000 / 0 | 1e-08 / 1.00000 / 0 | 2e-04 / 1.00045 / +0.0017 | 3e-04 / 1.00235 / +0.0078 | −0.9719 | 560 |
| AAPL TrendFollowing, 5 bp, band 2% | 1.31 | 2e-08 / 1.00000 / 0 | 2e-08 / 1.00000 / 0 | 2e-08 / 1.00000 / 0 | n/a (no band) | 0.4122 | 861 |
| 19 futures TSMOM, 5 bp, band 0.5% | 1.81 | 3e-10 / 1.00000 / 0 | 3e-10 / 1.00000 / 0 | 3e-10 / 1.00000 / 0 | n/a (no band) | 0.0717 | 5,446 |
| AAPL TrendFollowing, 5 bp *[real volume]* | 1.31 | 2e-08 / 1.00000 / 0 | 2e-08 / 1.00000 / 0 | 2e-08 / 1.00000 / 0 | 1e-02 / 0.98621 / −0.0091 | 0.4027 | 2,613 |
| 19 futures TSMOM, no costs *[real volume]* | 1.81 | 6e-03 / 0.99018 / −0.0075 | 6e-03 / 0.99018 / −0.0075 | 6e-03 / 0.99018 / −0.0075 | 1e-02 / 0.97382 / −0.0197 | 0.2350 | 42,258 |
| 8 equities CSMOM, 5 bp *[real volume]* | 0.33 | 3e-08 / 1.00000 / 0 | 3e-08 / 1.00000 / 0 | 3e-08 / 1.00000 / 0 | 7e-05 / 0.99988 / −0.0004 | 0.0635 | 8,989 |

### Order counts

The exact engines' order counts equal Nautilus's in every ample-volume case but one.
CSMOM without costs has 8,954 orders against 8,953: a single half-even rounding tie,
because we compute in float and Nautilus computes in `Decimal`.

### Why vectorbt `from_orders` misses

- **Cash caps longs at 1×.** In `from_orders` cash can't go negative: `buy_nb` limits
  each buy to the available cash. A `targetpercent` above 1 is filled only partially.
  I found no leverage option in 1.2.0.
  - Clipping AAPL's targets to |target| ≤ 1 brings `from_orders` within 2e-5 per day and
    0.0001 Sharpe of the exact engines.
  - At 1.31× it reproduces the capped result exactly: Sharpe 0.4330 against 0.4329.
  - Shorts are filled in full (`lock_cash=False`).
  - A workaround (phantom cash) needs equity to be known before sizing, so it isn't
    possible in `from_orders`.
- **Rounding.** `size_granularity` floors each *trade*. Nautilus rounds the *target*
  half-even. This is the residual 1e-5 to 1e-4 on the gross ≤ 1 cases.
- **No equity-relative band.** `min_size` is in units, and the band depends on equity
  along the path.
- **Non-positive prices.** An order at a price ≤ 0 raises `ValueError`.

### Why raptorbt needs a workaround

- **No resizing.** On `PortfolioSession` with `oms_type="netting"`:
  - An order that adds to an open position is rejected (`position_open`).
  - A sell larger than the long only closes it; it never flips to short.
- **One direction per instrument.** `add_instrument(direction=...)` must be 1 or −1.
- **No reduce-only.** Under `"hedging"`, every order opens a new position, and
  `reduce_only` is always rejected.
- **Exact P&L means a leg per delta.** The only exact mapping is one hedging leg per
  delta: net position is the sum of legs, and fees are charged per fill notional, so both
  are correct.
  - Locked margin grows with turnover, so `leverage` must be set high enough never to
    bind (1000 here).
  - Marking cost grows with the number of open legs, which makes runs superlinear.
- **No shared equity at a close.** Events interleave per instrument at one timestamp, so
  the adapter corrects `session.equity()` to the full close by hand before sizing.
- **Per-contract fees are approximate.** `BacktestConfig.fees` is a rate, and nothing
  takes a per-contract amount (`fee_segment` covers Indian schedules only). The adapter
  uses the rate at the mean contract notional, which gives the ES $2.25 row's 2e-4 error.
- **The array path can't express targets.** `run_single_backtest` /
  `BatchSingleItem(position_sizes=...)` reads the size only at entry and never resizes.
  Per the plan this is not worked around with entries/exits. There is no array portfolio
  path with per-bar sizes either: `run_portfolio_backtest` takes booleans.

## Speed

- **Timing:** warm medians of 5 runs, except the looped grid, which ran once.
- **Fees:** 5 bp in every run.
- **Sim:** the engine alone, on a prebuilt panel.
- **Signals + panel:** `compute_targets` plus building the panel. My builder is plain
  Python; a production one would be vectorised.

| Workload | Nautilus | Signals + panel | kernel | vbt `from_order_func` | vbt `from_orders` (numba / rust) | raptorbt session |
|---|---|---|---|---|---|---|
| AAPL TrendFollowing, 1 run (2,690 bars) | 186.8 ms | 1.0 ms | **0.03 ms** | 1.8 ms | 8.1 / 8.0 ms | 983 ms |
| 19 futures TSMOM, 1 run (5,063 closes × 19) | 3,905 ms | 25.4 ms | **0.63 ms** | 6.7 ms | 17.8 / 18.5 ms | 28,048 ms |

100-trial × 5-fold grid, AAPL TrendFollowing (fast 5–50 × slow 60–240), `make_folds(n_splits=5)`:

| Engine | Time |
|---|---|
| Nautilus, 500 `run_backtest` calls (as `WalkForward.run_folds`) | 11.50 s |
| Signals, once per trial on the full history (causal, so reusable across folds) | 0.093 s |
| kernel, 500 runs looped | **0.005 s** |
| kernel, batched: 5 calls × 100 trials (`prange`) | **0.001 s** |
| vbt `from_order_func`, 500 runs looped / batched (100 groups per call) | 0.78 s / 0.054 s |
| vbt `from_orders`, 500 runs looped (numba / rust) | 4.08 s / 4.17 s |
| vbt `from_orders`, batched (numba / rust) | 0.039 s / 0.031 s |
| raptorbt session, 500 runs looped | 5.25 s |

Grid end to end: signals plus simulation take about 0.1 s on the kernel, 0.9 s on vbt
`from_order_func` and 11.5 s on Nautilus. Signal generation, not simulation, is now the
cost.

First call in a fresh process with an empty numba cache, including JIT compilation:

| kernel | vbt `from_orders` | vbt `from_order_func` | raptorbt |
|---|---|---|---|
| 0.46 s (then cached on disk with `cache=True`) | 4.96 s numba; the rust engine needs no JIT | 2.18 s | 0.95 s (no JIT; that is the run) |

vectorbt's Rust engine is no faster here. The time goes to pandas wrapping and
`Portfolio` construction rather than the simulation loop. It also doesn't support
callbacks, so `from_order_func` is numba only.

## Fit

| Model feature | vectorbt `from_orders` | vectorbt `from_order_func` | raptorbt | In-house kernel |
|---|---|---|---|---|
| Per-bar targets in units of equity | native (`targetpercent`) | workaround: callback sizes the delta | workaround: we size each delta ourselves | native |
| One equity figure per close, after all marks | native (`cash_sharing`, `call_seq="auto"`) | native (pre-segment callback) | workaround: correct `equity()` by hand | native |
| Fill at that bar's close | native | native | native (`fill_timing="same_bar_close"`) | native |
| Gross > 1× (margin, leverage) | **impossible** (cash-capped) | workaround: phantom cash subtracted from value | workaround: `leverage=1000` so margin never binds | native |
| Shorts / flips | native | native | workaround: a hedging leg per delta | native |
| Contract multiplier | workaround: scale price | workaround: scale price | native (`InstrumentSpec`) | native |
| Half-even rounding of the target | **impossible** (floors the trade) | native (callback) | native (we round) | native |
| `commission_rate` | native | native | native | native |
| `commission_per_contract` | workaround: rate = fee ÷ (close × mult), exact | native (`fixed_fees` per order) | **approximate** (one rate) | native |
| `slippage_prob` (seeded one-tick) | approximate (`slippage` as a fixed fraction) | possible in callback (own RNG) | approximate (`fill_prob_slippage`) | possible (own RNG); can't share Nautilus's RNG stream |
| Rebalance band (path-dependent) | **impossible** | native (callback) | workaround (we skip) | native |
| Futures rolls (close old, open new, costs on both, no band) | native via panel | native via panel | native via panel | native via panel |
| Non-positive prices | **errors** | native (skipped) | native (skipped) | native (skipped) |
| Bar-volume liquidity (as Nautilus) | impossible | possible (callback) | not tested (`queue_fill_model`, `partial_fills`) | possible; not built |
| Batch over parameter sets | native (columns) | native (groups) | none (`batch_single_backtest` has no targets) | native (`prange`) |
| Licence | Apache 2.0 + Commons Clause | same | MIT | ours |

**Slippage** was left out of the accuracy matrix. Nautilus draws it from its own seeded
RNG, which no other engine can reproduce bar for bar. A screening engine should apply
the *expected* cost, `slippage_prob × tick` per contract, and leave the realised draw to
Nautilus validation.

## Implications for `execution/vector.py`

- **Kernel.** Copy `_simulate` and keep the panel semantics: per-symbol `RollSchedule`,
  roll legs forced past the band, non-positive closes skipped, equity read once per close.
- **Parity test.** Gate the kernel with a parity test against `run_backtest` on ample-volume
  synthetic data: final equity within 0.1% and Sharpe within 0.02. In practice it is
  within 1e-8.
- **Reuse signals.** Compute targets once per trial and slice them per fold. Every
  strategy is causal, and this is where the remaining grid time goes.
- **Validate on Nautilus.** Keep Nautilus validation of top trials mandatory. The kernel
  has no liquidity model (−0.0075 Sharpe on the real futures portfolio) and no realised
  slippage.
