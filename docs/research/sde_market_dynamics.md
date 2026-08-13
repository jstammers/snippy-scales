# SDEs and Neural SDEs for Trading: A Feasibility Assessment

**Question.** Can stochastic differential equations — classical or neural — be turned into
profitable trading by one person with a computational-physics PhD, ML/MLOps experience, and
retail access to CME futures?

**Short answer.** Yes, but almost certainly not in the way the question is usually meant. Using
a neural SDE to *predict direction* is the weakest available application and is where most
attempts die. Using one to *forecast the conditional distribution* — variance and tails — and
sizing positions from it is defensible, has a real mechanism behind it, and plays to the
strengths you actually have. The expected prize is modest: perhaps +0.1 to +0.2 of Sharpe on
top of a well-built classical futures programme, not a transformation.

The two things most likely to decide the outcome are not modelling choices at all. They are
**instrument selection** (which changes trading costs by 10×) and **search discipline** (which
decides whether your backtest means anything). Both are quantified below.

---

## 0. What this study found in the existing code

Before any of the analysis below, three defects in the current stack had to be fixed, because
they made the question unanswerable. Two of them were inflating reported performance by
roughly an order of magnitude.

| Finding | Effect | Status |
|---|---|---|
| `SignFlipInterpreter` reads only the **sign** of a position | All vol-scaled sizing silently discarded; `TrendFollowing`'s vol-scalar had zero effect on any result | Fixed — `TargetPositionEngine` |
| raptorbt annualises with **365**, not 252 | Sharpe overstated by √(365/252) ≈ **1.20×** | Fixed — `periods_per_year`, divergence pinned by test |
| raptorbt's **multi-leg path** (used by every evaluation fold) | Sharpe **4–7× higher** than its own single-instrument path on *identical* equity curves; `exposure_pct`, `omega_ratio` and trade-return fields silently zero | Fixed — documented, pinned by test |

The combined effect, measured end-to-end on a **random walk with negligible drift**:

| Engine | Best sweep Sharpe | Deflated Sharpe | Verdict it implies |
|---|---|---|---|
| raptorbt (previous default) | **10.72** | 1.000 | "certain skill" |
| `TargetPositionEngine` | **0.42** | 0.606 | "probably nothing" |

Equity curves and total returns were never affected — only the risk-adjusted ratios. But those
ratios are exactly what a research programme steers by. **Any Sharpe recorded in this repo
before this change should be treated as unusable.**

---

## 1. Method map, ranked by retail viability

| Family | Representative work | Verdict for you |
|---|---|---|
| **Classical SDE vol models** — Heston, SABR, OU, CIR | standard | **Start here.** Few parameters, fast to fit, parameters testable for stability. The baseline everything else must beat. |
| **Rough volatility** — RFSV, rough Bergomi | Gatheral–Jaisson–Rosenbaum; Bayer–Friz–Gatheral | **Genuinely promising.** RFSV beats AR/HAR out-of-sample on log-variance with essentially *one* parameter (H). Best effort-to-payoff ratio in this table. |
| **Signature / rough-path features** | Signature Trading (Futter et al.); signature kernels | **Practical.** Works on bars, no SDE solver, no GPU. Cheapest way to capture path-dependence. Dimension grows fast — truncate hard. |
| **Neural SDEs (generative)** | Kidger's SDE-GAN; signature-kernel scoring (Issa et al.) | **Strong for simulation, weak for alpha.** Use to generate synthetic paths for robustness testing, not to predict returns. Train non-adversarially — SDE-GANs need a Lipschitz discriminator and are notoriously unstable. |
| **Neural-SDE market models** | arbitrage-free option-surface models (Cohen, Reisinger, Wang) | **Best documented track record**, but monetisable retail only indirectly, via variance-risk-premium sizing. |
| **Deep hedging** | Buehler et al.; rough-vol extensions | Real, but it is a *cost-reduction* edge for someone who already runs an options book. It does not create alpha from nothing. |
| **Hawkes / neural Hawkes, LOB diffusion models** | TRADES, DiffVolume, LOBERT | **Out of reach.** Needs MBO data the ingestion layer does not support, plus colocation. See §6. |

---

## 2. The central argument

### Why directional neural SDEs fail

Daily futures returns have a signal-to-noise ratio around 0.05. A neural SDE has thousands of
parameters. This is the textbook overparameterised-model-in-low-SNR regime, and the failure is
not fixable by architecture, regularisation, or more data — there is not more data. Ten years
of daily ES bars is 2,520 observations. You cannot fit a flexible continuous-time generative
model to 2,520 noisy points and expect the drift term to generalise.

Worse, direction is close to a martingale by construction: if it were predictable at a scale
worth trading, it would be arbitraged by people with better data and lower latency than you.

### Why distributional forecasting is different

Volatility clustering is the most robust stylised fact in finance. Conditional variance *is*
predictable, persistently, across every liquid market, and has been for forty years. It is not
arbitraged away because it is not directly tradeable without an options book — it is a risk
input, not a price signal.

The mechanism by which a better variance forecast makes money is arithmetic, not speculative.
For a vol-targeted strategy, position size is `target_vol / forecast_vol`. Errors in the
denominator do two things: they push realised volatility away from target (so you are
mis-levered), and they correlate position size with subsequent volatility in the wrong
direction (so you are largest exactly when risk is highest). Sharpe on the *same* directional
signal improves when the sizing denominator improves. **You do not need to predict direction
better at all.**

This is also where your background is actually an edge. Calibrating stochastic processes,
diagnosing estimator bias, and building reproducible pipelines is what a computational physics
PhD with MLOps experience is genuinely better at than the median market participant. Predicting
direction is not — there you are competing with people who have the same skills, plus better
data, plus colocation.

### The honest caveat

"Better vol forecast → better Sharpe" is a real mechanism but a bounded one. GARCH-family and
HAR models already capture most of the predictable variance. The realistic increment from
rough-volatility or neural methods is at the margin — and the margin is where transaction
costs live. Which brings us to the arithmetic.

---

## 3. Cost arithmetic — and why instrument choice dominates

A futures contract's notional is `price × multiplier`. Commission is a flat cash amount per
contract; the bid-ask spread scales *with* the contract. So **larger contracts are cheaper per
unit of risk**, and the effect is not small.

Computed with `FuturesCostModel` (half-spread = 0.5 tick, retail commissions):

| Contract | Multiplier | Notional | Comm/side | Half-spread | **bp/side** | **bp/round-turn** |
|---|---:|---:|---:|---:|---:|---:|
| **NQ** | 20 | $440,000 | $2.00 | $2.50 | 0.102 | **0.205** |
| **MNQ** | 2 | $44,000 | $0.75 | $0.25 | 0.227 | **0.455** |
| **ES** | 50 | $300,000 | $2.00 | $6.25 | 0.275 | **0.550** |
| **MES** | 5 | $30,000 | $0.75 | $0.63 | 0.458 | **0.917** |
| **ZN** | 1,000 | $112,000 | $2.00 | $7.81 | 0.876 | **1.752** |
| **CL** | 1,000 | $70,000 | $2.50 | $5.00 | 1.071 | **2.143** |

**NQ is 10× cheaper to trade than CL per unit of notional.** That single decision swamps any
modelling improvement you are likely to achieve. Note also that the micro contracts cost
roughly 1.7–2.2× their full-size equivalents — the convenience of fine granularity is paid for.

### Annual cost drag by rebalancing frequency

Assuming average turnover of 0.5× notional per rebalance at 1× leverage:

| Rebalance | Rebalances/yr | ES drag | MES drag |
|---|---:|---:|---:|
| Daily | 252 | **0.69%** | **1.16%** |
| 4-hourly | 1,512 | 4.16% | 6.93% |
| Hourly | 5,796 | **15.94%** | **26.57%** |

This is the single most important table in this document.

At a 10% vol target and gross Sharpe 0.5, gross return is ~5%/yr. Daily rebalancing costs
0.7–1.2% of that — it survives, netting Sharpe ~0.42–0.44. **Hourly rebalancing costs 16–27%
of notional per year against a 5% gross return. It is not marginal; it is arithmetically
impossible.**

**Conclusion for the intraday regime you asked about:** 1m–1h bars are viable *as model inputs*
— using intraday data to build a better daily volatility estimate is free of this problem,
because you still only trade once a day. Trading *at* intraday frequency requires roughly 6–20×
more gross alpha than the daily regime to clear costs. Unless a specific signal justifies that,
use intraday data for estimation and trade daily.

### Capital thresholds

The binding constraint at small size is not cost but **integer-contract granularity**. One
contract carries a fixed amount of risk; if your risk budget per leg is smaller than that, you
round to zero and hold nothing.

One MES carries ~$4,800 of annualised volatility ($30k notional × 16% vol). For an
N-instrument portfolio at 10% total vol target with low cross-correlation, each leg gets
10%/√N of the risk budget:

| Assets | Per-leg vol budget | Capital for ~1 contract/leg | for ~2 contracts/leg |
|---:|---:|---:|---:|
| 1 | 10.00% | $48,000 | $96,000 |
| 4 | 5.00% | $96,000 | $192,000 |
| 8 | 3.54% | **$136,000** | $272,000 |
| 12 | 2.89% | $166,000 | $333,000 |
| 20 | 2.24% | $215,000 | $429,000 |

**Read this as:** below roughly **£50–70k**, you cannot hold a diversified futures book at a
sane vol target — you are forced into either one or two instruments (losing the diversification
that is the main free lunch available) or into higher leverage. **£130–270k** is where an
8-instrument micro programme becomes properly expressible. Above ~£500k the micros stop making
sense and you should be in full-size contracts, which are ~2× cheaper.

Fixed costs sit on top: budget for Databento (price your actual symbol/date range via
`Historical.metadata.get_cost` rather than guessing — daily CME bars are cheap, 1-minute is
moderate, MBO is not) plus a broker. At £20k of capital, data and infrastructure alone can
exceed plausible annual profit. At £250k they are a rounding error.

---

## 4. The statistics that decide whether you are fooling yourself

### Selection bias

Searching N configurations and reporting the best Sharpe is not an unbiased estimate. Under the
null hypothesis of *no skill whatsoever*, with trial dispersion 0.5:

| Configurations searched | Expected best Sharpe, purely from luck |
|---:|---:|
| 1 | 0.00 |
| 10 | 0.79 |
| 50 | 1.14 |
| 200 | 1.38 |
| 1,000 | 1.63 |

A 200-point grid search producing a best Sharpe of 1.3 has found *nothing*. This is why
`deflated_sharpe` is now a column in `summary_df()` and should be read before the Sharpe column.

The corollary is uncomfortable but important: **search breadth must be budgeted, not
maximised.** A genuine Sharpe-0.67 edge over 10 years of daily data scores DSR 0.91 when found
by a 3-point search and DSR 0.49 when found by a 50-point search. The edge is identical; only
your ability to demonstrate it changed. This is pinned as a test
(`test_broad_search_can_bury_a_real_edge`).

### How long until you know it worked

Minimum track record length at 95% confidence, daily bars:

| Annualised Sharpe | Days | **Years** |
|---:|---:|---:|
| 0.30 | 7,578 | **30.1** |
| 0.50 | 2,730 | **10.8** |
| 0.75 | 1,214 | 4.8 |
| 1.00 | 684 | 2.7 |
| 1.50 | 305 | 1.2 |
| 2.00 | 173 | 0.7 |

A Sharpe-0.5 strategy — a *good* result for retail futures — takes **eleven years** of live
trading to distinguish from luck at 95% confidence. You will never confirm your edge from live
P&L alone on a realistic horizon.

Three consequences follow, and they shape the whole programme:

1. Out-of-sample discipline in *research* is not good practice, it is the only evidence you
   will ever have.
2. Prefer strategies whose mechanism you can articulate. When statistics cannot adjudicate,
   economic reasoning is the tiebreaker.
3. Judge the volatility model on **forecast accuracy** (CRPS, PIT calibration), not on P&L.
   Every bar is an observation, so a vol forecast can be validated in months rather than
   decades. This is the single strongest methodological argument for the
   distributional-forecasting framing — it is the only part of the problem where you can
   actually get statistical power.

---

## 5. Realistic expected outcome

Anchor to what professionals achieve with vastly more resource: managed-futures CTAs run
long-term net Sharpe of roughly **0.4–0.7**. That is the honest benchmark for a diversified
trend/carry futures programme, and it already reflects institutional execution and research
budgets.

A realistic decomposition for you:

| Component | Plausible Sharpe contribution |
|---|---|
| Diversified trend + carry across 8–15 futures, vol-targeted | 0.4 – 0.7 |
| Better volatility estimation (range estimators, rough-vol forecasting) | +0.05 – 0.20 |
| Neural SDE over and above a well-tuned classical vol model | +0.00 – 0.10, highly uncertain |
| **Realistic total** | **0.5 – 0.9** |

At 10% vol target and Sharpe 0.6, expect ~6% annual return with occasional 15–20% drawdowns.
On £150k that is ~£9,000/year before tax, with multi-year stretches of underperformance that
will feel indistinguishable from being wrong. This is a serious research hobby that may
compound meaningfully over decades. It is not a salary.

**What would make this genuinely worthwhile** is not the P&L. It is that the resulting
artefact — a validated, cost-aware, statistically honest research pipeline with a live track
record — is a credible basis for managing external capital, where the economics are completely
different. Frame the work that way and the ten-year track-record problem becomes the *product*
rather than the obstacle.

---

## 6. Market making — the honest verdict

You excluded this from the build scope; here is the assessment for completeness.

The literature is richest here (neural Hawkes, TRADES, DiffVolume, LOBERT) and the retail
viability is lowest. Requirements: MBO or MBP-10 data (the ingestion layer supports OHLCV bars
only — `data/config.py` maps to `ohlcv-1s/1m/1h/1d/eod`), queue-position modelling, latency
simulation, and colocation. Competitive latency needs FPGA-class infrastructure at ~$50k/year
minimum, against firms spending millions.

The Avellaneda–Stoikov framework that dominates retail discussion of this was designed for
quote-driven markets and is a poor fit for price-time-priority order books like CME, where
queue position dominates spread capture. Crypto maker rebates are the usual suggested
workaround, but meaningful rebate tiers require $100M+ monthly volume.

**Verdict: do not pursue.** Not because the modelling is beyond you — it isn't — but because
the edge in market making is infrastructure and queue priority, and neither is purchasable at
retail scale.

---

## 7. Falsification plan

Ordered by cost. Each step can kill the thesis before the next is worth starting.

**Step 1 — Is the volatility even rough? (hours)**
Run `hurst_exponent` on log realised variance for ES, NQ, ZN, CL across walk-forward folds.

- H stably ≈ 0.1 → rough-volatility modelling is justified. Proceed.
- H ≈ 0.5 → the entire rough/neural motivation evaporates. Use Heston or HAR and stop here.
- H unstable across folds → the data does not support *any* fitted volatility model. Stop.

Also run `jump_ratio`. If jumps dominate quadratic variation, no continuous SDE will fit and a
jump-diffusion is needed instead.

⚠️ Check the answer's stability across lag ranges first. The variogram estimator is biased
downward at long lags — enough to manufacture a spurious "rough" reading. This bias is
documented in `roughness.py` and was caught during development, where it turned a true H = 0.7
into 0.61.

**Step 2 — Do better estimators improve the forecast? (days)**
Compare Yang–Zhang against close-to-close as inputs to a HAR forecast; score with `crps_ensemble`
and `calibration_error`. If range estimators do not improve out-of-sample CRPS, no downstream
model will rescue it.

**Step 3 — Does a better forecast improve risk-adjusted returns? (days)**
Same directional signal, different sizing denominators, through `TargetPositionEngine` with
`FuturesCostModel`. **This is the load-bearing experiment for the whole thesis.** If a
materially better vol forecast does not improve net Sharpe, the mechanism in §2 does not
operate on your data, and the neural programme has no premise.

**Step 4 — Does rough Bergomi beat Heston out-of-sample? (weeks)**
Both are already implemented. Compare on CRPS, not P&L. If the one-parameter rough model does
not beat the Markovian one, a thousand-parameter neural model will not either.

**Step 5 — Only now, a neural SDE. (months)**
Add the optional `jax` + `diffrax` dependency group. Train a latent SDE non-adversarially with
a signature-kernel score. Benchmark against Step 4 on CRPS and calibration, then on net Sharpe
with costs and a deflated Sharpe reported.

**Do not skip to Step 5.** Every step before it is cheap, and each one can save the months that
the next would cost.

---

## 8. Recommendation

1. **Build the boring thing first.** Diversified trend and carry across 8–15 liquid futures,
   vol-targeted, daily rebalance, honest costs. This is the 0.4–0.7 Sharpe that constitutes
   the bulk of the achievable outcome. Everything else is a refinement on top.
2. **Choose instruments by cost.** NQ/ES over CL/ZN where the signal permits. This is worth
   more than any model improvement you are likely to find.
3. **Rebalance daily, estimate intraday.** Use 1m–1h bars to build better volatility estimates;
   do not trade at that frequency.
4. **Run Steps 1–3 of the falsification plan** before adding a deep-learning dependency.
5. **Report deflated Sharpe on everything**, and budget search breadth deliberately.
6. **Treat the vol model as the research programme**, not the directional signal. It is the
   part that is predictable, the part where your skills are differentiated, and the only part
   you can validate on a human timescale.

The realistic prize is a Sharpe-0.6-ish programme that compounds quietly and constitutes a
credible track record. The realistic risk is spending two years building an elegant neural SDE
that a HAR model with Yang–Zhang inputs matches in ten lines. Steps 1–4 exist to tell you which
one you are in, cheaply, before you commit.

---

## Appendix: what was built

| Module | Purpose |
|---|---|
| `backtesting/continuous.py` | `TargetPositionEngine` — honours continuous position sizing; correct annualisation |
| `backtesting/costs.py` | `ProportionalCost`, `FuturesCostModel` — the §3 arithmetic, executable |
| `evaluation/statistics.py` | Deflated Sharpe, PBO (CSCV), minimum track record length — the §4 tables |
| `evaluation/scoring.py` | CRPS, pinball loss, PIT calibration, coverage — for scoring distributions |
| `research/volatility.py` | Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang |
| `research/roughness.py` | Hurst exponent, variogram, bipower variation, jump ratio — Step 1 |
| `research/sde.py` | OU (exact MLE), Heston, rough Bergomi — Steps 4–5 baselines |

Every estimator is validated against simulated data with known ground truth. That discipline
caught two real defects during development — an inverted bipower scaling constant that made
Gaussian noise read as 59% jumps, and the variogram lag bias described in Step 1 — either of
which would have produced a confident, wrong conclusion about market structure.

## Sources

- [Volatility is rough](https://www.tandfonline.com/doi/full/10.1080/14697688.2017.1393551) — Gatheral, Jaisson, Rosenbaum
- [Neural SDEs as Infinite-Dimensional GANs](https://proceedings.mlr.press/v139/kidger21b/kidger21b.pdf) — Kidger et al.
- [Non-adversarial training of Neural SDEs with signature kernel scores](https://arxiv.org/pdf/2305.16274)
- [Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework](https://arxiv.org/pdf/2308.15135)
- [Estimating risks of option books using neural-SDE market models](https://arxiv.org/pdf/2202.07148)
- [Forecasting Volatility with Machine Learning and Rough Volatility](https://arxiv.org/pdf/2311.04727)
- [The Probability of Backtest Overfitting](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110) — Bailey, Borwein, López de Prado, Zhu
- [Painting the market: generative diffusion models for LOB simulation](https://arxiv.org/abs/2509.05107)
- [Event-Based Limit Order Book Simulation under a Neural Hawkes Process](https://arxiv.org/pdf/2502.17417)
- [Databento metered pricing](https://databento.com/docs/api-reference-historical/basics/metered-pricing)
