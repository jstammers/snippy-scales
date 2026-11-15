# Replicating Moskowitz, Ooi & Pedersen (2012)

**Question.** `TimeSeriesMomentum` on S&P 500 stocks earned a much lower out-of-sample
Sharpe ratio than the "Sharpe greater than one" in *Time Series Momentum* (MOP). Is
the gap caused by a bug in our implementation, or is it what the strategy really
earns on our data?

**Approach.** Rebuild the paper's method from scratch, independently of the repo's
strategy and engine code, and run it on the same data. Then run the paper's rules
through the full report pipeline on futures. If the two agree, the gap is the data
(universe and period), not the code.

Script: [`scripts/replicate_tsmom.py`](../../scripts/replicate_tsmom.py). It
implements the paper in pandas and shares no strategy or engine code with the repo. It
uses `snippy_scales.data` only to read bars and recover the held contract.

Results are not kept in this repository: they live with the local, git-ignored
report instances (`reports/tsmom-futures/`, `reports/tsmom-futures-global/`).

## What the paper does

| | MOP (2012) | `TimeSeriesMomentum` |
|---|---|---|
| Universe | 58 futures: 24 commodities, 12 FX, 9 equity indices, 13 bonds | any bars |
| Period | 1985–2009 (1965–1985 out of sample: Sharpe 1.1) | — |
| Signal | sign of the past 12-month excess return, **no skip** | sign of the `lookback`-bar return, optional `skip_recent` |
| "12-1" | 12-month lookback, **1-month holding period** | docstring and docs used it to mean "skip the last month" |
| Volatility | EWMA of squared demeaned daily returns, centre of mass 60 days, × 261 | rolling std over `vol_lookback` bars, × 252, floored at 1e-6 |
| Sizing | 40% / σ, **no cap** | `vol_target` / σ, capped at `max_leverage` (default 2) |
| Rebalance | month end | every `rebal_freq` bars, counted from the first bar |
| Portfolio | equal weight over the instruments available | 1/N over all symbols, so names not yet listed sit in cash |

The paper's per-instrument gross Sharpe ratios (its Fig. 2) are mostly **0.1–0.6**. The
2-year Treasury is highest at about 1.05, and the S&P 500 is about 0.35. The diversified
portfolio passes 1 only because 58 low-correlated trend bets are averaged together. A
narrower or shorter universe, or a single asset class, should be expected to earn
less.

## The reports

* `reports/templates/configs/tsmom_futures.yaml` runs the paper's rules on the 34 CME
  `.ROLL` futures: 40% per instrument, no leverage cap, monthly rebalance, $2 per
  contract per side, the paper's lookback grid of 1, 3, 6, 9 and 12 months, and five
  walk-forward folds.
* `reports/templates/configs/tsmom_futures_global.yaml` uses the same rules on 42
  futures over the years both exchanges have data (2019 onward; ICE history starts in
  December 2018): the 34 CME futures, plus Brent, gasoil, London cocoa, robusta coffee,
  white sugar and the FTSE 100 from ICE Europe, and MSCI EM and MSCI EAFE from ICE US.

One USD margin account trades all of them. ICE contracts load as
`<ROOT>.<VENUE>.ROLL` (`BRN.IFEU.ROLL`) and are re-listed on the account's venue. The
two GBP-quoted contracts (cocoa, FTSE) are converted to USD settlement by settlement:
each day's price change is converted at that day's CME GBP/USD front-month close
(`futures.to_usd`). A futures position has no notional to convert, so this adds no
currency bet.

Nautilus's own "Sharpe Ratio (252 days)" statistic is computed on a return series with
one row per calendar day (weekends zero) and scaled by √252, so it understates the
Sharpe ratio. Use the report's figure.

## Problems found in `TimeSeriesMomentum`

None of these is a bug in the signal, but each one changes results.

1. **`max_leverage=2` silently drops low-volatility assets.** The cap binds whenever
   σ < `vol_target`/2. At a 40% target on futures, that covers every bond and FX
   contract: Treasury-note futures have about 1.5–5% volatility, and MOP levers them
   10–25×. With the cap in place, the book becomes mostly equity and commodity risk.
   Futures runs need a high `max_leverage` (the reports use 50).
2. **Volatility floor of 1e-6, combined with stale bars, produces extreme leverage.**
   Some stock series contain runs of flat, zero-volume placeholder bars. Over those
   runs the rolling std falls to about 0, so σ\*/σ̂ grows towards 10⁶, and only the
   leverage cap stops it. Fix: estimate volatility only on traded bars (`volume > 0`)
   and floor σ̂ at a sensible level, for example 2% annualised.
3. **"12-1" is mislabelled.** The docstring and `docs/strategies/momentum-ts.md` call
   `skip_recent=21` "the classic 12-1". In MOP, 12-1 means a 12-month lookback with a
   1-month holding period and **no skip**. The skip comes from cross-sectional momentum
   (Jegadeesh–Titman). It is fine as a searched parameter, but it is not the paper's
   specification.

Smaller differences: rolling vs EWMA volatility; rebalancing every 21 bars from the first
bar rather than at month end; and 1/N over the whole universe rather than over the names
alive at each date, which adds cash drag early in a ragged universe.

## Futures data and rolls

* **Source.** Databento daily bars of the calendar front and second month (`.c.0`,
  `.c.1`) for 40 CME roots (2010-06 onward) and 21 ICE roots (2018-12 onward). The
  configs are `configs/databento_tsmom_*.yaml`. Two ICE data quirks are handled at
  sync time:
  * Databento publishes each ICE bar twice, the second copy from the off-exchange
    block feed (venue XOFF). Only the exchange's own copy is kept.
  * Bars with every price undefined are dropped.
* **Rolls.** `.c.0` bars don't record which contract they came from. `algo data sync`
  stores each continuous symbol's contract-by-date mapping (a free Databento
  symbology lookup) beside the catalog.
  * `<ROOT>.ROLL` uses that mapping to split `.c.0`/`.c.1` back into real contracts.
  * It rolls to the next contract 5 trading days before the front switches, so it
    never holds into expiry.
  * Backtests trade the actual contracts; signals use the back-adjusted splice.
  * Weekend (Globex Sunday-session) bars are dropped, so 252 bars is a year for every
    root.
  * One-day price-scale errors (a bar more than 3× or under ⅓ of both neighbours) are
    dropped (`futures.drop_scale_glitches`). ICE's MSCI futures print a tenth of their
    price on 2021-08-13.
* **Exclusions:**
  * **Front month too sparse to price the rolls:** CME TN, 6M, PL, PA; ICE WTI, Long
    Gilt, SONIA and the ICE US softs and dollar index. For ICE US softs, the calendar
    front month has under 10% of trading days with a bar. Databento's volume-roll
    series (`.v.0`) would fix this if they are needed.
  * **Short-rate futures** (GE, SR3, Euribor), which the paper did not use.
* **Currencies.** The backtest engines run one USD account. ICE contracts in GBP are
  converted to USD settlement by settlement at the CME FX future's rate. The script
  uses local-currency returns, as in the paper, which differ only by the day's FX move
  on that day's return.
* **Sparse early contracts.** A rolled series starts at the first contract whose roll
  can be priced. This recovers ICE's MSCI EM and EAFE from July 2020. A series that
  would lose more than half its contracts this way is refused.
* **Narrower than the paper.** Only 34 CME roots (42 with ICE) from 2010, against
  MOP's 58. Four of the equity roots are near-duplicate US indices.

## Reproduce

```bash
SNIPPY_DATA_ROOT=/path/to/data uv run python scripts/replicate_tsmom.py --out /tmp/tsmom_rep
```

The script writes `results.json` and `monthly_returns.csv` (the TSMOM and passive
monthly series per universe) to `--out`. It only reads from `data/`.
