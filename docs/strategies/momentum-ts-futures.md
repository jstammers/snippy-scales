# Time-series momentum on futures (`momentum-ts`, MOP 2012)

This is the original setting of time-series momentum (TSMOM): Moskowitz, Ooi &
Pedersen (2012), *Time Series Momentum* (MOP). The paper runs one trend bet per
liquid futures contract across equity indices, government bonds, currencies and
commodities, each sized to the same ex-ante risk, and averages them into one
diversified portfolio. The strategy code is the same `momentum-ts` used on
stocks (see [`momentum-ts`](momentum-ts.md)). This page covers what the paper
did and how a report reproduces it.

## The paper's strategy

For instrument $s$ at a month end $t$, with $k$ the lookback in months:

$$
r^{\text{TSMOM},s}_{t,t+1} = \operatorname{sign}\!\left(r^s_{t-k,t}\right)
\frac{40\%}{\sigma^s_t}\, r^s_{t,t+1},
\qquad
r^{\text{TSMOM}}_{t,t+1} = \frac{1}{S_t}\sum_{s=1}^{S_t} r^{\text{TSMOM},s}_{t,t+1}.
$$

* $r^s$ is the futures excess return of the most liquid contract, rolled.
* $\sigma^s_t$ is the ex-ante annualised volatility: an exponentially weighted
  average of squared daily returns with a centre of mass of 60 days.
* $40\%$ is a target volatility per instrument. With low correlations across
  instruments it gives the portfolio about 12% volatility. There is no leverage
  cap: a 2-year Treasury future with about 1.5% volatility is held at more than
  25 times its capital weight.
* Positions are rebalanced every month and equally weighted across the $S_t$
  instruments available.
* The headline strategy is $k = 12$ with a one-month holding period. The paper's
  Table 2 also reports $k$ and holding periods from 1 to 48 months.

**Results, 1985–2009, 58 instruments.**
* Every one of the 58 instruments has a positive 12-month TSMOM Sharpe ratio,
  mostly between 0.1 and 0.6.
* The diversified portfolio's Sharpe ratio is above 1. Its monthly alpha is
  1.58% (t = 7.99) against MSCI World, bonds, commodities and the Fama–French
  factors.
* The return is largest in extreme markets, for example in late 2008. The payoff
  resembles a straddle on the equity market.
* The paper's 1966–1985 out-of-sample test has a Sharpe ratio of 1.1.

## How the report implements it

| Paper | Report |
|---|---|
| 58 futures, 4 asset classes | 34 CME futures: 6 equity indices, 5 Treasury futures, 7 currencies, 16 commodities. `tsmom-futures-global` adds 8 ICE futures (Brent, gasoil, London cocoa, robusta, white sugar, FTSE 100, MSCI EM, MSCI EAFE) over 2019–2026 |
| Returns in USD | One USD account. GBP contracts are converted settlement by settlement at the CME GBP/USD future's daily close |
| Most liquid contract, rolled | Front month (Databento calendar `.c.0`), rolled to the next contract 5 trading days before the front switches (`<ROOT>.ROLL`). Real contracts are traded. Signals use the backward-ratio-adjusted splice. |
| EWMA volatility, centre of mass 60 days | 63-bar rolling standard deviation (`vol_lookback`) |
| 40% / σ, no cap | `vol_target = 0.40`, `max_leverage = 50` |
| Monthly rebalance | `rebal_freq = 21` bars |
| 1/S_t over available instruments | Capital split 1/N over all 34 roots. RTY (2017) and KE (2014) hold cash before they list. |
| Gross of costs | $2 per contract per side, filled at the bar close |

The search covers the paper's lookback grid $k \in \{1, 3, 6, 9, 12\}$ months
(21–252 bars) with a one-month holding period. Its deflated Sharpe ratio is
corrected for those five trials.

## What to expect

* **The sample differs.** The paper's sample ends in 2009. Trend following had
  a weak 2010s: managed-futures indices earned Sharpe ratios around 0.2–0.5
  between 2011 and 2019. Hurst, Ooi & Pedersen (2017) document this weakening
  and still find positive returns over a century.
* **The universe is narrower.** 34 US-listed contracts with four near-duplicate
  US equity indices diversify less than 58 global ones. The paper's > 1 Sharpe
  came from that diversification: its per-instrument Sharpes were only 0.1–0.6.
* **A by-hand check.** `scripts/replicate_tsmom.py` implements the paper
  independently in pandas on the same roots (see
  `docs/research/tsmom_replication.md`). The report should land close to it. A
  much lower number would point to a bug.

## References

* Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics* 104(2), 228–250.
* Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management* 44(1).
* Kim, A. Y., Tse, Y. & Wald, J. K. (2016). Time series momentum and volatility scaling. *Journal of Financial Markets* 30.
* Huang, D., Li, J., Wang, L. & Zhou, G. (2020). Time series momentum: Is it there? *Journal of Financial Economics* 135(3).
