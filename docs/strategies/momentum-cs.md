# Cross-sectional momentum (`momentum-cs`)

Cross-sectional momentum (CSMOM) bets on **relative** strength. Rank the
universe on trailing return, buy the winners and sell the losers (Jegadeesh &
Titman, 1993). Time-series momentum asks *"is this asset trending?"*. CSMOM asks
*"is it trending more than its peers?"*. It is the natural extension of a TSMOM
study on the same universe, and the comparison separates momentum's two sources
of return.

## Signal

With lookback $L$ and skip $S$ as in [TSMOM](momentum-ts.md), on each rebalance
date $t$ (every `rebal_freq` bars) the momentum returns $m_{i,t}$ of the $N_t$
assets **listed at $t$** are ranked. With $q^{+}$ and $q^{-}$ the
`top_quantile` / `bottom_quantile` fractions,

$$
s_{i,t} =
\begin{cases}
+1 & m_{i,t} \ge Q_{1-q^{+}}(m_{\cdot,t}) \\
-1 & m_{i,t} \le Q_{q^{-}}(m_{\cdot,t}) \\
\phantom{+}0 & \text{otherwise,}
\end{cases}
\qquad
w_{i,t} = s_{i,t}\min\!\left(\frac{\sigma^*}{\hat\sigma_{i,t}},\lambda\right),
$$

where $Q_p$ is the cross-sectional $p$-quantile. Positions are carried between
rebalance dates. Each position is volatility-targeted as in TSMOM, so the book is
long/short in *names* but not necessarily dollar- or beta-neutral.

**Universe membership.** Members of an index universe list and delist at
different times. The engine aligns all symbols on the *union* of dates, and a
symbol is null (NaN) before its first and after its last bar. Its momentum is
then NaN, so it is simply absent from that date's ranking. A symbol whose data
ends is flattened on its last bar.

## TSMOM vs CSMOM

Following Moskowitz, Ooi & Pedersen (2012) and Goyal & Jegadeesh (2018), write
TSMOM's position as the cross-sectional deviation plus the average:

$$
\operatorname{sign}(m_{i,t}) \;\approx\;
\underbrace{\big(\operatorname{sign}(m_{i,t}) - \overline{\operatorname{sign}(m_{\cdot,t})}\big)}_{\text{cross-sectional (CSMOM-like)}}
\;+\;
\underbrace{\overline{\operatorname{sign}(m_{\cdot,t})}}_{\text{net market timing}} .
$$

TSMOM on stocks therefore carries a **time-varying net market exposure**: net
long after rallies, net short after sell-offs. CSMOM removes most of it. Goyal &
Jegadeesh find that this net exposure explains most of TSMOM's apparent edge over
CSMOM. Comparing the two reports on the same universe shows how much of the
TSMOM result is relative strength and how much is market timing.

## Economic rationale

* **Under-reaction to firm-specific news.** Earnings and other information is
  incorporated slowly, more so for stocks with less analyst coverage (Hong, Lim &
  Stein, 2000). Winners keep winning for 3–12 months.
* **Momentum crashes.** The long/short book is short past losers, which after a
  market crash are high-beta stocks. In a sharp rebound those losers rally the
  most and the strategy suffers its largest losses (Daniel & Moskowitz, 2016;
  2009 is the canonical case). Volatility scaling mitigates but does not remove
  this. Expect negative skew.

## Parameters

| Parameter | Role | Typical range |
|---|---|---|
| `lookback` | Ranking horizon $L$, bars | 126–252 |
| `skip_recent` | Bars skipped, $S$ | 0 or 21 |
| `top_quantile` / `bottom_quantile` | Fraction held long / short | 0.1–0.3 |
| `vol_target` / `vol_lookback` / `max_leverage` | Per-position risk sizing | 0.10 / 63 / 2 |
| `rebal_freq` | Bars between re-rankings | 5, 21 |

## Hypotheses a report tests

1. A positive, deflation-robust out-of-sample Sharpe after costs (CSMOM in
   US large caps has been weak since the 2000s, so a null result is plausible).
2. Lower correlation to the benchmark than TSMOM on the same universe.
3. Negative skew and the largest drawdowns in market rebounds.

## References

* Daniel, K. & Moskowitz, T. (2016). Momentum crashes. *Journal of Financial Economics* 122(2).
* Goyal, A. & Jegadeesh, N. (2018). Cross-sectional and time-series tests of return predictability: what is the difference? *Review of Financial Studies* 31(5).
* Hong, H., Lim, T. & Stein, J. (2000). Bad news travels slowly: size, analyst coverage, and the profitability of momentum strategies. *Journal of Finance* 55(1).
* Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance* 48(1).
* Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics* 104(2).
