# Time-series momentum (`momentum-ts`)

Time-series momentum (TSMOM) bets that an asset's own recent trend persists: hold
it long after a positive trailing return and short after a negative one, sized so
each position carries the same ex-ante risk. It is the core signal of most
systematic macro / managed-futures programmes (Moskowitz, Ooi & Pedersen, 2012;
Hurst, Ooi & Pedersen, 2017). Here it is applied **per stock** to a broad equity
universe, so the portfolio is a large, equally weighted book of independent trend
bets.

## Signal

For asset $i$ at bar close $t$, with lookback $L$ bars and the most recent $S$
bars skipped, the momentum return is

$$
m_{i,t} = \frac{P_{i,\,t-S}}{P_{i,\,t-S-L}} - 1,
\qquad
s_{i,t} = \operatorname{sign}(m_{i,t}) \in \{-1, 0, +1\}.
$$

Skipping $S \approx 21$ bars (the classic "12–1" formulation) removes the
one-month **short-term reversal** that contaminates raw trailing returns
(Jegadeesh, 1990).

## Sizing: volatility targeting

Each position targets an annualised volatility $\sigma^*$ using realised
volatility over $V$ bars:

$$
\hat\sigma_{i,t} = \sqrt{252}\;\operatorname{std}\!\left(r_{i,t-V+1},\dots,r_{i,t}\right),
\qquad
w_{i,t} = s_{i,t}\,\min\!\left(\frac{\sigma^*}{\hat\sigma_{i,t}},\; \lambda\right),
$$

where $r_{i,t}$ are daily simple returns and $\lambda$ caps leverage
(`max_leverage`). Positions are re-sized only every `rebal_freq` bars and carried
in between, which trades signal freshness against turnover.

The engine gives each of the $N$ symbols capital weight $1/N$, so the portfolio
return over the next bar is

$$
R_{t+1} = \frac{1}{N}\sum_{i=1}^{N} w_{i,t}\, r_{i,t+1} \;-\; \text{costs}_t .
$$

**Portfolio risk.** With equal per-name risk $\sigma^*$, average pairwise return
correlation $\bar\rho$ and every signal pointing the same way, the book's
volatility is

$$
\sigma_p = \frac{\sigma^*}{N}\sqrt{N + N(N-1)\bar\rho} \;\xrightarrow{N\to\infty}\; \sigma^*\sqrt{\bar\rho}.
$$

For US large caps ($\bar\rho \approx 0.3$) and $\sigma^* = 10\%$ that is at most
about 5.5%, and lower when signals disagree. Sharpe ratios are invariant to this
scale. Absolute returns and drawdowns in a report should be read against it, or
levered to a common volatility before comparison with a benchmark.

## Economic rationale

Two families of explanation predict positive autocorrelation in returns at
horizons of months:

* **Behavioural under-reaction, then over-reaction.** Investors anchor on old
  information and update slowly (conservatism), so prices drift in the direction
  of news (Barberis, Shleifer & Vishny, 1998). Information diffuses gradually
  across investors, and momentum traders chasing the drift later overshoot
  (Hong & Stein, 1999). Trends therefore persist for months and partially
  reverse over years. The persistence is the source of TSMOM's return.
* **Risk transfer and frictions.** Slow-moving capital, hedging pressure and
  institutional rebalancing make price moves serially correlated. Trend followers
  are paid for supplying liquidity in the direction of the flow.

TSMOM's payoff is **convex** in the underlying's move. It is long large rallies
and short large sell-offs, which resembles a long straddle. Historically it has
been profitable in prolonged crises ("crisis alpha"; Hurst, Ooi & Pedersen,
2017). Its weak spots are sharp **trend reversals**. A V-shaped crash and
recovery (e.g. March–April 2020) can whipsaw it twice.

On **individual stocks** the evidence is weaker than on diversified futures
because idiosyncratic noise dilutes the trend. Lim, Wang & Yao (2018) nonetheless
document time-series momentum in nearly a century of US stock returns.
Goyal & Jegadeesh (2018) show that TSMOM's edge over cross-sectional momentum
comes largely from its time-varying **net market exposure**. A TSMOM book on
stocks is partly a market-timing strategy (see [`momentum-cs`](momentum-cs.md)).

## Parameters

| Parameter | Symbol | Role | Typical range |
|---|---|---|---|
| `lookback` | $L$ | Trend horizon, bars | 21–252 |
| `skip_recent` | $S$ | Bars skipped to avoid short-term reversal | 0 or 21 |
| `vol_target` | $\sigma^*$ | Per-position annualised volatility target | 0.10 |
| `vol_lookback` | $V$ | Realised-volatility window, bars | 21–126 |
| `max_leverage` | $\lambda$ | Cap on $\sigma^*/\hat\sigma$ | 2 |
| `rebal_freq` | — | Re-size every N bars | 1, 5, 21 |

## Hypotheses a report tests

1. The out-of-sample Sharpe is positive after costs, and its **Deflated Sharpe**
   (corrected for the number of parameter sets tried) is significant.
2. Performance is a **plateau** over $L$ (any medium-term horizon works), not a
   spike at one value. A spike is the signature of overfitting, and the PBO
   quantifies it.
3. Longer `rebal_freq` cuts turnover and costs without destroying the signal.
4. Fold Sharpes are positive in most regimes. Expect weak or negative folds
   around sharp reversals.

## References

* Barberis, N., Shleifer, A. & Vishny, R. (1998). A model of investor sentiment. *Journal of Financial Economics* 49(3).
* Goyal, A. & Jegadeesh, N. (2018). Cross-sectional and time-series tests of return predictability: what is the difference? *Review of Financial Studies* 31(5).
* Hong, H. & Stein, J. (1999). A unified theory of underreaction, momentum trading, and overreaction in asset markets. *Journal of Finance* 54(6).
* Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management* 44(1).
* Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance* 45(3).
* Lim, B. Y., Wang, J. & Yao, Y. (2018). Time-series momentum in nearly 100 years of stock returns. *Journal of Banking & Finance* 97.
* Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics* 104(2).
