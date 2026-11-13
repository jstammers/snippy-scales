# Moving-average trend following (`trend-following`)

The oldest systematic trend rule: be long while a fast moving average of price is
above a slow one, short while it is below. It is the "dual moving average"
filter of the managed-futures industry. It is closely related to
[time-series momentum](momentum-ts.md): both are weighted sums of past returns,
differing only in the weights.

## Signal

With simple moving averages over $f < s$ bars,

$$
\mathrm{MA}^{(n)}_{i,t} = \frac{1}{n}\sum_{k=0}^{n-1} P_{i,t-k},
\qquad
d_{i,t} = \operatorname{sign}\!\left(\mathrm{MA}^{(f)}_{i,t} - \mathrm{MA}^{(s)}_{i,t}\right).
$$

**Relation to momentum.** Write prices as cumulated returns. A moving-average
difference is then a linear filter of past returns,
$\mathrm{MA}^{(f)}_t - \mathrm{MA}^{(s)}_t \approx \sum_{k\ge 0} \omega_k\, \Delta P_{t-k}$,
with weights $\omega_k$ that rise from 0, plateau, and fall back to 0 at lag $s$
(a trapezoid). TSMOM's $\operatorname{sign}(P_t/P_{t-L}-1)$ is the same idea with
flat weights on lags $0..L-1$. Levine & Pedersen (2016) show that most trend
rules are such return filters and perform similarly once matched on horizon. So
the crossover's effective horizon is roughly $(f+s)/2$, and results should be
compared with TSMOM at a similar lookback.

## Sizing

Identical in spirit to TSMOM: each position targets annualised volatility
$\sigma^*$ from the realised volatility over `vol_lookback` bars, capped at 5×:

$$
w_{i,t} = d_{i,t}\,\min\!\left(\frac{\sigma^*}{\hat\sigma_{i,t}},\, 5\right),
$$

and the engine weights each of the $N$ symbols by $1/N$. The signal is re-sized
every bar, so turnover is higher than a `rebal_freq`-throttled TSMOM. A
crossover's sign flips less often than a short-lookback momentum sign, though.

## Economic rationale

The same as TSMOM's: slow diffusion of information and under-reaction create
trends; frictions and flows sustain them; trend followers are paid for bearing
reversal risk (see [`momentum-ts`](momentum-ts.md)). The moving-average form
smooths the signal. Fewer sign flips on noise mean lower costs, at the price of
a slower response to genuine turns.

## Parameters

| Parameter | Role | Typical range |
|---|---|---|
| `fast_period` | Fast MA length, bars | 5–60 |
| `slow_period` | Slow MA length, bars | 60–250 |
| `vol_target` | Per-position annualised volatility target | 0.10 |
| `vol_lookback` | Realised-volatility window, bars | 20–63 |

Searches should keep `fast_period < slow_period`. Trials where they cross
produce an inverted (contrarian) rule, which the parameter-sensitivity plots
make visible.

## Hypotheses a report tests

1. A positive, deflation-robust out-of-sample Sharpe after costs.
2. A broad plateau over effective horizon $(f+s)/2$, comparable to the best TSMOM
   lookback.
3. Lower turnover than daily-rebalanced TSMOM for similar Sharpe.

## References

* Levine, A. & Pedersen, L. H. (2016). Which trend is your friend? *Financial Analysts Journal* 72(3).
* Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management* 44(1).
* Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics* 104(2).
