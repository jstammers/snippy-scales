# Trend and carry on futures (`trend-carry`)

`trend-carry` holds two positions in every futures contract: one that follows the
contract's own trend, and one that follows the slope of its futures curve relative
to its peers. The two are known to be nearly uncorrelated, so the blend should earn
a higher Sharpe ratio than either alone.

## Signals

**Trend** is time-series momentum (Moskowitz, Ooi & Pedersen, 2012; see
[`momentum-ts` on futures](momentum-ts-futures.md)): the sign of the return over the
past $L$ bars.

**Carry** is the return a future earns if prices do not change (Koijen, Moskowitz,
Pedersen & Vrugt, 2018, *Carry*). A futures position costs nothing to enter, so its
carry is the rate at which the contract rolls up or down the curve towards spot.
With $F^1_t$ and $F^2_t$ the closes of the front and second contract and $m$ the
months between their delivery months,

$$
c_{i,t} = \left(\frac{F^1_{i,t}}{F^2_{i,t}} - 1\right)\frac{12}{m}.
$$

* It is positive when the curve is in backwardation and negative in contango.
* It means something different in each asset class: dividend yield less the
  financing rate for equity indices, the interest-rate differential for
  currencies, the slope of the yield curve less financing for bonds, and
  convenience yield less storage for commodities.
* `load_market_data` computes it for every `<ROOT>.ROLL` symbol
  (`data.futures.roll_yield`) and passes it to the strategy as the `carry` column.
  Only closes at which both contracts traded are used.

The carry signal is **relative within an asset class**. Carry is averaged over
`carry_smooth` bars ($\bar c$; one year removes the seasonal shape of gas, grain and
livestock curves) and compared with the mean of the contract's asset class $g(i)$:

$$
s_{i,t} = (1 - w)\,\operatorname{sign}\!\left(r_{i,t-L,t}\right)
        + w\,\operatorname{sign}\!\left(\bar c_{i,t} - \bar c_{g(i),t}\right).
$$

The relative form takes no view on an asset class as a whole. That is left to
trend. It also cancels what the members of a class share, such as the level of US
interest rates in every equity-index and metal curve. A contract with no peer that
has carry on a date gets a zero carry signal.

## Sizing and rebalancing

Each daily target is sized to the same risk, as in MOP:
$s_{i,t}\,\min(\sigma^*/\hat\sigma_{i,t}, \lambda)$ with $\sigma^* = 40\%$,
$\hat\sigma$ a `vol_lookback`-bar standard deviation floored at `vol_floor`, and
capital split equally over the instruments.

The position held is the **average of the last $H$ = `rebal_freq` daily targets**.
This is the book of $H$ tranches, each rebalanced every $H$ bars on a different
day. A single monthly rebalance makes the result depend on which day of the month
it falls on, and for slow signals on a few dozen instruments that choice alone can
move a decade's Sharpe ratio by several tenths. Tranching earns the average over
the possible rebalance days and removes the luck.

## Long-or-flat groups

`long_only` names asset classes whose trend signal is clipped at zero: a negative
trend holds no position instead of a short one. Equity indices and bonds pay a risk
premium for being held, so a short position gives that premium up, and equity
sell-offs tend to reverse faster than a slow trend signal can follow. The carry signal
is unchanged. With `long_only = "equity"` the strategy is long equities in an uptrend
and out of them in a downtrend, while still trading currencies and commodities both
ways.

## Portfolio volatility target

Sizing each position to 40% volatility fixes every position's risk but not the
book's. When trends line up across markets (most contracts short in a crash, or
long in a broad rally) the positions become correlated and the book's risk rises,
often just before trends reverse. With `portfolio_vol_target` $= \sigma_P^*$ the whole
book is scaled each bar by

$$
k_t = \min\!\left(\frac{\sigma_P^*}{\sqrt{p_t^\top \hat\Sigma_t\, p_t}},\ k_{\max}\right),
$$

where $p_t$ is the book (positions over the number of symbols) and $\hat\Sigma_t$ the
annualised sample covariance of the last `cov_lookback` bars of returns. The risk
is **ex ante**: it is that of today's positions, not of the positions held over the
past months, so it reacts as soon as the book changes. All positions share one
scale, so the mix of positions is unchanged.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `lookback` | 252 | Trend window in bars |
| `carry_weight` | 0.5 | Weight $w$ of carry; `0` is trend alone, `1` carry alone |
| `carry_smooth` | 252 | Bars the roll yield is averaged over |
| `vol_target` | 0.40 | Annualised volatility target per instrument |
| `vol_lookback` | 63 | Volatility window in bars |
| `vol_floor` | 0.02 | Lowest volatility assumed |
| `max_leverage` | 50 | Cap on `vol_target / volatility` |
| `rebal_freq` | 21 | Holding period $H$ in bars (number of tranches) |
| `groups` | asset class | Peer group of each symbol for the carry signal |
| `long_only` | "" | Asset classes traded long-or-flat, comma-separated |
| `portfolio_vol_target` | 0 (off) | Annualised volatility the whole book is scaled to |
| `cov_lookback` | 126 | Bars of returns behind the book's ex-ante volatility |
| `max_portfolio_scale` | 2 | Cap on the book-level scale $k_{\max}$ |

## What to expect

* Koijen et al. report a Sharpe ratio of about 0.7–0.9 for a diversified carry
  portfolio over 1983–2012, with a low correlation to trend. Carry loses in
  flights to safety (2008), which is when trend earns most.
* The carry measure here uses only the first two contracts, so it is noisy where
  the second month is thin, and its asset-class groups are small (three metals,
  five bonds).
* The `carry` column needs rolled futures (`.ROLL` symbols). On other data the
  strategy is tranched time-series momentum.

(Literature figures are summarised from memory, not re-verified against the papers.)
