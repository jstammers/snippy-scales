# Options, crypto, news and arbitrage: what is feasible from the UK

**Question.** Beyond futures and US stocks, can this project feasibly research and
trade (a) options, (b) cryptocurrencies, (c) signals from news and social media
(Reddit, X), and (d) arbitrage, as a UK retail trader with the current stack?

**Short answer.** Ranked by value for effort:

| Area | Can a UK retail account trade it? | Can the stack research it today? | Verdict |
|---|---|---|---|
| Crypto, spot | Yes | Nearly: needs a data loader | **Do first.** Cheap, but long-or-flat only |
| News and events | Yes (it trades stocks) | No: needs timestamped text data | **Research only**, as an event study, before any strategy |
| Options | Yes, through a broker such as IBKR | No: no chain, strike or quote handling | **Later.** Get a data quote first; the build is large |
| Crypto derivatives | **No** (FCA retail ban) | — | Not available |
| True arbitrage | In principle | — | **Not realistic** at retail speed and fees |

Checked in November 2026 by web search. Regulatory and pricing statements come from
the sources listed at the end, several of them secondary: confirm with the FCA, the
broker or the vendor before acting. Items marked *(from memory)* were not re-checked.
Nothing here is tax or legal advice.

## Options

**Access.** Listed options on US stocks, indices and futures are available to UK
retail clients through brokers such as Interactive Brokers, subject to the broker's
options permissions *(from memory)*. Whether a given US ETF option is blocked by the
PRIIPs key-information rules is unsettled in the sources found; check per product.

**Data is the constraint.**

- US equity options are distributed through OPRA, and the feed is very large.
  Databento sells OPRA history pay-as-you-go by size. A full-chain history is costly;
  ask for the price of the exact request before buying (`metadata.get_cost` is
  free).
- Options on futures (for example on ES) are part of the CME dataset this project
  already uses *(from memory: verify the symbology and cost)*. Daily settlement
  data for one root is the cheapest way to get a real options history.
- Daily bars are not enough to model fills. Option spreads are wide, so a backtest
  needs quotes, or at least settlement prices with a spread assumption.

**Engine.** NautilusTrader has option instruments, option chains and Greeks (added
in v1.225, April 2026), with a local Black-Scholes calculator for backtests. This
project's own layers do not: strategies return one target per symbol in units of
equity, the vector engine and `SignalStrategy` fill at bar closes, and nothing
selects strikes or expiries. An options strategy needs a contract-selection layer,
an implied-volatility surface, and a quote-based fill model.

**What to test if built.** The best-documented options return is the volatility
risk premium: implied volatility exceeds realised volatility on average, so sellers
of index options are paid, with large losses in crashes. It is a negatively skewed
return, the opposite of trend following's, so the two combine well in principle.

**Verdict.** Tradable, but the research cost is high. First step: price a daily
history of ES options from Databento. Do not build the engine before that.

## Cryptocurrency

**What a UK retail trader may trade.**

- **Spot: yes.**
- **Derivatives: no.** The FCA banned the sale of crypto derivatives (CFDs, futures,
  options, perpetual futures) to retail clients in January 2021, and the ban remains.
  That removes shorting, leverage, and the funding-rate and futures-basis trades that
  most "crypto arbitrage" refers to. The same ban is why a UK retail broker account
  cannot trade CME bitcoin futures *(from memory)*.
- **Exchange-traded notes: yes**, since 8 October 2025.
- Elective professional status lifts the ban, but its tests (portfolio size, trading
  frequency, professional experience) are demanding *(from memory)*.

**Venues.** NautilusTrader ships adapters for Kraken, Coinbase, Binance, Bybit and
OKX, and a sandbox adapter for paper trading. Binance has not taken new UK customers
since October 2023. Kraken's and Coinbase's UK spot service was not verified in this
search. The UK's full cryptoasset regime starts on 25 October 2027 (applications
from 30 September 2026): a venue that is not authorised by then cannot serve UK
users, so venue choice may change.

**What it needs here.**

- A loader that writes exchange daily bars into the catalog. Exchanges publish
  historical candles free through their public APIs *(from memory)*.
- Crypto trades every day. `TRADING_DAYS_PER_YEAR = 252` understates its annualised
  volatility by about 17%, so volatility targets need the right factor.
  `summarize_study` already annualises by the observed frequency.
- Strategies must be long-or-flat. `TrendCarry` with `carry_weight = 0` and negative
  targets clipped to zero is the natural first test.

**Tax.** Each disposal of a cryptoasset is a capital-gains event, including
crypto-to-crypto trades, with a £3,000 annual allowance in 2026/27 (secondary
sources). A systematic strategy produces many disposals, so record keeping is a real
cost.

**Verdict.** Feasible and cheap to start. Expect a long-or-flat trend strategy on a
few liquid coins: a new, volatile risk asset for the trend book, not a market-neutral
strategy.

## News and social media

**Data access.**

- **X.** The API is pay-per-use, at about $0.005 per post read, with no free tier
  (secondary sources). A backtest needs years of history, so cost grows quickly:
  100,000 posts is about $500.
- **Reddit.** The free tier allows 100 queries a minute for non-commercial use under
  OAuth. Commercial use needs a contract with Reddit. Whether trading one's own
  account counts as commercial is for Reddit's terms to answer: read them. Bulk
  historical archives are no longer openly available *(from memory)*.
- **Cheaper sources with timestamps.** Alpaca's news API (included with the account
  this project already has, history from about 2015), SEC EDGAR filings, and GDELT
  *(all from memory)*.

**Two traps specific to text signals.**

1. **Timestamps.** A daily-bar backtest that uses a post published at 15:59 to trade
   that day's close has look-ahead. Text must be stored with its publication time
   and joined strictly before the bar it trades.
2. **Language models know the future.** Scoring historical headlines with a model
   whose training data ends after those headlines leaks the outcome into the score.
   Only headlines published after the model's cutoff give a clean test.

**Evidence.** The published effects of social-media sentiment on returns are small,
last hours to days, and are concentrated in small, expensive-to-trade stocks
*(from memory)*. That is a poor fit for daily bars and retail costs.

**Verdict.** Do not start with a strategy. Start with an event study: take Alpaca
news for the S&P 500 universe, measure the return over the following one to five
days by headline type, and only build a strategy if an effect survives costs.

## Arbitrage

- **True arbitrage** (the same asset at two prices, cross-exchange or index against
  constituents) is won on speed, fees and capital. A retail account on daily or
  minute bars will not capture it.
- **Crypto funding and basis trades** need perpetuals or futures, which UK retail
  clients cannot trade.
- **Statistical arbitrage** (pairs, short-term reversal) is not arbitrage: it is a
  risk-bearing strategy that needs cheap shorting and low costs. See the research log
  for what has been screened on the S&P 500 universe.
- **Carry** is the arbitrage-like return that is realistic here: being paid to hold
  the futures curve. `trend-carry` already trades it.
- **Betting exchanges.** NautilusTrader has a Betfair adapter. It is a different
  domain with its own data and models; noted only because the adapter exists.

**Verdict.** No true arbitrage is realistic. Risk premia that resemble it (carry)
are, and one is built.

## Sources

- [FCA reopens retail access to crypto ETNs (Charles Russell Speechlys)](https://www.charlesrussellspeechlys.com/en/insights/expert-insights/financial-services/2025/fca-reopens-retail-access-to-crypto-etns-a-strategic-turning-point-for-regulated-crypto-exposure/)
- [Is perp futures trading still legal for UK retail? (OneKey, 2026)](https://onekey.so/blog/ecosystem/is-perp-futures-trading-still-legal-for-uk-retail-2026-fca-ban-status-real-workarounds/)
- [UK crypto regulation 2026: what investors need to know (IG)](https://www.ig.com/uk/trading-strategies/uk-new-fca-crypto-rules-2026-what-investors-need-to-know-260706)
- [FCA publishes final rules on cryptoassets (DLA Piper)](https://www.dlapiper.com/en-cn/insights/publications/2026/07/uk-cryptoassets-and-digital-markets-developments-fca-publishes-final-rules-on-cryptoassets)
- [Binance stops accepting new UK customers (Finextra)](https://finextra.com/newsarticle/43130/binance-stops-accepting-new-uk-customers)
- [Binance plans UK return with FCA licence bid (crypto.news)](https://crypto.news/binance-plans-uk-return-with-fca-license-bid/)
- [Databento OPRA dataset](https://databento.com/catalog/opra/OPRA.PILLAR)
- [Databento: new OPRA pricing plans](https://databento.com/blog/introducing-new-opra-pricing-plans)
- [NautilusTrader: options](https://nautilustrader.io/docs/latest/concepts/options/)
- [NautilusTrader: Greeks](https://nautilustrader.io/docs/latest/concepts/greeks/)
- [NautilusTrader v1.225.0 release notes](https://newreleases.io/project/github/nautechsystems/nautilus_trader/release/v1.225.0)
- [IBKR: PRIIPs regulation](https://ibkb.interactivebrokers.com/node/2993)
- [X API pricing in 2026 (postproxy)](https://www.postproxy.dev/blog/x-api-pricing-2026/)
- [X API pay-per-use explained (opentweet)](https://opentweet.io/how-to/x-api-pay-per-use-explained)
- [Reddit API pricing 2026 (data365)](https://data365.co/blog/reddit-api-pricing)
- [Reddit API key, limits and pricing (socialcrawl)](https://www.socialcrawl.dev/blog/reddit-api-key-limits-alternatives-2026)
- [Crypto tax UK 2026/27 (Waltio)](https://www.waltio.com/uk/crypto-tax-uk/)
- [Day trading tax UK (CMC Markets)](https://www.cmcmarkets.com/en-gb/trading-guides/day-trading-tax-guide)
