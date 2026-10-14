# SnippyScales

A quantitative trading research platform built on
[NautilusTrader](https://github.com/nautechsystems/nautilus_trader).

## Design Principles

- Clear separation of research (ours) and execution (NautilusTrader)
- Same strategy code in backtests and live trading
- Volatility as a first-class input to all sizing decisions
- Incremental evolution in strategy sophistication
- Avoid unnecessary complexity early; build for scale from the outset

## Quick Links

- [Quickstart](guides/quickstart.md)
- [Architecture](architecture/overview.md)
- [Python API](api/python/index.md)
