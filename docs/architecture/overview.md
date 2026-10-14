# Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      snippy_scales                        │
│  research/  │  strategies/  │  evaluation/  │  cli/       │
└───────────────────────┬──────────────────────────────────┘
                        │ target positions, bars
┌───────────────────────▼──────────────────────────────────┐
│                     NautilusTrader                        │
│  ParquetDataCatalog │ BacktestEngine │ LiveNode + adapters │
└──────────────────────────────────────────────────────────┘
```

Our code handles research velocity: features, signals and evaluation.
NautilusTrader (Rust core, Python API) provides the data catalog, the
event-driven backtest engine and live execution. The migration is in progress —
see [the migration plan](nautilus_migration.md) for what has moved so far.
