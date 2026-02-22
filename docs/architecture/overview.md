# Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Python Layer                         │
│  Research  │  CLI (Typer)  │  Strategy Logic  │  Data   │
└──────────────────────┬──────────────────────────────────┘
                       │ PyO3 / maturin
┌──────────────────────▼──────────────────────────────────┐
│                      Rust Layer                          │
│  algo-core: Backtest Engine │ Execution │ Portfolio      │
└─────────────────────────────────────────────────────────┘
```

The Python layer handles research velocity; the Rust layer provides determinism, speed, and memory safety for the backtest and execution critical path.
