# Quickstart

## Prerequisites

- Rust (stable, ≥ 1.78) — [rustup.rs](https://rustup.rs)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [just](https://github.com/casey/just) — `cargo install just` or your package manager

## Setup

```bash
git clone https://github.com/jstammers/snippy-scales
cd snippy-scales

# Install tooling (Python deps, build Rust extension, activate git hooks)
just setup
```

That command is equivalent to:
```bash
uv sync                           # install Python dependencies
just dev                          # build Rust extension in dev mode
prek install                      # activate git hooks
```

## Verify Installation

```bash
# Check that the CLI works
just algo --help

# List available just commands
just
```

## First Backtest

```bash
# Ingest some data (requires DATABENTO_API_KEY env var)
just algo data ingest GLBX.MDP3 --symbol ES.c.0 --start 2020-01-01 --end 2024-12-31

# Run a backtest on the trend-following strategy
just algo backtest run trend_following --symbol ES.c.0
```

## Development Workflow

Most of your work will use these commands:

```bash
# Quick format + lint + typecheck + test (all in one)
just check

# Or individual commands
just fmt                 # format Python and Rust
just lint                # lint Python and Rust
just type-check          # Python type checking
just test                # all tests (Python + Rust)

# Specific to Python
just test-py             # just Python tests
just test-py-cov         # Python tests with coverage (HTML report)

# Specific to Rust
just test-rs             # just Rust tests
just bench               # run benchmarks

# Rebuild after Rust changes
just dev                 # build Rust extension in dev mode
just build               # build release wheel
```

## Running Backtests

```bash
# Run the CLI
just algo backtest run <strategy> [OPTIONS]

# Example: backtest the trend-following strategy
just algo backtest run trend_following --symbol ES.c.0 --start 2020-01-01 --end 2024-12-31

# List available strategies
just algo backtest strategies
```

## Adding a Strategy

See [Adding a Strategy](adding_strategy.md) for a detailed guide.

Quick version:
1. Create `python/snippy_scales/strategies/my_strategy.py`
2. Implement the `Strategy` interface
3. Register in `__init__.py`
4. Test with `just test-py`
5. Backtest with `just algo backtest run my_strategy`

## Documentation

```bash
# Build and serve docs locally (http://localhost:8000)
just docs-serve

# Rebuild docs (MkDocs + Rust API docs)
just docs-build

# Preview unreleased changes in changelog
just changelog
```

## Common Tasks

### Run all pre-commit hooks manually

```bash
just check-all
```

This runs all checks that will be executed when you commit.

### Clean build artifacts

```bash
just clean
```

### Rebuild Rust extension

```bash
just dev              # development (fast compile, debug symbols)
just build            # release (slow compile, optimized binary)
```

## Troubleshooting

### "maturin develop" fails

Make sure Rust is installed:
```bash
rustup update
rustc --version
```

Then rebuild:
```bash
just dev
```

### "cargo fmt" fails with "Failed to find targets"

Use `--all` to format all workspace members:
```bash
just fmt-rs
```

Or manually:
```bash
cargo fmt --all --manifest-path rust/Cargo.toml
```

### Python imports fail after changing Rust code

Restart the kernel and rebuild the extension:
```bash
just dev
```

## Next Steps

1. Read [Architecture Overview](../architecture/overview.md) to understand the codebase
2. Read [Backtesting Engine](../architecture/backtest_engine.md) to learn how to run backtests
3. Read [Adding a Strategy](adding_strategy.md) to implement your first strategy
4. Read [API Reference](../api/python/index.md) for Python API docs
5. Read [Rust API Docs](../api/rust) for Rust API docs

## Getting Help

- Check the [Contributing Guide](../CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub if something isn't working
- Run `just --help` to see all available commands
