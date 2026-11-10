# Quickstart

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [just](https://github.com/casey/just) — `uv tool install rust-just` or your package manager

## Setup

```bash
git clone https://github.com/jstammers/snippy-scales
cd snippy-scales

# Install Python deps and activate git hooks
just setup
```

That command is equivalent to:
```bash
uv sync --all-extras              # install Python dependencies
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
just fmt                 # format
just lint                # lint
just type-check          # type checking
just test                # all tests

# Specific to Python
just test-py             # just Python tests
just test-py-cov         # Python tests with coverage (HTML report)

# Build the sdist and wheel
just build
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

# Rebuild docs
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

## Troubleshooting

### `uv sync` refuses to install `nautilus_trader`

The project pins a NautilusTrader release candidate (`2.0.0rc6`). An exact
pre-release pin is allowed by uv's default resolution; if you have
`UV_PRERELEASE=disallow` set, unset it.

## Next Steps

1. Read [Architecture Overview](../architecture/overview.md) to understand the codebase
2. Read [Backtesting Engine](../architecture/backtest_engine.md) to learn how to run backtests
3. Read [Adding a Strategy](adding_strategy.md) to implement your first strategy
4. Read [API Reference](../api/python/index.md) for Python API docs

## Getting Help

- Check the [Contributing Guide](../CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub if something isn't working
- Run `just --help` to see all available commands
