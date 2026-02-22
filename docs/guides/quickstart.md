# Quickstart

## Prerequisites

- Rust (stable, ≥ 1.78) — [rustup.rs](https://rustup.rs)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup

```bash
git clone https://github.com/yourorg/snippy-scales
cd snippy-scales

# Install Python dependencies
uv sync

# Build the Rust extension and install into the venv
uv run maturin develop --release
```

## First backtest

```bash
# Ingest some data (requires DATABENTO_API_KEY env var)
uv run algo data ingest GLBX.MDP3 --symbol ES.c.0 --start 2020-01-01 --end 2024-12-31

# Run a backtest
uv run algo backtest run trend_following
```

## Development loop

```bash
# Lint + format Python
uv run ruff check . --fix && uv run ruff format .

# Type check
uv run ty check

# Test Python
uv run pytest

# Test + lint Rust
cargo clippy --workspace -- -D warnings
cargo test --workspace
```
