# SnippyScales

A modular Python–Rust quantitative trading research and execution platform.

## Architecture

```
python/          # Research, ML, orchestration, CLI
rust/
  algo-core/     # Pure Rust: backtesting engine, execution, market types
  algo-pyo3/     # PyO3 bindings — exposes Rust to Python via maturin
docs/            # Architecture decisions, API reference, guides
data/            # Local data (gitignored; raw + derived)
```

## Layers

| Layer | Language | Responsibility |
|---|---|---|
| Research | Python | Feature engineering, ML, prototyping |
| Strategy Logic | Python → Rust (gradual) | Signal generation |
| Backtesting Engine | Rust (`algo-core`) | Event-driven, fast, realistic |
| Execution Engine | Rust (`algo-core`) | Order routing, risk checks |
| Python Bindings | Rust (`algo-pyo3`) | PyO3/maturin bridge |
| CLI + Orchestration | Python (Typer) | Project control |

## Quickstart

```bash
# 1. Install tooling
curl -LsSf https://astral.sh/uv/install.sh | sh                           # uv
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh  # prek
cargo install git-cliff                                                     # changelog

# 2. Set up the repo
uv sync
uv run maturin develop    # build Rust extension (dev mode)
prek install              # activate git hooks from prek.toml

# 3. Use the CLI
uv run algo --help
```

## Development

```bash
# Python: lint, format, type-check, test
uv run ruff check . --fix && uv run ruff format .
uv run ty check
uv run pytest

# Rust: fmt, clippy, test, bench
cargo fmt --manifest-path rust/Cargo.toml
cargo clippy --manifest-path rust/Cargo.toml --workspace -- -D warnings
cargo test --manifest-path rust/Cargo.toml --workspace
cargo bench --manifest-path rust/Cargo.toml --workspace

# Run all git hooks manually
prek run --all-files
```

## Commit Convention

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/).
The `commit-msg` hook (prek) enforces this automatically on every `git commit`.

```
feat(backtest): add vol-adjusted slippage model    ← semver MINOR
fix(portfolio): handle zero-quantity close         ← semver PATCH
feat!: redesign Portfolio API                      ← semver MAJOR (breaking)
```

Allowed types: `feat`, `fix`, `perf`, `refactor`, `revert`, `docs`, `style`,
`test`, `build`, `ci`, `chore`

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Changelog

`CHANGELOG.md` is generated automatically by [git-cliff](https://git-cliff.org/)
from the conventional commit history. Never edit it by hand.

```bash
git cliff --unreleased          # preview next release
git cliff --unreleased --tag v1.2.0   # preview with a specific tag
git cliff -o CHANGELOG.md       # regenerate full changelog
```

Releases are triggered via the **Release** GitHub Actions workflow (manual
dispatch), which generates the changelog, bumps `pyproject.toml`, opens a PR,
and publishes to PyPI + GitHub Releases on merge.
