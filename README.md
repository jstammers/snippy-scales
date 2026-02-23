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
cargo install just                                                          # task runner

# 2. Set up the repo
just setup                # install deps, build Rust extension, activate git hooks

# 3. Use the CLI
just algo --help          # or: uv run algo --help
```

## Development

```bash
# Quick commands
just                      # list all available commands
just check                # run all checks (format, lint, type-check, test)
just test                 # run all tests (Python + Rust)
just fmt                  # format all code (Python + Rust)

# Python-specific
just lint-py              # ruff check with auto-fix
just fmt-py               # ruff format
just type-check           # ty check
just test-py              # pytest
just test-py-cov          # pytest with coverage report

# Rust-specific
just fmt-rs               # cargo fmt
just lint-rs              # cargo clippy
just test-rs              # cargo test
just bench                # cargo bench

# Other
just check-all            # run all pre-commit hooks (prek)
just clean                # remove build artifacts
just docs-serve           # serve documentation locally
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
just changelog                  # preview next release
just changelog-tag v1.2.0       # preview with a specific tag
just changelog-regen            # regenerate full changelog
```

Releases are triggered via the **Release** GitHub Actions workflow (manual
dispatch), which generates the changelog, bumps `pyproject.toml`, opens a PR,
and publishes to PyPI + GitHub Releases on merge.
