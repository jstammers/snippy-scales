# SnippyScales

A quantitative trading research platform. Research and signal generation are
ours; data storage, backtest execution and live trading are moving to
[NautilusTrader](https://github.com/nautechsystems/nautilus_trader) — see the
[migration plan](docs/architecture/nautilus_migration.md).

## Architecture

```
python/          # Research, strategies, evaluation, CLI
docs/            # Architecture decisions, API reference, guides
configs/         # Data ingestion configs
data/            # Local data (gitignored)
```

## Layers

| Layer | Provided by | Responsibility |
|---|---|---|
| Research | `snippy_scales.research` | Feature engineering, ML, prototyping |
| Strategy Logic | `snippy_scales.strategies` | Signal generation (target positions) |
| Evaluation | `snippy_scales.evaluation` | Walk-forward, parameter search, overfitting statistics |
| Data, Backtesting, Execution | NautilusTrader (migration in progress) | Catalog, event-driven engine, live trading |
| CLI + Orchestration | Python (Typer) | Project control |

## Quickstart

```bash
# 1. Install tooling
curl -LsSf https://astral.sh/uv/install.sh | sh                           # uv
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh  # prek
uv tool install git-cliff                                                   # changelog
uv tool install rust-just                                                   # task runner

# 2. Set up the repo
just setup                # install deps and activate git hooks

# 3. Use the CLI
just algo --help          # or: uv run algo --help
```

## Development

```bash
# Quick commands
just                      # list all available commands
just check                # run all checks (format, lint, type-check, test)
just test                 # run all tests
just fmt                  # format all code

# Python-specific
just lint-py              # ruff check with auto-fix
just fmt-py               # ruff format
just type-check           # ty check
just test-py              # pytest
just test-py-cov          # pytest with coverage report

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

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for the full workflow.

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
