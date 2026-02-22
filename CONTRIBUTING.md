# Contributing to SnippyScales

## Setup

```bash
# 1. Install Rust (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install prek (Rust-native git hook manager)
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh

# 4. Install git-cliff (changelog generator)
cargo install git-cliff

# 5. Clone and set up the repo
git clone https://github.com/yourorg/snippy-scales
cd snippy-scales
uv sync
uv run maturin develop   # build Rust extension in dev mode
prek install             # activate git hooks from prek.toml
```

## Commit Messages

This project enforces **Conventional Commits**. Every commit message must follow:

```
<type>(<optional scope>): <short description>

<optional body>

<optional footer>
```

### Allowed types

| Type | When to use | Semver impact |
|------|-------------|---------------|
| `feat` | New capability, API, or strategy | **MINOR** |
| `fix` | Bug fix | **PATCH** |
| `perf` | Performance improvement | PATCH |
| `refactor` | Code restructure, no behaviour change | — |
| `revert` | Reverts a previous commit | depends |
| `docs` | Documentation only | — |
| `style` | Formatting, whitespace (skipped in changelog) | — |
| `test` | Adding or fixing tests | — |
| `build` | Build system, maturin, Cargo | — |
| `ci` | GitHub Actions, prek hooks | — |
| `chore` | Maintenance (skipped in changelog) | — |

### Breaking changes

Add `!` after the type, or include a `BREAKING CHANGE:` footer:

```
feat!: redesign Portfolio API

BREAKING CHANGE: `Portfolio::apply_fill` now takes `&mut Fill` instead of `Fill`.
```

Breaking changes trigger a **MAJOR** semver bump.

### Scopes

Scopes are optional but encouraged for clarity:

```
feat(backtest): add volatility-adjusted slippage model
fix(execution): correct notional limit in RiskCheck
docs(cli): document data ingest flags
```

### Examples

```bash
# Good
git commit -m "feat(strategies): add carry-based futures strategy"
git commit -m "fix(portfolio): handle zero-quantity close correctly"
git commit -m "perf(backtest): parallelise event loop with rayon"
git commit -m "ci: cache Rust build artifacts in CI"

# Bad — will be rejected by the commit-msg hook
git commit -m "added stuff"
git commit -m "WIP"
git commit -m "fix things"
```

The `commit-msg` hook (managed by `prek`) will reject any message that
doesn't match the pattern. To amend a rejected message:

```bash
git commit --amend
```

## Running Hooks Manually

```bash
# Run all hooks against all files
prek run --all-files

# Run only a specific hook
prek run ruff-check

# Run hooks against files changed in the last commit
prek run --last-commit

# Auto-update hook versions (with 7-day cooldown for supply-chain safety)
prek auto-update --cooldown-days 7
```

## Changelog

The `CHANGELOG.md` is generated automatically by **git-cliff** from the
commit history. You never need to edit it by hand.

```bash
# Preview what the next release changelog will look like
git cliff --unreleased

# Preview with a specific tag
git cliff --unreleased --tag v1.2.0

# Generate the full changelog
git cliff -o CHANGELOG.md
```

Releases are triggered via the **Release** GitHub Actions workflow, which:
1. Validates the semver tag you provide
2. Runs `git cliff` to update `CHANGELOG.md`
3. Bumps the version in `pyproject.toml`
4. Opens a release PR
5. On merge, builds the wheel and publishes to PyPI + GitHub Releases

## Development Workflow

```bash
# Lint Python
uv run ruff check . --fix
uv run ruff format .

# Type-check Python
uv run ty check

# Run Python tests
uv run pytest -x -q

# Lint + test Rust
cargo fmt --manifest-path rust/Cargo.toml
cargo clippy --manifest-path rust/Cargo.toml --workspace -- -D warnings
cargo test --manifest-path rust/Cargo.toml --workspace

# Rebuild Rust extension after changes
uv run maturin develop

# Run benchmarks
cargo bench --manifest-path rust/Cargo.toml --workspace
```

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Always releasable; protected |
| `develop` | Integration branch |
| `feat/*` | New features |
| `fix/*` | Bug fixes |
| `release/vX.Y.Z` | Auto-created by the release workflow |

Use **squash-merges** when merging PRs so that git-cliff sees one clean
conventional commit per PR in the history.
