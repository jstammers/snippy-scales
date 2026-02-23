# SnippyScales development workflow commands

# Display available recipes
default:
    @just --list

# ── Setup ────────────────────────────────────────────────────────────────────

# Install all dependencies and set up the project
setup:
    uv sync
    uv run maturin develop
    prek install

# Install Python dependencies only
sync:
    uv sync

# Build Rust extension in development mode
dev:
    uv run maturin develop

# Build Rust extension in release mode
build:
    uv run maturin build --release

# ── Python ───────────────────────────────────────────────────────────────────

# Run Python linter (ruff) with auto-fix
lint-py:
    uv run ruff check . --fix

# Run Python linter (ruff) for CI with GitHub output format
lint-py-ci:
    uv run ruff check . --output-format=github

# Format Python code (ruff)
fmt-py:
    uv run ruff format .

# Check Python formatting without making changes
fmt-py-check:
    uv run ruff format --check .

# Run Python type checker (ty)
type-check:
    uv run ty check

# Run Python tests
test-py:
    uv run pytest

# Run Python tests with coverage (HTML report)
test-py-cov:
    uv run pytest --cov=snippy_scales --cov-report=html

# Run Python tests with coverage (XML report for CI)
test-py-cov-ci:
    uv run pytest --cov=snippy_scales --cov-report=xml -q

# Lint, format, type-check, and test Python code
check-py: lint-py fmt-py type-check test-py

# ── Rust ─────────────────────────────────────────────────────────────────────

# Format Rust code
fmt-rs:
    cargo fmt --manifest-path rust/Cargo.toml --all

# Check Rust formatting without making changes
fmt-rs-check:
    cargo fmt --manifest-path rust/Cargo.toml --all -- --check

# Run Rust linter (clippy)
lint-rs:
    cargo clippy --manifest-path rust/Cargo.toml --workspace -- -D warnings

# Run Rust tests
test-rs:
    cargo test --manifest-path rust/Cargo.toml --workspace

# Run Rust benchmarks
bench:
    cargo bench --manifest-path rust/Cargo.toml --workspace

# Format, lint, and test Rust code
check-rs: fmt-rs lint-rs test-rs

# ── Combined ─────────────────────────────────────────────────────────────────

# Format both Python and Rust code
fmt: fmt-py fmt-rs

# Lint both Python and Rust code
lint: lint-py lint-rs

# Run all tests (Python and Rust)
test: test-py test-rs

# Run all CI checks locally (prek hooks)
check-all:
    prek run --all-files

# Full check: format, lint, type-check, and test everything
check: fmt type-check lint test

# Clean build artifacts
clean:
    cargo clean --manifest-path rust/Cargo.toml
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf htmlcov
    rm -rf .coverage
    rm -rf dist
    rm -rf build
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# ── Documentation ────────────────────────────────────────────────────────────

# Build and serve documentation locally
docs-serve:
    uv run mkdocs serve

# Build documentation site
docs-build:
    cargo doc --manifest-path rust/Cargo.toml --workspace --no-deps
    rm -rf docs/api/rust
    cp -r rust/target/doc docs/api/rust
    uv run mkdocs build --strict

# ── Changelog & Release ──────────────────────────────────────────────────────

# Preview unreleased changes in changelog
changelog:
    git cliff --unreleased

# Preview changelog for a specific tag
changelog-tag TAG:
    git cliff --unreleased --tag {{TAG}}

# Regenerate full changelog
changelog-regen:
    git cliff -o CHANGELOG.md

# ── CLI ──────────────────────────────────────────────────────────────────────

# Run the algo CLI (pass arguments with: just algo -- --help)
algo *ARGS:
    uv run algo {{ARGS}}
