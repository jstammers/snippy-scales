# SnippyScales development workflow commands

# Display available recipes
default:
    @just --list

# ── Setup ────────────────────────────────────────────────────────────────────

# Install all dependencies and set up the project
setup:
    uv sync --all-extras
    prek install

# Install Python dependencies only
sync:
    uv sync

# Build the sdist and wheel into dist/
build:
    uv build

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

# ── Combined ─────────────────────────────────────────────────────────────────

# Format all code
fmt: fmt-py

# Lint all code
lint: lint-py

# Run all tests
test: test-py

# Run all CI checks locally (prek hooks)
check-all:
    prek run --all-files

# Full check: format, lint, type-check, and test everything
check: fmt type-check lint test

# Clean build artifacts
clean:
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf htmlcov
    rm -rf .coverage
    rm -rf dist
    rm -rf build
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# ── Data ─────────────────────────────────────────────────────────────────────

# Resolve and cache the S&P 500 ever-members universe (used by symbols_file configs)
update-universe *ARGS:
    uv run algo data update-universe {{ARGS}}

# Preview the S&P 500 1-minute Alpaca backfill plan (no download)
backfill-sp500-dry-run:
    uv run algo data backfill-sp500 --dry-run

# Run the S&P 500 1-minute Alpaca backfill (ALPACA_API_KEY/SECRET_KEY required)
backfill-sp500 *ARGS:
    uv run algo data backfill-sp500 {{ARGS}}

# ── Documentation ────────────────────────────────────────────────────────────

# Build and serve documentation locally
docs-serve:
    uv run mkdocs serve

# Build documentation site
docs-build:
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
    uv run --env-file .env algo {{ARGS}}
