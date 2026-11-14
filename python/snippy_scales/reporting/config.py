"""Report configuration: one YAML file per report, validated with pydantic.

A report's ``config.yaml`` says *what* to test (strategy, universe, period),
*how* (folds, costs, screening search, how many trials to validate on
Nautilus) and *where* its theory write-up lives. Everything the notebook
computes is derived from it, so the file plus the git commit reproduce a
report. See ``reports/templates/configs/`` for examples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from snippy_scales.data.paths import UNIVERSE_DIR

#: Repository root (the package is installed editable from ``python/``).
REPO_ROOT = Path(__file__).resolve().parents[3]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Model):
    """The universe and period.

    Attributes:
        symbols: Explicit symbols. Combined with *universe* when both are given.
        universe: Name of a symbols file in ``data/universe/`` (``<name>.txt``,
            one symbol per line, ``#`` comments), e.g. ``sp500_ever_members``.
        max_symbols: Keep only the first N symbols (quick runs); ``None`` = all.
        schema: Bar schema, e.g. ``ohlcv-1d``.
        venue: Catalog venue, e.g. ``ALPACA`` or ``GLBX`` (``None`` = resolve).
        start: First bar close evaluated (folds start after ``min_train`` bars
            from here). Earlier bars still warm up signals. ``None`` = first bar.
        end: Last bar close evaluated. ``None`` = last bar.
        benchmark: Symbol whose buy-and-hold returns are overlaid (``None`` = none).
        example_symbol: Symbol used to illustrate signals (``None`` = first).
    """

    symbols: list[str] = Field(default_factory=list)
    universe: str | None = None
    max_symbols: int | None = Field(default=None, ge=1)
    schema_: str = Field(default="ohlcv-1d", alias="schema")
    venue: str | None = None
    start: str | None = None
    end: str | None = None
    benchmark: str | None = None
    example_symbol: str | None = None

    @model_validator(mode="after")
    def _has_symbols(self) -> DataConfig:
        if not self.symbols and not self.universe:
            raise ValueError("data needs `symbols`, a `universe`, or both")
        return self

    def resolve_symbols(self, available: set[str] | None = None) -> list[str]:
        """The symbols to load, in file order, optionally only those *available*."""
        names = list(self.symbols)
        if self.universe:
            path = UNIVERSE_DIR / f"{self.universe}.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Universe file {path} not found; see `algo data update-universe`."
                )
            lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
            names += [line for line in lines if line]
        names = list(dict.fromkeys(names))
        if available is not None:
            names = [s for s in names if s in available]
        return names[: self.max_symbols] if self.max_symbols else names


class EvaluationConfig(_Model):
    """Walk-forward folds and backtest settings, shared by both engines.

    ``rebalance_band`` is a fraction of *total* equity per trade: with N symbols
    each position is only ~1/N of equity, so keep it well below ``1 / N`` or
    nothing trades.
    """

    n_splits: int = Field(default=5, ge=1)
    test_size: float = Field(default=0.15, gt=0)
    min_train: int = Field(default=252, ge=0)
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    commission_rate: float = Field(default=0.0005, ge=0)
    commission_per_contract: float = Field(default=0.0, ge=0)
    slippage_prob: float = Field(default=0.0, ge=0, le=1)
    rebalance_band: float = Field(default=0.0, ge=0)
    leverage: float = Field(default=10.0, gt=0)


class ScreeningConfig(_Model):
    """The parameter search on the screening engine.

    ``space`` uses :func:`snippy_scales.evaluation.suggest`'s format: a list is
    a categorical choice, ``{low, high, step?, log?}`` a range, a scalar fixed.
    """

    engine: Literal["vector", "nautilus"] = "vector"
    sampler: Literal["tpe", "random", "grid"] = "tpe"
    n_trials: int | None = Field(default=50, ge=1)
    seed: int = 42
    space: dict[str, Any] = Field(default_factory=dict)


class ValidationConfig(_Model):
    """How many of the best screened parameter sets to re-run on Nautilus.

    ``workers`` is the number of processes the Nautilus folds run in (each
    holds a copy of the data); ``None`` picks half the CPUs, capped at the
    number of fold runs. It does not change results, so it is not part of the
    fingerprint.
    """

    top_k: int = Field(default=5, ge=1)
    workers: int | None = Field(default=None, ge=1)


class ReportConfig(_Model):
    """A strategy research report.

    Attributes:
        name: Report slug; its directory is ``reports/<name>/``.
        title: Human-readable title.
        strategy: Registry name (``algo eval --help``), e.g. ``momentum-ts``.
        theory: Markdown file (repo-relative) explaining the strategy's maths
            and economics; defaults to ``docs/strategies/<strategy>.md``.
        question: The research question this report answers (one or two lines).
    """

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    title: str
    strategy: str
    theory: str | None = None
    question: str = ""
    data: DataConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @property
    def theory_path(self) -> Path:
        return REPO_ROOT / (self.theory or f"docs/strategies/{self.strategy}.md")

    def fingerprint(self) -> str:
        """Hash of everything that changes results (not title, question or theory).

        Stored on the report's Optuna studies so a changed config is never mixed
        with trials run under the old one.
        """
        payload = self.model_dump(
            mode="json", by_alias=True, include={"strategy", "data", "evaluation", "screening"}
        )
        payload["data"].pop("benchmark", None)
        payload["data"].pop("example_symbol", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", by_alias=True, exclude_none=True), sort_keys=False
        )


def load_report_config(path: Path) -> ReportConfig:
    """Read and validate a report config YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return ReportConfig.model_validate(raw)
