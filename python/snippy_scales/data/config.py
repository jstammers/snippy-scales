"""Configuration models for Databento data ingestion.

The ingestion pipeline is driven by a YAML config file.  Example::

    dataset: "GLBX.MDP3"
    tick_frequency: "1d"
    start: "2018-01-01"
    asset_classes:
      equity_index:
        symbols: [ES.c.0, NQ.c.0]
      rates:
        symbols: [ZN.c.0, ZB.c.0]

The ``tick_frequency`` field (and its CLI ``--frequency`` override) accepts
user-friendly strings that are mapped to the corresponding Databento OHLCV
schema names before any API call is made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml
from pandas.io.parsers.readers import Literal
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Frequency → Databento schema mapping
# ---------------------------------------------------------------------------

#: Supported user-facing frequency aliases and the Databento schema they map to.
#: Databento natively supports 1s, 1m, 1h, 1d, and eod bar schemas.
#: Sub-minute or arbitrary intervals (e.g. 5m) must be resampled after download.
_FREQUENCY_TO_SCHEMA: dict[str, str] = {
    # seconds
    "1s": "ohlcv-1s",
    "s": "ohlcv-1s",
    # minutes
    "1m": "ohlcv-1m",
    "1min": "ohlcv-1m",
    "min": "ohlcv-1m",
    # hours
    "1h": "ohlcv-1h",
    "h": "ohlcv-1h",
    "hourly": "ohlcv-1h",
    # daily
    "1d": "ohlcv-1d",
    "d": "ohlcv-1d",
    "daily": "ohlcv-1d",
    # end-of-day
    "eod": "ohlcv-eod",
}

_VALID_SCHEMAS: frozenset[str] = frozenset(_FREQUENCY_TO_SCHEMA.values())

VALID_STYPES = Literal["raw_symbol", "parent", "continuous", "instrument_id"]


def frequency_to_schema(frequency: str) -> str:
    """Convert a user-friendly frequency string to a Databento OHLCV schema name.

    Args:
        frequency: Frequency string such as ``"1d"``, ``"1h"``, ``"1m"``,
            or ``"daily"``.  Already-valid schema names (e.g. ``"ohlcv-1d"``)
            are passed through unchanged.

    Returns:
        The corresponding Databento schema string (e.g. ``"ohlcv-1d"``).

    Raises:
        ValueError: If *frequency* is not a recognised alias or schema name.

    Examples:
        >>> frequency_to_schema("daily")
        'ohlcv-1d'
        >>> frequency_to_schema("1h")
        'ohlcv-1h'
        >>> frequency_to_schema("ohlcv-1m")
        'ohlcv-1m'
    """
    normalized = frequency.lower().strip()
    if normalized in _FREQUENCY_TO_SCHEMA:
        return _FREQUENCY_TO_SCHEMA[normalized]
    if normalized in _VALID_SCHEMAS:
        return normalized
    valid = ", ".join(sorted(_FREQUENCY_TO_SCHEMA))
    raise ValueError(f"Unknown frequency {frequency!r}. Valid options: {valid}")


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class AssetClassConfig(BaseModel):
    """Configuration for a single asset class grouping."""

    symbols: list[str] = Field(
        ...,
        description=(
            "List of Databento symbol identifiers.  Use continuous-contract "
            "notation (e.g. ``ES.c.0``) or individual expiries (``ESZ2024``)."
        ),
    )


class IngestConfig(BaseModel):
    """Top-level configuration for a Databento ingestion run.

    Attributes:
        dataset: Databento dataset code (default ``"GLBX.MDP3"`` for CME Globex).
        tick_frequency: Bar frequency alias or schema name.  Overridable at
            runtime via the ``--frequency`` CLI flag.
        start: Earliest date to fetch (``YYYY-MM-DD``).
        end: Latest date to fetch (``YYYY-MM-DD``).  Defaults to today when
            ``None``.
        stype_in:
        asset_classes: Mapping of arbitrary asset-class labels to their
            :class:`AssetClassConfig`.
    """

    dataset: str = Field(
        default="GLBX.MDP3",
        description="Databento dataset identifier (e.g. 'GLBX.MDP3').",
    )
    tick_frequency: str = Field(
        default="1d",
        description=(
            "Bar frequency alias (e.g. '1d', '1h', '1m') or a Databento schema "
            "name (e.g. 'ohlcv-1d').  Can be overridden at runtime via "
            "--frequency."
        ),
    )
    start: str = Field(
        ...,
        description="Start date for data pull in ISO format (YYYY-MM-DD).",
    )
    end: str | None = Field(
        default=None,
        description=(
            "End date for data pull in ISO format (YYYY-MM-DD).  Defaults to today when omitted."
        ),
    )

    stype_in: VALID_STYPES | None = Field(
        default=None,
        description=(
            "Optional Databento symbology type for API calls (e.g. 'raw_symbol', 'parent')"
        ),
    )

    asset_classes: dict[str, AssetClassConfig] = Field(
        default_factory=dict,
        description="Mapping of asset-class label → AssetClassConfig.",
    )

    @model_validator(mode="after")
    def _validate_frequency(self) -> IngestConfig:
        """Ensure tick_frequency resolves to a known Databento schema."""
        frequency_to_schema(self.tick_frequency)  # raises ValueError if unknown
        return self

    @property
    def schema(self) -> str:
        """Databento OHLCV schema name derived from :attr:`tick_frequency`."""
        return frequency_to_schema(self.tick_frequency)

    @property
    def all_symbols(self) -> list[str]:
        """Flat list of every symbol across all asset classes (order preserved)."""
        return [sym for ac in self.asset_classes.values() for sym in ac.symbols]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: Path) -> IngestConfig:
    """Load and validate an ingestion configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated :class:`IngestConfig` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the YAML content fails Pydantic validation.

    Example::

        cfg = load_config(Path("configs/databento.yaml"))
        print(cfg.all_symbols)
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return IngestConfig.model_validate(raw)
