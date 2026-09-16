"""Configuration models for Databento data ingestion.

The ingestion pipeline is driven by a YAML config file.  Example::

    dataset: "GLBX.MDP3"
    schemas: ["1d"]
    start: "2018-01-01"
    asset_classes:
      equity_index:
        symbols: [ES.c.0, NQ.c.0]
      rates:
        symbols: [ZN.c.0, ZB.c.0]

The ``schemas`` field (and its CLI ``--schema`` override) accepts
user-friendly strings that are mapped to the corresponding Databento schema
names before any API call is made — both bar aliases (``"1d"``) and
event-level schema names (``"trades"``) are valid entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

import yaml
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
# Tick (event-level) schemas
# ---------------------------------------------------------------------------

#: Databento event-level schemas.  Unlike the OHLCV schemas these are not
#: aggregated into fixed-width bars, so a single trading day can contain
#: hundreds of millions of rows.  They are stored day-partitioned by
#: :mod:`snippy_scales.data.tick` rather than as one file per symbol.
TICK_SCHEMAS: frozenset[str] = frozenset({"trades", "mbo", "mbp-1", "mbp-10", "tbbo"})


def is_tick_schema(schema: str) -> bool:
    """Return ``True`` if *schema* is an event-level (non-bar) Databento schema.

    Args:
        schema: Databento schema name.

    Returns:
        Whether the schema is one of :data:`TICK_SCHEMAS`.
    """
    return schema.lower().strip() in TICK_SCHEMAS


def resolve_schema(value: str) -> str:
    """Resolve *value* to a Databento schema name, accepting ticks or bar aliases.

    This is the single entry point used by the CLI: it accepts event-level
    schema names (``"trades"``, ``"mbo"``) as-is and otherwise defers to
    :func:`frequency_to_schema` so bar aliases (``"1d"``, ``"daily"``) keep
    working unchanged.

    Args:
        value: A tick schema name, a bar frequency alias, or a bar schema name.

    Returns:
        The corresponding Databento schema string.

    Raises:
        ValueError: If *value* is neither a tick schema nor a known frequency.

    Examples:
        >>> resolve_schema("trades")
        'trades'
        >>> resolve_schema("daily")
        'ohlcv-1d'
    """
    normalized = value.lower().strip()
    if normalized in TICK_SCHEMAS:
        return normalized
    try:
        return frequency_to_schema(value)
    except ValueError as exc:
        valid = ", ".join(sorted(TICK_SCHEMAS | set(_FREQUENCY_TO_SCHEMA)))
        raise ValueError(f"Unknown schema {value!r}. Valid options: {valid}") from exc


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class AssetClassConfig(BaseModel):
    """Configuration for a single asset class grouping."""

    symbols: list[str] = Field(
        ...,
        description=(
            "List of Databento symbol identifiers.  Use continuous-contract "
            "notation (e.g. ``ES.c.0``), individual expiries (``ESZ2024``), or "
            "parent symbology (e.g. ``ES.FUT`` with ``stype_in: parent``) to "
            "pull every individual outright contract in one request."
        ),
    )


class IngestConfig(BaseModel):
    """Top-level configuration for a Databento ingestion run.

    Attributes:
        dataset: Databento dataset code (default ``"GLBX.MDP3"`` for CME Globex).
        schemas: Bar frequency aliases and/or event-level schema names to
            ingest (e.g. ``["1d", "trades"]``).  Overridable at runtime via
            the ``--schema`` CLI flag (which runs a single schema instead of
            the full list).
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
    schemas: list[str] = Field(
        default_factory=lambda: ["1d"],
        description=(
            "Bar frequency aliases (e.g. '1d', '1h', '1m') and/or event-level "
            "schema names (e.g. 'trades', 'mbo') to ingest.  Can be narrowed to "
            "a single schema at runtime via --schema."
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
    def _validate_schemas(self) -> IngestConfig:
        """Ensure every entry in schemas resolves to a known Databento schema."""
        for entry in self.schemas:
            resolve_schema(entry)  # raises ValueError if unknown
        return self

    @property
    def resolved_schemas(self) -> list[str]:
        """Databento schema names derived from :attr:`schemas` (order preserved)."""
        return [resolve_schema(entry) for entry in self.schemas]

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
