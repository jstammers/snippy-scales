"""Configuration models for syncing market data into the catalog.

``algo data sync`` is driven by a YAML config file.  Example::

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

from pathlib import Path  # noqa: TC003 — pydantic needs this at runtime (symbols_file: Path)
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Frequency → Databento schema mapping
# ---------------------------------------------------------------------------

#: Supported user-facing frequency aliases and the Databento schema they map to.
#: Databento natively supports 1s, 1m, 1h and 1d bar schemas (``ohlcv-eod`` has
#: no distinct Nautilus bar type, so it isn't supported). Other intervals (e.g.
#: 5m) are aggregated from these.
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

#: Databento event-level schemas. Unlike the OHLCV schemas these are not
#: aggregated into fixed-width bars, so a single trading day can contain
#: hundreds of millions of rows; :mod:`snippy_scales.data.databento` fetches
#: them one day per request. Must match ``databento.TICK_SPECS``.
TICK_SCHEMAS: frozenset[str] = frozenset(
    {"trades", "tbbo", "mbp-1", "bbo-1s", "bbo-1m", "mbo", "mbp-10"}
)


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


#: Providers that can supply data. Only Databento supports event-level (tick)
#: schemas and 1-second bars.
VALID_PROVIDERS = Literal["databento", "alpaca"]

#: Bar schemas Alpaca can serve.
ALPACA_SCHEMAS: frozenset[str] = frozenset({"ohlcv-1m", "ohlcv-1h", "ohlcv-1d"})


class AssetClassConfig(BaseModel):
    """Configuration for a single asset class grouping.

    Symbols can be listed inline, read from a file, or both (the two lists
    are concatenated, inline entries first, duplicates dropped). A file is
    the practical option for large universes — e.g. several hundred S&P 500
    tickers — that don't belong hand-typed into a YAML list.
    """

    symbols: list[str] = Field(
        default_factory=list,
        description=(
            "List of symbol identifiers. For Databento: continuous-contract "
            "notation (e.g. ``ES.c.0``), individual expiries (``ESZ2024``), or "
            "parent symbology (e.g. ``ES.FUT`` with ``stype_in: parent``) to "
            "pull every individual outright contract in one request. For "
            "Alpaca: plain equity tickers (e.g. ``AAPL``, ``BRK.B``)."
        ),
    )
    symbols_file: Path | None = Field(
        default=None,
        description=(
            "Optional path to a text file of symbols, one per line ('#' "
            "comments and blank lines ignored). Resolved relative to the "
            "config file's directory by load_config(). Merged with `symbols`."
        ),
    )

    @model_validator(mode="after")
    def _require_some_symbol_source(self) -> AssetClassConfig:
        """Ensure at least one of symbols/symbols_file was given."""
        if not self.symbols and self.symbols_file is None:
            raise ValueError("AssetClassConfig needs `symbols` and/or `symbols_file`.")
        return self


class AlpacaConfig(BaseModel):
    """Alpaca-specific ingestion options, used when ``IngestConfig.provider == "alpaca"``.

    Attributes:
        feed: Data feed to request. ``"sip"`` covers all US exchanges and is
            included in the free (Basic) historical data tier; ``"iex"`` is
            IEX-only (thinner, but sometimes preferred for consistency with a
            live-trading IEX feed).
        adjustment: Corporate-action adjustment Alpaca applies before
            returning bars. ``"all"`` (split + dividend) avoids spurious
            price jumps at split dates in multi-year 1-minute history.
        rate_limit_per_min: Historical API calls allowed per minute. The free
            tier allows 200; the default leaves headroom for jitter/retries.
        max_workers: Symbols fetched concurrently. Safe to raise well above
            the historical intuition for I/O-bound work — the shared
            :class:`~snippy_scales.data.alpaca.RateLimiter` is what
            actually bounds request throughput, not thread count.
    """

    feed: Literal["iex", "sip"] = "sip"
    adjustment: Literal["raw", "split", "dividend", "all"] = "all"
    rate_limit_per_min: int = Field(default=190, gt=0, le=200)
    max_workers: int = Field(default=4, gt=0)


class IngestConfig(BaseModel):
    """Top-level configuration for an ``algo data sync`` run.

    Attributes:
        provider: ``"databento"`` (default) or ``"alpaca"``. Alpaca supports
            only 1m/1h/1d bars.
        venue: Venue for instrument IDs in the catalog. Defaults to the
            dataset prefix for Databento (``GLBX.MDP3`` -> ``GLBX``) and
            ``ALPACA`` for Alpaca.
        dataset: Databento dataset code. Ignored for Alpaca.
        schemas: Bar frequency aliases and/or event-level schema names
            (e.g. ``["1d", "trades"]``). Narrowed at runtime via ``--schema``.
        start: Earliest date to fetch (``YYYY-MM-DD``).
        end: Latest date to fetch (exclusive, ``YYYY-MM-DD``). Defaults to today.
        stype_in: Databento symbology type, used for cost estimates. Inferred
            from each symbol's form when omitted. Ignored for Alpaca.
        alpaca: Alpaca-specific options. Ignored for Databento.
        asset_classes: Mapping of free-form labels to their symbols.
    """

    provider: VALID_PROVIDERS = Field(
        default="databento",
        description="Bar-data source: 'databento' or 'alpaca'.",
    )
    venue: str | None = Field(
        default=None,
        description="Venue for catalog instrument IDs. Defaults per provider.",
    )
    dataset: str = Field(
        default="GLBX.MDP3",
        description="Databento dataset identifier (e.g. 'GLBX.MDP3'). Ignored for Alpaca.",
    )
    schemas: list[str] = Field(
        default_factory=lambda: ["1d"],
        description=(
            "Bar frequency aliases (e.g. '1d', '1h', '1m') and/or event-level "
            "schema names (e.g. 'trades', 'mbo').  Can be narrowed to a single "
            "schema at runtime via --schema."
        ),
    )
    start: str = Field(
        ...,
        description="Start date for data pull in ISO format (YYYY-MM-DD).",
    )
    end: str | None = Field(
        default=None,
        description="End date (exclusive) in ISO format (YYYY-MM-DD). Defaults to today.",
    )
    stype_in: VALID_STYPES | None = Field(
        default=None,
        description="Databento symbology type for cost estimates. Ignored for Alpaca.",
    )
    alpaca: AlpacaConfig | None = Field(
        default=None,
        description="Alpaca-specific options. Only meaningful when provider == 'alpaca'.",
    )
    asset_classes: dict[str, AssetClassConfig] = Field(
        default_factory=dict,
        description="Mapping of asset-class label → AssetClassConfig.",
    )

    @model_validator(mode="after")
    def _validate_schemas(self) -> IngestConfig:
        """Ensure every schema resolves, and Alpaca configs list only supported bars."""
        for entry in self.schemas:
            resolved = resolve_schema(entry)  # raises ValueError if unknown
            if self.provider == "alpaca" and resolved not in ALPACA_SCHEMAS:
                raise ValueError(
                    f"provider='alpaca' does not support schema {resolved!r} — Alpaca "
                    f"supports {sorted(ALPACA_SCHEMAS)}. Use provider='databento'."
                )
        return self

    @property
    def resolved_schemas(self) -> list[str]:
        """Databento schema names derived from :attr:`schemas` (order preserved)."""
        return [resolve_schema(entry) for entry in self.schemas]

    @property
    def all_symbols(self) -> list[str]:
        """Flat list of every symbol across all asset classes (order preserved, deduplicated)."""
        return list(dict.fromkeys(sym for ac in self.asset_classes.values() for sym in ac.symbols))

    @property
    def alpaca_options(self) -> AlpacaConfig:
        """Effective Alpaca options — :attr:`alpaca` if set, otherwise defaults."""
        return self.alpaca or AlpacaConfig()

    @property
    def effective_venue(self) -> str:
        """:attr:`venue`, or the provider default."""
        if self.venue:
            return self.venue
        if self.provider == "alpaca":
            return "ALPACA"
        return self.dataset.split(".", 1)[0]


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
    return _resolve_symbols_files(IngestConfig.model_validate(raw), base_dir=path.parent)


def _load_symbols_file(path: Path) -> list[str]:
    """Read one symbol per line from *path*, ignoring blanks and '#' comments."""
    if not path.exists():
        raise FileNotFoundError(f"symbols_file not found: {path}")
    symbols: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            symbols.append(stripped)
    return symbols


def _resolve_symbols_files(config: IngestConfig, *, base_dir: Path) -> IngestConfig:
    """Merge each asset class's ``symbols_file`` (if any) into its ``symbols`` list.

    Relative ``symbols_file`` paths are resolved against *base_dir* (the
    config file's own directory), so a config and its companion symbols file
    can be moved together without editing the path. Duplicates between the
    inline list and the file are dropped, inline entries first.

    Args:
        config: A validated config, possibly with ``symbols_file`` entries.
        base_dir: Directory relative paths are resolved against.

    Returns:
        *config* unchanged if no asset class sets ``symbols_file``, otherwise
        a copy with every ``symbols`` list expanded.
    """
    if not any(ac.symbols_file is not None for ac in config.asset_classes.values()):
        return config

    resolved_classes: dict[str, AssetClassConfig] = {}
    for label, ac in config.asset_classes.items():
        if ac.symbols_file is None:
            resolved_classes[label] = ac
            continue
        file_path = ac.symbols_file if ac.symbols_file.is_absolute() else base_dir / ac.symbols_file
        file_symbols = _load_symbols_file(file_path)
        merged = list(dict.fromkeys([*ac.symbols, *file_symbols]))  # dedup, preserve order
        resolved_classes[label] = ac.model_copy(update={"symbols": merged})

    return config.model_copy(update={"asset_classes": resolved_classes})
