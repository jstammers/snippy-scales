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

from pathlib import Path  # noqa: TC003 — pydantic needs this at runtime (symbols_file: Path)
from typing import Any, Literal

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


#: Providers that can supply bar data. Only Databento supports event-level
#: (tick) schemas — a config with ``provider: alpaca`` may not list one.
VALID_PROVIDERS = Literal["databento", "alpaca"]

#: Databento delivery mechanism. ``"batch"`` (default) submits a batch job
#: and downloads it once complete — billed the same per byte as streaming,
#: but Databento keeps completed job output downloadable free of charge for
#: a retention window, so an accidental local-data loss doesn't have to be
#: repaid. ``"streaming"`` calls the Historical Streaming API directly —
#: lower latency, but every call is billed with no server-side retention.
#: Ignored for ``provider == "alpaca"`` (Alpaca has no batch equivalent).
DownloadMethod = Literal["batch", "streaming"]

#: Closed set of top-level storage classifications. This determines the
#: ``data/raw/<instrument_type>/`` subdirectory a symbol is stored under —
#: it is deliberately separate from ``asset_classes`` (below), which is a
#: free-form display/grouping label chosen per config file and must never
#: be used as a storage key (two configs could otherwise use different
#: labels for the same symbol and silently double-store it). Every symbol
#: must be unique within its ``instrument_type`` — see
#: :func:`~snippy_scales.data.ingest.find_cross_class_duplicates`.
InstrumentType = Literal["equities", "futures", "options", "fx_spot", "crypto"]


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
            :class:`~snippy_scales.data.ratelimit.RateLimiter` is what
            actually bounds request throughput, not thread count.
    """

    feed: Literal["iex", "sip"] = "sip"
    adjustment: Literal["raw", "split", "dividend", "all"] = "all"
    rate_limit_per_min: int = Field(default=190, gt=0, le=200)
    max_workers: int = Field(default=4, gt=0)


class IngestConfig(BaseModel):
    """Top-level configuration for a bar/tick ingestion run.

    Attributes:
        provider: Which data source to use — ``"databento"`` (default) or
            ``"alpaca"``. Alpaca only supports bar schemas, never tick
            schemas.
        instrument_type: Closed top-level storage classification (e.g.
            ``"equities"``, ``"futures"``) — determines the
            ``data/raw/<instrument_type>/`` subdirectory every symbol in
            this config is stored under. File-level (like ``provider``)
            because one config always covers one instrument type in
            practice; split into another file if that ever changes. Not to
            be confused with ``asset_classes``, which stays a free-form
            display/grouping label.
        dataset: Databento dataset code (default ``"GLBX.MDP3"``). Ignored
            when ``provider == "alpaca"``.
        schemas: Bar frequency aliases and/or event-level schema names to
            ingest (e.g. ``["1d", "trades"]``).  Overridable at runtime via
            the ``--schema`` CLI flag (which runs a single schema instead of
            the full list).
        start: Earliest date to fetch (``YYYY-MM-DD``).
        end: Latest date to fetch (``YYYY-MM-DD``).  Defaults to today when
            ``None``.
        stype_in: Databento symbology type. Ignored for Alpaca.
        download_method: Databento delivery mechanism — ``"batch"`` (default)
            or ``"streaming"``. Ignored for Alpaca.
        alpaca: Alpaca-specific options. Ignored (and optional) for Databento.
        asset_classes: Mapping of arbitrary asset-class labels to their
            :class:`AssetClassConfig`.
    """

    provider: VALID_PROVIDERS = Field(
        default="databento",
        description="Bar-data source: 'databento' or 'alpaca'.",
    )
    instrument_type: InstrumentType = Field(
        ...,
        description=(
            "Closed top-level storage classification (e.g. 'equities', "
            "'futures') — determines the data/raw/<instrument_type>/ "
            "subdirectory. Separate from asset_classes, which is a "
            "free-form display/grouping label only."
        ),
    )
    dataset: str = Field(
        default="GLBX.MDP3",
        description="Databento dataset identifier (e.g. 'GLBX.MDP3'). Ignored for Alpaca.",
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
            "Optional Databento symbology type for API calls (e.g. 'raw_symbol', 'parent'). "
            "Ignored for Alpaca."
        ),
    )

    download_method: DownloadMethod = Field(
        default="batch",
        description=(
            "Databento delivery mechanism: 'batch' (default, free re-download within "
            "Databento's retention window) or 'streaming' (lower latency, no retention). "
            "Ignored for Alpaca."
        ),
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
        """Ensure every schema resolves, and Alpaca configs list only bar schemas."""
        for entry in self.schemas:
            resolved = resolve_schema(entry)  # raises ValueError if unknown
            if self.provider == "alpaca" and is_tick_schema(resolved):
                raise ValueError(
                    f"provider='alpaca' does not support event-level schema {resolved!r} "
                    "— Alpaca is bar-data only. Use provider='databento' for tick schemas."
                )
        return self

    @property
    def resolved_schemas(self) -> list[str]:
        """Databento schema names derived from :attr:`schemas` (order preserved)."""
        return [resolve_schema(entry) for entry in self.schemas]

    @property
    def all_symbols(self) -> list[str]:
        """Flat list of every symbol across all asset classes (order preserved)."""
        return [sym for ac in self.asset_classes.values() for sym in ac.symbols]

    @property
    def alpaca_options(self) -> AlpacaConfig:
        """Effective Alpaca options — :attr:`alpaca` if set, otherwise defaults."""
        return self.alpaca or AlpacaConfig()


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
