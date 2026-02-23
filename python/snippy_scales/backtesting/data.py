"""OHLCV data normalisation and NumPy conversion.

Provides two layers of transformation:

1. :func:`normalize_ohlcv` — fills in missing columns (open/high/low/volume/ts)
   in a Polars DataFrame so that downstream code can always assume all columns
   are present.
2. :func:`to_numpy_ohlcv` — converts a normalised Polars DataFrame to a fully
   typed :class:`OhlcvArrays` frozen dataclass of aligned NumPy arrays.

:func:`extract_ohlcv` is the single public entry point that chains both steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl

# ── Synthetic OHLCV helpers ───────────────────────────────────────────────────

_BASE_NS: int = 1_577_836_800_000_000_000  # 2020-01-01 00:00 UTC in nanoseconds
_DAY_NS: int = 86_400_000_000_000


# ── OhlcvArrays ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OhlcvArrays:
    """Aligned NumPy arrays representing one instrument's bar data.

    All arrays have identical length.  This replaces the previous
    ``dict[str, np.ndarray]`` return type of :func:`extract_ohlcv` with an
    explicit, typed structure that makes illegal states unrepresentable.

    Attributes:
        timestamps: Bar open timestamps as int64 nanoseconds since epoch.
        open: Bar open prices (float64).
        high: Bar high prices (float64).
        low: Bar low prices (float64).
        close: Bar close prices (float64).
        volume: Bar volume (float64).
    """

    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


# ── Normalisation ─────────────────────────────────────────────────────────────


def normalize_ohlcv(bars: pl.DataFrame) -> pl.DataFrame:
    """Ensure all required OHLCV columns are present, synthesising any that are missing.

    Missing columns are synthesised deterministically from ``close``:

    * ``open``   — previous bar's close (bar 0 uses bar 0's close).
    * ``high``   — ``max(open, close) × 1.001``.
    * ``low``    — ``min(open, close) × 0.999``.
    * ``volume`` — constant 1.0.
    * ``ts``     — sequential daily timestamps starting 2020-01-01 UTC.

    Args:
        bars: Polars DataFrame with at minimum a ``close`` column.

    Returns:
        A new Polars DataFrame guaranteed to contain ``ts``, ``open``,
        ``high``, ``low``, ``close``, and ``volume`` columns.
    """
    import polars as pl

    n = len(bars)
    close = bars["close"].to_numpy().astype(np.float64)

    out = bars.clone()

    if "ts" not in out.columns:
        ts = _BASE_NS + np.arange(n, dtype=np.int64) * _DAY_NS
        out = out.with_columns(pl.Series("ts", ts))

    if "open" not in out.columns:
        open_arr = np.empty(n, dtype=np.float64)
        open_arr[0] = close[0]
        open_arr[1:] = close[:-1]
        out = out.with_columns(pl.Series("open", open_arr))

    if "high" not in out.columns:
        open_arr = out["open"].to_numpy()
        out = out.with_columns(pl.Series("high", np.maximum(open_arr, close) * 1.001))

    if "low" not in out.columns:
        open_arr = out["open"].to_numpy()
        out = out.with_columns(pl.Series("low", np.minimum(open_arr, close) * 0.999))

    if "volume" not in out.columns:
        out = out.with_columns(pl.Series("volume", np.ones(n, dtype=np.float64)))

    return out


def to_numpy_ohlcv(bars: pl.DataFrame) -> OhlcvArrays:
    """Convert a bar DataFrame to aligned NumPy arrays.

    Calls :func:`normalize_ohlcv` first to guarantee all columns are present,
    then casts each column to the appropriate NumPy dtype.

    Args:
        bars: Polars DataFrame with at minimum a ``close`` column.

    Returns:
        :class:`OhlcvArrays` with all six arrays aligned to the same length.
    """
    normalised = normalize_ohlcv(bars)
    return OhlcvArrays(
        timestamps=normalised["ts"].to_numpy().astype(np.int64),
        open=normalised["open"].to_numpy().astype(np.float64),
        high=normalised["high"].to_numpy().astype(np.float64),
        low=normalised["low"].to_numpy().astype(np.float64),
        close=normalised["close"].to_numpy().astype(np.float64),
        volume=normalised["volume"].to_numpy().astype(np.float64),
    )


def extract_ohlcv(bars: pl.DataFrame) -> OhlcvArrays:
    """Extract aligned OHLCV arrays from a Polars bar DataFrame.

    Convenience wrapper that chains :func:`normalize_ohlcv` and
    :func:`to_numpy_ohlcv`.  This is the primary entry point for converting
    strategy bar data into the format expected by the execution engine.

    Args:
        bars: Polars DataFrame with at minimum a ``close`` column.

    Returns:
        :class:`OhlcvArrays` with ``timestamps``, ``open``, ``high``,
        ``low``, ``close``, and ``volume`` arrays.
    """
    return to_numpy_ohlcv(bars)
