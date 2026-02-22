"""Databento data ingestion — downloads bar data and persists as Parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
DERIVED_DIR = Path("data/derived")


def ingest_databento(
    *,
    dataset: str,
    symbol: str,
    start: str,
    end: str,
    schema: str = "ohlcv-1d",
    output_dir: Path = RAW_DIR,
) -> Path:
    """Download bars from Databento and save to Parquet.

    Args:
        dataset: Databento dataset code (e.g. "GLBX.MDP3").
        symbol: Instrument symbol (e.g. "ES.c.0").
        start: ISO date string "YYYY-MM-DD".
        end: ISO date string "YYYY-MM-DD".
        schema: Databento schema name.
        output_dir: Directory to write raw Parquet files.

    Returns:
        Path to the written Parquet file.
    """
    import databento as db  # noqa: PLC0415 — optional dep

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_{schema}_{start}_{end}.parquet"

    if out_path.exists():
        logger.info("File already exists, skipping download: %s", out_path)
        return out_path

    client = db.Historical()
    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema=schema,
        start=start,
        end=end,
    )
    df = data.to_df()
    pl.from_pandas(df).write_parquet(out_path)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return out_path


def load_bars(path: Path) -> pl.DataFrame:
    """Load a Parquet file of bars into a Polars DataFrame."""
    return pl.read_parquet(path)
