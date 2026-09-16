"""Databento :class:`~snippy_scales.data.providers.base.BarProvider` implementation.

The Databento API call and the pandas → Polars conversion used to live
directly inside :func:`snippy_scales.data.ingest.upsert_symbol`. They're
factored out here so :func:`snippy_scales.data.ingest.upsert_bars` can treat
Databento as just one more provider alongside Alpaca.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from snippy_scales.data.batch import run_batch_job
from snippy_scales.data.schema import conform_bars

if TYPE_CHECKING:
    import databento as db
    import pandas as pd

    from snippy_scales.data.config import VALID_STYPES, DownloadMethod


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert a Databento pandas DataFrame to a Polars DataFrame.

    Promotes the ``ts_event`` index to a regular column when Databento
    returns it as the DataFrame index.

    Args:
        df: pandas DataFrame returned by ``DBNStore.to_df()``.

    Returns:
        Equivalent :class:`polars.DataFrame` with ``ts_event`` as a column.
    """
    if df.index.name is not None and df.index.name != "index":
        df = df.reset_index()
    return pl.from_pandas(df)


@dataclass
class DatabentoProvider:
    """Fetches OHLCV bars from the Databento Historical API.

    Attributes:
        dataset: Databento dataset code (e.g. ``"GLBX.MDP3"``).
        stype_in: Databento symbology type for the request.
        download_method: ``"batch"`` (default) or ``"streaming"`` — see
            :data:`~snippy_scales.data.config.DownloadMethod`.
    """

    dataset: str
    stype_in: VALID_STYPES = "raw_symbol"
    download_method: DownloadMethod = "batch"
    name: str = field(default="databento", init=False)
    resume_granularity: Literal["day", "timestamp"] = field(default="day", init=False)

    def fetch_bars(self, *, symbol: str, schema: str, start: str, end: str) -> pl.DataFrame:
        """Fetch bars for *symbol* over ``[start, end)`` and conform them.

        Args:
            symbol: Databento symbol (e.g. ``"ES.c.0"``).
            schema: Databento OHLCV schema name (e.g. ``"ohlcv-1d"``).
            start: Inclusive start date (``YYYY-MM-DD``).
            end: Exclusive end date (``YYYY-MM-DD``).

        Returns:
            Bars conformed to the shared schema, sorted by ``ts_event``.

        Raises:
            RuntimeError: If the Databento API call fails.
        """
        import databento as db  # noqa: PLC0415 — optional dep

        client = db.Historical()

        if self.download_method == "streaming":
            dfs = [
                self._fetch_streaming(client, symbol=symbol, schema=schema, start=start, end=end)
            ]
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                files = run_batch_job(
                    client,
                    dataset=self.dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=start,
                    end=end,
                    stype_in=self.stype_in,
                    split_duration="none",
                    output_dir=Path(tmp_dir),
                )
                dfs = [_to_polars(db.DBNStore.from_file(f).to_df()) for f in files]

        combined = pl.concat(dfs, how="diagonal") if len(dfs) > 1 else dfs[0]
        return conform_bars(combined, symbol=symbol, schema=schema).sort("ts_event")

    def _fetch_streaming(
        self, client: db.Historical, *, symbol: str, schema: str, start: str, end: str
    ) -> pl.DataFrame:
        """Fetch bars via the Historical Streaming API (``timeseries.get_range``)."""
        store = client.timeseries.get_range(
            dataset=self.dataset,
            symbols=[symbol],
            schema=schema,
            start=start,
            end=end,
            stype_in=self.stype_in,
        )
        return _to_polars(store.to_df())
