"""Tests for research feature helpers."""

from __future__ import annotations

import polars as pl

from snippy_scales.research.features import realised_vol, zscore


def test_realised_vol_length() -> None:
    close = pl.Series("close", [float(i + 100) for i in range(50)])
    rv = realised_vol(close, window=10)
    assert len(rv) == len(close)


def test_zscore_length() -> None:
    s = pl.Series("x", [float(i) for i in range(80)])
    z = zscore(s, window=20)
    assert len(z) == len(s)
