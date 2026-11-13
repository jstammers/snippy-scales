"""Shared filesystem locations for the data package.

* :data:`CATALOG_DIR` — the NautilusTrader ``ParquetDataCatalog`` holding all
  market data (bars, ticks, instrument definitions).
* :data:`RAW_DIR` — the pre-catalog Parquet store. Read only by the one-off
  migration script (``scripts/migrate_raw_to_catalog.py``); nothing writes to it.
* :data:`UNIVERSE_DIR` — resolved stock universes (e.g. S&P 500 membership).
* :data:`MANIFEST_DIR` — per-symbol job manifests for long backfills.
* :data:`CACHE_DIR` — small downloaded reference files (e.g. Databento publishers).

The datastore root is relocatable via the ``SNIPPY_DATA_ROOT`` env var
(e.g. to point at an external disk) — read once at import time, defaulting
to ``"data"`` (relative to the process's working directory) when unset.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SNIPPY_DATA_ROOT", "data"))
CATALOG_DIR = DATA_ROOT / "catalog"
RAW_DIR = DATA_ROOT / "raw"
UNIVERSE_DIR = DATA_ROOT / "universe"
MANIFEST_DIR = DATA_ROOT / "derived" / "manifests"
CACHE_DIR = DATA_ROOT / "cache"
