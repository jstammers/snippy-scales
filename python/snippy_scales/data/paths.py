"""Shared filesystem locations for the data-ingestion package.

Split out from :mod:`snippy_scales.data.ingest` so that module can import
:mod:`snippy_scales.data.tick` (for the unified config-driven ingestion
entrypoint) without an import cycle — both ``ingest.py`` and ``tick.py``
depend on this leaf module instead of on each other for their root path.

The datastore root is relocatable via the ``SNIPPY_DATA_ROOT`` env var
(e.g. to point at an external disk) — read once at import time, defaulting
to ``"data"`` (relative to the process's working directory) when unset.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SNIPPY_DATA_ROOT", "data"))
RAW_DIR = DATA_ROOT / "raw"
UNIVERSE_DIR = DATA_ROOT / "universe"
