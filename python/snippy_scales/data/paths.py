"""Shared filesystem locations for the data-ingestion package.

Split out from :mod:`snippy_scales.data.ingest` so that module can import
:mod:`snippy_scales.data.tick` (for the unified config-driven ingestion
entrypoint) without an import cycle — both ``ingest.py`` and ``tick.py``
depend on this leaf module instead of on each other for their root path.
"""

from __future__ import annotations

from pathlib import Path

RAW_DIR = Path("data/raw")
