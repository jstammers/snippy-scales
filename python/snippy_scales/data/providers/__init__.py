"""Bar-data providers.

Each provider implements :class:`~snippy_scales.data.providers.base.BarProvider`
and returns bars already conformed to the shared schema in
:mod:`snippy_scales.data.schema`, so
:func:`~snippy_scales.data.ingest.upsert_bars` and
:func:`~snippy_scales.data.ingest.load_bars` work identically regardless of
which provider fetched the data.
"""

from __future__ import annotations
