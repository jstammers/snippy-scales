"""Unit tests for snippy_scales.data.paths — the SNIPPY_DATA_ROOT env var."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import pytest


def _reload_paths() -> ModuleType:
    import snippy_scales.data.paths as paths_module

    return importlib.reload(paths_module)


class TestDataRoot:
    def test_defaults_to_data_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SNIPPY_DATA_ROOT", raising=False)
        paths = _reload_paths()
        assert Path("data") == paths.DATA_ROOT
        assert Path("data/raw") == paths.RAW_DIR
        assert Path("data/universe") == paths.UNIVERSE_DIR

    def test_honors_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SNIPPY_DATA_ROOT", "/mnt/bigdisk/algo-data")
        paths = _reload_paths()
        assert Path("/mnt/bigdisk/algo-data") == paths.DATA_ROOT
        assert Path("/mnt/bigdisk/algo-data/raw") == paths.RAW_DIR
        assert Path("/mnt/bigdisk/algo-data/universe") == paths.UNIVERSE_DIR

    def teardown_method(self) -> None:
        # Restore the module to its real-environment state for subsequent tests.
        _reload_paths()
