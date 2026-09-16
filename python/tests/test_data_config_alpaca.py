"""Unit tests for the Alpaca-related additions to snippy_scales.data.config:
`provider`, `AlpacaConfig`, and `AssetClassConfig.symbols_file`.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from snippy_scales.data.config import (
    AlpacaConfig,
    AssetClassConfig,
    IngestConfig,
    load_config,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestProviderField:
    def test_defaults_to_databento(self) -> None:
        cfg = IngestConfig(
            start="2020-01-01", asset_classes={"eq": AssetClassConfig(symbols=["ES.c.0"])}
        )
        assert cfg.provider == "databento"

    def test_alpaca_provider_accepted(self) -> None:
        cfg = IngestConfig(
            provider="alpaca",
            start="2020-01-01",
            schemas=["1m"],
            asset_classes={"eq": AssetClassConfig(symbols=["AAPL"])},
        )
        assert cfg.provider == "alpaca"

    def test_alpaca_rejects_tick_schema(self) -> None:
        with pytest.raises(ValueError, match="does not support event-level schema"):
            IngestConfig(
                provider="alpaca",
                start="2020-01-01",
                schemas=["trades"],
                asset_classes={"eq": AssetClassConfig(symbols=["AAPL"])},
            )

    def test_alpaca_options_default_when_alpaca_block_absent(self) -> None:
        cfg = IngestConfig(
            provider="alpaca",
            start="2020-01-01",
            asset_classes={"eq": AssetClassConfig(symbols=["AAPL"])},
        )
        assert cfg.alpaca_options == AlpacaConfig()

    def test_alpaca_options_uses_explicit_block(self) -> None:
        cfg = IngestConfig(
            provider="alpaca",
            start="2020-01-01",
            alpaca=AlpacaConfig(feed="iex", rate_limit_per_min=100),
            asset_classes={"eq": AssetClassConfig(symbols=["AAPL"])},
        )
        assert cfg.alpaca_options.feed == "iex"
        assert cfg.alpaca_options.rate_limit_per_min == 100


class TestAlpacaConfig:
    def test_defaults(self) -> None:
        options = AlpacaConfig()
        assert options.feed == "sip"
        assert options.adjustment == "all"
        assert options.rate_limit_per_min == 190
        assert options.max_workers == 4

    def test_rate_limit_bounds(self) -> None:
        with pytest.raises(ValueError):
            AlpacaConfig(rate_limit_per_min=0)
        with pytest.raises(ValueError):
            AlpacaConfig(rate_limit_per_min=201)


class TestAssetClassSymbolsFile:
    def test_requires_symbols_or_symbols_file(self) -> None:
        with pytest.raises(ValueError, match="symbols.*symbols_file"):
            AssetClassConfig()

    def test_symbols_file_alone_is_valid(self, tmp_path: Path) -> None:
        ac = AssetClassConfig(symbols_file=tmp_path / "tickers.txt")
        assert ac.symbols == []
        assert ac.symbols_file is not None

    def test_load_config_merges_symbols_file(self, tmp_path: Path) -> None:
        symbols_file = tmp_path / "sp500.txt"
        symbols_file.write_text("AAPL\nMSFT\n# a comment\n\nGOOGL\n")

        config_file = tmp_path / "alpaca.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                provider: alpaca
                schemas: ["1m"]
                start: "2021-09-15"
                asset_classes:
                  sp500:
                    symbols: [BRK.B]
                    symbols_file: sp500.txt
            """)
        )

        cfg = load_config(config_file)
        assert cfg.all_symbols == ["BRK.B", "AAPL", "MSFT", "GOOGL"]

    def test_symbols_file_path_relative_to_config_dir(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        (sub_dir / "tickers.txt").write_text("AAPL\n")

        config_file = sub_dir / "alpaca.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                provider: alpaca
                start: "2021-09-15"
                schemas: ["1d"]
                asset_classes:
                  eq:
                    symbols_file: tickers.txt
            """)
        )

        cfg = load_config(config_file)
        assert cfg.all_symbols == ["AAPL"]

    def test_missing_symbols_file_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "alpaca.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                provider: alpaca
                start: "2021-09-15"
                schemas: ["1d"]
                asset_classes:
                  eq:
                    symbols_file: does_not_exist.txt
            """)
        )

        with pytest.raises(FileNotFoundError):
            load_config(config_file)

    def test_no_symbols_file_leaves_config_untouched(self, tmp_path: Path) -> None:
        config_file = tmp_path / "databento.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                start: "2020-01-01"
                schemas: ["1d"]
                asset_classes:
                  eq:
                    symbols: [ES.c.0]
            """)
        )

        cfg = load_config(config_file)
        assert cfg.all_symbols == ["ES.c.0"]
