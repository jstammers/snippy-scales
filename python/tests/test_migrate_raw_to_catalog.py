"""Tests for scripts/migrate_raw_to_catalog.py (loaded from its file path)."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest
from nautilus_trader.model import (
    AggressorSide,
    InstrumentId,
    NautilusDataType,
    Price,
    Quantity,
    TradeId,
    TradeTick,
)

from snippy_scales.data import catalog as cat

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_raw_to_catalog.py"


@pytest.fixture(scope="module")
def mig() -> ModuleType:
    spec = importlib.util.spec_from_file_location("migrate_raw_to_catalog", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def _raw_bars(n: int, start: dt.datetime, step: dt.timedelta, base: float) -> pl.DataFrame:
    """A legacy bar file: Databento-shaped columns, ts_event at the bar open."""
    ts = [start + i * step for i in range(n)]
    return pl.DataFrame(
        {
            "ts_event": pl.Series(ts).cast(pl.Datetime("ns", "UTC")),
            "rtype": [35] * n,
            "publisher_id": [1] * n,
            "instrument_id": [123] * n,
            "open": [base + i * 0.25 for i in range(n)],
            "high": [base + i * 0.25 + 1 for i in range(n)],
            "low": [base + i * 0.25 - 1 for i in range(n)],
            "close": [base + i * 0.25 + 0.5 for i in range(n)],
            "volume": pl.Series([100 + i for i in range(n)], dtype=pl.UInt64),
            "symbol": ["X"] * n,
        }
    )


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    es = root / "futures" / "ES.c.0"
    es.mkdir(parents=True)
    _raw_bars(30, dt.datetime(2024, 1, 1), dt.timedelta(days=1), 4500.0).write_parquet(
        es / "ohlcv-1d.parquet"
    )
    aapl = root / "equities" / "AAPL"
    aapl.mkdir(parents=True)
    _raw_bars(120, dt.datetime(2024, 1, 2, 14, 30), dt.timedelta(minutes=1), 185.0).write_parquet(
        aapl / "ohlcv-1m.parquet"
    )
    # A schema the catalog doesn't support is skipped.
    _raw_bars(3, dt.datetime(2024, 1, 1), dt.timedelta(days=1), 1.0).write_parquet(
        es / "ohlcv-eod.parquet"
    )
    return root


def _digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_migrates_verifies_and_leaves_raw_untouched(
    mig: ModuleType, raw: Path, tmp_path: Path
) -> None:
    before = _digest(raw)
    catalog_root = tmp_path / "catalog"

    assert mig.main(["--raw", str(raw), "--catalog", str(catalog_root)]) == 0

    assert _digest(raw) == before
    es = cat.load_bars("ES.c.0", "ohlcv-1d", catalog_path=catalog_root)
    assert es.height == 30
    assert es["close"][0] == 4500.5
    assert (es["ts_init"] - es["ts_event"]).dt.total_days().unique().to_list() == [1]
    aapl = cat.load_bars("AAPL", "ohlcv-1m", venue="ALPACA", catalog_path=catalog_root)
    assert aapl.height == 120
    instruments = [str(i.id) for i in cat.open_catalog(catalog_root).instruments()]
    assert instruments == ["AAPL.ALPACA"]


def test_rerun_is_idempotent(mig: ModuleType, raw: Path, tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    mig.migrate_bars(raw, catalog_root, mig.DEFAULT_VENUES)
    again = mig.migrate_bars(raw, catalog_root, mig.DEFAULT_VENUES)
    assert {written for *_, written in again} == {0}


def test_verify_detects_missing_series(mig: ModuleType, raw: Path, tmp_path: Path) -> None:
    checks = mig.verify_bars(raw, tmp_path / "empty-catalog", mig.DEFAULT_VENUES)
    assert checks
    assert not any(c.ok for c in checks)
    assert mig.main(["--raw", str(raw), "--catalog", str(tmp_path / "c2"), "--verify-only"]) == 1


def test_dry_run_writes_nothing(mig: ModuleType, raw: Path, tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    assert mig.main(["--raw", str(raw), "--catalog", str(catalog_root), "--dry-run"]) == 0
    assert not catalog_root.exists()


class _FakeLoader:
    def load_trades(self, path: str, instrument_id: InstrumentId) -> list[TradeTick]:
        day = dt.datetime.fromisoformat(Path(path).name.removesuffix(".dbn.zst")).replace(
            tzinfo=dt.UTC
        )
        base = int(day.timestamp()) * 1_000_000_000
        return [
            TradeTick(
                instrument_id,
                Price(4500.25, 2),
                Quantity(1, 0),
                AggressorSide.BUY,
                TradeId(f"{base}-{i}"),
                base + i,
                base + i,
            )
            for i in range(5)
        ]


def test_ticks_rebuilt_from_dbn(mig: ModuleType, raw: Path, tmp_path: Path) -> None:
    dbn = raw / "futures" / "ES.c.0" / "trades" / "_dbn"
    dbn.mkdir(parents=True)
    for day in ("2024-01-02", "2024-01-03"):
        (dbn / f"{day}.dbn.zst").write_bytes(b"")
    catalog_root = tmp_path / "catalog"

    done = mig.migrate_ticks(raw, catalog_root, mig.DEFAULT_VENUES, loader=_FakeLoader())
    assert done == [("ES.c.0", "trades", 10)]
    again = mig.migrate_ticks(raw, catalog_root, mig.DEFAULT_VENUES, loader=_FakeLoader())
    assert again == [("ES.c.0", "trades", 0)]
    catalog = cat.open_catalog(catalog_root)
    assert catalog.query_last_timestamp(NautilusDataType.TradeTick, "ES.c.0.GLBX") is not None
