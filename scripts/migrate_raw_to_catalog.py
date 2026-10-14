"""One-off migration of the pre-catalog Parquet store into the Nautilus catalog.

Reads ``data/raw/<instrument_type>/<symbol>/...`` (the layout written before the
NautilusTrader migration) and writes it into ``data/catalog/``:

* Bar files ``<schema>.parquet`` -> catalog bars. Price precision is inferred
  from the data; ``ts_event`` was the bar open, so ``ts_init`` = open + bar length.
* Tick days are rebuilt from the lossless ``<schema>/_dbn/*.dbn.zst`` archive with
  ``DatabentoDataLoader`` (trades only). Days without a DBN file are reported.
* ``equities`` symbols also get an ``Equity`` instrument. Futures definitions come
  from Databento: run ``algo data sync configs/databento.yaml --instruments-only``.

Safety: ``data/raw`` is opened read-only and nothing in it is modified or
deleted. Writes are incremental (data already in the catalog is skipped), so the
script can be re-run after an interruption. Deleting ``data/raw`` afterwards is
a manual decision — see CLAUDE.md.

Usage::

    uv run python scripts/migrate_raw_to_catalog.py --dry-run
    uv run python scripts/migrate_raw_to_catalog.py
    uv run python scripts/migrate_raw_to_catalog.py --verify-only
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from nautilus_trader.model import InstrumentId, NautilusDataType

from snippy_scales.data import catalog as cat
from snippy_scales.data.paths import CATALOG_DIR, RAW_DIR

#: Default venue per legacy ``instrument_type`` directory.
DEFAULT_VENUES = {"futures": "GLBX", "equities": "ALPACA"}

_TICK_LOADERS = {"trades": "load_trades"}


@dataclass
class BarCheck:
    symbol: str
    venue: str
    schema: str
    raw_rows: int
    catalog_rows: int
    raw_checksum: int
    catalog_checksum: int

    @property
    def ok(self) -> bool:
        return self.raw_rows == self.catalog_rows and self.raw_checksum == self.catalog_checksum


def find_bar_files(raw_dir: Path) -> list[tuple[str, str, str, Path]]:
    """Every ``(instrument_type, symbol, schema, path)`` bar file under *raw_dir*."""
    found = []
    for path in sorted(raw_dir.glob("*/*/*.parquet")):
        schema = path.stem
        if schema in cat.BAR_SPECS:
            found.append((path.parent.parent.name, path.parent.name, schema, path))
    return found


def find_tick_dirs(raw_dir: Path) -> list[tuple[str, str, str, Path]]:
    """Every ``(instrument_type, symbol, schema, dbn_dir)`` tick archive under *raw_dir*."""
    return [
        (d.parent.parent.parent.name, d.parent.parent.name, d.parent.name, d)
        for d in sorted(raw_dir.glob("*/*/*/_dbn"))
        if d.is_dir()
    ]


def _read_raw_bars(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=["ts_event", "open", "high", "low", "close", "volume"])
    return df.sort("ts_event").unique("ts_event", keep="last", maintain_order=True)


def _checksum(close: np.ndarray, precision: int) -> int:
    return int(np.round(close * 10**precision).astype(np.int64).sum())


def migrate_bars(
    raw_dir: Path, catalog_root: Path, venues: dict[str, str], *, dry_run: bool = False
) -> list[tuple[str, str, str, int]]:
    """Copy every bar file into the catalog. Returns ``(symbol, venue, schema, written)``."""
    catalog = cat.open_catalog(catalog_root) if not dry_run else None
    done = []
    for instrument_type, symbol, schema, path in find_bar_files(raw_dir):
        venue = venues.get(instrument_type)
        if venue is None:
            print(f"skip {path}: no venue for instrument_type {instrument_type!r}")
            continue
        df = _read_raw_bars(path)
        if dry_run:
            print(
                f"would migrate {path} -> {cat.bar_type(symbol, venue, schema)} ({df.height} bars)"
            )
            continue
        assert catalog is not None  # noqa: S101
        bars = cat.frame_to_bars(
            df,
            bar_type=cat.bar_type(symbol, venue, schema),
            price_precision=cat.infer_price_precision(df),
            interval_ns=cat.bar_interval_ns(schema),
        )
        written = cat.write_bars(catalog, bars)
        if instrument_type == "equities":
            cat.write_instruments(catalog, [cat.equity_instrument(symbol, venue)])
        print(f"{symbol} {schema}: {written} of {df.height} bars written")
        done.append((symbol, venue, schema, written))
    return done


def migrate_ticks(
    raw_dir: Path,
    catalog_root: Path,
    venues: dict[str, str],
    *,
    dry_run: bool = False,
    loader: object = None,
) -> list[tuple[str, str, int]]:
    """Rebuild tick data from DBN archives. Returns ``(symbol, schema, records)``."""
    catalog = cat.open_catalog(catalog_root) if not dry_run else None
    done = []
    for instrument_type, symbol, schema, dbn_dir in find_tick_dirs(raw_dir):
        files = sorted(dbn_dir.glob("*.dbn.zst"))
        parquet_days = {
            p.parent.name.removeprefix("date=") for p in dbn_dir.parent.glob("date=*/data.parquet")
        }
        dbn_days = {f.name.removesuffix(".dbn.zst") for f in files}
        if missing := sorted(parquet_days - dbn_days):
            print(f"warning: {symbol} {schema}: {len(missing)} day(s) have no DBN file, skipped")
        venue = venues.get(instrument_type)
        method = _TICK_LOADERS.get(schema)
        if venue is None or method is None:
            print(f"skip {dbn_dir}: unsupported instrument_type/schema")
            continue
        if dry_run:
            print(f"would load {len(files)} DBN day(s) for {symbol} {schema}")
            continue
        if loader is None:
            from nautilus_trader.adapters.databento import DatabentoDataLoader  # noqa: PLC0415

            loader = DatabentoDataLoader()
        assert catalog is not None  # noqa: S101
        iid = InstrumentId.from_str(cat.instrument_id(symbol, venue))
        last = catalog.query_last_timestamp(NautilusDataType.TradeTick, str(iid))
        total = 0
        for f in files:
            records = [
                r for r in getattr(loader, method)(str(f), iid) if last is None or r.ts_init > last
            ]
            if records:
                catalog.write_trade_ticks(sorted(records, key=lambda r: r.ts_init))
                total += len(records)
        print(f"{symbol} {schema}: {total} records from {len(files)} DBN file(s)")
        done.append((symbol, schema, total))
    return done


def verify_bars(raw_dir: Path, catalog_root: Path, venues: dict[str, str]) -> list[BarCheck]:
    """Compare row counts and a close-price checksum between raw files and the catalog."""
    checks = []
    for instrument_type, symbol, schema, path in find_bar_files(raw_dir):
        venue = venues.get(instrument_type)
        if venue is None:
            continue
        raw = _read_raw_bars(path)
        precision = cat.infer_price_precision(raw)
        start = raw["ts_event"].min()
        end = raw["ts_event"].max()
        try:
            stored = cat.load_bars(symbol, schema, venue=venue, catalog_path=catalog_root)
            stored = stored.filter(pl.col("ts_event").is_between(start, end))
        except FileNotFoundError:
            stored = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
        checks.append(
            BarCheck(
                symbol,
                venue,
                schema,
                raw.height,
                stored.height,
                _checksum(raw["close"].to_numpy(), precision),
                _checksum(stored["close"].to_numpy(), precision),
            )
        )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="Legacy store (read-only).")
    parser.add_argument("--catalog", type=Path, default=CATALOG_DIR, help="Target catalog.")
    parser.add_argument("--futures-venue", default=DEFAULT_VENUES["futures"])
    parser.add_argument("--equities-venue", default=DEFAULT_VENUES["equities"])
    parser.add_argument("--dry-run", action="store_true", help="List what would be migrated.")
    parser.add_argument("--verify-only", action="store_true", help="Only run the comparison.")
    args = parser.parse_args(argv)

    if not args.raw.is_dir():
        print(f"No legacy store at {args.raw}")
        return 1
    venues = {"futures": args.futures_venue, "equities": args.equities_venue}

    if not args.verify_only:
        migrate_bars(args.raw, args.catalog, venues, dry_run=args.dry_run)
        migrate_ticks(args.raw, args.catalog, venues, dry_run=args.dry_run)
        if args.dry_run:
            return 0

    checks = verify_bars(args.raw, args.catalog, venues)
    failed = [c for c in checks if not c.ok]
    for c in checks:
        status = "ok" if c.ok else "MISMATCH"
        print(f"{status:8} {c.symbol:12} {c.schema:9} raw={c.raw_rows} catalog={c.catalog_rows}")
    print(f"{len(checks) - len(failed)}/{len(checks)} bar series verified")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
