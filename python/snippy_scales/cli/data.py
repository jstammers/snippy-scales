"""CLI commands for market data: sync into the catalog, inspect, maintain."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from nautilus_trader.persistence import ParquetDataCatalog

    from snippy_scales.data.catalog import SyncResult
    from snippy_scales.data.config import IngestConfig

app = typer.Typer(help="Market data commands (NautilusTrader catalog).")
console = Console()


def _fmt_cost(cost: float) -> str:
    if math.isnan(cost):
        return "[yellow]N/A[/yellow]"
    if cost == 0.0:
        return "[dim]—[/dim]"
    return f"[bold yellow]${cost:.4f}[/bold yellow]"


def _fmt_ns(ns: int | None) -> str:
    if ns is None:
        return "—"
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.UTC).strftime("%Y-%m-%d %H:%M")


def _fmt_ranges(ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return "[dim]up to date[/dim]"
    return ", ".join(f"{_fmt_ns(lo)} → {_fmt_ns(hi)}" for lo, hi in ranges)


def _day_ns(value: str | None) -> int:
    """``YYYY-MM-DD`` (or today when None) at 00:00 UTC, in UNIX ns."""
    day = dt.date.fromisoformat(value) if value else dt.datetime.now(dt.UTC).date()
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()) * 10**9


def _print_results(results: list[SyncResult]) -> None:
    table = Table(title="Sync results", show_header=True, header_style="bold cyan")
    table.add_column("Symbol")
    table.add_column("Schema")
    table.add_column("Written", justify="right")
    table.add_column("Status")
    for r in results:
        status = "[green]ok[/green]" if r.succeeded else f"[red]{r.error}[/red]"
        table.add_row(r.symbol, r.schema, f"{r.written:,}", status)
    console.print(table)


@app.command()
def sync(
    config: Annotated[Path, typer.Argument(help="Sync config YAML (see configs/).")],
    schema: Annotated[
        str | None,
        typer.Option("--schema", help="Sync only this schema ('1d', '1m', 'trades', ...)."),
    ] = None,
    symbol: Annotated[
        list[str] | None,
        typer.Option("--symbol", "-s", help="Sync only these symbols (repeatable)."),
    ] = None,
    catalog_path: Annotated[
        Path | None, typer.Option("--catalog", help="Catalog root. Default: data/catalog.")
    ] = None,
    instruments: Annotated[
        bool,
        typer.Option(
            "--instruments/--no-instruments",
            help=(
                "Also sync Databento instrument definitions. "
                "(Alpaca always writes Equity instruments.)"
            ),
        ),
    ] = True,
    instruments_only: Annotated[
        bool,
        typer.Option(
            "--instruments-only", help="Sync Databento instrument definitions only, no data."
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what is missing (and its cost); fetch nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Bring the catalog up to date for every symbol and schema in CONFIG.

    Only data missing from the catalog is requested: anything before the first
    or after the last stored record within the configured date range.

    Examples::

        algo data sync configs/databento.yaml --dry-run
        algo data sync configs/databento.yaml --schema 1h
        algo data sync configs/alpaca.yaml -s AAPL -s MSFT
        algo data sync configs/databento.yaml --instruments-only
    """
    from snippy_scales.data import catalog as cat  # noqa: PLC0415
    from snippy_scales.data.config import load_config, resolve_schema  # noqa: PLC0415

    cfg = load_config(config)
    schemas = [resolve_schema(schema)] if schema else cfg.resolved_schemas
    symbols = list(symbol) if symbol else cfg.all_symbols
    start, end = _day_ns(cfg.start), _day_ns(cfg.end)
    catalog = cat.open_catalog(catalog_path)

    if instruments_only:
        if cfg.provider != "databento":
            console.print("[red]--instruments-only applies to Databento configs only.[/]")
            raise typer.Exit(1)
        from snippy_scales.data.databento import DatabentoSync  # noqa: PLC0415

        syncer = DatabentoSync(catalog, dataset=cfg.dataset, venue=cfg.effective_venue)
        count = syncer.sync_instruments(symbols, start, end)
        console.print(f"[green]Instrument definitions processed: {count}[/]")
        return

    if cfg.provider == "alpaca":
        _sync_alpaca(cfg, catalog, symbols, schemas, start, end, dry_run, yes)
    else:
        _sync_databento(cfg, catalog, symbols, schemas, start, end, instruments, dry_run, yes)


def _sync_databento(
    cfg: IngestConfig,
    catalog: ParquetDataCatalog,
    symbols: list[str],
    schemas: list[str],
    start: int,
    end: int,
    instruments: bool,
    dry_run: bool,
    yes: bool,
) -> None:
    from snippy_scales.data.catalog import SyncResult  # noqa: PLC0415
    from snippy_scales.data.databento import DatabentoSync  # noqa: PLC0415

    syncer = DatabentoSync(catalog, dataset=cfg.dataset, venue=cfg.effective_venue)

    table = Table(title=f"Databento sync plan ({cfg.dataset})", header_style="bold cyan")
    table.add_column("Symbol")
    table.add_column("Schema")
    table.add_column("Missing")
    table.add_column("Cost (USD)", justify="right")
    total = 0.0
    todo: list[tuple[str, str]] = []
    for sym in symbols:
        for sch in schemas:
            ranges = syncer.plan(sym, sch, start, end)
            cost = 0.0
            if ranges:
                todo.append((sym, sch))
                try:
                    cost = syncer.estimate_cost(sym, sch, start, end, stype_in=cfg.stype_in)
                except Exception as exc:  # noqa: BLE001 — cost is advisory
                    console.print(f"[yellow]Cost estimate failed for {sym} {sch}: {exc}[/]")
                    cost = float("nan")
            total += cost
            table.add_row(sym, sch, _fmt_ranges(ranges), _fmt_cost(cost))
    console.print(table)
    console.print(f"Estimated total: {_fmt_cost(total)}")

    if dry_run:
        console.print("[yellow]Dry run — nothing fetched.[/]")
        return
    if not todo and not instruments:
        console.print("[green]Catalog is up to date.[/]")
        return
    if todo and not yes and not typer.confirm("Proceed?"):
        raise typer.Abort()

    if instruments:
        count = syncer.sync_instruments(symbols, start, end)
        console.print(f"[dim]Instrument definitions processed: {count}[/]")
    results = []
    for sym, sch in todo:
        try:
            results.append(SyncResult(sym, sch, written=syncer.sync(sym, sch, start, end)))
        except Exception as exc:  # noqa: BLE001 — reported per series
            results.append(SyncResult(sym, sch, error=f"{type(exc).__name__}: {exc}"))
    _print_results(results)


def _sync_alpaca(
    cfg: IngestConfig,
    catalog: ParquetDataCatalog,
    symbols: list[str],
    schemas: list[str],
    start: int,
    end: int,
    dry_run: bool,
    yes: bool,
) -> None:
    from nautilus_trader.model import NautilusDataType  # noqa: PLC0415

    from snippy_scales.data import catalog as cat  # noqa: PLC0415
    from snippy_scales.data.alpaca import AlpacaLoader  # noqa: PLC0415

    venue = cfg.effective_venue
    pending = 0
    for sch in schemas:
        for sym in symbols:
            ranges = cat.missing_ranges(
                catalog,
                NautilusDataType.Bar,
                cat.bar_type(sym, venue, sch),
                start,
                end,
                interval_ns=cat.bar_interval_ns(sch),
            )
            pending += bool(ranges)
    console.print(
        f"Alpaca: {pending} of {len(symbols) * len(schemas)} symbol/schema series need data "
        f"({_fmt_ns(start)} → {_fmt_ns(end)})."
    )
    if dry_run:
        console.print("[yellow]Dry run — nothing fetched.[/]")
        return
    if pending == 0:
        console.print("[green]Catalog is up to date.[/]")
        return
    if not yes and not typer.confirm("Proceed?"):
        raise typer.Abort()

    opts = cfg.alpaca_options
    loader = AlpacaLoader(
        feed=opts.feed,
        adjustment=opts.adjustment,
        rate_limit_per_min=opts.rate_limit_per_min,
        max_workers=opts.max_workers,
        venue=venue,
    )
    results = []
    for sch in schemas:
        results.extend(loader.sync(catalog, symbols, sch, start, end))
    _print_results(results)


@app.command()
def coverage(
    catalog_path: Annotated[
        Path | None, typer.Option("--catalog", help="Catalog root. Default: data/catalog.")
    ] = None,
) -> None:
    """List every bar series in the catalog with its first and last bar close."""
    from nautilus_trader.model import NautilusDataType  # noqa: PLC0415

    from snippy_scales.data import catalog as cat  # noqa: PLC0415

    catalog = cat.open_catalog(catalog_path)
    table = Table(title="Catalog bar coverage", header_style="bold cyan")
    for col in ("Symbol", "Venue", "Schema", "First close", "Last close"):
        table.add_column(col)
    rows = 0
    for sch in cat.BAR_SPECS:
        for sym, venue in cat.list_bar_series(sch, catalog_path):
            bt = cat.bar_type(sym, venue, sch)
            table.add_row(
                sym,
                venue,
                sch,
                _fmt_ns(catalog.query_first_timestamp(NautilusDataType.Bar, bt)),
                _fmt_ns(catalog.query_last_timestamp(NautilusDataType.Bar, bt)),
            )
            rows += 1
    if rows:
        console.print(table)
    else:
        console.print("[yellow]No bar data in the catalog yet.[/]")
    console.print(f"Instruments: {len(catalog.instruments())}")
    data_types = ", ".join(str(t).rsplit(".", 1)[-1] for t in catalog.list_data_types())
    console.print(f"Data types: {data_types or '—'}")


@app.command()
def consolidate(
    catalog_path: Annotated[
        Path | None, typer.Option("--catalog", help="Catalog root. Default: data/catalog.")
    ] = None,
) -> None:
    """Merge the many small files left by incremental syncs into larger ones."""
    from snippy_scales.data import catalog as cat  # noqa: PLC0415

    cat.open_catalog(catalog_path).consolidate_catalog()
    console.print("[green]Catalog consolidated.[/]")


@app.command(name="update-universe")
def update_universe(
    universe: Annotated[
        str, typer.Option(help="Universe to resolve. Currently only 'sp500'.")
    ] = "sp500",
    years: Annotated[int, typer.Option(help="How many years back from --end to include.")] = 5,
    end: Annotated[str | None, typer.Option(help="End date YYYY-MM-DD. Defaults to today.")] = None,
    output: Annotated[
        Path | None, typer.Option(help="Where to write the resolved ticker list.")
    ] = None,
) -> None:
    """Resolve and cache a stock universe as a `symbols_file` for a sync config.

    Currently only `sp500` is supported: every ticker that was an S&P 500
    constituent at any point in `[end - years, end]`, sourced from Wikipedia
    (see `snippy_scales.data.universe`), so a config that reads the file isn't
    survivorship-biased.

    Examples::

        algo data update-universe
        algo data update-universe --years 10 --output data/universe/sp500_10y.txt
    """
    from snippy_scales.data.backfill import (  # noqa: PLC0415
        DEFAULT_SYMBOLS_CACHE,
        resolve_backfill_window,
        save_symbols_cache,
    )
    from snippy_scales.data.universe import sp500_ever_members  # noqa: PLC0415

    if universe != "sp500":
        console.print(f"[bold red]Unknown universe:[/] {universe!r}. Only 'sp500' is supported.")
        raise typer.Exit(1)

    out_path = output if output is not None else DEFAULT_SYMBOLS_CACHE
    start_date, end_date = resolve_backfill_window(years, end)

    console.print(f"Resolving S&P 500 ever-members for {start_date} → {end_date} from Wikipedia …")
    symbols = sp500_ever_members(start_date, end_date)
    save_symbols_cache(out_path, symbols)
    console.print(f"[green]Resolved {len(symbols)} ticker(s)[/] — saved to {out_path}")


@app.command(name="backfill-sp500")
def backfill_sp500(
    schema: Annotated[
        str,
        typer.Option("--schema", help="Bar frequency: '1m' (default), '1h' or '1d'/'daily'."),
    ] = "1m",
    years: Annotated[int, typer.Option(help="How many years back from --end to backfill.")] = 5,
    end: Annotated[str | None, typer.Option(help="End date YYYY-MM-DD. Defaults to today.")] = None,
    catalog_path: Annotated[
        Path | None, typer.Option("--catalog", help="Catalog root. Default: data/catalog.")
    ] = None,
    manifest_path: Annotated[
        Path | None, typer.Option(help="CSV manifest of per-symbol results.")
    ] = None,
    symbols_cache: Annotated[
        Path | None,
        typer.Option(help="Where the resolved S&P 500 universe is cached for reproducibility."),
    ] = None,
    refresh_universe: Annotated[
        bool,
        typer.Option(
            "--refresh-universe",
            help="Re-fetch the S&P 500 universe from Wikipedia even if a cache exists.",
        ),
    ] = False,
    rate_limit_per_min: Annotated[
        int, typer.Option(help="Alpaca historical API call budget per minute (free tier: 200).")
    ] = 190,
    max_workers: Annotated[
        int,
        typer.Option(help="Symbols fetched concurrently (throughput is capped by the rate limit)."),
    ] = 8,
    feed: Annotated[Literal["iex", "sip"], typer.Option(help="Alpaca data feed.")] = "sip",
    adjustment: Annotated[
        Literal["raw", "split", "dividend", "all"],
        typer.Option(help="Corporate-action adjustment."),
    ] = "all",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Resolve the universe and print the plan; download nothing."
        ),
    ] = False,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed",
            help="Only re-run symbols marked failed in an existing --manifest-path.",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Backfill Alpaca bars into the catalog for every stock ever in the S&P 500.

    Pulls `--years` (default 5) of `--schema` (default `1m`) bars for every
    ticker that was a constituent at any point in that window, so the dataset
    isn't survivorship-biased. Respects the Alpaca rate limit across all
    threads and resumes automatically: re-running only fetches what the catalog
    is missing, including a widened range. Requires `ALPACA_API_KEY` /
    `ALPACA_SECRET_KEY`.

    Examples::

        algo data backfill-sp500 --dry-run
        algo data backfill-sp500 --schema 1d
        algo data backfill-sp500 --retry-failed
    """
    from snippy_scales.data import catalog as cat  # noqa: PLC0415
    from snippy_scales.data.alpaca import SUPPORTED_BAR_SCHEMAS, AlpacaLoader  # noqa: PLC0415
    from snippy_scales.data.backfill import (  # noqa: PLC0415
        DEFAULT_CATALOG_DIR,
        DEFAULT_SYMBOLS_CACHE,
        default_manifest_path,
        estimate_backfill_plan,
        load_symbols_cache,
        merge_manifest,
        read_manifest,
        resolve_backfill_window,
        save_symbols_cache,
    )
    from snippy_scales.data.config import resolve_schema  # noqa: PLC0415
    from snippy_scales.data.universe import sp500_ever_members  # noqa: PLC0415

    try:
        resolved_schema = resolve_schema(schema)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(1) from None
    if resolved_schema not in SUPPORTED_BAR_SCHEMAS:
        console.print(
            f"[bold red]--schema {schema!r} resolves to {resolved_schema!r}, which Alpaca "
            f"does not support.[/] Supported: {', '.join(sorted(SUPPORTED_BAR_SCHEMAS))}."
        )
        raise typer.Exit(1)

    root = catalog_path if catalog_path is not None else DEFAULT_CATALOG_DIR
    manifest = manifest_path if manifest_path is not None else default_manifest_path(schema)
    cache_path = symbols_cache if symbols_cache is not None else DEFAULT_SYMBOLS_CACHE
    start_date, end_date = resolve_backfill_window(years, end)

    if retry_failed:
        manifest_rows = read_manifest(manifest)
        if not manifest_rows:
            console.print(f"[bold red]No manifest found at {manifest} — nothing to retry.[/]")
            raise typer.Exit(1)
        symbols = sorted(
            row["symbol"] for row in manifest_rows.values() if row["succeeded"] != "True"
        )
        if not symbols:
            console.print("[green]Nothing to retry — every symbol in the manifest succeeded.[/]")
            raise typer.Exit(0)
        console.print(f"[cyan]Retrying {len(symbols)} previously-failed symbol(s).[/]")
    else:
        cached = None if refresh_universe else load_symbols_cache(cache_path)
        if cached is not None:
            console.print(
                f"[dim]Using cached universe ({len(cached)} tickers) from {cache_path}[/]"
            )
            symbols = cached
        else:
            console.print(
                f"Resolving S&P 500 ever-members for {start_date} → {end_date} from Wikipedia …"
            )
            symbols = sp500_ever_members(start_date, end_date)
            save_symbols_cache(cache_path, symbols)
            console.print(f"[green]Resolved {len(symbols)} ticker(s)[/] — cached to {cache_path}")

    estimate = estimate_backfill_plan(
        num_symbols=len(symbols),
        start=start_date,
        end=end_date,
        rate_limit_per_min=rate_limit_per_min,
        schema=resolved_schema,
    )
    table = Table(title="S&P 500 Backfill Plan", show_header=False)
    table.add_row("Schema", resolved_schema)
    table.add_row("Symbols", str(len(symbols)))
    table.add_row("Range", f"{start_date} → {end_date}  ({years}y)")
    table.add_row("Feed / Adjustment", f"{feed} / {adjustment}")
    table.add_row("Rate limit", f"{rate_limit_per_min}/min  ({max_workers} concurrent symbols)")
    table.add_row("Est. requests (upper bound)", f"{estimate['total_requests']:,.0f}")
    table.add_row("Est. runtime (upper bound)", f"{estimate['estimated_minutes'] / 60:.1f}h")
    table.add_row("Catalog", str(root))
    table.add_row("Manifest", str(manifest))
    console.print(table)
    console.print(
        "[dim]Estimate assumes every symbol needs the full range — data already in the "
        "catalog is skipped, so a resumed run is faster.[/]"
    )

    if dry_run:
        console.print("\n[yellow]Dry-run mode — no data downloaded.[/]")
        return
    if not yes and not typer.confirm("\nProceed with download?"):
        raise typer.Abort()

    loader = AlpacaLoader(
        feed=feed,
        adjustment=adjustment,
        rate_limit_per_min=rate_limit_per_min,
        max_workers=max_workers,
    )
    console.print(f"\n[bold]Starting backfill of {len(symbols)} symbol(s) …[/]")
    rows = loader.sync(
        cat.open_catalog(root),
        symbols,
        resolved_schema,
        _day_ns(start_date.isoformat()),
        _day_ns(end_date.isoformat()),
    )
    merged_manifest = merge_manifest(manifest, rows)

    succeeded = sum(1 for row in rows if row.succeeded)
    failed = len(rows) - succeeded
    console.print(
        f"\n[bold green]Done.[/]  {succeeded}/{len(rows)} symbol(s) succeeded this run"
        + (f", [bold red]{failed} failed[/]" if failed else ".")
    )
    console.print(f"[dim]Manifest ({len(merged_manifest)} symbol(s) total):[/] {manifest}")
    if failed:
        console.print(
            "[yellow]Re-run with[/] [bold]--retry-failed[/] [yellow]to retry just the failures.[/]"
        )
