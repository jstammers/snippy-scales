"""CLI commands for data ingestion and management."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from snippy_scales.data.config import VALID_STYPES, DownloadMethod

app = typer.Typer(help="Data ingestion commands.")
console = Console()


def _fmt_cost(cost: float, *, free_provider: bool = False) -> str:
    """Format a cost float for display in the terminal.

    Returns ``"free"`` for zero-cost when *free_provider* (Alpaca, which has
    no per-request charge at all), otherwise ``"cached"`` (already up to
    date), ``"N/A"`` for NaN (estimation failed), or a dollar-formatted
    string.
    """
    if cost == 0.0:
        return "[dim]free[/dim]" if free_provider else "[dim]cached[/dim]"
    if math.isnan(cost):
        return "[yellow]N/A[/yellow]"
    return f"[bold yellow]${cost:.4f}[/bold yellow]"


def _fmt_size(num_bytes: int) -> str:
    """Format a byte count using decimal units (Databento bills per GB)."""
    if num_bytes < 1_000_000:
        return f"{num_bytes / 1_000:.1f} KB"
    if num_bytes < 1_000_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    return f"{num_bytes / 1_000_000_000:.2f} GB"


def _ingest_bar(
    *,
    dataset: str,
    symbol: str,
    schema: str,
    start: str,
    end: str,
    stype_in: VALID_STYPES,
    instrument_type: str,
    output_dir: Path,
    yes: bool,
    download_method: DownloadMethod,
) -> None:
    """Cost-estimate, confirm, and download bar data for a single symbol."""
    from snippy_scales.data.ingest import estimate_cost, upsert_symbol  # noqa: PLC0415

    console.print(f"Estimating cost for [bold]{symbol}[/] ({schema})  {start} → {end} …")

    cost: float
    try:
        cost = estimate_cost(
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            start=start,
            end=end,
            instrument_type=instrument_type,
            output_dir=output_dir,
            stype_in=stype_in,
        )
    except Exception as exc:
        console.print(f"[yellow]Warning: cost estimation failed — {exc}[/]")
        cost = float("nan")

    if cost == 0.0:
        console.print("[dim]Symbol is already up to date — nothing to download.[/]")
        return

    console.print(f"Estimated cost: {_fmt_cost(cost)}")

    if not yes and not math.isnan(cost):
        if not typer.confirm("Proceed with download?"):
            raise typer.Abort()
    elif not yes and math.isnan(cost):
        console.print("[yellow]Cost estimate unavailable.[/]")
        if not typer.confirm("Proceed anyway?"):
            raise typer.Abort()

    console.print(
        f"Ingesting [bold]{symbol}[/] ({schema})  {start} → {end}  "
        f"[[dim]{dataset}, {download_method}[/]]"
    )
    if download_method == "batch":
        console.print("[dim]Submitting a Databento batch job — this may take a while...[/]")
    path = upsert_symbol(
        dataset=dataset,
        symbol=symbol,
        schema=schema,
        start=start,
        end=end,
        instrument_type=instrument_type,
        output_dir=output_dir,
        stype_in=stype_in,
        download_method=download_method,
    )
    console.print(f"[green]Saved:[/] {path}")


def _ingest_tick(
    *,
    dataset: str,
    symbol: str,
    schema: str,
    start: str,
    end: str,
    stype_in: VALID_STYPES,
    instrument_type: str,
    output_dir: Path,
    yes: bool,
    download_method: DownloadMethod,
) -> None:
    """Cost-estimate, confirm, and download tick data for a single symbol."""
    from snippy_scales.data.tick import estimate_tick_cost, upsert_ticks  # noqa: PLC0415

    console.print(f"Estimating cost for [bold]{symbol}[/] ({schema})  {start} → {end} …")

    try:
        estimate = estimate_tick_cost(
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            start=start,
            end=end,
            instrument_type=instrument_type,
            output_dir=output_dir,
            stype_in=stype_in,
        )
    except Exception as exc:
        console.print(f"[bold red]Cost estimation failed:[/] {exc}")
        raise typer.Exit(1) from exc

    if not estimate.missing_days:
        console.print(
            f"[dim]All {estimate.cached_days} day(s) in range are already stored — "
            "nothing to download.[/]"
        )
        return

    table = Table(title="Tick Ingestion Plan", show_header=True, header_style="bold cyan")
    table.add_column("Symbol")
    table.add_column("Schema")
    table.add_column("Range")
    table.add_column("Days to fetch", justify="right")
    table.add_column("Cached", justify="right")
    table.add_column("Billable size", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_row(
        symbol,
        schema,
        f"{start} → {end}",
        str(len(estimate.missing_days)),
        str(estimate.cached_days),
        _fmt_size(estimate.billable_bytes),
        _fmt_cost(estimate.cost_usd),
    )
    console.print(table)

    if not yes and not typer.confirm("\nProceed with download?"):
        raise typer.Abort()

    if download_method == "batch":
        console.print("[dim]Submitting Databento batch job(s) — this may take a while...[/]")
    written = upsert_ticks(
        dataset=dataset,
        symbol=symbol,
        schema=schema,
        start=start,
        end=end,
        instrument_type=instrument_type,
        output_dir=output_dir,
        stype_in=stype_in,
        download_method=download_method,
    )

    remaining = len(estimate.missing_days) - len(written)
    console.print(
        f"\n[bold green]Done.[/]  {len(written)} day(s) written to "
        f"{output_dir / instrument_type / symbol.replace('/', '_') / schema}"
        + (f"  [dim]({remaining} empty or failed)[/]" if remaining else "")
    )


@app.command()
def ingest(
    dataset: str = typer.Argument(..., help="Databento dataset identifier"),
    symbol: str = typer.Option(
        ...,
        "--symbol",
        "-s",
        help=(
            "Instrument symbol. Use Databento parent symbology (e.g. 'ES.FUT' with "
            "--stype-in parent) to pull every individual outright contract under a "
            "futures root in one request."
        ),
    ),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    schema: Annotated[
        str,
        typer.Option(
            "--schema",
            help=(
                "Bar frequency ('1d', '1h', '1m', 'daily', ...) or event-level schema "
                "('trades', 'mbo', 'mbp-1', 'mbp-10', 'tbbo')."
            ),
        ),
    ] = "1d",
    stype_in: Literal["raw_symbol", "continuous", "parent", "instrument_id"] = typer.Option(
        "raw_symbol", help="Databento symbology type"
    ),
    instrument_class: str = typer.Option(
        ...,
        "--instrument-class",
        help=(
            "Closed top-level storage classification — determines the "
            "data/raw/<instrument-class>/ subdirectory. One of: equities, "
            "futures, options, fx_spot, crypto."
        ),
    ),
    download_method: Literal["batch", "streaming"] = typer.Option(
        "batch",
        "--download-method",
        help=(
            "Databento delivery mechanism. 'batch' (default) submits a batch job — "
            "billed the same as streaming, but Databento keeps completed job output "
            "downloadable free of charge for a retention window. 'streaming' calls "
            "the Historical Streaming API directly — lower latency, no retention."
        ),
    ),
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Root directory for raw data."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the cost-confirmation prompt."),
    ] = False,
) -> None:
    """Download and upsert data for a single symbol.

    Routes automatically to the bar or event-level (tick) ingestion path based
    on ``--schema``: bar aliases (``1d``, ``1h``, ...) use the upsert-by-date
    Parquet store, event-level schemas (``trades``, ``mbo``, ...) use the
    day-partitioned tick store.

    Before downloading, the Databento metadata API is queried to estimate
    cost. You will be asked to confirm unless ``--yes`` is passed. Existing
    data is extended rather than re-downloaded, so symbols that are already up
    to date incur no cost and are skipped automatically.

    Examples::

        algo data ingest GLBX.MDP3 -s ES.c.0 --schema 1d --start 2010-01-01 --end 2026-01-01
        algo data ingest GLBX.MDP3 -s ES.c.0 --schema trades --start 2026-08-01 --end 2026-09-01
        algo data ingest GLBX.MDP3 -s ES.FUT --schema 1d --stype-in parent \\
            --start 2020-01-01 --end 2026-01-01
    """
    from snippy_scales.data.config import is_tick_schema, resolve_schema  # noqa: PLC0415
    from snippy_scales.data.ingest import RAW_DIR  # noqa: PLC0415

    try:
        resolved = resolve_schema(schema)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    out_dir = output_dir or RAW_DIR

    if is_tick_schema(resolved):
        _ingest_tick(
            dataset=dataset,
            symbol=symbol,
            schema=resolved,
            start=start,
            end=end,
            stype_in=stype_in,
            instrument_type=instrument_class,
            output_dir=out_dir,
            yes=yes,
            download_method=download_method,
        )
    else:
        _ingest_bar(
            dataset=dataset,
            symbol=symbol,
            schema=resolved,
            start=start,
            end=end,
            stype_in=stype_in,
            instrument_type=instrument_class,
            output_dir=out_dir,
            yes=yes,
            download_method=download_method,
        )


@app.command(name="ingest-config")
def ingest_config(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the YAML ingestion config file.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    schema: Annotated[
        str | None,
        typer.Option(
            "--schema",
            help=(
                "Restrict to a single schema instead of the config file's full "
                "`schemas` list.  Accepts bar aliases ('1d', '1h', ...) or "
                "event-level schema names ('trades', 'mbo', ...)."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Root directory for raw data."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show the ingestion plan and cost estimate without downloading.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the cost-confirmation prompt."),
    ] = False,
) -> None:
    """Batch-ingest every symbol across every schema defined in a YAML config file.

    Every entry in ``schemas`` is run for every symbol, routed automatically
    to the bar or event-level (tick) store based on whether it names a bar
    frequency or an event-level schema. Use Databento parent symbology (e.g.
    ``ES.FUT`` with ``stype_in: parent``) as a symbol to pull every
    individual outright contract under a futures root in one request.

    Uses upsert semantics: only data not already on disk is fetched, so
    repeated runs are cheap. Before any download the Databento metadata API
    is queried once per item to estimate per-item and total costs. You will
    be asked to confirm the total spend unless ``--yes`` is passed.

    Examples::

        algo data ingest-config configs/databento.yaml
        algo data ingest-config configs/databento.yaml --schema 1h
        algo data ingest-config configs/databento.yaml --yes
        algo data ingest-config configs/databento.yaml --dry-run
    """
    from snippy_scales.data.config import load_config, resolve_schema  # noqa: PLC0415
    from snippy_scales.data.ingest import (  # noqa: PLC0415
        RAW_DIR,
        estimate_costs_from_config,
        ingest_from_config,
    )

    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error loading config:[/] {exc}")
        raise typer.Exit(1) from exc

    if schema is not None:
        try:
            resolve_schema(schema)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(1) from exc

    out_dir = output_dir or RAW_DIR
    effective_schemas = [resolve_schema(schema)] if schema else cfg.resolved_schemas
    is_free_provider = cfg.provider == "alpaca"
    source_label = (
        f"alpaca, {cfg.alpaca_options.rate_limit_per_min}/min"
        if is_free_provider
        else f"{cfg.dataset}, {cfg.download_method}"
    )

    # --- Plan (cost estimation is skipped for Alpaca — it's free, only rate-limited) ---
    console.print(
        f"{'Building' if is_free_provider else 'Estimating costs for'} plan for "
        f"[bold]{len(cfg.asset_classes)}[/] asset class(es) across "
        f"{len(effective_schemas)} schema(s) [[dim]{source_label}[/]] …"
    )
    cost_rows = estimate_costs_from_config(cfg, schema_override=schema, output_dir=out_dir)

    # --- Build plan table with per-item costs ---
    table = Table(title="Ingestion Plan", show_header=True, header_style="bold cyan")
    table.add_column("Asset Class")
    table.add_column("Schema")
    table.add_column("Symbol")
    table.add_column("Cost (USD)", justify="right")

    for row in cost_rows:
        cost_display = _fmt_cost(row.cost_usd, free_provider=is_free_provider)
        table.add_row(row.asset_class, row.schema, row.symbol, cost_display)

    console.print(table)

    # --- Totals ---
    finite_costs = [row.cost_usd for row in cost_rows if not math.isnan(row.cost_usd)]
    has_nan = any(math.isnan(row.cost_usd) for row in cost_rows)
    total = sum(finite_costs)
    total_str = (
        "[dim]free[/dim]" if is_free_provider else f"[bold yellow]${total:.4f}[/bold yellow]"
    )
    if has_nan and not is_free_provider:
        total_str += " [yellow]+ N/A[/yellow]"

    cached_count = sum(1 for row in cost_rows if row.cached)
    download_count = len(cost_rows) - cached_count

    console.print(
        f"\n[dim]Source:[/] {source_label}  "
        f"[dim]Output:[/] {out_dir}\n"
        f"[dim]Items to download:[/] {download_count}  "
        f"[dim]Already cached:[/] {cached_count}  "
        f"[dim]Total estimated cost:[/] {total_str}"
    )

    if dry_run:
        console.print("\n[yellow]Dry-run mode — no data downloaded.[/]")
        return

    # Skip download if everything is already cached
    if download_count == 0:
        console.print("\n[dim]Everything is already up to date — nothing to download.[/]")
        return

    # --- Confirmation ---
    if not yes and not typer.confirm("\nProceed with download?"):
        raise typer.Abort()

    if not is_free_provider and cfg.download_method == "batch":
        console.print("[dim]Submitting Databento batch job(s) — this may take a while...[/]")

    result_rows = ingest_from_config(cfg, schema_override=schema, output_dir=out_dir)

    # --- Summary ---
    succeeded = sum(1 for row in result_rows if row.succeeded)
    failed = len(result_rows) - succeeded
    console.print(
        f"\n[bold green]Done.[/]  "
        f"{succeeded} item(s) succeeded" + (f", [bold red]{failed} failed[/]" if failed else ".")
    )


@app.command()
def coverage(
    symbol: Annotated[str, typer.Option("--symbol", "-s", help="Instrument symbol")],
    schema: Annotated[
        str, typer.Option("--schema", help="Event-level schema: trades, mbo, ...")
    ] = "trades",
    instrument_class: Annotated[
        str | None,
        typer.Option(
            "--instrument-class",
            help=(
                "Closed top-level storage classification the symbol was ingested "
                "under. Omit to auto-resolve — errors if the symbol exists under "
                "more than one classification."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Root directory for raw data."),
    ] = None,
) -> None:
    """Report which days of tick data are stored locally, and any gaps.

    Useful for verifying an ingestion run completed and for confirming that a
    repeat download would be a no-op.
    """
    import datetime  # noqa: PLC0415

    from snippy_scales.data.config import resolve_schema  # noqa: PLC0415
    from snippy_scales.data.ingest import RAW_DIR  # noqa: PLC0415
    from snippy_scales.data.tick import contiguous_ranges, covered_days  # noqa: PLC0415

    try:
        resolved = resolve_schema(schema)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    out_dir = output_dir or RAW_DIR

    if instrument_class is None:
        safe_symbol = symbol.replace("/", "_")
        matches = [p for p in out_dir.glob(f"*/{safe_symbol}/{resolved}") if p.is_dir()]
        if len(matches) > 1:
            classes = ", ".join(m.parent.parent.name for m in matches)
            console.print(
                f"[bold red]Error:[/] {symbol!r} ({resolved}) exists under more than "
                f"one instrument class ({classes}) — pass --instrument-class to disambiguate."
            )
            raise typer.Exit(1)
        instrument_class = matches[0].parent.parent.name if matches else "unclassified"

    days = sorted(covered_days(symbol, resolved, out_dir, instrument_type=instrument_class))

    if not days:
        console.print(f"[yellow]No {resolved} data stored for {symbol}.[/]")
        return

    first, last = days[0], days[-1]
    covered = set(days)
    gaps = [
        first + datetime.timedelta(days=offset)
        for offset in range((last - first).days + 1)
        if first + datetime.timedelta(days=offset) not in covered
    ]

    console.print(
        f"[bold]{symbol}[/] ({resolved})\n"
        f"  [dim]Covered days:[/] {len(days)}\n"
        f"  [dim]Range:[/] {first} → {last}"
    )

    if not gaps:
        console.print("  [green]No gaps.[/]")
        return

    console.print(f"  [yellow]Gaps:[/] {len(gaps)} day(s)")
    for gap_start, gap_end in contiguous_ranges(gaps):
        span = (gap_end - gap_start).days
        console.print(
            f"    {gap_start} → {gap_end - datetime.timedelta(days=1)}  [dim]({span} day(s))[/]"
        )


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
    """Resolve and cache a stock universe as a `symbols_file` for an ingest config.

    Currently only `sp500` is supported: every ticker that was an S&P 500
    constituent at any point in `[end - years, end]`, sourced from Wikipedia
    (see `snippy_scales.data.universe`) — not just today's 500 names, so a
    config that reads the resulting file isn't survivorship-biased.

    The output is a plain text file, one symbol per line, suitable for an
    `asset_classes.<label>.symbols_file` entry in a YAML ingest config (see
    `configs/alpaca_sp500_1m.yaml`).

    Examples::

        algo data update-universe
        algo data update-universe --years 10 --output data/universe/sp500_10y.txt
    """
    from snippy_scales.data.backfill import (
        DEFAULT_SYMBOLS_CACHE,  # noqa: PLC0415
        resolve_backfill_window,  # noqa: PLC0415
        save_symbols_cache,  # noqa: PLC0415
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
    years: Annotated[int, typer.Option(help="How many years back from --end to backfill.")] = 5,
    end: Annotated[str | None, typer.Option(help="End date YYYY-MM-DD. Defaults to today.")] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Root directory for raw Parquet files.")
    ] = None,
    manifest_path: Annotated[
        Path | None, typer.Option(help="CSV manifest of per-symbol ingestion results.")
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
    """Backfill 1-minute bars from Alpaca for every stock ever in the S&P 500.

    Pulls `--years` (default 5) of `ohlcv-1m` bars for every ticker that was
    an S&P 500 constituent at any point in that window — not just today's
    500 names — via `algo data update-universe`'s underlying resolver, so
    the dataset isn't survivorship-biased. Respects the Alpaca free-tier
    rate limit (shared across all symbols/threads) and resumes automatically
    if interrupted: re-running the same command only fetches what's still
    missing. Requires `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in the
    environment.

    Examples::

        # Preview the plan (universe size, estimated requests/runtime) — no download.
        algo data backfill-sp500 --dry-run

        # Run it (prompts for confirmation; ~3h for the full universe on the free tier).
        algo data backfill-sp500

        # Re-run, only retrying symbols that failed last time.
        algo data backfill-sp500 --retry-failed
    """
    from snippy_scales.data.backfill import (  # noqa: PLC0415
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SYMBOLS_CACHE,
        estimate_backfill_plan,
        load_symbols_cache,
        merge_manifest,
        read_manifest,
        resolve_backfill_window,
        save_symbols_cache,
    )
    from snippy_scales.data.config import (  # noqa: PLC0415
        AlpacaConfig,
        AssetClassConfig,
        IngestConfig,
    )
    from snippy_scales.data.ingest import ingest_from_config  # noqa: PLC0415
    from snippy_scales.data.universe import sp500_ever_members  # noqa: PLC0415

    out_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    manifest = manifest_path if manifest_path is not None else DEFAULT_MANIFEST
    cache_path = symbols_cache if symbols_cache is not None else DEFAULT_SYMBOLS_CACHE

    start_date, end_date = resolve_backfill_window(years, end)

    # --- Resolve the universe ---
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

    # --- Plan ---
    estimate = estimate_backfill_plan(
        num_symbols=len(symbols),
        start=start_date,
        end=end_date,
        rate_limit_per_min=rate_limit_per_min,
    )
    table = Table(title="S&P 500 1-Minute Backfill Plan", show_header=False)
    table.add_row("Symbols", str(len(symbols)))
    table.add_row("Range", f"{start_date} → {end_date}  ({years}y)")
    table.add_row("Feed / Adjustment", f"{feed} / {adjustment}")
    table.add_row("Rate limit", f"{rate_limit_per_min}/min  ({max_workers} concurrent symbols)")
    table.add_row("Est. requests (upper bound)", f"{estimate['total_requests']:,.0f}")
    table.add_row("Est. runtime (upper bound)", f"{estimate['estimated_minutes'] / 60:.1f}h")
    table.add_row("Output", str(out_dir))
    table.add_row("Manifest", str(manifest))
    console.print(table)
    console.print(
        "[dim]Estimate assumes every symbol needs the full range — already-downloaded "
        "data is skipped automatically (upsert semantics), so a resumed run is faster.[/]"
    )

    if dry_run:
        console.print("\n[yellow]Dry-run mode — no data downloaded.[/]")
        return

    if not yes and not typer.confirm("\nProceed with download?"):
        raise typer.Abort()

    cfg = IngestConfig(
        provider="alpaca",
        instrument_type="equities",
        schemas=["1m"],
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        alpaca=AlpacaConfig(
            feed=feed,
            adjustment=adjustment,
            rate_limit_per_min=rate_limit_per_min,
            max_workers=max_workers,
        ),
        asset_classes={"sp500": AssetClassConfig(symbols=symbols)},
    )

    console.print(f"\n[bold]Starting backfill of {len(symbols)} symbol(s) …[/]")
    rows = ingest_from_config(cfg, output_dir=out_dir)
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


@app.command(name="check-layout")
def check_layout(
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Root directory for raw data."),
    ] = None,
) -> None:
    """Report symbols that exist under more than one instrument classification.

    Every symbol is meant to be unique within its top-level
    ``instrument_type`` directory (e.g. ``equities``, ``futures``) — the
    ``asset_classes`` grouping in a YAML config is a separate, free-form
    display label and may legitimately repeat a symbol across configs.
    Finding the same symbol under two different ``instrument_type``
    directories signals a real misconfiguration. Exits with status 1 if any
    are found.
    """
    from snippy_scales.data.ingest import RAW_DIR, find_cross_class_duplicates  # noqa: PLC0415

    out_dir = output_dir or RAW_DIR
    duplicates = find_cross_class_duplicates(out_dir)

    if not duplicates:
        console.print(f"[green]No cross-class duplicates found under {out_dir}.[/]")
        return

    table = Table(title="Cross-Class Duplicates", show_header=True, header_style="bold red")
    table.add_column("Symbol")
    table.add_column("Instrument classes")
    for symbol, classes in sorted(duplicates.items()):
        table.add_row(symbol, ", ".join(classes))
    console.print(table)
    raise typer.Exit(1)


@app.command(name="migrate-layout")
def migrate_layout(
    instrument_class: Annotated[
        str,
        typer.Option(
            "--instrument-class",
            help=(
                "Closed top-level storage classification every currently-flat "
                "symbol directory should move under."
            ),
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Root directory for raw data."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt and move files for real."),
    ] = False,
) -> None:
    """Move symbol directories from the old flat layout under an instrument class.

    Old layout: ``<output-dir>/<symbol>/...``
    New layout: ``<output-dir>/<instrument-class>/<symbol>/...``

    Dry-run by default — always review the printed plan before passing
    ``--yes``. Each symbol directory is moved with a single, atomic
    ``Path.rename()`` (never a copy-then-delete, never a recursive delete of
    anything). Refuses outright, before moving anything, if the destination
    for any symbol already exists and is non-empty.
    """
    from snippy_scales.data.ingest import RAW_DIR  # noqa: PLC0415

    out_dir = output_dir or RAW_DIR

    if not out_dir.exists():
        console.print(f"[yellow]{out_dir} does not exist — nothing to migrate.[/]")
        return

    # Known instrument_type directories are left alone; only directories that
    # look like flat "<symbol>/<schema>.parquet-or-tick-store" entries are
    # treated as migration candidates — i.e. anything not already itself an
    # instrument_type bucket.
    known_classes = {"equities", "futures", "options", "fx_spot", "crypto"}
    candidates = sorted(p for p in out_dir.iterdir() if p.is_dir() and p.name not in known_classes)

    if not candidates:
        console.print(f"[dim]Nothing to migrate under {out_dir} — already fully classified.[/]")
        return

    dest_root = out_dir / instrument_class
    moves = [(p, dest_root / p.name) for p in candidates]

    conflicts = [dest for _, dest in moves if dest.exists() and any(dest.iterdir())]
    if conflicts:
        console.print("[bold red]Refusing to migrate — destination already exists:[/]")
        for dest in conflicts:
            console.print(f"  {dest}")
        raise typer.Exit(1)

    table = Table(title="Layout Migration Plan", show_header=True, header_style="bold cyan")
    table.add_column("From")
    table.add_column("To")
    for src, dest in moves:
        table.add_row(str(src), str(dest))
    console.print(table)
    console.print(f"\n[dim]{len(moves)} symbol director(y/ies) → {dest_root}[/]")

    if not yes:
        console.print("\n[yellow]Dry-run mode — pass --yes to move these directories for real.[/]")
        return

    dest_root.mkdir(parents=True, exist_ok=True)
    for src, dest in moves:
        src.rename(dest)
    console.print(f"\n[bold green]Done.[/]  Moved {len(moves)} symbol director(y/ies).")
