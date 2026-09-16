"""CLI commands for data ingestion and management."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from snippy_scales.data.config import VALID_STYPES

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
    output_dir: Path,
    yes: bool,
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

    console.print(f"Ingesting [bold]{symbol}[/] ({schema})  {start} → {end}  [[dim]{dataset}[/]]")
    path = upsert_symbol(
        dataset=dataset,
        symbol=symbol,
        schema=schema,
        start=start,
        end=end,
        output_dir=output_dir,
        stype_in=stype_in,
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
    output_dir: Path,
    yes: bool,
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

    written = upsert_ticks(
        dataset=dataset,
        symbol=symbol,
        schema=schema,
        start=start,
        end=end,
        output_dir=output_dir,
        stype_in=stype_in,
    )

    remaining = len(estimate.missing_days) - len(written)
    console.print(
        f"\n[bold green]Done.[/]  {len(written)} day(s) written to "
        f"{output_dir / symbol.replace('/', '_') / schema}"
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
            output_dir=out_dir,
            yes=yes,
        )
    else:
        _ingest_bar(
            dataset=dataset,
            symbol=symbol,
            schema=resolved,
            start=start,
            end=end,
            stype_in=stype_in,
            output_dir=out_dir,
            yes=yes,
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
        f"alpaca, {cfg.alpaca_options.rate_limit_per_min}/min" if is_free_provider else cfg.dataset
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
    days = sorted(covered_days(symbol, resolved, out_dir))

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
