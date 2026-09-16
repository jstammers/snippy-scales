#!/usr/bin/env python
"""Backfill 1-minute bars from Alpaca for every stock ever in the S&P 500.

Pulls ``--years`` (default 5) of ``ohlcv-1m`` bars for every ticker that was
an S&P 500 constituent at any point in that window — not just today's 500
names — so the resulting dataset doesn't have survivorship bias baked in.
Respects the Alpaca free-tier rate limit via
:class:`snippy_scales.data.ratelimit.RateLimiter` (shared across all
symbols/threads) and resumes automatically if interrupted: re-running the
same command only fetches what's still missing (see
:func:`snippy_scales.data.ingest.upsert_bars`).

Requires ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` in the environment.

Examples::

    # Preview the plan (universe size, estimated requests/runtime) — no download.
    uv run python scripts/pull_sp500_alpaca_1m.py --dry-run

    # Run it (prompts for confirmation; ~5h for the full universe on the free tier).
    uv run python scripts/pull_sp500_alpaca_1m.py

    # Re-run, only retrying symbols that failed last time.
    uv run python scripts/pull_sp500_alpaca_1m.py --retry-failed

All estimation, symbols-cache, and manifest logic lives in
:mod:`snippy_scales.data.backfill` (and is unit-tested there) — this script
is just argument parsing and orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_MANIFEST = Path("data/raw/_manifests/sp500_1m.csv")
DEFAULT_SYMBOLS_CACHE = Path("data/universe/sp500_ever_members.txt")


@app.command()
def main(
    years: Annotated[int, typer.Option(help="How many years back from --end to backfill.")] = 5,
    end: Annotated[str | None, typer.Option(help="End date YYYY-MM-DD. Defaults to today.")] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Root directory for raw Parquet files.")
    ] = DEFAULT_OUTPUT_DIR,
    manifest_path: Annotated[
        Path, typer.Option(help="CSV manifest of per-symbol ingestion results.")
    ] = DEFAULT_MANIFEST,
    symbols_cache: Annotated[
        Path,
        typer.Option(help="Where the resolved S&P 500 universe is cached for reproducibility."),
    ] = DEFAULT_SYMBOLS_CACHE,
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
    """Backfill 1-minute S&P 500 bars from Alpaca. See module docstring for details."""
    from snippy_scales.data.backfill import (
        estimate_backfill_plan,
        load_symbols_cache,
        merge_manifest,
        read_manifest,
        resolve_backfill_window,
        save_symbols_cache,
    )
    from snippy_scales.data.config import AlpacaConfig, AssetClassConfig, IngestConfig
    from snippy_scales.data.ingest import ingest_from_config
    from snippy_scales.data.universe import sp500_ever_members

    start_date, end_date = resolve_backfill_window(years, end)

    # --- Resolve the universe ---
    if retry_failed:
        manifest = read_manifest(manifest_path)
        if not manifest:
            console.print(f"[bold red]No manifest found at {manifest_path} — nothing to retry.[/]")
            raise typer.Exit(1)
        symbols = sorted(row["symbol"] for row in manifest.values() if row["succeeded"] != "True")
        if not symbols:
            console.print("[green]Nothing to retry — every symbol in the manifest succeeded.[/]")
            raise typer.Exit(0)
        console.print(f"[cyan]Retrying {len(symbols)} previously-failed symbol(s).[/]")
    else:
        cached = None if refresh_universe else load_symbols_cache(symbols_cache)
        if cached is not None:
            console.print(
                f"[dim]Using cached universe ({len(cached)} tickers) from {symbols_cache}[/]"
            )
            symbols = cached
        else:
            console.print(
                f"Resolving S&P 500 ever-members for {start_date} → {end_date} from Wikipedia …"
            )
            symbols = sp500_ever_members(start_date, end_date)
            save_symbols_cache(symbols_cache, symbols)
            console.print(
                f"[green]Resolved {len(symbols)} ticker(s)[/] — cached to {symbols_cache}"
            )

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
    table.add_row("Output", str(output_dir))
    table.add_row("Manifest", str(manifest_path))
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
    rows = ingest_from_config(cfg, output_dir=output_dir)
    manifest = merge_manifest(manifest_path, rows)

    succeeded = sum(1 for row in rows if row.succeeded)
    failed = len(rows) - succeeded
    console.print(
        f"\n[bold green]Done.[/]  {succeeded}/{len(rows)} symbol(s) succeeded this run"
        + (f", [bold red]{failed} failed[/]" if failed else ".")
    )
    console.print(f"[dim]Manifest ({len(manifest)} symbol(s) total):[/] {manifest_path}")
    if failed:
        console.print(
            "[yellow]Re-run with[/] [bold]--retry-failed[/] [yellow]to retry just the failures.[/]"
        )


if __name__ == "__main__":
    app()
