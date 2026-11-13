"""CLI for strategy research reports (marimo notebooks in ``reports/``).

Usage examples::

    # Create reports/tsmom-sp500/ from an example config
    algo report new tsmom-sp500 --from reports/templates/configs/tsmom_sp500.yaml

    # Execute it headlessly → reports/tsmom-sp500/report.html (+ tearsheet.html)
    algo report build tsmom-sp500

    # Open it in marimo's editor; list reports
    algo report edit tsmom-sp500
    algo report list
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.table import Table

from snippy_scales.cli._common import console

app = typer.Typer(help="Strategy research reports: marimo notebooks built from a template.")

_REPORTS = typer.Option(Path("reports"), "--reports-dir", help="Reports root")  # noqa: B008


def _paths(name: str, reports_dir: Path):  # noqa: ANN202
    from snippy_scales.reporting import ReportPaths

    paths = ReportPaths.named(name, reports_dir)
    if not paths.config.exists():
        console.print(
            f"[red]No report {name!r}[/] ({paths.config} missing). See `algo report list`."
        )
        raise typer.Exit(1)
    return paths


@app.command()
def new(
    name: str = typer.Argument(..., help="Report name (directory under reports/)"),
    source: Path = typer.Option(..., "--from", "-f", help="Config YAML to start from"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite an existing report's notebook"),
    reports_dir: Path = _REPORTS,
) -> None:
    """Create a report from the tracked template and a config."""
    from pydantic import ValidationError

    from snippy_scales.reporting import new_report

    try:
        paths = new_report(name, source, reports_dir=reports_dir, force=force)
    except (FileExistsError, FileNotFoundError, ValueError, ValidationError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Created[/] {paths.root}/  (config.yaml, report.py, notes.md)")
    console.print(f"Edit {paths.config}, then run [cyan]algo report build {name}[/]")


@app.command()
def build(
    name: str = typer.Argument(..., help="Report name"),
    refresh: bool = typer.Option(
        False, "--refresh-template", help="Replace the report's notebook with the current template"
    ),
    reports_dir: Path = _REPORTS,
) -> None:
    """Execute a report and export it to report.html (resumes its Optuna studies)."""
    from snippy_scales.reporting import export_html
    from snippy_scales.reporting.scaffold import refresh_template

    paths = _paths(name, reports_dir)
    if refresh:
        refresh_template(paths)
    console.print(f"[cyan]Building[/] {paths.notebook} …")
    try:
        export_html(paths)
    except subprocess.CalledProcessError as exc:
        console.print("[red]The notebook failed;[/] open it with `algo report edit` to debug.")
        raise typer.Exit(1) from exc
    console.print(f"[green]Report:[/] {paths.html}")
    if paths.tearsheet.exists():
        console.print(f"[green]Tearsheet:[/] {paths.tearsheet}")


@app.command()
def edit(name: str = typer.Argument(..., help="Report name"), reports_dir: Path = _REPORTS) -> None:
    """Open a report in marimo's interactive editor."""
    from snippy_scales.reporting.scaffold import edit as edit_notebook

    edit_notebook(_paths(name, reports_dir))


@app.command(name="list")
def list_(reports_dir: Path = _REPORTS) -> None:
    """List reports and whether they have been built."""
    from snippy_scales.reporting import list_reports, load_report_config

    reports = list_reports(reports_dir)
    if not reports:
        console.print("[yellow]No reports yet.[/]  Start one with [cyan]algo report new[/].")
        return
    table = Table(header_style="bold cyan")
    for column in ("report", "strategy", "title", "built"):
        table.add_column(column)
    for paths in reports:
        try:
            cfg = load_report_config(paths.config)
            strategy, title = cfg.strategy, cfg.title
        except Exception as exc:  # noqa: BLE001 — list every report, even broken ones
            strategy, title = "?", f"[red]invalid config: {exc.__class__.__name__}[/]"
        table.add_row(paths.root.name, strategy, title, _mtime(paths.html))
    console.print(table)


def _mtime(path: Path) -> str:
    import datetime as dt

    if not path.exists():
        return "—"
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")  # noqa: DTZ006
