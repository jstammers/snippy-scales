"""CLI command to launch the Streamlit trading dashboard.

Usage::

    algo dashboard launch
    algo dashboard launch --db data/analytics.duckdb --port 8501
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Launch the trading research dashboard.")
console = Console()


@app.command()
def launch(
    db: Path = typer.Option(  # noqa: B008
        Path("data/analytics.duckdb"), "--db", help="DuckDB database path"
    ),
    port: int = typer.Option(8501, "--port", help="Streamlit server port"),
    host: str = typer.Option("localhost", "--host", help="Streamlit server host"),
) -> None:
    """Start the Streamlit trading dashboard."""
    app_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    if not app_path.exists():
        console.print(f"[red]Dashboard app not found at {app_path}[/]")
        raise typer.Exit(1)

    env = os.environ.copy()
    env["SNIPPY_DB_PATH"] = str(db)

    console.print(f"[cyan]Starting dashboard:[/] {host}:{port}")
    console.print(f"[cyan]Database:[/] {db}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
            "--server.address",
            host,
            "--browser.gatherUsageStats",
            "false",
        ],
        env=env,
        check=False,
    )
