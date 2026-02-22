"""CLI commands for research utilities."""

from __future__ import annotations

import subprocess
import sys

import typer

app = typer.Typer(help="Research utilities.")


@app.command()
def notebook(name: str = typer.Argument("scratch")) -> None:
    """Launch a Jupyter notebook for the given research name."""
    subprocess.run([sys.executable, "-m", "jupyter", "lab", f"research/{name}.ipynb"])  # noqa: S603
