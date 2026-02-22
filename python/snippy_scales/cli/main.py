"""Main CLI entry point — `algo <command>`."""

from __future__ import annotations

import typer
from rich.console import Console

from snippy_scales.cli import backtest, data, research

app = typer.Typer(name="algo", help="SnippyScales platform CLI.", rich_markup_mode="rich")
console = Console()

app.add_typer(backtest.app, name="backtest")
app.add_typer(data.app, name="data")
app.add_typer(research.app, name="research")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print("[bold green]SnippyScales[/] — use [cyan]algo --help[/] for commands.")


if __name__ == "__main__":
    app()
