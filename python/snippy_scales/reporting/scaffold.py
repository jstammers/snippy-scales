"""Create report instances from the tracked template and export them to HTML.

``reports/templates/strategy_report.py`` (tracked) is the marimo notebook every
report runs. :func:`new_report` copies it into ``reports/<name>/`` with the
report's ``config.yaml`` and a ``notes.md`` for the analyst's conclusions;
:func:`export_html` executes it headlessly into ``report.html``. Everything
under ``reports/<name>/`` is git-ignored: results and conclusions are
proprietary, the template and the theory write-ups are not.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from snippy_scales.reporting.config import REPO_ROOT, ReportConfig, load_report_config
from snippy_scales.reporting.pipeline import REPORTS_DIR, ReportPaths

if TYPE_CHECKING:
    from pathlib import Path

#: The tracked marimo template every report instance is copied from.
TEMPLATE = REPO_ROOT / "reports" / "templates" / "strategy_report.py"

_NOTES = """\
<!--
Analyst notes for this report. Rendered verbatim in the "Conclusions" section,
below the automated checks. Not tracked by git (reports/ is proprietary).
-->

**Verdict:** _pending_ — fill in once the report has been built and reviewed.

**What the evidence says**

- …

**Caveats specific to this run**

- …

**Next steps**

- …
"""


def new_report(
    name: str,
    source: Path,
    *,
    reports_dir: Path = REPORTS_DIR,
    force: bool = False,
) -> ReportPaths:
    """Create ``reports/<name>/`` from the template and the config at *source*.

    The config's ``name`` is set to *name*. Existing ``notes.md`` is kept.

    Raises:
        FileExistsError: If the report exists and *force* is false.
        FileNotFoundError: If the template is missing.
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"report template not found at {TEMPLATE}")
    cfg = load_report_config(source)
    cfg = ReportConfig.model_validate({**cfg.model_dump(by_alias=True), "name": name})
    paths = ReportPaths.named(name, reports_dir)
    if paths.notebook.exists() and not force:
        raise FileExistsError(f"{paths.root} already exists (use --force to overwrite)")
    paths.root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, paths.notebook)
    paths.config.write_text(cfg.to_yaml())
    if not paths.notes.exists():
        paths.notes.write_text(_NOTES)
    return paths


def refresh_template(paths: ReportPaths) -> None:
    """Replace a report's notebook with the current template (config and notes kept)."""
    shutil.copyfile(TEMPLATE, paths.notebook)


def export_html(paths: ReportPaths, *, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run the report notebook headlessly and write ``report.html`` next to it.

    Raises:
        subprocess.CalledProcessError: If marimo fails (a cell raised).
    """
    cmd = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        str(paths.notebook),
        "-o",
        str(paths.html),
        "--force",
        "--no-include-code",
    ]
    return subprocess.run(cmd, check=True, timeout=timeout)  # noqa: S603


def edit(paths: ReportPaths) -> None:
    """Open the report notebook in marimo's editor (blocks until it is closed)."""
    subprocess.run([sys.executable, "-m", "marimo", "edit", str(paths.notebook)], check=False)  # noqa: S603


def list_reports(reports_dir: Path = REPORTS_DIR) -> list[ReportPaths]:
    """Report instances (directories with a ``config.yaml``), by name."""
    if not reports_dir.exists():
        return []
    return [
        ReportPaths(p)
        for p in sorted(reports_dir.iterdir())
        if p.is_dir() and p.name != "templates" and (p / "config.yaml").exists()
    ]
