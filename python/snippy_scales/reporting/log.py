"""The strategy log: one markdown document of every report's result, in date order.

Built from what the reports already store: each report's Optuna studies (results,
parameters, the config and git commit that produced them), its ``config.yaml`` and
the verdict in its ``notes.md``. Nothing is typed by hand, so the log can be
regenerated at any time (``algo report log``) and shows how results evolve as
strategies, data and code change.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import optuna
import yaml

from snippy_scales._provenance import commit_before
from snippy_scales.evaluation import summarize_study
from snippy_scales.reporting.config import REPO_ROOT, load_report_config
from snippy_scales.reporting.pipeline import REPORTS_DIR, ReportPaths
from snippy_scales.reporting.scaffold import list_reports

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_COMPLETE = optuna.trial.TrialState.COMPLETE
_FUTURES = re.compile(r"\.(ROLL|FUT|c\.\d+|v\.\d+)$")
_SHORT = 10  # characters of a commit hash shown


@dataclass(frozen=True)
class LogEntry:
    """One report's headline result and where it came from.

    Attributes:
        report: Report name.
        root: The report's directory.
        title: Report title.
        strategy: Registry name of the strategy.
        question: The report's research question.
        date: When the reported study's last trial finished.
        asset_group: ``"futures"`` or ``"equities"``, by the symbols traded. Results
            are only compared within a group.
        n_symbols: Symbols in the universe.
        engine: ``"nautilus"`` if the result was validated on Nautilus, else the
            screening engine.
        oos_start: First day of the stitched out-of-sample returns.
        oos_end: Last day.
        n_trials: Trials the result was selected from (the DSR's N).
        best_params: Parameters of the reported trial.
        sharpe: Annualised out-of-sample Sharpe ratio.
        dsr: Deflated Sharpe ratio.
        pbo: Probability of backtest overfitting (``None`` with fewer than 2 trials).
        total_return: Compounded out-of-sample return, as a fraction.
        max_drawdown: Worst drawdown, ``<= 0``.
        fold_sharpes: Sharpe per walk-forward fold.
        commit: Git commit of the code that ran the reported trial.
        commit_inferred: The study did not record its commit; *commit* is the last
            commit made before the trial finished.
        dirty: The working tree had uncommitted changes (``None`` if unknown).
        fingerprint: Hash of the result-affecting config, as stored on the study.
        config: The report config as run (from the study, else ``config.yaml``).
        verdict: First paragraph of the analyst notes, if written.
        error: Why the result could not be read, if it could not.
    """

    report: str
    root: Path
    title: str = ""
    strategy: str = ""
    question: str = ""
    date: dt.datetime | None = None
    asset_group: str = ""
    n_symbols: int = 0
    engine: str = ""
    oos_start: dt.date | None = None
    oos_end: dt.date | None = None
    n_trials: int = 0
    best_params: dict[str, Any] = field(default_factory=dict)
    sharpe: float = float("nan")
    dsr: float = float("nan")
    pbo: float | None = None
    total_return: float = float("nan")
    max_drawdown: float = float("nan")
    fold_sharpes: tuple[float, ...] = ()
    commit: str | None = None
    commit_inferred: bool = False
    dirty: bool | None = None
    fingerprint: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    error: str | None = None


def _load(name: str, storage: str) -> optuna.Study | None:
    try:
        study = optuna.load_study(study_name=name, storage=storage)
    except KeyError:
        return None
    return study if study.get_trials(deepcopy=False, states=(_COMPLETE,)) else None


def verdict(notes: Path) -> str | None:
    """The first paragraph of a report's ``notes.md`` after its comment header."""
    if not notes.exists():
        return None
    text = re.sub(r"<!--.*?-->", "", notes.read_text(), flags=re.DOTALL).strip()
    first = re.sub(r"^>\s?", "", text.split("\n\n", 1)[0], flags=re.MULTILINE)
    return " ".join(first.split()) or None


def asset_group(symbols: Sequence[str]) -> str:
    """``"futures"`` if most symbols are futures series, else ``"equities"``."""
    futures = sum(bool(_FUTURES.search(s)) for s in symbols)
    return "futures" if futures * 2 > len(symbols) else "equities"


def entry(paths: ReportPaths) -> LogEntry:
    """Read one report's result. Never raises: a broken report gets an ``error``."""
    name = paths.root.name
    try:
        cfg = load_report_config(paths.config)
    except Exception as exc:  # noqa: BLE001 — log every report, even broken ones
        return LogEntry(name, paths.root, error=f"invalid config: {exc.__class__.__name__}")
    base = LogEntry(
        name,
        paths.root,
        title=cfg.title,
        strategy=cfg.strategy,
        question=" ".join(cfg.question.split()),
        config=yaml.safe_load(cfg.to_yaml()),
        verdict=verdict(paths.notes),
    )
    if not (paths.root / "optuna.db").exists():
        return _replace(base, error="not built")
    try:
        screen = _load(f"{name}-screen", paths.storage)
        valid = _load(f"{name}-nautilus", paths.storage)
        study = valid or screen
        if study is None:
            return _replace(base, error="no completed trials")
        report = summarize_study(study, selection=screen if valid is not None else None)
    except Exception as exc:  # noqa: BLE001 — e.g. trial returns moved or deleted
        return _replace(base, error=f"{exc.__class__.__name__}: {exc}")

    trials = study.get_trials(deepcopy=False, states=(_COMPLETE,))
    best = max(trials, key=lambda t: t.value if t.value is not None else float("-inf"))
    finished = max((t.datetime_complete for t in trials if t.datetime_complete), default=None)
    commit, dirty, inferred = best.user_attrs.get("commit"), best.user_attrs.get("dirty"), False
    if commit is None and finished is not None:
        commit, inferred = commit_before(finished.isoformat(), REPO_ROOT), True
    days = report.returns["ts"].dt.date()
    selected_from = screen if valid is not None and screen is not None else study
    return _replace(
        base,
        date=finished,
        asset_group=asset_group(report.symbols),
        n_symbols=len(report.symbols),
        engine=str(study.user_attrs.get("engine", "")),
        oos_start=days.min(),
        oos_end=days.max(),
        n_trials=len(selected_from.get_trials(deepcopy=False, states=(_COMPLETE,))),
        best_params=report.best_params,
        sharpe=report.sharpe,
        dsr=report.dsr,
        pbo=report.pbo,
        total_return=report.total_return,
        max_drawdown=report.max_drawdown,
        fold_sharpes=tuple(report.fold_sharpes),
        commit=commit,
        commit_inferred=inferred,
        dirty=dirty,
        fingerprint=study.user_attrs.get("fingerprint"),
        config=study.user_attrs.get("config") or base.config,
    )


def _replace(base: LogEntry, **changes: Any) -> LogEntry:  # noqa: ANN401 — any LogEntry field
    return LogEntry(**{**base.__dict__, **changes})


def collect(reports_dirs: Sequence[Path] = (REPORTS_DIR,)) -> list[LogEntry]:
    """Every report under *reports_dirs*, oldest result first (unbuilt ones last).

    A report name found in several directories is kept once, from the first
    directory that has it built.
    """
    seen: dict[str, LogEntry] = {}
    for directory in reports_dirs:
        for paths in list_reports(directory):
            found = entry(paths)
            current = seen.get(found.report)
            if current is None or (current.date is None and found.date is not None):
                seen[found.report] = found
    return sorted(
        seen.values(), key=lambda e: (e.date is None, e.date or dt.datetime.min, e.report)
    )


def documents(directories: Sequence[Path]) -> list[tuple[Path, str, dt.date]]:
    """Markdown research documents in *directories*: ``(path, title, modified)``."""
    out = []
    for directory in directories:
        for path in sorted(directory.glob("*.md")) if directory.is_dir() else []:
            if path.stem.isupper():  # README, PR_BODY, STRATEGY_LOG: not research
                continue
            heading = next(
                (
                    ln.lstrip("# ").strip()
                    for ln in path.read_text().splitlines()
                    if ln.startswith("#")
                ),
                path.stem,
            )
            out.append((path, heading, dt.date.fromtimestamp(path.stat().st_mtime)))  # noqa: DTZ012
    return sorted(out, key=lambda d: (d[2], d[0].name))


# ── Rendering ─────────────────────────────────────────────────────────────────


def _num(value: float | None, spec: str = ".2f") -> str:
    return "—" if value is None or value != value else format(value, spec)  # noqa: PLR0124 — NaN


def _commit(e: LogEntry) -> str:
    if e.commit is None:
        return "unknown"
    mark = " (inferred)" if e.commit_inferred else " +changes" if e.dirty else ""
    return f"`{e.commit[:_SHORT]}`{mark}"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _timeline(entries: Sequence[LogEntry]) -> list[str]:
    lines = [
        "| Date | Report | Strategy | Universe | Out of sample | Sharpe | DSR | PBO | "
        "Return | Max DD | Trials | Best so far | Commit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    best: dict[str, float] = {}
    for e in entries:
        if e.error is not None or e.date is None:
            continue
        previous = best.get(e.asset_group)
        record = previous is None or e.sharpe > previous
        if record:
            best[e.asset_group] = e.sharpe
        running = f"**{_num(e.sharpe)}** ▲" if record else _num(best[e.asset_group])
        lines.append(
            f"| {e.date:%Y-%m-%d} | [{e.report}](#{e.report}) | `{e.strategy}` | "
            f"{e.n_symbols} {e.asset_group} | {e.oos_start} → {e.oos_end} | {_num(e.sharpe)} | "
            f"{_num(e.dsr)} | {_num(e.pbo)} | {_num(e.total_return, '+.0%')} | "
            f"{_num(e.max_drawdown, '.0%')} | {e.n_trials} | {running} | {_commit(e)} |"
        )
    return lines


def _section(e: LogEntry) -> list[str]:
    lines = [f"### {e.report}", "", f"**{e.title}** (`{e.strategy}`)", ""]
    if e.question:
        lines += [f"*Question.* {e.question}", ""]
    if e.verdict:
        lines += [f"*Verdict.* {e.verdict}", ""]
    if e.error is not None:
        lines += [f"*No result:* {e.error}.", ""]
    else:
        params = ", ".join(f"`{k}={v}`" for k, v in e.best_params.items())
        folds = ", ".join(_num(f) for f in e.fold_sharpes)
        changes = {True: "yes", False: "no", None: "unknown"}[e.dirty]
        lines += [
            f"- Result ({e.date:%Y-%m-%d %H:%M}, {e.engine}): Sharpe {_num(e.sharpe)}, "
            f"DSR {_num(e.dsr)}, PBO {_num(e.pbo)}, return {_num(e.total_return, '+.0%')}, "
            f"max drawdown {_num(e.max_drawdown, '.0%')}",
            f"- Out of sample: {e.oos_start} → {e.oos_end}, {e.n_symbols} {e.asset_group}; "
            f"fold Sharpes {folds}",
            f"- Selected from {e.n_trials} trials: {params}",
            f"- Code: commit {_commit(e)}, uncommitted changes: {changes}; "
            f"config fingerprint `{e.fingerprint}`",
        ]
        if e.commit_inferred:
            lines.append(
                "  - The study predates commit recording: this is the last commit made "
                "before it finished, not a recorded one."
            )
        lines.append("")
    lines += [f"- Files: `{_relative(e.root)}/` (`report.html`, `notes.md`, `optuna.db`)", ""]
    if e.config:
        dumped = yaml.safe_dump(e.config, sort_keys=False).rstrip()
        lines += ["<details><summary>Config</summary>", "", "```yaml", dumped, "```", ""]
        lines += ["</details>", ""]
    return lines


def render(
    entries: Sequence[LogEntry],
    docs: Sequence[tuple[Path, str, dt.date]] = (),
    *,
    now: dt.datetime | None = None,
) -> str:
    """The strategy log as markdown."""
    stamp = (now or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d %H:%M UTC")
    built = [e for e in entries if e.error is None]
    lines = [
        "# Strategy log",
        "",
        f"Generated {stamp} by `algo report log` from {len(built)} built reports. "
        "Do not edit: regenerate it. Conclusions belong in each report's `notes.md`.",
        "",
        "## Timeline",
        "",
        "Each row is a report's validated out-of-sample result, oldest first. "
        "*Best so far* is the highest Sharpe to date among reports on the same asset "
        "group; ▲ marks a new best. Sharpe ratios on different universes and periods "
        "are not like for like, so read a new best as a prompt to compare the two "
        "reports, not as proof of progress.",
        "",
        *_timeline(entries),
        "",
        "## Reports",
        "",
    ]
    for e in entries:
        lines += _section(e)
    if docs:
        lines += ["## Research documents", "", "| Updated | Document | File |", "|---|---|---|"]
        lines += [f"| {day} | {title} | `{_relative(path)}` |" for path, title, day in docs]
        lines.append("")
    return "\n".join(lines)


def write_log(
    output: Path,
    reports_dirs: Sequence[Path] = (REPORTS_DIR,),
    doc_dirs: Sequence[Path] | None = None,
) -> list[LogEntry]:
    """Collect every report and write the strategy log to *output*."""
    entries = collect(reports_dirs)
    if doc_dirs is None:
        doc_dirs = [REPO_ROOT / "docs" / "research", *reports_dirs]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(entries, documents(doc_dirs)))
    return entries
