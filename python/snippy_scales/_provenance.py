"""Which code produced a result: the git revision of the working tree."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_revision(root: Path = REPO_ROOT) -> dict[str, str | bool | None]:
    """The commit, branch and cleanliness of the checkout at *root*.

    Returns:
        ``commit`` (full hash), ``branch`` and ``dirty`` (uncommitted changes to
        tracked files, so the commit alone does not reproduce the result). Each is
        ``None`` if git or the repository is unavailable.
    """
    commit = _git("rev-parse", "HEAD", cwd=root)
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=root)
    return {
        "commit": commit or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root) or None,
        "dirty": None if status is None else bool(status),
    }


def commit_before(when: str, root: Path = REPO_ROOT) -> str | None:
    """The last commit on any branch made at or before *when* (ISO timestamp).

    A guess at the code behind a result that did not record its commit.
    """
    return _git("log", "--all", "-1", "--format=%H", f"--before={when}", cwd=root) or None
