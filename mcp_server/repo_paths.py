from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

PROJECT_DIR = REPO_ROOT / "project"
DOCS_DIR = REPO_ROOT / "docs"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"


def repo_relative(path: Path) -> str:
    """Return a repo-relative path when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def require_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path
