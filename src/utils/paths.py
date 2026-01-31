"""
Centralized, read-only path definitions for the project.

This module defines the canonical directory structure of the repository.
All scripts should import paths from here instead of hard-coding filesystem paths.

Design goals:
- Stable directory API (structure is frozen)
- Minimal logic: small helpers allowed (read latest run_id)
- No side effects at import time (no filesystem I/O on import)

Allowed:
- Add new path constants
- Add small helpers that return Path objects or parse identifiers

Do NOT:
- Rename existing attributes lightly (treat as public API)
- Perform heavy I/O or write operations in this module
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

# This file lives at: <repo>/src/utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------
# Top-level frozen directories (stable API)
# ---------------------------------------------------------------------

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
PROJECT_DIR = PROJECT_ROOT / "project"
DOCS_DIR = PROJECT_ROOT / "docs"


# ---------------------------------------------------------------------
# Data subdirectories (adjust/extend as needed, but keep roots stable)
# ---------------------------------------------------------------------

DATA_RAW_DIR = DATA_DIR / "raw"
DATA_HTML_RAW_DIR = DATA_DIR / "html_raw"
DATA_PDF_RAW_DIR = DATA_DIR / "pdf_raw"

DATA_CLEANED_DIR = DATA_DIR / "cleaned"
DATA_CLEANED_INDEX_DIR = DATA_CLEANED_DIR / "index"
DATA_CLEANED_SAMPLES_DIR = DATA_CLEANED_DIR / "samples"

# Large, regeneratable outputs (often ignored by git)
DATA_CLEANED_CONTENT_DIR = DATA_CLEANED_DIR / "content"
DATA_CLEANED_DEBUG_DIR = DATA_CLEANED_DIR / "debug"

DATA_LABELS_DIR = DATA_DIR / "labels"

# Optional exports (non-authoritative)
DATA_RESULTS_DIR = DATA_DIR / "results"


# ---------------------------------------------------------------------
# Runs: templates vs concrete run artifacts
# ---------------------------------------------------------------------

# Pointer to the latest run_id (single line)
RUNS_LATEST_FILE = RUNS_DIR / "latest.txt"

# Repository-level templates (blank templates, tracked in git)
RUN_TEMPLATE_MD = RUNS_DIR / "RUN_TEMPLATE.md"
BASELINE_CHECKLIST_MD = RUNS_DIR / "BASELINE_RUN_CHECKLIST.md"
RUN_META_TEMPLATE_JSON = RUNS_DIR / "meta_template.json"


# ---------------------------------------------------------------------
# Concrete run paths
# ---------------------------------------------------------------------

def run_dir(run_id: str) -> Path:
    """Directory for a given run_id, e.g., runs/run_YYYYMMDD_HHMM_xxx."""
    run_id = run_id.strip()
    if not run_id:
        raise ValueError("run_id is empty")
    return RUNS_DIR / run_id


def run_run_md(run_id: str) -> Path:
    """Path to runs/<run_id>/RUN.md."""
    return run_dir(run_id) / "RUN.md"


def run_meta_json(run_id: str) -> Path:
    """Path to runs/<run_id>/meta.json."""
    return run_dir(run_id) / "meta.json"


def read_latest_run_id(latest_file: Path = RUNS_LATEST_FILE) -> str:
    """
    Read latest run_id from runs/latest.txt.
    Rules:
    - Use the first non-empty, non-comment line
    - Comment lines start with '#'
    """
    if not latest_file.exists():
        raise FileNotFoundError(f"latest file not found: {latest_file}")

    text = latest_file.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s

    raise ValueError(f"No run_id found in: {latest_file}")


def latest_run_dir() -> Path:
    """Directory of the latest run (resolved via runs/latest.txt)."""
    return run_dir(read_latest_run_id())


def latest_run_md() -> Path:
    """Path to RUN.md of the latest run."""
    return run_run_md(read_latest_run_id())


def latest_meta_json() -> Path:
    """Path to meta.json of the latest run."""
    return run_meta_json(read_latest_run_id())
# ---------------------------------------------------------------------
# External data roots (user/machine specific)
# ---------------------------------------------------------------------

def zotero_storage_dir() -> Path:
    """
    Zotero storage root directory.

    This is an external, user-specific path and is intentionally
    resolved via environment variable instead of hard-coded.
    """
    import os
    v = os.getenv("ZOTERO_STORAGE_DIR", "").strip()
    if not v:
        raise RuntimeError(
            "ZOTERO_STORAGE_DIR is not set. "
            "Please set it to your Zotero storage root."
        )
    return Path(v)
