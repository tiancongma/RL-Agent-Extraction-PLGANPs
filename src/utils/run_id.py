#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

RUN_ID_REGEX = r"^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$"


def is_valid_run_id(s: str) -> bool:
    return bool(re.fullmatch(RUN_ID_REGEX, str(s or "").strip()))


def sanitize_token(s: str) -> str:
    """Normalize a token for run_id usage."""
    x = str(s).strip().lower().replace(" ", "_")
    x = re.sub(r"[^a-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def get_git_short_hash(project_root: Path | None = None) -> str:
    """Return git short hash (7 lowercase hex chars)."""
    try:
        cmd = ["git", "rev-parse", "--short=7", "HEAD"]
        out = subprocess.check_output(
            cmd,
            cwd=str(project_root) if project_root else None,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip().lower()
        if re.fullmatch(r"[0-9a-f]{7}", out):
            return out
    except Exception:
        pass
    raise RuntimeError("Failed to resolve git short hash (expected 7 hex chars).")


def build_run_id(
    subset: str,
    stage: str,
    version: int = 1,
    dt: datetime | None = None,
    git_hash: str | None = None,
) -> str:
    """Build run_id as: run_<YYYYMMDD>_<HHMM>_<git_short_hash>_<subset>_<stage>_v<version>."""
    now = dt or datetime.now()
    ts = now.strftime("%Y%m%d_%H%M")

    g = (str(git_hash).strip().lower() if git_hash is not None else get_git_short_hash()).strip()
    if not re.fullmatch(r"[0-9a-f]{7}", g):
        raise ValueError(f"Invalid git short hash for run_id: {g!r}")

    subset_tok = sanitize_token(subset) or "subset"
    stage_tok = sanitize_token(stage) or "stage"
    ver = int(version)

    rid = f"run_{ts}_{g}_{subset_tok}_{stage_tok}_v{ver}"
    if not is_valid_run_id(rid):
        raise ValueError(f"Generated invalid run_id: {rid}")
    return rid


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standardized run_id.")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--note", default="", help="Optional note for caller metadata.")
    args = parser.parse_args()

    rid = build_run_id(subset=args.subset, stage=args.stage, version=args.version)
    if not is_valid_run_id(rid):
        raise SystemExit(f"Generated run_id does not match required regex: {RUN_ID_REGEX}")

    print(rid)


if __name__ == "__main__":
    main()
