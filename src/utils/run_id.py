#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path


def sanitize_token(s: str) -> str:
    """Normalize a token for run_id usage."""
    x = str(s).strip().lower().replace(" ", "_")
    x = re.sub(r"[^a-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def get_git_short_hash(project_root: Path | None = None) -> str:
    """Return git short hash, or 'nogit' if unavailable."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        out = subprocess.check_output(
            cmd,
            cwd=str(project_root) if project_root else None,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out if out else "nogit"
    except Exception:
        return "nogit"


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

    g = sanitize_token(git_hash) if git_hash is not None else sanitize_token(get_git_short_hash())
    g = g or "nogit"

    subset_tok = sanitize_token(subset) or "subset"
    stage_tok = sanitize_token(stage) or "stage"
    ver = int(version)

    return f"run_{ts}_{g}_{subset_tok}_{stage_tok}_v{ver}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standardized run_id.")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    rid = build_run_id(subset=args.subset, stage=args.stage, version=args.version)

    parts = rid.split("_")
    assert rid.startswith("run_")
    assert len(parts) >= 7
    assert len(parts[1] + "_" + parts[2]) == 13
    assert len(parts[3]) >= 1

    print(rid)


if __name__ == "__main__":
    main()
