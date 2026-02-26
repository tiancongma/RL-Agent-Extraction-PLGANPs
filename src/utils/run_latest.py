#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.utils import paths
from src.utils.run_id import get_git_short_hash, is_valid_run_id


def _subset_stage_from_run_id(run_id: str) -> tuple[str, str]:
    parts = str(run_id).split("_")
    if len(parts) < 6:
        return ("", "")
    subset = parts[4]
    stage = "_".join(parts[5:]) if len(parts) > 5 else ""
    return (subset, stage)


def inputs_fingerprint(input_paths: Iterable[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted([Path(x) for x in input_paths], key=lambda x: str(x).lower()):
        rp = p.resolve() if p.exists() else p
        h.update(str(rp).encode("utf-8", errors="replace"))
        if p.exists() and p.is_file():
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        else:
            h.update(b"::missing_or_nonfile::")
    return h.hexdigest()[:12]


def read_latest(latest_path: Path | None = None) -> tuple[str, dict[str, str]]:
    p = latest_path or paths.RUNS_LATEST_FILE
    if not p.exists():
        return ("", {})
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return ("", {})
    rid = lines[0].strip()
    meta: dict[str, str] = {}
    for ln in lines[1:]:
        s = ln.strip()
        if not s.startswith("# "):
            continue
        payload = s[2:]
        if "=" not in payload:
            continue
        k, v = payload.split("=", 1)
        meta[k.strip()] = v.strip()
    return (rid, meta)


def write_latest(run_id: str, meta: dict, project_root: Path | None = None) -> Path:
    rid = str(run_id or "").strip()
    if not is_valid_run_id(rid):
        raise ValueError(f"Invalid run_id for latest pointer: {rid!r}")

    target = paths.RUNS_LATEST_FILE if project_root is None else (Path(project_root) / "runs" / "latest.txt")
    target.parent.mkdir(parents=True, exist_ok=True)

    merged = dict(meta or {})
    subset_default, stage_default = _subset_stage_from_run_id(rid)
    merged.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    merged.setdefault("git_short", get_git_short_hash(paths.PROJECT_ROOT))
    merged.setdefault("subset", subset_default)
    merged.setdefault("stage", stage_default)
    merged.setdefault("inputs_fingerprint", "")

    required = ["created_at", "git_short", "subset", "stage", "inputs_fingerprint"]
    missing = [k for k in required if not str(merged.get(k, "")).strip()]
    if missing:
        raise ValueError(f"Missing required latest metadata keys: {missing}")

    ordered_keys = ["created_at", "git_short", "subset", "stage", "inputs_fingerprint", "note"]
    lines = [rid]
    for k in ordered_keys:
        if k in merged and str(merged[k]).strip():
            lines.append(f"# {k}={merged[k]}")
    for k in sorted(merged.keys()):
        if k in ordered_keys:
            continue
        v = str(merged[k]).strip()
        if v:
            lines.append(f"# {k}={v}")

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
