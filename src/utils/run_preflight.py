#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.utils import paths
from src.utils.run_id import build_run_id, is_valid_run_id
from src.utils.run_latest import inputs_fingerprint, read_latest, write_latest


def _same_local_date(created_at_iso: str) -> bool:
    if not str(created_at_iso).strip():
        return False
    try:
        dt = datetime.fromisoformat(str(created_at_iso).strip())
    except ValueError:
        return False
    local_dt = dt.astimezone()
    return local_dt.date() == datetime.now().astimezone().date()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic run preflight with reuse/new policy.")
    p.add_argument("--subset", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--input-path", action="append", default=[], help="Repeatable input path for fingerprinting.")
    p.add_argument("--run-id", default="")
    p.add_argument("--note", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(x) for x in args.input_path]
    fp = inputs_fingerprint(input_paths)

    latest_path = paths.RUNS_LATEST_FILE
    latest_rid, latest_meta = read_latest(latest_path)

    chosen = ""
    reuse_reason = ""
    new_reason = ""

    explicit = str(args.run_id).strip()
    if explicit:
        if not is_valid_run_id(explicit):
            raise ValueError(f"Invalid --run-id (must match required regex): {explicit}")
        chosen = explicit
        reuse_reason = "explicit_run_id"
    else:
        can_reuse = (
            bool(latest_rid)
            and is_valid_run_id(latest_rid)
            and _same_local_date(latest_meta.get("created_at", ""))
            and latest_meta.get("inputs_fingerprint", "") == fp
        )
        if can_reuse:
            chosen = latest_rid
            reuse_reason = "latest_same_local_date_and_same_inputs_fingerprint"
        else:
            chosen = build_run_id(subset=args.subset, stage=args.stage, version=int(args.version))
            if not latest_rid:
                new_reason = "no_latest_found"
            elif not is_valid_run_id(latest_rid):
                new_reason = "latest_invalid_run_id"
            elif not _same_local_date(latest_meta.get("created_at", "")):
                new_reason = "latest_not_today_local"
            elif latest_meta.get("inputs_fingerprint", "") != fp:
                new_reason = "inputs_fingerprint_changed"
            else:
                new_reason = "generated_new_run_id"

    latest_written = write_latest(
        run_id=chosen,
        meta={
            "subset": str(args.subset).strip(),
            "stage": str(args.stage).strip(),
            "inputs_fingerprint": fp,
            "note": str(args.note).strip(),
        },
    )

    preview = latest_written.read_text(encoding="utf-8", errors="replace").splitlines()[:6]
    print(f"chosen_run_id={chosen}")
    print(f"regex_match={is_valid_run_id(chosen)}")
    print(f"reuse_reason={reuse_reason}")
    print(f"new_reason={new_reason}")
    print(f"inputs_fingerprint={fp}")
    print(f"latest_path={latest_written}")
    print("latest_preview_first6=")
    for ln in preview:
        print(ln)


if __name__ == "__main__":
    main()

