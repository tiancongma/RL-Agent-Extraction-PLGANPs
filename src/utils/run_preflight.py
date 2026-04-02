#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.utils import paths
from src.utils.model_policy import validate_models_or_raise
from src.utils.run_id import (
    classify_results_path,
    is_valid_legacy_run_id,
    resolve_results_write_target,
)
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
    p.add_argument("--subset", default="")
    p.add_argument("--stage", default="")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--input-path", action="append", default=[], help="Repeatable input path for fingerprinting.")
    p.add_argument("--model", action="append", default=[], help="Optional model name; repeatable.")
    p.add_argument("--models", default="", help="Optional comma-separated model names.")
    p.add_argument("--run-id", default="")
    p.add_argument(
        "--run-dir",
        default="",
        help="Explicit governed results directory to validate and use as-is. Existing legacy roots are reusable only; new writes must use v2 bucket/child paths.",
    )
    p.add_argument(
        "--naming-mode",
        choices=["v2", "legacy"],
        default="v2",
        help="Default write-target naming mode. v2 is the future-facing default; legacy is retained only for compatibility reporting and is rerouted to v2 for new writes.",
    )
    p.add_argument(
        "--cue",
        default="",
        help="Optional v2 child cue. Defaults to --stage when omitted, otherwise 'run'.",
    )
    p.add_argument("--note", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models: list[str] = []
    models.extend([str(m).strip() for m in args.model if str(m).strip()])
    if str(args.models).strip():
        models.extend([m.strip() for m in str(args.models).split(",") if m.strip()])
    if models:
        validate_models_or_raise(models, context="run_preflight model check")

    input_paths = [Path(x) for x in args.input_path]
    fp = inputs_fingerprint(input_paths)

    latest_path = paths.RUNS_LATEST_FILE
    latest_rid, latest_meta = read_latest(latest_path)

    chosen = ""
    chosen_ref = ""
    chosen_kind = ""
    bucket_ref = ""
    reuse_reason = ""
    new_reason = ""

    explicit_run_dir = str(args.run_dir).strip()
    if explicit_run_dir:
        run_dir = Path(explicit_run_dir).resolve()
        classification = classify_results_path(run_dir, results_dir=paths.DATA_RESULTS_DIR)
        if classification["path_kind"] not in {"legacy_run_root", "v2_child_execution"}:
            raise ValueError(f"Explicit --run-dir is not a supported results path: {run_dir}")
        if classification["path_kind"] == "legacy_run_root" and not run_dir.exists():
            raise ValueError(
                "Creating new legacy run roots is disabled. Use a v2 bucket/child path instead."
            )
        chosen = run_dir.name
        chosen_ref = str(run_dir)
        chosen_kind = classification["path_kind"]
        bucket_ref = str(run_dir.parent if classification["path_kind"] == "v2_child_execution" else run_dir)
        reuse_reason = "explicit_run_dir"
    else:
        chosen_kind = "v2_child_execution"

    explicit = str(args.run_id).strip()
    if not chosen:
        if explicit:
            if not is_valid_legacy_run_id(explicit):
                raise ValueError(f"Invalid --run-id (must match required legacy regex): {explicit}")
            chosen_ref_path = (paths.DATA_RESULTS_DIR / explicit).resolve()
            if not chosen_ref_path.exists():
                raise ValueError(
                    "Creating new legacy run roots is disabled. Use --run-dir for an existing "
                    "historical directory or let the v2 allocator create a compliant root."
                )
            chosen = explicit
            chosen_ref = str(chosen_ref_path)
            reuse_reason = "explicit_run_id_reuse_only"
            bucket_ref = chosen_ref
        else:
            cue = str(args.cue).strip() or str(args.stage).strip() or "run"
            target = resolve_results_write_target(
                results_root=paths.DATA_RESULTS_DIR,
                default_child_cue=cue,
            )
            chosen = target["run_basename"]
            chosen_ref = target["run_dir"]
            chosen_kind = target["path_kind"]
            bucket_ref = target["bucket_dir"]
            new_reason = target["selection_mode"]
            if args.naming_mode == "legacy":
                new_reason = "legacy_mode_rerouted_to_v2"

    latest_written = None
    if chosen_kind == "legacy_run_root":
        latest_meta = {
            "inputs_fingerprint": fp,
            "note": str(args.note).strip(),
        }
        if str(args.subset).strip():
            latest_meta["subset"] = str(args.subset).strip()
        if str(args.stage).strip():
            latest_meta["stage"] = str(args.stage).strip()
        latest_written = write_latest(
            run_id=chosen,
            meta=latest_meta,
        )

    preview: list[str] = []
    if latest_written is not None:
        preview = latest_written.read_text(encoding="utf-8", errors="replace").splitlines()[:6]
    print(f"chosen_run_id={chosen}")
    print(f"chosen_run_ref={chosen_ref}")
    print(f"chosen_path_kind={chosen_kind}")
    print(f"chosen_bucket_ref={bucket_ref}")
    print(f"legacy_regex_match={is_valid_legacy_run_id(chosen)}")
    print(f"reuse_reason={reuse_reason}")
    print(f"new_reason={new_reason}")
    print(f"inputs_fingerprint={fp}")
    print(f"latest_path={latest_written or ''}")
    if latest_written is not None:
        print("latest_preview_first6=")
        for ln in preview:
            print(ln)
    else:
        print("latest_preview_first6=")
        print("# skipped_for_non_legacy_explicit_run_dir")


if __name__ == "__main__":
    main()
