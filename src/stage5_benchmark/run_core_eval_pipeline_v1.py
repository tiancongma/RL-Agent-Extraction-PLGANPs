#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run core_eval_v1 pipeline: core projection -> core alignment -> core membership analysis."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    out_dir = Path(f"data/results/{run_id}/benchmark_goren_2025/core_eval_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmds = [
        [sys.executable, "src/stage5_benchmark/run_projection_core_to_curated_v1.py", "--run-id", run_id],
        [sys.executable, "src/stage5_benchmark/run_alignment_eval_core_v1.py", "--run-id", run_id],
        [sys.executable, "archive/code/benchmark_specific_audit_report/analyze_row_membership_core_v1.py", "--run-id", run_id],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True)

    metrics_path = out_dir / "core_metrics_summary.json"
    membership_path = out_dir / "core_row_membership_summary.json"
    if not metrics_path.exists() or not membership_path.exists():
        raise FileNotFoundError("Expected core eval outputs were not produced.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    memb = json.loads(membership_path.read_text(encoding="utf-8"))

    mode_metrics = metrics.get("metrics_by_mode", {})
    print(f"run_id={run_id}")
    print("per_mode_metrics=" + json.dumps({m: {"recall": v.get("recall"), "precision": v.get("precision")} for m, v in mode_metrics.items()}, ensure_ascii=False))
    print(
        "doi_multiplicity_flags="
        + json.dumps(
            {
                "core_gt_curated_doi_count": memb.get("core_gt_curated_doi_count"),
                "core_lt_curated_doi_count": memb.get("core_lt_curated_doi_count"),
                "equal_doi_count": memb.get("equal_doi_count"),
            },
            ensure_ascii=False,
        )
    )
    print("top10_abs_difference_dois=" + json.dumps(memb.get("top_abs_difference_dois", [])[:10], ensure_ascii=False))
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()


