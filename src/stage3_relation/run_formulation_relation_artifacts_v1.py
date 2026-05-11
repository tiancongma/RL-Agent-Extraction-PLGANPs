#!/usr/bin/env python3
from __future__ import annotations

"""
Run the deterministic Stage 3 relation builder in a run-scoped results location.

This wrapper exists to:
- enforce explicit `run_id` usage,
- keep outputs under `data/results/<run_id>/<out_subdir>/`,
- write a reproducibility-grade `RUN_CONTEXT.md` for the Stage 3 relation layer.

It is a stage-local helper and not a hidden end-to-end pipeline runner.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.stage3_relation.build_formulation_relation_artifacts_v1 import (
    build_relation_artifacts,
)
from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
from src.utils.run_id import resolve_results_write_target, validate_artifact_subdir


VALID_RUN_TYPES = {
    "intermediate_diagnostic_run",
    "component_regression_run",
    "full_pipeline_benchmark_run",
}
def validate_out_subdir(out_subdir: str) -> str:
    return validate_artifact_subdir(out_subdir, param_name="out_subdir")


def render_run_context(
    *,
    run_id: str,
    run_type: str,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    out_subdir: str,
    weak_labels_tsv: Path,
    weak_labels_jsonl: Path | None,
    scope_manifest_tsv: Path | None,
    out_dir: Path,
    stats: dict[str, object],
) -> str:
    jsonl_line = (
        f"- weak_labels_jsonl: `{weak_labels_jsonl}`"
        if weak_labels_jsonl is not None
        else "- weak_labels_jsonl: `not provided`"
    )
    manifest_line = (
        f"- scope_manifest_tsv: `{scope_manifest_tsv}`"
        if scope_manifest_tsv is not None
        else "- scope_manifest_tsv: `not provided`"
    )
    return "\n".join(
        [
            "# RUN_CONTEXT",
            "",
            "## 1. Run ID",
            "",
            f"- `{run_id}`",
            "",
            "## 2. Run type",
            "",
            f"- `{run_type}`",
            f"- run_dir_kind: `{run_dir_kind}`",
            f"- run_selection_mode: `{run_selection_mode}`",
            f"- bucket_dir: `{bucket_dir}`",
            "",
            "## 3. Purpose",
            "",
            "- Build deterministic Stage 3 formulation relation artifacts from an existing Stage 2 candidate-instance artifact.",
            "",
            "## 4. Starting input artifacts",
            "",
            f"- weak_labels_tsv: `{weak_labels_tsv}`",
            jsonl_line,
            manifest_line,
            "",
            "## 5. Exact script execution order",
            "",
            "1. Run `src/stage3_relation/run_formulation_relation_artifacts_v1.py` with explicit `--run-id`, `--out-subdir`, and `--weak-labels-tsv`.",
            "2. The wrapper invokes `src/stage3_relation/build_formulation_relation_artifacts_v1.py` inside the requested run-scoped subdirectory.",
            "",
            "## 6. Script paths used",
            "",
            "- `src/stage3_relation/run_formulation_relation_artifacts_v1.py`",
            "- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`",
            "",
            "## 7. Intermediate artifacts",
            "",
            "- No hidden intermediate files beyond the explicit Stage 3 relation artifacts.",
            "",
            "## 8. Final outputs",
            "",
            f"- `{out_dir / 'formulation_relation_records_v1.tsv'}`",
            f"- `{out_dir / 'formulation_logic_graph_v1.jsonl'}`",
            f"- `{out_dir / 'formulation_relation_summary_v1.tsv'}`",
            f"- `{out_dir / 'resolved_relation_fields_v1.tsv'}`",
            f"- `{out_dir.parent / 'RUN_CONTEXT.md'}`",
            "",
            "## 9. Benchmark-valid vs diagnostic-only status",
            "",
            "- `diagnostic-only, not benchmark-valid final output`",
            "- Reason: this run materializes the deterministic Stage 3 relation layer only and does not execute the later Stage 5 final-table comparison.",
            "",
            "## 10. Reproduction steps",
            "",
            "```powershell",
            "$env:PYTHONPATH='c:\\Users\\tianc\\Downloads\\GitHub\\RL-Agent-Extraction-PLGANPs'; "
            f"python src/stage3_relation/run_formulation_relation_artifacts_v1.py --run-id {run_id} "
            f"--out-subdir {out_subdir} --weak-labels-tsv {weak_labels_tsv.as_posix()}",
            "```",
            "",
            "## 11. Outcome summary",
            "",
            f"- paper_count: `{stats['paper_count']}`",
            f"- candidate_count: `{stats['candidate_count']}`",
            f"- relation_row_count: `{stats['relation_row_count']}`",
            f"- resolved_relation_field_row_count: `{stats['resolved_relation_field_row_count']}`",
        ]
    ) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Stage 3 formulation relation builder in a run-scoped results subdirectory."
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Explicit legacy compatibility run_id. New writes default to MDEC084 v2 bucket/child naming when omitted.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Explicit results run directory. Supports legacy roots and v2 child execution paths under data/results/.",
    )
    parser.add_argument(
        "--execution-cue",
        default="relation",
        help="Future-facing child cue used only when auto-allocating a new v2 child execution path.",
    )
    parser.add_argument("--out-subdir", required=True)
    parser.add_argument("--weak-labels-tsv", required=True, type=Path)
    parser.add_argument("--weak-labels-jsonl", type=Path, default=None)
    parser.add_argument("--scope-manifest-tsv", type=Path, default=None)
    parser.add_argument(
        "--run-type",
        default="component_regression_run",
        choices=sorted(VALID_RUN_TYPES),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target = resolve_results_write_target(
        results_root=DATA_RESULTS_DIR,
        default_child_cue=args.execution_cue,
        explicit_run_dir=args.run_dir,
        explicit_legacy_run_id=args.run_id,
    )
    run_id = target["run_basename"]
    run_dir = Path(target["run_dir"])
    run_dir_kind = target["path_kind"]
    run_selection_mode = target["selection_mode"]
    bucket_dir = Path(target["bucket_dir"])
    out_subdir = validate_out_subdir(args.out_subdir)
    if args.run_type == "full_pipeline_benchmark_run":
        raise ValueError(
            "Stage 3 relation-only runs must not be labeled full_pipeline_benchmark_run because they stop before Stage 5 benchmark comparison."
        )

    out_dir = run_dir / out_subdir
    if out_dir.exists():
        raise FileExistsError(f"Output subdirectory already exists: {out_dir}")
    if run_dir_kind == "v2_child_execution":
        bucket_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = build_relation_artifacts(
        weak_labels_tsv=args.weak_labels_tsv,
        out_dir=out_dir,
        weak_labels_jsonl=args.weak_labels_jsonl,
        scope_manifest_tsv=args.scope_manifest_tsv,
    )

    run_context_path = run_dir / "RUN_CONTEXT.md"
    run_context_text = render_run_context(
        run_id=run_id,
        run_type=args.run_type,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        out_subdir=out_subdir,
        weak_labels_tsv=args.weak_labels_tsv,
        weak_labels_jsonl=args.weak_labels_jsonl,
        scope_manifest_tsv=args.scope_manifest_tsv,
        out_dir=out_dir,
        stats=stats,
    )
    if run_context_path.exists():
        existing = run_context_path.read_text(encoding="utf-8")
        if run_context_text.strip() not in existing:
            run_context_path.write_text(existing.rstrip() + "\n\n" + run_context_text, encoding="utf-8")
    else:
        run_context_path.write_text(run_context_text, encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "utils" / "update_run_context_with_feature_activation_v1.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        check=True,
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_type": args.run_type,
                "out_dir": str(out_dir),
                "relation_records_path": str(stats["relation_records_path"]),
                "relation_graph_jsonl_path": str(stats["relation_graph_jsonl_path"]),
                "relation_summary_path": str(stats["relation_summary_path"]),
                "resolved_relation_fields_path": str(stats["resolved_relation_fields_path"]),
                "paper_count": stats["paper_count"],
                "candidate_count": stats["candidate_count"],
                "relation_row_count": stats["relation_row_count"],
                "resolved_relation_field_row_count": stats["resolved_relation_field_row_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
