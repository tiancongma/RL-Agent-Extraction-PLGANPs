#!/usr/bin/env python3
from __future__ import annotations

"""
NON-CANONICAL, STAGE5-ONLY HELPER.

This script is a thin convenience wrapper around
`src/stage5_benchmark/build_minimal_final_output_v1.py`.
It exists only to package Stage 5A final-output closure into a run-scoped
directory with `RUN_CONTEXT.md`.

It is not part of the canonical manual production path and it must not be used
to stand in for the full Stage 0 to Stage 5 pipeline.
"""

import argparse
import json
import subprocess
import re
import sys
from pathlib import Path

from src.stage5_benchmark.build_minimal_final_output_v1 import (
    build_minimal_final_output,
)
from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
from src.utils.run_id import resolve_results_write_target


VALID_RUN_TYPES = {
    "intermediate_diagnostic_run",
    "component_regression_run",
    "full_pipeline_benchmark_run",
}

def render_run_context(
    run_id: str,
    run_type: str,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    input_tsv: Path,
    relation_records_tsv: Path,
    resolved_relation_fields_tsv: Path,
    run_dir: Path,
    stats: dict[str, object],
) -> str:
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
            "- Execute phase 1 of the minimal final-output layer on a current engineering Stage2 candidate-instance artifact.",
            "",
            "## 4. Starting input artifact(s)",
            "",
            f"- candidate_input_tsv: `{input_tsv}`",
            f"- relation_records_tsv: `{relation_records_tsv}`",
            f"- resolved_relation_fields_tsv: `{resolved_relation_fields_tsv}`",
            "",
            "## 5. Exact script execution order",
            "",
            "1. Run `src/stage5_benchmark/run_minimal_final_output_v1.py` with explicit `--run-id` and `--input-tsv`.",
            "2. The runner invokes `src/stage5_benchmark/build_minimal_final_output_v1.py` to build the final-output artifacts in the run root.",
            "",
            "## 6. Script paths used",
            "",
            "- `src/stage5_benchmark/run_minimal_final_output_v1.py`",
            "- `src/stage5_benchmark/build_minimal_final_output_v1.py`",
            "",
            "## 7. Intermediate artifacts",
            "",
            "- No separate intermediate files are emitted in phase 1 beyond the explicit decision trace.",
            "",
            "## 8. Final outputs",
            "",
            f"- `{run_dir / 'final_formulation_table_v1.tsv'}`",
            f"- `{run_dir / 'downstream_variant_records_v1.tsv'}`",
            f"- `{run_dir / 'final_output_decision_trace_v1.tsv'}`",
            f"- `{run_dir / 'final_output_summary_v1.md'}`",
            f"- `{run_dir / 'RUN_CONTEXT.md'}`",
            "",
            "## 9. Benchmark-valid vs diagnostic-only status",
            "",
            "- `diagnostic-only, not benchmark-valid final output`",
            "- Reason: this helper stops after Stage 5A final-output closure and does not execute the Stage 5B final-table GT comparison required for benchmark-valid reporting.",
            "",
            "## 10. Reproduction steps",
            "",
            "```powershell",
            "$env:PYTHONPATH='c:\\Users\\tianc\\Downloads\\GitHub\\RL-Agent-Extraction-PLGANPs'; "
            f"python src/stage5_benchmark/run_minimal_final_output_v1.py --run-id {run_id} "
            f"--input-tsv {input_tsv.as_posix()} --relation-records-tsv {relation_records_tsv.as_posix()} "
            f"--resolved-relation-fields-tsv {resolved_relation_fields_tsv.as_posix()} --run-type {run_type}",
            "```",
            "",
            "## 11. Environment assumptions",
            "",
            "- Repository checkout matches the code used for this run.",
            "- Input TSV already exists and is readable.",
            "- Python environment can import `src.*` modules through `PYTHONPATH`.",
            "",
            "## 12. Outcome summary",
            "",
            f"- input_rows: `{stats['input_rows']}`",
            f"- final_rows: `{stats['final_rows']}`",
            f"- kept_rows: `{stats['kept_rows']}`",
            f"- filtered_rows: `{stats['filtered_rows']}`",
            f"- collapsed_rows: `{stats['collapsed_rows']}`",
            f"- downstream_variant_rows: `{stats['downstream_variant_rows']}`",
        ]
    ) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run phase-1 minimal final-output closure on a Stage2 candidate-instance TSV."
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
        default="stage5",
        help="Future-facing child cue used only when auto-allocating a new v2 child execution path.",
    )
    parser.add_argument("--input-tsv", required=True, type=Path)
    parser.add_argument("--relation-records-tsv", required=True, type=Path)
    parser.add_argument("--resolved-relation-fields-tsv", required=True, type=Path)
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
    if run_dir_kind == "v2_child_execution":
        bucket_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.run_type == "full_pipeline_benchmark_run":
        raise ValueError(
            "Stage 5A final-output-only runs must not be labeled full_pipeline_benchmark_run because they stop before the Stage 5B final-table GT comparison step."
        )

    stats = build_minimal_final_output(
        args.input_tsv,
        run_dir,
        relation_records_tsv=args.relation_records_tsv,
        resolved_relation_fields_tsv=args.resolved_relation_fields_tsv,
    )
    run_context = render_run_context(
        run_id,
        args.run_type,
        run_dir_kind,
        run_selection_mode,
        bucket_dir,
        args.input_tsv,
        args.relation_records_tsv,
        args.resolved_relation_fields_tsv,
        run_dir,
        stats,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")
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
                "input_tsv": str(args.input_tsv),
                "relation_records_tsv": (
                    str(args.relation_records_tsv) if args.relation_records_tsv else ""
                ),
                "resolved_relation_fields_tsv": str(args.resolved_relation_fields_tsv),
                "run_dir": str(run_dir),
                "final_table_path": str(stats["final_table_path"]),
                "downstream_variant_path": str(stats["downstream_variant_path"]),
                "decision_trace_path": str(stats["decision_trace_path"]),
                "summary_path": str(stats["summary_path"]),
                "input_rows": stats["input_rows"],
                "final_rows": stats["final_rows"],
                "filtered_rows": stats["filtered_rows"],
                "collapsed_rows": stats["collapsed_rows"],
                "downstream_variant_rows": stats["downstream_variant_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
