#!/usr/bin/env python3
"""
Run a diagnostic-only DEV15 full pipeline with NVIDIA hosted API as the Stage2 backend.

This utility preserves the governed downstream path:
- Stage2 legacy extractor output surface
- Stage3 deterministic relation materialization
- Stage5 final output materialization
- Stage5 final-table count comparison

It does not overwrite existing runs and does not modify GT authority artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from src.stage3_relation.build_formulation_relation_artifacts_v1 import build_relation_artifacts
    from src.stage5_benchmark.build_minimal_final_output_v1 import build_minimal_final_output
    from src.stage5_benchmark.compare_final_table_to_gt_v1 import compare_final_table_to_gt
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage3_relation.build_formulation_relation_artifacts_v1 import build_relation_artifacts
    from src.stage5_benchmark.build_minimal_final_output_v1 import build_minimal_final_output
    from src.stage5_benchmark.compare_final_table_to_gt_v1 import compare_final_table_to_gt
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target


NVIDIA_HOSTED_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_SCOPE_NAME = "dev15_nvidia_full_pipeline_diagnostic"


def make_run_target() -> dict[str, str]:
    return resolve_results_write_target(
        results_root=DATA_RESULTS_DIR,
        default_child_cue="dev15_nvidia_full_pipeline",
    )


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def run_stage2(
    *,
    scope_manifest_tsv: Path,
    out_dir: Path,
    model: str,
    max_items: int,
    max_chars: int,
    retries: int,
    sleep_sec: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "auto_extract_weak_labels_v7pilot_r3_fixparse.py"),
        "--manifest-tsv",
        str(scope_manifest_tsv),
        "--model",
        model,
        "--llm-backend",
        "nvidia",
        "--max-items",
        str(max_items),
        "--max-chars",
        str(max_chars),
        "--retries",
        str(retries),
        "--sleep",
        str(sleep_sec),
        "--out-dir",
        str(out_dir),
        "--verbose",
    ]
    print("Running Stage2 with NVIDIA hosted backend:")
    print("  " + " ".join(command))
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        check=True,
    )
    summary_path = out_dir / "pilot_summary.json"
    require_exists(summary_path, "Stage2 summary JSON")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_run_context(
    *,
    run_dir: Path,
    source_run_context: dict[str, Any],
    scope_manifest_tsv: Path,
    gt_workbook_xlsx: Path,
    gt_skeleton_tsv: Path | None,
    stage2_out_dir: Path,
    model: str,
    stage2_summary: dict[str, Any],
    stage3_stats: dict[str, Any],
    stage5_stats: dict[str, Any],
    compare_stats: dict[str, Any],
) -> str:
    generated_at = datetime.now().isoformat(timespec="seconds")
    gt_skeleton_line = (
        f"- gt_skeleton_tsv: `{gt_skeleton_tsv}`" if gt_skeleton_tsv is not None else "- gt_skeleton_tsv: `not resolved`"
    )
    return "\n".join(
        [
            "# RUN_CONTEXT",
            "",
            "## 1. Run Type",
            "",
            "- `diagnostic-only`",
            "",
            "## 2. Purpose",
            "",
            "- NVIDIA-backed DEV15 full pipeline diagnostic run.",
            "- NVIDIA replaced the Stage2 LLM backend only.",
            "- Stage3 and Stage5 remained on the governed deterministic path.",
            "- This run is not an official benchmark-valid replacement output.",
            "",
            "## 3. Source Authority Resolution",
            "",
            f"- source_resolution: `{source_run_context['resolution_source']}`",
            f"- source_run_id: `{source_run_context['run_id']}`",
            f"- source_run_dir: `{source_run_context['run_dir']}`",
            f"- active_run_pointer_path: `{source_run_context.get('pointer_path') or ''}`",
            f"- source_manifest_scope: `{scope_manifest_tsv}`",
            f"- gt_workbook_xlsx: `{gt_workbook_xlsx}`",
            gt_skeleton_line,
            "",
            "## 4. Backend Configuration",
            "",
            f"- endpoint: `{NVIDIA_HOSTED_ENDPOINT}`",
            f"- model: `{model}`",
            "- required_environment_variable: `NVIDIA_API_KEY`",
            "- hosted_vs_local_note: This run uses the NVIDIA hosted API Catalog endpoint and does not assume a local NIM server such as `localhost:8000`.",
            "",
            "## 5. Exact Script Execution Order",
            "",
            "1. Run `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` with `--llm-backend nvidia` and the DEV15 scope manifest.",
            "2. Run `src/stage3_relation/build_formulation_relation_artifacts_v1.py` on the Stage2 TSV/JSONL outputs.",
            "3. Run `src/stage5_benchmark/build_minimal_final_output_v1.py` with the Stage2 TSV plus Stage3 relation artifacts.",
            "4. Run `src/stage5_benchmark/compare_final_table_to_gt_v1.py` on the completed Stage5 final table.",
            "",
            "## 6. Final Outputs",
            "",
            f"- `{run_dir / 'RUN_CONTEXT.md'}`",
            f"- `{stage2_out_dir / 'weak_labels__v7pilot_r3_fixparse.tsv'}`",
            f"- `{stage2_out_dir / 'weak_labels__v7pilot_r3_fixparse.jsonl'}`",
            f"- `{stage2_out_dir / 'raw_responses'}`",
            f"- `{run_dir / 'formulation_relation_v1' / 'formulation_relation_records_v1.tsv'}`",
            f"- `{run_dir / 'formulation_relation_v1' / 'resolved_relation_fields_v1.tsv'}`",
            f"- `{run_dir / 'final_formulation_table_v1.tsv'}`",
            f"- `{run_dir / 'final_output_decision_trace_v1.tsv'}`",
            f"- `{run_dir / 'final_output_summary_v1.md'}`",
            f"- `{run_dir / 'final_table_vs_gt_counts.tsv'}`",
            f"- `{run_dir / 'final_table_vs_gt_summary.md'}`",
            "",
            "## 7. Metrics",
            "",
            f"- generated_at: `{generated_at}`",
            f"- stage2_papers: `{stage2_summary.get('n_papers', '')}`",
            f"- stage2_formulations: `{stage2_summary.get('n_formulations', '')}`",
            f"- stage3_relation_rows: `{stage3_stats.get('relation_row_count', '')}`",
            f"- stage3_resolved_relation_field_rows: `{stage3_stats.get('resolved_relation_field_row_count', '')}`",
            f"- stage5_final_rows: `{stage5_stats.get('final_rows', '')}`",
            f"- compare_matched_papers: `{compare_stats.get('papers_matching', '')}`",
            f"- compare_mismatched_papers: `{compare_stats.get('papers_mismatching', '')}`",
            "",
            "## 8. Benchmark Status",
            "",
            "- `diagnostic-only, not benchmark-valid final output`",
            "- Reason: this run swaps the Stage2 LLM backend to NVIDIA for controlled comparison work and does not replace the current official Gemini-governed benchmark lineage.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DEV15 full pipeline with NVIDIA hosted API as the diagnostic Stage2 backend."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--scope-name", default=DEFAULT_SCOPE_NAME)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=50000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY is missing. Set it in the environment or .env before running.")

    source_run_context = resolve_run_context()
    scope_manifest_tsv = resolve_artifact_path(
        explicit_path=None,
        run_context=source_run_context,
        pointer_key="scope_manifest_tsv",
        required=True,
    )
    gt_workbook_xlsx = resolve_artifact_path(
        explicit_path=None,
        run_context=source_run_context,
        pointer_key="gt_workbook_xlsx",
        required=True,
    )
    gt_skeleton_tsv = resolve_artifact_path(
        explicit_path=None,
        run_context=source_run_context,
        pointer_key="gt_skeleton_tsv",
        required=False,
    )
    assert scope_manifest_tsv is not None
    assert gt_workbook_xlsx is not None
    require_exists(scope_manifest_tsv, "scope manifest TSV")
    require_exists(gt_workbook_xlsx, "GT workbook XLSX")

    target = make_run_target()
    run_dir = Path(target["run_dir"])
    bucket_dir = Path(target["bucket_dir"])
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if target["path_kind"] == "v2_child_execution":
        bucket_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    stage2_out_dir = run_dir / "weak_labels_v7pilot_r3_fixparse"
    stage2_out_dir.mkdir(parents=True, exist_ok=False)

    print(f"Resolved source run: {source_run_context['run_id']}")
    print(f"Resolved scope manifest: {scope_manifest_tsv}")
    print(f"Resolved GT workbook: {gt_workbook_xlsx}")
    print(f"Output run directory: {run_dir}")

    stage2_summary = run_stage2(
        scope_manifest_tsv=scope_manifest_tsv,
        out_dir=stage2_out_dir,
        model=args.model,
        max_items=args.max_items,
        max_chars=args.max_chars,
        retries=args.retries,
        sleep_sec=args.sleep,
    )

    weak_labels_tsv = stage2_out_dir / "weak_labels__v7pilot_r3_fixparse.tsv"
    weak_labels_jsonl = stage2_out_dir / "weak_labels__v7pilot_r3_fixparse.jsonl"
    require_exists(weak_labels_tsv, "Stage2 weak-label TSV")
    require_exists(weak_labels_jsonl, "Stage2 weak-label JSONL")

    print("Running Stage3 deterministic relation materialization...")
    stage3_stats = build_relation_artifacts(
        weak_labels_tsv=weak_labels_tsv,
        out_dir=run_dir / "formulation_relation_v1",
        weak_labels_jsonl=weak_labels_jsonl,
        scope_manifest_tsv=scope_manifest_tsv,
    )
    print(
        "Stage3 complete: "
        f"paper_count={stage3_stats['paper_count']} "
        f"candidate_count={stage3_stats['candidate_count']} "
        f"relation_row_count={stage3_stats['relation_row_count']}"
    )

    relation_records_tsv = Path(stage3_stats["relation_records_path"])
    resolved_relation_fields_tsv = Path(stage3_stats["resolved_relation_fields_path"])

    print("Running Stage5 final output materialization...")
    stage5_stats = build_minimal_final_output(
        weak_labels_tsv,
        run_dir,
        relation_records_tsv=relation_records_tsv,
        resolved_relation_fields_tsv=resolved_relation_fields_tsv,
    )
    print(
        "Stage5 complete: "
        f"input_rows={stage5_stats['input_rows']} "
        f"final_rows={stage5_stats['final_rows']} "
        f"filtered_rows={stage5_stats['filtered_rows']}"
    )

    final_table_tsv = Path(stage5_stats["final_table_path"])
    require_exists(final_table_tsv, "Stage5 final formulation table")

    print("Running governed final-table GT count comparison...")
    compare_stats = compare_final_table_to_gt(
        final_table_tsv=final_table_tsv,
        gt_xlsx=gt_workbook_xlsx,
        scope_manifest_tsv=scope_manifest_tsv,
        out_dir=run_dir,
        scope_name=args.scope_name,
    )
    print(
        "GT count comparison complete: "
        f"matched_papers={compare_stats['papers_matching']} "
        f"mismatched_papers={compare_stats['papers_mismatching']}"
    )

    run_context = build_run_context(
        run_dir=run_dir,
        source_run_context=source_run_context,
        scope_manifest_tsv=scope_manifest_tsv,
        gt_workbook_xlsx=gt_workbook_xlsx,
        gt_skeleton_tsv=gt_skeleton_tsv,
        stage2_out_dir=stage2_out_dir,
        model=args.model,
        stage2_summary=stage2_summary,
        stage3_stats=stage3_stats,
        stage5_stats=stage5_stats,
        compare_stats=compare_stats,
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

    print("Full pipeline run completed successfully.")
    print(f"Run directory: {run_dir}")
    print(f"Final table: {final_table_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
