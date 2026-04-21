#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import (
        AUTHORITY_REATTACHMENT_SIDECAR_NAME,
        CONTRACT_TSV_NAME,
        EXECUTION_LEDGER_NAME,
        FUNCTION_UNIT_ACTIVATION_NAME,
        LEGACY_JSONL_NAME,
        LEGACY_TSV_NAME,
        SUMMARY_JSON_NAME,
        TRACE_TSV_NAME,
        run_projection,
    )
    from src.stage2_sampling_labels.run_stage2_composite_v1 import COMPAT_SUBDIR
    from src.stage2_sampling_labels.run_stage2_s2_6_contract_validation_v1 import REPORT_NAME, REPORT_SUBDIR
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import (
        AUTHORITY_REATTACHMENT_SIDECAR_NAME,
        CONTRACT_TSV_NAME,
        EXECUTION_LEDGER_NAME,
        FUNCTION_UNIT_ACTIVATION_NAME,
        LEGACY_JSONL_NAME,
        LEGACY_TSV_NAME,
        SUMMARY_JSON_NAME,
        TRACE_TSV_NAME,
        run_projection,
    )
    from src.stage2_sampling_labels.run_stage2_composite_v1 import COMPAT_SUBDIR
    from src.stage2_sampling_labels.run_stage2_s2_6_contract_validation_v1 import REPORT_NAME, REPORT_SUBDIR
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target


RUN_METADATA_NAME = "stage2_s2_7_run_metadata_v1.json"
DOE_GUARD_NAME = "numbered_doe_regression_guard_v1.tsv"


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume an S2-6-validated S2-5 semantic surface and materialize the maintained S2-7 completed Stage2 artifact."
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
        default="s2_7_compatibility_projection",
        help="Future-facing child cue used only when auto-allocating a new v2 child execution path.",
    )
    parser.add_argument(
        "--s2-6-run-dir",
        required=True,
        help="Run directory produced by the maintained S2-6 contract-validation runner.",
    )
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    source_s2_6_run_dir: Path,
    source_validation_report_path: Path,
    source_validation_status: str,
    source_s2_5_run_dir: Path,
    semantic_jsonl_path: Path,
    authority_sidecar_path: Path,
    compat_dir: Path,
    projection_summary_path: Path,
    function_unit_activation_report_path: Path,
    execution_ledger_path: Path,
    projected_row_count: int,
    projected_document_count: int,
) -> str:
    return f"""# RUN_CONTEXT

## 1. Run ID
`{run_id}`

## 1a. Run Path
- run_dir: `{run_dir}`
- run_dir_kind: `{run_dir_kind}`
- run_selection_mode: `{run_selection_mode}`
- bucket_dir: `{bucket_dir}`

## 2. Run Type
`intermediate_diagnostic_run`

Benchmark reporting rule:
- This run is `diagnostic-only, not benchmark-valid final output`.
- It materializes the frozen `S2-7` completed Stage2 artifact only.
- It does not execute `Stage3`, `Stage4`, or `Stage5`.

## 3. Purpose
- Consume the `S2-6`-validated `S2-5` semantic-intermediate surface only.
- Invoke the maintained compatibility-projection logic without any new LLM call.
- Materialize the completed Stage2 artifact required by unchanged downstream `Stage3` consumers.
- Stop before `Stage3` execution.

## 4. Stage Boundary
- current_stage_boundary: `S2-7`
- boundary_class: `mainline_resume_boundary`
- authoritative_for_downstream: `yes`
- lawful_resume_boundary: `yes`
- resume_entrypoint: `src/stage3_relation/run_formulation_relation_artifacts_v1.py`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::run_projection`
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::project_document`
- stop_boundary: `completed_stage2_artifact_written`
- next_lawful_step: `Stage3 relation materialization`

## 5. Inputs
- source_s2_6_run_dir: `{source_s2_6_run_dir}`
- source_validation_report: `{source_validation_report_path}`
- source_validation_status: `{source_validation_status}`
- source_s2_5_run_dir: `{source_s2_5_run_dir}`
- semantic_jsonl: `{semantic_jsonl_path}`
- authority_reattachment_sidecar: `{authority_sidecar_path}`
- input_contract_note:
  - this runner requires a passing `S2-6` validation report before projection
  - the semantic JSONL path is resolved from the `S2-6` validation surface
  - authority reopen metadata is reattached from the deterministic S2-5 sidecar, not from LLM semantic content
  - this runner does not consume raw `S2-4b` responses
  - this runner does not perform semantic parsing or contract validation
  - this runner does not invoke any live model call

## 6. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
2. `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::run_projection`

## 7. Outputs
- completed Stage2 artifact directory:
  - `{compat_dir}`
- completed Stage2 weak-label TSV:
  - `{compat_dir / LEGACY_TSV_NAME}`
- completed Stage2 weak-label JSONL:
  - `{compat_dir / LEGACY_JSONL_NAME}`
- compatibility trace TSV:
  - `{compat_dir / TRACE_TSV_NAME}`
- compatibility summary JSON:
  - `{projection_summary_path}`
- function-unit activation report v2:
  - `{function_unit_activation_report_path}`
- execution ledger v2:
  - `{execution_ledger_path}`
- numbered DOE guard TSV:
  - `{compat_dir / DOE_GUARD_NAME}`
- projection contract TSV:
  - `{compat_dir / CONTRACT_TSV_NAME}`
- run context:
  - `{run_dir / 'RUN_CONTEXT.md'}`
- machine-readable run metadata:
  - `{run_dir / RUN_METADATA_NAME}`

## 8. Stop Rule
- This run stops after `S2-7` compatibility projection artifacts are written.
- The following stages were not executed:
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 9. Run Summary
- projected_document_count: `{projected_document_count}`
- projected_row_count: `{projected_row_count}`
- completed_stage2_artifact_status: `written`
- lawful_stage3_resume_boundary: `yes`

## 10. Function Unit Execution
- activation_report_v2:
  - `{function_unit_activation_report_path}`
- execution_ledger_v2:
  - `{execution_ledger_path}`
- tracked_fields:
  - `was_unit_considered`
  - `was_unit_authorized`
  - `was_unit_called`
  - `rows_emitted`
  - `rows_retained_after_projection`
  - `skip_reason`
"""


def main() -> None:
    args = parse_args()
    source_s2_6_run_dir = repo_path(args.s2_6_run_dir)
    if not source_s2_6_run_dir.exists():
        raise FileNotFoundError(f"S2-6 run directory not found: {source_s2_6_run_dir}")

    source_validation_report_path = source_s2_6_run_dir / REPORT_SUBDIR / REPORT_NAME
    if not source_validation_report_path.exists():
        raise FileNotFoundError(f"S2-6 validation report not found: {source_validation_report_path}")
    source_validation_report = load_json(source_validation_report_path)
    source_validation_status = str(source_validation_report.get("status", "")).strip()
    if source_validation_status != "pass":
        raise ValueError(f"S2-6 validation report must have status=pass, found: {source_validation_status or '<blank>'}")

    source_s2_5_run_dir_text = str(source_validation_report.get("inputs", {}).get("source_s2_5_run_dir", "")).strip()
    semantic_jsonl_text = str(source_validation_report.get("inputs", {}).get("semantic_jsonl", "")).strip()
    if not source_s2_5_run_dir_text or not semantic_jsonl_text:
        raise ValueError("S2-6 validation report is missing required upstream S2-5 input paths.")

    source_s2_5_run_dir = repo_path(source_s2_5_run_dir_text)
    semantic_jsonl_path = repo_path(semantic_jsonl_text)
    authority_sidecar_path = source_s2_5_run_dir / "semantic_stage2_objects" / AUTHORITY_REATTACHMENT_SIDECAR_NAME
    if not source_s2_5_run_dir.exists():
        raise FileNotFoundError(f"Upstream S2-5 run directory not found: {source_s2_5_run_dir}")
    if not semantic_jsonl_path.exists():
        raise FileNotFoundError(f"Upstream semantic JSONL not found: {semantic_jsonl_path}")
    if not authority_sidecar_path.exists():
        raise FileNotFoundError(f"Upstream authority reattachment sidecar not found: {authority_sidecar_path}")

    target = resolve_results_write_target(
        results_root=DATA_RESULTS_DIR,
        default_child_cue=args.execution_cue,
        explicit_run_dir=repo_path(args.run_dir) if str(args.run_dir).strip() else None,
        explicit_legacy_run_id=args.run_id,
    )
    run_dir = Path(target["run_dir"])
    run_id = target["run_basename"]
    run_dir_kind = target["path_kind"]
    run_selection_mode = target["selection_mode"]
    bucket_dir = Path(target["bucket_dir"])
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if run_dir_kind == "v2_child_execution":
        bucket_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)

    compat_dir = run_dir / COMPAT_SUBDIR
    compat_dir.mkdir(parents=True, exist_ok=False)
    contract_path = compat_dir / CONTRACT_TSV_NAME
    summary = run_projection(
        input_path=semantic_jsonl_path,
        output_dir=compat_dir,
        contract_path=contract_path,
        authority_sidecar_path=authority_sidecar_path,
    )
    analysis_dir = run_dir / "analysis"
    function_unit_activation_report_path = analysis_dir / FUNCTION_UNIT_ACTIVATION_NAME
    execution_ledger_path = analysis_dir / EXECUTION_LEDGER_NAME
    write_tsv(
        function_unit_activation_report_path,
        list(summary.get("function_unit_activation_rows", [])),
        [
            "document_key",
            "function_unit",
            "was_unit_considered",
            "was_unit_authorized",
            "was_unit_called",
            "rows_emitted",
            "rows_retained_after_projection",
            "skip_reason",
            "status",
        ],
    )
    write_tsv(
        execution_ledger_path,
        list(summary.get("execution_ledger_rows", [])),
        [
            "document_key",
            "function_unit",
            "table_id",
            "scope_id",
            "table_type",
            "marker_provenance",
            "was_unit_considered",
            "was_unit_authorized",
            "was_unit_called",
            "rows_emitted",
            "rows_retained_after_projection",
            "varying_variable_count",
            "varying_variables",
            "table_path",
            "skip_reason",
        ],
    )

    projection_summary_path = compat_dir / SUMMARY_JSON_NAME
    run_context = build_run_context(
        run_id=run_id,
        run_dir=run_dir,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        source_s2_6_run_dir=source_s2_6_run_dir,
        source_validation_report_path=source_validation_report_path,
        source_validation_status=source_validation_status,
        source_s2_5_run_dir=source_s2_5_run_dir,
        semantic_jsonl_path=semantic_jsonl_path,
        authority_sidecar_path=authority_sidecar_path,
        compat_dir=compat_dir,
        projection_summary_path=projection_summary_path,
        function_unit_activation_report_path=function_unit_activation_report_path,
        execution_ledger_path=execution_ledger_path,
        projected_row_count=int(summary["projected_rows"]),
        projected_document_count=int(summary["documents"]),
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")

    run_metadata = {
        "schema": "stage2_s2_7_run_metadata_v1",
        "run_id": run_id,
        "run_dir": to_repo_rel(run_dir),
        "stage_boundary": "S2-7",
        "boundary_class": "mainline_resume_boundary",
        "authoritative_for_downstream": True,
        "lawful_resume_boundary": True,
        "resume_entrypoint": "src/stage3_relation/run_formulation_relation_artifacts_v1.py",
        "owner_script": "src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py",
        "owner_function_surface": [
            "src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::run_projection",
            "src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::project_document",
        ],
        "inputs": {
            "source_s2_6_run_dir": to_repo_rel(source_s2_6_run_dir),
            "source_validation_report": to_repo_rel(source_validation_report_path),
            "source_validation_status": source_validation_status,
            "source_s2_5_run_dir": to_repo_rel(source_s2_5_run_dir),
            "semantic_jsonl": to_repo_rel(semantic_jsonl_path),
            "authority_reattachment_sidecar": to_repo_rel(authority_sidecar_path),
        },
        "outputs": {
            "compat_dir": to_repo_rel(compat_dir),
            "completed_stage2_tsv": to_repo_rel(compat_dir / LEGACY_TSV_NAME),
            "completed_stage2_jsonl": to_repo_rel(compat_dir / LEGACY_JSONL_NAME),
            "compatibility_trace_tsv": to_repo_rel(compat_dir / TRACE_TSV_NAME),
            "compatibility_summary_json": to_repo_rel(projection_summary_path),
            "feature_activation_report_v2_tsv": to_repo_rel(function_unit_activation_report_path),
            "execution_ledger_v2_tsv": to_repo_rel(execution_ledger_path),
            "numbered_doe_guard_tsv": to_repo_rel(compat_dir / DOE_GUARD_NAME),
            "projection_contract_tsv": to_repo_rel(contract_path),
            "run_context": to_repo_rel(run_dir / "RUN_CONTEXT.md"),
        },
        "stop_boundary": "completed_stage2_artifact_written",
        "next_lawful_step": "Stage3 relation materialization",
        "not_executed": [
            "Stage3",
            "Stage4",
            "Stage5",
        ],
        "projection_summary": {
            "documents": int(summary["documents"]),
            "projected_rows": int(summary["projected_rows"]),
            "stage2_semantic_source_modes": summary["stage2_semantic_source_modes"],
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / RUN_METADATA_NAME).write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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

    print(f"source_s2_6_run_dir={source_s2_6_run_dir}")
    print(f"source_validation_report={source_validation_report_path}")
    print(f"source_s2_5_run_dir={source_s2_5_run_dir}")
    print(f"semantic_jsonl={semantic_jsonl_path}")
    print(f"authority_reattachment_sidecar={authority_sidecar_path}")
    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"compat_dir={compat_dir}")
    print(f"completed_stage2_tsv={compat_dir / LEGACY_TSV_NAME}")
    print(f"completed_stage2_jsonl={compat_dir / LEGACY_JSONL_NAME}")
    print(f"projection_summary={projection_summary_path}")
    print(f"feature_activation_report_v2={function_unit_activation_report_path}")
    print(f"execution_ledger_v2={execution_ledger_path}")


if __name__ == "__main__":
    main()
