#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from src.stage2_sampling_labels.run_stage2_s2_5_semantic_parsing_v1 import (
        RUN_METADATA_NAME as S2_5_RUN_METADATA_NAME,
        SEMANTIC_SUBDIR,
        TARGETED_MANIFEST_NAME,
    )
    from src.stage2_sampling_labels.validate_stage2_semantic_authority_contract_v1 import (
        ALLOWED_MODES,
        collect_mode_values,
        normalize_text,
        read_jsonl,
        summarize_authority_reattachment_sidecar,
        validate_semantic_documents,
    )
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.run_stage2_s2_5_semantic_parsing_v1 import (
        RUN_METADATA_NAME as S2_5_RUN_METADATA_NAME,
        SEMANTIC_SUBDIR,
        TARGETED_MANIFEST_NAME,
    )
    from src.stage2_sampling_labels.validate_stage2_semantic_authority_contract_v1 import (
        ALLOWED_MODES,
        collect_mode_values,
        normalize_text,
        read_jsonl,
        summarize_authority_reattachment_sidecar,
        validate_semantic_documents,
    )
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target


SEMANTIC_JSONL_NAME = "semantic_stage2_v2_objects.jsonl"
SEMANTIC_SUMMARY_NAME = "semantic_stage2_v2_summary.tsv"
RUN_METADATA_NAME = "stage2_s2_6_run_metadata_v1.json"
REPORT_SUBDIR = "analysis"
REPORT_NAME = "stage2_semantic_authority_contract_report_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume S2-5 semantic-intermediate artifacts only and validate the maintained S2-6 semantic authority contract."
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
        default="s2_6_contract_validation",
        help="Future-facing child cue used only when auto-allocating a new v2 child execution path.",
    )
    parser.add_argument(
        "--s2-5-run-dir",
        required=True,
        help="Run directory produced by the maintained S2-5 semantic-parsing runner.",
    )
    parser.add_argument(
        "--authority-reattachment-sidecar",
        default="",
        help="Optional diagnostic S2-5b authority reattachment sidecar file or run/root directory. Diagnostic-only; does not create semantic authorization.",
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


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    source_s2_5_run_dir: Path,
    semantic_jsonl_path: Path,
    semantic_summary_path: Path | None,
    source_targeted_manifest_path: Path | None,
    source_s2_5_metadata_path: Path | None,
    authority_reattachment_sidecar_path: Path | None,
    authority_reattachment_diagnostics: dict[str, object],
    report_path: Path,
    declared_mode: str,
    document_count: int,
    warning_count: int,
    error_count: int,
) -> str:
    source_items = [
        f"- source_s2_5_run_dir: `{source_s2_5_run_dir}`",
        f"- semantic_jsonl: `{semantic_jsonl_path}`",
    ]
    if semantic_summary_path is not None:
        source_items.append(f"- semantic_summary_tsv: `{semantic_summary_path}`")
    if source_targeted_manifest_path is not None:
        source_items.append(f"- targeted_manifest_tsv: `{source_targeted_manifest_path}`")
    if source_s2_5_metadata_path is not None:
        source_items.append(f"- source_s2_5_metadata: `{source_s2_5_metadata_path}`")
    if authority_reattachment_sidecar_path is not None:
        source_items.append(f"- authority_reattachment_sidecar: `{authority_reattachment_sidecar_path}`")
    source_block = "\n".join(source_items)
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
- It materializes the frozen `S2-6` contract-validation surface only.
- It does not create the authoritative completed Stage2 artifact and it is not a lawful Stage3 resume boundary.

## 3. Purpose
- Consume `S2-5` semantic-intermediate artifacts only.
- Validate semantic-source mode declaration, semantic scope provenance, marker readiness governance, and maintained semantic-intermediate contract compliance.
- Stop before `S2-7` compatibility projection.

## 4. Stage Boundary
- current_stage_boundary: `S2-6`
- boundary_class: `internal_intermediate`
- authoritative_for_downstream: `no`
- lawful_resume_boundary: `no`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_semantic_documents`
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_table_scope`
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_semantic_signals`
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_formulation_candidate`
- stop_boundary: `semantic_contract_validation_artifacts_written`
- next_lawful_step: `S2-7 compatibility projection`

## 5. Inputs
{source_block}
- input_contract_note:
  - the frozen `S2-5` semantic JSONL is the only required semantic input
  - this runner does not consume raw `S2-4b` responses
  - this runner does not reread clean text or evidence blocks as the primary semantic input
  - this runner does not invoke any live model call

## 6. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
2. `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_semantic_documents`

## 7. Outputs
- validation report:
  - `{report_path}`
- run context:
  - `{run_dir / 'RUN_CONTEXT.md'}`
- machine-readable run metadata:
  - `{run_dir / RUN_METADATA_NAME}`

## 8. Stop Rule
- This run stops after `S2-6` validation artifacts are written.
- The following substeps were not executed:
  - `S2-5 semantic parsing`
  - `S2-7 compatibility projection`
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 9. Run Summary
- declared_mode: `{declared_mode or '<blank>'}`
- document_count: `{document_count}`
- warning_count: `{warning_count}`
- error_count: `{error_count}`
- authority_reattachment_semantic_signal_count: `{authority_reattachment_diagnostics.get('semantic_signal_count', 0)}`
- authority_reattachment_reattached_target_count: `{authority_reattachment_diagnostics.get('reattached_target_count', 0)}`
- authority_reattachment_unresolved_target_count: `{authority_reattachment_diagnostics.get('unresolved_target_count', 0)}`
- authority_reattachment_ambiguous_target_count: `{authority_reattachment_diagnostics.get('ambiguous_target_count', 0)}`
- allowed_modes:
  - `{sorted(ALLOWED_MODES)}`
- success_status: `{"pass" if error_count == 0 else "fail"}`
"""


def main() -> None:
    args = parse_args()
    source_s2_5_run_dir = repo_path(args.s2_5_run_dir)
    if not source_s2_5_run_dir.exists():
        raise FileNotFoundError(f"S2-5 run directory not found: {source_s2_5_run_dir}")

    semantic_jsonl_path = source_s2_5_run_dir / SEMANTIC_SUBDIR / SEMANTIC_JSONL_NAME
    if not semantic_jsonl_path.exists():
        raise FileNotFoundError(f"S2-5 semantic JSONL not found: {semantic_jsonl_path}")
    semantic_summary_path = source_s2_5_run_dir / SEMANTIC_SUBDIR / SEMANTIC_SUMMARY_NAME
    if not semantic_summary_path.exists():
        semantic_summary_path = None
    source_targeted_manifest_path = source_s2_5_run_dir / TARGETED_MANIFEST_NAME
    if not source_targeted_manifest_path.exists():
        source_targeted_manifest_path = None
    source_s2_5_metadata_path = source_s2_5_run_dir / S2_5_RUN_METADATA_NAME
    if not source_s2_5_metadata_path.exists():
        source_s2_5_metadata_path = None
    authority_reattachment_sidecar_path = repo_path(args.authority_reattachment_sidecar) if normalize_text(args.authority_reattachment_sidecar) else None
    if authority_reattachment_sidecar_path is None:
        inferred = source_s2_5_run_dir / "semantic_stage2_objects" / "authority_reattachment"
        if inferred.exists():
            authority_reattachment_sidecar_path = inferred
    authority_reattachment_diagnostics = summarize_authority_reattachment_sidecar(authority_reattachment_sidecar_path)

    documents = read_jsonl(semantic_jsonl_path)
    semantic_validation = validate_semantic_documents(documents)
    document_modes, row_modes, observed_modes, declared_mode = collect_mode_values(documents, None)
    report = {
        "schema": "stage2_semantic_authority_contract_report_v1",
        "validation_scope": "s2_6_semantic_intermediate_only",
        "source_boundary": "S2-5",
        "next_lawful_step": "S2-7 compatibility projection",
        "declared_mode": declared_mode,
        "document_modes": document_modes,
        "row_modes": row_modes,
        "document_count": len(documents),
        "row_count": 0,
        "error_count": len(semantic_validation["errors"]),
        "warning_count": len(semantic_validation["warnings"]),
        "errors": semantic_validation["errors"],
        "warnings": semantic_validation["warnings"],
        "status": "pass" if not semantic_validation["errors"] else "fail",
        "inputs": {
            "source_s2_5_run_dir": to_repo_rel(source_s2_5_run_dir),
            "semantic_jsonl": to_repo_rel(semantic_jsonl_path),
            "semantic_summary_tsv": to_repo_rel(semantic_summary_path) if semantic_summary_path is not None else "",
            "targeted_manifest_tsv": to_repo_rel(source_targeted_manifest_path) if source_targeted_manifest_path is not None else "",
            "source_s2_5_metadata": to_repo_rel(source_s2_5_metadata_path) if source_s2_5_metadata_path is not None else "",
            "authority_reattachment_sidecar": to_repo_rel(authority_reattachment_sidecar_path) if authority_reattachment_sidecar_path is not None else "",
        },
        "authority_reattachment_diagnostics": authority_reattachment_diagnostics,
        "observed_modes": observed_modes,
    }

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

    analysis_dir = run_dir / REPORT_SUBDIR
    analysis_dir.mkdir(parents=True, exist_ok=False)
    report_path = analysis_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    run_context = build_run_context(
        run_id=run_id,
        run_dir=run_dir,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        source_s2_5_run_dir=source_s2_5_run_dir,
        semantic_jsonl_path=semantic_jsonl_path,
        semantic_summary_path=semantic_summary_path,
        source_targeted_manifest_path=source_targeted_manifest_path,
        source_s2_5_metadata_path=source_s2_5_metadata_path,
        authority_reattachment_sidecar_path=authority_reattachment_sidecar_path,
        authority_reattachment_diagnostics=authority_reattachment_diagnostics,
        report_path=report_path,
        declared_mode=declared_mode,
        document_count=len(documents),
        warning_count=len(semantic_validation["warnings"]),
        error_count=len(semantic_validation["errors"]),
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")

    run_metadata = {
        "schema": "stage2_s2_6_run_metadata_v1",
        "run_id": run_id,
        "run_dir": to_repo_rel(run_dir),
        "stage_boundary": "S2-6",
        "boundary_class": "internal_intermediate",
        "authoritative_for_downstream": False,
        "lawful_resume_boundary": False,
        "owner_script": "src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py",
        "owner_function_surface": [
            "src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_semantic_documents",
            "src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_selection_marker",
            "src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_inheritance_marker",
            "src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::has_llm_declared_doe_scope",
            "src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::has_table_formulation_scope_marker",
        ],
        "inputs": report["inputs"],
        "outputs": {
            "validation_report": to_repo_rel(report_path),
            "run_context": to_repo_rel(run_dir / "RUN_CONTEXT.md"),
        },
        "stop_boundary": "semantic_contract_validation_artifacts_written",
        "next_lawful_step": "S2-7 compatibility projection",
        "not_executed": [
            "S2-5 semantic parsing",
            "S2-7 compatibility projection",
            "Stage3",
            "Stage4",
            "Stage5",
        ],
        "status": report["status"],
        "error_count": report["error_count"],
        "warning_count": report["warning_count"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / RUN_METADATA_NAME).write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"source_s2_5_run_dir={source_s2_5_run_dir}")
    print(f"semantic_jsonl={semantic_jsonl_path}")
    if semantic_summary_path is not None:
        print(f"semantic_summary={semantic_summary_path}")
    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"validation_report={report_path}")
    print(f"document_count={report['document_count']}")
    print(f"warning_count={report['warning_count']}")
    print(f"error_count={report['error_count']}")
    if semantic_validation["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
