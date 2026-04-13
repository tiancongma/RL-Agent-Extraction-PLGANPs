#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        OUTPUT_JSONL_NAME,
        OUTPUT_SUMMARY_NAME,
        convert_legacy_raw_response_to_v2,
        read_tsv,
        summary_row,
        write_tsv,
    )
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        OUTPUT_JSONL_NAME,
        OUTPUT_SUMMARY_NAME,
        convert_legacy_raw_response_to_v2,
        read_tsv,
        summary_row,
        write_tsv,
    )
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target


RAW_RESPONSE_FILENAME_TEMPLATE = "{paper_key}__stage2_v2_raw_response.json"
RUN_METADATA_NAME = "stage2_s2_5_run_metadata_v1.json"
TARGETED_MANIFEST_NAME = "targeted_manifest.tsv"
SEMANTIC_SUBDIR = "semantic_stage2_objects"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume frozen S2-4b raw responses and materialize S2-5 semantic-intermediate artifacts only."
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
        default="s2_5_semantic_parsing",
        help="Future-facing child cue used only when auto-allocating a new v2 child execution path.",
    )
    parser.add_argument(
        "--manifest-tsv",
        required=True,
        help="TSV manifest containing the selected papers and their text_path provenance.",
    )
    parser.add_argument(
        "--raw-responses-dir",
        required=True,
        help="Directory containing frozen S2-4b raw responses named <paper_key>__stage2_v2_raw_response.json.",
    )
    parser.add_argument(
        "--paper-key",
        action="append",
        dest="paper_keys",
        default=[],
        help="Repeatable paper key filter. Default: all manifest rows with matching frozen raw responses.",
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


def raw_response_path(raw_responses_dir: Path, paper_key: str) -> Path:
    return raw_responses_dir / RAW_RESPONSE_FILENAME_TEMPLATE.format(paper_key=paper_key)


def write_manifest_subset(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No manifest rows selected for S2-5 semantic parsing.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    manifest_tsv: Path,
    targeted_manifest_tsv: Path,
    raw_responses_dir: Path,
    selected_paper_keys: list[str],
    selected_raw_response_paths: list[Path],
    semantic_dir: Path,
    semantic_jsonl_path: Path,
    semantic_summary_path: Path,
    success_count: int,
    failure_count: int,
) -> str:
    key_block = "\n".join(f"- `{key}`" for key in selected_paper_keys) if selected_paper_keys else "- `none`"
    raw_block = "\n".join(f"- `{path}`" for path in selected_raw_response_paths) if selected_raw_response_paths else "- `none`"
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
- It materializes the frozen `S2-5` semantic-parsing surface only.
- It does not create the authoritative completed Stage2 artifact and it is not a lawful Stage3 resume boundary.

## 3. Purpose
- Consume frozen `S2-4b` raw-response payloads only.
- Parse those payloads into governed Stage2 semantic-intermediate object artifacts.
- Preserve object families, raw expressions, marker readiness, evidence handoff, and provenance-carrying intermediate structure.
- Stop before `S2-6` contract validation and before `S2-7` compatibility projection.

## 4. Stage Boundary
- current_stage_boundary: `S2-5`
- boundary_class: `internal_intermediate`
- authoritative_for_downstream: `no`
- lawful_resume_boundary: `no`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::convert_legacy_raw_response_to_v2`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::normalize_replayed_live_document`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_v2_document`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::finalize_llm_first_document`
- stop_boundary: `semantic_intermediate_artifacts_written`
- next_lawful_step: `S2-6 contract validation`

## 5. Inputs
- manifest_tsv: `{manifest_tsv}`
- targeted_manifest_tsv: `{targeted_manifest_tsv}`
- raw_responses_dir: `{raw_responses_dir}`
- selected_paper_keys:
{key_block}
- selected_raw_response_files:
{raw_block}
- input_contract_note:
  - frozen raw responses are the primary semantic input
  - manifest provenance is used only to resolve per-paper metadata, `text_path`, and table-file provenance
  - no clean text or evidence blocks are reread as the primary semantic input

## 6. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
2. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::convert_legacy_raw_response_to_v2`
3. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::summary_row`

## 7. Outputs
- semantic-intermediate directory:
  - `{semantic_dir}`
- semantic-intermediate JSONL:
  - `{semantic_jsonl_path}`
- semantic-intermediate summary TSV:
  - `{semantic_summary_path}`
- run context:
  - `{run_dir / 'RUN_CONTEXT.md'}`
- machine-readable run metadata:
  - `{run_dir / RUN_METADATA_NAME}`

## 8. Stop Rule
- This run stops after `S2-5` semantic-intermediate artifacts are written.
- The following substeps were not executed:
  - `S2-6 contract validation`
  - `S2-7 compatibility projection`
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 9. Run Summary
- parsed_document_count: `{success_count}`
- failures: `{failure_count}`
- success_status: `{"pass" if failure_count == 0 else "partial_fail"}`
"""


def main() -> None:
    args = parse_args()
    manifest_tsv = repo_path(args.manifest_tsv)
    raw_responses_dir = repo_path(args.raw_responses_dir)
    if not manifest_tsv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_tsv}")
    if not raw_responses_dir.exists():
        raise FileNotFoundError(f"Raw responses directory not found: {raw_responses_dir}")

    manifest_rows = read_tsv(manifest_tsv)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest_tsv}")
    manifest_by_key = {
        str(row.get("key", "")).strip(): row
        for row in manifest_rows
        if str(row.get("key", "")).strip()
    }

    requested_keys = [str(key).strip() for key in args.paper_keys if str(key).strip()]
    if requested_keys:
        missing_manifest_keys = [key for key in requested_keys if key not in manifest_by_key]
        if missing_manifest_keys:
            raise ValueError(f"Manifest missing requested paper keys: {missing_manifest_keys}")
        selected_keys = requested_keys
    else:
        selected_keys = [
            key
            for key in manifest_by_key
            if raw_response_path(raw_responses_dir, key).exists()
        ]
    if not selected_keys:
        raise ValueError("No S2-4b raw responses were selected for S2-5 semantic parsing.")

    selected_rows = [manifest_by_key[key] for key in selected_keys]
    selected_raw_paths = []
    for key in selected_keys:
        path = raw_response_path(raw_responses_dir, key)
        if not path.exists():
            raise FileNotFoundError(f"Frozen raw response not found for {key}: {path}")
        selected_raw_paths.append(path)

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

    targeted_manifest_tsv = run_dir / TARGETED_MANIFEST_NAME
    write_manifest_subset(targeted_manifest_tsv, selected_rows)

    semantic_dir = run_dir / SEMANTIC_SUBDIR
    semantic_dir.mkdir(parents=True, exist_ok=False)
    semantic_jsonl_path = semantic_dir / OUTPUT_JSONL_NAME
    semantic_summary_path = semantic_dir / OUTPUT_SUMMARY_NAME

    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with semantic_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in selected_rows:
            key = str(record.get("key", "")).strip()
            current_raw_path = raw_response_path(raw_responses_dir, key)
            try:
                document = convert_legacy_raw_response_to_v2(
                    record=record,
                    raw_response_path=current_raw_path,
                    raw_response_text=current_raw_path.read_text(encoding="utf-8", errors="replace"),
                )
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
                summary_rows.append(summary_row(document))
            except Exception as exc:
                failures.append(
                    {
                        "paper_key": key,
                        "raw_response_path": to_repo_rel(current_raw_path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                raise

    write_tsv(
        semantic_summary_path,
        summary_rows,
        [
            "document_key",
            "doi",
            "source_mode",
            "stage2_semantic_source_mode",
            "formulation_count",
            "component_count",
            "variable_count",
            "measurement_count",
            "relation_hint_count",
            "evidence_span_count",
            "unassigned_observation_count",
            "ph_variable_count",
            "doe_factor_count",
            "doe_scope_declared",
            "pdi_measurement_present",
            "zeta_measurement_present",
            "multi_component_formulation_count",
        ],
    )

    run_context = build_run_context(
        run_id=run_id,
        run_dir=run_dir,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        manifest_tsv=manifest_tsv,
        targeted_manifest_tsv=targeted_manifest_tsv,
        raw_responses_dir=raw_responses_dir,
        selected_paper_keys=selected_keys,
        selected_raw_response_paths=selected_raw_paths,
        semantic_dir=semantic_dir,
        semantic_jsonl_path=semantic_jsonl_path,
        semantic_summary_path=semantic_summary_path,
        success_count=len(summary_rows),
        failure_count=len(failures),
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")

    run_metadata = {
        "schema": "stage2_s2_5_run_metadata_v1",
        "run_id": run_id,
        "run_dir": to_repo_rel(run_dir),
        "stage_boundary": "S2-5",
        "boundary_class": "internal_intermediate",
        "authoritative_for_downstream": False,
        "lawful_resume_boundary": False,
        "owner_script": "src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py",
        "owner_function_surface": [
            "src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::convert_legacy_raw_response_to_v2",
            "src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::normalize_replayed_live_document",
            "src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_v2_document",
            "src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::finalize_llm_first_document",
            "src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::summary_row",
        ],
        "inputs": {
            "manifest_tsv": to_repo_rel(manifest_tsv),
            "targeted_manifest_tsv": to_repo_rel(targeted_manifest_tsv),
            "raw_responses_dir": to_repo_rel(raw_responses_dir),
            "selected_raw_response_files": [to_repo_rel(path) for path in selected_raw_paths],
        },
        "outputs": {
            "semantic_dir": to_repo_rel(semantic_dir),
            "semantic_jsonl": to_repo_rel(semantic_jsonl_path),
            "semantic_summary_tsv": to_repo_rel(semantic_summary_path),
            "run_context": to_repo_rel(run_dir / "RUN_CONTEXT.md"),
        },
        "stop_boundary": "semantic_intermediate_artifacts_written",
        "next_lawful_step": "S2-6 contract validation",
        "not_executed": [
            "S2-6 contract validation",
            "S2-7 compatibility projection",
            "Stage3",
            "Stage4",
            "Stage5",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "failure_count": len(failures),
        "failures": failures,
    }
    (run_dir / RUN_METADATA_NAME).write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"semantic_jsonl={semantic_jsonl_path}")
    print(f"semantic_summary={semantic_summary_path}")
    print(f"parsed_document_count={len(summary_rows)}")
    print(f"failure_count={len(failures)}")


if __name__ == "__main__":
    main()
