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
        NORMALIZED_TABLE_PAYLOADS_FILENAME,
        REQUEST_METADATA_FILENAME_TEMPLATE,
        build_stage2_authority_metadata,
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
        NORMALIZED_TABLE_PAYLOADS_FILENAME,
        REQUEST_METADATA_FILENAME_TEMPLATE,
        build_stage2_authority_metadata,
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
AUTHORITY_REATTACHMENT_SIDECAR_NAME = "authority_reattachment_sidecar_v1.json"


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
        "--fallback-legacy-raw-responses-dir",
        default="",
        help="Optional richer legacy raw-response directory used when a replayed live-v2 raw response collapses to the minimal shrunken contract.",
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


def request_metadata_path(raw_responses_dir: Path, paper_key: str) -> Path:
    return raw_responses_dir.parent / "request_metadata" / REQUEST_METADATA_FILENAME_TEMPLATE.format(paper_key=paper_key)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompt_rows(prompt_jsonl_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with prompt_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            key = str(row.get("paper_key", "")).strip()
            if key:
                rows[key] = row
    return rows


def build_locator_index(authority_payload_root: Path, paper_key: str) -> dict[str, Any]:
    manifest_path = authority_payload_root / paper_key / NORMALIZED_TABLE_PAYLOADS_FILENAME
    if not manifest_path.exists():
        return {
            "locator_manifest_path": "",
            "table_scope_locators": [],
            "locator_count": 0,
        }
    payload = read_json(manifest_path)
    table_scope_locators = []
    for item in payload.get("normalized_table_payloads", []):
        if not isinstance(item, dict):
            continue
        locator = {
            "table_id": str(item.get("table_id") or item.get("source_table_id") or "").strip(),
            "source_table_asset_id": str(item.get("source_table_asset_id") or "").strip(),
            "source_table_reference": str(item.get("source_table_reference") or item.get("source_csv_path") or "").strip(),
        }
        if locator["table_id"] or locator["source_table_asset_id"] or locator["source_table_reference"]:
            table_scope_locators.append(locator)
    return {
        "locator_manifest_path": to_repo_rel(manifest_path),
        "table_scope_locators": table_scope_locators,
        "locator_count": len(table_scope_locators),
    }


def resolve_authority_reattachment_entry(
    *,
    raw_responses_dir: Path,
    paper_key: str,
    prompt_cache: dict[Path, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    path = request_metadata_path(raw_responses_dir, paper_key)
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = read_json(path)
        except Exception:
            payload = {}
    authority_run_dir = str(payload.get("authority_run_dir", "")).strip()
    authority_payload_root = str(payload.get("authority_payload_root", "")).strip()
    source_evidence_artifact_path = ""
    resolution_source = ""
    failure_reason = ""
    if authority_run_dir and authority_payload_root:
        resolution_source = "request_metadata_explicit"
    else:
        prompt_jsonl_text = str(payload.get("source_prompts_jsonl_path", "")).strip()
        if not prompt_jsonl_text:
            failure_reason = "source_prompts_jsonl_missing"
        else:
            prompt_jsonl_path = repo_path(prompt_jsonl_text)
            if not prompt_jsonl_path.exists():
                failure_reason = "source_prompts_jsonl_not_found"
            else:
                prompt_rows = prompt_cache.setdefault(prompt_jsonl_path, load_prompt_rows(prompt_jsonl_path))
                prompt_row = prompt_rows.get(paper_key)
                if not isinstance(prompt_row, dict):
                    failure_reason = "paper_key_missing_in_source_prompts_jsonl"
                else:
                    source_evidence_artifact_path = str(prompt_row.get("source_evidence_artifact_path", "")).strip()
                    if not source_evidence_artifact_path:
                        failure_reason = "source_evidence_artifact_path_missing"
                    else:
                        authority_metadata = build_stage2_authority_metadata(
                            stage2_artifact_path=repo_path(source_evidence_artifact_path)
                        )
                        authority_run_dir = str(authority_metadata.get("authority_run_dir", "")).strip()
                        authority_payload_root = str(authority_metadata.get("authority_payload_root", "")).strip()
                        resolution_source = "source_prompts_jsonl_evidence_artifact"
    locator_payload = {
        "locator_manifest_path": "",
        "table_scope_locators": [],
        "locator_count": 0,
    }
    if authority_payload_root:
        authority_payload_root_path = repo_path(authority_payload_root)
        if authority_payload_root_path.exists():
            locator_payload = build_locator_index(authority_payload_root_path, paper_key)
        elif not failure_reason:
            failure_reason = "authority_payload_root_missing"
    resolution_status = "resolved" if authority_run_dir and authority_payload_root else "unresolved"
    if resolution_status == "unresolved" and not failure_reason:
        failure_reason = "authority_metadata_unresolved"
    return {
        "paper_key": paper_key,
        "authority_run_dir": authority_run_dir,
        "authority_payload_root": authority_payload_root,
        "source_evidence_artifact_path": source_evidence_artifact_path,
        "resolution_status": resolution_status,
        "resolution_source": resolution_source,
        "failure_reason": failure_reason,
        **locator_payload,
    }


def read_authority_metadata(raw_responses_dir: Path, paper_key: str, prompt_cache: dict[Path, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    entry = resolve_authority_reattachment_entry(
        raw_responses_dir=raw_responses_dir,
        paper_key=paper_key,
        prompt_cache=prompt_cache,
    )
    try:
        return {
            "authority_run_dir": str(entry.get("authority_run_dir", "")).strip(),
            "authority_payload_root": str(entry.get("authority_payload_root", "")).strip(),
        }
    except Exception:
        return {}


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
    authority_sidecar_path: Path,
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
- Preserve the shrunken live semantic-understanding contract plus governed provenance metadata needed by downstream maintained steps.
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
- authority reattachment sidecar:
  - `{authority_sidecar_path}`
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
    fallback_legacy_raw_responses_dir: Path | None = None
    if str(args.fallback_legacy_raw_responses_dir).strip():
        fallback_legacy_raw_responses_dir = repo_path(args.fallback_legacy_raw_responses_dir)
        if not fallback_legacy_raw_responses_dir.exists():
            raise FileNotFoundError(
                f"Fallback legacy raw responses directory not found: {fallback_legacy_raw_responses_dir}"
            )

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
    prompt_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    authority_sidecar_entries: dict[str, dict[str, Any]] = {}
    with semantic_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in selected_rows:
            key = str(record.get("key", "")).strip()
            current_raw_path = raw_response_path(raw_responses_dir, key)
            try:
                authority_entry = resolve_authority_reattachment_entry(
                    raw_responses_dir=raw_responses_dir,
                    paper_key=key,
                    prompt_cache=prompt_cache,
                )
                authority_sidecar_entries[key] = authority_entry
                document = convert_legacy_raw_response_to_v2(
                    record=record,
                    raw_response_path=current_raw_path,
                    raw_response_text=current_raw_path.read_text(encoding="utf-8", errors="replace"),
                    authority_metadata=read_authority_metadata(raw_responses_dir, key, prompt_cache),
                    fallback_legacy_raw_dir=fallback_legacy_raw_responses_dir,
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
            "table_scope_count",
            "formulation_candidate_count",
            "has_variable_sweep",
            "has_sequential_optimization",
            "has_parent_child_table_relation",
            "has_downstream_non_synthesis_variants",
            "has_measurement_only_variants",
            "primary_preparation_method_hint",
            "primary_variable_count",
            "selected_condition_hint_count",
        ],
    )
    authority_sidecar_path = semantic_dir / AUTHORITY_REATTACHMENT_SIDECAR_NAME
    authority_sidecar_payload = {
        "schema": "authority_reattachment_sidecar_v1",
        "source_stage_boundary": "S2-5",
        "source_raw_responses_dir": to_repo_rel(raw_responses_dir),
        "entries_by_paper_key": authority_sidecar_entries,
    }
    authority_sidecar_path.write_text(
        json.dumps(authority_sidecar_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
        authority_sidecar_path=authority_sidecar_path,
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
            "authority_reattachment_sidecar": to_repo_rel(authority_sidecar_path),
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
    print(f"authority_reattachment_sidecar={authority_sidecar_path}")
    print(f"parsed_document_count={len(summary_rows)}")
    print(f"failure_count={len(failures)}")


if __name__ == "__main__":
    main()
