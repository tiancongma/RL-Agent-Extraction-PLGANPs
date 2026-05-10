#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        build_live_prompt,
        build_prompt_preview_row,
        render_selected_table_candidate,
    )
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        build_live_prompt,
        build_prompt_preview_row,
        render_selected_table_candidate,
    )
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import resolve_results_write_target


EVIDENCE_BLOCKS_FILENAME = "evidence_blocks_v1.json"
PROMPT_TEMPLATE_NAME = "s2_4a_prompt_template_v1.txt"
PROMPTS_JSONL_NAME = "s2_4a_prompts_v1.jsonl"
PROMPT_AUDIT_NAME = "s2_4a_prompt_audit_v1.tsv"
RUN_METADATA_NAME = "stage2_s2_4a_run_metadata_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize frozen S2-4a prompt-construction artifacts from canonical S2-2 evidence_blocks_v1.json inputs."
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
        default="s2_4a_prompt_construction",
        help="Future-facing child cue used only when auto-allocating a new v2 child execution path.",
    )
    parser.add_argument(
        "--evidence-blocks-root",
        required=True,
        help="Root directory containing <paper_key>/evidence_blocks_v1.json folders.",
    )
    parser.add_argument(
        "--paper-key",
        action="append",
        dest="paper_keys",
        default=[],
        help="Repeatable paper key filter. Default: all evidence artifact directories under --evidence-blocks-root.",
    )
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def selected_keys(args: argparse.Namespace, evidence_root: Path) -> list[str]:
    requested = [str(key).strip() for key in args.paper_keys if str(key).strip()]
    if requested:
        return requested
    return sorted(
        child.name
        for child in evidence_root.iterdir()
        if child.is_dir() and (child / EVIDENCE_BLOCKS_FILENAME).exists()
    )


def extract_record_from_evidence_artifact(artifact: dict[str, Any]) -> dict[str, str]:
    key = str(artifact.get("paper_key", "")).strip()
    doi = ""
    title = ""
    for block in artifact.get("evidence_blocks") or []:
        if not isinstance(block, dict):
            continue
        if str(block.get("block_type", "")).strip() != "metadata":
            continue
        lines = str(block.get("text_content", "")).splitlines()
        for line in lines:
            if line.startswith("key: "):
                key = line.split(":", 1)[1].strip() or key
            elif line.startswith("doi: "):
                doi = line.split(":", 1)[1].strip()
            elif line.startswith("title: "):
                title = line.split(":", 1)[1].strip()
        break
    if not key:
        raise ValueError("Evidence artifact is missing paper_key metadata.")
    return {"key": key, "doi": doi, "title": title}


def rehydrate_table_summary_blocks(artifact: dict[str, Any]) -> dict[str, Any]:
    """Rebuild selected table-summary block text from source table assets.

    S2-4a prompt freezes may be regenerated from frozen S2-2/S2-3 evidence roots.
    Those older evidence roots can legally identify selected table assets while
    their cached `text_content` still contains stale prompt-rendering lines such
    as `key_columns`, `table_role_hint`, or `semantic_summary`.  The S2-4a
    materialization boundary owns the final prompt text, so rebuild table-summary
    text from `origin_locator` whenever possible and preserve all non-table
    blocks unchanged.
    """
    rehydrated = copy.deepcopy(artifact)
    blocks = []
    for block in rehydrated.get("evidence_blocks") or []:
        if isinstance(block, dict) and str(block.get("source_type", "")).strip() == "table_summary":
            block["text_content"] = render_selected_table_candidate(block)
        blocks.append(block)
    rehydrated["evidence_blocks"] = blocks
    return rehydrated


def build_template_from_artifact(artifact: dict[str, Any]) -> str:
    placeholder_artifact = copy.deepcopy(rehydrate_table_summary_blocks(artifact))
    placeholder_artifact["input_contract"] = dict(placeholder_artifact.get("input_contract") or {})
    placeholder_artifact["input_contract"]["ordered_block_order"] = ["{ordered_block_order}"]
    placeholder_artifact["evidence_blocks"] = [
        {
            "block_id": "template_placeholder",
            "block_type": "placeholder",
            "text_content": "{evidence_pack}",
        }
    ]
    placeholder_record = {
        "key": "{paper_key}",
        "doi": "{doi}",
        "title": "{title}",
    }
    return build_live_prompt(placeholder_record, placeholder_artifact)


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    evidence_blocks_root: Path,
    selected_paper_keys: list[str],
    analysis_dir: Path,
    prompt_template_path: Path,
    prompts_jsonl_path: Path,
    prompt_audit_path: Path,
    success_count: int,
    failure_count: int,
) -> str:
    key_block = "\n".join(f"- `{key}`" for key in selected_paper_keys) if selected_paper_keys else "- `all discovered evidence artifacts`"
    return f"""# RUN_CONTEXT

## 1. Run ID
`{run_id}`

## 1a. Run Path
- run_dir: `{run_dir}`
- run_dir_kind: `{run_dir_kind}`
- run_selection_mode: `{run_selection_mode}`
- bucket_dir: `{bucket_dir}`

## 2. Run Type
`S2-4a_prompt_freeze_only`

Scope note:
- This run materializes the frozen S2-4a prompt-construction surface only.
- It does not run S2-4b live LLM inference or any downstream stage.
- No benchmark comparison is part of this S2-4a-only artifact.

## 3. Purpose
- Materialize the frozen S2-4a prompt-construction surface from canonical S2-2 evidence artifacts.
- Make the pre-LLM prompt set independently runnable, independently auditable, and independently traceable.
- Stop after writing prompt template, per-paper prompt payloads, and prompt-audit artifacts.

## 4. Stage Boundary
- current_stage_boundary: `S2-4a`
- upstream_frozen_dependency: `S2-3 prompt assembly from frozen evidence_blocks_v1.json`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_prompt`
- stop_boundary: `frozen_prompt_artifacts_written`
- next_lawful_step: `S2-4 live LLM call`

## 5. Inputs
- evidence_blocks_root: `{evidence_blocks_root}`
- selected_paper_keys:
{key_block}

## 6. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
2. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_prompt`

## 7. Outputs
- prompt template:
  - `{prompt_template_path}`
- per-paper prompt payloads:
  - `{prompts_jsonl_path}`
- prompt audit:
  - `{prompt_audit_path}`
- run-local analysis directory:
  - `{analysis_dir}`
- run context:
  - `{run_dir / 'RUN_CONTEXT.md'}`
- machine-readable run metadata:
  - `{run_dir / RUN_METADATA_NAME}`

## 8. Stop Rule
- This run stops after S2-4a prompt artifacts are written.
- The following substeps were not executed:
  - `S2-4 live LLM call`
  - `S2-5 semantic parsing`
  - `S2-6 contract validation`
  - `S2-7 compatibility projection`
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 9. Run Summary
- prompt_count: `{success_count}`
- failures: `{failure_count}`
- success_status: `{"pass" if failure_count == 0 else "partial_fail"}`
"""


def main() -> None:
    args = parse_args()
    evidence_root = repo_path(args.evidence_blocks_root)
    if not evidence_root.exists():
        raise FileNotFoundError(f"Evidence blocks root not found: {evidence_root}")

    keys = selected_keys(args, evidence_root)
    if not keys:
        raise ValueError("No evidence block artifacts were selected.")

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

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=False)
    prompts_jsonl_path = analysis_dir / PROMPTS_JSONL_NAME
    prompt_template_path = analysis_dir / PROMPT_TEMPLATE_NAME
    prompt_audit_path = analysis_dir / PROMPT_AUDIT_NAME

    prompt_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    prompt_template_text = ""

    for key in keys:
        evidence_path = evidence_root / key / EVIDENCE_BLOCKS_FILENAME
        if not evidence_path.exists():
            failures.append({"paper_key": key, "reason": "missing_evidence_artifact"})
            continue
        source_artifact = read_json(evidence_path)
        if str(source_artifact.get("contract_version", "")).strip() != "s2_2_evidence_blocks_v1":
            failures.append({"paper_key": key, "reason": "unexpected_evidence_contract"})
            continue
        artifact = rehydrate_table_summary_blocks(source_artifact)
        record = extract_record_from_evidence_artifact(artifact)
        prompt_text = build_live_prompt(record, artifact)
        prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        evidence_blocks = [block for block in artifact.get("evidence_blocks") or [] if isinstance(block, dict)]
        ordered_block_order = artifact.get("input_contract", {}).get("ordered_block_order") or []
        if not prompt_template_text:
            prompt_template_text = build_template_from_artifact(artifact)
        prompt_rows.append(
            {
                "paper_key": record["key"],
                "doi": record["doi"],
                "title": record["title"],
                "source_evidence_artifact_path": to_repo_rel(evidence_path),
                "authority_run_dir": str(artifact.get("authority_run_dir", "")).strip(),
                "authority_payload_root": str(artifact.get("authority_payload_root", "")).strip(),
                "prompt_sha256": prompt_sha256,
                "ordered_block_order": ordered_block_order,
                "prompt_text": prompt_text,
            }
        )
        preview_row = build_prompt_preview_row(
            document={
                "document_key": record["key"],
                "doi": record["doi"],
                "source_mode": "frozen_s2_4a",
            },
            prompt_text=prompt_text,
            table_mode_value=str(artifact.get("input_contract", {}).get("table_mode") or ""),
            summary_enhanced=bool(artifact.get("input_contract", {}).get("summary_first_column_enhancement")),
            input_packing_mode_value=str(artifact.get("input_contract", {}).get("input_packing_mode") or ""),
            ordered_block_order=" > ".join(str(value) for value in ordered_block_order if str(value).strip()),
            evidence_artifact_path=to_repo_rel(evidence_path),
            evidence_artifact=artifact,
            technical_status_overall=str(artifact.get("technical_status", {}).get("overall") or ""),
            design_status_overall=str(artifact.get("design_status", {}).get("overall") or ""),
        )
        issues = []
        if str(preview_row.get("s2_3_ready_overall")) != "pass":
            issues.append(str(preview_row.get("s2_3_readiness_reasons") or "upstream_prompt_nonconformant"))
        status = "pass" if not issues else "fail"
        audit_rows.append(
            {
                "paper_key": record["key"],
                "source_evidence_artifact_path": to_repo_rel(evidence_path),
                "authority_run_dir": str(artifact.get("authority_run_dir", "")).strip(),
                "authority_payload_root": str(artifact.get("authority_payload_root", "")).strip(),
                "evidence_block_count": len(evidence_blocks),
                "ordered_block_order": " > ".join(str(value) for value in ordered_block_order if str(value).strip()),
                "prompt_length": len(prompt_text),
                "prompt_sha256": prompt_sha256,
                "live_prompt_header_mode": preview_row.get("live_prompt_header_mode", ""),
                "runtime_metadata_removed_from_live_prompt": preview_row.get("runtime_metadata_removed_from_live_prompt", ""),
                "live_prompt_contains_runtime_metadata": preview_row.get("live_prompt_contains_runtime_metadata", ""),
                "uses_evidence_pack_only": preview_row.get("uses_evidence_pack_only", ""),
                "all_selected_blocks_included": preview_row.get("all_selected_blocks_included", ""),
                "truncation_detected": preview_row.get("truncation_detected", ""),
                "exact_duplicate_block_count": preview_row.get("exact_duplicate_block_count", ""),
                "prompt_size_policy_status": preview_row.get("prompt_size_policy_status", ""),
                "upstream_design_status_overall": str(artifact.get("design_status", {}).get("overall") or ""),
                "status": status,
                "issues": "|".join(issue for issue in issues if issue),
            }
        )

    prompts_jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompt_rows),
        encoding="utf-8",
    )
    prompt_template_path.write_text(prompt_template_text, encoding="utf-8")
    write_tsv(
        prompt_audit_path,
        [
            "paper_key",
            "source_evidence_artifact_path",
            "authority_run_dir",
            "authority_payload_root",
            "evidence_block_count",
            "ordered_block_order",
            "prompt_length",
            "prompt_sha256",
            "live_prompt_header_mode",
            "runtime_metadata_removed_from_live_prompt",
            "live_prompt_contains_runtime_metadata",
            "uses_evidence_pack_only",
            "all_selected_blocks_included",
            "truncation_detected",
            "exact_duplicate_block_count",
            "prompt_size_policy_status",
            "upstream_design_status_overall",
            "status",
            "issues",
        ],
        audit_rows,
    )

    run_context = build_run_context(
        run_id=run_id,
        run_dir=run_dir,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        evidence_blocks_root=evidence_root,
        selected_paper_keys=keys,
        analysis_dir=analysis_dir,
        prompt_template_path=prompt_template_path,
        prompts_jsonl_path=prompts_jsonl_path,
        prompt_audit_path=prompt_audit_path,
        success_count=len(prompt_rows),
        failure_count=len(failures),
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")

    run_metadata = {
        "schema": "stage2_s2_4a_run_metadata_v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage_boundary": "S2-4a",
        "entrypoint_script": "src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py",
        "upstream_dependency": "S2-2 evidence_blocks_v1.json",
        "frozen_substep_role": "prompt_construction_only",
        "input_evidence_blocks_root": to_repo_rel(evidence_root),
        "selected_paper_keys": keys,
        "outputs": {
            "prompt_template": to_repo_rel(prompt_template_path),
            "prompts_jsonl": to_repo_rel(prompts_jsonl_path),
            "prompt_audit_tsv": to_repo_rel(prompt_audit_path),
        },
        "stop_boundary": "frozen_prompt_artifacts_written",
        "next_lawful_step": "S2-4 live LLM call",
        "failures": failures,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / RUN_METADATA_NAME).write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"prompt_template={prompt_template_path}")
    print(f"prompts_jsonl={prompts_jsonl_path}")
    print(f"prompt_audit={prompt_audit_path}")
    print(f"success_count={len(prompt_rows)}")
    print(f"failure_count={len(failures)}")


if __name__ == "__main__":
    main()
