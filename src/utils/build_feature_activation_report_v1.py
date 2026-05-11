#!/usr/bin/env python3
from __future__ import annotations

"""
Build a run-scoped feature activation report from deterministic artifact signals.

Purpose:
- Distinguish repository feature existence from run-local feature activation.
- Make lineage reuse failures visible when a child validation run proves a fix,
  but a benchmark-valid parent run still reuses older artifacts.

This utility is intentionally conservative:
- mark a feature as active only when run artifacts provide direct evidence
- prefer "unclear" over invented activation
- ignore descendant lineage child runs when inspecting a parent run
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

try:
    from src.utils.paths import PROJECT_DIR
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a run-scoped feature activation report.")
    parser.add_argument(
        "--registry",
        default=str(PROJECT_DIR / "feature_units" / "feature_unit_registry.json"),
        help="Path to the feature unit registry JSON.",
    )
    parser.add_argument(
        "--matrix",
        default=str(PROJECT_DIR / "feature_units" / "feature_intervention_matrix.tsv"),
        help="Path to the feature intervention matrix TSV.",
    )
    parser.add_argument("--run-dir", required=True, help="Target run directory.")
    parser.add_argument(
        "--out-tsv",
        default="",
        help="Optional explicit output TSV path. Default: <run-dir>/analysis/feature_activation_report_v1.tsv",
    )
    return parser.parse_args()


def load_registry(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrix(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return {
            row["feature_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def is_lineage_child_path(run_dir: Path, path: Path) -> bool:
    rel_parts = path.relative_to(run_dir).parts
    return len(rel_parts) >= 2 and rel_parts[0] == "lineage" and rel_parts[1] == "children"


def find_run_files(run_dir: Path, name: str) -> list[Path]:
    matches = []
    for path in run_dir.rglob(name):
        if is_lineage_child_path(run_dir, path):
            continue
        matches.append(path)
    return sorted(matches, key=lambda p: (len(p.relative_to(run_dir).parts), str(p).lower()))


def first_run_file(run_dir: Path, name: str) -> Path | None:
    matches = find_run_files(run_dir, name)
    return matches[0] if matches else None


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR.parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def looks_like_numeric_label(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    digits = "".join(ch for ch in text if ch.isdigit())
    return bool(digits) and text.replace(".", "").replace(" ", "").isdigit()


def is_governed_numbered_doe_candidate_source(value: str) -> bool:
    source = str(value or "").strip()
    return source in {
        "doe_numbered_table_row",
        "doe_numbered_table_row_recovery",
    }


def detect_surfaces(run_dir: Path) -> dict[str, Any]:
    weak_labels_path = first_run_file(run_dir, "weak_labels__v7pilot_r3_fixparse.tsv")
    final_table_path = first_run_file(run_dir, "final_formulation_table_v1.tsv")
    decision_trace_path = first_run_file(run_dir, "final_output_decision_trace_v1.tsv")
    compare_counts_path = first_run_file(run_dir, "final_table_vs_gt_counts_by_doi.tsv")
    prompt_preview_path = first_run_file(run_dir, "stage2_prompt_preview_v1.tsv")
    candidate_segmentation_debug_path = first_run_file(run_dir, "candidate_segmentation_debug_v1.tsv")
    candidate_blocks_paths = find_run_files(run_dir, "candidate_blocks_v1.json")
    normalized_table_payload_paths = find_run_files(run_dir, "normalized_table_payloads_v1.json")
    evidence_blocks_paths = find_run_files(run_dir, "evidence_blocks_v1.json")
    run_context_path = run_dir / "RUN_CONTEXT.md"
    surfaces = {
        "weak_labels_path": weak_labels_path,
        "final_table_path": final_table_path,
        "decision_trace_path": decision_trace_path,
        "compare_counts_path": compare_counts_path,
        "prompt_preview_path": prompt_preview_path,
        "candidate_segmentation_debug_path": candidate_segmentation_debug_path,
        "candidate_blocks_paths": candidate_blocks_paths,
        "normalized_table_payload_paths": normalized_table_payload_paths,
        "evidence_blocks_paths": evidence_blocks_paths,
        "run_context_path": run_context_path if run_context_path.exists() else None,
        "stage2_active": weak_labels_path is not None,
        "stage2_child_validation": "validation" in run_dir.name.lower(),
        "stage3_resolution": first_run_file(run_dir, "formulation_relation_summary_v1.tsv") is not None,
        "stage4_eval": first_run_file(run_dir, "per_doi_formulation_instance_summary.tsv") is not None,
        "stage5_final": final_table_path is not None and decision_trace_path is not None,
        "benchmark_compare": compare_counts_path is not None,
        "run_context": run_context_path.exists(),
        "regression_guard": first_run_file(run_dir, "numbered_doe_regression_guard_v1.tsv") is not None,
    }
    if weak_labels_path is not None:
        weak_rows = read_tsv(weak_labels_path)
        surfaces["contains_ufxx"] = any(row.get("key") == "UFXX9WXE" for row in weak_rows)
    else:
        surfaces["contains_ufxx"] = False
    if prompt_preview_path is not None:
        preview_rows = read_tsv(prompt_preview_path)
        first_preview = preview_rows[0] if preview_rows else {}
        surfaces["stage2_input_packing_mode"] = first_preview.get("input_packing_mode", "")
        surfaces["stage2_prompt_layout_class"] = first_preview.get("prompt_layout_class", "")
        surfaces["stage2_ordered_block_order"] = first_preview.get("ordered_block_order", "")
        surfaces["stage2_prompt_preview_evidence_artifact_path"] = first_preview.get("evidence_artifact_path", "")
        surfaces["stage2_prompt_preview_technical_status"] = first_preview.get("technical_status_overall", "")
        surfaces["stage2_prompt_preview_design_status"] = first_preview.get("design_status_overall", "")
    else:
        surfaces["stage2_input_packing_mode"] = ""
        surfaces["stage2_prompt_layout_class"] = ""
        surfaces["stage2_ordered_block_order"] = ""
        surfaces["stage2_prompt_preview_evidence_artifact_path"] = ""
        surfaces["stage2_prompt_preview_technical_status"] = ""
        surfaces["stage2_prompt_preview_design_status"] = ""
    return surfaces


def derive_activation_state(expected_for_run: str, observed_activation: str, activation_status: str) -> str:
    if activation_status == "active":
        return "active"
    if activation_status == "unclear":
        return "processing_error"
    if expected_for_run == "yes":
        return "evidence_missing" if observed_activation == "missing" else "not_invoked"
    if activation_status == "not_expected" or observed_activation == "not_expected":
        return "not_invoked"
    return "not_invoked"


def evidence_artifacts_record_duplicate_suppression(surfaces: dict[str, Any]) -> bool:
    for path in surfaces.get("evidence_blocks_paths") or []:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("duplicate_table_suppression_events"), list) and payload.get("duplicate_table_suppression_events"):
            return True
    return False


def evidence_artifacts_record_primary_table_guardrail(surfaces: dict[str, Any]) -> bool:
    for path in surfaces.get("normalized_table_payload_paths") or []:
        try:
            payload = read_json(path)
        except Exception:
            continue
        for table_payload in payload.get("normalized_table_payloads") or [] if isinstance(payload, dict) else []:
            if not isinstance(table_payload, dict):
                continue
            if normalize_text(table_payload.get("primary_table_guardrail_status")) or normalize_text(table_payload.get("structure_first_guardrail_status")):
                return True
    for path in surfaces.get("candidate_blocks_paths") or []:
        try:
            payload = read_json(path)
        except Exception:
            continue
        snapshot = payload.get("feature_activation_snapshot", {}) if isinstance(payload, dict) else {}
        if isinstance(snapshot, dict) and snapshot.get("s2_2a_primary_table_guardrail"):
            return True
    return False


def expected_for_run(feature_id: str, matrix_row: dict[str, str], surfaces: dict[str, Any]) -> str:
    if feature_id in {"variant_aware_gt_authority_switch", "benchmark_doi_level_gt_count_audit"}:
        return "yes" if surfaces.get("benchmark_compare") else "no"
    if feature_id == "family_variant_retention_governance":
        return "yes" if surfaces.get("stage5_final") else "no"
    if feature_id == "s2_2_duplicate_table_suppression":
        return "yes" if surfaces.get("stage2_active") and evidence_artifacts_record_duplicate_suppression(surfaces) else "no"
    if feature_id == "s2_2a_primary_table_guardrail":
        return "yes" if surfaces.get("stage2_active") and evidence_artifacts_record_primary_table_guardrail(surfaces) else "no"
    if feature_id == "stage2_input_evidence_packing":
        if surfaces.get("stage2_input_packing_mode") == "ordered_blocks":
            return "yes"
        return "no"
    if feature_id in {
        "s2_2_evidence_artifact_contract",
        "s2_2_design_success_split",
        "s2_2_prompt_preview_derived_from_evidence_artifact",
        "s2_2_evidence_priority_selection",
        "s2_2_duplicate_table_suppression",
    }:
        return "yes" if surfaces.get("stage2_active") else "no"
    if feature_id == "s2_2a_table_authority_ranking":
        if surfaces.get("candidate_blocks_paths") or surfaces.get("normalized_table_payload_paths"):
            return "yes"
        return "no"
    if feature_id in {"s2_candidate_section_aware_split", "s2_candidate_noise_filtering"}:
        return "yes" if surfaces.get("stage2_active") else "no"
    if feature_id == "s2_candidate_table_isolation":
        if not surfaces.get("stage2_active"):
            return "no"
        for path in surfaces.get("candidate_blocks_paths") or []:
            try:
                payload = read_json(path)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            blocks = payload.get("candidate_blocks") or []
            if any(isinstance(block, dict) and block.get("candidate_type") == "table" for block in blocks):
                return "yes"
        return "no"
    if feature_id == "s2_2_doe_overlay_selection":
        if not surfaces.get("stage2_active"):
            return "no"
        for path in surfaces.get("evidence_blocks_paths") or []:
            try:
                payload = read_json(path)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("feature_activation_snapshot", {}).get("doe_pre_llm_detection"):
                return "yes"
        return "no"
    if feature_id in {
        "numbered_doe_row_enumeration_priority",
        "numbered_doe_regression_guard",
        "table_first_evidence_binding",
    } and not surfaces.get("contains_ufxx"):
        return "no"

    surface_columns = [
        "stage2_active",
        "stage2_child_validation",
        "stage3_resolution",
        "stage4_eval",
        "stage5_final",
        "benchmark_compare",
        "run_context",
        "regression_guard",
    ]
    for column in surface_columns:
        if matrix_row.get(column) == "required" and surfaces.get(column):
            return "yes"
    return "no"


def compute_activation_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    required_rows = [row for row in rows if row["expected_for_run"] == "yes"]
    missing_required = [row["feature_id"] for row in required_rows if row["activation_status"] == "missing"]
    unclear_required = [row["feature_id"] for row in required_rows if row["activation_status"] == "unclear"]
    if missing_required:
        gate = "fail"
    elif unclear_required:
        gate = "warn"
    else:
        gate = "pass"
    return {
        "required_feature_units": [row["feature_id"] for row in required_rows],
        "missing_required_feature_units": missing_required,
        "unclear_required_feature_units": unclear_required,
        "run_activation_gate": gate,
    }


def observe_numbered_doe_regression_guard(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    guard_path = first_run_file(run_dir, "numbered_doe_regression_guard_v1.tsv")
    if guard_path is None:
        return {
            "observed_activation": "missing",
            "activation_status": "missing",
            "evidence_path": "",
            "evidence_detail": "No numbered_doe_regression_guard_v1.tsv found in the run-local artifacts.",
            "notes": "The guard may exist in code but is not evidenced in this run.",
        }
    rows = read_tsv(guard_path)
    fail_count = sum(1 for row in rows if row.get("guard_status") == "fail")
    warn_count = sum(1 for row in rows if row.get("guard_status") == "warn")
    return {
        "observed_activation": "active",
        "activation_status": "active",
        "evidence_path": to_repo_rel(guard_path),
        "evidence_detail": f"guard_rows={len(rows)} fail={fail_count} warn={warn_count}",
        "notes": "Active because the run emitted a guard artifact.",
    }


def observe_variant_aware_gt_authority_switch(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    counts_files = find_run_files(run_dir, "final_table_vs_gt_counts_by_doi.tsv")
    accepted_authority_tokens = (
        "dev15_formulation_skeleton_review_v2_variantaware.xlsx",
        "data/cleaned/gt_authority/v1/dev15_layer1_gt_counts.tsv",
        "dev15_layer1_gt_counts.tsv",
    )
    for path in counts_files:
        rows = read_tsv(path)
        authority_values = [row.get("gt_authority_file", "") for row in rows]
        if any(
            any(token in authority_value for token in accepted_authority_tokens)
            for authority_value in authority_values
        ):
            observed = next(
                authority_value
                for authority_value in authority_values
                if any(token in authority_value for token in accepted_authority_tokens)
            )
            return {
                "observed_activation": "active",
                "activation_status": "active",
                "evidence_path": to_repo_rel(path),
                "evidence_detail": f"Compare artifact records accepted DEV15 GT authority: {observed}",
                "notes": "The feature is active only because the run-local compare artifact proves the accepted DEV15 GT authority was used.",
            }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": "",
        "evidence_detail": "No run-local compare artifact proved use of the accepted DEV15 variant-aware/frozen Layer1 GT authority.",
        "notes": "Code existence alone is not enough for activation.",
    }


def observe_numbered_doe_row_enumeration_priority(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    weak_labels_path = surfaces.get("weak_labels_path")
    if weak_labels_path is None:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No weak-label TSV found.",
            "notes": "Cannot inspect Stage2 activation without run-local weak labels.",
        }
    rows = [row for row in read_tsv(weak_labels_path) if row.get("key") == "UFXX9WXE"]
    if not rows:
        return {
            "observed_activation": "not_expected",
            "activation_status": "not_expected",
            "evidence_path": to_repo_rel(weak_labels_path),
            "evidence_detail": "UFXX9WXE is not present in this run's Stage2 weak labels.",
            "notes": "Current activation logic for this feature is grounded on UFXX9WXE-class detectable papers.",
        }
    doe_count = sum(1 for row in rows if is_governed_numbered_doe_candidate_source(row.get("candidate_source", "")))
    llm_numeric_count = sum(
        1 for row in rows
        if row.get("candidate_source") == "llm_extracted" and looks_like_numeric_label(row.get("raw_formulation_label", ""))
    )
    if doe_count > 0:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(weak_labels_path),
            "evidence_detail": f"UFXX9WXE governed_numbered_doe_rows={doe_count} llm_numeric_rows={llm_numeric_count}",
            "notes": "Active because governed structured DOE rows reached the run-local Stage2 candidate surface.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": to_repo_rel(weak_labels_path),
        "evidence_detail": f"UFXX9WXE rows={len(rows)} governed_numbered_doe_rows=0 llm_numeric_rows={llm_numeric_count}",
        "notes": "UFXX9WXE is in scope, but the run-local Stage2 artifact does not carry governed deterministic numbered DOE rows.",
    }


def observe_table_first_evidence_binding(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    weak_labels_path = surfaces.get("weak_labels_path")
    if weak_labels_path is None:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No weak-label TSV found.",
            "notes": "Cannot inspect table-first evidence anchors without run-local Stage2 rows.",
        }
    rows = [row for row in read_tsv(weak_labels_path) if row.get("key") == "UFXX9WXE"]
    table_rows = [
        row for row in rows
        if is_governed_numbered_doe_candidate_source(row.get("candidate_source", ""))
        and row.get("instance_evidence_region_type") == "table_row"
        and "numbered_doe_table" in row.get("evidence_section", "")
    ]
    if table_rows:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(weak_labels_path),
            "evidence_detail": f"UFXX9WXE table_row_anchors={len(table_rows)}",
            "notes": "Active because run-local Stage2 rows retain explicit table_row anchors.",
        }
    if rows:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": to_repo_rel(weak_labels_path),
            "evidence_detail": "UFXX9WXE rows are present, but no explicit numbered table_row anchors were found.",
            "notes": "Current artifacts do not prove table-first binding for this run.",
        }
    return {
        "observed_activation": "not_expected",
        "activation_status": "not_expected",
        "evidence_path": to_repo_rel(weak_labels_path),
        "evidence_detail": "UFXX9WXE is not present in this run's Stage2 weak labels.",
        "notes": "Current detection for this feature is grounded on UFXX9WXE-class structured DOE papers.",
    }


def observe_stage2_input_evidence_packing(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    prompt_preview_path = surfaces.get("prompt_preview_path")
    if prompt_preview_path is None:
        if surfaces.get("stage2_active"):
            return {
                "observed_activation": "missing",
                "activation_status": "missing",
                "evidence_path": "",
                "evidence_detail": "No stage2_prompt_preview_v1.tsv found in the run-local Stage2 artifacts.",
                "notes": "The live Stage2 run did not emit an input-assembly preview artifact.",
            }
        return {
            "observed_activation": "not_expected",
            "activation_status": "not_expected",
            "evidence_path": "",
            "evidence_detail": "Stage2 prompt preview is not expected because this run does not include active Stage2 extraction artifacts.",
            "notes": "This feature is only relevant to Stage2 live prompt assembly.",
        }
    rows = read_tsv(prompt_preview_path)
    if not rows:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": to_repo_rel(prompt_preview_path),
            "evidence_detail": "Prompt preview file exists but contains no rows.",
            "notes": "Treat empty prompt previews conservatively as a processing error.",
        }
    row = rows[0]
    layout_class = row.get("prompt_layout_class", "")
    input_mode = row.get("input_packing_mode", "")
    block_order = row.get("ordered_block_order", "")
    if input_mode != "ordered_blocks":
        return {
            "observed_activation": "not_expected",
            "activation_status": "not_expected",
            "evidence_path": to_repo_rel(prompt_preview_path),
            "evidence_detail": f"rows={len(rows)} prompt_layout_class={layout_class or 'unknown'} input_packing_mode={input_mode or 'off'}",
            "notes": "The run emitted prompt preview evidence, but the governed controlled-order execution path was not enabled.",
        }
    return {
        "observed_activation": "active",
        "activation_status": "active",
        "evidence_path": to_repo_rel(prompt_preview_path),
        "evidence_detail": f"rows={len(rows)} prompt_layout_class={layout_class or 'unknown'} input_packing_mode={input_mode or 'unknown'} block_order={block_order or 'unknown'}",
        "notes": "Active because the run emitted a prompt-preview artifact and the controlled ordering mode was enabled in the live Stage2 path.",
    }


def observe_s2_2_evidence_artifact_contract(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    evidence_paths = surfaces.get("evidence_blocks_paths") or []
    if not evidence_paths:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No evidence_blocks_v1.json artifacts found in the run-local Stage2 outputs.",
            "notes": "The maintained S2-2 contract requires persisted evidence blocks before live prompt assembly.",
        }
    required_top_level = {
        "paper_key",
        "source_clean_text_path",
        "source_manifest_path",
        "input_contract",
        "producer_script",
        "contract_version",
        "evidence_blocks",
        "coverage_summary",
        "feature_activation_snapshot",
        "technical_status",
        "design_status",
    }
    required_block_fields = {
        "block_id",
        "block_type",
        "source_type",
        "origin_locator",
        "selection_reason",
        "selection_feature",
        "rank_score",
        "order_index",
        "text_content",
    }
    valid_count = 0
    for path in evidence_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not required_top_level.issubset(payload.keys()):
            continue
        blocks = payload.get("evidence_blocks")
        if not isinstance(blocks, list) or not blocks:
            continue
        if not all(isinstance(block, dict) and required_block_fields.issubset(block.keys()) for block in blocks):
            continue
        valid_count += 1
    if valid_count == len(evidence_paths):
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(evidence_paths[0]),
            "evidence_detail": f"evidence_artifacts={len(evidence_paths)} valid_contract_artifacts={valid_count}",
            "notes": "Active because the run-local Stage2 path persisted canonical S2-2 evidence artifacts with the required contract fields.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(evidence_paths[0]),
        "evidence_detail": f"evidence_artifacts={len(evidence_paths)} valid_contract_artifacts={valid_count}",
        "notes": "Some evidence artifacts exist, but at least one did not expose the full governed S2-2 contract.",
    }


def observe_s2_candidate_section_aware_split(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    candidate_paths = surfaces.get("candidate_blocks_paths") or []
    if not candidate_paths:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No candidate_blocks_v1.json artifacts found for segmentation inspection.",
            "notes": "The explicit candidate-segmentation boundary is only auditable when the maintained Stage2 path persists candidate artifacts.",
        }
    valid_count = 0
    for path in candidate_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("segmentation_profile") != "section_aware_candidate_segmentation_v1":
            continue
        blocks = payload.get("candidate_blocks") or []
        if not isinstance(blocks, list) or not blocks:
            continue
        prose_with_structure = [
            block
            for block in blocks
            if isinstance(block, dict)
            and block.get("candidate_type") == "prose"
            and "segmentation_method" in block
            and "section_kind" in block
        ]
        if prose_with_structure:
            valid_count += 1
    if valid_count == len(candidate_paths):
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(candidate_paths[0]),
            "evidence_detail": f"candidate_artifacts={len(candidate_paths)} section_aware_artifacts={valid_count}",
            "notes": "Active because the run-local candidate artifacts record the maintained section-aware prose segmentation boundary.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(candidate_paths[0]),
        "evidence_detail": f"candidate_artifacts={len(candidate_paths)} section_aware_artifacts={valid_count}",
        "notes": "Candidate artifacts exist, but at least one does not expose the maintained section-aware segmentation fields cleanly.",
    }


def observe_s2_candidate_table_isolation(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    candidate_paths = surfaces.get("candidate_blocks_paths") or []
    if not candidate_paths:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No candidate_blocks_v1.json artifacts found for table-isolation inspection.",
            "notes": "Table-isolation activation is only auditable when the maintained candidate artifact exists.",
        }
    readable_count = 0
    isolated_count = 0
    for path in candidate_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        readable_count += 1
        snapshot = payload.get("feature_activation_snapshot", {})
        blocks = payload.get("candidate_blocks") or []
        table_blocks = [block for block in blocks if isinstance(block, dict) and block.get("candidate_type") == "table"]
        if snapshot.get("table_isolation") and table_blocks:
            isolated_count += 1
    if readable_count == 0:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No readable candidate artifacts found for table-isolation inspection.",
            "notes": "Could not inspect the candidate table-isolation surface.",
        }
    if isolated_count > 0:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(candidate_paths[0]),
            "evidence_detail": f"candidate_artifacts_with_table_isolation={isolated_count} readable_candidate_artifacts={readable_count}",
            "notes": "Active because the maintained candidate artifacts record isolated table candidates before selector prioritization.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": to_repo_rel(candidate_paths[0]),
        "evidence_detail": f"candidate_artifacts_with_table_isolation={isolated_count} readable_candidate_artifacts={readable_count}",
        "notes": "The maintained candidate artifacts were readable, but none recorded isolated table candidates.",
    }


def observe_s2_2a_table_authority_ranking(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    candidate_paths = surfaces.get("candidate_blocks_paths") or []
    normalized_paths = surfaces.get("normalized_table_payload_paths") or []
    if not candidate_paths or not normalized_paths:
        status = "missing" if surfaces.get("stage2_active") else "not_expected"
        return {
            "observed_activation": status,
            "activation_status": status,
            "evidence_path": "",
            "evidence_detail": "Candidate or normalized table payload artifacts are missing for table-authority ranking inspection.",
            "notes": "The S2-2a ranking feature is only auditable when both the candidate and execution-facing authority surfaces exist.",
        }
    readable_candidates = 0
    ranked_candidates = 0
    for path in candidate_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        readable_candidates += 1
        blocks = payload.get("candidate_blocks") or []
        table_blocks = [
            block for block in blocks
            if isinstance(block, dict) and block.get("candidate_type") == "table"
        ]
        if not table_blocks:
            continue
        snapshot = payload.get("feature_activation_snapshot", {})
        if not snapshot.get("table_authority_ranking"):
            continue
        if all(
            "authority_rank" in block and "authority_score" in block and "authority_tier" in block
            for block in table_blocks
        ):
            ranked_candidates += 1

    readable_normalized = 0
    ranked_normalized = 0
    for path in normalized_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        readable_normalized += 1
        records = payload.get("normalized_table_payloads") or []
        if not isinstance(records, list) or not records:
            continue
        if all(
            isinstance(record, dict)
            and "authority_rank" in record
            and "authority_score" in record
            and "authority_tier" in record
            for record in records
        ):
            ranked_normalized += 1

    if readable_candidates == 0 or readable_normalized == 0:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No readable candidate or normalized table payload artifacts were available for authority-ranking inspection.",
            "notes": "Could not inspect the maintained S2-2a table-authority ranking surface.",
        }
    if ranked_candidates > 0 and ranked_normalized > 0:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(normalized_paths[0]),
            "evidence_detail": (
                f"ranked_candidate_artifacts={ranked_candidates}/{readable_candidates} "
                f"ranked_normalized_payloads={ranked_normalized}/{readable_normalized}"
            ),
            "notes": "Active because the run-local S2-2a artifacts expose ranked table authority metadata on both the candidate surface and the preserved normalized table payload surface.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(normalized_paths[0]),
        "evidence_detail": (
            f"ranked_candidate_artifacts={ranked_candidates}/{readable_candidates} "
            f"ranked_normalized_payloads={ranked_normalized}/{readable_normalized}"
        ),
        "notes": "At least one maintained S2-2a artifact exists without the required authority-ranking metadata.",
    }


def observe_s2_candidate_noise_filtering(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    candidate_paths = surfaces.get("candidate_blocks_paths") or []
    debug_path = surfaces.get("candidate_segmentation_debug_path")
    if not candidate_paths:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No candidate_blocks_v1.json artifacts found for noise-filter inspection.",
            "notes": "The candidate noise-filtering feature is only auditable when the maintained candidate artifact exists.",
        }
    valid_count = 0
    for path in candidate_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        snapshot = payload.get("feature_activation_snapshot", {})
        if not snapshot.get("noise_filtering"):
            continue
        blocks = payload.get("candidate_blocks") or []
        if not isinstance(blocks, list) or not blocks:
            continue
        valid_count += 1
    if valid_count == len(candidate_paths):
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(debug_path) if debug_path is not None else to_repo_rel(candidate_paths[0]),
            "evidence_detail": f"candidate_artifacts={len(candidate_paths)} noise_filtered_artifacts={valid_count} debug_surface={'yes' if debug_path is not None else 'no'}",
            "notes": "Active because the maintained candidate artifacts record conservative noise filtering and the run may expose candidate-level debug rows.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(candidate_paths[0]),
        "evidence_detail": f"candidate_artifacts={len(candidate_paths)} noise_filtered_artifacts={valid_count} debug_surface={'yes' if debug_path is not None else 'no'}",
        "notes": "Candidate artifacts exist, but at least one did not record conservative noise-filtering activation cleanly.",
    }


def observe_s2_2_design_success_split(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    evidence_paths = surfaces.get("evidence_blocks_paths") or []
    if not evidence_paths:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No evidence_blocks_v1.json artifacts found for technical/design status inspection.",
            "notes": "The S2-2 status split is only auditable when the canonical evidence artifact exists.",
        }
    valid_count = 0
    for path in evidence_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        technical_status = payload.get("technical_status")
        design_status = payload.get("design_status")
        if not isinstance(technical_status, dict) or not isinstance(design_status, dict):
            continue
        if not {"artifact_readable", "required_top_level_fields_present", "required_block_fields_present", "non_empty_evidence_blocks", "overall"}.issubset(technical_status.keys()):
            continue
        if not {"input_contract_satisfied", "required_features_active", "overall"}.issubset(design_status.keys()):
            continue
        valid_count += 1
    if valid_count == len(evidence_paths):
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(evidence_paths[0]),
            "evidence_detail": f"evidence_artifacts={len(evidence_paths)} with_status_split={valid_count}",
            "notes": "Active because the canonical evidence artifacts distinguish technical completeness from design conformance.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(evidence_paths[0]),
        "evidence_detail": f"evidence_artifacts={len(evidence_paths)} with_status_split={valid_count}",
        "notes": "At least one artifact exists without the required technical/design status split.",
    }


def observe_s2_2_prompt_preview_derived_from_evidence_artifact(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    prompt_preview_path = surfaces.get("prompt_preview_path")
    if prompt_preview_path is None:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No stage2_prompt_preview_v1.tsv found in the run-local analysis directory.",
            "notes": "The derived observability contract requires the prompt preview to point back to canonical evidence artifacts.",
        }
    rows = read_tsv(prompt_preview_path)
    if not rows:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": to_repo_rel(prompt_preview_path),
            "evidence_detail": "Prompt preview file exists but contains no rows.",
            "notes": "Treat empty prompt previews conservatively as a processing error.",
        }
    linked_rows = 0
    for row in rows:
        rel = row.get("evidence_artifact_path", "").strip()
        if not rel:
            continue
        artifact_path = PROJECT_DIR.parent / rel
        if not artifact_path.exists():
            continue
        try:
            payload = read_json(artifact_path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if row.get("technical_status_overall", "").strip() != str(payload.get("technical_status", {}).get("overall", "")).strip():
            continue
        if row.get("design_status_overall", "").strip() != str(payload.get("design_status", {}).get("overall", "")).strip():
            continue
        linked_rows += 1
    if linked_rows == len(rows):
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(prompt_preview_path),
            "evidence_detail": f"prompt_preview_rows={len(rows)} linked_rows={linked_rows}",
            "notes": "Active because every prompt-preview row resolves to a canonical S2-2 evidence artifact with matching status summaries.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(prompt_preview_path),
        "evidence_detail": f"prompt_preview_rows={len(rows)} linked_rows={linked_rows}",
        "notes": "Prompt preview rows exist, but at least one row does not cleanly resolve back to the canonical evidence artifact.",
    }


def observe_s2_2_evidence_priority_selection(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    evidence_paths = surfaces.get("evidence_blocks_paths") or []
    if not evidence_paths:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No evidence_blocks_v1.json artifacts found for evidence-priority selector inspection.",
            "notes": "Evidence-driven S2-2 activation is only auditable when the canonical evidence artifact exists.",
        }
    valid_count = 0
    for path in evidence_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if normalize_text(payload.get("selection_mode")) != "evidence_priority_v1":
            continue
        blocks = payload.get("evidence_blocks") or []
        if not isinstance(blocks, list) or not blocks:
            continue
        if not any(isinstance(block, dict) and block.get("evidence_kind") for block in blocks):
            continue
        if not any(isinstance(block, dict) and block.get("char_count") for block in blocks):
            continue
        valid_count += 1
    if valid_count == len(evidence_paths):
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(evidence_paths[0]),
            "evidence_detail": f"evidence_artifacts={len(evidence_paths)} evidence_priority_contracts={valid_count}",
            "notes": "Active because the canonical evidence artifacts record evidence-driven selection mode, compact block contracts, and suppression-aware selection summaries.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(evidence_paths[0]),
        "evidence_detail": f"evidence_artifacts={len(evidence_paths)} evidence_priority_contracts={valid_count}",
        "notes": "At least one evidence artifact exists without the required evidence-priority selector fields.",
    }


def observe_s2_2_doe_overlay_selection(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    evidence_paths = surfaces.get("evidence_blocks_paths") or []
    if not evidence_paths:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No evidence_blocks_v1.json artifacts found for archetype-metadata inspection.",
            "notes": "Archetype detection is only auditable when the canonical evidence artifact exists.",
        }
    doe_candidates = 0
    doe_active = 0
    evidence_path = to_repo_rel(evidence_paths[0])
    for path in evidence_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        snapshot = payload.get("feature_activation_snapshot", {})
        if snapshot.get("doe_pre_llm_detection"):
            doe_candidates += 1
            if snapshot.get("archetype_detection_metadata_only"):
                doe_active += 1
                evidence_path = to_repo_rel(path)
    if doe_candidates == 0:
        return {
            "observed_activation": "not_expected",
            "activation_status": "not_expected",
            "evidence_path": evidence_path,
            "evidence_detail": "No DOE-like papers were detected in this run's S2-2 evidence artifacts.",
            "notes": "DOE-like signals are only expected when the maintained pre-LLM signals indicate a DOE or optimization paper.",
        }
    if doe_active == doe_candidates:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": evidence_path,
            "evidence_detail": f"doe_detected_artifacts={doe_candidates} metadata_only_archetype_artifacts={doe_active}",
            "notes": "Active because every DOE-detected evidence artifact records metadata-only archetype detection rather than a selector overlay.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": evidence_path,
        "evidence_detail": f"doe_detected_artifacts={doe_candidates} metadata_only_archetype_artifacts={doe_active}",
        "notes": "At least one DOE-detected artifact did not record metadata-only archetype detection cleanly.",
    }


def observe_s2_2_duplicate_table_suppression(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    evidence_paths = surfaces.get("evidence_blocks_paths") or []
    if not evidence_paths:
        return {
            "observed_activation": "missing" if surfaces.get("stage2_active") else "not_expected",
            "activation_status": "missing" if surfaces.get("stage2_active") else "not_expected",
            "evidence_path": "",
            "evidence_detail": "No evidence_blocks_v1.json artifacts found for duplicate-table suppression inspection.",
            "notes": "Duplicate-table suppression is only auditable when the canonical evidence artifact exists.",
        }
    active_count = 0
    candidate_count = 0
    evidence_path = to_repo_rel(evidence_paths[0])
    for path in evidence_paths:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        candidate_count += 1
        events = payload.get("duplicate_table_suppression_events") or []
        if isinstance(events, list) and events:
            active_count += 1
            evidence_path = to_repo_rel(path)
    if candidate_count == 0:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No readable evidence artifacts found for duplicate-table suppression inspection.",
            "notes": "Could not inspect the duplicate-table suppression surface.",
        }
    if active_count > 0:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": evidence_path,
            "evidence_detail": f"artifacts_with_duplicate_suppression={active_count} readable_artifacts={candidate_count}",
            "notes": "Active because at least one canonical evidence artifact recorded suppressed duplicate tables.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": to_repo_rel(evidence_paths[0]),
        "evidence_detail": f"artifacts_with_duplicate_suppression={active_count} readable_artifacts={candidate_count}",
        "notes": "Role-aware selection was inspected, but this run did not record any duplicate-table suppression events.",
    }


def observe_family_variant_retention_governance(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    trace_path = surfaces.get("decision_trace_path")
    if trace_path is None:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No final_output_decision_trace_v1.tsv found.",
            "notes": "Cannot inspect Stage5 governance activation without a run-local decision trace.",
        }
    rows = read_tsv(trace_path)
    variant_rows = [row for row in rows if row.get("confidence_or_rule_scope") == "phase1_variant_governance"]
    if variant_rows:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(trace_path),
            "evidence_detail": f"phase1_variant_governance_rows={len(variant_rows)}",
            "notes": "Active because the run-local Stage5 trace records variant-governance intervention.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": to_repo_rel(trace_path),
        "evidence_detail": "No phase1_variant_governance rows were found in the run-local decision trace.",
        "notes": "Child replays do not count as activation for the parent run.",
    }


def observe_feature_unit_governance_layer(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    report_path = run_dir / "analysis" / "feature_activation_report_v1.tsv"
    registry_path = PROJECT_DIR / "feature_units" / "feature_unit_registry.json"
    matrix_path = PROJECT_DIR / "feature_units" / "feature_intervention_matrix.tsv"
    if report_path.exists() and registry_path.exists() and matrix_path.exists():
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(report_path),
            "evidence_detail": (
                "Run-local activation report exists and the project-level registry plus matrix are available."
            ),
            "notes": "Active because this run has already materialized feature governance into a report.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": "",
        "evidence_detail": "Run-local activation report or project-level registry/matrix is missing.",
        "notes": "The governance layer only counts as active when the run materializes it into local evidence.",
    }


def observe_run_context_feature_activation_integration(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    run_context_path = surfaces.get("run_context_path")
    if run_context_path is None:
        return {
            "observed_activation": "missing",
            "activation_status": "missing",
            "evidence_path": "",
            "evidence_detail": "RUN_CONTEXT.md was not found for this run.",
            "notes": "Cannot prove run-context activation integration without RUN_CONTEXT.md.",
        }
    text = run_context_path.read_text(encoding="utf-8")
    if "## Feature Unit Activation" in text and "run_activation_gate" in text and "feature_activation_report_path" in text:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(run_context_path),
            "evidence_detail": "RUN_CONTEXT.md contains the generated Feature Unit Activation section.",
            "notes": "Active because the run metadata includes feature activation status and gate details.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": to_repo_rel(run_context_path),
        "evidence_detail": "RUN_CONTEXT.md exists but does not contain the generated Feature Unit Activation section.",
        "notes": "The updater must be run for this feature to count as active.",
    }


def observe_doi_level_gt_vs_pred_count_audit(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    compare_counts_path = surfaces.get("compare_counts_path")
    if compare_counts_path is None:
        return {
            "observed_activation": "missing",
            "activation_status": "missing",
            "evidence_path": "",
            "evidence_detail": "No final_table_vs_gt_counts_by_doi.tsv found.",
            "notes": "The DOI-level count audit feature requires a run-local compare artifact.",
        }
    rows = read_tsv(compare_counts_path)
    required_columns = {"doi", "gt_count", "pred_count", "delta_count", "count_status"}
    observed_columns = set(rows[0].keys()) if rows else set()
    if rows and required_columns.issubset(observed_columns):
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(compare_counts_path),
            "evidence_detail": f"rows={len(rows)} required_columns_present=yes",
            "notes": "Active because the run emitted the DOI-level GT versus prediction count audit table.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(compare_counts_path),
        "evidence_detail": "Counts-by-doi artifact exists but did not expose the full required schema.",
        "notes": "Keep this conservative until the compare artifact clearly exposes the audit columns.",
    }


OBSERVERS = {
    "numbered_doe_regression_guard": observe_numbered_doe_regression_guard,
    "variant_aware_gt_authority_switch": observe_variant_aware_gt_authority_switch,
    "numbered_doe_row_enumeration_priority": observe_numbered_doe_row_enumeration_priority,
    "table_first_evidence_binding": observe_table_first_evidence_binding,
    "stage2_input_evidence_packing": observe_stage2_input_evidence_packing,
    "s2_candidate_section_aware_split": observe_s2_candidate_section_aware_split,
    "s2_candidate_table_isolation": observe_s2_candidate_table_isolation,
    "s2_2a_table_authority_ranking": observe_s2_2a_table_authority_ranking,
    "s2_candidate_noise_filtering": observe_s2_candidate_noise_filtering,
    "s2_2_evidence_artifact_contract": observe_s2_2_evidence_artifact_contract,
    "s2_2_design_success_split": observe_s2_2_design_success_split,
    "s2_2_prompt_preview_derived_from_evidence_artifact": observe_s2_2_prompt_preview_derived_from_evidence_artifact,
    "s2_2_evidence_priority_selection": observe_s2_2_evidence_priority_selection,
    "s2_2_doe_overlay_selection": observe_s2_2_doe_overlay_selection,
    "s2_2_duplicate_table_suppression": observe_s2_2_duplicate_table_suppression,
    "family_variant_retention_governance": observe_family_variant_retention_governance,
    "feature_unit_governance_layer": observe_feature_unit_governance_layer,
    "run_context_feature_activation_integration": observe_run_context_feature_activation_integration,
    "doi_level_gt_vs_pred_count_audit": observe_doi_level_gt_vs_pred_count_audit,
}


def build_report_rows(
    *,
    registry: list[dict[str, Any]],
    matrix: dict[str, dict[str, str]],
    run_dir: Path,
) -> list[dict[str, str]]:
    surfaces = detect_surfaces(run_dir)
    rows: list[dict[str, str]] = []
    for feature in registry:
        feature_id = feature["feature_id"]
        matrix_row = matrix[feature_id]
        expected = expected_for_run(feature_id, matrix_row, surfaces)
        observer = OBSERVERS.get(feature_id)
        if observer is None:
            observed = {
                "observed_activation": "unclear",
                "activation_status": "unclear",
                "evidence_path": "",
                "evidence_detail": "No observer implemented for this feature.",
                "notes": "Add a deterministic artifact-based observer before relying on this feature in reports.",
            }
        else:
            observed = observer(run_dir, surfaces)
        if expected == "no" and observed["activation_status"] not in {"active"}:
            observed["activation_status"] = "not_expected"
            observed["observed_activation"] = "not_expected"
        rows.append(
            {
                "feature_id": feature_id,
                "expected_for_run": expected,
                "observed_activation": observed["observed_activation"],
                "activation_status": observed["activation_status"],
                "activation_state": derive_activation_state(
                    expected, observed["observed_activation"], observed["activation_status"]
                ),
                "evidence_path": observed["evidence_path"],
                "evidence_detail": observed["evidence_detail"],
                "notes": observed["notes"],
            }
        )
    return rows


def write_report_tsv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature_id",
        "expected_for_run",
        "observed_activation",
        "activation_status",
        "activation_state",
        "evidence_path",
        "evidence_detail",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    registry = load_registry(Path(args.registry))
    matrix = load_matrix(Path(args.matrix))
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_path = Path(args.out_tsv) if args.out_tsv else run_dir / "analysis" / "feature_activation_report_v1.tsv"
    rows = build_report_rows(registry=registry, matrix=matrix, run_dir=run_dir)
    write_report_tsv(output_path, rows)
    print(str(output_path))


if __name__ == "__main__":
    main()
