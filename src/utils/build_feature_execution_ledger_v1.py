
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import DOCS_DIR, PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DOCS_DIR, PROJECT_ROOT

STATE_EXPECTED_ACTIVE = "expected_active"
STATE_ACTIVE_OBSERVED = "active_observed"
STATE_ACTIVE_INFERRED = "active_inferred_from_upstream"
STATE_NOT_APPLICABLE = "not_applicable_for_run_scope"
STATE_NOT_REACHABLE = "not_reachable_due_to_resume_boundary"
STATE_EXPECTED_NOT_OBSERVED = "expected_but_not_observed"
STATE_INTENTIONALLY_DISABLED = "intentionally_disabled"
STATE_REPLAY_HIDDEN = "replay_hidden"
STATE_UNKNOWN = "unknown_needs_review"

STATE_COLUMNS = [
    "run_id", "feature_key", "feature_name", "expected_state_for_this_run",
    "actual_state_for_this_run", "state_reason", "evidence_paths",
    "evidence_summary", "mismatch_flag", "mismatch_type", "severity",
    "entrypoint_scope", "source_mode", "resume_boundary",
    "upstream_applied_before_run", "visible_in_current_run",
    "replay_visibility_limit", "recommended_debug_order", "notes",
]
SCHEMA_COLUMNS = [
    "feature_key", "feature_name", "architecture_layer", "stage_scope",
    "feature_type", "current_status", "default_expectation", "activation_mode",
    "expected_run_scopes", "lawful_resume_behavior",
    "upstream_persistence_behavior", "observability_requirements",
    "expected_artifacts_if_active", "expected_artifacts_if_upstream_applied",
    "cannot_be_reobserved_after_resume", "notes",
    "adjudicated_feature_class", "adjudicated_default_expectation",
    "belongs_to_feature_unit_system", "input_contract_feature",
    "mismatch_should_block_debugging", "mismatch_severity_tier",
]
BACKFILL_SUMMARY_COLUMNS = [
    "run_id", "run_dir", "entrypoint_scope", "source_mode", "resume_boundary",
    "ledger_rows", "active_observed", "active_inferred_from_upstream",
    "expected_but_not_observed", "intentionally_disabled", "replay_hidden",
    "not_reachable_due_to_resume_boundary", "not_applicable_for_run_scope",
    "unknown_needs_review", "mismatch_rows", "highest_severity",
]

@dataclass(frozen=True)
class RunProfile:
    run_id: str
    run_dir: Path
    run_type: str
    source_mode: str
    entrypoint_scope: str
    resume_boundary: str
    execution_order: list[str]
    start_stage: int
    stage2_live: bool
    stage2_replay: bool
    stage3_local: bool
    stage5_final_local: bool
    compare_local: bool
    full_pipeline_local: bool
    diagnostic_only: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-run feature execution ledgers.")
    parser.add_argument("--inventory", default=str(DOCS_DIR / "feature_governance" / "feature_master_inventory_v1.tsv"))
    parser.add_argument("--schema-out", default=str(DOCS_DIR / "feature_governance" / "feature_applicability_schema_v1.tsv"))
    parser.add_argument("--backfill-summary-out", default=str(DOCS_DIR / "feature_governance" / "feature_execution_ledger_backfill_summary_v1.json"))
    parser.add_argument("--backfill-summary-tsv", default=str(DOCS_DIR / "feature_governance" / "feature_execution_ledger_backfill_index_v1.tsv"))
    parser.add_argument("--system-report-out", default=str(DOCS_DIR / "feature_governance" / "feature_execution_ledger_system_report_v1.md"))
    parser.add_argument("--guide-out", default=str(DOCS_DIR / "feature_governance" / "how_to_use_feature_execution_ledger_v1.md"))
    parser.add_argument("--run-dir", action="append", dest="run_dirs", default=[])
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_run_context_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    patterns = [
        r"^- ([A-Za-z0-9_]+):\s*`([^`]*)`",
        r"^- `([A-Za-z0-9_]+)`: `([^`]*)`",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.MULTILINE):
            values[match.group(1)] = match.group(2).strip()
    return values


def parse_execution_order(text: str) -> list[str]:
    order: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\d+\.\s+(.*)$", line.strip())
        if not match:
            continue
        content = match.group(1)
        order.extend(re.findall(r"`([^`]*src/[^`]*)`", content))
        if not order:
            inline = re.search(r"(src/[A-Za-z0-9_./-]+\.py)", content)
            if inline:
                order.append(inline.group(1))
    return order


def first_existing(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    return None


def find_run_file(run_dir: Path, name: str) -> Path | None:
    matches = sorted(run_dir.rglob(name), key=lambda p: (len(p.relative_to(run_dir).parts), str(p).lower()))
    return matches[0] if matches else None


def parse_boolish(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def infer_run_profile(run_dir: Path) -> tuple[RunProfile, dict[str, str], dict[str, Path | None]]:
    ctx = run_dir / "RUN_CONTEXT.md"
    text = read_text(ctx)
    values = parse_run_context_values(text)
    order = parse_execution_order(text)
    run_id = values.get("run_id") or run_dir.name
    run_type_match = re.search(r"## 2\. Run Type\s*\n`([^`]*)`", text) or re.search(r"## 2\. Run type\s*\n\n- `([^`]*)`", text)
    run_type = run_type_match.group(1).strip() if run_type_match else values.get("run_type", "unknown")
    source_mode = values.get("source_mode") or values.get("source_resolution") or values.get("replay_mode") or "unknown"
    artifacts = {
        "run_context": ctx,
        "prompt_preview": first_existing(run_dir, ["analysis/stage2_prompt_preview_v1.tsv"]),
        "table_selection_debug": first_existing(run_dir, ["analysis/table_selection_debug_v1.json"]),
        "feature_activation": first_existing(run_dir, ["analysis/feature_activation_report_v1.tsv"]),
        "contract_report": first_existing(run_dir, ["analysis/stage2_semantic_authority_contract_report_v1.json"]),
        "compatibility_summary": find_run_file(run_dir, "compatibility_projection_summary_v1.json"),
        "semantic_objects": find_run_file(run_dir, "semantic_stage2_v2_objects.jsonl"),
        "relation_records": find_run_file(run_dir, "formulation_relation_records_v1.tsv"),
        "resolved_fields": find_run_file(run_dir, "resolved_relation_fields_v1.tsv"),
        "final_table": find_run_file(run_dir, "final_formulation_table_v1.tsv"),
        "decision_trace": find_run_file(run_dir, "final_output_decision_trace_v1.tsv"),
        "compare_counts": find_run_file(run_dir, "final_table_vs_gt_counts_by_doi.tsv"),
        "audit_ready": find_run_file(run_dir, "final_formulation_table_audit_ready_v1.tsv"),
        "risk_assessment": find_run_file(run_dir, "paper_risk_assessment.tsv"),
        "raw_responses_dir": Path(values["legacy_raw_responses_dir"]) if values.get("legacy_raw_responses_dir") else None,
    }
    stage2_local = any("run_stage2_composite_v1.py" in step or "extract_semantic_stage2_objects_v2.py" in step for step in order)
    stage3_local = any("build_formulation_relation_artifacts_v1.py" in step for step in order) or artifacts["relation_records"] is not None
    stage5_final_local = any("build_minimal_final_output_v1.py" in step for step in order)
    compare_local = any("compare_final_table_to_gt_v1.py" in step for step in order) or artifacts["compare_counts"] is not None
    stage2_replay = stage2_local and source_mode == "legacy_llm_replay"
    stage2_live = stage2_local and not stage2_replay
    full_pipeline_local = stage2_local and stage3_local and stage5_final_local
    diagnostic_only = "diagnostic-only" in text.lower() or "diagnostic" in run_type.lower()
    if stage2_local:
        start_stage = 2
        scope = "stage2_replay" if stage2_replay else ("full_pipeline_live" if full_pipeline_local else "stage2_live")
    elif compare_local:
        start_stage = 6
        scope = "compare_terminal"
    elif stage3_local:
        start_stage = 3
        scope = "stage3_resume"
    elif stage5_final_local:
        start_stage = 5
        scope = "stage5_final_resume"
    else:
        start_stage = 0
        scope = "unknown_needs_review"
    profile = RunProfile(run_id, run_dir, run_type, source_mode, scope, values.get("boundary_class", "unknown"), order, start_stage, stage2_live, stage2_replay, stage3_local, stage5_final_local, compare_local, full_pipeline_local, diagnostic_only)
    return profile, values, artifacts


def feature_category(feature_key: str) -> str:
    if feature_key in {"stage2_input_evidence_packing_ordered_blocks", "stage2_prompt_preview_observability", "stage2_table_summary_mode", "stage2_summary_first_column_enhancement", "stage2_default_raw_prefix_then_table_excerpts_layout", "stage2_json_sanitation_path1"}:
        return "stage2_live_visibility"
    if feature_key in {"stage2_composite_governed_entrypoint", "stage2_llm_first_composite_authority", "stage2_semantic_authority_contract_validator", "stage2_completed_stage2_only_resume_boundary", "stage2_compatibility_projection_completion", "stage2_non_doe_table_row_expansion", "stage2_sequential_optimization_interpreter", "stage2_doe_expansion_with_llm_scope", "stage2_partial_selection_marker_preservation", "stage2_partial_inheritance_marker_preservation", "stage2_execution_ready_only_downstream_handshake"}:
        return "stage2_completed"
    if feature_key == "stage2_livev2_raw_response_rehydration":
        return "stage2_replay"
    if feature_key in {"run_feature_activation_report", "run_context_feature_activation_section", "boundary_governance_metadata"}:
        return "run_metadata"
    if feature_key in {"active_data_source_authority_resolution", "benchmark_doi_level_gt_count_audit", "variant_aware_gt_authority_switch"}:
        return "compare"
    if feature_key in {"family_variant_retention_governance"}:
        return "stage5_final"
    if feature_key in {"stage5_audit_ready_export", "stage5_layer2_risk_assessment"}:
        return "stage5_review"
    if feature_key == "memory_bootstrap_debug_surface":
        return "governance_support"
    if feature_key in {"stage2_5_shadow_evidence_pack_builder", "legacy_table_first_block_packing_fixparse", "legacy_synthesis_method_materials_procurement_ordering", "legacy_table_heavy_row_enumeration_prompt_hint", "deterministic_semantic_emitter_fallback_mode"}:
        return "historical_or_fallback"
    return "review_or_gap"


def derive_schema_row(feature: dict[str, str], existing: dict[str, str] | None = None) -> dict[str, str]:
    existing = existing or {}
    category = feature_category(feature["feature_key"])
    expected_scopes = {"stage2_live_visibility": "stage2_live,stage2_replay,full_pipeline_local", "stage2_completed": "stage2_live,stage2_replay,full_pipeline_local", "stage2_replay": "stage2_replay", "run_metadata": "all_runs", "compare": "compare_local,benchmark_terminal,full_pipeline_local", "stage5_final": "stage5_final_local,full_pipeline_local,compare_local", "stage5_review": "stage5_review_local", "governance_support": "governance_debug_only", "historical_or_fallback": "historical_only", "review_or_gap": "review_needed_only"}[category]
    resume_behavior = {"stage2_live_visibility": "not_reachable_after_stage2_boundary", "stage2_completed": "inferred_from_completed_stage2_input_when_resumed_downstream", "stage2_replay": "replay_only_then_inferred_from_completed_stage2_input", "run_metadata": "re-emitted_per_run", "compare": "compare_local_only_or_terminal_resume", "stage5_final": "inferred_from_upstream_final_table_or_trace_after_resume", "stage5_review": "review_workflow_only", "governance_support": "manual_debugging_only", "historical_or_fallback": "not_expected_in_current_mainline_resume", "review_or_gap": "review_needed"}[category]
    persistence = {"stage2_live_visibility": "prompt_build_visibility_does_not_persist_downstream", "stage2_completed": "completed_stage2_outputs_persist", "stage2_replay": "completed_stage2_outputs_persist_but_prompt_visibility_may_not", "run_metadata": "run_local_only", "compare": "compare_artifacts_persist", "stage5_final": "final_table_and_trace_persist", "stage5_review": "review_exports_persist", "governance_support": "manual_only", "historical_or_fallback": "historical_only", "review_or_gap": "varies_review_needed"}[category]
    observability = {"stage2_live_visibility": "RUN_CONTEXT stage2 settings plus prompt preview or parse audit", "stage2_completed": "completed Stage2 artifacts and compatibility summary", "stage2_replay": "RUN_CONTEXT source_mode=legacy_llm_replay and legacy_raw_responses_dir", "run_metadata": "RUN_CONTEXT and feature activation report", "compare": "compare outputs and GT authority paths", "stage5_final": "final table and decision trace", "stage5_review": "review export artifacts", "governance_support": "explicit memory-query or audit evidence", "historical_or_fallback": "historical docs or explicit legacy execution", "review_or_gap": "manual audit evidence"}[category]
    active_artifacts = {"stage2_live_visibility": "analysis/stage2_prompt_preview_v1.tsv; RUN_CONTEXT.md", "stage2_completed": "semantic_to_widerow_adapter/compatibility_projection_summary_v1.json; weak_labels__v7pilot_r3_fixparse.tsv", "stage2_replay": "RUN_CONTEXT.md; semantic_to_widerow_adapter/*", "run_metadata": "RUN_CONTEXT.md; analysis/feature_activation_report_v1.tsv", "compare": "final_table_vs_gt_counts_by_doi.tsv; RUN_CONTEXT.md", "stage5_final": "final_formulation_table_v1.tsv; final_output_decision_trace_v1.tsv", "stage5_review": "final_formulation_table_audit_ready_v1.tsv; paper_risk_assessment.tsv", "governance_support": "memory query transcript or audit notes", "historical_or_fallback": "explicit legacy/fallback run artifacts", "review_or_gap": "manual audit artifacts"}[category]
    upstream_artifacts = {"stage2_live_visibility": "parent live stage2_prompt_preview_v1.tsv only if traced", "stage2_completed": "upstream weak_labels__v7pilot_r3_fixparse.tsv or compatibility summary", "stage2_replay": "upstream raw responses and parent live prompt preview", "run_metadata": "none", "compare": "upstream final_formulation_table_v1.tsv and compare outputs", "stage5_final": "upstream final_formulation_table_v1.tsv and final_output_decision_trace_v1.tsv", "stage5_review": "upstream review exports", "governance_support": "none", "historical_or_fallback": "historical docs", "review_or_gap": "manual audit notes"}[category]
    row = {key: feature[key] for key in ["feature_key", "feature_name", "architecture_layer", "stage_scope", "feature_type", "current_status", "default_expectation", "activation_mode"]}
    row.update({"expected_run_scopes": expected_scopes, "lawful_resume_behavior": resume_behavior, "upstream_persistence_behavior": persistence, "observability_requirements": observability, "expected_artifacts_if_active": active_artifacts, "expected_artifacts_if_upstream_applied": upstream_artifacts, "cannot_be_reobserved_after_resume": "true" if category in {"stage2_live_visibility", "stage2_replay"} else "false", "notes": feature.get("notes", "")})
    row["adjudicated_feature_class"] = feature.get("adjudicated_feature_class", existing.get("adjudicated_feature_class", ""))
    row["adjudicated_default_expectation"] = existing.get("adjudicated_default_expectation", "")
    row["belongs_to_feature_unit_system"] = existing.get("belongs_to_feature_unit_system", "")
    row["input_contract_feature"] = existing.get("input_contract_feature", "")
    row["mismatch_should_block_debugging"] = existing.get("mismatch_should_block_debugging", "")
    row["mismatch_severity_tier"] = existing.get("mismatch_severity_tier", "")
    return row

def load_compatibility_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(read_text(path))


def load_prompt_preview(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows = read_tsv(path)
    return rows[0] if rows else {}


def load_feature_activation(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row["feature_id"]: row for row in read_tsv(path)}


def load_table_selection_debug(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(read_text(path))
    if isinstance(payload, list) and payload:
        first = payload[0]
        return first if isinstance(first, dict) else {}
    return payload if isinstance(payload, dict) else {}


def find_parent_live_prompt_preview(raw_responses_dir: Path | None) -> Path | None:
    if raw_responses_dir is None:
        return None
    preview = raw_responses_dir.parent.parent / "analysis" / "stage2_prompt_preview_v1.tsv"
    return preview if preview.exists() else None


def build_upstream_trace(profile: RunProfile, values: dict[str, str], artifacts: dict[str, Path | None]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    tags: list[str] = []
    if not profile.stage2_live and not profile.stage2_replay and values.get("weak_labels_tsv"):
        tags.extend(["stage2_composite_governed_entrypoint", "stage2_compatibility_projection_completion", "stage2_execution_ready_only_downstream_handshake"])
        items.append({"stage": "stage2", "status": "already_applied_upstream", "reason": "Run starts from completed Stage2 weak-label artifacts.", "evidence_path": values["weak_labels_tsv"]})
    if not profile.stage3_local and values.get("relation_records_tsv"):
        tags.append("stage3_relation_materialization")
        items.append({"stage": "stage3", "status": "already_applied_upstream", "reason": "Run starts from Stage3 relation artifacts.", "evidence_path": values["relation_records_tsv"]})
    if not profile.stage5_final_local and values.get("final_table_tsv"):
        tags.extend(["family_variant_retention_governance"])
        items.append({"stage": "stage5_final", "status": "already_applied_upstream", "reason": "Run starts from a completed Stage5 final table.", "evidence_path": values["final_table_tsv"]})
    if profile.stage2_replay:
        parent_preview = find_parent_live_prompt_preview(artifacts.get("raw_responses_dir"))
        items.append({"stage": "stage2_prompt_assembly", "status": "replay_hidden_upstream", "reason": "Current run replays saved raw responses; original live prompt construction happened upstream.", "evidence_path": values.get("legacy_raw_responses_dir", ""), "parent_prompt_preview_path": repo_rel(parent_preview) if parent_preview else ""})
    return {"run_id": profile.run_id, "run_dir": repo_rel(profile.run_dir), "entrypoint_scope": profile.entrypoint_scope, "source_mode": profile.source_mode, "resume_boundary": profile.resume_boundary, "upstream_features_or_stages": sorted(dict.fromkeys(tags)), "transformations_before_current_entrypoint": items}


def expected_state(feature: dict[str, str], profile: RunProfile, values: dict[str, str], schema_row: dict[str, str]) -> str:
    key = feature["feature_key"]
    status = feature["current_status"]
    category = feature_category(key)
    adjudicated_class = schema_row.get("adjudicated_feature_class", "").strip()
    if status in {"historical_reference", "experimental_nonmainline"}:
        return STATE_NOT_APPLICABLE
    if status == "review_needed":
        return STATE_UNKNOWN
    if adjudicated_class == "core_input_contract":
        if profile.stage2_live or profile.full_pipeline_local:
            return STATE_EXPECTED_ACTIVE
        if profile.stage2_replay or profile.start_stage > 2:
            return STATE_NOT_REACHABLE
        return STATE_NOT_APPLICABLE
    if category == "governance_support" or category == "stage5_review":
        return STATE_NOT_APPLICABLE
    if category == "compare":
        return STATE_EXPECTED_ACTIVE if profile.compare_local else STATE_NOT_APPLICABLE
    if category == "stage5_final":
        if profile.stage5_final_local or profile.full_pipeline_local:
            return STATE_EXPECTED_ACTIVE
        return STATE_NOT_REACHABLE if profile.start_stage > 5 else STATE_NOT_APPLICABLE
    if category == "stage2_replay":
        return STATE_EXPECTED_ACTIVE if profile.stage2_replay else STATE_NOT_APPLICABLE
    if category == "stage2_completed":
        if profile.stage2_live or profile.stage2_replay or profile.full_pipeline_local:
            return STATE_EXPECTED_ACTIVE
        return STATE_NOT_REACHABLE if profile.start_stage > 2 else STATE_NOT_APPLICABLE
    if category == "stage2_live_visibility":
        if profile.stage2_live or profile.stage2_replay or profile.full_pipeline_local:
            if key == "stage2_input_evidence_packing_ordered_blocks":
                return STATE_EXPECTED_ACTIVE if values.get("stage2_input_evidence_packing_mode") == "ordered_blocks" else STATE_INTENTIONALLY_DISABLED
            if key == "stage2_table_summary_mode":
                mode = values.get("stage2_table_mode", "summary" if "summary" in values.get("summary_table_mode_activation", "") else "")
                return STATE_EXPECTED_ACTIVE if mode == "summary" else STATE_INTENTIONALLY_DISABLED
            if key == "stage2_summary_first_column_enhancement":
                return STATE_EXPECTED_ACTIVE if parse_boolish(values.get("stage2_table_summary_first_column_enhancement", "")) else STATE_INTENTIONALLY_DISABLED
            return STATE_EXPECTED_ACTIVE
        return STATE_NOT_REACHABLE if profile.start_stage > 2 else STATE_NOT_APPLICABLE
    return STATE_EXPECTED_ACTIVE if category == "run_metadata" else STATE_UNKNOWN


def actual_state(feature: dict[str, str], profile: RunProfile, values: dict[str, str], artifacts: dict[str, Path | None], upstream: dict[str, Any], compatibility: dict[str, Any], preview: dict[str, str], activation: dict[str, dict[str, str]], schema_row: dict[str, str], table_selection_debug: dict[str, Any]) -> tuple[str, str, list[Path | None]]:
    key = feature["feature_key"]
    expected = expected_state(feature, profile, values, schema_row)
    if expected == STATE_INTENTIONALLY_DISABLED:
        return STATE_INTENTIONALLY_DISABLED, "Feature is conditional and was left off in this run.", [artifacts.get("run_context"), artifacts.get("prompt_preview")]
    if expected == STATE_NOT_APPLICABLE:
        return STATE_NOT_APPLICABLE, "Feature is outside the local run scope.", []

    if key == "stage2_composite_governed_entrypoint":
        if profile.stage2_live or profile.stage2_replay:
            return STATE_ACTIVE_OBSERVED, "Run executed the governed Stage2 composite entrypoint locally.", [artifacts["run_context"]]
        return STATE_ACTIVE_INFERRED, "Completed Stage2 artifacts were already supplied before this run started.", [artifacts["run_context"]]
    if key == "stage2_llm_first_composite_authority":
        if profile.stage2_live or profile.stage2_replay:
            if values.get("stage2_semantic_source_mode") == "llm_first_composite":
                return STATE_ACTIVE_OBSERVED, "RUN_CONTEXT records llm_first_composite for local Stage2 execution.", [artifacts["run_context"]]
        if profile.start_stage > 2:
            return STATE_ACTIVE_INFERRED, "Downstream run starts from a completed Stage2 artifact presumed to carry llm-first composite authority.", [artifacts["run_context"]]
    if key == "stage2_semantic_authority_contract_validator":
        if artifacts.get("contract_report") is not None:
            return STATE_ACTIVE_OBSERVED, "Run emitted the Stage2 semantic authority contract report.", [artifacts["contract_report"]]
        if profile.start_stage > 2:
            return STATE_ACTIVE_INFERRED, "Validator is upstream of the current resume boundary.", [artifacts["run_context"]]
    if key in {"stage2_completed_stage2_only_resume_boundary", "stage2_compatibility_projection_completion", "stage2_execution_ready_only_downstream_handshake"}:
        path = find_run_file(profile.run_dir, "weak_labels__v7pilot_r3_fixparse.tsv")
        if profile.stage2_live or profile.stage2_replay or profile.full_pipeline_local:
            return STATE_ACTIVE_OBSERVED, "Completed Stage2 compatibility artifacts were built locally.", [path, artifacts.get("compatibility_summary")]
        return STATE_ACTIVE_INFERRED, "Run begins after the completed Stage2 compatibility boundary.", [artifacts["run_context"]]
    if key == "stage2_livev2_raw_response_rehydration":
        if profile.stage2_replay and values.get("legacy_raw_responses_dir"):
            return STATE_ACTIVE_OBSERVED, "Run rehydrated Stage2 outputs from saved raw responses.", [artifacts["run_context"], artifacts.get("raw_responses_dir")]
        return STATE_NOT_APPLICABLE, "Replay rehydration applies only to legacy_llm_replay runs.", []
    if key == "stage2_input_evidence_packing_ordered_blocks":
        if preview.get("input_packing_mode") == "ordered_blocks":
            return STATE_ACTIVE_OBSERVED, "Prompt preview records ordered block packing.", [artifacts["prompt_preview"]]
        if profile.stage2_replay and find_parent_live_prompt_preview(artifacts.get("raw_responses_dir")) is not None:
            return STATE_REPLAY_HIDDEN, "Replay run does not rebuild the original live prompt preview; parent live preview must be inspected.", [find_parent_live_prompt_preview(artifacts.get("raw_responses_dir")), artifacts["run_context"]]
    if key == "stage2_prompt_preview_observability":
        if artifacts.get("prompt_preview") is not None:
            return STATE_ACTIVE_OBSERVED, "Run emitted a Stage2 prompt preview artifact.", [artifacts["prompt_preview"]]
        if profile.stage2_replay and find_parent_live_prompt_preview(artifacts.get("raw_responses_dir")) is not None:
            return STATE_REPLAY_HIDDEN, "Prompt preview is visible only in the parent live run for replay executions.", [find_parent_live_prompt_preview(artifacts.get("raw_responses_dir")), artifacts["run_context"]]
    if key == "stage2_table_summary_mode" and (values.get("stage2_table_mode") == "summary" or "summary" in values.get("summary_table_mode_activation", "")):
        return STATE_ACTIVE_OBSERVED, "RUN_CONTEXT records summary table mode.", [artifacts["run_context"], artifacts.get("prompt_preview")]
    if key == "stage2_summary_first_column_enhancement" and parse_boolish(values.get("stage2_table_summary_first_column_enhancement", "")):
        return STATE_ACTIVE_OBSERVED, "RUN_CONTEXT records summary first-column enhancement as enabled.", [artifacts["run_context"], artifacts.get("prompt_preview")]
    if key == "stage2_table_selection_strategy":
        selected_tables = [str(item).strip() for item in ensure_list(table_selection_debug.get("selected_tables")) if str(item).strip()]
        ranking_mode = normalize_text(table_selection_debug.get("selection_ranking_mode"))
        if ranking_mode == "score_ranked_top_k" and selected_tables:
            if normalize_text(table_selection_debug.get("document_key")) == "QLYKLPKT":
                if any(name in selected_tables for name in ["QLYKLPKT__table_08__pdf_table.csv", "QLYKLPKT__table_09__pdf_table.csv"]):
                    return STATE_ACTIVE_OBSERVED, "Selector debug artifact shows ranked selection and includes the QLYK optimization tables.", [artifacts.get("table_selection_debug"), artifacts.get("prompt_preview")]
            return STATE_ACTIVE_OBSERVED, "Selector debug artifact shows ranked summary-table selection with non-empty chosen tables.", [artifacts.get("table_selection_debug"), artifacts.get("prompt_preview")]
        if profile.stage2_replay and find_parent_live_prompt_preview(artifacts.get("raw_responses_dir")) is not None:
            return STATE_REPLAY_HIDDEN, "Replay run does not rebuild the original ranked table-selection debug artifact; inspect the parent live run.", [find_parent_live_prompt_preview(artifacts.get("raw_responses_dir")), artifacts["run_context"]]
    if key == "stage2_default_raw_prefix_then_table_excerpts_layout" and preview.get("prompt_layout_class") == "raw_prefix_then_table_excerpts":
        return STATE_ACTIVE_OBSERVED, "Prompt preview shows the default raw-prefix then table-excerpts layout.", [artifacts["prompt_preview"]]
    if key == "stage2_non_doe_table_row_expansion" and compatibility.get("table_row_expansion_rows", 0) > 0:
        return STATE_ACTIVE_OBSERVED, "Compatibility summary records emitted non-DOE table-row expansion rows.", [artifacts["compatibility_summary"]]
    if key == "stage2_sequential_optimization_interpreter" and compatibility.get("sequential_optimization_resolved_rows", 0) > 0:
        return STATE_ACTIVE_OBSERVED, "Compatibility summary records sequential optimization resolved rows.", [artifacts["compatibility_summary"]]
    if key == "stage2_doe_expansion_with_llm_scope" and (compatibility.get("numbered_doe_recovered_rows", 0) > 0 or compatibility.get("doe_enumeration_mode") == "explicit_only"):
        return STATE_ACTIVE_OBSERVED if profile.stage2_live or profile.stage2_replay or profile.full_pipeline_local else STATE_ACTIVE_INFERRED, "Compatibility summary records DOE recovery or explicit-only DOE enumeration.", [artifacts["compatibility_summary"]]
    if key in {"stage2_partial_selection_marker_preservation", "stage2_partial_inheritance_marker_preservation"}:
        if artifacts.get("semantic_objects") is not None:
            return STATE_ACTIVE_OBSERVED, "Stage2 semantic objects are present and this contract belongs to the semantic intermediate layer.", [artifacts["semantic_objects"], artifacts.get("contract_report")]
        if profile.start_stage > 2:
            return STATE_ACTIVE_INFERRED, "Marker contracts are upstream semantic behavior and cannot be re-observed after Stage2 resume.", [artifacts["run_context"]]
    if key == "stage2_json_sanitation_path1" and find_run_file(profile.run_dir, "path1_parse_audit.tsv") is not None:
        return STATE_ACTIVE_OBSERVED, "Path 1 parse-audit artifact exists.", [find_run_file(profile.run_dir, "path1_parse_audit.tsv")]
    if key == "run_feature_activation_report" and artifacts.get("feature_activation") is not None:
        return STATE_ACTIVE_OBSERVED, "Run emitted a local feature activation report.", [artifacts["feature_activation"]]
    if key == "run_context_feature_activation_section" and "## Feature Unit Activation" in read_text(artifacts["run_context"]):
        return STATE_ACTIVE_OBSERVED, "RUN_CONTEXT contains the feature activation section.", [artifacts["run_context"]]
    if key == "boundary_governance_metadata" and "## Boundary Governance" in read_text(artifacts["run_context"]):
        return STATE_ACTIVE_OBSERVED, "RUN_CONTEXT contains boundary-governance metadata.", [artifacts["run_context"]]
    if key == "active_data_source_authority_resolution" and profile.compare_local and (values.get("source_resolution") or values.get("source_run_dir")):
        return STATE_ACTIVE_OBSERVED, "Compare run records explicit source authority resolution.", [artifacts["run_context"], artifacts.get("compare_counts")]
    if key == "family_variant_retention_governance":
        if artifacts.get("decision_trace") is not None:
            state = STATE_ACTIVE_OBSERVED if profile.stage5_final_local or profile.full_pipeline_local else STATE_ACTIVE_INFERRED
            reason = "Run-local final-output decision trace is present." if state == STATE_ACTIVE_OBSERVED else "Persisted final-output decision trace is visible from upstream Stage5 materialization."
            return state, reason, [artifacts["decision_trace"]]
        if profile.compare_local and values.get("final_table_tsv"):
            return STATE_ACTIVE_INFERRED, "Compare node starts from a final table whose materialization already includes family/variant retention upstream.", [artifacts["run_context"]]
    if key == "stage5_audit_ready_export":
        return (STATE_ACTIVE_OBSERVED, "Run-local audit-ready export exists.", [artifacts["audit_ready"]]) if artifacts.get("audit_ready") is not None else (STATE_NOT_APPLICABLE, "Audit-ready export is a supporting review workflow and was not part of this run.", [])
    if key == "stage5_layer2_risk_assessment":
        return (STATE_ACTIVE_OBSERVED, "Run-local risk assessment artifact exists.", [artifacts["risk_assessment"]]) if artifacts.get("risk_assessment") is not None else (STATE_NOT_APPLICABLE, "Layer2 risk assessment is a supporting review workflow and was not part of this run.", [])
    if key == "benchmark_doi_level_gt_count_audit" and artifacts.get("compare_counts") is not None:
        return STATE_ACTIVE_OBSERVED, "Run emitted DOI-level compare counts.", [artifacts["compare_counts"]]
    if key == "variant_aware_gt_authority_switch" and artifacts.get("compare_counts") is not None:
        if any("variantaware.xlsx" in row.get("gt_authority_file", "").lower() for row in read_tsv(artifacts["compare_counts"])):
            return STATE_ACTIVE_OBSERVED, "Compare output records the variant-aware GT authority workbook.", [artifacts["compare_counts"]]
    if key == "memory_bootstrap_debug_surface":
        return STATE_NOT_APPLICABLE, "This feature is a debugging support surface, not a runtime run-local behavior.", []
    if key == "replay_mode_prompt_visibility_gap":
        parent = find_parent_live_prompt_preview(artifacts.get("raw_responses_dir"))
        if profile.stage2_replay and parent is not None and artifacts.get("prompt_preview") is None:
            return STATE_ACTIVE_OBSERVED, "This replay run exhibits the prompt-visibility gap: parent live preview exists but replay-local preview does not.", [parent, artifacts["run_context"]]
        if profile.stage2_replay:
            return STATE_UNKNOWN, "Replay run detected, but parent live prompt preview could not be located automatically.", [artifacts["run_context"]]
    if key == "maintained_summary_features_missing_from_feature_unit_registry":
        if values.get("stage2_table_mode") == "summary" or "summary" in values.get("summary_table_mode_activation", ""):
            return STATE_ACTIVE_OBSERVED, "Summary-mode behavior is visible even though the current feature-unit registry does not cover it.", [artifacts["run_context"], artifacts.get("feature_activation")]
        return STATE_NOT_APPLICABLE, "Run did not use summary-mode behavior.", []
    if key == "docs_tool_index_stage2_authority_consistency":
        return STATE_NOT_APPLICABLE, "This is a repo-level documentation consistency issue, not a run-local execution feature.", []
    if key == "locality_preserving_literal_binding_pack":
        if "QLYKLPKT" in read_text(artifacts["run_context"]) and profile.stage2_live:
            return STATE_UNKNOWN, "Sequential-optimization locality-preserving pack is tracked but not adopted as a maintained runtime feature.", [artifacts["run_context"], artifacts.get("prompt_preview")]
        return STATE_NOT_APPLICABLE, "Design candidate not adopted for this run.", []
    if key == "deterministic_semantic_emitter_fallback_mode":
        if any("emit_semantic_objects_from_cleaned_papers_v1.py" in step for step in profile.execution_order):
            return STATE_ACTIVE_OBSERVED, "Run explicitly executed the deterministic semantic emitter fallback.", [artifacts["run_context"]]
        return STATE_NOT_APPLICABLE, "Fallback deterministic emitter was not used in this maintained-path run.", []
    if feature["current_status"] == "review_needed":
        return STATE_UNKNOWN, "Feature remains in a review-needed bucket and has no executable maintained contract.", []
    if expected == STATE_NOT_REACHABLE:
        return STATE_NOT_REACHABLE, "Feature belongs upstream of the current resume boundary.", [artifacts["run_context"]]
    if expected == STATE_EXPECTED_ACTIVE and activation.get(key, {}).get("activation_status") == "active":
        return STATE_ACTIVE_OBSERVED, "Existing run-local feature activation report marks this feature active.", [artifacts.get("feature_activation")]
    if expected == STATE_EXPECTED_ACTIVE:
        return STATE_EXPECTED_NOT_OBSERVED, "Feature was expected for this run but direct run-local evidence was not found.", []
    return STATE_UNKNOWN, "No deterministic classifier matched this feature for the current run.", []

def mismatch_for(expected: str, actual: str) -> tuple[str, str]:
    compatible = {
        (STATE_EXPECTED_ACTIVE, STATE_ACTIVE_OBSERVED),
        (STATE_EXPECTED_ACTIVE, STATE_ACTIVE_INFERRED),
        (STATE_INTENTIONALLY_DISABLED, STATE_INTENTIONALLY_DISABLED),
        (STATE_NOT_APPLICABLE, STATE_NOT_APPLICABLE),
        (STATE_NOT_REACHABLE, STATE_NOT_REACHABLE),
        (STATE_NOT_REACHABLE, STATE_ACTIVE_OBSERVED),
        (STATE_NOT_REACHABLE, STATE_ACTIVE_INFERRED),
        (STATE_NOT_REACHABLE, STATE_REPLAY_HIDDEN),
        (STATE_UNKNOWN, STATE_UNKNOWN),
        (STATE_UNKNOWN, STATE_NOT_APPLICABLE),
        (STATE_UNKNOWN, STATE_ACTIVE_OBSERVED),
    }
    if (expected, actual) in compatible:
        return "no", "none"
    if expected == STATE_EXPECTED_ACTIVE and actual in {STATE_EXPECTED_NOT_OBSERVED, STATE_REPLAY_HIDDEN}:
        return "yes", "expected_missing_or_hidden"
    if actual == STATE_UNKNOWN:
        return "yes", "unknown_needs_review"
    return "yes", "classification_mismatch"


def severity_for(feature: dict[str, str], mismatch_flag: str, actual: str) -> str:
    if mismatch_flag == "no":
        return "none"
    if feature["omission_risk"] == "high" or actual in {STATE_EXPECTED_NOT_OBSERVED, STATE_REPLAY_HIDDEN}:
        return "high"
    return "medium"


def recommended_debug_order(feature: dict[str, str]) -> str:
    category = feature_category(feature["feature_key"])
    if category == "run_metadata":
        return "1"
    if category in {"stage2_live_visibility", "stage2_replay", "stage2_completed"}:
        return "2"
    if category == "stage5_final":
        return "3"
    if category == "compare":
        return "4"
    return "5"


def build_ledger_rows(inventory: list[dict[str, str]], profile: RunProfile, values: dict[str, str], artifacts: dict[str, Path | None]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    upstream = build_upstream_trace(profile, values, artifacts)
    compatibility = load_compatibility_summary(artifacts.get("compatibility_summary"))
    preview = load_prompt_preview(artifacts.get("prompt_preview"))
    table_selection_debug = load_table_selection_debug(artifacts.get("table_selection_debug"))
    activation = load_feature_activation(artifacts.get("feature_activation"))
    schema_path = DOCS_DIR / "feature_governance" / "feature_applicability_schema_v1.tsv"
    schema_rows = read_tsv(schema_path) if schema_path.exists() else []
    schema_by_feature = {row["feature_key"]: row for row in schema_rows}
    rows: list[dict[str, str]] = []
    for feature in inventory:
        schema_row = schema_by_feature.get(feature["feature_key"], {})
        expected = expected_state(feature, profile, values, schema_row)
        actual, reason, evidence = actual_state(feature, profile, values, artifacts, upstream, compatibility, preview, activation, schema_row, table_selection_debug)
        mismatch_flag, mismatch_type = mismatch_for(expected, actual)
        rows.append({
            "run_id": profile.run_id,
            "feature_key": feature["feature_key"],
            "feature_name": feature["feature_name"],
            "expected_state_for_this_run": expected,
            "actual_state_for_this_run": actual,
            "state_reason": reason,
            "evidence_paths": "; ".join(repo_rel(path) for path in evidence if path is not None and path.exists()),
            "evidence_summary": feature.get("expected_runtime_artifact", ""),
            "mismatch_flag": mismatch_flag,
            "mismatch_type": mismatch_type,
            "severity": severity_for(feature, mismatch_flag, actual),
            "entrypoint_scope": profile.entrypoint_scope,
            "source_mode": profile.source_mode,
            "resume_boundary": profile.resume_boundary,
            "upstream_applied_before_run": "yes" if actual == STATE_ACTIVE_INFERRED else "no",
            "visible_in_current_run": "no" if actual in {STATE_ACTIVE_INFERRED, STATE_NOT_REACHABLE, STATE_REPLAY_HIDDEN} else "yes",
            "replay_visibility_limit": "yes" if profile.stage2_replay and actual == STATE_REPLAY_HIDDEN else "no",
            "recommended_debug_order": recommended_debug_order(feature),
            "notes": feature.get("notes", ""),
        })
    return rows, upstream


def summarize_run(rows: list[dict[str, str]], profile: RunProfile) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["actual_state_for_this_run"]] = counts.get(row["actual_state_for_this_run"], 0) + 1
    severities = [row["severity"] for row in rows if row["severity"] != "none"]
    highest = "high" if "high" in severities else ("medium" if "medium" in severities else "none")
    return {"run_id": profile.run_id, "run_dir": repo_rel(profile.run_dir), "entrypoint_scope": profile.entrypoint_scope, "source_mode": profile.source_mode, "resume_boundary": profile.resume_boundary, "ledger_rows": len(rows), "active_observed": counts.get(STATE_ACTIVE_OBSERVED, 0), "active_inferred_from_upstream": counts.get(STATE_ACTIVE_INFERRED, 0), "expected_but_not_observed": counts.get(STATE_EXPECTED_NOT_OBSERVED, 0), "intentionally_disabled": counts.get(STATE_INTENTIONALLY_DISABLED, 0), "replay_hidden": counts.get(STATE_REPLAY_HIDDEN, 0), "not_reachable_due_to_resume_boundary": counts.get(STATE_NOT_REACHABLE, 0), "not_applicable_for_run_scope": counts.get(STATE_NOT_APPLICABLE, 0), "unknown_needs_review": counts.get(STATE_UNKNOWN, 0), "mismatch_rows": sum(1 for row in rows if row["mismatch_flag"] == "yes"), "highest_severity": highest}


def write_run_report(path: Path, profile: RunProfile, upstream: dict[str, Any], rows: list[dict[str, str]]) -> None:
    expected_local = [row for row in rows if row["expected_state_for_this_run"] == STATE_EXPECTED_ACTIVE]
    hidden = [row for row in rows if row["actual_state_for_this_run"] in {STATE_REPLAY_HIDDEN, STATE_NOT_REACHABLE}]
    mismatches = [row for row in rows if row["mismatch_flag"] == "yes"]
    mismatches.sort(key=lambda row: ({"high": 0, "medium": 1, "none": 2}.get(row["severity"], 3), row["recommended_debug_order"], row["feature_key"]))
    lines = ["# Feature Execution Ledger Report v1", "", "Diagnostic-only governance support artifact.", "", "## Run Scope", "", f"- run_id: `{profile.run_id}`", f"- run_dir: `{repo_rel(profile.run_dir)}`", f"- entrypoint_scope: `{profile.entrypoint_scope}`", f"- source_mode: `{profile.source_mode}`", f"- resume_boundary: `{profile.resume_boundary}`", f"- run_type: `{profile.run_type}`", "", "## Upstream Already Applied Before This Run", ""]
    if upstream["transformations_before_current_entrypoint"]:
        for item in upstream["transformations_before_current_entrypoint"]:
            lines.append(f"- `{item['stage']}`: `{item['status']}` via `{item.get('evidence_path', '')}`. {item['reason']}")
    else:
        lines.append("- none detected from explicit run inputs")
    lines.extend(["", "## Expected Local Features", "", f"- expected_active_count: `{len(expected_local)}`"])
    for row in expected_local[:12]:
        lines.append(f"- `{row['feature_key']}` -> `{row['actual_state_for_this_run']}`")
    lines.extend(["", "## Hidden Or Upstream-Limited Features", ""])
    for row in hidden[:12] or [{"feature_key": "none", "actual_state_for_this_run": "none", "state_reason": "none"}]:
        lines.append(f"- `{row['feature_key']}` -> `{row['actual_state_for_this_run']}`: {row['state_reason']}")
    lines.extend(["", "## Highest-Priority Mismatches", ""])
    for row in mismatches[:12] or [{"feature_key": "none", "severity": "none", "expected_state_for_this_run": "none", "actual_state_for_this_run": "none", "state_reason": "no mismatches detected"}]:
        lines.append(f"- `{row['feature_key']}` [{row['severity']}] expected `{row['expected_state_for_this_run']}` but saw `{row['actual_state_for_this_run']}`. {row['state_reason']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_system_report(path: Path, schema_path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = ["# Feature Execution Ledger System Report v1", "", "Diagnostic-only governance-support artifact.", "", "## What This Adds", "", f"- global run-evaluable schema: `{repo_rel(schema_path)}`", "- per-run ledger rows for every recovered feature", "- explicit upstream processing trace for resume and replay runs", "- run-local markdown reports that answer expected-vs-actual before semantic analysis", "", "## Backfilled Run Types", ""]
    for summary in summaries:
        lines.append(f"- `{summary['run_id']}` -> `{summary['entrypoint_scope']}` with `{summary['source_mode']}`; mismatches=`{summary['mismatch_rows']}`")
    lines.extend(["", "## State Model", "", f"- `{STATE_EXPECTED_ACTIVE}`", f"- `{STATE_ACTIVE_OBSERVED}`", f"- `{STATE_ACTIVE_INFERRED}`", f"- `{STATE_NOT_APPLICABLE}`", f"- `{STATE_NOT_REACHABLE}`", f"- `{STATE_EXPECTED_NOT_OBSERVED}`", f"- `{STATE_INTENTIONALLY_DISABLED}`", f"- `{STATE_REPLAY_HIDDEN}`", f"- `{STATE_UNKNOWN}`", "", "## Use Before Debugging", "", "1. Open the run-local `analysis/feature_execution_ledger_v1.tsv`.", "2. Read `analysis/feature_upstream_processing_trace_v1.json`.", "3. Read `analysis/feature_execution_ledger_report_v1.md`.", "4. Only then inspect semantic extraction or GT comparison details."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_guide(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# How To Use Feature Execution Ledger v1\n\nDiagnostic-only governance-support guide.\n\n1. Open the run's `analysis/feature_execution_ledger_v1.tsv`.\n2. Filter `mismatch_flag=yes`.\n3. Check `expected_state_for_this_run` versus `actual_state_for_this_run`.\n4. Open `analysis/feature_upstream_processing_trace_v1.json` to see which stages already happened upstream.\n5. Treat `replay_hidden` as a prompt-visibility limitation, not as proof that the feature was absent.\n6. Treat `active_inferred_from_upstream` as upstream-applied behavior, not run-local execution.\n7. Only after that, inspect raw prompts, semantic rows, or GT deltas.\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    inventory = read_tsv(Path(args.inventory))
    schema_path = Path(args.schema_out)
    existing_schema_rows = read_tsv(schema_path) if schema_path.exists() else []
    existing_schema_by_feature = {row["feature_key"]: row for row in existing_schema_rows}
    schema_rows = [derive_schema_row(feature, existing_schema_by_feature.get(feature["feature_key"])) for feature in inventory]
    write_tsv(schema_path, schema_rows, SCHEMA_COLUMNS)
    summaries: list[dict[str, Any]] = []
    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        profile, values, artifacts = infer_run_profile(run_dir)
        rows, upstream = build_ledger_rows(inventory, profile, values, artifacts)
        write_tsv(run_dir / "analysis" / "feature_execution_ledger_v1.tsv", rows, STATE_COLUMNS)
        write_json(run_dir / "analysis" / "feature_execution_ledger_v1.json", {"run_id": profile.run_id, "run_dir": repo_rel(run_dir), "entrypoint_scope": profile.entrypoint_scope, "rows": rows})
        write_json(run_dir / "analysis" / "feature_upstream_processing_trace_v1.json", upstream)
        write_run_report(run_dir / "analysis" / "feature_execution_ledger_report_v1.md", profile, upstream, rows)
        summaries.append(summarize_run(rows, profile))
    write_json(Path(args.backfill_summary_out), {"runs": summaries})
    write_tsv(Path(args.backfill_summary_tsv), summaries, BACKFILL_SUMMARY_COLUMNS)
    build_system_report(Path(args.system_report_out), schema_path, summaries)
    build_guide(Path(args.guide_out))
    print(str(schema_path))
    for summary in summaries:
        print(summary["run_dir"])


if __name__ == "__main__":
    main()
