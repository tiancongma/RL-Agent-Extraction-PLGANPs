from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from .repo_paths import DOCS_DIR, PROJECT_DIR, REPO_ROOT, RESULTS_DIR, repo_relative
from .schemas import JSONDict

logger = logging.getLogger(__name__)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _safe_read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return _read_text(path)


def _safe_read_json(path: Path) -> JSONDict | None:
    if not path.exists():
        return None
    return json.loads(_read_text(path))


def _safe_read_tsv(path: Path) -> list[JSONDict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _extract_bullets(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    bullets: list[str] = []
    capture = False
    for line in lines:
        if line.strip().startswith("#"):
            normalized = line.strip("#").strip()
            if normalized == heading:
                capture = True
                continue
            if capture:
                break
        if capture:
            stripped = line.strip()
            if stripped.startswith("- "):
                bullets.append(stripped[2:].strip())
    return bullets


def load_active_run_json() -> JSONDict | None:
    return _safe_read_json(RESULTS_DIR / "ACTIVE_RUN.json")


def load_maintained_registry() -> list[JSONDict]:
    return _safe_read_tsv(DOCS_DIR / "maintained_script_surface.tsv")


def load_script_registry() -> list[JSONDict]:
    return _safe_read_tsv(DOCS_DIR / "src_script_registry.tsv")


def load_active_pipeline_authority() -> JSONDict:
    flow_path = PROJECT_DIR / "ACTIVE_PIPELINE_FLOW.md"
    architecture_path = PROJECT_DIR / "2_ARCHITECTURE.md"
    readme_path = REPO_ROOT / "README.md"

    flow_text = _safe_read_text(flow_path) or ""
    architecture_text = _safe_read_text(architecture_path) or ""
    readme_text = _safe_read_text(readme_path) or ""
    primary_text = "\n".join([flow_text, architecture_text])

    primary_namespace_matches = re.findall(r"`(src/stage[0-9_]+[^`]*/)`", primary_text)
    helper_namespace_matches = re.findall(r"`(src/stage[0-9_]+[^`]*/)`", readme_text)
    namespace_matches = _ordered_unique(primary_namespace_matches + helper_namespace_matches)
    active_namespaces = [
        match for match in namespace_matches if not match.startswith("src/stage3_gt")
    ]

    reserved_namespaces = []
    if "src/stage3_gt/" in primary_text or "src/stage3_gt/" in readme_text:
        reserved_namespaces.append("src/stage3_gt/")

    benchmark_endpoints = _ordered_unique(
        re.findall(r"`(final_[^`]+?\.tsv)`", primary_text)
        + re.findall(r"`(final_[^`]+?\.tsv)`", readme_text)
    )
    boundary_classes = []
    for name in (
        "internal_intermediate",
        "diagnostic_boundary",
        "mainline_resume_boundary",
        "benchmark_terminal_boundary",
    ):
        if name in primary_text or name in readme_text:
            boundary_classes.append(name)

    return {
        "resource_name": "active_pipeline_authority",
        "source_files": [
            repo_relative(flow_path),
            repo_relative(architecture_path),
            repo_relative(readme_path),
        ],
        "active_stage_namespaces": active_namespaces,
        "reserved_reference_namespaces": reserved_namespaces,
        "benchmark_valid_endpoint": "final_formulation_table_v1.tsv",
        "comparison_node_entrypoint": "src/stage5_benchmark/compare_final_table_to_gt_v1.py",
        "run_boundary_classes": boundary_classes,
        "notes": {
            "stage2_authority_summary": (
                "Stage2 authority is composite: LLM semantic discovery plus "
                "deterministic post-LLM completion inside Stage2."
            ),
            "stage4_role": "Stage4 is a diagnostic branch, not the benchmark-valid endpoint.",
            "stage6_status": "No active Stage6 namespace is allowed by current authority.",
            "authority_extraction_priority": (
                "Primary extraction prefers ACTIVE_PIPELINE_FLOW.md and 2_ARCHITECTURE.md; "
                "README.md is used only as a secondary helper."
            ),
        },
        "benchmark_endpoint_mentions": benchmark_endpoints,
        "raw_excerpt_available": bool(flow_text and architecture_text),
    }


def load_active_data_source_authority() -> JSONDict:
    contract_path = PROJECT_DIR / "ACTIVE_DATA_SOURCE_CONTRACT.md"
    contract_text = _safe_read_text(contract_path) or ""
    active_run = load_active_run_json()

    forbidden = [
        name
        for name in (
            "latest-by-sort",
            "latest-by-mtime",
            "parent fallback",
            "glob-first match",
            "silent defaulting",
        )
        if name in contract_text
    ]

    return {
        "resource_name": "active_data_source_authority",
        "source_files": [
            repo_relative(contract_path),
            repo_relative(RESULTS_DIR / "ACTIVE_RUN.json"),
        ],
        "resolution_precedence": [
            "explicit CLI path such as --run-dir",
            "data/results/ACTIVE_RUN.json",
            "otherwise hard error",
        ],
        "forbidden_resolution_behaviors": forbidden,
        "active_run_pointer": active_run,
        "gt_authority_lock": None if active_run is None else active_run.get("gt_authority_lock"),
    }


def load_maintained_entrypoint_registry_resource() -> JSONDict:
    registry_path = DOCS_DIR / "maintained_script_surface.tsv"
    rows = load_maintained_registry()
    return {
        "resource_name": "maintained_entrypoint_registry",
        "source_files": [repo_relative(registry_path)],
        "row_count": len(rows),
        "rows": rows,
    }


def load_stage2_contract_and_frozen_substeps() -> JSONDict:
    runbook_path = PROJECT_DIR / "ACTIVE_PIPELINE_RUNBOOK.md"
    map_path = PROJECT_DIR / "PIPELINE_SCRIPT_MAP.md"
    decisions_path = PROJECT_DIR / "4_DECISIONS_LOG.md"

    runbook_text = _safe_read_text(runbook_path) or ""
    map_text = _safe_read_text(map_path) or ""
    decisions_text = _safe_read_text(decisions_path) or ""
    registry_rows = load_maintained_registry()

    stage2_rows = [
        row
        for row in registry_rows
        if row.get("stage_or_workflow", "").startswith("stage2_")
    ]

    substeps: list[JSONDict] = []
    for substep_id, script_suffix in (
        ("S2-4a", "run_stage2_s2_4a_prompt_construction_v1.py"),
        ("S2-4b", "run_stage2_s2_4b_live_llm_call_v1.py"),
        ("S2-5", "run_stage2_s2_5_semantic_parsing_v1.py"),
        ("S2-6", "run_stage2_s2_6_contract_validation_v1.py"),
        ("S2-7", "run_stage2_s2_7_compatibility_projection_v1.py"),
    ):
        matched = next((row for row in stage2_rows if row.get("script_path", "").endswith(script_suffix)), None)
        if matched:
            substeps.append(
                {
                    "substep_id": substep_id,
                    "script_path": matched.get("script_path"),
                    "status": matched.get("status"),
                    "role": matched.get("role"),
                    "notes": matched.get("notes"),
                }
            )

    for substep_id, owner_text in (
        ("S2-2a", "build_candidate_segmentation_artifact"),
        ("S2-2b", "build_evidence_blocks_artifact"),
        ("S2-3", "build_live_prompt"),
    ):
        if owner_text in map_text:
            substeps.append(
                {
                    "substep_id": substep_id,
                    "owner_surface_hint": owner_text,
                    "status": "internal_frozen_boundary",
                }
            )

    return {
        "resource_name": "stage2_contract_and_frozen_substeps",
        "source_files": [
            repo_relative(runbook_path),
            repo_relative(map_path),
            repo_relative(decisions_path),
            repo_relative(DOCS_DIR / "maintained_script_surface.tsv"),
        ],
        "authority_summary": {
            "llm_semantic_discovery": "authoritative",
            "deterministic_post_llm_completion_inside_stage2": "authoritative",
            "deterministic_semantic_emitters": "fallback_or_diagnostic_only",
        },
        "stage2_contract_markers": {
            "llm_first_composite": "present" if "llm_first_composite" in runbook_text or "llm_first_composite" in map_text else "not_explicitly_found",
            "doe_scope_rule": "present" if "LLM-declared DOE scope" in runbook_text or "LLM-declared DOE scope" in decisions_text else "not_explicitly_found",
            "non_doe_table_authorization_rule": "present" if "non-DOE table" in runbook_text or "non-DOE table" in map_text else "not_explicitly_found",
        },
        "frozen_substeps": substeps,
    }


def load_naming_and_run_layout_contract() -> JSONDict:
    naming_path = PROJECT_DIR / "FILE_NAMING_AND_VERSIONING.md"
    runbook_path = PROJECT_DIR / "ACTIVE_PIPELINE_RUNBOOK.md"
    naming_text = _safe_read_text(naming_path) or ""
    runbook_text = _safe_read_text(runbook_path) or ""

    prohibited_patterns = re.findall(r"- (\*_[^. ]+\.\*)", naming_text)
    if not prohibited_patterns:
        prohibited_patterns = ["*_final.*", "*_final_v2.*", "*_new.*", "*_latest.*", "*_fix.*"]

    allowed_dataset_roots = _extract_bullets(naming_text, "Allowed Dataset Roots")

    return {
        "resource_name": "naming_and_run_layout_contract",
        "source_files": [
            repo_relative(naming_path),
            repo_relative(runbook_path),
        ],
        "prohibited_name_patterns": prohibited_patterns,
        "future_results_layout": {
            "bucket_pattern": "data/results/YYYYMMDD_<short_hash>/",
            "child_pattern": "data/results/YYYYMMDD_<short_hash>/NN_<cue>/",
        },
        "legacy_results_layout": {
            "allowed_pattern": r"^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$",
            "compatibility_only": True,
        },
        "allowed_dataset_roots": allowed_dataset_roots,
        "run_layout_rule_present": "MDEC084" in naming_text or "MDEC084" in runbook_text,
    }


def resolve_active_run(run_dir: str | None = None) -> JSONDict:
    active_run = load_active_run_json()
    if run_dir:
        explicit_path = (REPO_ROOT / run_dir).resolve() if not Path(run_dir).is_absolute() else Path(run_dir).resolve()
        return {
            "resolution_mode": "explicit_run_dir",
            "resolved_run_dir": repo_relative(explicit_path),
            "resolved_run_id": explicit_path.name,
            "active_run_pointer_present": active_run is not None,
            "authoritative_terminal_files": active_run.get("authoritative_terminal_files", {}) if active_run else {},
        }
    if active_run is None:
        return {"resolution_mode": "error", "error": "ACTIVE_RUN.json is missing and no explicit run_dir was provided."}
    return {
        "resolution_mode": "active_run_pointer",
        "resolved_run_dir": active_run.get("active_run_dir"),
        "resolved_run_id": active_run.get("active_run_id"),
        "authoritative_terminal_files": active_run.get("authoritative_terminal_files", {}),
        "gt_authority_lock": active_run.get("gt_authority_lock"),
    }


def list_maintained_entrypoints(stage_or_workflow: str) -> JSONDict:
    query = stage_or_workflow.strip().lower()
    rows = load_maintained_registry()
    matched = [
        row
        for row in rows
        if query in row.get("stage_or_workflow", "").lower()
        or query in row.get("script_path", "").lower()
        or query in row.get("role", "").lower()
    ]
    return {
        "query": stage_or_workflow,
        "matches": matched,
        "match_count": len(matched),
    }


def check_script_selection(script_path: str, task_type: str) -> JSONDict:
    maintained_rows = load_maintained_registry()
    script_rows = [row for row in maintained_rows if row.get("script_path") == script_path]
    advisory_rows = [row for row in load_script_registry() if row.get("script_path") == script_path]
    normalized_task_type = task_type.strip().lower()
    matched_maintained_row = script_rows[0] if script_rows else None
    matched_advisory_row = advisory_rows[0] if advisory_rows else None

    strict_execution_tasks = {
        "execution",
        "production",
        "production_execution",
        "benchmark_execution",
        "stage_execution",
        "mainline_execution",
    }
    compare_audit_tasks = {
        "compare",
        "comparison",
        "audit",
        "review",
        "workbook",
        "workbook_generation",
    }

    if not script_rows and not advisory_rows:
        return {
            "allowed": False,
            "reason": "Script is not present in the maintained registry or advisory registry.",
            "task_type": task_type,
            "normalized_task_type": normalized_task_type,
            "script_path": script_path,
            "matched_maintained_row": None,
            "matched_advisory_row": None,
        }

    if matched_maintained_row:
        row = matched_maintained_row
        status = row.get("status", "")
        allowed_statuses = {"maintained_entrypoint"}
        if normalized_task_type in compare_audit_tasks:
            allowed_statuses = {"maintained_entrypoint", "supporting_nondefault"}
        elif normalized_task_type in strict_execution_tasks:
            allowed_statuses = {"maintained_entrypoint"}
        elif normalized_task_type in {"alignment"}:
            allowed_statuses = {"maintained_entrypoint", "supporting_nondefault"}
        else:
            allowed_statuses = {"maintained_entrypoint", "supporting_nondefault"}

        if status == "wrapper_nondefault" and normalized_task_type in strict_execution_tasks:
            allowed = False
            reason = (
                f"Denied for task_type={normalized_task_type or task_type} because "
                "wrapper_nondefault scripts are not lawful execution-facing defaults and "
                "require an explicit user request for that wrapper."
            )
        else:
            allowed = status in allowed_statuses

        if allowed:
            reason = (
                f"Allowed for task_type={normalized_task_type or task_type} because "
                f"maintained registry status '{status}' is permitted."
            )
        elif status != "wrapper_nondefault" or normalized_task_type not in strict_execution_tasks:
            reason = (
                f"Denied for task_type={normalized_task_type or task_type} because "
                f"maintained registry status '{status}' is not permitted. "
                f"Allowed statuses for this task are {sorted(allowed_statuses)}."
            )
        return {
            "allowed": allowed,
            "script_path": script_path,
            "task_type": task_type,
            "normalized_task_type": normalized_task_type,
            "maintained_registry_status": status,
            "must_use_active_data_source_contract": row.get("must_use_active_data_source_contract"),
            "allowed_statuses_for_task_type": sorted(allowed_statuses),
            "reason": reason,
            "notes": row.get("notes"),
            "matched_maintained_row": row,
            "matched_advisory_row": matched_advisory_row,
        }

    return {
        "allowed": False,
        "script_path": script_path,
        "task_type": task_type,
        "normalized_task_type": normalized_task_type,
        "reason": "Script appears only in the advisory script registry and should not be treated as a default execution whitelist entry.",
        "matched_maintained_row": None,
        "matched_advisory_row": matched_advisory_row,
        "advisory_registry_status": matched_advisory_row.get("status"),
        "advisory_registry_role": matched_advisory_row.get("current_pipeline_role"),
    }


def validate_file_path(proposed_path: str, intended_role: str | None = None) -> JSONDict:
    normalized = proposed_path.replace("\\", "/").strip("/")
    parts = normalized.split("/") if normalized else []
    path_name = Path(normalized).name
    lower_name = path_name.lower()

    violations: list[str] = []
    warnings: list[str] = []

    if not normalized:
        violations.append("Path must not be empty.")

    if parts[:1] == ["mcp_server"]:
        warnings.append("mcp_server/ is the intended read-only location for this repo-specific MCP server.")

    prohibited_markers = ["_final.", "_final_v2.", "_new.", "_latest.", "_fix."]
    if any(marker in lower_name for marker in prohibited_markers):
        violations.append("Filename matches a prohibited naming pattern from FILE_NAMING_AND_VERSIONING.md.")

    if parts[:1] == ["project"]:
        violations.append("Do not create new governance files under project/ without explicit instruction.")

    if parts[:1] == ["src"] and len(parts) >= 2 and parts[1].startswith("stage"):
        violations.append(
            "New files under src/stage* are denied by default by repo governance. "
            "Active stage namespaces are not open-ended file-creation targets, and no whitelist matched this path."
        )

    if parts[:2] == ["data", "results"]:
        if len(parts) >= 3 and parts[2].startswith("run_"):
            warnings.append("Legacy run_* results layout is compatibility-only.")
        if len(parts) >= 4 and re.match(r"^\d{8}_[0-9a-f]{7}$", parts[2]) and not re.match(r"^\d{2,3}_[A-Za-z0-9_]+$", parts[3]):
            violations.append("Future child execution folders under data/results bucket roots must use ordinal child names such as NN_<cue>.")

    if parts[:2] == ["data", "cleaned"] and len(parts) >= 4:
        allowed_roots = {"index", "content", "text", "sections", "tables", "analysis"}
        root_name = parts[3]
        if root_name not in allowed_roots:
            violations.append("Dataset-scoped cleaned outputs are limited to index/content/text/sections/tables/analysis.")

    return {
        "allowed": not violations,
        "proposed_path": proposed_path,
        "intended_role": intended_role,
        "governance_scope": (
            "repo_specific_mcp_server"
            if parts[:1] == ["mcp_server"]
            else "governance_layer"
            if parts[:1] == ["project"]
            else "active_stage_namespace"
            if parts[:1] == ["src"] and len(parts) >= 2 and parts[1].startswith("stage")
            else "general_repo_path"
        ),
        "violations": violations,
        "warnings": warnings,
    }


def _resolve_layer_gt_path(active_run: JSONDict, normalized_layer: str) -> tuple[str | None, str | None]:
    layer_key_map = {
        "layer1": "layer1_gt_path",
        "layer2": "layer2_gt_path",
        "layer3": "layer3_gt_path",
    }
    direct_key = layer_key_map[normalized_layer]

    direct_value = active_run.get(direct_key)
    if isinstance(direct_value, str) and direct_value.strip():
        return direct_value, f"top_level.{direct_key}"

    terminal_files = active_run.get("authoritative_terminal_files")
    if isinstance(terminal_files, dict):
        nested_value = terminal_files.get(direct_key)
        if isinstance(nested_value, str) and nested_value.strip():
            return nested_value, f"authoritative_terminal_files.{direct_key}"

    return None, None


def get_authoritative_gt(layer: str, explicit_path: str | None = None) -> JSONDict:
    active_run = load_active_run_json()
    if active_run is None:
        return {"allowed": False, "error": "ACTIVE_RUN.json is missing."}

    normalized_layer = layer.strip().lower()
    if normalized_layer not in {"layer1", "layer2", "layer3"}:
        return {
            "allowed": False,
            "error": "Unsupported GT layer. Use layer1, layer2, or layer3.",
            "requested_layer": layer,
        }

    pinned, resolved_from = _resolve_layer_gt_path(active_run, normalized_layer)
    if not pinned:
        return {
            "allowed": False,
            "requested_layer": normalized_layer,
            "gt_authority_lock": active_run.get("gt_authority_lock"),
            "error": (
                f"Could not resolve a contracted GT path for {normalized_layer}. "
                "Checked top-level ACTIVE_RUN.json keys and authoritative_terminal_files only."
            ),
        }

    normalized_explicit = None if explicit_path is None else explicit_path.replace("\\", "/")
    matches = normalized_explicit is None or normalized_explicit == pinned
    return {
        "allowed": matches,
        "requested_layer": normalized_layer,
        "gt_authority_lock": active_run.get("gt_authority_lock"),
        "authoritative_gt_path": pinned,
        "resolved_from": resolved_from,
        "explicit_path": explicit_path,
        "explicit_path_matches_contract": matches,
        "reason": (
            f"Resolved GT path from {resolved_from}."
            if matches
            else "Explicit GT path does not match the contracted GT path."
        ),
    }


def check_resume_boundary(boundary_name: str) -> JSONDict:
    name = boundary_name.strip()
    normalized = name.lower()

    boundary_map: dict[str, JSONDict] = {
        "internal_intermediate": {
            "boundary_class": "internal_intermediate",
            "stage_local_next_step_allowed": False,
            "lawful_downstream_resume_boundary": False,
            "next_lawful_step": None,
            "notes": "Internal stage-local artifact only; not itself a lawful resume boundary.",
        },
        "diagnostic_boundary": {
            "boundary_class": "diagnostic_boundary",
            "stage_local_next_step_allowed": False,
            "lawful_downstream_resume_boundary": False,
            "next_lawful_step": None,
            "notes": "Replay or audit artifact only; inspection is allowed, but it does not by itself authorize mainline continuation.",
        },
        "mainline_resume_boundary": {
            "boundary_class": "mainline_resume_boundary",
            "stage_local_next_step_allowed": True,
            "lawful_downstream_resume_boundary": True,
            "next_lawful_step": None,
            "notes": "Contract-complete upstream artifact that a maintained downstream stage may legally consume.",
        },
        "benchmark_terminal_boundary": {
            "boundary_class": "benchmark_terminal_boundary",
            "stage_local_next_step_allowed": False,
            "lawful_downstream_resume_boundary": False,
            "next_lawful_step": None,
            "notes": "Benchmark-facing terminal surface rather than a general downstream resume boundary.",
        },
        "s2-4a": {
            "boundary_class": "internal_intermediate",
            "stage_local_next_step_allowed": True,
            "lawful_downstream_resume_boundary": False,
            "next_lawful_step": "S2-4b live LLM call",
            "notes": "Frozen prompt-construction handoff only. It supports the next Stage2 substep but is not a general downstream resume boundary.",
        },
        "s2-4b": {
            "boundary_class": "diagnostic_boundary",
            "stage_local_next_step_allowed": True,
            "lawful_downstream_resume_boundary": False,
            "next_lawful_step": "S2-5 semantic parsing",
            "notes": "Raw-response freeze is a stage-local handoff into Stage2 semantic parsing only. It is not a general lawful downstream resume boundary for later stages.",
        },
        "s2-5": {
            "boundary_class": "diagnostic_boundary",
            "stage_local_next_step_allowed": True,
            "lawful_downstream_resume_boundary": False,
            "next_lawful_step": "S2-6 contract validation",
            "notes": "Semantic intermediates support the next maintained Stage2 validation step only. They are not Stage3-ready and do not by themselves authorize downstream continuation.",
        },
        "s2-6": {
            "boundary_class": "internal_intermediate",
            "stage_local_next_step_allowed": True,
            "lawful_downstream_resume_boundary": False,
            "next_lawful_step": "S2-7 compatibility projection",
            "notes": "Validation artifacts are a Stage2-local legality gate and handoff into compatibility projection only.",
        },
        "s2-7": {
            "boundary_class": "mainline_resume_boundary",
            "stage_local_next_step_allowed": True,
            "lawful_downstream_resume_boundary": True,
            "next_lawful_step": "Stage3 relation materialization",
            "notes": "Completed Stage2 compatibility projection is the lawful downstream resume boundary for Stage3.",
        },
        "weak_labels__v7pilot_r3_fixparse.tsv": {
            "boundary_class": "mainline_resume_boundary",
            "stage_local_next_step_allowed": True,
            "lawful_downstream_resume_boundary": True,
            "next_lawful_step": "Stage3 relation materialization",
            "notes": "Canonical completed Stage2 artifact name. This is the lawful downstream resume boundary for Stage3 relation materialization.",
        },
        "stage5_final_formulation_table_v1": {
            "boundary_class": "benchmark_terminal_boundary",
            "stage_local_next_step_allowed": True,
            "lawful_downstream_resume_boundary": False,
            "next_lawful_step": "Identity freeze gate, compare node, or downstream frozen-final helpers",
            "notes": "The final table is the benchmark-final object, but benchmark legality additionally depends on the identity-freeze gate and compare workflow. It is not exposed as a generic mainline resume boundary.",
        },
    }

    key = normalized
    if normalized == "final_formulation_table_v1.tsv":
        key = "stage5_final_formulation_table_v1"

    result = boundary_map.get(key)
    if result is None:
        return {
            "known": False,
            "boundary_name": boundary_name,
            "reason": "Boundary is not encoded in the v1 read-only MCP skeleton.",
        }
    return {"known": True, "boundary_name": name, **result}
