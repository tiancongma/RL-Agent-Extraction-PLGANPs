#!/usr/bin/env python3
from __future__ import annotations

"""
Governed DOE row expansion function unit for Stage2 llm_first_composite mode.

Semantic authority remains with the LLM. This unit only materializes explicit
numbered DOE table rows that already exist in Stage1 assets and only when the
LLM has declared DOE scope that authorizes deterministic row expansion.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from src.stage2_sampling_labels.auto_extract_weak_labels_v7pilot_r3_fixparse import (
    build_numbered_doe_guard_row,
    flatten_row,
    write_numbered_doe_guard_artifact,
)
from src.stage2_sampling_labels.build_numbered_doe_row_candidates_v1 import (
    PaperRecord,
    enumerate_numbered_doe_candidates_for_explicit_tables,
)


NUMBERED_DOE_RECOVERY_ENV = "STAGE2_ENABLE_NUMBERED_DOE_RECOVERY"
NUMBERED_DOE_RECOVERY_MIN_ROWS_ENV = "STAGE2_NUMBERED_DOE_MIN_ROWS"
DOE_ENUMERATION_MODE_ENV = "STAGE2_DOE_ENUMERATION_MODE"
LLM_FIRST_COMPOSITE_MODE = "llm_first_composite"
LLM_SEMANTIC_DISCOVERY = "llm_semantic_discovery"
LLM_PARSED = "llm_parsed"
LLM_DECLARED_SCOPE = "llm_declared_scope"
DOE_SCOPE_KIND = "doe_table_row_enumeration_scope"
FUNCTION_UNIT_ID = "doe_row_expansion_function_unit_v1"
ROW_MATERIALIZATION_MODE = "deterministic_row_expansion_within_llm_scope"
RECOVERY_CANDIDATE_SOURCE = "doe_numbered_table_row_recovery"
NORMALIZED_TABLE_PAYLOADS_SUBDIR = "normalized_table_payloads"
NORMALIZED_TABLE_PAYLOADS_FILENAME = "normalized_table_payloads_v1.json"


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def env_flag(name: str, default: bool = False) -> bool:
    value = normalize_text(os.getenv(name, "")).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def doe_enumeration_mode() -> str:
    value = normalize_text(os.getenv(DOE_ENUMERATION_MODE_ENV, "")).lower()
    if value in {"off", "explicit_only"}:
        return value
    return "explicit_only"


def numbered_doe_recovery_enabled() -> bool:
    return doe_enumeration_mode() == "explicit_only" or env_flag(NUMBERED_DOE_RECOVERY_ENV, default=False)


def numbered_doe_recovery_min_rows() -> int:
    raw = normalize_text(os.getenv(NUMBERED_DOE_RECOVERY_MIN_ROWS_ENV, "8"))
    try:
        return max(1, int(raw))
    except Exception:
        return 8


def stage2_semantic_source_mode(document: dict[str, Any]) -> str:
    return normalize_text(document.get("stage2_semantic_source_mode")) or LLM_FIRST_COMPOSITE_MODE


def semantic_scope_declarations(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in ensure_list(document.get("semantic_scope_declarations")) if isinstance(item, dict)]


def resolve_llm_declared_doe_scope(document: dict[str, Any]) -> dict[str, Any] | None:
    for declaration in semantic_scope_declarations(document):
        if normalize_text(declaration.get("scope_kind")) != DOE_SCOPE_KIND:
            continue
        if normalize_text(declaration.get("declared_by")) not in {LLM_SEMANTIC_DISCOVERY, LLM_PARSED}:
            continue
        modes = {
            normalize_text(mode)
            for mode in ensure_list(declaration.get("authorizes_row_materialization_modes"))
            if normalize_text(mode)
        }
        if ROW_MATERIALIZATION_MODE not in modes:
            continue
        if normalize_text(declaration.get("row_enumeration_required")).lower() != "yes":
            continue
        return declaration
    return None


def resolve_document_text_path(document: dict[str, Any]) -> Path | None:
    text_path = normalize_text(document.get("source_text_path"))
    if not text_path:
        return None
    path = Path(text_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _normalize_table_label(value: Any) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _table_scope_records(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in ensure_list(document.get("table_formulation_scopes")) if isinstance(item, dict)]


def _boundary_marker_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    markers: dict[str, dict[str, Any]] = {}
    for marker in ensure_list(document.get("boundary_markers")):
        if not isinstance(marker, dict):
            continue
        table_id = normalize_text(marker.get("table_id"))
        if not table_id:
            continue
        markers[_normalize_table_label(table_id)] = marker
    return markers


def _variable_role_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for role in ensure_list(document.get("table_variable_roles")):
        if not isinstance(role, dict):
            continue
        table_id = normalize_text(role.get("table_id"))
        if not table_id:
            continue
        roles[_normalize_table_label(table_id)] = role
    return roles


def _resolve_semantic_stage2_root(document: dict[str, Any]) -> Path | None:
    raw_response_path = normalize_text(document.get("source_raw_response_path"))
    if raw_response_path:
        candidate = Path(raw_response_path)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        parent = candidate.parent
        if parent.name == "raw_responses":
            return parent.parent
    return None


def _load_normalized_table_payloads(document: dict[str, Any]) -> list[dict[str, Any]]:
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    if not document_key:
        return []
    semantic_root = _resolve_semantic_stage2_root(document)
    if semantic_root is None:
        return []
    manifest_path = semantic_root / NORMALIZED_TABLE_PAYLOADS_SUBDIR / document_key / NORMALIZED_TABLE_PAYLOADS_FILENAME
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [item for item in ensure_list(payload.get("normalized_table_payloads")) if isinstance(item, dict)]


def _resolve_normalized_payload_for_scope(
    *,
    matching_scope: dict[str, Any],
    normalized_payloads: list[dict[str, Any]],
) -> dict[str, Any] | None:
    wanted_table_path = normalize_text(matching_scope.get("table_path")).replace("\\", "/").lower()
    wanted_asset_id = normalize_text(matching_scope.get("table_asset_id")).lower()
    wanted_table_id = _normalize_table_label(matching_scope.get("table_id"))
    for item in normalized_payloads:
        source_csv_path = normalize_text(item.get("source_csv_path")).replace("\\", "/").lower()
        source_table_id = _normalize_table_label(item.get("source_table_id"))
        source_asset_id = normalize_text(item.get("source_table_asset_id") or item.get("table_asset_id")).lower()
        if wanted_table_path and source_csv_path == wanted_table_path:
            return item
        if wanted_asset_id and source_asset_id == wanted_asset_id:
            return item
        if wanted_table_id and source_table_id == wanted_table_id:
            return item
    return None


def resolve_authorized_doe_targets(
    document: dict[str, Any],
    semantic_scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    carrier_refs = [normalize_text(item) for item in ensure_list(semantic_scope.get("table_scope_refs")) if normalize_text(item)]
    boundary_map = _boundary_marker_map(document)
    variable_role_map = _variable_role_map(document)
    normalized_scopes = _table_scope_records(document)
    normalized_payloads = _load_normalized_table_payloads(document)
    targets: list[dict[str, Any]] = []
    unresolved_refs: list[str] = []
    for ref in carrier_refs:
        wanted = _normalize_table_label(ref)
        matching_scope = None
        for scope in normalized_scopes:
            if _normalize_table_label(scope.get("table_id")) == wanted:
                matching_scope = scope
                break
        if matching_scope is None:
            unresolved_refs.append(ref)
            continue
        boundary = boundary_map.get(wanted)
        if boundary is None or not bool(boundary.get("is_doe")):
            unresolved_refs.append(ref)
            continue
        table_asset_id = normalize_text(matching_scope.get("table_asset_id"))
        if not table_asset_id and not normalize_text(matching_scope.get("table_id")):
            unresolved_refs.append(ref)
            continue
        normalized_payload = _resolve_normalized_payload_for_scope(
            matching_scope=matching_scope,
            normalized_payloads=normalized_payloads,
        )
        normalized_csv_path = normalize_text(normalized_payload.get("normalized_csv_path")) if normalized_payload else ""
        if not normalized_payload or not normalized_csv_path:
            unresolved_refs.append(ref)
            continue
        execution_table_path = normalized_csv_path
        targets.append(
            {
                "table_id": normalize_text(matching_scope.get("table_id")) or ref,
                "table_path": execution_table_path,
                "source_table_path": normalize_text(matching_scope.get("table_path")),
                "table_asset_id": table_asset_id,
                "table_type": normalize_text(matching_scope.get("table_type")),
                "evidence_span": normalize_text(matching_scope.get("evidence_span")),
                "scope_id": normalize_text(matching_scope.get("scope_id")),
                "variable_role_present": "yes" if wanted in variable_role_map else "no",
                "normalized_payload_path": normalized_csv_path,
                "normalized_payload_used": "yes" if normalized_csv_path else "no",
            }
        )
    binding = {
        "authorized_target_carrier": "|".join(carrier_refs),
        "resolved_execution_target": "|".join(
            f"{normalize_text(item.get('table_id'))}:{normalize_text(item.get('table_path')) or normalize_text(item.get('table_asset_id'))}"
            for item in targets
        ),
        "binding_success": bool(targets),
        "unresolved_authorized_target_refs": "|".join(unresolved_refs),
        "normalized_payload_used": "yes" if any(normalize_text(item.get("normalized_payload_used")) == "yes" for item in targets) else "no",
    }
    return targets, binding


def build_existing_forms_for_recovery(document: dict[str, Any]) -> list[dict[str, Any]]:
    existing_forms: list[dict[str, Any]] = []
    for item in ensure_list(document.get("formulation_identity_candidates")):
        if not isinstance(item, dict):
            continue
        raw_label = normalize_text(
            item.get("raw_formulation_label")
            or item.get("raw_label")
            or item.get("candidate_id")
            or item.get("formulation_candidate_id")
        )
        formulation_id = normalize_text(
            item.get("formulation_candidate_id")
            or item.get("candidate_id")
            or raw_label
        )
        if not formulation_id and not raw_label:
            continue
        existing_forms.append(
            {
                "raw_formulation_label": raw_label,
                "formulation_id": formulation_id or raw_label,
                "candidate_source": "llm_extracted",
            }
        )
    return existing_forms


def is_governed_doe_recovery_candidate_source(value: str) -> bool:
    return normalize_text(value) in {"doe_numbered_table_row", RECOVERY_CANDIDATE_SOURCE}


def _parse_numeric_label(raw_label: str) -> str:
    match = re.fullmatch(r"(\d{1,3})\s*\.?", normalize_text(raw_label))
    if not match:
        return ""
    return str(int(match.group(1)))


def prefer_governed_doe_rows_over_llm_numeric_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    doe_numbers = {
        _parse_numeric_label(row.get("raw_formulation_label", ""))
        for row in rows
        if is_governed_doe_recovery_candidate_source(row.get("candidate_source", ""))
    }
    doe_numbers.discard("")
    if not doe_numbers:
        return rows, []

    preferred: list[dict[str, str]] = []
    suppressed: list[dict[str, str]] = []
    for row in rows:
        numeric_label = _parse_numeric_label(row.get("raw_formulation_label", ""))
        if (
            row.get("row_materialization_mode") == LLM_SEMANTIC_DISCOVERY
            and not is_governed_doe_recovery_candidate_source(row.get("candidate_source", ""))
            and numeric_label in doe_numbers
        ):
            suppressed.append(row)
            continue
        preferred.append(row)
    return preferred, suppressed


def build_governed_numbered_doe_guard_row(
    *,
    document: dict[str, Any],
    rows: list[dict[str, str]],
    recovery_summary: dict[str, Any] | None,
) -> dict[str, str]:
    pseudo_forms: list[dict[str, Any]] = []
    for row in rows:
        candidate_source = normalize_text(row.get("candidate_source"))
        if is_governed_doe_recovery_candidate_source(candidate_source):
            guard_source = "doe_numbered_table_row"
        elif normalize_text(row.get("row_materialization_mode")) == LLM_SEMANTIC_DISCOVERY:
            guard_source = "llm_extracted"
        else:
            guard_source = candidate_source
        pseudo_forms.append(
            {
                "raw_formulation_label": normalize_text(row.get("raw_formulation_label")),
                "instance_kind": normalize_text(row.get("instance_kind")),
                "candidate_source": guard_source,
            }
        )
    return build_numbered_doe_guard_row(
        paper={
            "doi": normalize_text(document.get("doi")),
            "key": normalize_text(document.get("document_key") or document.get("key")),
        },
        forms=pseudo_forms,
        doe_summary=recovery_summary,
    )


def run_doe_row_expansion_function_unit(
    *,
    document: dict[str, Any],
    model_name: str,
    semantic_scope: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    source_mode = stage2_semantic_source_mode(document)
    if source_mode != LLM_FIRST_COMPOSITE_MODE:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "document_key": document_key,
            "considered": True,
            "authorized": False,
            "called": False,
            "emitted_row_count": 0,
            "retained_row_count": 0,
            "skip_reason": f"semantic_source_mode_not_llm_first:{source_mode}",
            "enabled": False,
            "mode": doe_enumeration_mode(),
            "candidate_count": 0,
            "row_count": 0,
            "table_count": 0,
            "tables_dir": "",
            "notes": f"semantic_source_mode_not_llm_first:{source_mode}",
        }
    if not numbered_doe_recovery_enabled():
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "document_key": document_key,
            "considered": True,
            "authorized": False,
            "called": False,
            "emitted_row_count": 0,
            "retained_row_count": 0,
            "skip_reason": "recovery_disabled",
            "enabled": False,
            "mode": doe_enumeration_mode(),
            "candidate_count": 0,
            "row_count": 0,
            "table_count": 0,
            "tables_dir": "",
            "notes": "recovery_disabled",
        }
    if not semantic_scope:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "document_key": document_key,
            "considered": True,
            "authorized": False,
            "called": False,
            "emitted_row_count": 0,
            "retained_row_count": 0,
            "skip_reason": "missing_llm_declared_doe_scope",
            "enabled": True,
            "mode": doe_enumeration_mode(),
            "candidate_count": 0,
            "row_count": 0,
            "table_count": 0,
            "tables_dir": "",
            "notes": "missing_llm_declared_doe_scope",
        }
    authorized_targets, binding = resolve_authorized_doe_targets(document, semantic_scope)
    if not authorized_targets:
        notes = "authorized_target_unresolved"
        if binding["unresolved_authorized_target_refs"]:
            notes = f"{notes}:{binding['unresolved_authorized_target_refs']}"
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "document_key": document_key,
            "considered": True,
            "authorized": True,
            "called": False,
            "emitted_row_count": 0,
            "retained_row_count": 0,
            "skip_reason": "authorized_target_unresolved",
            "enabled": True,
            "mode": doe_enumeration_mode(),
            "candidate_count": 0,
            "row_count": 0,
            "table_count": 0,
            "tables_dir": "",
            "semantic_scope_ref": normalize_text(semantic_scope.get("scope_id")) or f"{document_key}__llm_declared_doe_scope__01",
            "authorized_target_carrier": binding["authorized_target_carrier"],
            "resolved_execution_target": "",
            "binding_success": "no",
            "doe_expansion_attempted": "no",
            "normalized_payload_used": "no",
            "notes": notes,
        }

    text_path = resolve_document_text_path(document)
    raw_text = ""
    text_mode = "full_text_available"
    if text_path is not None and text_path.exists():
        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
    else:
        text_mode = "table_assets_only_missing_source_text_path"
        text_path = REPO_ROOT / "data" / "cleaned" / "__missing_text__" / f"{document_key}.txt"

    paper = PaperRecord(
        key=document_key,
        doi=normalize_text(document.get("doi")),
        title=normalize_text(document.get("title")),
        text_path=text_path,
    )
    emitted_forms, artifact_rows, summary = enumerate_numbered_doe_candidates_for_explicit_tables(
        paper=paper,
        raw_text=raw_text,
        explicit_targets=authorized_targets,
        existing_forms=build_existing_forms_for_recovery(document),
        min_numbered_rows=numbered_doe_recovery_min_rows(),
    )

    scope_ref = normalize_text(semantic_scope.get("scope_id")) or f"{document_key}__llm_declared_doe_scope__01"
    rows: list[dict[str, str]] = []
    traces: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []
    for candidate, artifact_row in zip(emitted_forms, artifact_rows):
        instance_evidence = candidate.get("instance_evidence") if isinstance(candidate, dict) else {}
        if not isinstance(instance_evidence, dict):
            instance_evidence = {}
        row_form = dict(candidate)
        row_form["instance_kind_reconciliation_note"] = "deterministic_explicit_numbered_doe_row_recovery_v1"
        row_form["candidate_source"] = RECOVERY_CANDIDATE_SOURCE
        row_form["stage2_semantic_source_mode"] = LLM_FIRST_COMPOSITE_MODE
        row_form["semantic_universe_authority"] = LLM_SEMANTIC_DISCOVERY
        row_form["row_materialization_mode"] = ROW_MATERIALIZATION_MODE
        row_form["semantic_scope_authority"] = LLM_DECLARED_SCOPE
        row_form["semantic_scope_ref"] = scope_ref
        row = flatten_row(document_key, paper.doi, model_name, row_form, instance_evidence)
        rows.append(row)
        traces.append(
            {
                "document_key": document_key,
                "formulation_id": normalize_text(row.get("formulation_id")),
                "legacy_field": "numbered_doe_row_recovery",
                "source_replacement_objects": json.dumps(
                    [
                        {
                            "table_id": normalize_text(artifact_row.get("table_id")),
                            "table_csv_path": normalize_text(artifact_row.get("table_csv_path")),
                            "formulation_label": normalize_text(artifact_row.get("formulation_label")),
                            "evidence_snippet": normalize_text(artifact_row.get("evidence_snippet")),
                        }
                    ],
                    ensure_ascii=False,
                ),
                "mapping_status": "direct",
                "direct_or_derived": "direct",
                "notes": "Recovered only from explicit numbered DOE table rows present in Stage1 table assets within LLM-declared DOE scope.",
            }
        )
        jsonl_rows.append({"key": document_key, "doi": paper.doi, "formulation_id": normalize_text(row.get("formulation_id")), "legacy_row": row})

    recovery_summary = dict(summary)
    recovery_summary.update(
        {
            "function_unit": FUNCTION_UNIT_ID,
            "document_key": document_key,
            "considered": True,
            "authorized": True,
            "called": True,
            "emitted_row_count": len(rows),
            "retained_row_count": len(rows),
            "skip_reason": "" if rows else "no_rows_emitted",
            "enabled": True,
            "mode": doe_enumeration_mode(),
            "candidate_count": len(rows),
            "row_count": len(rows),
            "table_count": int(recovery_summary.get("selected_table_count", 0) or 0),
            "tables_dir": normalize_text(recovery_summary.get("tables_dir")),
            "semantic_scope_ref": scope_ref,
            "authorized_target_carrier": binding["authorized_target_carrier"],
            "resolved_execution_target": binding["resolved_execution_target"],
            "binding_success": "yes" if binding["binding_success"] else "no",
            "doe_expansion_attempted": "yes",
            "normalized_payload_used": binding["normalized_payload_used"],
            "text_mode": text_mode,
        }
    )
    if not rows:
        recovery_summary["notes"] = (
            f"{normalize_text(recovery_summary.get('notes'))} "
            f"Authorized binding succeeded against {binding['resolved_execution_target']} but no rows were emitted."
        ).strip()
    return rows, traces, jsonl_rows, recovery_summary


__all__ = [
    "FUNCTION_UNIT_ID",
    "RECOVERY_CANDIDATE_SOURCE",
    "ROW_MATERIALIZATION_MODE",
    "build_governed_numbered_doe_guard_row",
    "doe_enumeration_mode",
    "is_governed_doe_recovery_candidate_source",
    "numbered_doe_recovery_enabled",
    "numbered_doe_recovery_min_rows",
    "prefer_governed_doe_rows_over_llm_numeric_rows",
    "resolve_llm_declared_doe_scope",
    "run_doe_row_expansion_function_unit",
    "write_numbered_doe_guard_artifact",
]
