#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


FUNCTION_UNIT_ID = "table_row_expansion_v1"
ROW_MATERIALIZATION_MODE = "table_row_expansion_v1"
RECOVERY_CANDIDATE_SOURCE = "table_row_expansion_v1"
SCOPE_KIND = "table_formulation_authorization_scope"
METHOD_GROUP_SIGNATURE_HINT_FIELD = "method_group_signature_hint"

TABLE_SCOPE_FIELD = "table_formulation_scopes_json"
TABLE_VARIABLE_ROLE_FIELD = "table_variable_roles_json"
SELECTION_MARKER_FIELD = "selection_markers_json"
INHERITANCE_MARKER_FIELD = "inheritance_markers_json"
BOUNDARY_MARKER_FIELD = "boundary_markers_json"
TABLE_ID_FIELD = "table_id"
TABLE_ROW_ID_FIELD = "table_row_id"
TABLE_ASSIGNMENTS_FIELD = "table_row_variable_assignments_json"
PREPARATION_INHERITANCE_FIELD = "preparation_inheritance_json"
IDENTITY_VARIABLES_FIELD = "identity_variables_json"

LLM_MARKER_SOURCES = {"llm_explicit", "llm_parsed"}
MARKER_READINESS_FIELD = "marker_readiness"
EXECUTION_READY_MARKER = "execution_ready"
PARTIAL_SEMANTIC_MARKER = "partial_semantic"
VALID_MARKER_READINESS = {EXECUTION_READY_MARKER, PARTIAL_SEMANTIC_MARKER}
RISK_LABEL_FIELD = "risk_label"
RISK_REASON_FIELD = "risk_reason"
REVIEW_RISK_LABEL = "review"
SELECTION_RISK_REASONS = {
    "missing_source_table",
    "missing_selected_variable",
    "missing_selected_value",
}
INHERITANCE_RISK_REASONS = {
    "missing_source_table",
    "missing_target_table",
    "cross_table_link_unresolved",
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_token(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9%:/.+-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def stringify_json(value: Any) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    text = normalize_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def source_table_paths(document: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for raw in ensure_list(document.get("source_table_files")):
        text = normalize_text(raw)
        if not text:
            continue
        path = Path(text)
        if path.exists():
            paths.append(path)
    return paths


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [[normalize_text(cell) for cell in row] for row in csv.reader(handle)]


def parse_candidate_values(value_text: str) -> list[str]:
    text = normalize_text(value_text)
    if not text:
        return []
    compact = text.replace(" and ", ",").replace(" or ", ",")
    parts = [normalize_text(part) for part in compact.split(",") if normalize_text(part)]
    shared_unit = ""
    if parts:
        last_part = parts[-1]
        unit_match = re.match(r"^[-+]?\d+(?:\.\d+)?\s*(?P<unit>.+)$", last_part)
        if unit_match:
            shared_unit = normalize_text(unit_match.group("unit"))
    seen: set[str] = set()
    values: list[str] = []
    for part in parts:
        cleaned = part.rstrip(".")
        if shared_unit and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
            cleaned = f"{cleaned} {shared_unit}"
        key = normalize_token(cleaned)
        if key and key not in seen:
            seen.add(key)
            values.append(cleaned)
    return values


def extract_table_label(table_path: Path, rows: list[list[str]]) -> str:
    for row in rows[:120]:
        joined = " ".join(cell for cell in row if cell)
        match = re.search(r"\bTable\s+\d+\b", joined, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    stem_match = re.search(r"__table_(\d+)__", table_path.name)
    if stem_match:
        return f"AssetTable {int(stem_match.group(1))}"
    return table_path.stem


def marker_provenance(marker: dict[str, Any]) -> str:
    provenance = normalize_text(marker.get("marker_provenance"))
    return provenance if provenance in LLM_MARKER_SOURCES else ""


def infer_selection_marker_readiness(marker: dict[str, Any]) -> str:
    if (
        normalize_text(marker.get("source_table_id"))
        and normalize_text(marker.get("selected_variable"))
        and normalize_text(marker.get("selected_value"))
    ):
        return EXECUTION_READY_MARKER
    return PARTIAL_SEMANTIC_MARKER


def infer_inheritance_marker_readiness(marker: dict[str, Any]) -> str:
    if (
        normalize_text(marker.get("from_table"))
        and normalize_text(marker.get("to_table"))
        and normalize_text(marker.get("inherit_type"))
        and normalize_text(marker.get("variable"))
        and normalize_text(marker.get("value"))
    ):
        return EXECUTION_READY_MARKER
    return PARTIAL_SEMANTIC_MARKER


def normalize_marker_readiness(marker: dict[str, Any], *, family: str) -> str:
    readiness = normalize_text(marker.get(MARKER_READINESS_FIELD))
    if readiness in VALID_MARKER_READINESS:
        return readiness
    if family == "selection":
        return infer_selection_marker_readiness(marker)
    if family == "inheritance":
        return infer_inheritance_marker_readiness(marker)
    return EXECUTION_READY_MARKER


def marker_is_execution_ready(marker: dict[str, Any]) -> bool:
    return normalize_text(marker.get(MARKER_READINESS_FIELD)) == EXECUTION_READY_MARKER


def execution_ready_markers(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [marker for marker in markers if isinstance(marker, dict) and marker_is_execution_ready(marker)]


def selection_marker_risk_reason(marker: dict[str, Any]) -> str:
    if not normalize_text(marker.get("source_table_id")):
        return "missing_source_table"
    if not normalize_text(marker.get("selected_variable")):
        return "missing_selected_variable"
    if not normalize_text(marker.get("selected_value")):
        return "missing_selected_value"
    return ""


def inheritance_marker_risk_reason(marker: dict[str, Any]) -> str:
    missing_from = not normalize_text(marker.get("from_table"))
    missing_to = not normalize_text(marker.get("to_table"))
    if missing_from and missing_to:
        return "cross_table_link_unresolved"
    if missing_from:
        return "missing_source_table"
    if missing_to:
        return "missing_target_table"
    return ""


def normalize_table_scope(scope: dict[str, Any], *, document: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "scope_id": normalize_text(scope.get("scope_id")),
        "table_id": normalize_text(scope.get("table_id")),
        "table_path": normalize_text(scope.get("table_path")),
        "table_asset_id": normalize_text(scope.get("table_asset_id")),
        "is_formulation_table": bool(scope.get("is_formulation_table")),
        "table_type": normalize_text(scope.get("table_type")),
        "confidence": normalize_text(scope.get("confidence")),
        "evidence_span": normalize_text(scope.get("evidence_span")),
        "marker_provenance": marker_provenance(scope),
    }
    if not normalized["table_path"]:
        table_path = resolve_table_path_for_id(normalized["table_id"], document)
        if table_path is not None:
            normalized["table_path"] = str(table_path)
            normalized["table_asset_id"] = normalize_text(normalized["table_asset_id"]) or table_path.stem
    return normalized


def normalize_variable_role(role: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": normalize_text(role.get("table_id")),
        "varying_variables": [normalize_text(item) for item in ensure_list(role.get("varying_variables")) if normalize_text(item)],
        "constant_variables": [normalize_text(item) for item in ensure_list(role.get("constant_variables")) if normalize_text(item)],
        "new_variables_introduced": [normalize_text(item) for item in ensure_list(role.get("new_variables_introduced")) if normalize_text(item)],
        "variable_source": normalize_text(role.get("variable_source")),
        "marker_provenance": marker_provenance(role),
    }


def normalize_selection_marker(marker: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "source_table_id": normalize_text(marker.get("source_table_id")),
        "selected_variable": normalize_text(marker.get("selected_variable")),
        "selected_value": normalize_text(marker.get("selected_value")),
        "explicit": bool(marker.get("explicit")),
        "evidence_span": normalize_text(marker.get("evidence_span")),
        "marker_provenance": marker_provenance(marker),
        MARKER_READINESS_FIELD: normalize_marker_readiness(marker, family="selection"),
    }
    if normalized[MARKER_READINESS_FIELD] == PARTIAL_SEMANTIC_MARKER:
        normalized[RISK_LABEL_FIELD] = REVIEW_RISK_LABEL
        normalized[RISK_REASON_FIELD] = selection_marker_risk_reason(normalized)
    return normalized


def normalize_inheritance_marker(marker: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "from_table": normalize_text(marker.get("from_table")),
        "to_table": normalize_text(marker.get("to_table")),
        "inherit_type": normalize_text(marker.get("inherit_type")),
        "variable": normalize_text(marker.get("variable")),
        "value": normalize_text(marker.get("value")),
        "evidence_span": normalize_text(marker.get("evidence_span")),
        "marker_provenance": marker_provenance(marker),
        MARKER_READINESS_FIELD: normalize_marker_readiness(marker, family="inheritance"),
    }
    if normalized[MARKER_READINESS_FIELD] == PARTIAL_SEMANTIC_MARKER:
        normalized[RISK_LABEL_FIELD] = REVIEW_RISK_LABEL
        normalized[RISK_REASON_FIELD] = inheritance_marker_risk_reason(normalized)
    return normalized


def normalize_preparation_marker(marker: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": normalize_text(marker.get("table_id")),
        "inherits_from_preparation": bool(marker.get("inherits_from_preparation")),
        "evidence_span": normalize_text(marker.get("evidence_span")),
        "marker_provenance": marker_provenance(marker),
    }


def normalize_boundary_marker(marker: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": normalize_text(marker.get("table_id")),
        "is_doe": bool(marker.get("is_doe")),
        "marker_provenance": marker_provenance(marker),
    }


def resolve_table_path_for_id(table_id: str, document: dict[str, Any]) -> Path | None:
    wanted = normalize_text(table_id).lower()
    if not wanted:
        return None
    for path in source_table_paths(document):
        rows = read_csv_rows(path)
        if extract_table_label(path, rows).lower() == wanted:
            return path
    return None


def augment_document_with_table_markers(document: dict[str, Any]) -> dict[str, Any]:
    table_scopes = [
        normalize_table_scope(item, document=document)
        for item in ensure_list(document.get("table_formulation_scopes"))
        if isinstance(item, dict)
    ]
    table_roles = [
        normalize_variable_role(item)
        for item in ensure_list(document.get("table_variable_roles"))
        if isinstance(item, dict)
    ]
    selection_markers = [
        normalize_selection_marker(item)
        for item in ensure_list(document.get("selection_markers"))
        if isinstance(item, dict)
    ]
    inheritance_markers = [
        normalize_inheritance_marker(item)
        for item in ensure_list(document.get("inheritance_markers"))
        if isinstance(item, dict)
    ]
    preparation_markers = [
        normalize_preparation_marker(item)
        for item in ensure_list(document.get("preparation_inheritance_markers"))
        if isinstance(item, dict)
    ]
    boundary_markers = [
        normalize_boundary_marker(item)
        for item in ensure_list(document.get("boundary_markers"))
        if isinstance(item, dict)
    ]

    document["table_formulation_scopes"] = table_scopes
    document["table_variable_roles"] = table_roles
    document["selection_markers"] = selection_markers
    document["preparation_inheritance_markers"] = preparation_markers
    document["inheritance_markers"] = inheritance_markers
    document["boundary_markers"] = boundary_markers
    return document


def text_matches_value(text: str, value: str) -> bool:
    normalized_text = normalize_token(text)
    normalized_value = normalize_token(value)
    if not normalized_text or not normalized_value:
        return False
    return normalized_value in normalized_text or any(
        token and token in normalized_text.split("_")
        for token in normalized_value.split("_")
    )


def _extract_row_assignments(table_path: Path, candidate_values: list[str]) -> list[dict[str, str]]:
    rows = read_csv_rows(table_path)
    assignments: list[dict[str, str]] = []
    seen_values: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        row_text = " ".join(cell for cell in row if cell)
        normalized_row_text = normalize_token(row_text)
        if not normalized_row_text or "note:" in row_text.lower():
            continue
        matched_value = ""
        for value in candidate_values:
            if text_matches_value(row_text, value):
                matched_value = value
                break
        if not matched_value:
            continue
        value_key = normalize_token(matched_value)
        if value_key in seen_values:
            continue
        seen_values.add(value_key)
        assignments.append(
            {
                "row_ordinal": str(row_index),
                "variable_value": matched_value,
                "row_text": row_text,
            }
        )
    return assignments


def candidate_values_for_variable(document: dict[str, Any], variable_name: str) -> list[str]:
    wanted = normalize_text(variable_name)
    if not wanted:
        return []
    for variable in ensure_list(document.get("variable_candidates")):
        if not isinstance(variable, dict):
            continue
        if normalize_text(variable.get("variable_name")) != wanted:
            continue
        values = parse_candidate_values(normalize_text(variable.get("value_text")))
        if values:
            return values
    return []


def run_table_row_expansion(
    *,
    document: dict[str, Any],
    compatibility_columns: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    augment_document_with_table_markers(document)
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    doi = normalize_text(document.get("doi"))
    model_name = normalize_text(document.get("model_name") or document.get("source_mode")) or "stage2_v2_semantic_objects"
    scopes = [item for item in ensure_list(document.get("table_formulation_scopes")) if isinstance(item, dict)]
    variable_roles = {
        normalize_text(item.get("table_id")): item
        for item in ensure_list(document.get("table_variable_roles"))
        if isinstance(item, dict) and normalize_text(item.get("table_id"))
    }
    boundary_markers = {
        normalize_text(item.get("table_id")): item
        for item in ensure_list(document.get("boundary_markers"))
        if isinstance(item, dict) and normalize_text(item.get("table_id"))
    }
    selection_markers = execution_ready_markers(
        [item for item in ensure_list(document.get("selection_markers")) if isinstance(item, dict)]
    )
    inheritance_markers = execution_ready_markers(
        [item for item in ensure_list(document.get("inheritance_markers")) if isinstance(item, dict)]
    )
    rows: list[dict[str, str]] = []
    traces: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []
    table_row_count = 0
    group_hint = f"{document_key}__table_formulation_group__01"

    for scope in scopes:
        table_id = normalize_text(scope.get("table_id"))
        if not table_id or not scope.get("is_formulation_table"):
            continue
        if marker_provenance(scope) not in LLM_MARKER_SOURCES:
            continue
        boundary = boundary_markers.get(table_id, {})
        if bool(boundary.get("is_doe")):
            continue
        table_path_text = normalize_text(scope.get("table_path"))
        if not table_path_text:
            continue
        table_path = Path(table_path_text)
        if not table_path.exists():
            continue
        role_info = variable_roles.get(table_id, {})
        if marker_provenance(role_info) not in LLM_MARKER_SOURCES:
            continue
        varying_variables = [normalize_text(item) for item in ensure_list(role_info.get("varying_variables")) if normalize_text(item)]
        if len(varying_variables) != 1:
            continue
        varying_variable = varying_variables[0]
        candidate_values = candidate_values_for_variable(document, varying_variable)
        if not candidate_values:
            continue
        assignment_rows = _extract_row_assignments(table_path, candidate_values)
        scope_id = normalize_text(scope.get("scope_id"))
        if not scope_id:
            continue
        table_selection_markers = [
            marker
            for marker in selection_markers
            if normalize_text(marker.get("source_table_id")) == table_id and marker_provenance(marker) in LLM_MARKER_SOURCES
        ]
        table_inheritance_markers = [
            marker
            for marker in inheritance_markers
            if (
                normalize_text(marker.get("to_table")) == table_id
                or normalize_text(marker.get("table_id")) == table_id
            )
            and marker_provenance(marker) in LLM_MARKER_SOURCES
        ]
        for assignment in assignment_rows:
            row = {column: "" for column in compatibility_columns}
            row_id = f"{document_key}__{normalize_token(table_id)}__row_{int(assignment['row_ordinal']):02d}"
            value = normalize_text(assignment.get("variable_value"))
            row.update(
                {
                    "key": document_key,
                    "doi": doi,
                    "model": model_name,
                    "local_instance_id": row_id,
                    "formulation_id": row_id,
                    "raw_formulation_label": f"{table_id} row {int(assignment['row_ordinal'])} ({varying_variable}={value})",
                    "instance_kind": "new_formulation",
                    "instance_kind_raw": "new_formulation",
                    "instance_kind_inferred": "new_formulation",
                    "instance_confidence": "reported",
                    "candidate_source": RECOVERY_CANDIDATE_SOURCE,
                    "stage2_semantic_source_mode": normalize_text(document.get("stage2_semantic_source_mode")),
                    "semantic_universe_authority": normalize_text(document.get("semantic_universe_authority")),
                    "row_materialization_mode": ROW_MATERIALIZATION_MODE,
                    "semantic_scope_authority": "llm_declared_scope",
                    "semantic_scope_ref": scope_id,
                    "instance_evidence_region_type": "table_row",
                    "evidence_section": table_id,
                    "evidence_span_text": normalize_text(assignment.get("row_text")),
                    "formulation_role": "reported",
                    "instance_context_tags": stringify_json(["table_row_expansion"]),
                    "change_context_tags": stringify_json(["table_authorized_variation"]),
                    "change_descriptions": stringify_json([f"{varying_variable}={value}"]),
                    "change_role": "table_row_variation",
                    IDENTITY_VARIABLES_FIELD: stringify_json(
                        [
                            {
                                "name": normalize_token(varying_variable),
                                "name_raw": varying_variable,
                                "value": value,
                                "value_raw": value,
                            }
                        ]
                    ),
                    METHOD_GROUP_SIGNATURE_HINT_FIELD: group_hint,
                    TABLE_ID_FIELD: table_id,
                    TABLE_ROW_ID_FIELD: f"{table_id}::row_{int(assignment['row_ordinal']):02d}",
                    TABLE_ASSIGNMENTS_FIELD: stringify_json([{varying_variable: value}]),
                    TABLE_SCOPE_FIELD: stringify_json(scope),
                    TABLE_VARIABLE_ROLE_FIELD: stringify_json(role_info),
                    SELECTION_MARKER_FIELD: stringify_json(table_selection_markers),
                    INHERITANCE_MARKER_FIELD: stringify_json(table_inheritance_markers),
                    BOUNDARY_MARKER_FIELD: stringify_json(boundary),
                    PREPARATION_INHERITANCE_FIELD: stringify_json(
                        [
                            marker
                            for marker in table_inheritance_markers
                            if bool(marker.get("inherits_from_preparation"))
                        ]
                    ),
                    "supporting_evidence_refs": stringify_json(
                        [
                            {
                                "source_region_type": "table_row",
                                "source_locator_text": f"{table_id}::row_{int(assignment['row_ordinal']):02d}",
                                "supporting_snippet": normalize_text(assignment.get("row_text")),
                                "target_field_name": varying_variable,
                            }
                        ]
                    ),
                }
            )
            rows.append(row)
            jsonl_rows.append(dict(row))
            traces.append(
                {
                    "key": document_key,
                    "local_instance_id": row_id,
                    "projection_step": FUNCTION_UNIT_ID,
                    "projection_status": "added_row",
                    "detail": f"{table_id}::{varying_variable}={value}",
                }
            )
            table_row_count += 1
    summary = {
        "document_key": document_key,
        "emitted_row_count": table_row_count,
        "table_count": sum(1 for item in scopes if bool(item.get("is_formulation_table"))),
        "group_hint": group_hint if table_row_count else "",
        "status": "emitted_rows" if table_row_count else "no_rows_emitted",
    }
    return rows, traces, jsonl_rows, summary


def mark_llm_summary_rows_as_helpers(rows: list[dict[str, str]], jsonl_rows: list[dict[str, Any]], group_hint: str) -> None:
    def is_summary_row(row: dict[str, str]) -> bool:
        label = normalize_text(row.get("raw_formulation_label")).lower()
        identity_blob = normalize_text(row.get("identity_variables_json")).lower()
        if "optimal" in label:
            return True
        return bool(identity_blob and (", " in identity_blob or " and " in identity_blob or " or " in identity_blob))

    jsonl_by_id = {
        normalize_text(item.get("formulation_id")): item
        for item in jsonl_rows
        if isinstance(item, dict) and normalize_text(item.get("formulation_id"))
    }
    for row in rows:
        row[METHOD_GROUP_SIGNATURE_HINT_FIELD] = normalize_text(row.get(METHOD_GROUP_SIGNATURE_HINT_FIELD)) or group_hint
        if normalize_text(row.get("candidate_source")) == RECOVERY_CANDIDATE_SOURCE:
            continue
        if not is_summary_row(row):
            continue
        row["instance_kind_raw"] = "candidate_non_formulation"
        row["instance_kind_inferred"] = "candidate_non_formulation"
        row["instance_kind"] = "candidate_non_formulation"
        row["formulation_role"] = normalize_text(row.get("formulation_role")) or "unclear"
        row["change_context_tags"] = stringify_json(["table_summary_helper"])
        item = jsonl_by_id.get(normalize_text(row.get("formulation_id")))
        if isinstance(item, dict):
            item["instance_kind"] = "candidate_non_formulation"
