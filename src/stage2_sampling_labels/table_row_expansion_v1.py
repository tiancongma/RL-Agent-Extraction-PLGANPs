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
DOE_SCOPE_KIND = "doe_table_row_enumeration_scope"
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
REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_TABLE_PAYLOADS_SUBDIR = "normalized_table_payloads"
NORMALIZED_TABLE_PAYLOADS_FILENAME = "normalized_table_payloads_v1.json"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_token(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9%:/.+-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _normalize_table_label(value: Any) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


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


def is_llm_first_document(document: dict[str, Any]) -> bool:
    return normalize_text(document.get("stage2_semantic_source_mode")) == "llm_first_composite"


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


def marker_provenance(marker: dict[str, Any], *, document: dict[str, Any] | None = None) -> str:
    provenance = normalize_text(marker.get("marker_provenance"))
    if provenance in LLM_MARKER_SOURCES:
        return provenance
    # Backward-compatibility for older llm-first replay payloads that carried
    # governed table markers without the explicit provenance field.
    if document and is_llm_first_document(document):
        return "llm_parsed"
    return ""


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
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    table_id = normalize_text(scope.get("table_id"))
    scope_id = normalize_text(scope.get("scope_id"))
    if not scope_id and document_key and table_id:
        scope_id = f"{document_key}__table_formulation_scope__{normalize_token(table_id)}"
    normalized = {
        "scope_id": scope_id,
        "table_id": table_id,
        "table_path": normalize_text(scope.get("table_path")),
        "table_asset_id": normalize_text(scope.get("table_asset_id")),
        "variable_name": normalize_text(scope.get("variable_name")),
        "candidate_values": [
            normalize_text(item)
            for item in ensure_list(scope.get("candidate_values"))
            if normalize_text(item)
        ],
        "is_formulation_table": bool(scope.get("is_formulation_table")),
        "table_type": normalize_text(scope.get("table_type")),
        "confidence": normalize_text(scope.get("confidence")),
        "evidence_span": normalize_text(scope.get("evidence_span")),
        "marker_provenance": marker_provenance(scope, document=document),
    }
    if not normalized["table_path"]:
        payload = resolve_table_authority_payload_for_scope(
            normalized,
            normalized_payloads=_load_normalized_table_payloads(document),
        )
        if payload is not None:
            normalized["table_path"] = normalize_text(payload.get("normalized_csv_path"))
            normalized["table_asset_id"] = normalize_text(normalized["table_asset_id"]) or normalize_text(
                payload.get("source_table_asset_id")
            )
    return normalized


def normalize_variable_role(role: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": normalize_text(role.get("table_id")),
        "varying_variables": [normalize_text(item) for item in ensure_list(role.get("varying_variables")) if normalize_text(item)],
        "constant_variables": [normalize_text(item) for item in ensure_list(role.get("constant_variables")) if normalize_text(item)],
        "new_variables_introduced": [normalize_text(item) for item in ensure_list(role.get("new_variables_introduced")) if normalize_text(item)],
        "variable_source": normalize_text(role.get("variable_source")),
        "marker_provenance": "",
    }


def normalize_selection_marker(marker: dict[str, Any], *, document: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "source_table_id": normalize_text(marker.get("source_table_id")),
        "selected_variable": normalize_text(marker.get("selected_variable")),
        "selected_value": normalize_text(marker.get("selected_value")),
        "explicit": bool(marker.get("explicit")),
        "evidence_span": normalize_text(marker.get("evidence_span")),
        "marker_provenance": marker_provenance(marker, document=document),
        MARKER_READINESS_FIELD: normalize_marker_readiness(marker, family="selection"),
    }
    if normalized[MARKER_READINESS_FIELD] == PARTIAL_SEMANTIC_MARKER:
        normalized[RISK_LABEL_FIELD] = REVIEW_RISK_LABEL
        normalized[RISK_REASON_FIELD] = selection_marker_risk_reason(normalized)
    return normalized


def normalize_inheritance_marker(marker: dict[str, Any], *, document: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "from_table": normalize_text(marker.get("from_table")),
        "to_table": normalize_text(marker.get("to_table")),
        "inherit_type": normalize_text(marker.get("inherit_type")),
        "variable": normalize_text(marker.get("variable")),
        "value": normalize_text(marker.get("value")),
        "evidence_span": normalize_text(marker.get("evidence_span")),
        "marker_provenance": marker_provenance(marker, document=document),
        MARKER_READINESS_FIELD: normalize_marker_readiness(marker, family="inheritance"),
    }
    if normalized[MARKER_READINESS_FIELD] == PARTIAL_SEMANTIC_MARKER:
        normalized[RISK_LABEL_FIELD] = REVIEW_RISK_LABEL
        normalized[RISK_REASON_FIELD] = inheritance_marker_risk_reason(normalized)
    return normalized


def normalize_preparation_marker(marker: dict[str, Any], *, document: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": normalize_text(marker.get("table_id")),
        "inherits_from_preparation": bool(marker.get("inherits_from_preparation")),
        "evidence_span": normalize_text(marker.get("evidence_span")),
        "marker_provenance": marker_provenance(marker, document=document),
    }


def normalize_boundary_marker(marker: dict[str, Any], *, document: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": normalize_text(marker.get("table_id")),
        "is_doe": bool(marker.get("is_doe")),
        "marker_provenance": marker_provenance(marker, document=document),
    }


def resolve_table_path_for_id(table_id: str, document: dict[str, Any]) -> Path | None:
    wanted = normalize_text(table_id).lower()
    if not wanted:
        return None
    wanted_number_match = re.search(r"\btable\s+(\d+)\b", wanted, flags=re.IGNORECASE)
    wanted_number = str(int(wanted_number_match.group(1))) if wanted_number_match else ""
    for path in source_table_paths(document):
        rows = read_csv_rows(path)
        label = extract_table_label(path, rows).lower()
        if label == wanted:
            return path
        if wanted_number:
            label_number_match = re.search(r"\btable\s+(\d+)\b", label, flags=re.IGNORECASE)
            if label_number_match and str(int(label_number_match.group(1))) == wanted_number:
                return path
            stem_number_match = re.search(r"__table_(\d+)__", path.name, flags=re.IGNORECASE)
            if stem_number_match and str(int(stem_number_match.group(1))) == wanted_number:
                return path
    return None


def resolve_table_authority_payload_for_scope(
    scope: dict[str, Any],
    *,
    normalized_payloads: list[dict[str, Any]],
) -> dict[str, Any] | None:
    wanted_table_id = _normalize_table_label(scope.get("table_id"))
    wanted_table_path = normalize_text(scope.get("table_path")).replace("\\", "/").lower()
    wanted_asset_id = normalize_text(scope.get("table_asset_id")).lower()
    for item in normalized_payloads:
        payload_table_id = _normalize_table_label(item.get("table_id") or item.get("source_table_id"))
        payload_source_ref = normalize_text(
            item.get("source_table_reference") or item.get("source_csv_path")
        ).replace("\\", "/").lower()
        payload_asset_id = normalize_text(item.get("source_table_asset_id") or item.get("table_asset_id")).lower()
        if wanted_table_id and payload_table_id == wanted_table_id:
            return item
        if wanted_table_path and payload_source_ref == wanted_table_path:
            return item
        if wanted_asset_id and payload_asset_id == wanted_asset_id:
            return item
    return None


def authority_row_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [item for item in ensure_list(payload.get("normalized_rows")) if isinstance(item, dict)]
    if rows:
        return rows
    matrix = [row for row in ensure_list(payload.get("normalized_matrix")) if isinstance(row, list)]
    if not matrix:
        normalized_csv_path = normalize_text(payload.get("normalized_csv_path"))
        if normalized_csv_path:
            csv_path = Path(normalized_csv_path)
            if not csv_path.is_absolute():
                csv_path = (REPO_ROOT / csv_path).resolve()
            if csv_path.exists():
                matrix = read_csv_rows(csv_path)
    header_structure = payload.get("header_structure") if isinstance(payload.get("header_structure"), dict) else {}
    header_row_count = int(header_structure.get("header_row_count") or 0)
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(matrix[header_row_count:], start=header_row_count + 1):
        cells = [normalize_text(cell) for cell in row]
        entries.append(
            {
                "row_index": index,
                "row_number": "",
                "cells": cells,
                "row_text": " | ".join(value for value in cells if value),
            }
        )
    return entries


def infer_table_scopes_from_table_anchored_formulations(document: dict[str, Any]) -> list[dict[str, Any]]:
    if not is_llm_first_document(document):
        return []
    if any(isinstance(item, dict) for item in ensure_list(document.get("table_formulation_scopes"))):
        return []

    evidence_by_id = {
        normalize_text(item.get("span_id") or item.get("evidence_span_id")): item
        for item in ensure_list(document.get("evidence_spans"))
        if isinstance(item, dict) and normalize_text(item.get("span_id") or item.get("evidence_span_id"))
    }
    doe_scope_table_refs = {
        normalize_text(ref)
        for declaration in ensure_list(document.get("semantic_scope_declarations"))
        if isinstance(declaration, dict)
        and normalize_text(declaration.get("scope_kind")) == DOE_SCOPE_KIND
        for ref in ensure_list(declaration.get("table_scope_refs"))
        if normalize_text(ref)
    }
    doe_scope_table_numbers = {
        str(int(match.group(1)))
        for ref in doe_scope_table_refs
        for match in [re.search(r"\btable\s+(\d+)\b", ref, flags=re.IGNORECASE)]
        if match
    }
    for variable in ensure_list(document.get("variable_candidates")):
        if not isinstance(variable, dict):
            continue
        if normalize_text(variable.get("variable_role")) != "doe_factor":
            continue
        for span_id in ensure_list(variable.get("evidence_span_ids")):
            span = evidence_by_id.get(normalize_text(span_id))
            if not span:
                continue
            locator = normalize_text(span.get("source_locator_text"))
            match = re.search(r"\btable\s+(\d+)\b", locator, flags=re.IGNORECASE)
            if match:
                doe_scope_table_numbers.add(str(int(match.group(1))))
    table_hits: dict[str, dict[str, Any]] = {}
    for candidate in ensure_list(document.get("formulation_candidates")):
        if not isinstance(candidate, dict):
            continue
        candidate_id = normalize_text(candidate.get("candidate_id"))
        if not candidate_id:
            continue
        for span_id in ensure_list(candidate.get("evidence_span_ids")):
            span = evidence_by_id.get(normalize_text(span_id))
            if not span:
                continue
            region = normalize_text(span.get("source_region_type")).lower()
            locator = normalize_text(span.get("source_locator_text"))
            if region not in {"table_row", "table_cell"}:
                continue
            match = re.search(r"\btable\s+(\d+)\b", locator, flags=re.IGNORECASE)
            if not match:
                continue
            table_number = str(int(match.group(1)))
            table_id = f"Table {table_number}"
            if (
                table_id in doe_scope_table_refs
                or locator in doe_scope_table_refs
                or table_number in doe_scope_table_numbers
            ):
                continue
            bucket = table_hits.setdefault(
                table_id,
                {
                    "candidate_ids": set(),
                    "evidence_span": normalize_text(span.get("supporting_text")) or locator,
                },
            )
            bucket["candidate_ids"].add(candidate_id)

    inferred: list[dict[str, Any]] = []
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    next_index = 1
    for table_id in sorted(table_hits):
        candidate_ids = table_hits[table_id]["candidate_ids"]
        if len(candidate_ids) < 2:
            continue
        table_path = resolve_table_path_for_id(table_id, document)
        if table_path is None:
            continue
        inferred.append(
            {
                "scope_id": f"{document_key}__table_formulation_scope__{next_index:02d}",
                "table_id": table_id,
                "table_path": str(table_path),
                "table_asset_id": table_path.stem,
                "variable_name": "",
                "candidate_values": [],
                "is_formulation_table": True,
                "table_type": "full_formulation",
                "confidence": "medium",
                "evidence_span": table_hits[table_id]["evidence_span"],
                "marker_provenance": "llm_parsed",
            }
        )
        next_index += 1
    return inferred


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
    for role in table_roles:
        role["marker_provenance"] = marker_provenance(role, document=document)
    selection_markers = [
        normalize_selection_marker(item, document=document)
        for item in ensure_list(document.get("selection_markers"))
        if isinstance(item, dict)
    ]
    inheritance_markers = [
        normalize_inheritance_marker(item, document=document)
        for item in ensure_list(document.get("inheritance_markers"))
        if isinstance(item, dict)
    ]
    preparation_markers = [
        normalize_preparation_marker(item, document=document)
        for item in ensure_list(document.get("preparation_inheritance_markers"))
        if isinstance(item, dict)
    ]
    boundary_markers = [
        normalize_boundary_marker(item, document=document)
        for item in ensure_list(document.get("boundary_markers"))
        if isinstance(item, dict)
    ]
    if not table_scopes:
        inferred_scopes = [
            normalize_table_scope(item, document=document)
            for item in infer_table_scopes_from_table_anchored_formulations(document)
        ]
        if inferred_scopes:
            table_scopes = inferred_scopes
            known_boundaries = {
                normalize_text(item.get("table_id"))
                for item in boundary_markers
                if isinstance(item, dict) and normalize_text(item.get("table_id"))
            }
            for scope in inferred_scopes:
                table_id = normalize_text(scope.get("table_id"))
                if not table_id or table_id in known_boundaries:
                    continue
                boundary_markers.append(
                    normalize_boundary_marker(
                        {
                            "table_id": table_id,
                            "is_doe": False,
                            "marker_provenance": "llm_parsed",
                        },
                        document=document,
                    )
                )
                known_boundaries.add(table_id)

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


def _extract_row_assignments_from_authority(
    row_entries: list[dict[str, Any]],
    candidate_values: list[str],
) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    seen_values: set[str] = set()
    for entry in row_entries:
        if not isinstance(entry, dict):
            continue
        cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells")) if normalize_text(cell)]
        row_text = normalize_text(entry.get("row_text")) or " ".join(cells)
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
        row_ordinal = normalize_text(entry.get("row_number")) or normalize_text(entry.get("row_index"))
        assignments.append(
            {
                "row_ordinal": row_ordinal or str(len(assignments) + 1),
                "variable_value": matched_value,
                "row_text": row_text,
            }
        )
    return assignments


def candidate_values_for_variable(document: dict[str, Any], variable_name: str, *, scope: dict[str, Any] | None = None) -> list[str]:
    wanted = normalize_text(variable_name)
    if not wanted:
        return []
    variable_records = ensure_list(document.get("variable_candidates")) or ensure_list(
        document.get("variable_or_factor_candidates")
    )
    for variable in variable_records:
        if not isinstance(variable, dict):
            continue
        variable_name_raw = normalize_text(variable.get("variable_name") or variable.get("factor_name_raw"))
        if variable_name_raw != wanted:
            continue
        values = parse_candidate_values(
            normalize_text(variable.get("value_text") or variable.get("factor_expression_raw"))
        )
        if values:
            return values
    if scope:
        scoped_values = [
            normalize_text(item)
            for item in ensure_list(scope.get("candidate_values"))
            if normalize_text(item)
        ]
        if scoped_values:
            return scoped_values
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
    normalized_payloads = _load_normalized_table_payloads(document)
    rows: list[dict[str, str]] = []
    traces: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []
    table_row_count = 0
    group_hint = f"{document_key}__table_formulation_group__01"
    table_activation_rows: list[dict[str, str]] = []
    if not scopes and normalized_payloads:
        table_activation_rows.append(
            {
                "function_unit": FUNCTION_UNIT_ID,
                "document_key": document_key,
                "table_id": "",
                "scope_id": "",
                "table_type": "",
                "marker_provenance": "",
                "considered": "yes",
                "authorized": "no",
                "called": "no",
                "rows_emitted": "0",
                "rows_retained_after_projection": "0",
                "skip_reason": "missing_table_formulation_scopes",
                "table_path": "",
                "varying_variable_count": "0",
                "varying_variables": "",
            }
        )

    for scope in scopes:
        table_id = normalize_text(scope.get("table_id"))
        activation_row = {
            "function_unit": FUNCTION_UNIT_ID,
            "document_key": document_key,
            "table_id": table_id,
            "scope_id": normalize_text(scope.get("scope_id")),
            "table_type": normalize_text(scope.get("table_type")),
            "marker_provenance": marker_provenance(scope),
            "considered": "yes",
            "authorized": "no",
            "called": "no",
            "rows_emitted": "0",
            "rows_retained_after_projection": "0",
            "skip_reason": "",
            "table_path": normalize_text(scope.get("table_path")),
            "varying_variable_count": "0",
            "varying_variables": "",
        }
        if not table_id:
            activation_row["skip_reason"] = "missing_table_id"
            table_activation_rows.append(activation_row)
            continue
        if not scope.get("is_formulation_table"):
            activation_row["skip_reason"] = "not_formulation_table"
            table_activation_rows.append(activation_row)
            continue
        if marker_provenance(scope) not in LLM_MARKER_SOURCES:
            activation_row["skip_reason"] = "scope_not_llm_authorized"
            table_activation_rows.append(activation_row)
            continue
        boundary = boundary_markers.get(table_id, {})
        if bool(boundary.get("is_doe")):
            activation_row["skip_reason"] = "blocked_by_doe_boundary"
            table_activation_rows.append(activation_row)
            continue
        authority_payload = resolve_table_authority_payload_for_scope(
            scope,
            normalized_payloads=normalized_payloads,
        )
        if authority_payload is None:
            activation_row["authorized"] = "yes"
            activation_row["skip_reason"] = "missing_table_authority_payload"
            table_activation_rows.append(activation_row)
            continue
        authority_table_id = normalize_text(authority_payload.get("table_id") or authority_payload.get("source_table_id"))
        authority_table_path = normalize_text(authority_payload.get("normalized_csv_path"))
        scope["table_path"] = authority_table_path
        if authority_table_id:
            scope["table_id"] = authority_table_id
            table_id = authority_table_id
            activation_row["table_id"] = authority_table_id
        activation_row["table_path"] = authority_table_path
        role_info = variable_roles.get(table_id, {})
        if marker_provenance(role_info) not in LLM_MARKER_SOURCES:
            activation_row["authorized"] = "yes"
            activation_row["skip_reason"] = "missing_llm_variable_roles"
            table_activation_rows.append(activation_row)
            continue
        varying_variables = [normalize_text(item) for item in ensure_list(role_info.get("varying_variables")) if normalize_text(item)]
        activation_row["authorized"] = "yes"
        activation_row["called"] = "yes"
        activation_row["varying_variable_count"] = str(len(varying_variables))
        activation_row["varying_variables"] = "|".join(varying_variables)
        if len(varying_variables) != 1:
            activation_row["skip_reason"] = f"unsupported_varying_variable_count:{len(varying_variables)}"
            table_activation_rows.append(activation_row)
            continue
        varying_variable = varying_variables[0]
        candidate_values = candidate_values_for_variable(document, varying_variable, scope=scope)
        if not candidate_values:
            activation_row["skip_reason"] = "missing_candidate_values"
            table_activation_rows.append(activation_row)
            continue
        assignment_rows = _extract_row_assignments_from_authority(
            authority_row_entries(authority_payload),
            candidate_values,
        )
        scope_id = normalize_text(scope.get("scope_id"))
        if not scope_id:
            scope_id = f"{document_key}__table_formulation_scope__{normalize_token(table_id)}"
            scope["scope_id"] = scope_id
            activation_row["scope_id"] = scope_id
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
        activation_row["called"] = "yes"
        activation_row["rows_emitted"] = str(len(assignment_rows))
        activation_row["rows_retained_after_projection"] = str(len(assignment_rows))
        activation_row["skip_reason"] = "" if assignment_rows else "no_assignment_rows_matched"
        table_activation_rows.append(activation_row)
    summary = {
        "function_unit": FUNCTION_UNIT_ID,
        "document_key": document_key,
        "considered": bool(scopes or source_table_paths(document)),
        "authorized": any(row.get("authorized") == "yes" for row in table_activation_rows),
        "called": any(row.get("called") == "yes" for row in table_activation_rows),
        "emitted_row_count": table_row_count,
        "retained_row_count": table_row_count,
        "skip_reason": "" if table_row_count else (
            next(
                (
                    row.get("skip_reason", "")
                    for row in table_activation_rows
                    if row.get("skip_reason")
                ),
                "no_table_scopes",
            )
        ),
        "document_key": document_key,
        "emitted_row_count": table_row_count,
        "table_count": sum(1 for item in scopes if bool(item.get("is_formulation_table"))),
        "group_hint": group_hint if table_row_count else "",
        "status": "emitted_rows" if table_row_count else "no_rows_emitted",
        "table_activation_rows": table_activation_rows,
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
