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


def _resolve_authority_payload_root(document: dict[str, Any]) -> Path | None:
    authority_payload_root = normalize_text(document.get("authority_payload_root"))
    if not authority_payload_root:
        return None
    path = Path(authority_payload_root)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _resolve_legacy_payload_root(document: dict[str, Any]) -> Path | None:
    raw_response_path = normalize_text(document.get("source_raw_response_path"))
    if not raw_response_path:
        return None
    candidate = Path(raw_response_path)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    parent = candidate.parent
    if parent.name == "raw_responses":
        return parent.parent / NORMALIZED_TABLE_PAYLOADS_SUBDIR
    return None


def _load_normalized_table_payloads(document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    if not document_key:
        return [], {
            "reopen_source_type": "",
            "reopen_resolution_status": "failed",
            "reopen_failure_reason": "payload_locator_missing",
            "normalized_payload_used": "no",
        }
    explicit_root = _resolve_authority_payload_root(document)
    reopen_source_type = ""
    if explicit_root is not None:
        reopen_source_type = "normalized_table_payloads_explicit"
        manifest_path = explicit_root / document_key / NORMALIZED_TABLE_PAYLOADS_FILENAME
        if not manifest_path.exists():
            return [], {
                "reopen_source_type": reopen_source_type,
                "reopen_resolution_status": "failed",
                "reopen_failure_reason": "authority_root_missing",
                "normalized_payload_used": "no",
            }
    else:
        legacy_root = _resolve_legacy_payload_root(document)
        if legacy_root is None:
            return [], {
                "reopen_source_type": "",
                "reopen_resolution_status": "failed",
                "reopen_failure_reason": "authority_root_missing",
                "normalized_payload_used": "no",
            }
        reopen_source_type = "legacy_raw_response_derived"
        manifest_path = legacy_root / document_key / NORMALIZED_TABLE_PAYLOADS_FILENAME
    if not manifest_path.exists():
        return [], {
            "reopen_source_type": reopen_source_type,
            "reopen_resolution_status": "failed",
            "reopen_failure_reason": "authority_root_missing",
            "normalized_payload_used": "no",
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return [], {
            "reopen_source_type": reopen_source_type,
            "reopen_resolution_status": "failed",
            "reopen_failure_reason": "authority_root_missing",
            "normalized_payload_used": "no",
        }
    items = [item for item in ensure_list(payload.get("normalized_table_payloads")) if isinstance(item, dict)]
    return items, {
        "reopen_source_type": reopen_source_type,
        "reopen_resolution_status": "resolved",
        "reopen_failure_reason": "",
        "normalized_payload_used": "yes" if items else "no",
    }


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
        "source_table_asset_id": normalize_text(scope.get("source_table_asset_id")),
        "source_table_reference": normalize_text(scope.get("source_table_reference")),
        "table_scope_locators": parse_json_maybe(scope.get("table_scope_locators")) if normalize_text(scope.get("table_scope_locators")) else scope.get("table_scope_locators"),
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
        normalized_payloads, _ = _load_normalized_table_payloads(document)
        payload, _ = resolve_table_authority_payload_for_scope(normalized, normalized_payloads=normalized_payloads)
        if payload is not None:
            normalized["table_path"] = normalize_text(payload.get("normalized_csv_path"))
            normalized["table_asset_id"] = normalize_text(normalized["table_asset_id"]) or normalize_text(
                payload.get("source_table_asset_id")
            )
            normalized["source_table_asset_id"] = normalize_text(payload.get("source_table_asset_id"))
            normalized["source_table_reference"] = normalize_text(
                payload.get("source_table_reference") or payload.get("source_csv_path")
            )
            normalized["table_scope_locators"] = {
                "table_id": normalize_text(payload.get("table_id") or payload.get("source_table_id")),
                "source_table_asset_id": normalize_text(payload.get("source_table_asset_id")),
                "source_table_reference": normalize_text(payload.get("source_table_reference") or payload.get("source_csv_path")),
            }
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
) -> tuple[dict[str, Any] | None, str]:
    wanted_table_id = _normalize_table_label(scope.get("table_id"))
    scope_locators = scope.get("table_scope_locators") if isinstance(scope.get("table_scope_locators"), dict) else {}
    wanted_table_path = normalize_text(
        scope_locators.get("source_table_reference") or scope.get("source_table_reference") or scope.get("table_path")
    ).replace("\\", "/").lower()
    wanted_asset_id = normalize_text(
        scope_locators.get("source_table_asset_id") or scope.get("source_table_asset_id") or scope.get("table_asset_id")
    ).lower()
    matches: list[dict[str, Any]] = []
    for item in normalized_payloads:
        payload_table_id = _normalize_table_label(item.get("table_id") or item.get("source_table_id"))
        payload_source_ref = normalize_text(
            item.get("source_table_reference") or item.get("source_csv_path")
        ).replace("\\", "/").lower()
        payload_asset_id = normalize_text(item.get("source_table_asset_id") or item.get("table_asset_id")).lower()
        if wanted_table_id and payload_table_id == wanted_table_id:
            matches.append(item)
            continue
        if wanted_table_path and payload_source_ref == wanted_table_path:
            matches.append(item)
            continue
        if wanted_asset_id and payload_asset_id == wanted_asset_id:
            matches.append(item)
            continue
    if len(matches) > 1:
        return None, "multiple_candidate_payloads"
    if len(matches) == 1:
        return matches[0], ""
    if not any([wanted_table_id, wanted_table_path, wanted_asset_id]):
        return None, "payload_locator_missing"
    return None, "authorized_target_unresolved"


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


def normalize_minus_signs(text: Any) -> str:
    return normalize_text(text).replace("−", "-").replace("–", "-").replace("—", "-")


def parse_formulation_row_label_info(value: Any) -> dict[str, Any] | None:
    normalized = normalize_minus_signs(value)
    numeric_match = re.fullmatch(r"(\d{1,3})\s*[\.\):]?", normalized)
    if numeric_match:
        return {
            "label": normalize_text(value),
            "number": int(numeric_match.group(1)),
            "label_style": "numeric",
        }
    f_match = re.fullmatch(r"([Ff])\s*[- ]?(\d{1,3})\s*[\.\):]?", normalized)
    if f_match:
        return {
            "label": f"{f_match.group(1).upper()}{int(f_match.group(2))}",
            "number": int(f_match.group(2)),
            "label_style": "f_numeric",
        }
    return None


MEASUREMENT_HEADER_PATTERNS = [
    r"\bmean size\b",
    r"\bsize\b",
    r"\bdiameter\b",
    r"\bz-average\b",
    r"\bpdi\b",
    r"\bpi[a-z]?\b",
    r"\bpolydispersity\b",
    r"\bzeta\b",
    r"\bentrapp?ment\b",
    r"\bencapsulation\b",
    r"\bloading\b",
    r"\brecovery\b",
    r"\bdrug content\b",
    r"\bafter freeze-drying\b",
    r"\bbefore freeze-drying\b",
    r"\bmeasured responses\b",
    r"\bresponse\b",
]


def is_measurement_header(header: str) -> bool:
    low = normalize_text(header).lower()
    return any(re.search(pattern, low) for pattern in MEASUREMENT_HEADER_PATTERNS)


def normalize_assignment_name(name: str) -> str:
    return normalize_text(name)


def compatibility_field_for_assignment(name: str) -> str:
    low = normalize_assignment_name(name).lower()
    if "plga" in low or "polymer" in low:
        return "plga_mass_mg"
    if "pva" in low or "surfactant" in low or "stabilizer" in low:
        return "surfactant_concentration_text"
    if "drug" in low or "pf" in low:
        return "drug_feed_amount_text"
    return ""


def maybe_number_text(value: str) -> str:
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", normalize_text(value))
    return match.group(0).replace(",", ".") if match else ""


SINGLE_VARIABLE_STOPWORDS = {
    "amount",
    "concentration",
    "content",
    "value",
    "values",
    "variable",
    "variables",
    "phase",
    "levels",
    "level",
    "of",
    "the",
    "and",
}

SINGLE_VARIABLE_CONTRACT_PATTERNS = [
    r"only one parameter was changed in each series of experiments",
    r"only one parameter was changed",
    r"only one variable was changed",
    r"only one parameter was varied",
    r"one parameter was changed in each series",
]

FORMULATION_HEADER_NOISE_PATTERNS = [
    r"\bformulation characters?\b",
    r"\boptimized nanoparticle formulations?\b",
]


def unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        key = normalize_token(text)
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9%]+", normalize_text(text).lower())
        if token and token not in SINGLE_VARIABLE_STOPWORDS
    ]


def phrase_pattern(phrase: str) -> str:
    tokens = [re.escape(token) for token in re.findall(r"[A-Za-z0-9%]+", normalize_text(phrase))]
    if not tokens:
        return ""
    return r"[\s\-/–—]*".join(tokens)


def load_document_source_text(document: dict[str, Any]) -> str:
    text_path = normalize_text(document.get("source_text_path"))
    candidate_paths: list[Path] = []
    if text_path:
        path = Path(text_path)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        candidate_paths.append(path)
        if "content_goren_2025/text" in text_path:
            fallback = text_path.replace("content_goren_2025/text", "content/text")
            fallback_path = Path(fallback)
            if not fallback_path.is_absolute():
                fallback_path = (REPO_ROOT / fallback_path).resolve()
            candidate_paths.append(fallback_path)
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
    return ""


def is_generic_formulation_title(text: str) -> bool:
    low = normalize_text(text).lower()
    return any(re.search(pattern, low) for pattern in FORMULATION_HEADER_NOISE_PATTERNS)


def clean_formulation_header_part(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\(\s*mean\s*[±\+\-\/]*\s*sd\s*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmean\s*[±\+\-\/]*\s*sd\b", "", cleaned, flags=re.IGNORECASE)
    return normalize_text(cleaned)


def formulation_role_from_label(label: str) -> str:
    low = normalize_text(label).lower()
    if "empty" in low or "drug free" in low or "drug-free" in low:
        return "control"
    return "reported"


def horizontal_forward_fill(cells: list[str], *, width: int) -> list[str]:
    padded = [normalize_text(cells[idx]) if idx < len(cells) else "" for idx in range(width)]
    current = ""
    output = padded[:]
    for idx in range(1, width):
        if output[idx]:
            current = output[idx]
            continue
        if current:
            output[idx] = current
    return output


def is_pure_enumerator_row(cells: list[str]) -> bool:
    normalized = [normalize_text(cell) for cell in cells if normalize_text(cell)]
    return bool(normalized) and all(re.fullmatch(r"\d{1,3}", cell) for cell in normalized)


def expand_header_row_for_formulation_columns(cells: list[str], *, width: int) -> list[str]:
    normalized = [normalize_text(cell) for cell in cells if normalize_text(cell)]
    if not normalized or is_pure_enumerator_row(normalized):
        return [""] * width
    if len(normalized) == width:
        return normalized[:]
    if len(normalized) == width - 1:
        return [""] + normalized
    formulation_slots = max(0, width - 1)
    if formulation_slots == 0:
        return normalized[:width]
    if len(normalized) == 1:
        return [""] + [""] * formulation_slots
    if formulation_slots % len(normalized) == 0:
        span = formulation_slots // len(normalized)
        expanded = [""]
        for cell in normalized:
            expanded.extend([cell] * span)
        return (expanded + [""] * width)[:width]
    return ([""] + normalized + [""] * width)[:width]


def measurement_rows_start_index(row_entries: list[dict[str, Any]]) -> int | None:
    for idx, entry in enumerate(row_entries):
        cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
        if not cells:
            continue
        if not is_measurement_header(cells[0]):
            continue
        value_count = sum(1 for cell in cells[1:] if normalize_text(cell))
        if value_count >= 2:
            return idx
    return None


def infer_column_assignment_name(part: str, ordinal: int) -> str:
    low = normalize_text(part).lower()
    if "empty" in low or "drug loaded" in low or "drug-free" in low or "drug free" in low:
        return "loading_status"
    if "plga" in low or "pcl" in low or "polymer" in low:
        return "polymer_variant"
    return f"formulation_header_part_{ordinal}"


def extract_column_anchor_rows_from_authority(
    *,
    authority_payload: dict[str, Any],
    row_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    start_index = measurement_rows_start_index(row_entries)
    if start_index is None:
        return [], "no_measurement_axis_detected"
    if start_index < 2:
        return [], "insufficient_column_header_rows"
    width = max(len(ensure_list(entry.get("cells"))) for entry in row_entries)
    header_rows = [
        horizontal_forward_fill(
            expand_header_row_for_formulation_columns(
                [normalize_text(cell) for cell in ensure_list(entry.get("cells"))],
                width=width,
            ),
            width=width,
        )
        for entry in row_entries[:start_index]
    ]
    measurement_rows = [
        [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
        for entry in row_entries[start_index:]
        if ensure_list(entry.get("cells"))
    ]
    formulation_columns: list[dict[str, Any]] = []
    for col_idx in range(1, width):
        header_parts = unique_nonempty(
            [
                clean_formulation_header_part(header_rows[row_idx][col_idx])
                for row_idx in range(len(header_rows))
                if col_idx < len(header_rows[row_idx])
                and not is_generic_formulation_title(header_rows[row_idx][col_idx])
                and clean_formulation_header_part(header_rows[row_idx][col_idx])
            ]
        )
        if not header_parts:
            continue
        measurements: list[dict[str, str]] = []
        for row in measurement_rows:
            if col_idx >= len(row):
                continue
            measure_name = normalize_text(row[0])
            measure_value = normalize_text(row[col_idx])
            if not measure_name or not measure_value:
                continue
            measurements.append({"name": measure_name, "value": measure_value})
        if len(measurements) < 2:
            continue
        formulation_columns.append(
            {
                "column_index": col_idx,
                "header_parts": header_parts,
                "measurements": measurements,
            }
        )
    if len(formulation_columns) < 2:
        return [], "insufficient_formulation_columns"
    extracted_rows: list[dict[str, Any]] = []
    for column in formulation_columns:
        header_parts = column["header_parts"]
        label = " / ".join(header_parts)
        assignments = [
            {
                "name": infer_column_assignment_name(part, idx + 1),
                "value": part,
            }
            for idx, part in enumerate(header_parts)
        ]
        measurement_text = " | ".join(
            f"{item['name']}={item['value']}" for item in column["measurements"]
        )
        extracted_rows.append(
            {
                "label": label,
                "label_number": "",
                "row_text": measurement_text,
                "assignments": assignments,
                "instance_role": formulation_role_from_label(label),
                "measurement_summary": column["measurements"],
            }
        )
    return extracted_rows, ""


def source_text_context_window(text: str, pattern_match: re.Match[str], *, chars_before: int = 1400, chars_after: int = 250) -> str:
    start = max(0, pattern_match.start() - chars_before)
    end = min(len(text), pattern_match.end() + chars_after)
    return text[start:end]


def extract_single_variable_level_list(text: str, variable_name: str) -> list[str]:
    pattern = phrase_pattern(variable_name)
    if not pattern:
        return []
    match = re.search(pattern + r"\s*\(([^)]{3,120})\)", text, flags=re.IGNORECASE)
    if not match:
        return []
    return parse_candidate_values(match.group(1))


def extract_baseline_assignment_from_text(text: str, variable_name: str) -> str:
    tokens = content_tokens(variable_name)
    if not tokens:
        return ""
    compact_tokens = [token.replace("%", "") for token in tokens]
    best_score = -1
    best_value = ""
    for match in re.finditer(r"([A-Za-z0-9%/\- ]{0,80})\(([^)]{1,40})\)", text):
        prefix = normalize_minus_signs(match.group(1)).lower()
        prefix_compact = re.sub(r"\s+", "", prefix)
        score = 0
        for token in compact_tokens:
            if not token:
                continue
            if token in prefix_compact or re.search(rf"\b{re.escape(token)}\b", prefix):
                score += 1
        if score > best_score:
            best_score = score
            best_value = normalize_text(match.group(2))
    return best_value if best_score > 0 else ""


def build_single_variable_recovery_contract(
    *,
    document: dict[str, Any],
    require_anchor_rows: bool,
) -> dict[str, Any]:
    semantic_signals = document.get("semantic_signals") if isinstance(document.get("semantic_signals"), dict) else {}
    primary_variable_names = [
        normalize_text(item)
        for item in ensure_list(semantic_signals.get("primary_variable_names"))
        if normalize_text(item)
    ]
    if not bool(semantic_signals.get("has_variable_sweep")):
        return {
            "detected": False,
            "failure_reason": "semantic_signal_missing_variable_sweep",
        }
    if not require_anchor_rows:
        return {
            "detected": False,
            "failure_reason": "missing_explicit_anchor_rows",
        }
    if not primary_variable_names:
        return {
            "detected": False,
            "failure_reason": "missing_primary_variable_names",
        }
    source_text = load_document_source_text(document)
    if not source_text:
        return {
            "detected": False,
            "failure_reason": "source_text_missing",
        }
    contract_match = None
    for pattern in SINGLE_VARIABLE_CONTRACT_PATTERNS:
        contract_match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if contract_match:
            break
    if contract_match is None:
        return {
            "detected": False,
            "failure_reason": "single_variable_contract_not_found",
        }
    context = source_text_context_window(source_text, contract_match)
    groups: list[dict[str, Any]] = []
    baseline_assignments: dict[str, str] = {}
    for variable_name in primary_variable_names:
        levels = extract_single_variable_level_list(context, variable_name)
        baseline_value = extract_baseline_assignment_from_text(context, variable_name)
        if baseline_value:
            baseline_assignments[variable_name] = baseline_value
        if len(levels) >= 2:
            groups.append(
                {
                    "variable_name": variable_name,
                    "levels": levels,
                    "baseline_value": baseline_value,
                }
            )
    if not groups:
        return {
            "detected": False,
            "failure_reason": "single_variable_levels_not_found",
        }
    missing_baseline = [group["variable_name"] for group in groups if not group.get("baseline_value")]
    if missing_baseline:
        return {
            "detected": False,
            "failure_reason": "held_constant_context_incomplete",
            "variable_axes": missing_baseline,
        }
    return {
        "detected": True,
        "source_type": "explicit_narrative_single_variable_contract",
        "groups": groups,
        "baseline_assignments": baseline_assignments,
        "held_constant_context_source": "source_text_baseline_clause",
        "evidence_span": normalize_text(context),
    }


def emit_single_variable_recovery_rows(
    *,
    document: dict[str, Any],
    compatibility_columns: list[str],
    contract: dict[str, Any],
    scope: dict[str, Any],
    scope_id: str,
    table_id: str,
    group_hint_prefix: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]], int]:
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    doi = normalize_text(document.get("doi"))
    model_name = normalize_text(document.get("model_name") or document.get("source_mode")) or "stage2_v2_semantic_objects"
    rows: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []
    traces: list[dict[str, str]] = []
    emitted = 0
    def equivalent_level_value(left: str, right: str) -> bool:
        return re.sub(r"\s+", "", normalize_minus_signs(left).lower()) == re.sub(
            r"\s+", "", normalize_minus_signs(right).lower()
        )

    baseline_assignments = {
        normalize_text(name): normalize_text(value)
        for name, value in (contract.get("baseline_assignments") or {}).items()
        if normalize_text(name) and normalize_text(value)
    }
    evidence_span = normalize_text(contract.get("evidence_span"))
    for group in ensure_list(contract.get("groups")):
        if not isinstance(group, dict):
            continue
        variable_name = normalize_text(group.get("variable_name"))
        baseline_value = normalize_text(group.get("baseline_value"))
        if not variable_name or not baseline_value:
            continue
        group_hint = f"{group_hint_prefix}__{normalize_token(variable_name)}"
        for value in ensure_list(group.get("levels")):
            value_text = normalize_text(value)
            if not value_text:
                continue
            if equivalent_level_value(value_text, baseline_value):
                continue
            assignment_map = dict(baseline_assignments)
            assignment_map[variable_name] = value_text
            identity_variables = [
                {
                    "name": normalize_token(name),
                    "name_raw": name,
                    "value": assignment_map[name],
                    "value_raw": assignment_map[name],
                }
                for name in assignment_map
            ]
            row = {column: "" for column in compatibility_columns}
            row_label = f"{variable_name}={value_text}"
            row_id = f"{document_key}__single_variable__{normalize_token(variable_name)}__{normalize_token(value_text)}"
            row.update(
                {
                    "key": document_key,
                    "doi": doi,
                    "model": model_name,
                    "local_instance_id": row_id,
                    "formulation_id": row_id,
                    "raw_formulation_label": row_label,
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
                    "instance_evidence_region_type": "narrative_text",
                    "evidence_section": table_id or "single_variable_context",
                    "evidence_span_text": evidence_span,
                    "formulation_role": "reported",
                    "instance_context_tags": stringify_json(["single_variable_recovery"]),
                    "change_context_tags": stringify_json(["single_variable_family"]),
                    "change_descriptions": stringify_json(
                        [f"{name}={assignment_map[name]}" for name in assignment_map]
                    ),
                    "change_role": "single_variable_variation",
                    IDENTITY_VARIABLES_FIELD: stringify_json(identity_variables),
                    METHOD_GROUP_SIGNATURE_HINT_FIELD: group_hint,
                    TABLE_ID_FIELD: table_id,
                    TABLE_ROW_ID_FIELD: f"{table_id}::{normalize_token(row_label)}" if table_id else "",
                    TABLE_ASSIGNMENTS_FIELD: stringify_json([assignment_map]),
                    TABLE_SCOPE_FIELD: stringify_json(scope),
                    "supporting_evidence_refs": stringify_json(
                        [
                            {
                                "source_region_type": "narrative_text",
                                "source_locator_text": f"{table_id or 'document'}::single_variable_contract::{variable_name}",
                                "supporting_snippet": evidence_span,
                                "target_field_name": variable_name,
                            }
                        ]
                    ),
                }
            )
            for name, assignment_value in assignment_map.items():
                compat_field = compatibility_field_for_assignment(name)
                if not compat_field:
                    continue
                row[f"{compat_field}_value"] = maybe_number_text(assignment_value) or assignment_value
                row[f"{compat_field}_value_text"] = assignment_value
                row[f"{compat_field}_membership_confidence"] = "reported"
                row[f"{compat_field}_evidence_region_type"] = "narrative_text"
            rows.append(row)
            jsonl_rows.append(dict(row))
            traces.append(
                {
                    "key": document_key,
                    "local_instance_id": row_id,
                    "projection_step": FUNCTION_UNIT_ID,
                    "projection_status": "added_row",
                    "detail": f"single_variable::{variable_name}={value_text}",
                }
            )
            emitted += 1
    return rows, jsonl_rows, traces, emitted


def first_cell_value(entry: dict[str, Any]) -> str:
    cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
    return cells[0] if cells else ""


def explicit_formulation_row_entries(row_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    explicit: list[dict[str, Any]] = []
    for entry in row_entries:
        if not isinstance(entry, dict):
            continue
        label_info = parse_formulation_row_label_info(first_cell_value(entry))
        if label_info is None:
            continue
        copied = dict(entry)
        copied["row_label_info"] = label_info
        explicit.append(copied)
    return explicit


def combined_prelude_headers(
    row_entries: list[dict[str, Any]],
    *,
    first_explicit_row_index: int,
    width: int,
) -> list[str]:
    combined = [""] * width
    primary_cells: list[str] = []
    widest = -1
    for entry in row_entries:
        if int(entry.get("row_index") or 0) >= first_explicit_row_index:
            continue
        cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
        if len(cells) > widest:
            widest = len(cells)
            primary_cells = cells[:]
        for idx in range(min(width, len(cells))):
            if not cells[idx]:
                continue
            combined[idx] = f"{combined[idx]} {cells[idx]}".strip()
    resolved: list[str] = []
    for idx in range(width):
        primary = normalize_text(primary_cells[idx]) if idx < len(primary_cells) else ""
        resolved.append(primary or normalize_text(combined[idx]))
    return resolved


def infer_variable_columns_from_authority(
    row_entries: list[dict[str, Any]],
    explicit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not explicit_rows:
        return []
    width = max(len(ensure_list(row.get("cells"))) for row in explicit_rows)
    first_explicit_row_index = min(int(row.get("row_index") or 0) for row in explicit_rows)
    headers = combined_prelude_headers(
        row_entries,
        first_explicit_row_index=first_explicit_row_index,
        width=width,
    )
    if headers:
        first_header = normalize_text(headers[0]).lower()
        if first_header and not re.search(r"\b(formulation|factorial|run|sample)\b", first_header):
            headers = [""] + headers[:-1]
    variable_columns: list[dict[str, Any]] = []
    for col_idx in range(1, width):
        header = headers[col_idx] if col_idx < len(headers) else ""
        if is_measurement_header(header):
            break
        if not header:
            continue
        variable_columns.append({"column_index": col_idx, "header": header})
    return variable_columns


def extract_direct_formulation_rows_from_authority(
    *,
    authority_payload: dict[str, Any],
    row_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    explicit_rows = explicit_formulation_row_entries(row_entries)
    if len(explicit_rows) < 2:
        return [], "insufficient_explicit_row_labels"
    variable_columns = infer_variable_columns_from_authority(row_entries, explicit_rows)
    if not variable_columns:
        return [], "no_formulation_variable_columns"
    extracted_rows: list[dict[str, Any]] = []
    for entry in explicit_rows:
        cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
        assignments: list[dict[str, str]] = []
        for column in variable_columns:
            col_idx = int(column["column_index"])
            if col_idx >= len(cells):
                continue
            value = normalize_text(cells[col_idx])
            if not value:
                continue
            assignments.append(
                {
                    "name": normalize_assignment_name(column["header"]),
                    "value": value,
                }
            )
        if not assignments:
            continue
        extracted_rows.append(
            {
                "label": entry["row_label_info"]["label"],
                "label_number": entry["row_label_info"]["number"],
                "row_text": normalize_text(entry.get("row_text")) or " | ".join(value for value in cells if value),
                "assignments": assignments,
            }
        )
    if not extracted_rows:
        return [], "no_assignment_rows_matched"
    return extracted_rows, ""


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
    doe_summary: dict[str, Any] | None = None,
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
    normalized_payloads, reopen_binding = _load_normalized_table_payloads(document)
    rows: list[dict[str, str]] = []
    traces: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []
    table_row_count = 0
    group_hint = f"{document_key}__table_formulation_group__01"
    table_activation_rows: list[dict[str, str]] = []
    explicit_table_rows_emitted = 0
    single_variable_recovery_attempted = "no"
    single_variable_rows_emitted = 0
    non_doe_single_variable_groups_detected = 0
    single_variable_recovery_source_type = ""
    single_variable_recovery_failure_reason = ""
    held_constant_context_source = ""
    variable_axes_detected: list[str] = []
    single_variable_recovery_consumed = False
    doe_path_attempted = "yes" if isinstance(doe_summary, dict) and normalize_text(doe_summary.get("doe_path_attempted") or doe_summary.get("doe_expansion_attempted")) == "yes" else "no"
    doe_rows_emitted = int((doe_summary or {}).get("doe_rows_emitted") or (doe_summary or {}).get("emitted_row_count") or 0)
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
                "reopen_source_type": reopen_binding.get("reopen_source_type", ""),
                "reopen_resolution_status": reopen_binding.get("reopen_resolution_status", ""),
                "reopen_failure_reason": reopen_binding.get("reopen_failure_reason", ""),
                "normalized_payload_used": reopen_binding.get("normalized_payload_used", "no"),
                "doe_path_attempted": doe_path_attempted,
                "doe_rows_emitted": str(doe_rows_emitted),
                "fell_back_to_table_expansion": "no",
                "fallback_reason": "",
                "explicit_table_rows_emitted": "0",
                "non_doe_single_variable_groups_detected": "0",
                "single_variable_recovery_attempted": "no",
                "single_variable_rows_emitted": "0",
                "single_variable_recovery_source_type": "",
                "single_variable_recovery_failure_reason": "",
                "held_constant_context_source": "",
                "variable_axis_detected": "",
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
            "reopen_source_type": reopen_binding.get("reopen_source_type", ""),
            "reopen_resolution_status": reopen_binding.get("reopen_resolution_status", ""),
            "reopen_failure_reason": "",
            "normalized_payload_used": reopen_binding.get("normalized_payload_used", "no"),
            "doe_path_attempted": doe_path_attempted,
            "doe_rows_emitted": str(doe_rows_emitted),
            "fell_back_to_table_expansion": "no",
            "fallback_reason": "",
            "explicit_table_rows_emitted": "0",
            "non_doe_single_variable_groups_detected": "0",
            "single_variable_recovery_attempted": "no",
            "single_variable_rows_emitted": "0",
            "single_variable_recovery_source_type": "",
            "single_variable_recovery_failure_reason": "",
            "held_constant_context_source": "",
            "variable_axis_detected": "",
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
            if doe_rows_emitted > 0:
                activation_row["skip_reason"] = "blocked_by_successful_doe_emission"
                table_activation_rows.append(activation_row)
                continue
            activation_row["fell_back_to_table_expansion"] = "yes"
            activation_row["fallback_reason"] = "doe_emitted_zero_rows"
        authority_payload, payload_failure_reason = resolve_table_authority_payload_for_scope(
            scope,
            normalized_payloads=normalized_payloads,
        )
        if authority_payload is None:
            activation_row["authorized"] = "yes"
            activation_row["skip_reason"] = "missing_table_authority_payload"
            activation_row["reopen_resolution_status"] = "failed"
            activation_row["reopen_failure_reason"] = payload_failure_reason or reopen_binding.get("reopen_failure_reason", "")
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
        activation_row["reopen_resolution_status"] = "resolved"
        activation_row["normalized_payload_used"] = reopen_binding.get("normalized_payload_used", "yes")
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
        authority_entries = authority_row_entries(authority_payload)
        direct_rows, direct_failure_reason = extract_direct_formulation_rows_from_authority(
            authority_payload=authority_payload,
            row_entries=authority_entries,
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
        emitted_rows_for_scope = 0
        explicit_rows_for_scope = 0
        if direct_rows:
            for direct_row in direct_rows:
                row = {column: "" for column in compatibility_columns}
                label = normalize_text(direct_row.get("label")) or f"row_{len(rows) + 1}"
                row_id = f"{document_key}__{normalize_token(table_id)}__{normalize_token(label)}"
                assignment_map = {
                    normalize_assignment_name(item["name"]): normalize_text(item["value"])
                    for item in direct_row.get("assignments", [])
                    if normalize_assignment_name(item.get("name")) and normalize_text(item.get("value"))
                }
                identity_variables = [
                    {
                        "name": normalize_token(name),
                        "name_raw": name,
                        "value": value,
                        "value_raw": value,
                    }
                    for name, value in assignment_map.items()
                ]
                change_descriptions = [f"{name}={value}" for name, value in assignment_map.items()]
                row.update(
                    {
                        "key": document_key,
                        "doi": doi,
                        "model": model_name,
                        "local_instance_id": row_id,
                        "formulation_id": row_id,
                        "raw_formulation_label": label,
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
                        "evidence_span_text": normalize_text(direct_row.get("row_text")),
                        "formulation_role": normalize_text(direct_row.get("instance_role")) or "reported",
                        "instance_context_tags": stringify_json(["table_row_expansion", "explicit_table_anchor"]),
                        "change_context_tags": stringify_json(["table_authorized_variation"]),
                        "change_descriptions": stringify_json(change_descriptions),
                        "change_role": "table_row_variation",
                        IDENTITY_VARIABLES_FIELD: stringify_json(identity_variables),
                        METHOD_GROUP_SIGNATURE_HINT_FIELD: group_hint,
                        TABLE_ID_FIELD: table_id,
                        TABLE_ROW_ID_FIELD: f"{table_id}::{label}",
                        TABLE_ASSIGNMENTS_FIELD: stringify_json([assignment_map]),
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
                                    "source_locator_text": f"{table_id}::{label}",
                                    "supporting_snippet": normalize_text(direct_row.get("row_text")),
                                    "target_field_name": "|".join(assignment_map.keys()),
                                }
                            ]
                        ),
                    }
                )
                for name, value in assignment_map.items():
                    compat_field = compatibility_field_for_assignment(name)
                    if not compat_field:
                        continue
                    row[f"{compat_field}_value"] = maybe_number_text(value) or value
                    row[f"{compat_field}_value_text"] = value
                    row[f"{compat_field}_membership_confidence"] = "reported"
                    row[f"{compat_field}_evidence_region_type"] = "table_cell"
                rows.append(row)
                jsonl_rows.append(dict(row))
                traces.append(
                    {
                        "key": document_key,
                        "local_instance_id": row_id,
                        "projection_step": FUNCTION_UNIT_ID,
                        "projection_status": "added_row",
                        "detail": f"{table_id}::{label}",
                    }
                )
                table_row_count += 1
                emitted_rows_for_scope += 1
                explicit_table_rows_emitted += 1
                explicit_rows_for_scope += 1
        else:
            column_rows, column_failure_reason = extract_column_anchor_rows_from_authority(
                authority_payload=authority_payload,
                row_entries=authority_entries,
            )
            if column_rows:
                for column_row in column_rows:
                    row = {column: "" for column in compatibility_columns}
                    label = normalize_text(column_row.get("label")) or f"column_{len(rows) + 1}"
                    row_id = f"{document_key}__{normalize_token(table_id)}__{normalize_token(label)}"
                    assignment_map = {
                        normalize_assignment_name(item["name"]): normalize_text(item["value"])
                        for item in column_row.get("assignments", [])
                        if normalize_assignment_name(item.get("name")) and normalize_text(item.get("value"))
                    }
                    identity_variables = [
                        {
                            "name": normalize_token(name),
                            "name_raw": name,
                            "value": value,
                            "value_raw": value,
                        }
                        for name, value in assignment_map.items()
                    ]
                    measurement_summary = ensure_list(column_row.get("measurement_summary"))
                    row.update(
                        {
                            "key": document_key,
                            "doi": doi,
                            "model": model_name,
                            "local_instance_id": row_id,
                            "formulation_id": row_id,
                            "raw_formulation_label": label,
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
                            "instance_evidence_region_type": "table_column",
                            "evidence_section": table_id,
                            "evidence_span_text": normalize_text(column_row.get("row_text")),
                            "formulation_role": normalize_text(column_row.get("instance_role")) or "reported",
                            "instance_context_tags": stringify_json(["table_row_expansion", "explicit_table_anchor"]),
                            "change_context_tags": stringify_json(["column_oriented_formulation_table"]),
                            "change_descriptions": stringify_json(
                                [f"{name}={value}" for name, value in assignment_map.items()]
                            ),
                            "change_role": "table_column_variation",
                            IDENTITY_VARIABLES_FIELD: stringify_json(identity_variables),
                            METHOD_GROUP_SIGNATURE_HINT_FIELD: group_hint,
                            TABLE_ID_FIELD: table_id,
                            TABLE_ROW_ID_FIELD: f"{table_id}::{normalize_token(label)}",
                            TABLE_ASSIGNMENTS_FIELD: stringify_json([assignment_map]),
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
                                        "source_region_type": "table_column",
                                        "source_locator_text": f"{table_id}::{label}",
                                        "supporting_snippet": normalize_text(column_row.get("row_text")),
                                        "target_field_name": "|".join(assignment_map.keys()),
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
                            "detail": f"{table_id}::{label}",
                        }
                    )
                    table_row_count += 1
                    emitted_rows_for_scope += 1
                    explicit_table_rows_emitted += 1
                    explicit_rows_for_scope += 1
                direct_failure_reason = ""
            else:
                direct_failure_reason = column_failure_reason or direct_failure_reason
            if len(varying_variables) != 1:
                if emitted_rows_for_scope == 0:
                    activation_row["skip_reason"] = direct_failure_reason or f"unsupported_varying_variable_count:{len(varying_variables)}"
                    table_activation_rows.append(activation_row)
                    continue
            elif emitted_rows_for_scope == 0:
                varying_variable = varying_variables[0]
                candidate_values = candidate_values_for_variable(document, varying_variable, scope=scope)
                if not candidate_values:
                    activation_row["skip_reason"] = "missing_candidate_values"
                    table_activation_rows.append(activation_row)
                    continue
                assignment_rows = _extract_row_assignments_from_authority(
                    authority_entries,
                    candidate_values,
                )
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
                            "raw_formulation_label": f"{table_id} row {int(assignment['row_ordinal']):02d} ({varying_variable}={value})",
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
                    emitted_rows_for_scope += 1
                    explicit_table_rows_emitted += 1
                    explicit_rows_for_scope += 1
        if (
            emitted_rows_for_scope > 0
            and not single_variable_recovery_consumed
            and doe_rows_emitted == 0
        ):
            single_variable_recovery_attempted = "yes"
            single_variable_contract = build_single_variable_recovery_contract(
                document=document,
                require_anchor_rows=explicit_rows_for_scope > 0,
            )
            if bool(single_variable_contract.get("detected")):
                single_variable_recovery_source_type = normalize_text(single_variable_contract.get("source_type"))
                held_constant_context_source = normalize_text(single_variable_contract.get("held_constant_context_source"))
                groups = [
                    item for item in ensure_list(single_variable_contract.get("groups")) if isinstance(item, dict)
                ]
                non_doe_single_variable_groups_detected = len(groups)
                variable_axes_detected = [
                    normalize_text(item.get("variable_name"))
                    for item in groups
                    if normalize_text(item.get("variable_name"))
                ]
                recovered_rows, recovered_jsonl, recovered_traces, recovered_count = emit_single_variable_recovery_rows(
                    document=document,
                    compatibility_columns=compatibility_columns,
                    contract=single_variable_contract,
                    scope=scope,
                    scope_id=scope_id,
                    table_id=table_id,
                    group_hint_prefix=f"{document_key}__single_variable_group",
                )
                rows.extend(recovered_rows)
                jsonl_rows.extend(recovered_jsonl)
                traces.extend(recovered_traces)
                table_row_count += recovered_count
                emitted_rows_for_scope += recovered_count
                single_variable_rows_emitted = recovered_count
                if recovered_count == 0:
                    single_variable_recovery_failure_reason = "no_nonbaseline_levels_emitted"
                single_variable_recovery_consumed = True
            else:
                single_variable_recovery_failure_reason = normalize_text(
                    single_variable_contract.get("failure_reason")
                )
                variable_axes_detected = [
                    normalize_text(item)
                    for item in ensure_list(single_variable_contract.get("variable_axes"))
                    if normalize_text(item)
                ]
                single_variable_recovery_consumed = True
        activation_row["called"] = "yes"
        activation_row["rows_emitted"] = str(emitted_rows_for_scope)
        activation_row["rows_retained_after_projection"] = str(emitted_rows_for_scope)
        activation_row["skip_reason"] = "" if emitted_rows_for_scope else (direct_failure_reason or "no_assignment_rows_matched")
        activation_row["explicit_table_rows_emitted"] = str(explicit_rows_for_scope)
        activation_row["non_doe_single_variable_groups_detected"] = str(non_doe_single_variable_groups_detected)
        activation_row["single_variable_recovery_attempted"] = single_variable_recovery_attempted
        activation_row["single_variable_rows_emitted"] = str(single_variable_rows_emitted)
        activation_row["single_variable_recovery_source_type"] = single_variable_recovery_source_type
        activation_row["single_variable_recovery_failure_reason"] = single_variable_recovery_failure_reason
        activation_row["held_constant_context_source"] = held_constant_context_source
        activation_row["variable_axis_detected"] = "|".join(variable_axes_detected)
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
        "reopen_source_type": reopen_binding.get("reopen_source_type", ""),
        "reopen_resolution_status": (
            "resolved"
            if any(row.get("reopen_resolution_status") == "resolved" for row in table_activation_rows)
            else reopen_binding.get("reopen_resolution_status", "")
        ),
        "reopen_failure_reason": (
            next((row.get("reopen_failure_reason", "") for row in table_activation_rows if row.get("reopen_failure_reason")), "")
            or reopen_binding.get("reopen_failure_reason", "")
        ),
        "normalized_payload_used": (
            "yes"
            if any(row.get("normalized_payload_used") == "yes" for row in table_activation_rows)
            else reopen_binding.get("normalized_payload_used", "no")
        ),
        "doe_path_attempted": doe_path_attempted,
        "doe_rows_emitted": doe_rows_emitted,
        "fell_back_to_table_expansion": (
            "yes"
            if any(row.get("fell_back_to_table_expansion") == "yes" for row in table_activation_rows)
            else "no"
        ),
        "fallback_reason": next(
            (row.get("fallback_reason", "") for row in table_activation_rows if row.get("fallback_reason")),
            "",
        ),
        "explicit_table_rows_emitted": explicit_table_rows_emitted,
        "non_doe_single_variable_groups_detected": non_doe_single_variable_groups_detected,
        "single_variable_recovery_attempted": single_variable_recovery_attempted,
        "single_variable_rows_emitted": single_variable_rows_emitted,
        "single_variable_recovery_source_type": single_variable_recovery_source_type,
        "single_variable_recovery_failure_reason": single_variable_recovery_failure_reason,
        "held_constant_context_source": held_constant_context_source,
        "variable_axis_detected": "|".join(variable_axes_detected),
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
