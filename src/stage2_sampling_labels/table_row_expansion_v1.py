#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from src.stage2_sampling_labels.table_structure_dictionary_v1 import (
    canonical_field_for_header as shared_canonical_field_for_header,
    extract_continuation_group_context,
    is_numeric_index_row,
    recover_row_entry_headers,
)


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
TABLE_CELL_BINDINGS_FIELD = "table_cell_bindings_json"
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


def canonical_field_for_header(header: str, *, paper_key: str = "") -> str:
    return shared_canonical_field_for_header(header, paper_key=paper_key)


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
        "source_text_caption_only": bool(scope.get("source_text_caption_only")),
        "table_scope_locators": parse_json_maybe(scope.get("table_scope_locators")) if normalize_text(scope.get("table_scope_locators")) else scope.get("table_scope_locators"),
        "variable_name": normalize_text(scope.get("variable_name")),
        "candidate_values": [
            normalize_text(item)
            for item in ensure_list(scope.get("candidate_values"))
            if normalize_text(item)
        ],
        "is_formulation_table": bool(scope.get("is_formulation_table")),
        "table_type": normalize_text(scope.get("table_type")),
        "parent_table_hint": normalize_text(scope.get("parent_table_hint")),
        "confidence": normalize_text(scope.get("confidence")),
        "evidence_span": normalize_text(scope.get("evidence_span")),
        "marker_provenance": marker_provenance(scope, document=document),
    }
    if not normalized["table_path"] and not normalized.get("source_text_caption_only") and normalized.get("source_table_reference") != "source_text":
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


def iter_table_scope_locator_dicts(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(value) if normalize_text(value) else value
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def table_number_aliases(*values: Any) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        normalized = normalize_token(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            aliases.append(normalized)
        for match in re.finditer(r"(?:^|[^a-z0-9])table[_\s\-]*(\d{1,3})(?:[^a-z0-9]|$)", text, flags=re.IGNORECASE):
            alias = normalize_token(f"Table {int(match.group(1))}")
            if alias and alias not in seen:
                seen.add(alias)
                aliases.append(alias)
        for match in re.finditer(r"__table_(\d{1,3})__", text, flags=re.IGNORECASE):
            alias = normalize_token(f"Table {int(match.group(1))}")
            if alias and alias not in seen:
                seen.add(alias)
                aliases.append(alias)
    return aliases


def resolve_table_authority_payload_for_scope(
    scope: dict[str, Any],
    *,
    normalized_payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    wanted_table_id = _normalize_table_label(scope.get("table_id"))
    wanted_aliases = {
        alias
        for alias in table_number_aliases(
            scope.get("table_id"),
            scope.get("parent_table_hint"),
            scope.get("evidence_span"),
            scope.get("source_snippet"),
            scope.get("table_caption"),
            scope.get("caption_or_title"),
        )
        if alias
    }
    scope_locators = iter_table_scope_locator_dicts(scope.get("table_scope_locators"))

    def choose(matches: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
        if len(matches) == 1:
            return matches[0], ""
        if not matches:
            return None, "authorized_target_unresolved"
        matches.sort(
            key=lambda item: (
                0 if bool(item.get("preserved_by_authority_ranking")) else 1,
                int(item.get("authority_rank") or 10_000),
                -float(item.get("authority_score") or 0.0),
            )
        )
        top = matches[0]
        top_rank = int(top.get("authority_rank") or 10_000)
        same_top_rank = [item for item in matches if int(item.get("authority_rank") or 10_000) == top_rank]
        if len(same_top_rank) == 1:
            return top, ""
        top_score = float(top.get("authority_score") or 0.0)
        same_top_score = [item for item in same_top_rank if float(item.get("authority_score") or 0.0) == top_score]
        if len(same_top_score) == 1:
            return top, ""
        return None, "multiple_candidate_payloads"

    def payload_matches_locator(item: dict[str, Any], locator: dict[str, Any]) -> bool:
        payload_table_id = _normalize_table_label(item.get("table_id") or item.get("source_table_id"))
        payload_source_ref = normalize_text(
            item.get("source_table_reference") or item.get("source_csv_path")
        ).replace("\\", "/").lower()
        payload_asset_id = normalize_text(item.get("source_table_asset_id") or item.get("table_asset_id")).lower()
        locator_path = normalize_text(locator.get("source_table_reference") or locator.get("source_csv_path") or locator.get("table_path")).replace("\\", "/").lower()
        locator_asset_id = normalize_text(locator.get("source_table_asset_id") or locator.get("table_asset_id")).lower()
        locator_table_id = _normalize_table_label(locator.get("table_id") or locator.get("source_table_id"))
        # When the LLM/source repair provides an execution locator, prefer exact
        # source asset/path identity over caption table number.  Some parser
        # readthroughs duplicate captions across adjacent assets (for example a
        # real table plus page-continuation text both labelled "Table 3"); an OR
        # on table_id alone can bind the scope to the wrong payload and silently
        # drop source-backed rows.
        if locator_path or locator_asset_id:
            return bool(
                (locator_path and payload_source_ref == locator_path)
                or (locator_asset_id and payload_asset_id == locator_asset_id)
            )
        return bool(locator_table_id and payload_table_id == locator_table_id)

    locator_matches: list[dict[str, Any]] = []
    if scope_locators:
        for item in normalized_payloads:
            if any(payload_matches_locator(item, locator) for locator in scope_locators):
                locator_matches.append(item)
        if locator_matches:
            return choose(locator_matches)

    wanted_table_path = normalize_text(
        scope.get("source_table_reference") or scope.get("table_path")
    ).replace("\\", "/").lower()
    wanted_asset_id = normalize_text(
        scope.get("source_table_asset_id") or scope.get("table_asset_id")
    ).lower()
    matches: list[dict[str, Any]] = []
    for item in normalized_payloads:
        payload_table_id = _normalize_table_label(item.get("table_id") or item.get("source_table_id"))
        payload_source_ref = normalize_text(
            item.get("source_table_reference") or item.get("source_csv_path")
        ).replace("\\", "/").lower()
        payload_asset_id = normalize_text(item.get("source_table_asset_id") or item.get("table_asset_id")).lower()
        payload_aliases = set(
            table_number_aliases(
                item.get("table_id"),
                item.get("source_table_id"),
                item.get("source_table_asset_id"),
                item.get("source_table_reference"),
                item.get("source_csv_path"),
                item.get("source_caption_or_title"),
            )
        )
        if wanted_table_path and payload_source_ref == wanted_table_path:
            matches.append(item)
            continue
        if wanted_asset_id and payload_asset_id == wanted_asset_id:
            matches.append(item)
            continue
        if wanted_table_id and payload_table_id == wanted_table_id:
            matches.append(item)
            continue
        if wanted_aliases and payload_aliases and wanted_aliases & payload_aliases:
            matches.append(item)
            continue
    if not matches:
        if not any([wanted_table_id, wanted_table_path, wanted_asset_id]) and not scope_locators:
            return None, "payload_locator_missing"
        return None, "authorized_target_unresolved"
    return choose(matches)


def source_csv_authority_payload_from_scope(scope: dict[str, Any]) -> dict[str, Any] | None:
    """Build a bounded execution payload from an explicit LLM/source table locator.

    This is a post-authorization fallback for cases where S2-2 preserved a
    source CSV/table locator in the semantic scope, but no normalized payload was
    reattached under that exact locator.  It does not discover new semantic
    tables: callers only invoke it after an LLM-authorized formulation scope has
    failed ordinary payload resolution.
    """
    source_ref = normalize_text(
        scope.get("source_table_reference")
        or scope.get("source_csv_path")
        or scope.get("table_path")
    )
    if not source_ref or source_ref == "source_text":
        return None
    source_path = Path(source_ref)
    if not source_path.is_absolute():
        source_path = (REPO_ROOT / source_path).resolve()
    if not source_path.exists() or source_path.suffix.lower() != ".csv":
        return None
    rows = read_csv_rows(source_path)
    table_id = extract_table_label(source_path, rows) or normalize_text(scope.get("table_id"))
    return {
        "table_id": table_id,
        "source_table_id": table_id,
        "source_csv_path": str(source_path),
        "normalized_csv_path": str(source_path),
        "source_table_reference": source_ref,
        "source_table_asset_id": normalize_text(scope.get("source_table_asset_id") or scope.get("table_asset_id")) or source_path.stem,
        "representation_status": "source_csv_locator_fallback",
        "normalization_actions": ["preserve_coordinate_grid"],
    }


def source_text_authority_payload_from_scope(
    *,
    document: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any] | None:
    """Create an empty authority shell for LLM-authorized source-text tables.

    Used only when normalized payload resolution fails but the clean source text
    still contains the LLM-authorized table anchor.  Row extraction remains
    constrained to existing source-text table extractors.
    """
    table_id = normalize_text(scope.get("table_id"))
    table_type = normalize_text(scope.get("table_type")).lower()
    if not table_id or table_type not in {"full_formulation", "doe_table"}:
        return None
    source_text = load_document_source_text(document)
    if not source_text or not re.search(rf"\b{re.escape(table_id)}\b", source_text, flags=re.IGNORECASE):
        return None
    return {
        "table_id": table_id,
        "source_table_id": table_id,
        "normalized_csv_path": "",
        "source_csv_path": "",
        "source_table_reference": "source_text",
        "row_entries": [],
        "representation_status": "source_text_locator_fallback",
    }


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


def infer_table_scopes_from_source_sweep_tables(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer execution-ready table scopes from LLM-declared variable-sweep papers.

    This is a post-LLM repair path: it does not decide semantic relevance before
    the LLM.  It only activates after the LLM has declared a formulation variable
    sweep / formulation candidates, then reopens source table files that contain
    row-level concentration-sweep evidence.  The goal is to recover rows when the
    LLM collapses a source-backed sweep into family-level candidates.
    """
    if not is_llm_first_document(document):
        return []
    semantic_signals = document.get("semantic_signals") if isinstance(document.get("semantic_signals"), dict) else {}
    if not bool(semantic_signals.get("has_variable_sweep")):
        return []
    formulation_candidates = [
        item
        for item in ensure_list(document.get("formulation_candidates"))
        if isinstance(item, dict)
    ]
    if not formulation_candidates:
        formulation_candidates = [
            item
            for item in ensure_list(document.get("formulation_identity_candidates"))
            if isinstance(item, dict)
        ]
    if not formulation_candidates:
        return []
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    inferred: list[dict[str, Any]] = []
    seen_table_ids: set[str] = set()
    for path in source_table_paths(document):
        rows = read_csv_rows(path)
        table_id = extract_table_label(path, rows)
        table_key = normalize_token(table_id)
        if not table_id or table_key in seen_table_ids:
            continue
        raw_text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        payload = {
            "representation_status": "repaired_summary",
            "normalization_actions": ["preserve_coordinate_grid"],
            "source_csv_path": str(path),
        }
        split_rows, _ = extract_split_column_concentration_sweep_rows_from_source_csv(
            authority_payload=payload,
            document=document,
            scope={"table_id": table_id, "is_formulation_table": True},
        )
        sample_hits = re.findall(
            r"(?:Empty|[A-Za-z0-9\-]+-loaded)\s+nanocapsules?\s*\([^)]*\)",
            raw_text,
            flags=re.IGNORECASE,
        )
        has_caption_sample_surface = len(unique_nonempty(sample_hits)) >= 2 and re.search(
            r"\b(?:diameter|zeta|incorporation|efficien|PI|polydispersity)\b",
            raw_text,
            flags=re.IGNORECASE,
        )
        if len(split_rows) < 4 and not has_caption_sample_surface:
            continue
        inferred.append(
            {
                "scope_id": f"{document_key}__source_sweep_table_scope__{len(inferred) + 1:02d}",
                "table_id": table_id,
                "table_path": str(path),
                "table_asset_id": path.stem,
                "source_table_asset_id": path.stem,
                "source_table_reference": str(path),
                "table_scope_locators": {
                    "table_id": table_id,
                    "source_table_asset_id": path.stem,
                    "source_table_reference": str(path),
                },
                "variable_name": "",
                "candidate_values": [],
                "is_formulation_table": True,
                "table_type": "full_formulation",
                "confidence": "medium",
                "evidence_span": table_id,
                "marker_provenance": "llm_parsed",
                "declaration_basis": "llm_variable_sweep_plus_source_table_row_evidence",
            }
        )
        seen_table_ids.add(table_key)

    # Some PDF text extractors preserve compact characterization/caption tables in
    # the clean text even when the coordinate CSV asset was split across pages or
    # assigned an asset-local label.  After the LLM has already authorized this as
    # a variable-sweep formulation paper, reopen only source-text table captions
    # that expose explicit formulation sample labels plus measured-result columns.
    source_text = load_document_source_text(document)
    if source_text:
        for match in re.finditer(r"\bTable\s+(\d{1,3})\b", source_text, flags=re.IGNORECASE):
            table_id = f"Table {int(match.group(1))}"
            table_key = normalize_token(table_id)
            if table_key in seen_table_ids:
                continue
            window = normalize_text(source_text[match.start(): match.start() + 1800])
            has_measured_result_surface = bool(
                re.search(r"\b(?:diameter|zeta|incorporation|efficien|PI|polydispersity|final\s+concentration)\b", window, flags=re.IGNORECASE)
            )
            sample_hits = re.findall(
                r"(?:Empty|[A-Za-z0-9\-]+-loaded)\s+nano(?:sphere|capsule)s?\s*(?:\([^)]*\))?",
                window,
                flags=re.IGNORECASE,
            )
            has_empty_loaded_characterization_surface = bool(
                re.search(r"\bempty\s+and\s+loaded\b", window, flags=re.IGNORECASE)
                and re.search(r"\bEmpty\s+nano(?:sphere|capsule)s?\s+XAN\s+nano(?:sphere|capsule)s?", window, flags=re.IGNORECASE)
                and re.search(r"\b3-?MeOXAN\s+nano(?:sphere|capsule)s?", window, flags=re.IGNORECASE)
            )
            if not has_measured_result_surface or (len(unique_nonempty(sample_hits)) < 2 and not has_empty_loaded_characterization_surface):
                continue
            inferred.append(
                {
                    "scope_id": f"{document_key}__source_text_table_scope__{len(inferred) + 1:02d}",
                    "table_id": table_id,
                    "table_path": "",
                    "table_asset_id": "",
                    "source_table_asset_id": "",
                    "source_table_reference": "source_text",
                    "source_text_caption_only": True,
                    "table_scope_locators": {"table_id": table_id, "source_table_reference": "source_text"},
                    "variable_name": "",
                    "candidate_values": [],
                    "is_formulation_table": True,
                    "table_type": "full_formulation",
                    "confidence": "medium",
                    "evidence_span": table_id,
                    "marker_provenance": "llm_parsed",
                    "declaration_basis": "llm_variable_sweep_plus_source_text_table_evidence",
                }
            )
            seen_table_ids.add(table_key)
    return inferred


def infer_table_scopes_from_table_anchored_formulations(document: dict[str, Any]) -> list[dict[str, Any]]:
    if not is_llm_first_document(document):
        return []
    existing_scopes = [
        item
        for item in ensure_list(document.get("table_formulation_scopes"))
        if isinstance(item, dict)
    ]
    if any(bool(item.get("is_formulation_table")) for item in existing_scopes):
        return []

    source_sweep_scopes = infer_table_scopes_from_source_sweep_tables(document)
    if source_sweep_scopes:
        return source_sweep_scopes

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
    has_formulation_bearing_scope = any(
        bool(scope.get("is_formulation_table"))
        for scope in table_scopes
        if isinstance(scope, dict)
    )
    if not has_formulation_bearing_scope:
        inferred_scopes = [
            normalize_table_scope(item, document=document)
            for item in infer_table_scopes_from_table_anchored_formulations(document)
        ]
        if inferred_scopes:
            existing_scope_keys = {
                (normalize_text(scope.get("table_id")), normalize_text(scope.get("source_table_reference")))
                for scope in table_scopes
                if isinstance(scope, dict)
            }
            table_scopes.extend(
                scope
                for scope in inferred_scopes
                if (normalize_text(scope.get("table_id")), normalize_text(scope.get("source_table_reference")))
                not in existing_scope_keys
            )
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


def row_identity_surface_kind(explicit_rows: list[dict[str, Any]]) -> str:
    styles = {
        normalize_text((row.get("row_label_info") or {}).get("label_style"))
        for row in explicit_rows
        if isinstance(row, dict)
    }
    styles.discard("")
    if styles == {"numeric"}:
        return "numeric_first_column"
    if styles == {"f_numeric"}:
        return "f_numeric_first_column"
    if styles:
        return "mixed_explicit_first_column"
    return ""


MEASUREMENT_HEADER_PATTERNS = [
    r"\bmean size\b",
    r"\bsizes?\b",
    r"\bdiameter\b",
    r"\bmajor axis\b",
    r"\bminor axis\b",
    r"\bferet\b",
    r"\baspect ratio\b",
    r"\bz-average\b",
    r"\bp\.?\s*i\.?\b",
    r"\bpdi\b",
    r"\bpi[a-z]?\b",
    r"\bpolydispersity\b",
    r"\byield\b",
    r"\bd\.?\s*c\.?\b",
    r"\bdrug content\b",
    r"\be\.?\s*e\.?\b",
    r"\bentrapp?ment\b",
    r"\bencapsulation\b",
    r"\bloading\b",
    r"\brecovery\b",
    r"\bafter freeze-drying\b",
    r"\bbefore freeze-drying\b",
    r"\bmeasured responses\b",
    r"\bresponse\b",
    r"\bzeta\b",
    r"\bzp\b",
]

MEASUREMENT_HEADER_EXACT_TOKENS = {
    "ar",
}


def is_measurement_header(header: str) -> bool:
    low = normalize_text(header).lower()
    if any(re.search(pattern, low) for pattern in MEASUREMENT_HEADER_PATTERNS):
        return True
    compact_tokens = [token for token in re.findall(r"[a-z0-9]+", low) if token]
    return bool(compact_tokens) and all(token in MEASUREMENT_HEADER_EXACT_TOKENS for token in compact_tokens)


def normalize_assignment_name(name: str) -> str:
    return normalize_text(name)


def compatibility_field_for_assignment(name: str) -> str:
    canonical_header = canonical_field_for_header(name)
    if canonical_header:
        return {
            "ee_percent": "encapsulation_efficiency_percent",
            "lc_percent": "loading_content_percent",
            "dl_percent": "dl_percent",
            "particle_size_nm": "size_nm",
            "pdi": "pdi",
            "zeta_mV": "zeta_mV",
        }.get(canonical_header, canonical_header)
    low = normalize_assignment_name(name).lower()
    compact_tokens = re.findall(r"[a-z0-9]+", low)
    compact = " ".join(compact_tokens)
    if "plga" in low or "polymer" in low:
        return "plga_mass_mg"
    if "pva" in low or "surfactant" in low or "stabilizer" in low:
        return "surfactant_concentration_text"
    if "drug" in low or "pf" in low:
        return "drug_feed_amount_text"
    if "size" in low or "z-average" in low:
        return "size_nm"
    if "zeta" in low:
        return "zeta_mV"
    if (
        "entrap" in low
        or "encapsulation" in low
        or low in {"ee", "e e"}
        or compact in {"e e", "ee"}
        or compact.startswith("e e ")
        or compact.startswith("ee ")
    ):
        return "encapsulation_efficiency_percent"
    return ""


def maybe_number_text(value: str) -> str:
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", normalize_text(value))
    return match.group(0).replace(",", ".") if match else ""


def is_temporal_followup_label(label: str) -> bool:
    low = normalize_text(label).lower()
    if not low:
        return False
    return bool(
        re.search(r"\b(day|days|week|weeks|month|months|hour|hours|hr|hrs|min|mins|minute|minutes|time)\b", low)
        and re.search(r"\d", low)
    )


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


def formulation_identity_label_looks_primary(value: Any) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    lowered = text.lower()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?%", lowered):
        return False
    primary_cues = (
        "plga",
        "pcl",
        "peg",
        "nanocaps",
        "nanospher",
        "nanoparticle",
        "loaded",
        "theoretical concentration",
        "mg/ml",
        "mg mL",
        "drug",
        "polymer",
        "sc6oh",
        "xan",
        "meoxan",
        "mb ",
    )
    if any(cue in lowered for cue in primary_cues):
        return True
    token_count = len(re.findall(r"[a-z0-9]+", lowered))
    if token_count >= 3 and any(symbol in text for symbol in ("/", "(", ")", ":", "-")):
        return True
    return False


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


REVERSIBLE_HEAD_NOUNS = {
    "amount",
    "concentration",
    "ratio",
    "content",
    "loading",
    "volume",
}


def phrase_patterns(phrase: str) -> list[str]:
    raw_tokens = [token for token in re.findall(r"[A-Za-z0-9%]+", normalize_text(phrase)) if token]
    if not raw_tokens:
        return []
    escaped = [re.escape(token) for token in raw_tokens]
    patterns = [r"[\s\-/–—]*".join(escaped)]
    if len(raw_tokens) >= 2 and raw_tokens[-1].lower() in REVERSIBLE_HEAD_NOUNS:
        head = re.escape(raw_tokens[-1])
        tail = [re.escape(token) for token in raw_tokens[:-1]]
        patterns.append(head + r"[\s\-/–—]*(?:of[\s\-/–—]*)?" + r"[\s\-/–—]*".join(tail))
    deduped: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        deduped.append(pattern)
    return deduped


def phrase_pattern(phrase: str) -> str:
    patterns = phrase_patterns(phrase)
    return patterns[0] if patterns else ""


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


def _family_candidates_from_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [item for item in ensure_list(document.get("formulation_candidates")) if isinstance(item, dict)]
    if candidates:
        return [
            item
            for item in candidates
            if normalize_text(item.get("candidate_kind") or item.get("instance_kind")) in {"formulation_family", "family"}
        ]
    legacy = [item for item in ensure_list(document.get("formulation_identity_candidates")) if isinstance(item, dict)]
    if legacy:
        return [
            item
            for item in legacy
            if normalize_text(item.get("instance_kind") or item.get("candidate_kind")) == "formulation_family"
            or normalize_text(item.get("raw_formulation_label")).lower().endswith("formulations")
            or "formulations with varying" in normalize_text(item.get("raw_formulation_label")).lower()
        ]
    return []


def _first_family_label_from_document(document: dict[str, Any]) -> str:
    for candidate in _family_candidates_from_document(document):
        label = normalize_text(candidate.get("label_hint") or candidate.get("raw_formulation_label"))
        if not label:
            continue
        match = re.search(r"([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-NPs?)", label)
        if match:
            return normalize_text(match.group(1))
    return ""


def _extract_drug_name_for_family(source_text: str, family_label: str) -> str:
    low_family = normalize_text(family_label).lower()
    if low_family.startswith("ap-"):
        match = re.search(r"acetylpuerarin\s*\(\s*AP\s*\)", source_text, flags=re.IGNORECASE)
        if match:
            return "acetylpuerarin"
        return "AP"
    prefix = normalize_text(family_label).split("-")[0]
    return prefix if prefix and prefix.lower() not in {"plga", "nps", "np"} else ""


def extract_characterization_pair_rows_from_source_text(
    *,
    document: dict[str, Any],
    scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    source_text = load_document_source_text(document)
    if not source_text:
        return [], "source_text_missing"
    if len(_family_candidates_from_document(document)) != 1:
        return [], "not_single_family_document"
    comparator_match = re.search(
        r"zeta\s+potential\s+of\s+(?P<loaded_label>[A-Za-z0-9\-]+NPs?)\s+was\s+found\s+to\s+be\s+(?P<loaded_zeta>[−\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?\s*mV),\s*whereas\s+that\s+of\s+(?P<blank_state>empty|blank|drug\s*free|drug-free)\s+NPs\s+was\s+(?P<blank_zeta>[−\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?\s*mV)",
        source_text,
        flags=re.IGNORECASE,
    )
    if not comparator_match:
        return [], "no_explicit_loaded_blank_characterization_pair"
    loaded_label = normalize_text(comparator_match.group("loaded_label")) or _first_family_label_from_document(document)
    if not loaded_label:
        return [], "family_label_missing"
    loaded_sentence = normalize_text(comparator_match.group(0))
    polymer_identity = "PLGA" if "plga" in loaded_label.lower() else ""
    if not polymer_identity:
        family_labels = detect_polymer_family_labels(source_text)
        polymer_identity = normalize_text(family_labels[0]) if family_labels else ""
    if not polymer_identity:
        return [], "polymer_identity_missing"
    characterization_match = re.search(
        r"particle\s+size\s+of\s+(?P<size>\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?\s*nm),\s+a\s+polydispersity\s+index\s+of\s+(?P<pdi>\d+(?:\.\d+)?),\s+and\s+a\s+zeta\s+potential\s+of\s+(?P<loaded_zeta>[−\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?\s*mV)\.\s+The\s+optimized\s+EE\s+and\s+DL\s+values\s+were\s+(?P<ee>\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?%)\s+and\s+(?P<dl>\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?%)",
        source_text,
        flags=re.IGNORECASE,
    )
    loaded_assignments = [
        {"name": "polymer_identity", "value": polymer_identity},
    ]
    drug_name = _extract_drug_name_for_family(source_text, loaded_label)
    if drug_name:
        loaded_assignments.append({"name": "drug identity", "value": drug_name})
    if characterization_match:
        loaded_assignments.extend(
            [
                {"name": "particle size", "value": normalize_text(characterization_match.group("size"))},
                {"name": "zeta potential", "value": normalize_text(characterization_match.group("loaded_zeta"))},
                {"name": "encapsulation efficiency", "value": normalize_text(characterization_match.group("ee"))},
            ]
        )
        loaded_row_text = normalize_text(characterization_match.group(0))
    else:
        loaded_assignments.append({"name": "zeta potential", "value": normalize_text(comparator_match.group("loaded_zeta"))})
        loaded_row_text = loaded_sentence
    blank_assignments = [
        {"name": "polymer_identity", "value": polymer_identity},
        {"name": "zeta potential", "value": normalize_text(comparator_match.group("blank_zeta"))},
    ]
    return (
        [
            {
                "label": f"{loaded_label} / Drug loaded",
                "assignments": loaded_assignments,
                "row_text": loaded_row_text,
                "instance_role": "reported",
                "source_region_type": "narrative_text",
                "evidence_span_text": loaded_row_text,
                "change_context_tag": "characterization_pair_recovery",
            },
            {
                "label": f"{loaded_label} / Empty",
                "assignments": blank_assignments,
                "row_text": loaded_sentence,
                "instance_role": "control",
                "source_region_type": "narrative_text",
                "evidence_span_text": loaded_sentence,
                "change_context_tag": "characterization_pair_recovery",
            },
        ],
        "",
    )


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
    padded = [normalize_text(cells[idx]) if idx < len(cells) else "" for idx in range(width)]
    # If the recovered/normalized matrix preserved blank placeholder cells, keep
    # the two-dimensional column positions.  Collapsing to non-empty cells shifts
    # grouped headers left and can turn auxiliary columns (for example a later
    # after-storage timepoint) into apparent formulation headers.
    if len(cells) >= width and any(not cell for cell in padded) and any(cell for cell in padded):
        return padded
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


AUXILIARY_MEASUREMENT_COLUMN_PATTERNS = [
    r"\bafter\s+storage\b",
    r"\bstor(?:age|ed)\b",
    r"\bstability\b",
]


def is_auxiliary_measurement_column_header(header_parts: list[str]) -> bool:
    """Return True for column groups that are follow-up measurements, not new formulations.

    Column-oriented formulation tables sometimes append a stability/storage
    timepoint column beside true preparation columns.  The lower header level may
    repeat a real formulation coordinate (for example a drug:polymer ratio), but
    the top-level group is an after-storage characterization condition.  Such a
    column should feed value evidence for the base formulation only; it must not
    instantiate a separate formulation row.
    """
    header_text = " ".join(normalize_text(part) for part in header_parts if normalize_text(part)).lower()
    if not header_text:
        return False
    return any(re.search(pattern, header_text) for pattern in AUXILIARY_MEASUREMENT_COLUMN_PATTERNS)


def extract_column_anchor_rows_from_authority(
    *,
    authority_payload: dict[str, Any],
    row_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    candidate_entry_sets: list[list[dict[str, Any]]] = []
    if row_entries:
        candidate_entry_sets.append(row_entries)
    matrix = [row for row in ensure_list(authority_payload.get("normalized_matrix")) if isinstance(row, list)]
    if not matrix:
        normalized_csv_path = normalize_text(authority_payload.get("normalized_csv_path"))
        if normalized_csv_path:
            csv_path = Path(normalized_csv_path)
            if not csv_path.is_absolute():
                csv_path = (REPO_ROOT / csv_path).resolve()
            if csv_path.exists():
                matrix = read_csv_rows(csv_path)
    if matrix:
        raw_entries = [
            {
                "row_index": index,
                "row_number": "",
                "cells": [normalize_text(cell) for cell in row],
                "row_text": " | ".join(normalize_text(cell) for cell in row if normalize_text(cell)),
            }
            for index, row in enumerate(matrix, start=1)
        ]
        if not candidate_entry_sets or raw_entries != candidate_entry_sets[0]:
            candidate_entry_sets.append(raw_entries)
    best_reason = "no_measurement_axis_detected"
    for candidate_rows in candidate_entry_sets:
        start_index = measurement_rows_start_index(candidate_rows)
        if start_index is None:
            best_reason = "no_measurement_axis_detected"
            continue
        if start_index < 2:
            best_reason = "insufficient_column_header_rows"
            continue
        width = max(len(ensure_list(entry.get("cells"))) for entry in candidate_rows)
        header_rows = [
            horizontal_forward_fill(
                expand_header_row_for_formulation_columns(
                    [normalize_text(cell) for cell in ensure_list(entry.get("cells"))],
                    width=width,
                ),
                width=width,
            )
            for entry in candidate_rows[:start_index]
        ]
        measurement_rows = [
            [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
            for entry in candidate_rows[start_index:]
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
            if is_auxiliary_measurement_column_header(header_parts):
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
            best_reason = "insufficient_formulation_columns"
            continue
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
    return [], best_reason


def source_text_context_window(text: str, pattern_match: re.Match[str], *, chars_before: int = 1400, chars_after: int = 250) -> str:
    start = max(0, pattern_match.start() - chars_before)
    end = min(len(text), pattern_match.end() + chars_after)
    return text[start:end]


def extract_single_variable_level_list(text: str, variable_name: str) -> list[str]:
    patterns = phrase_patterns(variable_name)
    if patterns:
        for pattern in patterns:
            match = re.search(pattern + r"\s*\(([^)]{3,120})\)", text, flags=re.IGNORECASE)
            if match:
                return parse_candidate_values(match.group(1))
    variable_low = normalize_text(variable_name).lower()
    if "poloxamer 188" in variable_low or ("surfactant" in variable_low and "concentration" in variable_low):
        match = re.search(
            r"containing\s+([0-9][0-9\.,\sand]{1,80})\s*mg/mL\s+of[^.]{0,80}?(?:poloxamer\s+188|surfactant)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            numeric_tokens = re.findall(r"\d+(?:\.\d+)?", match.group(1))
            if len(numeric_tokens) >= 4:
                numeric_tokens = numeric_tokens[-4:]
            normalized = [f"{value} mg/mL" for value in numeric_tokens if value]
            return unique_nonempty(normalized)
    if "ratio" in variable_low and "plga" in variable_low and "itz" in variable_low:
        match = re.search(
            r"ratios?\s+of\s+([0-9:\.,\sand]{3,120})\s*,?\s*respectively",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return parse_candidate_values(match.group(1))
    return []


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


def normalize_family_label(label: str) -> str:
    text = normalize_text(label).upper()
    text = re.sub(r"\bPLGA\b", "PLGA ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"PLGA\s+(\d{1,3})\s*/\s*(\d{1,3})", r"PLGA \1/\2", text)
    return text


def detect_polymer_family_labels(text: str) -> list[str]:
    labels: list[str] = []
    for match in re.finditer(r"\bPLGA\s*\d{1,3}\s*/\s*\d{1,3}\b|\bPCL\b", text, flags=re.IGNORECASE):
        label = normalize_family_label(match.group(0))
        if label == "PCL":
            labels.append("PCL")
        else:
            labels.append(label)
    return unique_nonempty(labels)


def has_explicit_blank_control_contract(text: str) -> bool:
    low = normalize_text(text).lower()
    return bool(
        re.search(r"drug\s+free\s+nanoparticles\s+were\s+pre[\s-]*pared", low)
        and re.search(r"without\s+the\s+addition\s+of\s+[a-z0-9\-]+", low)
        and re.search(r"characterized", low)
    )


def variable_supports_blank_control_family_recovery(variable_name: str) -> bool:
    low = normalize_text(variable_name).lower()
    return "drug" not in low and "etoposide" not in low


def _numeric_value(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", normalize_text(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _format_mg_level(value: str) -> str:
    value_text = normalize_text(value)
    if re.search(r"\bmg\b", value_text, flags=re.IGNORECASE):
        return value_text
    return f"{value_text} mg" if value_text else ""


def _optimized_baseline_for_axis(source_text: str, variable_name: str) -> str:
    low_axis = normalize_text(variable_name).lower()
    text = normalize_text(source_text)
    match = re.search(
        r"amount\s+of\s+drug\s*,\s*amount\s+of\s+polymer\s*,\s*and\s*concentration\s+of\s+stabilizer\s+were\s+optimized\s+to\s+"
        r"(\d+(?:\.\d+)?\s*mg)\s*,\s*(\d+(?:\.\d+)?\s*mg)\s*,\s*and\s*(\d+(?:\.\d+)?\s*%\s*w/v)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    if "drug" in low_axis or "etoposide" in low_axis:
        return normalize_text(match.group(1))
    if "polymer" in low_axis and "amount" in low_axis:
        return normalize_text(match.group(2))
    if "stabilizer" in low_axis or "surfactant" in low_axis:
        return normalize_text(match.group(3))
    return ""


def _drug_amount_observed_endpoint_levels(source_text: str, variable_name: str) -> list[str]:
    low_axis = normalize_text(variable_name).lower()
    if "drug" not in low_axis and "etoposide" not in low_axis:
        return []
    text = normalize_text(source_text)
    match = re.search(
        r"(?:increase\s+in\s+the\s+)?amount\s+of\s+etoposide\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*mg\s+increased\s+.{0,320}?\bPLGA\s*50\s*/\s*50\b.{0,320}?\bPCL\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    return [_format_mg_level(match.group(1)), _format_mg_level(match.group(2))]


def source_supported_single_variable_levels(
    *,
    source_text: str,
    variable_name: str,
    levels: list[str],
    baseline_value: str,
) -> list[str]:
    """Prune broad single-variable sweep ranges to source-supported formulation rows.

    This stays evidence-gated: it activates only for observed endpoint language that
    binds levels to measured formulation outcomes, or for above-baseline polymer
    amount language where lower-bound rows are under-anchored.
    """
    low_axis = normalize_text(variable_name).lower()
    normalized_levels = [normalize_text(level) for level in levels if normalize_text(level)]
    if not normalized_levels:
        return []
    observed_drug_levels = _drug_amount_observed_endpoint_levels(source_text, variable_name)
    if observed_drug_levels:
        observed_keys = {re.sub(r"\s+", "", item.lower()) for item in observed_drug_levels}
        return [level for level in normalized_levels if re.sub(r"\s+", "", level.lower()) in observed_keys]
    if "polymer" in low_axis and "amount" in low_axis:
        text_low = normalize_text(source_text).lower()
        if "polymer amount was increased from" in text_low and "above 50 mg" in text_low:
            baseline = _optimized_baseline_for_axis(source_text, variable_name) or baseline_value
            baseline_number = _numeric_value(baseline)
            if baseline_number is not None:
                kept = [
                    level
                    for level in normalized_levels
                    if (_numeric_value(level) is not None and _numeric_value(level) > baseline_number)
                ]
                if kept:
                    return kept
    return normalized_levels


def family_contexts_for_variable_group(*, source_text: str, variable_name: str) -> tuple[list[str], list[str], bool]:
    low = normalize_text(source_text).lower()
    family_labels = detect_polymer_family_labels(source_text)
    if not family_labels:
        return [], [], True
    variable_low = normalize_text(variable_name).lower()
    if (
        "stabilizer" in variable_low
        and has_explicit_blank_control_contract(source_text)
        and "all the polymers in the study" in low
    ):
        return family_labels, ["blank_control"], False
    if _drug_amount_observed_endpoint_levels(source_text, variable_name):
        observed_families = [family for family in ["PLGA 50/50", "PCL"] if family in family_labels]
        if observed_families:
            return observed_families, ["drug_loaded"], False
    return [], [], True


def parse_anchor_family_and_state(label: str) -> tuple[str, str]:
    match = re.match(r"(.+?)\s*/\s*(Empty|Drug loaded)$", normalize_text(label), flags=re.IGNORECASE)
    if not match:
        return "", ""
    family_label = normalize_family_label(match.group(1))
    payload_state = normalize_text(match.group(2)).lower().replace(" ", "_")
    if payload_state == "empty":
        payload_state = "blank_control"
    return family_label, payload_state


def optimized_family_labels_from_text(source_text: str) -> list[str]:
    family_labels: list[str] = []
    for pattern in [
        r"optimized formulations prepared with(.{0,900})",
        r"transmission electron microscopy image of formulation prepared with(.{0,500})",
    ]:
        match = re.search(pattern, source_text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        family_labels.extend(detect_polymer_family_labels(match.group(0)))
    return unique_nonempty(family_labels)


def _is_downstream_only_variable_axis(variable_name: str) -> bool:
    low = normalize_text(variable_name).lower()
    return any(token in low for token in ["lyoprotectant", "freeze", "resuspend", "sf/si", "reconstit", "post-processing"])


def infer_selected_optimized_variable_axes(
    *,
    selected_condition_hints: list[str],
    source_text: str,
    existing_variable_names: list[str],
) -> list[str]:
    """Recover source-supported sequential-optimization axes from neutral S2 hints.

    This is intentionally source-gated: a hint such as "surfactant concentration
    optimized" is promoted only when the source text yields at least two concrete
    levels for the inferred axis.
    """
    variables = list(existing_variable_names)
    seen = {normalize_token(item) for item in variables if normalize_text(item)}
    for hint in selected_condition_hints:
        hint_text = normalize_text(hint)
        if not hint_text:
            continue
        candidates: list[str] = []
        optimized_match = re.search(r"(.+?)\s+(?:was\s+)?optimized\b", hint_text, flags=re.IGNORECASE)
        if optimized_match:
            candidates.append(normalize_text(optimized_match.group(1)))
        ratio_match = re.search(r"([A-Za-z0-9: /\-]+?\s+ratios?)\b", hint_text, flags=re.IGNORECASE)
        if ratio_match:
            candidates.append(normalize_text(ratio_match.group(1)).rstrip("s"))
        for candidate in candidates:
            if not candidate:
                continue
            key = normalize_token(candidate)
            if key in seen:
                continue
            if len(extract_single_variable_level_list(source_text, candidate)) < 2:
                continue
            seen.add(key)
            variables.append(candidate)
    return variables


def source_indicates_anchorless_sequential_optimization(
    *,
    source_text: str,
    selected_condition_hints: list[str],
    primary_variable_names: list[str],
) -> bool:
    if len(primary_variable_names) < 2:
        return False
    low = normalize_text(source_text).lower()
    if "after the optimal" in low or "after optimal" in low:
        return True
    if re.search(r"optimized\s+by\s+modifying", low) and any("optimized" in hint.lower() for hint in selected_condition_hints):
        return True
    return False


def build_single_variable_recovery_contract(
    *,
    document: dict[str, Any],
    require_anchor_rows: bool,
) -> dict[str, Any]:
    semantic_signals = document.get("semantic_signals") if isinstance(document.get("semantic_signals"), dict) else {}
    selected_condition_hints = [
        normalize_text(item)
        for item in ensure_list(semantic_signals.get("selected_condition_hints"))
        if normalize_text(item)
    ]
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
    allow_anchorless_sequential = bool(
        selected_condition_hints
        and (semantic_signals.get("has_sequential_optimization") or semantic_signals.get("has_variable_sweep"))
    )
    if not require_anchor_rows and not allow_anchorless_sequential:
        return {
            "detected": False,
            "failure_reason": "missing_explicit_anchor_rows",
        }
    if not primary_variable_names:
        return {
            "detected": False,
            "failure_reason": "missing_primary_variable_names",
        }
    if not require_anchor_rows and allow_anchorless_sequential:
        primary_variable_names = [
            name for name in primary_variable_names if not _is_downstream_only_variable_axis(name)
        ]
        if not primary_variable_names:
            return {
                "detected": False,
                "failure_reason": "anchorless_sequential_has_only_downstream_variable_axes",
            }
    source_text = load_document_source_text(document)
    if not source_text:
        return {
            "detected": False,
            "failure_reason": "source_text_missing",
        }
    primary_variable_names = infer_selected_optimized_variable_axes(
        selected_condition_hints=selected_condition_hints,
        source_text=source_text,
        existing_variable_names=primary_variable_names,
    )
    allow_anchorless_sequential = bool(
        selected_condition_hints
        and (
            semantic_signals.get("has_sequential_optimization")
            or source_indicates_anchorless_sequential_optimization(
                source_text=source_text,
                selected_condition_hints=selected_condition_hints,
                primary_variable_names=primary_variable_names,
            )
        )
    )
    contract_match = None
    for pattern in SINGLE_VARIABLE_CONTRACT_PATTERNS:
        contract_match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if contract_match:
            break
    if contract_match is None and not allow_anchorless_sequential:
        return {
            "detected": False,
            "failure_reason": "single_variable_contract_not_found",
        }
    context = source_text_context_window(source_text, contract_match) if contract_match is not None else source_text
    groups: list[dict[str, Any]] = []
    baseline_assignments: dict[str, str] = {}
    for variable_name in primary_variable_names:
        levels = extract_single_variable_level_list(context, variable_name)
        if not levels and selected_condition_hints:
            levels = extract_single_variable_level_list(source_text, variable_name)
        if not levels and ("drug" in normalize_text(variable_name).lower() or "etoposide" in normalize_text(variable_name).lower()):
            for drug_axis_alias in ("etoposide amount", "amount of etoposide"):
                levels = extract_single_variable_level_list(source_text, drug_axis_alias)
                if levels:
                    break
        baseline_value = extract_baseline_assignment_from_text(context, variable_name)
        if not baseline_value and selected_condition_hints:
            baseline_value = extract_baseline_assignment_from_text(source_text, variable_name)
        optimized_axis_baseline = _optimized_baseline_for_axis(source_text, variable_name)
        if optimized_axis_baseline and (
            not baseline_value
            or not re.search(r"\d+(?:\.\d+)?\s*mg\b", baseline_value, flags=re.IGNORECASE)
            and ("drug" in normalize_text(variable_name).lower() or "etoposide" in normalize_text(variable_name).lower())
        ):
            baseline_value = optimized_axis_baseline
        if baseline_value:
            baseline_assignments[variable_name] = baseline_value
        if len(levels) >= 2:
            levels = source_supported_single_variable_levels(
                source_text=source_text,
                variable_name=variable_name,
                levels=levels,
                baseline_value=baseline_value,
            )
        if len(levels) >= 2:
            family_contexts, payload_states, emit_generic_row = family_contexts_for_variable_group(
                source_text=source_text,
                variable_name=variable_name,
            )
            groups.append(
                {
                    "variable_name": variable_name,
                    "levels": levels,
                    "baseline_value": baseline_value,
                    "family_contexts": family_contexts,
                    "payload_states": payload_states,
                    "emit_generic_row": emit_generic_row,
                }
            )
    if not groups:
        return {
            "detected": False,
            "failure_reason": "single_variable_levels_not_found",
        }
    missing_baseline = [group["variable_name"] for group in groups if not group.get("baseline_value")]
    if missing_baseline and not (allow_anchorless_sequential and contract_match is None):
        return {
            "detected": False,
            "failure_reason": "held_constant_context_incomplete",
            "variable_axes": missing_baseline,
        }
    if allow_anchorless_sequential and contract_match is None:
        baseline_assignments = {}
        for group in groups:
            if isinstance(group, dict):
                group["baseline_value"] = ""
    return {
        "detected": True,
        "source_type": "anchorless_sequential_selected_chain" if contract_match is None and allow_anchorless_sequential else "explicit_narrative_single_variable_contract",
        "groups": groups,
        "baseline_assignments": baseline_assignments,
        "held_constant_context_source": "selected_condition_hints_or_stagewise_text" if contract_match is None and allow_anchorless_sequential else "source_text_baseline_clause",
        "evidence_span": normalize_text(context),
        "source_text": source_text,
        "optimized_family_labels": optimized_family_labels_from_text(source_text),
        "blank_control_supported": has_explicit_blank_control_contract(source_text),
    }


def emit_supplemental_family_anchor_rows(
    *,
    document: dict[str, Any],
    compatibility_columns: list[str],
    contract: dict[str, Any],
    scope: dict[str, Any],
    scope_id: str,
    table_id: str,
    existing_rows: list[dict[str, Any]],
    group_hint_prefix: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]], int]:
    source_text = normalize_text(contract.get("source_text"))
    if not source_text:
        return [], [], [], 0
    if not bool(contract.get("blank_control_supported")):
        return [], [], [], 0
    optimized_families = [normalize_text(item) for item in ensure_list(contract.get("optimized_family_labels")) if normalize_text(item)]
    if not optimized_families:
        return [], [], [], 0
    baseline_assignments = {
        normalize_text(name): normalize_text(value)
        for name, value in (contract.get("baseline_assignments") or {}).items()
        if normalize_text(name) and normalize_text(value)
    }
    if not baseline_assignments:
        return [], [], [], 0
    def anchor_pair_from_existing_row(row: dict[str, Any]) -> tuple[str, str]:
        label = normalize_text(row.get("label") or row.get("raw_formulation_label"))
        return parse_anchor_family_and_state(label)

    existing_pairs = {
        anchor_pair_from_existing_row(row)
        for row in existing_rows
        if anchor_pair_from_existing_row(row) != ("", "")
    }
    if not existing_pairs:
        return [], [], [], 0
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    doi = normalize_text(document.get("doi"))
    model_name = normalize_text(document.get("model_name") or document.get("source_mode")) or "stage2_v2_semantic_objects"
    evidence_span = normalize_text(source_text_context_window(source_text, re.search(r"optimized formulations prepared with", source_text, flags=re.IGNORECASE) or re.search(r"transmission electron microscopy image of formulation prepared with", source_text, flags=re.IGNORECASE) or re.search(r"table\s*1", source_text, flags=re.IGNORECASE) or re.search(r"fig\.\s*2", source_text, flags=re.IGNORECASE)))
    rows: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []
    traces: list[dict[str, str]] = []
    emitted = 0
    for family_label in optimized_families:
        for payload_state, payload_label in [("blank_control", "Empty"), ("drug_loaded", "Drug loaded")]:
            if (family_label, payload_state) in existing_pairs:
                continue
            assignment_map = {"polymer_identity": family_label}
            for name, value in baseline_assignments.items():
                low = name.lower()
                if payload_state == "blank_control" and ("drug" in low or "etoposide" in low):
                    continue
                assignment_map[name] = value
            row = {column: "" for column in compatibility_columns}
            row_label = f"{family_label} / {payload_label}"
            row_id = f"{document_key}__{normalize_token(table_id)}__{normalize_token(row_label)}"
            identity_variables = [
                {
                    "name": normalize_token(name),
                    "name_raw": name,
                    "value": assignment_map[name],
                    "value_raw": assignment_map[name],
                }
                for name in assignment_map
            ]
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
                    "evidence_section": table_id,
                    "evidence_span_text": evidence_span,
                    "formulation_role": "control" if payload_state == "blank_control" else "reported",
                    "instance_context_tags": stringify_json(["table_row_expansion", "narrative_anchor_completion"]),
                    "change_context_tags": stringify_json(["optimized_family_anchor_completion"]),
                    "change_descriptions": stringify_json([f"{name}={assignment_map[name]}" for name in assignment_map]),
                    "change_role": "table_anchor_completion",
                    IDENTITY_VARIABLES_FIELD: stringify_json(identity_variables),
                    METHOD_GROUP_SIGNATURE_HINT_FIELD: f"{group_hint_prefix}__optimized_family_anchor_completion",
                    TABLE_ID_FIELD: table_id,
                    TABLE_ROW_ID_FIELD: f"{table_id}::{normalize_token(row_label)}" if table_id else "",
                    TABLE_ASSIGNMENTS_FIELD: stringify_json([assignment_map]),
                    TABLE_SCOPE_FIELD: stringify_json(scope),
                    "supporting_evidence_refs": stringify_json(
                        [
                            {
                                "source_region_type": "narrative_text",
                                "source_locator_text": f"{table_id}::optimized_family_anchor_completion::{family_label}",
                                "supporting_snippet": evidence_span,
                                "target_field_name": "|".join(assignment_map.keys()),
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
                    "detail": f"optimized_anchor::{family_label}::{payload_state}",
                }
            )
            emitted += 1
    return rows, jsonl_rows, traces, emitted


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
        if not variable_name:
            continue
        group_hint = f"{group_hint_prefix}__{normalize_token(variable_name)}"
        family_contexts = [normalize_text(item) for item in ensure_list(group.get("family_contexts")) if normalize_text(item)]
        payload_states = [normalize_text(item) for item in ensure_list(group.get("payload_states")) if normalize_text(item)]
        emit_generic_row = bool(group.get("emit_generic_row", True))
        for value in ensure_list(group.get("levels")):
            value_text = normalize_text(value)
            if not value_text:
                continue
            if baseline_value and equivalent_level_value(value_text, baseline_value):
                continue
            row_contexts: list[tuple[dict[str, str], str, str, str]] = []
            if family_contexts and payload_states:
                for family_label in family_contexts:
                    for payload_state in payload_states:
                        assignment_map = {"polymer_identity": family_label}
                        for name, assignment_value in baseline_assignments.items():
                            low = name.lower()
                            if payload_state == "blank_control" and ("drug" in low or "etoposide" in low):
                                continue
                            assignment_map[name] = assignment_value
                        assignment_map[variable_name] = value_text
                        row_contexts.append(
                            (
                                assignment_map,
                                payload_state,
                                f"{family_label} [{variable_name}={value_text}] / {'Empty' if payload_state == 'blank_control' else 'Drug loaded'}",
                                f"{document_key}__single_variable__{normalize_token(family_label)}__{normalize_token(variable_name)}__{normalize_token(value_text)}__{payload_state}",
                            )
                        )
            if emit_generic_row or not row_contexts:
                assignment_map = dict(baseline_assignments)
                assignment_map[variable_name] = value_text
                row_contexts.append(
                    (
                        assignment_map,
                        "reported",
                        f"{variable_name}={value_text}",
                        f"{document_key}__single_variable__{normalize_token(variable_name)}__{normalize_token(value_text)}",
                    )
                )
            for assignment_map, payload_state, row_label, row_id in row_contexts:
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
                        "formulation_role": "control" if payload_state == "blank_control" else "reported",
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
                        "detail": f"single_variable::{row_label}",
                    }
                )
                emitted += 1
    return rows, jsonl_rows, traces, emitted


def _assignment_value_key(value: str) -> str:
    return re.sub(r"\s+", "", normalize_minus_signs(normalize_text(value)).lower())


def _existing_axis_value_supports_level(existing_value_key: str, level: str) -> bool:
    level_key = _assignment_value_key(level)
    return bool(level_key and (existing_value_key == level_key or level_key in existing_value_key))


def existing_assignment_values_for_axis(
    *,
    existing_rows: list[dict[str, Any]],
    variable_name: str,
) -> set[str]:
    """Return values already materialized for a variable axis in this scope.

    Single-variable narrative recovery is supplemental. When row-level table
    expansion has already emitted the same axis/value, the narrative axis must
    not duplicate those formulation identities.
    """
    axis_keys = {normalize_token(variable_name), normalize_text(variable_name).lower()}
    values: set[str] = set()
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        for blob_field in (TABLE_ASSIGNMENTS_FIELD, IDENTITY_VARIABLES_FIELD):
            parsed = parse_json_maybe(row.get(blob_field))
            for item in ensure_list(parsed):
                if not isinstance(item, dict):
                    continue
                if "name" in item and "value" in item:
                    name = item.get("name_raw") or item.get("name")
                    if normalize_token(name) in axis_keys or normalize_text(name).lower() in axis_keys:
                        value_key = _assignment_value_key(item.get("value_raw") or item.get("value"))
                        if value_key:
                            values.add(value_key)
                else:
                    for name, value in item.items():
                        if normalize_token(name) in axis_keys or normalize_text(name).lower() in axis_keys:
                            value_key = _assignment_value_key(value)
                            if value_key:
                                values.add(value_key)
    return values


def suppress_single_variable_levels_already_materialized(
    *,
    contract: dict[str, Any],
    existing_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not existing_rows or not isinstance(contract, dict):
        return contract
    filtered_groups: list[dict[str, Any]] = []
    changed = False
    for group in ensure_list(contract.get("groups")):
        if not isinstance(group, dict):
            continue
        levels = [normalize_text(item) for item in ensure_list(group.get("levels")) if normalize_text(item)]
        existing_values = existing_assignment_values_for_axis(
            existing_rows=existing_rows,
            variable_name=normalize_text(group.get("variable_name")),
        )
        if not levels or not existing_values:
            filtered_groups.append(group)
            continue
        retained_levels = [
            level
            for level in levels
            if not any(_existing_axis_value_supports_level(existing_value, level) for existing_value in existing_values)
        ]
        if retained_levels != levels:
            changed = True
        if retained_levels:
            updated = dict(group)
            updated["levels"] = retained_levels
            filtered_groups.append(updated)
    if not changed:
        return contract
    updated_contract = dict(contract)
    updated_contract["groups"] = filtered_groups
    return updated_contract


def first_cell_value(entry: dict[str, Any]) -> str:
    cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
    return cells[0] if cells else ""


def explicit_formulation_row_entries(row_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    explicit: list[dict[str, Any]] = []
    for entry in row_entries:
        if not isinstance(entry, dict):
            continue
        cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
        if is_numeric_index_row(cells):
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
    return recover_row_entry_headers(
        row_entries,
        first_explicit_row_index=first_explicit_row_index,
        width=width,
    )


def infer_measurement_columns_from_authority(
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
    measurement_columns: list[dict[str, Any]] = []
    for col_idx in range(1, width):
        header = headers[col_idx] if col_idx < len(headers) else ""
        if not header or not is_measurement_header(header):
            continue
        measurement_columns.append({"column_index": col_idx, "header": header})
    return measurement_columns


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
        if first_header and not re.search(r"\b(formulation|factorial|run|sample|number)\b", first_header):
            headers = [""] + headers[:-1]
    variable_columns: list[dict[str, Any]] = []
    for col_idx in range(1, width):
        header = headers[col_idx] if col_idx < len(headers) else ""
        if is_measurement_header(header):
            break
        if not header:
            continue
        variable_columns.append(
            {
                "column_index": col_idx,
                "header": header,
                "canonical_field": canonical_field_for_header(header),
            }
        )
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
    measurement_columns = infer_measurement_columns_from_authority(row_entries, explicit_rows)
    if not variable_columns and not measurement_columns:
        return [], "no_formulation_variable_columns"
    source_csv_path = normalize_text(authority_payload.get("source_csv_path"))
    extracted_rows: list[dict[str, Any]] = []
    current_group_context: dict[str, str] = {}
    row_entries_by_index = {int(entry.get("row_index") or 0): entry for entry in row_entries if isinstance(entry, dict)}
    ordered_row_indices = [int(entry.get("row_index") or 0) for entry in row_entries if isinstance(entry, dict)]

    def _append_assignment(assignments: list[dict[str, str]], *, raw_header: str, canonical_field: str, value: str, source_row_index: str, source_column_index: str) -> None:
        assignments.append(
            {
                "name": raw_header,
                "value": value,
                "canonical_field": canonical_field,
                "source_row_index": source_row_index,
                "source_column_index": source_column_index,
            }
        )

    for entry in explicit_rows:
        cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
        source_row_index = normalize_text(entry.get("row_index"))
        assignments: list[dict[str, str]] = []
        cell_bindings: list[dict[str, str]] = []
        used_columns: set[int] = set()
        for column in variable_columns:
            col_idx = int(column["column_index"])
            if col_idx >= len(cells):
                continue
            value = normalize_text(cells[col_idx])
            if not value:
                continue
            raw_header = normalize_assignment_name(column["header"])
            canonical_field = normalize_text(column.get("canonical_field")) or canonical_field_for_header(raw_header)
            _append_assignment(assignments, raw_header=raw_header, canonical_field=canonical_field, value=value, source_row_index=source_row_index, source_column_index=str(col_idx))
            if canonical_field:
                cell_bindings.append(
                    {
                        "source_csv_path": source_csv_path,
                        "source_row_index": source_row_index,
                        "source_column_index": str(col_idx),
                        "raw_header": raw_header,
                        "canonical_field": canonical_field,
                        "raw_cell_value": value,
                        "binding_rule": "stage2_header_alias_cell_binding",
                        "ambiguity_status": "unique_header_cell",
                    }
                )
            used_columns.add(col_idx)
        for carried_field, carried_value in current_group_context.items():
            if not carried_value:
                continue
            if any(item.get("canonical_field") == carried_field for item in assignments):
                continue
            _append_assignment(assignments, raw_header=carried_field, canonical_field=carried_field, value=carried_value, source_row_index=source_row_index, source_column_index="")
        for column in measurement_columns:
            col_idx = int(column["column_index"])
            if col_idx in used_columns or col_idx >= len(cells):
                continue
            value = normalize_text(cells[col_idx])
            if not value:
                continue
            raw_header = normalize_assignment_name(column["header"])
            canonical_field = canonical_field_for_header(raw_header)
            _append_assignment(assignments, raw_header=raw_header, canonical_field=canonical_field, value=value, source_row_index=source_row_index, source_column_index=str(col_idx))
            if canonical_field:
                cell_bindings.append(
                    {
                        "source_csv_path": source_csv_path,
                        "source_row_index": source_row_index,
                        "source_column_index": str(col_idx),
                        "raw_header": raw_header,
                        "canonical_field": canonical_field,
                        "raw_cell_value": value,
                        "binding_rule": "stage2_header_alias_cell_binding",
                        "ambiguity_status": "unique_header_cell",
                    }
                )
            used_columns.add(col_idx)
        if not assignments:
            continue
        extracted_rows.append(
            {
                "label": entry["row_label_info"]["label"],
                "label_number": entry["row_label_info"]["number"],
                "row_text": normalize_text(entry.get("row_text")) or " | ".join(value for value in cells if value),
                "assignments": assignments,
                "table_cell_bindings": cell_bindings,
            }
        )
        explicit_row_index = int(entry.get("row_index") or 0)
        try:
            ordered_pos = ordered_row_indices.index(explicit_row_index)
        except ValueError:
            ordered_pos = -1
        if ordered_pos >= 0:
            for next_row_index in ordered_row_indices[ordered_pos + 1 :]:
                next_entry = row_entries_by_index.get(next_row_index)
                if not isinstance(next_entry, dict):
                    continue
                next_cells = [normalize_text(cell) for cell in ensure_list(next_entry.get("cells"))]
                if next_cells and parse_formulation_row_label_info(next_cells[0]) is not None:
                    break
                group_context = extract_continuation_group_context(next_entry, variable_columns=variable_columns)
                if not group_context:
                    continue
                current_group_context.update(group_context)
                for carried_field, carried_value in group_context.items():
                    if not carried_value:
                        continue
                    replaced = False
                    for item in extracted_rows[-1]["assignments"]:
                        if item.get("canonical_field") == carried_field:
                            item["value"] = carried_value
                            replaced = True
                            break
                    if not replaced:
                        _append_assignment(extracted_rows[-1]["assignments"], raw_header=carried_field, canonical_field=carried_field, value=carried_value, source_row_index=source_row_index, source_column_index="")
                break
    if not extracted_rows:
        return [], "no_assignment_rows_matched"
    return extracted_rows, ""


def measurement_like_cell_count(cells: list[str], *, start_index: int) -> int:
    count = 0
    for value in cells[start_index:]:
        text = normalize_text(value)
        if not text:
            continue
        if re.search(r"\d", text):
            count += 1
    return count


def is_probable_footnote_label(label: str) -> bool:
    low = normalize_text(label).lower()
    if not low:
        return True
    if len(low) > 80:
        return True
    if low.startswith(("a ", "b ", "c ", "note", "where ")):
        return True
    if re.search(r"\bamounts?\b.*\bmaintained\b", low):
        return True
    if re.search(r"\bprepared with\b.*\bmethod\b", low):
        return True
    return False


def inline_table_value_score(tokens: list[str]) -> int:
    score = 0
    for token in tokens:
        text = normalize_text(token)
        if not text:
            continue
        if text in {"-", "–", "—", "−"}:
            score += 1
            continue
        if re.search(r"\d", text):
            score += 1
    return score


def extract_compact_inline_table_rows_from_source_text(
    *,
    document: dict[str, Any],
    scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    source_text = load_document_source_text(document)
    table_id = normalize_text(scope.get("table_id"))
    table_number_match = re.search(r"table\s*(\d{1,3})", table_id, flags=re.IGNORECASE)
    if not source_text or table_number_match is None:
        return [], "source_text_or_table_number_missing"
    table_number = table_number_match.group(1)
    anchor_match = re.search(rf"table\s*{re.escape(table_number)}\b", source_text, flags=re.IGNORECASE)
    if anchor_match is None:
        return [], "table_anchor_not_found_in_source_text"
    window = normalize_text(source_text[anchor_match.start(): anchor_match.start() + 1800])
    row_label_pattern = re.compile(r"\b[A-Z]{2,6}\d{1,3}\b")
    row_matches = list(row_label_pattern.finditer(window))
    if len(row_matches) < 3:
        return [], "insufficient_inline_row_labels"
    header_block = window[: row_matches[0].start()]
    formulation_anchor = re.search(r"Formulation\b", header_block, flags=re.IGNORECASE)
    if formulation_anchor is not None:
        header_block = header_block[formulation_anchor.start():]
    header_names = re.findall(r"([A-Z][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9\-]+)*\s*\([^)]{1,20}\))", header_block)
    header_names = [normalize_text(name) for name in header_names if normalize_text(name)]
    if len(header_names) < 2:
        return [], "inline_header_parse_failed"
    data_column_count = len(header_names)
    extracted_rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for idx, match in enumerate(row_matches):
        next_start = row_matches[idx + 1].start() if idx + 1 < len(row_matches) else len(window)
        row_label = normalize_text(match.group(0))
        if row_label in seen_labels:
            break
        token_block = window[match.end():next_start]
        value_tokens = [normalize_text(token) for token in re.findall(r"[A-Za-z0-9.%/-]+|[–—−-]", token_block) if normalize_text(token)]
        if len(value_tokens) < data_column_count:
            continue
        value_tokens = value_tokens[:data_column_count]
        if inline_table_value_score(value_tokens) < max(2, data_column_count - 1):
            continue
        assignments: list[dict[str, str]] = []
        for header_name, value in zip(header_names, value_tokens):
            if value in {"-", "–", "—", "−"}:
                continue
            assignments.append(
                {
                    "name": normalize_assignment_name(header_name),
                    "value": value,
                }
            )
        extracted_rows.append(
            {
                "label": row_label,
                "label_number": str(idx + 1),
                "row_text": normalize_text(f"{row_label} {' '.join(value_tokens)}"),
                "assignments": assignments,
            }
        )
        seen_labels.add(row_label)
    if len(extracted_rows) < 3:
        return [], "inline_row_parse_failed"
    return extracted_rows, ""


def _authority_source_csv_path(authority_payload: dict[str, Any]) -> Path | None:
    source_csv_path = normalize_text(authority_payload.get("source_csv_path"))
    if not source_csv_path or "#" in source_csv_path:
        return None
    candidate = Path(source_csv_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate if candidate.exists() else None


def _corrupted_table_family_labels(text: str) -> list[str]:
    normalized = normalize_text(text)
    noun = ""
    if re.search(r"\bnanocapsules\b", normalized, flags=re.IGNORECASE):
        noun = "nanocapsules"
    elif re.search(r"\bnanospheres\b", normalized, flags=re.IGNORECASE):
        noun = "nanospheres"
    payload_tokens: list[str] = []
    for token in ["XAN", "3-MeOXAN", "Empty"]:
        if re.search(rf"\b{re.escape(token)}\b", normalized, flags=re.IGNORECASE):
            payload_tokens.append(token)
    if noun and len(payload_tokens) >= 2:
        return [f"{token} {noun}" for token in payload_tokens]
    labels = re.findall(
        r"((?:Empty|[A-Za-z0-9\-]+(?:-loaded)?)\s+nano(?:sphere|capsule)s?)",
        text,
        flags=re.IGNORECASE,
    )
    return unique_nonempty([normalize_text(label) for label in labels])


def _segment_looks_like_sweep_row(segment: str) -> bool:
    text = normalize_text(segment)
    if not text:
        return False
    if "crystals" in text.lower():
        return True
    tail = re.sub(r"^\s*\d+(?:\.\d+)?\s*", "", text)
    numeric_hits = re.findall(r"\d+(?:\.\d+)?(?:G\d+(?:\.\d+)?)?", tail)
    return len(numeric_hits) >= 2


def _leading_theoretical_concentration(segment: str) -> str:
    match = re.match(r"\s*(\d+(?:\.\d+)?)\b", normalize_text(segment))
    return match.group(1) if match else ""


def _build_sweep_direct_row(*, label: str, concentration_value: str, row_text: str, crystals: bool = False) -> dict[str, Any]:
    assignments = [
        {"name": "formulation_identity_label", "value": label},
        {"name": "theoretical concentration", "value": f"{concentration_value} mg/mL"},
    ]
    if crystals:
        assignments.append({"name": "crystallization_status", "value": "crystals_observed"})
    return {
        "label": label,
        "label_number": concentration_value,
        "row_text": normalize_text(row_text),
        "assignments": assignments,
        "instance_role": formulation_role_from_label(label),
    }


def extract_split_column_concentration_sweep_rows_from_source_csv(
    *,
    authority_payload: dict[str, Any],
    document: dict[str, Any],
    scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    semantic_signals = document.get("semantic_signals") if isinstance(document.get("semantic_signals"), dict) else {}
    if not bool(semantic_signals.get("has_variable_sweep")):
        return [], "semantic_signal_missing_variable_sweep"
    representation_status = normalize_text(authority_payload.get("representation_status")).lower()
    normalization_actions = {
        normalize_text(action).lower()
        for action in ensure_list(authority_payload.get("normalization_actions"))
        if normalize_text(action)
    }
    # Some preserved PDF coordinate-grid assets are only summary-repaired enough for
    # prompt visibility but still retain a lawful split-column sweep in the original
    # CSV.  Do not let the newer representation label suppress an LLM-authorized
    # variable-sweep scope before the source CSV has been inspected.
    source_csv = _authority_source_csv_path(authority_payload)
    if source_csv is None:
        return [], "source_csv_missing"
    raw_text = source_csv.read_text(encoding="utf-8", errors="replace")
    family_labels = _corrupted_table_family_labels("\n".join(raw_text.splitlines()[:14]))
    source_has_split_sweep_surface = len(family_labels) >= 2 and any(
        _leading_theoretical_concentration(line) and _segment_looks_like_sweep_row(line)
        for line in raw_text.splitlines()
    )
    if representation_status not in {"repair_insufficient", "unrepaired_corrupted"} and not (
        representation_status == "repaired_summary" and "preserve_coordinate_grid" in normalization_actions
    ) and not source_has_split_sweep_surface:
        return [], "representation_not_corrupted_sweep_source"
    if len(family_labels) < 2:
        return [], "insufficient_family_labels"
    extracted_rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    lines = [line.strip() for line in raw_text.splitlines() if normalize_text(line)]
    for line in lines:
        if re.search(r"values\s+express", line, flags=re.IGNORECASE):
            if extracted_rows:
                break
            continue
        parts = [normalize_text(part) for part in line.split(",")]
        left = parts[0] if parts else ""
        right = parts[1] if len(parts) > 1 else ""
        left_conc = _leading_theoretical_concentration(left)
        right_conc = _leading_theoretical_concentration(right)
        if left_conc and right_conc and len(family_labels) >= 2 and _segment_looks_like_sweep_row(left) and _segment_looks_like_sweep_row(right):
            paired = [(family_labels[0], left_conc, left), (family_labels[1], right_conc, right)]
        elif left_conc and len(family_labels) >= 2 and _segment_looks_like_sweep_row(left):
            # In two-family split tables, a damaged coordinate row may either be
            # a shared same-concentration row whose right-side cells were folded
            # into the left segment (for example two ND/crystal outcomes), or a
            # genuinely left-only tail row.  Replicating every left-only segment
            # to both families invents a false right-family formulation.
            replicate_to_both = (
                len(re.findall(r"\bND\b", left, flags=re.IGNORECASE)) >= 2
                or len(re.findall(r"\d+(?:\.\d+)?", left)) >= 5
                or all(label.lower() in left.lower() for label in family_labels[:2])
            )
            if not replicate_to_both and re.search(r"\bND\b", left, flags=re.IGNORECASE) and "crystals" in left.lower():
                continue
            paired = (
                [(family_label, left_conc, left) for family_label in family_labels[:2]]
                if replicate_to_both
                else [(family_labels[0], left_conc, left)]
            )
        else:
            continue
        for family_label, conc, segment_text in paired:
            label = f"{family_label} (Theoretical concentration {conc} mg/mL)"
            label_key = normalize_token(label)
            if not label_key or label_key in seen_labels:
                continue
            seen_labels.add(label_key)
            extracted_rows.append(
                _build_sweep_direct_row(
                    label=label,
                    concentration_value=conc,
                    row_text=segment_text or line,
                    crystals="crystals" in normalize_text(segment_text or line).lower(),
                )
            )
    if len(extracted_rows) < 4:
        return [], "corrupted_split_column_sweep_parse_failed"
    return extracted_rows, ""


def extract_caption_sample_rows_from_source_text(
    *,
    document: dict[str, Any],
    scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    source_text = load_document_source_text(document)
    table_id = normalize_text(scope.get("table_id"))
    table_number_match = re.search(r"table\s*(\d{1,3})", table_id, flags=re.IGNORECASE)
    if not source_text or table_number_match is None:
        return [], "source_text_or_table_number_missing"
    table_number = table_number_match.group(1)
    block_match = re.search(
        rf"Table\s*{re.escape(table_number)}\b(.{{0,1800}}?)((?:[A-Za-z0-9\-]+-loaded|Empty)\s+nanocapsules?.{{0,400}})",
        source_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if block_match is None:
        return [], "table_block_not_found"
    window = normalize_text(block_match.group(0))
    extracted_rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    sample_pattern = re.compile(
        r"((?:Empty|[A-Za-z0-9\-]+-loaded)\s+nanocapsules?\s*\([^)]*?\))",
        flags=re.IGNORECASE,
    )
    for sample_match in sample_pattern.finditer(window):
        label = normalize_text(sample_match.group(1))
        if not label:
            continue
        label_key = normalize_token(label)
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)
        concentration_match = re.search(r"theoretical\s+concentration\s+of\s+(\d+(?:\.\d+)?)\s*mg/mL", label, flags=re.IGNORECASE)
        assignments = [{"name": "formulation_identity_label", "value": label}]
        if concentration_match:
            assignments.append({"name": "theoretical concentration", "value": f"{concentration_match.group(1)} mg/mL"})
        if re.search(r"without\s+xanthones", label, flags=re.IGNORECASE):
            assignments.append({"name": "payload_state", "value": "empty"})
        extracted_rows.append(
            {
                "label": label,
                "label_number": str(len(extracted_rows) + 1),
                "row_text": window,
                "assignments": assignments,
                "instance_role": formulation_role_from_label(label),
            }
        )
    if len(extracted_rows) < 2:
        return [], "caption_sample_row_parse_failed"
    return extracted_rows, ""


def extract_empty_control_characterization_row_from_source_text(
    *,
    document: dict[str, Any],
    scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Recover measured blank-control rows from compact characterization tables.

    This post-LLM path is intentionally narrow: it only emits the empty/control
    row from a table whose caption states it characterizes empty and loaded
    nanospheres/nanocapsules.  Loaded rows in these characterization tables often
    duplicate already-enumerated concentration-sweep formulation rows.
    """
    source_text = load_document_source_text(document)
    table_id = normalize_text(scope.get("table_id"))
    table_number_match = re.search(r"table\s*(\d{1,3})", table_id, flags=re.IGNORECASE)
    if not source_text or table_number_match is None:
        return [], "source_text_or_table_number_missing"
    table_number = table_number_match.group(1)
    anchor_match = re.search(
        rf"\bTable\s*{re.escape(table_number)}\s+Mean\s+diameter\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if anchor_match is None:
        return [], "table_anchor_not_found_in_source_text"
    window = normalize_text(source_text[anchor_match.start(): anchor_match.start() + 1600])
    if not re.search(r"\bMean\s+diameter\b", window, flags=re.IGNORECASE):
        return [], "not_characterization_table"
    if not re.search(r"\bempty\s+and\s+loaded\b", window, flags=re.IGNORECASE):
        return [], "not_empty_loaded_characterization_table"
    if not (
        re.search(r"\bEmpty\s+nano(?:sphere|capsule)s?\s+XAN\s+nano(?:sphere|capsule)s?", window, flags=re.IGNORECASE)
        and re.search(r"\b3-?MeOXAN\s+nano(?:sphere|capsule)s?", window, flags=re.IGNORECASE)
    ):
        return [], "empty_loaded_header_missing"
    family_match = re.search(r"\b(nanospheres|nanocapsules)\b", window, flags=re.IGNORECASE)
    if family_match is None:
        return [], "nanoparticle_family_missing"
    family = family_match.group(1).lower()
    label = f"Empty {family}"
    assignments = [{"name": "formulation_identity_label", "value": label}, {"name": "payload_state", "value": "empty"}]
    metric_match = re.search(
        r"Diameter\s*\(nm\)\s+(?P<diameter>\d+G\d+).*?PI\s+(?P<pi>\d+\.\d+G\d+\.\d+).*?z\s*\(mV\)\s+(?P<zeta>[K−-]?\d+\.\d+G\d+\.\d+)",
        window,
        flags=re.IGNORECASE,
    )
    if metric_match:
        assignments.extend(
            [
                {"name": "particle size", "value": normalize_text(metric_match.group("diameter"))},
                {"name": "polydispersity index", "value": normalize_text(metric_match.group("pi"))},
                {"name": "zeta potential", "value": normalize_text(metric_match.group("zeta"))},
            ]
        )
    return [
        {
            "label": label,
            "label_number": "1",
            "row_text": window,
            "assignments": assignments,
            "instance_role": "control",
            "source_region_type": "source_text_table",
            "evidence_span_text": window,
            "change_context_tag": "empty_control_characterization_recovery",
        }
    ], ""


def extract_first_column_identity_rows_from_authority(
    *,
    authority_payload: dict[str, Any],
    row_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if len(row_entries) < 2:
        return [], "insufficient_row_entries"
    if len(row_entries) > 16:
        return [], "first_column_identity_row_count_out_of_bounds"
    headers = [normalize_text(value) for value in ensure_list((authority_payload.get("header_structure") or {}).get("flattened_headers"))]
    if not headers:
        matrix = ensure_list(authority_payload.get("normalized_matrix"))
        if matrix:
            headers = [normalize_text(value) for value in ensure_list(matrix[0])]
    if not headers:
        return [], "missing_header_surface"
    measurement_start_idx = None
    for idx, header in enumerate(headers):
        if is_measurement_header(header):
            measurement_start_idx = idx
            break
    if measurement_start_idx is None and row_entries:
        first_cells = [normalize_text(cell) for cell in ensure_list(row_entries[0].get("cells"))]
        for idx, header in enumerate(first_cells):
            if is_measurement_header(header):
                measurement_start_idx = idx
                break
    if measurement_start_idx is None and row_entries and isinstance(row_entries[0].get("cell_map"), dict):
        header_candidates = [normalize_text(key) for key in row_entries[0].get("cell_map", {}).keys()]
        for idx, header in enumerate(header_candidates):
            if is_measurement_header(header):
                measurement_start_idx = idx
                break
    if measurement_start_idx is None or measurement_start_idx < 1:
        return [], "first_column_identity_missing_measurement_tail"
    if len(headers) - measurement_start_idx < 2:
        return [], "insufficient_measurement_columns"
    extracted_rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    temporal_label_hits = 0
    for ordinal, entry in enumerate(row_entries, start=1):
        cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
        if len(cells) < measurement_start_idx + 1:
            continue
        raw_label = normalize_text(cells[0])
        if not raw_label or is_measurement_header(raw_label) or is_probable_footnote_label(raw_label):
            continue
        if is_temporal_followup_label(raw_label):
            temporal_label_hits += 1
        if measurement_like_cell_count(cells, start_index=measurement_start_idx) < 2:
            continue
        label_key = normalize_token(raw_label)
        if not label_key or label_key in seen_labels:
            continue
        seen_labels.add(label_key)
        assignments = [
            {
                "name": "formulation_identity_label",
                "value": raw_label,
            }
        ]
        for col_idx in range(measurement_start_idx, min(len(cells), len(headers))):
            value = normalize_text(cells[col_idx])
            header = normalize_text(headers[col_idx])
            if not value or not header or not is_measurement_header(header):
                continue
            assignments.append(
                {
                    "name": normalize_assignment_name(header),
                    "value": value,
                }
            )
        extracted_rows.append(
            {
                "label": raw_label,
                "label_number": str(ordinal),
                "row_text": normalize_text(entry.get("row_text")) or " | ".join(value for value in cells if value),
                "assignments": assignments,
            }
        )
    if temporal_label_hits >= max(2, len(extracted_rows)):
        return [], "temporal_followup_series_not_formulation_table"
    if len(extracted_rows) < 2:
        return [], "no_first_column_identity_rows"
    return extracted_rows, ""


def extract_rowwise_formulation_rows_from_authority(
    *,
    authority_payload: dict[str, Any],
    row_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if len(row_entries) < 2:
        return [], "insufficient_row_entries"
    if len(row_entries) > 12:
        return [], "rowwise_contract_row_count_out_of_bounds"
    headers = [normalize_text(value) for value in ensure_list((authority_payload.get("header_structure") or {}).get("flattened_headers"))]
    if not headers:
        matrix = ensure_list(authority_payload.get("normalized_matrix"))
        if matrix:
            headers = [normalize_text(value) for value in ensure_list(matrix[0])]
    if not headers:
        return [], "missing_header_surface"
    assignment_columns: list[dict[str, Any]] = []
    measurement_column_count = 0
    for idx, header in enumerate(headers):
        normalized_header = normalize_text(header)
        if not normalized_header:
            continue
        if is_measurement_header(normalized_header):
            measurement_column_count += 1
            continue
        if measurement_column_count == 0:
            assignment_columns.append({"column_index": idx, "header": normalized_header})
    if len(assignment_columns) < 2 or len(assignment_columns) > 5:
        return [], "insufficient_assignment_columns"
    if measurement_column_count < 2:
        return [], "insufficient_measurement_columns"
    extracted_rows: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(row_entries, start=1):
        cells = [normalize_text(cell) for cell in ensure_list(entry.get("cells"))]
        if len(cells) < len(assignment_columns):
            return [], "rowwise_assignment_cells_incomplete"
        assignments: list[dict[str, str]] = []
        for column in assignment_columns:
            col_idx = int(column["column_index"])
            if col_idx >= len(cells):
                return [], "rowwise_assignment_cells_incomplete"
            value = normalize_text(cells[col_idx])
            if not value:
                return [], "rowwise_assignment_cells_incomplete"
            assignments.append(
                {
                    "name": normalize_assignment_name(column["header"]),
                    "value": value,
                }
            )
        label = f"row_{ordinal:02d}"
        raw_label = normalize_text(cells[0]) if cells else ""
        if raw_label:
            label = f"{label}__{normalize_token(raw_label)}"
        extracted_rows.append(
            {
                "label": label,
                "label_number": str(ordinal),
                "row_text": normalize_text(entry.get("row_text")) or " | ".join(value for value in cells if value),
                "assignments": assignments,
            }
        )
    if len(extracted_rows) < 2:
        return [], "no_rowwise_assignment_rows"
    return extracted_rows, ""

def evaluate_simple_table_enumeration_contract(
    *,
    scope: dict[str, Any],
    boundary: dict[str, Any],
    explicit_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> tuple[bool, str, str]:
    table_type = normalize_text(scope.get("table_type")).lower()
    if bool(boundary.get("is_doe")):
        return False, "doe_scope_table", ""
    if table_type and table_type != "full_formulation":
        return False, f"table_type_not_simple:{table_type}", ""
    if len(explicit_rows) < 2:
        return False, "no_stable_row_identity_surface", ""
    identity_surface = row_identity_surface_kind(explicit_rows)
    if identity_surface not in {"numeric_first_column", "f_numeric_first_column"}:
        return False, "unsupported_row_identity_surface", identity_surface
    if not direct_rows:
        return False, "no_simple_row_assignments", identity_surface
    return True, "", identity_surface


def direct_rows_look_like_aggregate_variant_list(direct_rows: list[dict[str, Any]]) -> bool:
    for row in direct_rows:
        if not isinstance(row, dict):
            continue
        values = [
            normalize_text(item.get("value"))
            for item in ensure_list(row.get("assignments"))
            if isinstance(item, dict) and normalize_text(item.get("value"))
        ]
        values.append(normalize_text(row.get("label")))
        values.append(normalize_text(row.get("row_text")))
        values.append(normalize_text(row.get("evidence_span_text")))
        if row_values_look_like_aggregate_variant_list(values):
            return True
    return False


def weak_ratio_coordinate_text(value: Any) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    low = text.lower()
    if not re.search(r"\d\s*[:/]\s*\d", low):
        return False
    alpha_tokens = re.findall(r"[a-z]+", low)
    allowed_alpha = {"plga", "pla", "pcl", "peg", "np", "nps", "w", "o", "ow", "wo"}
    return all(token in allowed_alpha for token in alpha_tokens)


def row_values_look_like_aggregate_variant_list(values: list[Any]) -> bool:
    for raw_value in values:
        value = normalize_text(raw_value)
        value_low = value.lower()
        if " | " in value and sum(1 for marker in ("nanoparticle", "nanoparticles", "np", "nps") if marker in value_low) >= 2:
            return True
        if " | " in value and sum(1 for marker in ("blank", "fitc", "99mtc", "drug-free", "drug free") if marker in value_low) >= 2:
            return True
    return False


def direct_rows_look_like_weak_ratio_coordinate_surface(direct_rows: list[dict[str, Any]]) -> bool:
    if len(direct_rows) < 2:
        return False
    labels = [normalize_text(row.get("label")) for row in direct_rows if isinstance(row, dict)]
    if labels and len(labels) == len(direct_rows) and all(weak_ratio_coordinate_text(label) for label in labels):
        return True
    ratio_like_rows = 0
    for row in direct_rows:
        if not isinstance(row, dict):
            continue
        assignments = [item for item in ensure_list(row.get("assignments")) if isinstance(item, dict)]
        if not assignments:
            continue
        names = [normalize_assignment_name(item.get("name")) for item in assignments]
        values = [normalize_text(item.get("value")) for item in assignments]
        if all(name.startswith("formulation_header_part") for name in names if name) and any(weak_ratio_coordinate_text(value) for value in values):
            ratio_like_rows += 1
    return ratio_like_rows == len(direct_rows)


def assignment_rows_look_like_aggregate_variant_list(assignment_rows: list[dict[str, Any]]) -> bool:
    values = [
        assignment.get("variable_value")
        for assignment in assignment_rows
        if isinstance(assignment, dict) and normalize_text(assignment.get("variable_value"))
    ]
    return row_values_look_like_aggregate_variant_list(values)


def should_block_downstream_measurement_only_table(
    *,
    document: dict[str, Any],
    scope: dict[str, Any],
    authority_payload: dict[str, Any],
    direct_rows: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Block post-synthesis measurement/process-condition tables from row-universe expansion.

    This guard is intentionally narrow: it requires the LLM/document to flag
    downstream or measurement-only variants and the table surface to be a
    cryoprotectant / freeze-drying characterization surface.  Such rows describe
    processing/characterization conditions applied to an already selected NP, not
    independent synthesized formulation instances.
    """
    semantic_signals = document.get("semantic_signals") if isinstance(document.get("semantic_signals"), dict) else {}
    if not (
        semantic_signals.get("has_downstream_non_synthesis_variants")
        or semantic_signals.get("has_measurement_only_variants")
    ):
        return False, ""
    primary_axes = " ".join(
        normalize_text(item).lower()
        for item in ensure_list(semantic_signals.get("primary_variable_names"))
        if normalize_text(item)
    )
    scope_blob = " ".join(
        [
            normalize_text(scope.get("table_id")),
            normalize_text(scope.get("table_type")),
            normalize_text(scope.get("variable_name")),
            normalize_text(authority_payload.get("source_caption_or_title")),
            normalize_text(authority_payload.get("source_csv_path")),
            normalize_text(authority_payload.get("normalized_csv_path")),
            " ".join(normalize_text(row.get("row_text")) for row in direct_rows if isinstance(row, dict)),
            " ".join(
                normalize_text(item.get("name"))
                for row in direct_rows
                if isinstance(row, dict)
                for item in ensure_list(row.get("assignments"))
                if isinstance(item, dict)
            ),
        ]
    ).lower()
    if "cryoprotect" not in (primary_axes + " " + scope_blob):
        return False, ""
    if not re.search(r"\b(?:freeze[-\s]?dry|lyophili[sz])", scope_blob + " " + load_document_source_text(document).lower()[:20000]):
        return False, ""
    return True, "downstream_cryoprotectant_measurement_table"


def should_block_explicit_table_row_universe_surface(
    *,
    scope: dict[str, Any],
    boundary: dict[str, Any],
    role_info: dict[str, Any],
    direct_rows: list[dict[str, Any]] | None = None,
    assignment_rows: list[dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    table_type = normalize_text(scope.get("table_type")).lower()
    direct_rows = direct_rows or []
    assignment_rows = assignment_rows or []
    if direct_rows_look_like_aggregate_variant_list(direct_rows):
        return True, "aggregate_variant_list_table_row"
    if assignment_rows_look_like_aggregate_variant_list(assignment_rows):
        return True, "aggregate_variant_list_assignment_row"
    auxiliary_surface = bool(table_type and table_type not in {"full_formulation", "doe_table"} and not bool(boundary.get("is_doe")))
    if not auxiliary_surface:
        return False, ""
    if direct_rows_look_like_weak_ratio_coordinate_surface(direct_rows):
        return True, f"weak_ratio_coordinate_auxiliary_table:{table_type or 'doe'}"
    varying_variables = [
        normalize_text(item).lower()
        for item in ensure_list(role_info.get("varying_variables"))
        if normalize_text(item)
    ]
    scope_variable = normalize_text(scope.get("variable_name") or scope.get("declared_variable_name"))
    if scope_variable:
        varying_variables.append(scope_variable.lower())
    ratio_axis = any("ratio" in item or ":" in item for item in varying_variables)
    if ratio_axis and assignment_rows:
        values = [normalize_text(item.get("variable_value")) for item in assignment_rows if isinstance(item, dict)]
        if values and all(weak_ratio_coordinate_text(value) for value in values):
            return True, f"weak_ratio_assignment_auxiliary_table:{table_type or 'doe'}"
    return False, ""


def non_primary_direct_rows_look_measurement_only(
    *,
    scope: dict[str, Any],
    role_info: dict[str, Any],
    direct_rows: list[dict[str, Any]],
) -> tuple[bool, str]:
    table_type = normalize_text(scope.get("table_type")).lower()
    if not table_type or table_type == "full_formulation":
        return False, ""
    assignment_names: list[str] = []
    canonical_fields: set[str] = set()
    synthesis_like_assignments = 0
    measurement_or_output_assignments = 0
    generic_identity_or_unknown_assignments = 0
    formulation_identity_values: list[str] = []
    for direct_row in direct_rows:
        if not isinstance(direct_row, dict):
            continue
        row_label_text = normalize_text(direct_row.get("label") or direct_row.get("formulation_identity_label"))
        if row_label_text:
            formulation_identity_values.append(row_label_text)
        for item in ensure_list(direct_row.get("assignments")):
            if not isinstance(item, dict):
                continue
            name = normalize_assignment_name(item.get("name"))
            if not name:
                continue
            assignment_names.append(name)
            canonical_field = canonical_field_for_header(name)
            if canonical_field:
                canonical_fields.add(canonical_field)
            compat_field = compatibility_field_for_assignment(name)
            if compat_field in {
                "particle_size_nm",
                "size_nm",
                "pdi",
                "zeta_mV",
                "encapsulation_efficiency_percent",
                "loading_content_percent",
                "drug_loading_percent",
                "drug_loading_pct",
                "drug_content_percent",
                "yield_percent",
            } or is_measurement_header(name):
                measurement_or_output_assignments += 1
                continue
            if compat_field in {
                "polymer_name",
                "drug_name",
                "surfactant_name",
                "plga_mass_mg",
                "drug_feed_amount_text",
                "surfactant_concentration_text",
                "organic_phase_volume_ml",
                "external_aqueous_phase_volume_mL",
                "o_w_ratio",
                "drug_polymer_ratio",
                "polymer_ratio",
            }:
                synthesis_like_assignments += 1
                continue
            normalized_name = normalize_token(name)
            if normalized_name in {
                "formulationidentitylabel",
                "formulation_identity_label",
                "identitylabel",
                "identity_label",
                "label",
                "rowlabel",
                "row_label",
                "used",
            }:
                if normalized_name in {
                    "formulationidentitylabel",
                    "formulation_identity_label",
                    "identitylabel",
                    "identity_label",
                    "label",
                    "rowlabel",
                    "row_label",
                }:
                    formulation_identity_values.append(normalize_text(item.get("value")))
                generic_identity_or_unknown_assignments += 1
                continue
            if not compat_field:
                generic_identity_or_unknown_assignments += 1
    varying_variables = [
        normalize_text(item) for item in ensure_list(role_info.get("varying_variables")) if normalize_text(item)
    ]
    if varying_variables:
        assignment_blob = " | ".join(assignment_names).lower()
        varying_hits = 0
        for variable in varying_variables:
            low = variable.lower()
            tokens = [token for token in re.findall(r"[a-z0-9]+", low) if token not in SINGLE_VARIABLE_STOPWORDS]
            if low and low in assignment_blob:
                varying_hits += 1
                continue
            if tokens and all(token in assignment_blob for token in tokens):
                varying_hits += 1
        if varying_hits > 0:
            return False, ""
    aggregate_assignment_values = [
        normalize_text(item.get("value"))
        for direct_row in direct_rows
        for item in ensure_list(direct_row.get("assignments"))
        if isinstance(item, dict) and normalize_text(item.get("value"))
    ]
    for value in aggregate_assignment_values:
        value_low = value.lower()
        if " | " in value and sum(1 for marker in ("nanoparticle", "nanoparticles", "np", "nps") if marker in value_low) >= 2:
            return True, f"aggregate_variant_list_nonprimary_table:{table_type}"
        if " | " in value and sum(1 for marker in ("blank", "fitc", "99mtc", "drug-free", "drug free") if marker in value_low) >= 2:
            return True, f"aggregate_variant_list_nonprimary_table:{table_type}"
    if synthesis_like_assignments > 0:
        return False, ""
    identity_label_is_primary = any(
        formulation_identity_label_looks_primary(value) for value in formulation_identity_values if normalize_text(value)
    )
    if assignment_names and measurement_or_output_assignments + generic_identity_or_unknown_assignments >= len(assignment_names):
        if identity_label_is_primary:
            return False, ""
        return True, f"measurement_only_nonprimary_table:{table_type}"
    if assignment_names and not varying_variables and not canonical_fields:
        return True, f"nonprimary_table_without_synthesis_contract:{table_type}"
    if assignment_names and not varying_variables and measurement_or_output_assignments > 0:
        return True, f"measurement_only_nonprimary_table:{table_type}"
    return False, ""


def weak_duplicate_label_surface_labels(labels: list[str]) -> bool:
    if not labels:
        return False
    weak_count = 0
    for label in labels:
        text = normalize_text(label)
        low = text.lower()
        alpha_tokens = re.findall(r"[a-z]+", low)
        has_ratio_or_number = bool(re.search(r"\d\s*[:/]\s*\d|\d", low))
        allowed_alpha = all(token in {"plga", "pcl", "peg", "pla", "np", "nps"} for token in alpha_tokens)
        if has_ratio_or_number and (not alpha_tokens or allowed_alpha):
            weak_count += 1
    return weak_count == len(labels)


def should_block_nonprimary_duplicate_label_surface(
    *,
    scope: dict[str, Any],
    direct_rows: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
) -> bool:
    table_type = normalize_text(scope.get("table_type")).lower()
    if not direct_rows or not existing_rows:
        return False
    direct_labels = [normalize_token(row.get("label")) for row in direct_rows if normalize_token(row.get("label"))]
    if len(direct_labels) < 2:
        return False
    direct_label_set = set(direct_labels)
    if len(direct_label_set) < 2:
        return False
    auxiliary_surface = bool(table_type and table_type != "full_formulation") or weak_duplicate_label_surface_labels(direct_labels)
    if not auxiliary_surface:
        return False
    existing_label_set = set()
    for row in existing_rows:
        raw_label = normalize_token(row.get("raw_formulation_label"))
        if raw_label:
            existing_label_set.add(raw_label)
            continue
        local_instance_id = normalize_text(row.get("local_instance_id"))
        if "__table_" in local_instance_id and "__" in local_instance_id:
            suffix = local_instance_id.split("__", 2)[-1]
            suffix = suffix.replace("_/_", "/").replace("_", " ")
            derived_label = normalize_token(suffix)
            if derived_label:
                existing_label_set.add(derived_label)
    if len(existing_label_set) < 2:
        return False
    return direct_label_set.issubset(existing_label_set)


def should_block_as_doe_companion_duplicate(
    *,
    document: dict[str, Any],
    scope: dict[str, Any],
    boundary: dict[str, Any],
    direct_rows: list[dict[str, Any]],
    doe_rows_emitted: int,
) -> bool:
    if doe_rows_emitted <= 0:
        return False
    if bool(boundary.get("is_doe")):
        return False
    if normalize_text(scope.get("table_type")).lower() != "full_formulation":
        return False
    semantic_signals = document.get("semantic_signals") if isinstance(document.get("semantic_signals"), dict) else {}
    if semantic_signals.get("has_variable_sweep") is not True:
        return False
    if not direct_rows or len(direct_rows) != doe_rows_emitted:
        return False
    label_numbers: list[int] = []
    for row in direct_rows:
        number = normalize_text(row.get("label_number"))
        if not number.isdigit():
            return False
        label_numbers.append(int(number))
    expected = list(range(1, doe_rows_emitted + 1))
    return sorted(label_numbers) == expected


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
        row_text_low = row_text.lower()
        if re.search(r"\boptimized\s+by\s+(?:modifying|varying|changing|adjusting)\b", row_text_low):
            continue
        if re.search(r"\b(?:modified|varied|changed|adjusted)\s+.*\b(?:ratio|concentration|amount|level)s?\b", row_text_low) and "their" in row_text_low:
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
    simple_table_enumeration_attempted = "no"
    simple_table_enumeration_activated = "no"
    simple_table_rows_emitted = 0
    simple_table_block_reason = ""
    row_identity_surface_used = ""
    single_variable_recovery_attempted = "no"
    single_variable_rows_emitted = 0
    non_doe_single_variable_groups_detected = 0
    single_variable_recovery_source_type = ""
    single_variable_recovery_failure_reason = ""
    held_constant_context_source = ""
    variable_axes_detected: list[str] = []
    single_variable_recovery_consumed = False
    document_has_explicit_anchor_scope = False
    doe_path_attempted = "yes" if isinstance(doe_summary, dict) and normalize_text(doe_summary.get("doe_path_attempted") or doe_summary.get("doe_expansion_attempted")) == "yes" else "no"
    doe_rows_emitted = int((doe_summary or {}).get("doe_rows_emitted") or (doe_summary or {}).get("emitted_row_count") or 0)
    for preview_scope in scopes:
        if not preview_scope.get("is_formulation_table"):
            continue
        if marker_provenance(preview_scope) not in LLM_MARKER_SOURCES:
            continue
        preview_payload, _ = resolve_table_authority_payload_for_scope(preview_scope, normalized_payloads=normalized_payloads)
        if preview_payload is None:
            continue
        preview_entries = authority_row_entries(preview_payload)
        preview_direct_rows, _ = extract_direct_formulation_rows_from_authority(
            authority_payload=preview_payload,
            row_entries=preview_entries,
        )
        if not preview_direct_rows:
            preview_direct_rows, _ = extract_rowwise_formulation_rows_from_authority(
                authority_payload=preview_payload,
                row_entries=preview_entries,
            )
        if not preview_direct_rows:
            preview_direct_rows, _ = extract_first_column_identity_rows_from_authority(
                authority_payload=preview_payload,
                row_entries=preview_entries,
            )
        preview_column_rows, _ = extract_column_anchor_rows_from_authority(
            authority_payload=preview_payload,
            row_entries=preview_entries,
        )
        if preview_direct_rows or preview_column_rows:
            document_has_explicit_anchor_scope = True
            break
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
                "simple_table_enumeration_attempted": "no",
                "simple_table_enumeration_activated": "no",
                "simple_table_rows_emitted": "0",
                "simple_table_block_reason": "",
                "row_identity_surface_used": "",
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
        semantic_table_id = normalize_text(scope.get("table_id"))
        table_id = semantic_table_id
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
            "simple_table_enumeration_attempted": "no",
            "simple_table_enumeration_activated": "no",
            "simple_table_rows_emitted": "0",
            "simple_table_block_reason": "",
            "row_identity_surface_used": "",
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
        if bool(scope.get("source_text_caption_only")):
            authority_payload = {
                "table_id": table_id,
                "source_table_id": table_id,
                "normalized_csv_path": "",
                "source_csv_path": "",
                "row_entries": [],
            }
            payload_failure_reason = ""
            activation_row["reopen_resolution_status"] = "source_text_only"
            activation_row["normalized_payload_used"] = "no"
        else:
            authority_payload, payload_failure_reason = resolve_table_authority_payload_for_scope(
                scope,
                normalized_payloads=normalized_payloads,
            )
        if authority_payload is None:
            authority_payload = source_csv_authority_payload_from_scope(scope)
            if authority_payload is not None:
                payload_failure_reason = ""
                activation_row["reopen_resolution_status"] = "resolved_source_csv_locator_fallback"
                activation_row["normalized_payload_used"] = "no"
            else:
                authority_payload = source_text_authority_payload_from_scope(document=document, scope=scope)
                if authority_payload is not None:
                    payload_failure_reason = ""
                    activation_row["reopen_resolution_status"] = "resolved_source_text_locator_fallback"
                    activation_row["normalized_payload_used"] = "no"
        if authority_payload is None:
            activation_row["authorized"] = "yes"
            activation_row["skip_reason"] = "missing_table_authority_payload"
            activation_row["reopen_resolution_status"] = "failed"
            activation_row["reopen_failure_reason"] = payload_failure_reason or reopen_binding.get("reopen_failure_reason", "")
            table_activation_rows.append(activation_row)
            continue
        authority_table_id = normalize_text(authority_payload.get("table_id") or authority_payload.get("source_table_id"))
        authority_table_path = normalize_text(authority_payload.get("normalized_csv_path"))
        original_scope_table_id = semantic_table_id
        if original_scope_table_id:
            scope.setdefault("original_scope_table_id", original_scope_table_id)
            scope.setdefault("semantic_table_id", original_scope_table_id)
        scope["table_path"] = authority_table_path
        if authority_table_id:
            scope["table_id"] = authority_table_id
            table_id = authority_table_id
            activation_row["table_id"] = authority_table_id
        activation_row["table_path"] = authority_table_path
        activation_row["reopen_resolution_status"] = "resolved"
        activation_row["normalized_payload_used"] = reopen_binding.get("normalized_payload_used", "yes")
        activation_row["authorized"] = "yes"
        activation_row["called"] = "yes"
        role_info = variable_roles.get(table_id) or variable_roles.get(original_scope_table_id, {})
        authority_entries = authority_row_entries(authority_payload)
        explicit_rows = explicit_formulation_row_entries(authority_entries)
        direct_rows, direct_failure_reason = extract_direct_formulation_rows_from_authority(
            authority_payload=authority_payload,
            row_entries=authority_entries,
        )
        if not direct_rows:
            direct_rows, direct_failure_reason = extract_rowwise_formulation_rows_from_authority(
                authority_payload=authority_payload,
                row_entries=authority_entries,
            )
        if not direct_rows:
            direct_rows, direct_failure_reason = extract_first_column_identity_rows_from_authority(
                authority_payload=authority_payload,
                row_entries=authority_entries,
            )
        if not direct_rows:
            direct_rows, direct_failure_reason = extract_compact_inline_table_rows_from_source_text(
                document=document,
                scope=scope,
            )
        if not direct_rows:
            direct_rows, direct_failure_reason = extract_split_column_concentration_sweep_rows_from_source_csv(
                authority_payload=authority_payload,
                document=document,
                scope=scope,
            )
        if not direct_rows:
            direct_rows, direct_failure_reason = extract_caption_sample_rows_from_source_text(
                document=document,
                scope=scope,
            )
        if not direct_rows:
            direct_rows, direct_failure_reason = extract_empty_control_characterization_row_from_source_text(
                document=document,
                scope=scope,
            )
        if not direct_rows:
            direct_rows, direct_failure_reason = extract_characterization_pair_rows_from_source_text(
                document=document,
                scope=scope,
            )
        if direct_rows:
            blocked_downstream_table, downstream_block_reason = should_block_downstream_measurement_only_table(
                document=document,
                scope=scope,
                authority_payload=authority_payload,
                direct_rows=direct_rows,
            )
            if blocked_downstream_table:
                activation_row["skip_reason"] = downstream_block_reason
                table_activation_rows.append(activation_row)
                continue
            blocked_surface, blocked_reason = should_block_explicit_table_row_universe_surface(
                scope=scope,
                boundary=boundary,
                role_info=role_info,
                direct_rows=direct_rows,
            )
            if blocked_surface:
                activation_row["skip_reason"] = blocked_reason
                table_activation_rows.append(activation_row)
                continue
        if should_block_as_doe_companion_duplicate(
            document=document,
            scope=scope,
            boundary=boundary,
            direct_rows=direct_rows,
            doe_rows_emitted=doe_rows_emitted,
        ):
            activation_row["skip_reason"] = "blocked_by_successful_doe_companion_duplicate"
            table_activation_rows.append(activation_row)
            continue
        anchor_direct_rows = list(direct_rows)
        blocked_nonprimary_direct_rows = False
        blocked_nonprimary_reason = ""
        if direct_rows and normalize_text(scope.get("table_type")).lower() != "full_formulation":
            filtered_direct_rows: list[dict[str, Any]] = []
            blocked_row_reasons: list[str] = []
            for candidate_row in direct_rows:
                blocked_row, blocked_reason = non_primary_direct_rows_look_measurement_only(
                    scope=scope,
                    role_info=role_info,
                    direct_rows=[candidate_row],
                )
                if blocked_row:
                    blocked_nonprimary_direct_rows = True
                    if blocked_reason:
                        blocked_row_reasons.append(blocked_reason)
                    continue
                filtered_direct_rows.append(candidate_row)
            if blocked_row_reasons and not filtered_direct_rows:
                blocked_nonprimary_reason = blocked_row_reasons[0]
                direct_failure_reason = blocked_nonprimary_reason
            direct_rows = filtered_direct_rows
        if direct_rows and should_block_nonprimary_duplicate_label_surface(
            scope=scope,
            direct_rows=direct_rows,
            existing_rows=[candidate for candidate in rows if normalize_text(candidate.get("key")) == document_key],
        ):
            activation_row["skip_reason"] = "duplicate_nonprimary_label_surface"
            table_activation_rows.append(activation_row)
            continue
        simple_table_enumeration_attempted = "yes"
        simple_table_activated, simple_table_block_reason, row_identity_surface_used = evaluate_simple_table_enumeration_contract(
            scope=scope,
            boundary=boundary,
            explicit_rows=explicit_rows,
            direct_rows=direct_rows,
        )
        activation_row["simple_table_enumeration_attempted"] = "yes"
        activation_row["simple_table_enumeration_activated"] = "yes" if simple_table_activated else "no"
        activation_row["simple_table_block_reason"] = simple_table_block_reason
        activation_row["row_identity_surface_used"] = row_identity_surface_used
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
        simple_rows_for_scope = 0
        if direct_rows:
            if simple_table_activated:
                simple_table_enumeration_activated = "yes"
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
                        "instance_evidence_region_type": normalize_text(direct_row.get("source_region_type")) or "table_row",
                        "evidence_section": table_id,
                        "evidence_span_text": normalize_text(direct_row.get("evidence_span_text") or direct_row.get("row_text")),
                        "formulation_role": normalize_text(direct_row.get("instance_role")) or "reported",
                        "instance_context_tags": stringify_json(
                            [
                                "table_row_expansion",
                                "explicit_table_anchor" if normalize_text(direct_row.get("source_region_type")) != "narrative_text" else "characterization_pair_recovery",
                            ]
                        ),
                        "change_context_tags": stringify_json([normalize_text(direct_row.get("change_context_tag")) or "table_authorized_variation"]),
                        "change_descriptions": stringify_json(change_descriptions),
                        "change_role": "table_row_variation",
                        IDENTITY_VARIABLES_FIELD: stringify_json(identity_variables),
                        METHOD_GROUP_SIGNATURE_HINT_FIELD: group_hint,
                        TABLE_ID_FIELD: table_id,
                        TABLE_ROW_ID_FIELD: f"{table_id}::{label}",
                        TABLE_ASSIGNMENTS_FIELD: stringify_json([assignment_map]),
                        TABLE_CELL_BINDINGS_FIELD: stringify_json(ensure_list(direct_row.get("table_cell_bindings"))),
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
                                    "source_region_type": normalize_text(direct_row.get("source_region_type")) or "table_row",
                                    "source_locator_text": f"{table_id}::{label}",
                                    "supporting_snippet": normalize_text(direct_row.get("evidence_span_text") or direct_row.get("row_text")),
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
                    row[f"{compat_field}_evidence_region_type"] = "narrative_text" if normalize_text(direct_row.get("source_region_type")) == "narrative_text" else "table_cell"
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
                if simple_table_activated:
                    simple_table_rows_emitted += 1
                    simple_rows_for_scope += 1
        else:
            role_info = variable_roles.get(table_id) or variable_roles.get(original_scope_table_id, {})
            if marker_provenance(role_info) not in LLM_MARKER_SOURCES:
                activation_row["authorized"] = "yes"
                activation_row["skip_reason"] = "missing_llm_variable_roles"
                table_activation_rows.append(activation_row)
                continue
            varying_variables = [normalize_text(item) for item in ensure_list(role_info.get("varying_variables")) if normalize_text(item)]
            activation_row["varying_variable_count"] = str(len(varying_variables))
            activation_row["varying_variables"] = "|".join(varying_variables)
            column_rows, column_failure_reason = extract_column_anchor_rows_from_authority(
                authority_payload=authority_payload,
                row_entries=authority_entries,
            )
            if column_rows:
                blocked_surface, blocked_reason = should_block_explicit_table_row_universe_surface(
                    scope=scope,
                    boundary=boundary,
                    role_info=role_info,
                    direct_rows=column_rows,
                )
                if blocked_surface:
                    activation_row["skip_reason"] = blocked_reason
                    table_activation_rows.append(activation_row)
                    continue
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
            allow_anchorless_single_variable_recovery = bool(
                isinstance(document.get("semantic_signals"), dict)
                and document.get("semantic_signals", {}).get("has_variable_sweep")
                and bool(ensure_list(document.get("semantic_signals", {}).get("selected_condition_hints")))
            )
            if len(varying_variables) != 1:
                if emitted_rows_for_scope == 0 and not allow_anchorless_single_variable_recovery and not anchor_direct_rows:
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
                blocked_surface, blocked_reason = should_block_explicit_table_row_universe_surface(
                    scope=scope,
                    boundary=boundary,
                    role_info=role_info,
                    assignment_rows=assignment_rows,
                )
                if blocked_surface:
                    activation_row["skip_reason"] = blocked_reason
                    table_activation_rows.append(activation_row)
                    continue
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
            not single_variable_recovery_consumed
            and doe_rows_emitted == 0
            and (
                emitted_rows_for_scope > 0
                or allow_anchorless_single_variable_recovery
                or bool(
                    isinstance(document.get("semantic_signals"), dict)
                    and document.get("semantic_signals", {}).get("has_sequential_optimization")
                )
            )
        ):
            single_variable_recovery_attempted = "yes"
            single_variable_contract = build_single_variable_recovery_contract(
                document=document,
                require_anchor_rows=explicit_rows_for_scope > 0 or bool(anchor_direct_rows),
            )
            if bool(single_variable_contract.get("detected")):
                existing_rows_for_scope = [
                    candidate
                    for candidate in rows
                    if normalize_text(candidate.get(TABLE_ID_FIELD)) == table_id
                    and normalize_text(candidate.get("semantic_scope_ref")) == scope_id
                ]
                supplemental_anchor_rows, supplemental_anchor_jsonl, supplemental_anchor_traces, supplemental_anchor_count = emit_supplemental_family_anchor_rows(
                    document=document,
                    compatibility_columns=compatibility_columns,
                    contract=single_variable_contract,
                    scope=scope,
                    scope_id=scope_id,
                    table_id=table_id,
                    existing_rows=existing_rows_for_scope,
                    group_hint_prefix=f"{document_key}__single_variable_group",
                )
                if supplemental_anchor_count:
                    rows.extend(supplemental_anchor_rows)
                    jsonl_rows.extend(supplemental_anchor_jsonl)
                    traces.extend(supplemental_anchor_traces)
                    table_row_count += supplemental_anchor_count
                    emitted_rows_for_scope += supplemental_anchor_count
                    explicit_table_rows_emitted += supplemental_anchor_count
                    explicit_rows_for_scope += supplemental_anchor_count
                    existing_rows_for_scope.extend(supplemental_anchor_rows)
                single_variable_contract = suppress_single_variable_levels_already_materialized(
                    contract=single_variable_contract,
                    existing_rows=existing_rows_for_scope,
                )
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
        activation_row["simple_table_rows_emitted"] = str(simple_rows_for_scope)
        activation_row["non_doe_single_variable_groups_detected"] = str(non_doe_single_variable_groups_detected)
        activation_row["single_variable_recovery_attempted"] = single_variable_recovery_attempted
        activation_row["single_variable_rows_emitted"] = str(single_variable_rows_emitted)
        activation_row["single_variable_recovery_source_type"] = single_variable_recovery_source_type
        activation_row["single_variable_recovery_failure_reason"] = single_variable_recovery_failure_reason
        activation_row["held_constant_context_source"] = held_constant_context_source
        activation_row["variable_axis_detected"] = "|".join(variable_axes_detected)
        table_activation_rows.append(activation_row)

    if table_row_count == 0 and not single_variable_recovery_consumed and doe_rows_emitted == 0:
        anchorless_contract = build_single_variable_recovery_contract(
            document=document,
            require_anchor_rows=False,
        )
        anchorless_activation_row = {
            "function_unit": FUNCTION_UNIT_ID,
            "document_key": document_key,
            "table_id": "single_variable_context",
            "scope_id": f"{document_key}__anchorless_single_variable_scope__01",
            "table_type": "anchorless_single_variable_context",
            "marker_provenance": "llm_parsed",
            "considered": "yes",
            "authorized": "yes" if bool(anchorless_contract.get("detected")) else "no",
            "called": "no",
            "rows_emitted": "0",
            "rows_retained_after_projection": "0",
            "skip_reason": "",
            "table_path": "source_text",
            "varying_variable_count": "0",
            "varying_variables": "",
            "reopen_source_type": reopen_binding.get("reopen_source_type", ""),
            "reopen_resolution_status": "source_text_anchorless" if bool(anchorless_contract.get("detected")) else reopen_binding.get("reopen_resolution_status", ""),
            "reopen_failure_reason": "",
            "normalized_payload_used": "no",
            "doe_path_attempted": doe_path_attempted,
            "doe_rows_emitted": str(doe_rows_emitted),
            "fell_back_to_table_expansion": "no",
            "fallback_reason": "",
            "explicit_table_rows_emitted": "0",
            "simple_table_enumeration_attempted": "no",
            "simple_table_enumeration_activated": "no",
            "simple_table_rows_emitted": "0",
            "simple_table_block_reason": "",
            "row_identity_surface_used": "",
            "non_doe_single_variable_groups_detected": "0",
            "single_variable_recovery_attempted": "yes",
            "single_variable_rows_emitted": "0",
            "single_variable_recovery_source_type": normalize_text(anchorless_contract.get("source_type")),
            "single_variable_recovery_failure_reason": normalize_text(anchorless_contract.get("failure_reason")),
            "held_constant_context_source": normalize_text(anchorless_contract.get("held_constant_context_source")),
            "variable_axis_detected": "|".join(
                normalize_text(item.get("variable_name"))
                for item in ensure_list(anchorless_contract.get("groups"))
                if isinstance(item, dict) and normalize_text(item.get("variable_name"))
            ),
        }
        single_variable_recovery_attempted = "yes"
        if bool(anchorless_contract.get("detected")):
            anchorless_scope_id = anchorless_activation_row["scope_id"]
            anchorless_scope = {
                "scope_id": anchorless_scope_id,
                "table_id": "single_variable_context",
                "is_formulation_table": True,
                "table_type": "anchorless_single_variable_context",
                "marker_provenance": "llm_parsed",
                "declaration_basis": "llm_variable_sweep_plus_source_text_anchorless_contract",
            }
            recovered_rows, recovered_jsonl, recovered_traces, recovered_count = emit_single_variable_recovery_rows(
                document=document,
                compatibility_columns=compatibility_columns,
                contract=anchorless_contract,
                scope=anchorless_scope,
                scope_id=anchorless_scope_id,
                table_id="single_variable_context",
                group_hint_prefix=f"{document_key}__single_variable_group",
            )
            rows.extend(recovered_rows)
            jsonl_rows.extend(recovered_jsonl)
            traces.extend(recovered_traces)
            table_row_count += recovered_count
            single_variable_rows_emitted = recovered_count
            non_doe_single_variable_groups_detected = len(
                [item for item in ensure_list(anchorless_contract.get("groups")) if isinstance(item, dict)]
            )
            variable_axes_detected = [
                normalize_text(item.get("variable_name"))
                for item in ensure_list(anchorless_contract.get("groups"))
                if isinstance(item, dict) and normalize_text(item.get("variable_name"))
            ]
            single_variable_recovery_source_type = normalize_text(anchorless_contract.get("source_type"))
            held_constant_context_source = normalize_text(anchorless_contract.get("held_constant_context_source"))
            single_variable_recovery_failure_reason = "" if recovered_count else "no_nonbaseline_levels_emitted"
            anchorless_activation_row["called"] = "yes"
            anchorless_activation_row["rows_emitted"] = str(recovered_count)
            anchorless_activation_row["rows_retained_after_projection"] = str(recovered_count)
            anchorless_activation_row["single_variable_rows_emitted"] = str(recovered_count)
            anchorless_activation_row["single_variable_recovery_failure_reason"] = single_variable_recovery_failure_reason
            anchorless_activation_row["non_doe_single_variable_groups_detected"] = str(non_doe_single_variable_groups_detected)
            anchorless_activation_row["skip_reason"] = "" if recovered_count else single_variable_recovery_failure_reason
        else:
            single_variable_recovery_failure_reason = normalize_text(anchorless_contract.get("failure_reason"))
            anchorless_activation_row["skip_reason"] = single_variable_recovery_failure_reason or "anchorless_single_variable_contract_not_found"
        table_activation_rows.append(anchorless_activation_row)

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
        "simple_table_enumeration_attempted": simple_table_enumeration_attempted,
        "simple_table_enumeration_activated": simple_table_enumeration_activated,
        "simple_table_rows_emitted": simple_table_rows_emitted,
        "simple_table_block_reason": next(
            (row.get("simple_table_block_reason", "") for row in table_activation_rows if row.get("simple_table_block_reason")),
            "",
        ),
        "row_identity_surface_used": next(
            (row.get("row_identity_surface_used", "") for row in table_activation_rows if row.get("row_identity_surface_used")),
            "",
        ),
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
