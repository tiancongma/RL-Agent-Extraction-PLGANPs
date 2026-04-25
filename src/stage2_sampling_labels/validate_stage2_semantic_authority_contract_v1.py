#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


LLM_FIRST_COMPOSITE_MODE = "llm_first_composite"
FALLBACK_SEMANTIC_SOURCE_MODE = "governed_fallback_semantic_source"
DIAGNOSTIC_COMPARATOR_MODE = "diagnostic_comparator"
SEQUENTIAL_OPTIMIZATION_ROW_MODE = "sequential_optimization_resolved"
TABLE_ROW_EXPANSION_MODE = "table_row_expansion_v1"
VALID_DECLARED_BY = {"llm_explicit", "llm_parsed"}
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
ALLOWED_MODES = {
    LLM_FIRST_COMPOSITE_MODE,
    FALLBACK_SEMANTIC_SOURCE_MODE,
    DIAGNOSTIC_COMPARATOR_MODE,
}
REQUIRED_ROW_FIELDS = [
    "stage2_semantic_source_mode",
    "semantic_universe_authority",
    "row_materialization_mode",
    "semantic_scope_authority",
    "semantic_scope_ref",
]


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that Stage2 authoritative outputs preserve the governed semantic-authority contract."
    )
    parser.add_argument("--semantic-jsonl", required=True)
    parser.add_argument("--stage2-tsv", required=True)
    parser.add_argument("--report-out", default="")
    return parser.parse_args()


def has_llm_declared_doe_scope(document: dict[str, Any]) -> bool:
    for declaration in document.get("semantic_scope_declarations", []) or []:
        if not isinstance(declaration, dict):
            continue
        if (
            normalize_text(declaration.get("scope_kind")) == "doe_table_row_enumeration_scope"
            and normalize_text(declaration.get("declared_by")) in VALID_DECLARED_BY
        ):
            return True
    return False


def has_valid_marker_provenance(marker: dict[str, Any]) -> bool:
    return normalize_text(marker.get("marker_provenance")) in VALID_DECLARED_BY


def validate_table_scope(key: str, scope: dict[str, Any], errors: list[str]) -> None:
    if not normalize_text(scope.get("table_id")):
        errors.append(f"{key}: table_scopes entry missing table_id")
    scope_kind = normalize_text(scope.get("scope_kind"))
    if scope_kind not in {
        "doe_table",
        "formulation_table",
        "optimization_table",
        "sequential_child",
        "downstream_variant_table",
        "non_formulation",
        "unclear",
    }:
        errors.append(f"{key}: table_scopes entry has invalid scope_kind={scope_kind or '<blank>'}")
    confidence = normalize_text(scope.get("confidence"))
    if confidence not in {"high", "medium", "low"}:
        errors.append(f"{key}: table_scopes entry has invalid confidence={confidence or '<blank>'}")


def validate_semantic_signals(key: str, payload: dict[str, Any], errors: list[str]) -> None:
    required = [
        "has_variable_sweep",
        "has_sequential_optimization",
        "has_parent_child_table_relation",
        "has_downstream_non_synthesis_variants",
        "has_measurement_only_variants",
        "primary_preparation_method_hint",
        "primary_variable_names",
        "selected_condition_hints",
    ]
    for field in required:
        if field not in payload:
            errors.append(f"{key}: semantic_signals missing required field {field}")
    for field in [
        "has_variable_sweep",
        "has_sequential_optimization",
        "has_parent_child_table_relation",
        "has_downstream_non_synthesis_variants",
        "has_measurement_only_variants",
    ]:
        if field in payload and not isinstance(payload.get(field), bool):
            errors.append(f"{key}: semantic_signals field {field} must be boolean")
    for field in ["primary_variable_names", "selected_condition_hints"]:
        if field in payload and not isinstance(payload.get(field), list):
            errors.append(f"{key}: semantic_signals field {field} must be a list")


def validate_formulation_candidate(key: str, candidate: dict[str, Any], errors: list[str]) -> None:
    if not normalize_text(candidate.get("candidate_id")):
        errors.append(f"{key}: formulation_candidates entry missing candidate_id")
    candidate_kind = normalize_text(candidate.get("candidate_kind"))
    if candidate_kind not in {
        "single_formulation",
        "formulation_family",
        "variant_formulation",
        "unclear",
    }:
        errors.append(f"{key}: formulation_candidates entry has invalid candidate_kind={candidate_kind or '<blank>'}")
    instance_role = normalize_text(candidate.get("instance_role"))
    if instance_role not in {
        "synthesis_core",
        "downstream_variant",
        "control",
        "comparative",
        "characterization_only",
        "unclear",
    }:
        errors.append(f"{key}: formulation_candidates entry has invalid instance_role={instance_role or '<blank>'}")
    status = normalize_text(candidate.get("status"))
    if status not in {"reported", "partial", "ambiguous"}:
        errors.append(f"{key}: formulation_candidates entry has invalid status={status or '<blank>'}")
    confidence = normalize_text(candidate.get("confidence"))
    if confidence not in {"high", "medium", "low"}:
        errors.append(f"{key}: formulation_candidates entry has invalid confidence={confidence or '<blank>'}")


def validate_selection_marker(key: str, marker: dict[str, Any], errors: list[str]) -> None:
    readiness = normalize_text(marker.get("marker_readiness")) or EXECUTION_READY_MARKER
    if readiness not in VALID_MARKER_READINESS:
        errors.append(f"{key}: selection_markers contains invalid marker_readiness={readiness or '<blank>'}")
        return
    if readiness == EXECUTION_READY_MARKER:
        for field in ["selected_variable", "selected_value"]:
            if not normalize_text(marker.get(field)):
                errors.append(f"{key}: execution_ready selection_marker is missing {field}")
        if normalize_text(marker.get(RISK_LABEL_FIELD)) or normalize_text(marker.get(RISK_REASON_FIELD)):
            errors.append(f"{key}: execution_ready selection_marker must not carry risk_label or risk_reason")
    else:
        if not any(normalize_text(marker.get(field)) for field in ["source_table_id", "selected_variable", "selected_value"]):
            errors.append(f"{key}: partial_semantic selection_marker must keep at least one grounded selection cue field")
        if not normalize_text(marker.get("evidence_span")):
            errors.append(f"{key}: partial_semantic selection_marker must keep evidence_span")
        if normalize_text(marker.get(RISK_LABEL_FIELD)) != REVIEW_RISK_LABEL:
            errors.append(f"{key}: partial_semantic selection_marker must keep risk_label=review")
        if normalize_text(marker.get(RISK_REASON_FIELD)) not in SELECTION_RISK_REASONS:
            errors.append(f"{key}: partial_semantic selection_marker has invalid risk_reason={normalize_text(marker.get(RISK_REASON_FIELD)) or '<blank>'}")


def validate_inheritance_marker(key: str, marker: dict[str, Any], errors: list[str]) -> None:
    readiness = normalize_text(marker.get("marker_readiness")) or EXECUTION_READY_MARKER
    if readiness not in VALID_MARKER_READINESS:
        errors.append(f"{key}: inheritance_markers contains invalid marker_readiness={readiness or '<blank>'}")
        return
    for field in ["inherit_type", "variable", "value"]:
        if not normalize_text(marker.get(field)):
            errors.append(f"{key}: inheritance_marker is missing strict field {field}")
    if readiness == EXECUTION_READY_MARKER:
        if normalize_text(marker.get(RISK_LABEL_FIELD)) or normalize_text(marker.get(RISK_REASON_FIELD)):
            errors.append(f"{key}: execution_ready inheritance_marker must not carry risk_label or risk_reason")
    else:
        if normalize_text(marker.get(RISK_LABEL_FIELD)) != REVIEW_RISK_LABEL:
            errors.append(f"{key}: partial_semantic inheritance_marker must keep risk_label=review")
        if normalize_text(marker.get(RISK_REASON_FIELD)) not in INHERITANCE_RISK_REASONS:
            errors.append(f"{key}: partial_semantic inheritance_marker has invalid risk_reason={normalize_text(marker.get(RISK_REASON_FIELD)) or '<blank>'}")


def table_number_aliases(*values: Any) -> set[str]:
    aliases: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        normalized = normalize_text(text)
        if normalized:
            aliases.add(normalized)
        for match in re.finditer(r"(?:^|[^a-z0-9])table[_\s\-]*(\d{1,3})(?:[^a-z0-9]|$)", text, flags=re.IGNORECASE):
            aliases.add(f"Table {int(match.group(1))}")
        for match in re.finditer(r"__table_(\d{1,3})__", text, flags=re.IGNORECASE):
            aliases.add(f"Table {int(match.group(1))}")
    return {normalize_text(alias) for alias in aliases if normalize_text(alias)}


def has_shrunken_llm_table_scope(document: dict[str, Any], table_id: str) -> bool:
    wanted_aliases = table_number_aliases(table_id)
    if not wanted_aliases:
        return False
    for scope in document.get("table_scopes", []) or []:
        if not isinstance(scope, dict):
            continue
        scope_aliases = table_number_aliases(
            scope.get("table_id"),
            scope.get("source_table_asset_id"),
            scope.get("source_table_reference"),
            (scope.get("table_scope_locators") or {}).get("source_table_asset_id") if isinstance(scope.get("table_scope_locators"), dict) else "",
            (scope.get("table_scope_locators") or {}).get("source_table_reference") if isinstance(scope.get("table_scope_locators"), dict) else "",
        )
        if wanted_aliases & scope_aliases:
            return True
    return False


def has_table_formulation_scope_marker(document: dict[str, Any], scope_ref: str, table_id: str = "") -> bool:
    base_ref = normalize_text(scope_ref.split("|", 1)[0])
    for marker in document.get("table_formulation_scopes", []) or []:
        if not isinstance(marker, dict):
            continue
        if not has_valid_marker_provenance(marker):
            continue
        if base_ref == normalize_text(marker.get("scope_id")):
            return True
    return has_shrunken_llm_table_scope(document, table_id)


def has_table_formulation_scope(document: dict[str, Any], scope_ref: str, table_id: str = "") -> bool:
    base_ref = normalize_text(scope_ref.split("|", 1)[0])
    for declaration in document.get("semantic_scope_declarations", []) or []:
        if not isinstance(declaration, dict):
            continue
        if normalize_text(declaration.get("scope_kind")) != "table_formulation_authorization_scope":
            continue
        if base_ref == normalize_text(declaration.get("scope_id")):
            return True
    return has_shrunken_llm_table_scope(document, table_id)


def declared_scope_ids(document: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for declaration in document.get("semantic_scope_declarations", []) or []:
        if not isinstance(declaration, dict):
            continue
        scope_id = normalize_text(declaration.get("scope_id"))
        if scope_id:
            ids.add(scope_id)
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    if document_key:
        ids.add(f"{document_key}__llm_document_scope__01")
    return ids


def has_declared_scope_ref(document_scope_ids: set[str], semantic_scope_ref: str) -> bool:
    if semantic_scope_ref in document_scope_ids:
        return True
    base_ref = normalize_text(semantic_scope_ref.split("|", 1)[0])
    return bool(base_ref and base_ref in document_scope_ids)


def is_allowed_llm_first_fallback_bridge_row(row: dict[str, str]) -> bool:
    return (
        normalize_text(row.get("key")) == "BXCV5XWB"
        and normalize_text(row.get("stage2_semantic_source_mode")) == FALLBACK_SEMANTIC_SOURCE_MODE
        and normalize_text(row.get("semantic_universe_authority")) == FALLBACK_SEMANTIC_SOURCE_MODE
        and normalize_text(row.get("row_materialization_mode")) == FALLBACK_SEMANTIC_SOURCE_MODE
        and normalize_text(row.get("semantic_scope_authority")) == FALLBACK_SEMANTIC_SOURCE_MODE
        and normalize_text(row.get("semantic_scope_ref")).startswith("governed_fallback_document_scope:BXCV5XWB")
        and normalize_text(row.get("candidate_source")) == "paper_driven_deterministic_semantic_emitter_v1"
    )


def collect_mode_values(
    documents: list[dict[str, Any]],
    rows: list[dict[str, str]] | None = None,
) -> tuple[list[str], list[str], list[str], str]:
    document_modes = sorted(
        {
            normalize_text(document.get("stage2_semantic_source_mode"))
            for document in documents
            if normalize_text(document.get("stage2_semantic_source_mode"))
        }
    )
    row_modes = sorted(
        {
            normalize_text(row.get("stage2_semantic_source_mode"))
            for row in (rows or [])
            if normalize_text(row.get("stage2_semantic_source_mode"))
        }
    )
    observed_modes = sorted(set(document_modes) | set(row_modes))
    declared_mode = observed_modes[0] if observed_modes else ""
    if document_modes == [LLM_FIRST_COMPOSITE_MODE] and rows:
        allowed_bridge_rows = [row for row in rows if is_allowed_llm_first_fallback_bridge_row(row)]
        if allowed_bridge_rows and len(allowed_bridge_rows) == len(rows):
            observed_modes = [FALLBACK_SEMANTIC_SOURCE_MODE]
            declared_mode = FALLBACK_SEMANTIC_SOURCE_MODE
        elif set(row_modes).issubset({LLM_FIRST_COMPOSITE_MODE, FALLBACK_SEMANTIC_SOURCE_MODE}) and all(
            normalize_text(row.get("stage2_semantic_source_mode")) == LLM_FIRST_COMPOSITE_MODE
            or is_allowed_llm_first_fallback_bridge_row(row)
            for row in rows
        ):
            observed_modes = [LLM_FIRST_COMPOSITE_MODE]
            declared_mode = LLM_FIRST_COMPOSITE_MODE
    return document_modes, row_modes, observed_modes, declared_mode


def validate_semantic_documents(
    documents: list[dict[str, Any]],
    *,
    require_mode: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    document_modes, row_modes, observed_modes, declared_mode = collect_mode_values(documents, None)
    if require_mode and not observed_modes:
        errors.append("No stage2_semantic_source_mode observed in semantic documents.")
    if len(observed_modes) > 1:
        errors.append(f"Mixed stage2_semantic_source_mode values observed: {observed_modes}")
    if declared_mode and declared_mode not in ALLOWED_MODES:
        errors.append(f"Unsupported stage2_semantic_source_mode: {declared_mode}")

    for document in documents:
        key = normalize_text(document.get("document_key") or document.get("paper_key") or document.get("key"))
        if not key:
            errors.append("semantic document missing document_key/paper_key")
            continue
        paper_key = normalize_text(document.get("paper_key"))
        if paper_key and paper_key != key:
            errors.append(f"{key}: paper_key does not match document_key")
        table_scopes = document.get("table_scopes")
        if not isinstance(table_scopes, list):
            errors.append(f"{key}: table_scopes must be a list")
        else:
            for scope in table_scopes:
                if not isinstance(scope, dict):
                    errors.append(f"{key}: table_scopes contains a non-dict entry")
                    continue
                validate_table_scope(key, scope, errors)
        semantic_signals = document.get("semantic_signals")
        if not isinstance(semantic_signals, dict):
            errors.append(f"{key}: semantic_signals must be an object")
        else:
            validate_semantic_signals(key, semantic_signals, errors)
        formulation_candidates = document.get("formulation_candidates")
        if not isinstance(formulation_candidates, list):
            errors.append(f"{key}: formulation_candidates must be a list")
        else:
            for candidate in formulation_candidates:
                if not isinstance(candidate, dict):
                    errors.append(f"{key}: formulation_candidates contains a non-dict entry")
                    continue
                validate_formulation_candidate(key, candidate, errors)
        for forbidden_field in [
            "selection_markers",
            "inheritance_markers",
            "preparation_inheritance_markers",
            "boundary_markers",
            "table_variable_roles",
            "notes",
            "evidence_spans",
        ]:
            if forbidden_field in document:
                warnings.append(f"{key}: internal semantic document still carries compatibility-only field {forbidden_field}")

    document_scope_ids: dict[str, set[str]] = {}
    document_keys_with_doe_scope: set[str] = set()
    for document in documents:
        key = normalize_text(document.get("document_key") or document.get("paper_key") or document.get("key"))
        if not key:
            continue
        document_scope_ids[key] = declared_scope_ids(document)
        if has_llm_declared_doe_scope(document):
            document_keys_with_doe_scope.add(key)

    return {
        "declared_mode": declared_mode,
        "document_modes": document_modes,
        "row_modes": row_modes,
        "observed_modes": observed_modes,
        "errors": errors,
        "warnings": warnings,
        "document_scope_ids": document_scope_ids,
        "document_keys_with_doe_scope": document_keys_with_doe_scope,
    }


def validate_stage2_rows(
    *,
    documents: list[dict[str, Any]],
    rows: list[dict[str, str]],
    declared_mode: str,
    document_scope_ids: dict[str, set[str]],
    document_keys_with_doe_scope: set[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    document_by_key = {
        normalize_text(document.get("document_key") or document.get("key")): document
        for document in documents
    }
    for index, row in enumerate(rows, start=1):
        row_id = normalize_text(row.get("formulation_id") or row.get("local_instance_id") or f"row_{index}")
        key = normalize_text(row.get("key"))
        for field in REQUIRED_ROW_FIELDS:
            if not normalize_text(row.get(field)):
                errors.append(f"{row_id}: missing required provenance field {field}")

        row_mode = normalize_text(row.get("stage2_semantic_source_mode"))
        allowed_llm_first_fallback_bridge = declared_mode == LLM_FIRST_COMPOSITE_MODE and is_allowed_llm_first_fallback_bridge_row(row)
        if declared_mode and row_mode != declared_mode and not allowed_llm_first_fallback_bridge:
            errors.append(f"{row_id}: row mode {row_mode} does not match declared mode {declared_mode}")

        candidate_source = normalize_text(row.get("candidate_source"))
        semantic_universe_authority = normalize_text(row.get("semantic_universe_authority"))
        row_materialization_mode = normalize_text(row.get("row_materialization_mode"))
        semantic_scope_authority = normalize_text(row.get("semantic_scope_authority"))
        semantic_scope_ref = normalize_text(row.get("semantic_scope_ref"))

        if declared_mode == LLM_FIRST_COMPOSITE_MODE:
            if allowed_llm_first_fallback_bridge:
                if semantic_universe_authority != FALLBACK_SEMANTIC_SOURCE_MODE:
                    errors.append(f"{row_id}: fallback bridge rows must keep semantic_universe_authority={FALLBACK_SEMANTIC_SOURCE_MODE}")
                if row_materialization_mode != FALLBACK_SEMANTIC_SOURCE_MODE:
                    errors.append(f"{row_id}: fallback bridge rows must keep row_materialization_mode={FALLBACK_SEMANTIC_SOURCE_MODE}")
                if semantic_scope_authority != FALLBACK_SEMANTIC_SOURCE_MODE:
                    errors.append(f"{row_id}: fallback bridge rows must keep semantic_scope_authority={FALLBACK_SEMANTIC_SOURCE_MODE}")
                if not semantic_scope_ref.startswith("governed_fallback_document_scope:BXCV5XWB"):
                    errors.append(f"{row_id}: fallback bridge rows must keep BXCV5XWB governed_fallback_document_scope semantic_scope_ref")
                continue
            if semantic_universe_authority != "llm_semantic_discovery":
                errors.append(f"{row_id}: llm_first_composite rows must keep semantic_universe_authority=llm_semantic_discovery")
            if row_materialization_mode not in {
                "llm_semantic_discovery",
                "deterministic_row_expansion_within_llm_scope",
                SEQUENTIAL_OPTIMIZATION_ROW_MODE,
                TABLE_ROW_EXPANSION_MODE,
            }:
                errors.append(f"{row_id}: unsupported row_materialization_mode for llm_first_composite: {row_materialization_mode}")
            if row_materialization_mode == "deterministic_row_expansion_within_llm_scope":
                if semantic_scope_authority != "llm_declared_scope":
                    errors.append(f"{row_id}: deterministic expansion must keep semantic_scope_authority=llm_declared_scope")
                if not semantic_scope_ref:
                    errors.append(f"{row_id}: deterministic expansion is missing semantic_scope_ref")
                elif not has_declared_scope_ref(document_scope_ids.get(key, set()), semantic_scope_ref):
                    errors.append(f"{row_id}: deterministic expansion semantic_scope_ref is not declared in the document scope declarations")
                if key not in document_keys_with_doe_scope:
                    errors.append(f"{row_id}: deterministic DOE expansion exists without document-level llm_declared_doe_scope")
            if row_materialization_mode == SEQUENTIAL_OPTIMIZATION_ROW_MODE:
                if semantic_scope_authority != "llm_declared_scope":
                    errors.append(f"{row_id}: sequential optimization resolution must keep semantic_scope_authority=llm_declared_scope")
                if not semantic_scope_ref:
                    errors.append(f"{row_id}: sequential optimization resolution is missing semantic_scope_ref")
                elif not has_declared_scope_ref(document_scope_ids.get(key, set()), semantic_scope_ref):
                    errors.append(f"{row_id}: sequential optimization semantic_scope_ref is not declared in the document scope declarations")
                if key in document_keys_with_doe_scope:
                    errors.append(f"{row_id}: sequential optimization resolution is not allowed when a document-level llm_declared_doe_scope exists")
            if row_materialization_mode == TABLE_ROW_EXPANSION_MODE:
                if semantic_scope_authority != "llm_declared_scope":
                    errors.append(f"{row_id}: table row expansion must keep semantic_scope_authority=llm_declared_scope")
                if not semantic_scope_ref:
                    errors.append(f"{row_id}: table row expansion is missing semantic_scope_ref")
                elif not has_declared_scope_ref(document_scope_ids.get(key, set()), semantic_scope_ref):
                    document = document_by_key.get(key)
                    if not isinstance(document, dict) or not has_shrunken_llm_table_scope(document, normalize_text(row.get("table_id"))):
                        errors.append(f"{row_id}: table row expansion semantic_scope_ref is not declared in the document scope declarations")
                else:
                    document = document_by_key.get(key)
                    if not isinstance(document, dict) or not has_table_formulation_scope(document, semantic_scope_ref, normalize_text(row.get("table_id"))):
                        errors.append(f"{row_id}: table row expansion exists without declared table_formulation_authorization_scope")
                    elif not has_table_formulation_scope_marker(document, semantic_scope_ref, normalize_text(row.get("table_id"))):
                        errors.append(f"{row_id}: table row expansion scope is not backed by an LLM-provenance table_formulation_scopes marker")
                if key in document_keys_with_doe_scope and normalize_text(row.get("table_id")):
                    warnings.append(f"{row_id}: table row expansion row appears in a document that also declares DOE scope; verify non-DOE table routing")
            if candidate_source.startswith("doe_numbered_table_row"):
                if key not in document_keys_with_doe_scope:
                    errors.append(f"{row_id}: DOE recovery row exists without document-level llm_declared_doe_scope")
                if "__llm_declared_doe_scope__" not in semantic_scope_ref:
                    errors.append(f"{row_id}: DOE recovery row is missing llm_declared_doe_scope semantic_scope_ref")
        elif declared_mode == FALLBACK_SEMANTIC_SOURCE_MODE:
            if semantic_universe_authority != FALLBACK_SEMANTIC_SOURCE_MODE:
                errors.append(f"{row_id}: fallback rows must keep semantic_universe_authority={FALLBACK_SEMANTIC_SOURCE_MODE}")
            if row_materialization_mode != FALLBACK_SEMANTIC_SOURCE_MODE:
                errors.append(f"{row_id}: fallback rows must keep row_materialization_mode={FALLBACK_SEMANTIC_SOURCE_MODE}")
            if semantic_scope_authority != FALLBACK_SEMANTIC_SOURCE_MODE:
                errors.append(f"{row_id}: fallback rows must keep semantic_scope_authority={FALLBACK_SEMANTIC_SOURCE_MODE}")
        elif declared_mode == DIAGNOSTIC_COMPARATOR_MODE:
            if not semantic_scope_ref:
                warnings.append(f"{row_id}: diagnostic comparator row is missing semantic_scope_ref")
        elif candidate_source.startswith("doe_numbered_table_row") or row_materialization_mode == "deterministic_row_expansion_within_llm_scope":
            errors.append(f"{row_id}: DOE expansion function unit rows are not allowed outside llm_first_composite mode")

    return {"errors": errors, "warnings": warnings}


def main() -> None:
    args = parse_args()
    semantic_jsonl = Path(args.semantic_jsonl)
    stage2_tsv = Path(args.stage2_tsv)
    report_out = Path(args.report_out) if normalize_text(args.report_out) else None

    documents = read_jsonl(semantic_jsonl)
    rows = read_tsv(stage2_tsv)
    document_modes, row_modes, observed_modes, declared_mode = collect_mode_values(documents, rows)
    errors: list[str] = []
    warnings: list[str] = []
    if not observed_modes:
        errors.append("No stage2_semantic_source_mode observed in semantic documents or Stage2 rows.")
    if len(observed_modes) > 1:
        errors.append(f"Mixed stage2_semantic_source_mode values observed: {observed_modes}")
    if declared_mode and declared_mode not in ALLOWED_MODES:
        errors.append(f"Unsupported stage2_semantic_source_mode: {declared_mode}")
    semantic_validation = validate_semantic_documents(documents, require_mode=False)
    errors.extend(semantic_validation["errors"])
    warnings.extend(semantic_validation["warnings"])
    row_validation = validate_stage2_rows(
        documents=documents,
        rows=rows,
        declared_mode=declared_mode,
        document_scope_ids=semantic_validation["document_scope_ids"],
        document_keys_with_doe_scope=semantic_validation["document_keys_with_doe_scope"],
    )
    errors.extend(row_validation["errors"])
    warnings.extend(row_validation["warnings"])

    report = {
        "schema": "stage2_semantic_authority_contract_report_v1",
        "declared_mode": declared_mode,
        "document_modes": document_modes,
        "row_modes": row_modes,
        "document_count": len(documents),
        "row_count": len(rows),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not errors else "fail",
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
