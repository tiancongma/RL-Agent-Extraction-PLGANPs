#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
TABLE_MARKER_FAMILIES = [
    "table_formulation_scopes",
    "table_variable_roles",
    "selection_markers",
    "inheritance_markers",
    "preparation_inheritance_markers",
    "boundary_markers",
]
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


def has_table_formulation_scope_marker(document: dict[str, Any], scope_ref: str) -> bool:
    base_ref = normalize_text(scope_ref.split("|", 1)[0])
    for marker in document.get("table_formulation_scopes", []) or []:
        if not isinstance(marker, dict):
            continue
        if not has_valid_marker_provenance(marker):
            continue
        if base_ref == normalize_text(marker.get("scope_id")):
            return True
    return False


def has_table_formulation_scope(document: dict[str, Any], scope_ref: str) -> bool:
    base_ref = normalize_text(scope_ref.split("|", 1)[0])
    for declaration in document.get("semantic_scope_declarations", []) or []:
        if not isinstance(declaration, dict):
            continue
        if normalize_text(declaration.get("scope_kind")) != "table_formulation_authorization_scope":
            continue
        if base_ref == normalize_text(declaration.get("scope_id")):
            return True
    return False


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

    document_keys_with_doe_scope = {
        normalize_text(document.get("document_key") or document.get("key"))
        for document in documents
        if has_llm_declared_doe_scope(document)
    }
    document_scope_ids: dict[str, set[str]] = {}
    for document in documents:
        key = normalize_text(document.get("document_key") or document.get("key"))
        for family in TABLE_MARKER_FAMILIES:
            for marker in document.get(family, []) or []:
                if not isinstance(marker, dict):
                    errors.append(f"{key}: {family} contains a non-dict marker payload")
                    continue
                if not has_valid_marker_provenance(marker):
                    errors.append(f"{key}: {family} contains a marker without llm_explicit/llm_parsed provenance")
                    continue
                if family == "selection_markers":
                    validate_selection_marker(key, marker, errors)
                elif family == "inheritance_markers":
                    validate_inheritance_marker(key, marker, errors)
        for declaration in document.get("semantic_scope_declarations", []) or []:
            if not isinstance(declaration, dict):
                errors.append(f"{key}: semantic_scope_declarations contains a non-dict declaration")
                continue
            declared_by = normalize_text(declaration.get("declared_by"))
            if declared_by not in VALID_DECLARED_BY:
                errors.append(f"{key}: semantic scope declaration has invalid declared_by={declared_by or '<blank>'}")
            if normalize_text(declaration.get("scope_kind")) == "table_formulation_authorization_scope":
                scope_id = normalize_text(declaration.get("scope_id"))
                if not has_table_formulation_scope_marker(document, scope_id):
                    errors.append(f"{key}: table formulation scope declaration lacks an LLM-provenance table_formulation_scopes marker")
        document_scope_ids[key] = declared_scope_ids(document)

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
        if declared_mode and row_mode != declared_mode:
            errors.append(f"{row_id}: row mode {row_mode} does not match declared mode {declared_mode}")

        candidate_source = normalize_text(row.get("candidate_source"))
        semantic_universe_authority = normalize_text(row.get("semantic_universe_authority"))
        row_materialization_mode = normalize_text(row.get("row_materialization_mode"))
        semantic_scope_authority = normalize_text(row.get("semantic_scope_authority"))
        semantic_scope_ref = normalize_text(row.get("semantic_scope_ref"))

        if declared_mode == LLM_FIRST_COMPOSITE_MODE:
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
                    errors.append(f"{row_id}: table row expansion semantic_scope_ref is not declared in the document scope declarations")
                else:
                    document = document_by_key.get(key)
                    if not isinstance(document, dict) or not has_table_formulation_scope(document, semantic_scope_ref):
                        errors.append(f"{row_id}: table row expansion exists without declared table_formulation_authorization_scope")
                    elif not has_table_formulation_scope_marker(document, semantic_scope_ref):
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
