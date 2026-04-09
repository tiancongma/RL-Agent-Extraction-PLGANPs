#!/usr/bin/env python3
from __future__ import annotations

"""
Governed sequential optimization function unit for Stage2 llm_first_composite mode.

This unit resolves one final formulation row only when the paper explicitly
describes a stagewise optimization chain with selected values and no governed
DOE scope. It does not enumerate a formulation universe and it does not perform
cross-table Cartesian reconstruction.
"""

import json
import re
from pathlib import Path
from typing import Any

from src.stage2_sampling_labels.auto_extract_weak_labels_v7pilot_r3_fixparse import CORE_FIELDS


REPO_ROOT = Path(__file__).resolve().parents[3]

LLM_FIRST_COMPOSITE_MODE = "llm_first_composite"
LLM_SEMANTIC_DISCOVERY = "llm_semantic_discovery"
LLM_DECLARED_SCOPE = "llm_declared_scope"
DOCUMENT_SCOPE_KIND = "llm_document_scope"
DOE_SCOPE_KIND = "doe_table_row_enumeration_scope"
FUNCTION_UNIT_ID = "sequential_optimization_interpreter_v1"
ROW_MATERIALIZATION_MODE = "sequential_optimization_resolved"
RECOVERY_CANDIDATE_SOURCE = "sequential_optimization_function_unit"


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def semantic_scope_declarations(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in ensure_list(document.get("semantic_scope_declarations")) if isinstance(item, dict)]


def stage2_semantic_source_mode(document: dict[str, Any]) -> str:
    return normalize_text(document.get("stage2_semantic_source_mode")) or LLM_FIRST_COMPOSITE_MODE


def resolve_llm_document_scope(document: dict[str, Any]) -> dict[str, Any] | None:
    for declaration in semantic_scope_declarations(document):
        if normalize_text(declaration.get("scope_kind")) != DOCUMENT_SCOPE_KIND:
            continue
        if normalize_text(declaration.get("declared_by")) != LLM_SEMANTIC_DISCOVERY:
            continue
        return declaration
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    if not document_key:
        return None
    return {
        "scope_id": f"{document_key}__llm_document_scope__01",
        "scope_kind": DOCUMENT_SCOPE_KIND,
        "declared_by": LLM_SEMANTIC_DISCOVERY,
    }


def has_llm_declared_doe_scope(document: dict[str, Any]) -> bool:
    for declaration in semantic_scope_declarations(document):
        if normalize_text(declaration.get("scope_kind")) != DOE_SCOPE_KIND:
            continue
        if normalize_text(declaration.get("declared_by")) != LLM_SEMANTIC_DISCOVERY:
            continue
        return True
    return False


def resolve_document_text_path(document: dict[str, Any]) -> Path | None:
    text_path = normalize_text(document.get("source_text_path"))
    if not text_path:
        return None
    path = Path(text_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def normalize_variable_name(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def first_number(text: str) -> str:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return match.group(0) if match else ""


def collapse_snippet(raw_text: str, start: int, end: int, radius: int = 120) -> str:
    left = max(0, start - radius)
    right = min(len(raw_text), end + radius)
    return re.sub(r"\s+", " ", raw_text[left:right]).strip()


def compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)


def extract_stagewise_selected_values(raw_text: str) -> list[dict[str, str]]:
    patterns = [
        {
            "stage_id": "stage_1_surfactant",
            "variable_name": "poloxamer 188 concentration",
            "pattern": r"(?P<value>\d+(?:\.\d+)?)\s*mg/mL\s+was\s+(?:then\s+)?selected\s+as\s+optimal\s+surfactant\s+concentration",
            "text_pattern": r"(?P<value>\d+(?:\.\d+)?)\s*mg/mL\s+of\s+poloxamer\s+188\s+was\s+chosen",
        },
        {
            "stage_id": "stage_2_ratio",
            "variable_name": "PLGA:ITZ (w/w) ratio",
            "pattern": r"ratio\s+of\s+(?P<value>\d+:\d+)\s+was\s+(?:then\s+)?selected",
            "text_pattern": r"PLGA:ITZ\s+ratio\s+of\s+(?P<value>\d+:\d+)\s+was\s+selected",
        },
        {
            "stage_id": "stage_3_lyoprotectant_concentration",
            "variable_name": "lyoprotectant concentration",
            "pattern": r"(?P<value>\d+(?:\.\d+)?%\s*(?:w/v)?)\s+(?P<material>sucrose|mannitol|dextrose|HP-β-CD|HP-\w+-CD)\s+was\s+(?:then\s+)?selected\s+as\s+optimal\s+lyoprotectant",
            "text_pattern": r"(?P<value>\d+(?:\.\d+)?%\s*(?:w/v)?)\s+(?P<material>sucrose|mannitol|dextrose|HP-β-CD|HP-\w+-CD)\s+was\s+selected\s+as\s+a\s+lyoprotectant",
        },
    ]
    selected: list[dict[str, str]] = []
    for entry in patterns:
        match = compile_pattern(entry["pattern"]).search(raw_text) or compile_pattern(entry["text_pattern"]).search(raw_text)
        if not match:
            continue
        value = normalize_text(match.groupdict().get("value"))
        if entry["variable_name"] == "poloxamer 188 concentration" and value and "mg/ml" not in value.lower():
            value = f"{value} mg/mL"
        material = normalize_text(match.groupdict().get("material"))
        evidence = collapse_snippet(raw_text, match.start(), match.end())
        selected.append(
            {
                "stage_id": entry["stage_id"],
                "variable_name": entry["variable_name"],
                "value_text": value,
                "material_name": material,
                "evidence_text": evidence,
                "evidence_region_type": "results_sentence" if "selected" in evidence.lower() or "chosen" in evidence.lower() else "methods_sentence",
            }
        )
    if selected:
        return selected

    generic_matches = [
        (
            "stage_1_surfactant",
            "poloxamer 188 concentration",
            r"poloxamer\s+188\s+was\s+selected",
        ),
        (
            "stage_2_ratio",
            "PLGA:ITZ (w/w) ratio",
            r"PLGA:ITZ\s+ratio\s+of\s+(?P<value>\d+:\d+)",
        ),
    ]
    fallback_selected: list[dict[str, str]] = []
    for stage_id, variable_name, pattern in generic_matches:
        match = compile_pattern(pattern).search(raw_text)
        if not match:
            continue
        fallback_selected.append(
            {
                "stage_id": stage_id,
                "variable_name": variable_name,
                "value_text": normalize_text(match.groupdict().get("value")),
                "material_name": "",
                "evidence_text": collapse_snippet(raw_text, match.start(), match.end()),
                "evidence_region_type": "results_sentence",
            }
        )
    return fallback_selected


def build_identity_variables_json(selected_values: list[dict[str, str]]) -> str:
    payload: list[dict[str, str]] = []
    for item in selected_values:
        variable_name = normalize_text(item.get("variable_name"))
        value_text = normalize_text(item.get("value_text"))
        material_name = normalize_text(item.get("material_name"))
        if variable_name == "lyoprotectant concentration" and value_text:
            payload.append(
                {
                    "name": normalize_variable_name(variable_name),
                    "name_raw": variable_name,
                    "value": value_text.lower(),
                    "value_raw": value_text,
                }
            )
        if variable_name == "poloxamer 188 concentration" and value_text:
            payload.append(
                {
                    "name": normalize_variable_name(variable_name),
                    "name_raw": variable_name,
                    "value": value_text.lower(),
                    "value_raw": value_text,
                }
            )
        if variable_name == "PLGA:ITZ (w/w) ratio" and value_text:
            payload.append(
                {
                    "name": normalize_variable_name(variable_name),
                    "name_raw": variable_name,
                    "value": value_text.lower(),
                    "value_raw": value_text,
                }
            )
        if material_name:
            payload.append(
                {
                    "name": "lyoprotectant_type",
                    "name_raw": "lyoprotectant type",
                    "value": material_name.lower(),
                    "value_raw": material_name,
                }
            )
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in payload:
        key = (item["name"], item["name_raw"], item["value"], item["value_raw"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return json.dumps(deduped, ensure_ascii=False)


def selected_value_map(selected_values: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {normalize_text(item.get("variable_name")): item for item in selected_values}


def copy_field_bundle(target: dict[str, Any], source: dict[str, Any], field_name: str) -> None:
    for suffix in ["value", "value_text", "scope", "membership_confidence", "evidence_region_type", "missing_reason"]:
        key = f"{field_name}_{suffix}"
        target[key] = source.get(key, "")


def merge_supporting_refs(*values: str) -> str:
    merged: list[dict[str, Any]] = []
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            merged.extend(item for item in parsed if isinstance(item, dict))
    return json.dumps(merged, ensure_ascii=False)


def build_selection_supporting_refs(selected_values: list[dict[str, str]]) -> str:
    refs = [
        {
            "source_region_type": normalize_text(item.get("evidence_region_type")),
            "source_locator_text": normalize_text(item.get("evidence_text")),
            "supporting_snippet": normalize_text(item.get("evidence_text")),
            "target_field_name": normalize_variable_name(item.get("variable_name")),
        }
        for item in selected_values
        if normalize_text(item.get("evidence_text"))
    ]
    return json.dumps(refs, ensure_ascii=False)


def prefer_resolved_sequential_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved_ids = {
        normalize_text(row.get("formulation_id"))
        for row in rows
        if normalize_text(row.get("candidate_source")) == RECOVERY_CANDIDATE_SOURCE
    }
    if not resolved_ids:
        return rows, []
    preferred: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for row in rows:
        formulation_id = normalize_text(row.get("formulation_id"))
        if (
            formulation_id in resolved_ids
            and normalize_text(row.get("candidate_source")) != RECOVERY_CANDIDATE_SOURCE
        ):
            suppressed.append(row)
            continue
        preferred.append(row)
    return preferred, suppressed


def run_sequential_optimization_interpreter(
    *,
    document: dict[str, Any],
    existing_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    source_mode = stage2_semantic_source_mode(document)
    if source_mode != LLM_FIRST_COMPOSITE_MODE:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": f"semantic_source_mode_not_llm_first:{source_mode}",
        }
    if has_llm_declared_doe_scope(document):
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": "blocked_by_doe_scope",
        }

    document_scope = resolve_llm_document_scope(document)
    if document_scope is None:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": "missing_llm_document_scope",
        }

    text_path = resolve_document_text_path(document)
    if text_path is None or not text_path.exists():
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": "missing_source_text_path",
        }

    raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
    lower_text = raw_text.lower()
    if not any(token in lower_text for token in ["kept constant", "selected", "chosen", "optimal", "optimum"]):
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": "missing_optimization_language",
        }

    selected_values = extract_stagewise_selected_values(raw_text)
    if len(selected_values) < 3:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": "missing_stagewise_selected_values",
        }

    if "kept constant" not in lower_text and "after the optimal surfactant concentration had been determined" not in lower_text:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": "ambiguous_stage_dependency",
        }

    formulation_candidates = [item for item in ensure_list(document.get("formulation_identity_candidates")) if isinstance(item, dict)]
    optimized_candidate = next(
        (
            item
            for item in formulation_candidates
            if normalize_text(item.get("formulation_role")) == "optimized"
            or "optimal" in normalize_text(item.get("raw_formulation_label")).lower()
            or "optimum" in normalize_text(item.get("raw_formulation_label")).lower()
        ),
        None,
    )
    if optimized_candidate is None:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": "missing_final_selection_candidate",
        }

    optimized_id = normalize_text(optimized_candidate.get("formulation_candidate_id"))
    parent_id = normalize_text(optimized_candidate.get("parent_candidate_id"))
    optimized_row = next((row for row in existing_rows if normalize_text(row.get("formulation_id")) == optimized_id), None)
    parent_row = next((row for row in existing_rows if normalize_text(row.get("formulation_id")) == parent_id), None)
    if optimized_row is None:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "triggered": False,
            "materialized": False,
            "candidate_count": 0,
            "notes": "missing_optimized_row",
        }

    resolved_row = dict(optimized_row)
    if parent_row is not None:
        for field_name in CORE_FIELDS:
            if not normalize_text(resolved_row.get(f"{field_name}_value_text")) and normalize_text(parent_row.get(f"{field_name}_value_text")):
                copy_field_bundle(resolved_row, parent_row, field_name)
        if not normalize_text(resolved_row.get("polymer_identity")):
            resolved_row["polymer_identity"] = normalize_text(parent_row.get("polymer_identity"))
        if not normalize_text(resolved_row.get("polymer_name_raw")):
            resolved_row["polymer_name_raw"] = normalize_text(parent_row.get("polymer_name_raw"))

    selected_map = selected_value_map(selected_values)
    surfactant_value = normalize_text(selected_map.get("poloxamer 188 concentration", {}).get("value_text"))
    if surfactant_value:
        resolved_row["surfactant_concentration_text_value"] = first_number(surfactant_value) or surfactant_value
        resolved_row["surfactant_concentration_text_value_text"] = surfactant_value
        resolved_row["surfactant_concentration_text_scope"] = "instance_specific"
        resolved_row["surfactant_concentration_text_membership_confidence"] = "projected_direct"
        resolved_row["surfactant_concentration_text_evidence_region_type"] = normalize_text(
            selected_map["poloxamer 188 concentration"].get("evidence_region_type")
        )
        resolved_row["surfactant_concentration_text_missing_reason"] = ""

    resolved_row["candidate_source"] = RECOVERY_CANDIDATE_SOURCE
    resolved_row["stage2_semantic_source_mode"] = LLM_FIRST_COMPOSITE_MODE
    resolved_row["semantic_universe_authority"] = LLM_SEMANTIC_DISCOVERY
    resolved_row["row_materialization_mode"] = ROW_MATERIALIZATION_MODE
    resolved_row["semantic_scope_authority"] = LLM_DECLARED_SCOPE
    resolved_row["semantic_scope_ref"] = normalize_text(document_scope.get("scope_id"))
    resolved_row["instance_kind_reconciliation_note"] = FUNCTION_UNIT_ID
    resolved_row["change_role"] = "synthesis_defining"
    resolved_row["instance_confidence"] = "high"
    resolved_row["formulation_role"] = "optimized"
    resolved_row["identity_variables_json"] = build_identity_variables_json(selected_values)
    resolved_row["change_descriptions"] = json.dumps(
        [
            f"{normalize_text(item.get('variable_name'))} -> {normalize_text(item.get('value_text') or item.get('material_name'))}"
            for item in selected_values
        ],
        ensure_ascii=False,
    )
    resolved_row["instance_context_tags"] = json.dumps(["sequential_optimization_resolved"], ensure_ascii=False)
    resolved_row["change_context_tags"] = json.dumps(["stagewise_selection"], ensure_ascii=False)
    resolved_row["supporting_evidence_refs"] = merge_supporting_refs(
        normalize_text(parent_row.get("supporting_evidence_refs")) if parent_row is not None else "",
        normalize_text(optimized_row.get("supporting_evidence_refs")),
        build_selection_supporting_refs(selected_values),
    )
    final_evidence = selected_values[-1]
    resolved_row["instance_evidence_region_type"] = normalize_text(final_evidence.get("evidence_region_type"))
    resolved_row["evidence_section"] = normalize_text(final_evidence.get("evidence_text"))
    resolved_row["evidence_span_text"] = normalize_text(final_evidence.get("evidence_text"))
    resolved_row["evidence_span_start"] = ""
    resolved_row["evidence_span_end"] = ""

    stage_chain = [
        {
            "stage_id": normalize_text(item.get("stage_id")),
            "variable_name": normalize_text(item.get("variable_name")),
            "value_text": normalize_text(item.get("value_text")),
            "material_name": normalize_text(item.get("material_name")),
            "evidence_text": normalize_text(item.get("evidence_text")),
        }
        for item in selected_values
    ]
    trace_note = {
        "parent_formulation_id": parent_id,
        "optimized_formulation_id": optimized_id,
        "selected_variable_values": stage_chain,
        "guardrails": {
            "cross_table_cartesian_merge": "no",
            "doe_scope_present": "no",
            "materialization_basis": "explicit_stagewise_selection_text_plus_parent_inheritance",
        },
    }
    traces = [
        {
            "document_key": normalize_text(document.get("document_key") or document.get("key")),
            "formulation_id": optimized_id,
            "legacy_field": "sequential_optimization_resolution",
            "source_replacement_objects": json.dumps(trace_note, ensure_ascii=False),
            "mapping_status": "direct",
            "direct_or_derived": "direct",
            "notes": "Resolved one optimized formulation from explicit stagewise selection statements without cross-table enumeration.",
        }
    ]
    jsonl_rows = [
        {
            "key": normalize_text(document.get("document_key") or document.get("key")),
            "doi": normalize_text(document.get("doi")),
            "formulation_id": normalize_text(resolved_row.get("formulation_id")),
            "legacy_row": resolved_row,
        }
    ]
    return [resolved_row], traces, jsonl_rows, {
        "function_unit": FUNCTION_UNIT_ID,
        "triggered": True,
        "materialized": True,
        "candidate_count": 1,
        "semantic_scope_ref": normalize_text(document_scope.get("scope_id")),
        "notes": "resolved_final_formulation",
    }


__all__ = [
    "FUNCTION_UNIT_ID",
    "RECOVERY_CANDIDATE_SOURCE",
    "ROW_MATERIALIZATION_MODE",
    "prefer_resolved_sequential_rows",
    "run_sequential_optimization_interpreter",
]
