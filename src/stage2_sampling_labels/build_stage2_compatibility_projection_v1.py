#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.preparation_method_fields_v1 import PREPARATION_METHOD_FIELDNAMES

try:
    from src.stage2_sampling_labels.auto_extract_weak_labels_v7pilot_r3_fixparse import (
        CORE_FIELDS,
        build_output_columns,
    )
except ModuleNotFoundError as exc:
    if exc.name != "pandas":
        raise
    CORE_FIELDS = [
        "emul_type",
        "emul_method",
        "la_ga_ratio",
        "polymer_mw_kDa",
        "plga_mass_mg",
        "surfactant_name",
        "surfactant_concentration_text",
        "pva_conc_percent",
        "organic_solvent",
        "drug_name",
        "drug_feed_amount_text",
        "size_nm",
        "pdi",
        "zeta_mV",
        "encapsulation_efficiency_percent",
        "loading_content_percent",
    ]

    def build_output_columns() -> list[str]:
        cols = [
            "key",
            "doi",
            "model",
            "local_instance_id",
            "formulation_id",
            "raw_formulation_label",
            "polymer_identity",
            "polymer_name_raw",
            "instance_kind",
            "parent_instance_id",
            "change_descriptions",
            "change_role",
            "instance_context_tags",
            "change_context_tags",
            "supporting_evidence_refs",
            "formulation_role",
            "instance_confidence",
            "candidate_source",
            "stage2_semantic_source_mode",
            "semantic_universe_authority",
            "row_materialization_mode",
            "semantic_scope_authority",
            "semantic_scope_ref",
            "instance_evidence_region_type",
            "evidence_section",
            "evidence_span_text",
            "evidence_span_start",
            "evidence_span_end",
            "instance_kind_raw",
            "instance_kind_inferred",
            "instance_kind_reconciliation_note",
        ]
        for field_name in CORE_FIELDS:
            cols.extend(
                [
                    f"{field_name}_value",
                    f"{field_name}_value_text",
                    f"{field_name}_scope",
                    f"{field_name}_membership_confidence",
                    f"{field_name}_evidence_region_type",
                    f"{field_name}_missing_reason",
                ]
            )
        cols.extend(PREPARATION_METHOD_FIELDNAMES)
        return cols
try:
    from src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1 import (
        FUNCTION_UNIT_ID,
        build_governed_numbered_doe_guard_row,
        doe_enumeration_mode,
        is_governed_doe_recovery_candidate_source,
        numbered_doe_recovery_enabled,
        numbered_doe_recovery_min_rows,
        prefer_governed_doe_rows_over_llm_numeric_rows,
        resolve_llm_declared_doe_scope,
        run_doe_row_expansion_function_unit,
        write_numbered_doe_guard_artifact,
    )
except ModuleNotFoundError as exc:
    if exc.name != "pandas":
        raise
    FUNCTION_UNIT_ID = "doe_row_expansion_function_unit_v1"

    def build_governed_numbered_doe_guard_row(*, document_key: str, **_: Any) -> dict[str, Any]:
        return {
            "document_key": document_key,
            "status": "skipped_missing_optional_dependency",
            "skip_reason": "pandas_not_available",
        }

    def doe_enumeration_mode() -> str:
        return "disabled_missing_optional_dependency"

    def is_governed_doe_recovery_candidate_source(candidate_source: str) -> bool:
        return False

    def numbered_doe_recovery_enabled() -> bool:
        return False

    def numbered_doe_recovery_min_rows() -> int:
        return 0

    def prefer_governed_doe_rows_over_llm_numeric_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        return rows, []

    def resolve_llm_declared_doe_scope(document: dict[str, Any]) -> dict[str, Any]:
        return {}

    def run_doe_row_expansion_function_unit(*, document: dict[str, Any], model_name: str, semantic_scope: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
        return [], [], [], {
            "function_unit": FUNCTION_UNIT_ID,
            "document_key": normalize_text(document.get("document_key")),
            "considered": True,
            "authorized": False,
            "called": False,
            "candidate_count": 0,
            "retained_row_count": 0,
            "skip_reason": "pandas_not_available",
            "status": "skipped_missing_optional_dependency",
        }

    def write_numbered_doe_guard_artifact(output_dir: Path, guard_rows: list[dict[str, Any]]) -> dict[str, Any]:
        guard_path = output_dir / "numbered_doe_guard_v1.tsv"
        write_tsv(guard_path, guard_rows, sorted({key for row in guard_rows for key in row.keys()}) or ["document_key", "status", "skip_reason"])
        return {"guard_path": str(guard_path.resolve()), "fail_count": 0, "warn_count": 0}

try:
    from src.stage2_sampling_labels.function_units.sequential_optimization_interpreter_v1 import (
        FUNCTION_UNIT_ID as SEQUENTIAL_OPTIMIZATION_FUNCTION_UNIT_ID,
        RECOVERY_CANDIDATE_SOURCE as SEQUENTIAL_OPTIMIZATION_CANDIDATE_SOURCE,
        prefer_resolved_sequential_rows,
        run_sequential_optimization_interpreter,
    )
except ModuleNotFoundError as exc:
    if exc.name != "pandas":
        raise
    SEQUENTIAL_OPTIMIZATION_FUNCTION_UNIT_ID = "sequential_optimization_interpreter_v1"
    SEQUENTIAL_OPTIMIZATION_CANDIDATE_SOURCE = "sequential_optimization_interpreter_disabled_missing_optional_dependency"

    def prefer_resolved_sequential_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        return rows, []

    def run_sequential_optimization_interpreter(*, document: dict[str, Any], existing_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
        return [], [], [], {
            "function_unit": SEQUENTIAL_OPTIMIZATION_FUNCTION_UNIT_ID,
            "document_key": normalize_text(document.get("document_key")),
            "considered": True,
            "authorized": False,
            "called": False,
            "emitted_row_count": 0,
            "retained_row_count": 0,
            "skip_reason": "pandas_not_available",
            "status": "skipped_missing_optional_dependency",
            "replaced_row_count": 0,
        }
from src.stage2_sampling_labels.table_row_expansion_v1 import (
    BOUNDARY_MARKER_FIELD,
    execution_ready_markers,
    INHERITANCE_MARKER_FIELD,
    METHOD_GROUP_SIGNATURE_HINT_FIELD,
    PREPARATION_INHERITANCE_FIELD,
    SELECTION_MARKER_FIELD,
    TABLE_ASSIGNMENTS_FIELD,
    TABLE_ID_FIELD,
    TABLE_ROW_ID_FIELD,
    TABLE_SCOPE_FIELD,
    TABLE_VARIABLE_ROLE_FIELD,
    augment_document_with_table_markers,
    mark_llm_summary_rows_as_helpers,
    run_table_row_expansion,
)


LEGACY_TSV_NAME = "weak_labels__v7pilot_r3_fixparse.tsv"
LEGACY_JSONL_NAME = "weak_labels__v7pilot_r3_fixparse.jsonl"
TRACE_TSV_NAME = "compatibility_projection_trace_v1.tsv"
SUMMARY_JSON_NAME = "compatibility_projection_summary_v1.json"
CONTRACT_TSV_NAME = "stage2_replacement_compatibility_projection_contract.tsv"
FUNCTION_UNIT_ACTIVATION_NAME = "feature_activation_report_v2.tsv"
EXECUTION_LEDGER_NAME = "execution_ledger_v2.tsv"
AUTHORITY_REATTACHMENT_SIDECAR_NAME = "authority_reattachment_sidecar_v1.json"
IDENTITY_VARIABLES_FIELD = "identity_variables_json"
LLM_FIRST_COMPOSITE_MODE = "llm_first_composite"
FALLBACK_SEMANTIC_SOURCE_MODE = "governed_fallback_semantic_source"
DIAGNOSTIC_COMPARATOR_MODE = "diagnostic_comparator"

# Stage2 internal handoff contract:
# - The governed composite Stage2 runner writes semantic intermediates from
#   `extract_semantic_stage2_objects_v2.py`.
# - Those canonical semantic-intermediate payloads use object-family names such
#   as `formulation_candidates`, `variable_candidates`, `relation_hints`, and
#   `evidence_spans`.
# - Older replacement-path and deterministic comparator surfaces may still use
#   the earlier semantic-object names such as
#   `formulation_identity_candidates`, `variable_or_factor_candidates`,
#   `relation_cues`, and `evidence_handoffs`.
# - This completion script accepts both explicitly, normalizes them into one
#   compatibility-projection view, and then emits the only authoritative
#   completed Stage2 artifacts consumed by Stage3.

DIRECT = "direct"
DERIVED = "derived"
COMPRESSED = "compressed"
UNAVAILABLE = "unavailable"

LEGACY_OBJECT_KEYS = {
    "formulation_identity_candidate": "formulation_identity_candidates",
    "component_candidate": "component_candidates",
    "phase_candidate": "phase_candidates",
    "process_step_candidate": "process_step_candidates",
    "variable_or_factor_candidate": "variable_or_factor_candidates",
    "measurement_candidate": "measurement_candidates",
    "relation_cue": "relation_cues",
    "evidence_handoff": "evidence_handoffs",
}

CANONICAL_STAGE2_V2_KEYS = {
    "formulation_identity_candidate": "formulation_candidates",
    "component_candidate": "component_candidates",
    "phase_candidate": "phase_candidates",
    "process_step_candidate": "process_step_candidates",
    "variable_or_factor_candidate": "variable_candidates",
    "measurement_candidate": "measurement_candidates",
    "relation_cue": "relation_hints",
    "evidence_handoff": "evidence_spans",
}

MEASUREMENT_ALIASES = {
    "size_nm": ["size", "particle size", "size_nm", "mean particle size"],
    "pdi": ["pdi", "polydispersity index", "polydispersity"],
    "zeta_mV": ["zeta", "zeta potential", "zeta_mv"],
    "encapsulation_efficiency_percent": ["encapsulation efficiency", "ee", "entrapment efficiency"],
    "loading_content_percent": ["loading content", "drug loading", "dl", "loading efficiency"],
}

ROLE_ALIASES = {
    "polymer": {"polymer", "copolymer", "matrix polymer"},
    "surfactant": {"surfactant", "stabilizer", "emulsifier"},
    "organic_solvent": {"organic_solvent", "solvent", "organic solvent", "co-solvent", "cosolvent"},
    "drug": {"drug", "active", "api", "payload"},
}

SHARED_SCOPE_FIELDS = {
    "emul_type",
    "emul_method",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "surfactant_name",
    "organic_solvent",
    "drug_name",
    "preparation_method",
    "emulsion_structure",
}


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_text(value).lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return default


def normalize_token(value: Any) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = normalize_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_string_list(value: Any) -> list[str]:
    parsed = parse_json_maybe(value)
    if isinstance(parsed, list):
        return [normalize_text(item) for item in parsed if normalize_text(item)]
    text = normalize_text(parsed)
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def stringify_json(value: Any) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def compatibility_output_columns() -> list[str]:
    columns = list(build_output_columns())
    if IDENTITY_VARIABLES_FIELD not in columns:
        columns.append(IDENTITY_VARIABLES_FIELD)
    for field in [
        METHOD_GROUP_SIGNATURE_HINT_FIELD,
        TABLE_ID_FIELD,
        TABLE_ROW_ID_FIELD,
        TABLE_ASSIGNMENTS_FIELD,
        TABLE_SCOPE_FIELD,
        TABLE_VARIABLE_ROLE_FIELD,
        SELECTION_MARKER_FIELD,
        INHERITANCE_MARKER_FIELD,
        BOUNDARY_MARKER_FIELD,
        PREPARATION_INHERITANCE_FIELD,
    ]:
        if field not in columns:
            columns.append(field)
    return columns


def stage2_semantic_source_mode(document: dict[str, Any]) -> str:
    return normalize_text(document.get("stage2_semantic_source_mode")) or LLM_FIRST_COMPOSITE_MODE


def default_semantic_scope_ref(identity: dict[str, Any], document: dict[str, Any]) -> str:
    if stage2_semantic_source_mode(document) == FALLBACK_SEMANTIC_SOURCE_MODE:
        return f"governed_fallback_document_scope:{normalize_text(document.get('document_key') or document.get('key'))}"
    formulation_id = normalize_text(identity.get("formulation_candidate_id") or identity.get("candidate_id"))
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    return f"{document_key}__llm_document_scope__01|candidate:{formulation_id}" if formulation_id else f"{document_key}__llm_document_scope__01"


def choose_first(items: list[dict[str, Any]], *keys: str) -> str:
    for item in items:
        for key in keys:
            text = normalize_text(item.get(key))
            if text:
                return text
    return ""


def first_number(text: str) -> str:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return match.group(0) if match else ""


def normalize_variable_name(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_variable_value(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).lower()).strip()


def build_identity_variables_payload(factors: list[dict[str, Any]]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for factor in factors:
        if normalize_text(factor.get("identity_defining_signal")).lower() != "yes":
            continue
        name_raw = normalize_text(factor.get("factor_name_raw"))
        value_raw = normalize_text(factor.get("factor_expression_raw"))
        name = normalize_variable_name(name_raw)
        value = normalize_variable_value(value_raw)
        if not name or not value:
            continue
        key = (name, value)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "name": name,
                "value": value,
                "name_raw": name_raw,
                "value_raw": value_raw,
            }
        )
    return sorted(items, key=lambda item: (item["name"], item["value"], item["name_raw"], item["value_raw"]))


def infer_polymer_identity(name: str) -> str:
    token = normalize_token(name)
    if "plga" in token or "lactic glycolic" in token:
        return "PLGA"
    if "pcl" in token or "polycaprolactone" in token:
        return "PCL"
    if "pla" in token or "polylactic" in token:
        return "PLA"
    if "peg" in token and "plga" in token:
        return "PEG-PLGA"
    return name.strip()


def value_or_first_number(value: Any) -> str:
    text = normalize_text(value)
    return first_number(text) or text


def build_target_object_ref(prefix: str, item_id: str, formulation_id: str) -> str:
    if item_id:
        return f"{prefix}:{item_id}"
    if formulation_id:
        return f"{prefix}:{formulation_id}"
    return prefix


def is_shrunken_stage2_document(document: dict[str, Any]) -> bool:
    return all(key in document for key in ["table_scopes", "semantic_signals", "formulation_candidates"])


def scope_kind_to_table_type(scope_kind: str) -> str:
    normalized = normalize_text(scope_kind)
    mapping = {
        "doe_table": "doe_table",
        "formulation_table": "full_formulation",
        "optimization_table": "partial_formulation",
        "sequential_child": "sequential_child",
        "downstream_variant_table": "sequential_child",
        "non_formulation": "non_formulation",
        "unclear": "partial_formulation",
    }
    return mapping.get(normalized, "partial_formulation")


def scope_kind_is_formulation_bearing(scope_kind: str, explicit_flag: Any) -> bool:
    if explicit_flag is not None:
        return bool(explicit_flag)
    return normalize_text(scope_kind) not in {"non_formulation", "unclear"}


def role_to_legacy_formulation_role(instance_role: str) -> str:
    normalized = normalize_text(instance_role)
    mapping = {
        "downstream_variant": "variant",
        "control": "control",
        "comparative": "comparative",
        "characterization_only": "characterization_only",
        "synthesis_core": "unclear",
        "unclear": "unclear",
    }
    return mapping.get(normalized, "unclear")


def kind_to_change_role(candidate_kind: str, instance_role: str) -> str:
    if normalize_text(instance_role) == "downstream_variant" or normalize_text(candidate_kind) == "variant_formulation":
        return "non_synthesis"
    return "unclear"


def normalize_stage2_document_for_projection(document: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize the Stage2 semantic-intermediate payload into the legacy-shaped
    semantic object families consumed by the deterministic completion step.

    Canonical governed Stage2 v2 payloads come from
    `extract_semantic_stage2_objects_v2.py` and use names like
    `formulation_candidates` and `variable_candidates`.
    Older replacement-path artifacts may already use the legacy-shaped semantic
    families such as `formulation_identity_candidates`.
    """

    legacy_identity_key = LEGACY_OBJECT_KEYS["formulation_identity_candidate"]
    if is_shrunken_stage2_document(document):
        document_key = normalize_text(document.get("document_key") or document.get("paper_key") or document.get("key"))
        model_name = normalize_text(document.get("model_name") or document.get("source_mode")) or "stage2_v2_semantic_objects"
        source_mode = normalize_text(document.get("source_mode"))
        semantic_signals = document.get("semantic_signals") if isinstance(document.get("semantic_signals"), dict) else {}
        primary_variable_names = [
            normalize_text(item)
            for item in ensure_list(semantic_signals.get("primary_variable_names"))
            if normalize_text(item)
        ]
        selected_condition_hints = [
            normalize_text(item)
            for item in ensure_list(semantic_signals.get("selected_condition_hints"))
            if normalize_text(item)
        ]
        normalized: dict[str, Any] = {
            "document_key": document_key,
            "doi": normalize_text(document.get("doi")),
            "model_name": model_name,
            "source_mode": source_mode,
            "stage2_semantic_source_mode": normalize_text(document.get("stage2_semantic_source_mode")),
            "semantic_universe_authority": normalize_text(document.get("semantic_universe_authority")),
            "semantic_scope_declarations": ensure_list(document.get("semantic_scope_declarations")),
            "semantic_signals": semantic_signals,
            "authority_run_dir": normalize_text(document.get("authority_run_dir")),
            "authority_payload_root": normalize_text(document.get("authority_payload_root")),
            "source_text_path": normalize_text(document.get("source_text_path")),
            "source_raw_response_path": normalize_text(document.get("source_raw_response_path")),
            "source_table_files": ensure_list(document.get("source_table_files")),
            "title": normalize_text(document.get("title")),
        }
        normalized["table_formulation_scopes"] = [
            {
                "scope_id": f"{document_key}__table_formulation_scope__{index:02d}",
                "table_id": normalize_text(item.get("table_id")),
                "table_path": "",
                "table_asset_id": "",
                "source_table_asset_id": normalize_text(item.get("source_table_asset_id")),
                "source_table_reference": normalize_text(item.get("source_table_reference")),
                "table_scope_locators": item.get("table_scope_locators") if isinstance(item.get("table_scope_locators"), dict) else {},
                "variable_name": "",
                "candidate_values": [],
                "is_formulation_table": scope_kind_is_formulation_bearing(item.get("scope_kind"), item.get("is_formulation_bearing")),
                "table_type": scope_kind_to_table_type(normalize_text(item.get("scope_kind"))),
                "confidence": normalize_text(item.get("confidence")) or "low",
                "evidence_span": "",
                "marker_provenance": "llm_parsed",
            }
            for index, item in enumerate(ensure_list(document.get("table_scopes")), start=1)
            if isinstance(item, dict) and normalize_text(item.get("table_id"))
        ]
        normalized["table_variable_roles"] = [
            {
                "table_id": normalize_text(item.get("table_id")),
                "varying_variables": primary_variable_names,
                "constant_variables": [],
                "new_variables_introduced": [],
                "marker_provenance": "llm_parsed",
            }
            for item in ensure_list(document.get("table_scopes"))
            if isinstance(item, dict)
            and normalize_text(item.get("table_id"))
            and scope_kind_is_formulation_bearing(item.get("scope_kind"), item.get("is_formulation_bearing"))
            and normalize_text(item.get("scope_kind")) != "non_formulation"
            and primary_variable_names
        ]
        normalized["selection_markers"] = []
        normalized["inheritance_markers"] = []
        normalized["boundary_markers"] = [
            {
                "table_id": normalize_text(item.get("table_id")),
                "is_doe": bool(item.get("is_doe")),
                "marker_provenance": "llm_parsed",
            }
            for item in ensure_list(document.get("table_scopes"))
            if isinstance(item, dict) and normalize_text(item.get("table_id"))
        ]
        normalized_identities: list[dict[str, Any]] = []
        normalized_relation_cues: list[dict[str, Any]] = []
        normalized_processes: list[dict[str, Any]] = []
        normalized_factors: list[dict[str, Any]] = []
        method_hint = normalize_text(semantic_signals.get("primary_preparation_method_hint"))
        for index, item in enumerate(ensure_list(document.get("formulation_candidates")), start=1):
            if not isinstance(item, dict):
                continue
            formulation_id = normalize_text(item.get("candidate_id"))
            if not formulation_id:
                continue
            raw_label = normalize_text(item.get("label_hint")) or formulation_id
            instance_role = normalize_text(item.get("instance_role"))
            candidate_kind = normalize_text(item.get("candidate_kind"))
            normalized_identities.append(
                {
                    "formulation_candidate_id": formulation_id,
                    "raw_formulation_label": raw_label,
                    "parent_candidate_id": normalize_text(item.get("parent_candidate_hint")),
                    "instance_kind": candidate_kind or "unclear",
                    "formulation_role": role_to_legacy_formulation_role(instance_role),
                    "identity_confidence": normalize_text(item.get("confidence")) or normalize_text(item.get("status")) or "reported",
                    "candidate_source": normalize_text(document.get("source_mode")) or "stage2_v2_semantic_objects",
                    "stage2_semantic_source_mode": normalize_text(item.get("stage2_semantic_source_mode") or document.get("stage2_semantic_source_mode")),
                    "semantic_universe_authority": normalize_text(item.get("semantic_universe_authority") or document.get("semantic_universe_authority")),
                    "row_materialization_mode": normalize_text(item.get("row_materialization_mode")),
                    "semantic_scope_authority": normalize_text(item.get("semantic_scope_authority")),
                    "semantic_scope_ref": normalize_text(item.get("semantic_scope_ref")),
                    "change_role": kind_to_change_role(candidate_kind, instance_role),
                    "change_descriptions": [normalize_text(item.get("core_change_hint"))] if normalize_text(item.get("core_change_hint")) else [],
                    "instance_context_tags": [instance_role] if instance_role and instance_role != "unclear" else [],
                    "change_context_tags": [],
                }
            )
            if method_hint:
                normalized_processes.append(
                    {
                        "process_step_id": f"{document_key}__{formulation_id}__process_01",
                        "formulation_candidate_id": formulation_id,
                        "process_name_raw": method_hint,
                        "process_step_order_hint": "1",
                    }
                )
            parent_hint = normalize_text(item.get("parent_candidate_hint"))
            if parent_hint:
                normalized_relation_cues.append(
                    {
                        "relation_id": f"{document_key}__relation_{len(normalized_relation_cues) + 1:02d}",
                        "source_object_ref": build_target_object_ref("formulation_identity_candidate", formulation_id, formulation_id),
                        "target_object_ref": build_target_object_ref("formulation_identity_candidate", parent_hint, parent_hint),
                        "relation_type_raw": "inherits_from",
                        "relation_note_raw": normalize_text(item.get("shared_context_hint") or item.get("core_change_hint") or "Derived from parent_candidate_hint."),
                    }
                )
        factor_expression = " | ".join(selected_condition_hints)
        for index, variable_name in enumerate(primary_variable_names, start=1):
            normalized_factors.append(
                {
                    "factor_id": f"{document_key}__signal_factor_{index:02d}",
                    "formulation_candidate_id": "",
                    "factor_name_raw": variable_name,
                    "factor_expression_raw": factor_expression,
                    "factor_kind": "doe_factor" if normalize_bool(semantic_signals.get("has_variable_sweep"), False) else "identity_signal",
                    "identity_defining_signal": "yes",
                }
            )
        normalized[legacy_identity_key] = normalized_identities
        normalized[LEGACY_OBJECT_KEYS["component_candidate"]] = []
        normalized[LEGACY_OBJECT_KEYS["phase_candidate"]] = []
        normalized[LEGACY_OBJECT_KEYS["process_step_candidate"]] = normalized_processes
        normalized[LEGACY_OBJECT_KEYS["variable_or_factor_candidate"]] = normalized_factors
        normalized[LEGACY_OBJECT_KEYS["measurement_candidate"]] = []
        normalized[LEGACY_OBJECT_KEYS["relation_cue"]] = normalized_relation_cues
        normalized[LEGACY_OBJECT_KEYS["evidence_handoff"]] = []
        return normalized

    canonical_identity_key = CANONICAL_STAGE2_V2_KEYS["formulation_identity_candidate"]
    has_canonical = any(key in document for key in CANONICAL_STAGE2_V2_KEYS.values())
    has_legacy = any(key in document for key in LEGACY_OBJECT_KEYS.values())
    has_canonical_identity = canonical_identity_key in document
    has_legacy_identity = legacy_identity_key in document
    if not has_canonical and not has_legacy:
        raise ValueError(
            "Stage2 semantic document is missing recognized semantic-object families for compatibility projection."
        )
    if has_legacy_identity and not has_canonical_identity:
        return document

    normalized: dict[str, Any] = {
        "document_key": normalize_text(document.get("document_key") or document.get("key")),
        "doi": normalize_text(document.get("doi")),
        "model_name": normalize_text(document.get("model_name") or document.get("source_mode")) or "stage2_v2_semantic_objects",
        "source_mode": normalize_text(document.get("source_mode")),
        "stage2_semantic_source_mode": normalize_text(document.get("stage2_semantic_source_mode")),
        "semantic_universe_authority": normalize_text(document.get("semantic_universe_authority")),
        "semantic_scope_declarations": ensure_list(document.get("semantic_scope_declarations")),
        "semantic_signals": document.get("semantic_signals") if isinstance(document.get("semantic_signals"), dict) else {},
        "table_formulation_scopes": ensure_list(document.get("table_formulation_scopes")),
        "table_variable_roles": ensure_list(document.get("table_variable_roles")),
        "selection_markers": ensure_list(document.get("selection_markers")),
        "inheritance_markers": ensure_list(document.get("inheritance_markers")),
        "boundary_markers": ensure_list(document.get("boundary_markers")),
        "authority_run_dir": normalize_text(document.get("authority_run_dir")),
        "authority_payload_root": normalize_text(document.get("authority_payload_root")),
        # Preserve source-path metadata so bounded deterministic recovery can
        # still resolve the explicit Stage1 table anchors from the semantic
        # intermediate, even after the semantic payload is normalized into the
        # legacy-shaped compatibility view.
        "source_text_path": normalize_text(document.get("source_text_path")),
        "source_raw_response_path": normalize_text(document.get("source_raw_response_path")),
        "source_table_files": ensure_list(document.get("source_table_files")),
        "title": normalize_text(document.get("title")),
    }

    evidence_spans = [
        item
        for item in ensure_list(document.get(CANONICAL_STAGE2_V2_KEYS["evidence_handoff"]))
        if isinstance(item, dict)
    ]
    evidence_by_id = {
        normalize_text(item.get("span_id") or item.get("evidence_span_id")): item
        for item in evidence_spans
        if normalize_text(item.get("span_id") or item.get("evidence_span_id"))
    }

    formulation_candidates = [
        item
        for item in ensure_list(document.get(canonical_identity_key))
        if isinstance(item, dict)
    ]
    normalized_identities: list[dict[str, Any]] = []
    normalized_handoffs: list[dict[str, Any]] = []

    for item in formulation_candidates:
        formulation_id = normalize_text(item.get("formulation_candidate_id") or item.get("candidate_id"))
        raw_label = normalize_text(item.get("raw_formulation_label") or item.get("raw_label") or formulation_id)
        evidence_span_ids = [
            normalize_text(span_id)
            for span_id in ensure_list(item.get("evidence_span_ids"))
            if normalize_text(span_id)
        ]
        normalized_identities.append(
            {
                "formulation_candidate_id": formulation_id,
                "raw_formulation_label": raw_label,
                "parent_candidate_id": normalize_text(item.get("parent_candidate_id")),
                "instance_kind": normalize_text(item.get("instance_kind")) or "unclear",
                "formulation_role": normalize_text(item.get("formulation_role")) or "unclear",
                "identity_confidence": normalize_text(item.get("identity_confidence") or item.get("status")) or "reported",
                "candidate_source": normalize_text(item.get("candidate_source") or document.get("source_mode")) or "stage2_v2_semantic_objects",
                "stage2_semantic_source_mode": normalize_text(item.get("stage2_semantic_source_mode") or document.get("stage2_semantic_source_mode")),
                "semantic_universe_authority": normalize_text(item.get("semantic_universe_authority") or document.get("semantic_universe_authority")),
                "row_materialization_mode": normalize_text(item.get("row_materialization_mode")),
                "semantic_scope_authority": normalize_text(item.get("semantic_scope_authority")),
                "semantic_scope_ref": normalize_text(item.get("semantic_scope_ref")),
                "change_role": normalize_text(item.get("change_role")) or "unclear",
                "change_descriptions": ensure_list(item.get("change_descriptions")),
                "instance_context_tags": ensure_list(item.get("instance_context_tags")),
                "change_context_tags": ensure_list(item.get("change_context_tags")),
            }
        )
        for span_id in evidence_span_ids:
            span = evidence_by_id.get(span_id)
            if not span:
                continue
            normalized_handoffs.append(
                {
                    "source_region_type": normalize_text(span.get("source_region_type")),
                    "source_locator_text": normalize_text(span.get("source_locator_text") or span.get("locator_raw")),
                    "supporting_snippet": normalize_text(span.get("supporting_text") or span.get("span_text_raw")),
                    "target_object_ref": build_target_object_ref("formulation_identity_candidate", formulation_id, formulation_id),
                    "target_field_name": "formulation_identity",
                }
            )

    normalized_components: list[dict[str, Any]] = []
    for item in ensure_list(document.get(CANONICAL_STAGE2_V2_KEYS["component_candidate"])):
        if not isinstance(item, dict):
            continue
        expressions = [expr for expr in ensure_list(item.get("expressions")) if isinstance(expr, dict)]
        expression_texts = [
            normalize_text(
                expr.get("expression_text_raw")
                or expr.get("expression_text")
                or " ".join(
                    part
                    for part in [
                        normalize_text(expr.get("value_raw")),
                        normalize_text(expr.get("unit_raw")),
                    ]
                    if part
                )
            )
            for expr in expressions
        ]
        expression_texts = [text for text in expression_texts if text]
        parsed_values = [
            normalize_text(expr.get("value_raw")) or first_number(normalize_text(expr.get("expression_text_raw")))
            for expr in expressions
            if normalize_text(expr.get("value_raw")) or first_number(normalize_text(expr.get("expression_text_raw")))
        ]
        properties = [
            {
                "name": normalize_text(expr.get("expression_type")) or "expression",
                "value": normalize_text(expr.get("expression_text_raw") or expr.get("expression_text")),
                "raw_value": normalize_text(expr.get("qualifier_raw")),
            }
            for expr in expressions
            if normalize_text(expr.get("expression_text_raw") or expr.get("expression_text"))
        ]
        amount_expression = " | ".join(expression_texts) if expression_texts else normalize_text(item.get("amount_text"))
        parsed_value = " | ".join(parsed_values) if parsed_values else value_or_first_number(item.get("amount_text"))
        normalized_components.append(
            {
                "component_id": normalize_text(item.get("component_id")),
                "formulation_candidate_id": normalize_text(item.get("formulation_candidate_id")),
                "component_name_raw": normalize_text(item.get("component_name_raw") or item.get("component_name") or item.get("name_raw")),
                "component_role_raw": normalize_text(item.get("component_role_raw") or item.get("component_role")),
                "amount_expression_raw": amount_expression,
                "parsed_value_raw": parsed_value,
                "component_properties_raw": properties,
                "phase_hint_raw": normalize_text(item.get("phase_hint_raw") or item.get("phase_hint")),
            }
        )

    normalized_factors: list[dict[str, Any]] = []
    for item in ensure_list(document.get(CANONICAL_STAGE2_V2_KEYS["variable_or_factor_candidate"])):
        if not isinstance(item, dict):
            continue
        variable_role = normalize_text(item.get("variable_role"))
        identity_signal = "yes" if variable_role in {"identity_signal", "doe_factor"} else "no"
        normalized_factors.append(
            {
                "factor_id": normalize_text(item.get("factor_id") or item.get("variable_id")),
                "formulation_candidate_id": normalize_text(item.get("formulation_candidate_id")),
                "factor_name_raw": normalize_text(item.get("factor_name_raw") or item.get("variable_name")),
                "factor_expression_raw": normalize_text(item.get("factor_expression_raw") or item.get("value_text")),
                "factor_kind": variable_role,
                "identity_defining_signal": identity_signal,
            }
        )

    normalized_measurements: list[dict[str, Any]] = []
    for item in ensure_list(document.get(CANONICAL_STAGE2_V2_KEYS["measurement_candidate"])):
        if not isinstance(item, dict):
            continue
        normalized_measurements.append(
            {
                "measurement_id": normalize_text(item.get("measurement_id")),
                "formulation_candidate_id": normalize_text(item.get("formulation_candidate_id")),
                "measurement_name_raw": normalize_text(item.get("measurement_name_raw") or item.get("measurement_name")),
                "measurement_value_raw": normalize_text(item.get("measurement_value_raw") or item.get("value_text")),
                "measurement_unit_raw": normalize_text(item.get("measurement_unit_raw") or item.get("unit_text")),
            }
        )

    normalized_relation_cues: list[dict[str, Any]] = []
    for item in ensure_list(document.get(CANONICAL_STAGE2_V2_KEYS["relation_cue"])):
        if not isinstance(item, dict):
            continue
        normalized_relation_cues.append(
            {
                "relation_id": normalize_text(item.get("relation_id")),
                "source_object_ref": build_target_object_ref(
                    "formulation_identity_candidate",
                    normalize_text(item.get("source_candidate_id")),
                    normalize_text(item.get("source_candidate_id")),
                ),
                "target_object_ref": build_target_object_ref(
                    "formulation_identity_candidate",
                    normalize_text(item.get("target_candidate_id")),
                    normalize_text(item.get("target_candidate_id")),
                ),
                "relation_type_raw": normalize_text(item.get("relation_type")),
                "relation_note_raw": normalize_text(item.get("note")),
            }
        )

    normalized[legacy_identity_key] = normalized_identities
    normalized[LEGACY_OBJECT_KEYS["component_candidate"]] = normalized_components
    normalized[LEGACY_OBJECT_KEYS["phase_candidate"]] = [
        item
        for item in ensure_list(document.get(CANONICAL_STAGE2_V2_KEYS["phase_candidate"]))
        if isinstance(item, dict)
    ]
    normalized[LEGACY_OBJECT_KEYS["process_step_candidate"]] = [
        item
        for item in ensure_list(document.get(CANONICAL_STAGE2_V2_KEYS["process_step_candidate"]))
        if isinstance(item, dict)
    ]
    normalized[LEGACY_OBJECT_KEYS["variable_or_factor_candidate"]] = normalized_factors
    normalized[LEGACY_OBJECT_KEYS["measurement_candidate"]] = normalized_measurements
    normalized[LEGACY_OBJECT_KEYS["relation_cue"]] = normalized_relation_cues
    normalized[LEGACY_OBJECT_KEYS["evidence_handoff"]] = normalized_handoffs
    return normalized


def load_jsonl_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            documents.append(normalize_stage2_document_for_projection(json.loads(line)))
    return documents


def load_authority_sidecar(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entries = payload.get("entries_by_paper_key")
    if not isinstance(entries, dict):
        return {}
    return {
        normalize_text(key): value
        for key, value in entries.items()
        if normalize_text(key) and isinstance(value, dict)
    }


def merge_table_scope_locators(document: dict[str, Any], locator_entries: list[dict[str, Any]]) -> None:
    if not locator_entries:
        return
    locator_by_ref: dict[str, dict[str, str]] = {}
    for entry in locator_entries:
        locator = {
            "table_id": normalize_text(entry.get("table_id")),
            "source_table_asset_id": normalize_text(entry.get("source_table_asset_id")),
            "source_table_reference": normalize_text(entry.get("source_table_reference")),
        }
        if not any(locator.values()):
            continue
        for candidate in locator.values():
            normalized = normalize_token(candidate)
            if normalized and normalized not in locator_by_ref:
                locator_by_ref[normalized] = dict(locator)

    def resolve_locator(value: Any) -> dict[str, str] | None:
        normalized = normalize_token(value)
        if not normalized:
            return None
        return locator_by_ref.get(normalized)

    for scope in ensure_list(document.get("table_scopes")):
        if not isinstance(scope, dict):
            continue
        locator = resolve_locator(scope.get("table_id"))
        if locator is None:
            continue
        scope["table_scope_locators"] = dict(locator)
        if locator.get("source_table_asset_id"):
            scope["source_table_asset_id"] = locator["source_table_asset_id"]
        if locator.get("source_table_reference"):
            scope["source_table_reference"] = locator["source_table_reference"]

    for scope in ensure_list(document.get("table_formulation_scopes")):
        if not isinstance(scope, dict):
            continue
        locator = resolve_locator(scope.get("table_id"))
        if locator is None:
            continue
        scope["table_scope_locators"] = dict(locator)
        if locator.get("source_table_asset_id"):
            scope["table_asset_id"] = locator["source_table_asset_id"]
            scope["source_table_asset_id"] = locator["source_table_asset_id"]
        if locator.get("source_table_reference"):
            scope["source_table_reference"] = locator["source_table_reference"]

    for declaration in ensure_list(document.get("semantic_scope_declarations")):
        if not isinstance(declaration, dict):
            continue
        refs = [item for item in ensure_list(declaration.get("table_scope_refs")) if normalize_text(item)]
        locators = []
        for ref in refs:
            locator = resolve_locator(ref)
            if locator is not None:
                locators.append(dict(locator))
        if locators:
            declaration["table_scope_locators"] = locators


def merge_authority_sidecar(document: dict[str, Any], sidecar_entry: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(sidecar_entry, dict):
        return {
            "reattachment_status": "missing",
            "reattachment_source": "",
            "reattachment_failure_reason": "sidecar_entry_missing",
        }
    authority_run_dir = normalize_text(sidecar_entry.get("authority_run_dir"))
    authority_payload_root = normalize_text(sidecar_entry.get("authority_payload_root"))
    if authority_run_dir:
        document["authority_run_dir"] = authority_run_dir
    if authority_payload_root:
        document["authority_payload_root"] = authority_payload_root
    merge_table_scope_locators(
        document,
        [item for item in ensure_list(sidecar_entry.get("table_scope_locators")) if isinstance(item, dict)],
    )
    return {
        "reattachment_status": normalize_text(sidecar_entry.get("resolution_status"))
        or ("resolved" if authority_payload_root else "unresolved"),
        "reattachment_source": normalize_text(sidecar_entry.get("resolution_source")),
        "reattachment_failure_reason": normalize_text(sidecar_entry.get("failure_reason")),
    }


def object_rows(document: dict[str, Any], object_type: str) -> list[dict[str, Any]]:
    value = document.get(LEGACY_OBJECT_KEYS[object_type], [])
    return [item for item in ensure_list(value) if isinstance(item, dict)]


def ranked_sort_key(item: dict[str, Any], key_name: str) -> tuple[int, str]:
    order = item.get(key_name)
    number = first_number(str(order))
    return (int(float(number)) if number else 10_000, normalize_text(order))


def field_bundle_empty() -> dict[str, str]:
    return {
        "value": "",
        "value_text": "",
        "scope": "",
        "membership_confidence": "",
        "evidence_region_type": "",
        "missing_reason": "",
    }


def add_trace(
    traces: list[dict[str, str]],
    document_key: str,
    formulation_id: str,
    legacy_field: str,
    source_refs: list[str],
    mapping_status: str,
    direct_or_derived: str,
    notes: str,
) -> None:
    traces.append(
        {
            "document_key": document_key,
            "formulation_id": formulation_id,
            "legacy_field": legacy_field,
            "source_replacement_objects": " | ".join(source_refs),
            "mapping_status": mapping_status,
            "direct_or_derived": direct_or_derived,
            "notes": notes,
        }
    )


def component_matches(component: dict[str, Any], role: str) -> bool:
    raw_role = normalize_token(component.get("component_role_raw"))
    aliases = {normalize_token(alias) for alias in ROLE_ALIASES[role]}
    return raw_role in aliases


def choose_components(components: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    matched = [item for item in components if component_matches(item, role)]
    return sorted(matched, key=lambda item: normalize_text(item.get("component_id")))


def parse_component_properties(component: dict[str, Any]) -> list[dict[str, Any]]:
    props = parse_json_maybe(component.get("component_properties_raw"))
    if isinstance(props, dict):
        return [{"name": key, "value": value} for key, value in props.items()]
    if isinstance(props, list):
        result: list[dict[str, Any]] = []
        for item in props:
            if isinstance(item, dict):
                result.append(item)
            else:
                result.append({"name": normalize_text(item), "value": normalize_text(item)})
        return result
    text = normalize_text(props)
    return [{"name": "raw", "value": text}] if text else []


def find_property(component: dict[str, Any], *needles: str) -> str:
    needles_norm = [normalize_token(item) for item in needles]
    for prop in parse_component_properties(component):
        name = normalize_token(prop.get("name"))
        value = normalize_text(prop.get("value"))
        raw = normalize_text(prop.get("raw_value"))
        text = value or raw
        haystack = f"{name} {normalize_token(text)}"
        if any(needle in haystack for needle in needles_norm):
            return text
    return ""


def polymer_mw_projection_text(component: dict[str, Any]) -> str:
    return find_property(component, "molecular weight", "mw", "kda")


def polymer_mw_projection_value(component: dict[str, Any]) -> str:
    text = polymer_mw_projection_text(component)
    if not text:
        return ""
    token_text = normalize_token(text)
    if "viscosity" in token_text or "dl g" in token_text:
        return ""
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:kda|da)\b", text, flags=re.IGNORECASE)
    if range_match:
        first = range_match.group(1)
        second = range_match.group(2)
        if "da" in text.lower() and "kda" not in text.lower():
            return f"{float(first) / 1000:.6g}-{float(second) / 1000:.6g}"
        return f"{first}-{second}"
    single_match = re.search(r"(\d+(?:\.\d+)?)\s*(kda|da)\b", text, flags=re.IGNORECASE)
    if single_match:
        value = float(single_match.group(1))
        if single_match.group(2).lower() == "da":
            value /= 1000.0
        return f"{value:.6g}"
    # Preserve article-native polymer grade strings as text-only support
    # rather than incorrectly reading grade digits as a numeric MW value.
    if any(marker in token_text for marker in ["resomer", "rg502", "rg503", "rg504", "rg505", "rg506", "rg750", "rg752", "rg753", "rg756"]):
        return ""
    return first_number(text)


def measurement_target_name(item: dict[str, Any]) -> str:
    return normalize_token(item.get("measurement_name_raw"))


def best_handoff(
    handoffs: list[dict[str, Any]],
    formulation_id: str,
    target_prefix: str = "",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for handoff in handoffs:
        target_ref = normalize_text(handoff.get("target_object_ref"))
        if formulation_id and formulation_id in target_ref:
            if not target_prefix or normalize_text(handoff.get("target_field_name")).startswith(target_prefix):
                results.append(handoff)
    return results


def handoff_projection(handoffs: list[dict[str, Any]]) -> tuple[str, str, str]:
    if not handoffs:
        return "", "", ""
    return (
        choose_first(handoffs, "source_region_type"),
        choose_first(handoffs, "source_locator_text"),
        choose_first(handoffs, "supporting_snippet"),
    )


def project_choice(items: list[dict[str, Any]], value_getter, text_getter=None) -> tuple[str, str, str]:
    if not items:
        return "", "", UNAVAILABLE
    if len(items) == 1:
        value = normalize_text(value_getter(items[0]))
        text = normalize_text(text_getter(items[0]) if text_getter else value)
        return value, text, DIRECT
    values = [normalize_text(value_getter(item)) for item in items if normalize_text(value_getter(item))]
    texts = [
        normalize_text(text_getter(item) if text_getter else value_getter(item))
        for item in items
        if normalize_text(text_getter(item) if text_getter else value_getter(item))
    ]
    return " | ".join(values), " | ".join(texts), COMPRESSED


def base_row(identity: dict[str, Any], document_key: str, doi: str, model_name: str) -> dict[str, str]:
    row = {column: "" for column in compatibility_output_columns()}
    change_descriptions = parse_string_list(identity.get("change_descriptions"))
    instance_context_tags = parse_string_list(identity.get("instance_context_tags"))
    change_context_tags = parse_string_list(identity.get("change_context_tags"))
    row.update(
        {
            "key": document_key,
            "doi": doi,
            "model": model_name,
            "local_instance_id": normalize_text(identity.get("formulation_candidate_id")),
            "formulation_id": normalize_text(identity.get("formulation_candidate_id")),
            "raw_formulation_label": normalize_text(identity.get("raw_formulation_label")),
            "parent_instance_id": normalize_text(identity.get("parent_candidate_id")),
            "instance_kind_raw": normalize_text(identity.get("instance_kind")),
            "instance_kind_inferred": normalize_text(identity.get("instance_kind")),
            "instance_kind_reconciliation_note": "compatibility_projection_v1",
            "instance_kind": normalize_text(identity.get("instance_kind")),
            "change_descriptions": stringify_json(change_descriptions),
            "change_role": normalize_text(identity.get("change_role")) or "unclear",
            "instance_context_tags": stringify_json(instance_context_tags),
            "change_context_tags": stringify_json(change_context_tags),
            "formulation_role": normalize_text(identity.get("formulation_role")),
            "instance_confidence": normalize_text(identity.get("identity_confidence")) or "projected",
            "candidate_source": normalize_text(identity.get("candidate_source")) or "compatibility_projection",
            "stage2_semantic_source_mode": normalize_text(identity.get("stage2_semantic_source_mode")),
            "semantic_universe_authority": normalize_text(identity.get("semantic_universe_authority")),
            "row_materialization_mode": normalize_text(identity.get("row_materialization_mode")),
            "semantic_scope_authority": normalize_text(identity.get("semantic_scope_authority")),
            "semantic_scope_ref": normalize_text(identity.get("semantic_scope_ref")),
        }
    )
    return row


def project_document(
    document: dict[str, Any],
    *,
    authority_sidecar_entry: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any], dict[str, str] | None]:
    reattachment_summary = merge_authority_sidecar(document, authority_sidecar_entry)
    augment_document_with_table_markers(document)
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    doi = normalize_text(document.get("doi"))
    model_name = normalize_text(document.get("model_name")) or "stage2_replacement_compatibility_projection_v1"
    source_mode = stage2_semantic_source_mode(document)
    identities = sorted(
        object_rows(document, "formulation_identity_candidate"),
        key=lambda item: normalize_text(item.get("formulation_candidate_id")),
    )
    components = object_rows(document, "component_candidate")
    phases = sorted(
        object_rows(document, "phase_candidate"),
        key=lambda item: ranked_sort_key(item, "phase_order_hint"),
    )
    processes = sorted(
        object_rows(document, "process_step_candidate"),
        key=lambda item: ranked_sort_key(item, "process_step_order_hint"),
    )
    factors = object_rows(document, "variable_or_factor_candidate")
    measurements = object_rows(document, "measurement_candidate")
    handoffs = object_rows(document, "evidence_handoff")

    rows: list[dict[str, str]] = []
    traces: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []

    for identity in identities:
        formulation_id = normalize_text(identity.get("formulation_candidate_id"))
        row = base_row(identity, document_key, doi, model_name)
        row["stage2_semantic_source_mode"] = row.get("stage2_semantic_source_mode") or source_mode
        row["semantic_universe_authority"] = row.get("semantic_universe_authority") or (
            "governed_fallback_semantic_source" if source_mode == FALLBACK_SEMANTIC_SOURCE_MODE else "llm_semantic_discovery"
        )
        row["row_materialization_mode"] = row.get("row_materialization_mode") or (
            "governed_fallback_semantic_source" if source_mode == FALLBACK_SEMANTIC_SOURCE_MODE else "llm_semantic_discovery"
        )
        row["semantic_scope_authority"] = row.get("semantic_scope_authority") or (
            "governed_fallback_semantic_source" if source_mode == FALLBACK_SEMANTIC_SOURCE_MODE else "llm_declared_scope"
        )
        row["semantic_scope_ref"] = row.get("semantic_scope_ref") or default_semantic_scope_ref(identity, document)
        owned_components = [item for item in components if normalize_text(item.get("formulation_candidate_id")) == formulation_id]
        owned_phases = [item for item in phases if normalize_text(item.get("formulation_candidate_id")) == formulation_id]
        owned_processes = [item for item in processes if normalize_text(item.get("formulation_candidate_id")) == formulation_id]
        owned_factors = [item for item in factors if normalize_text(item.get("formulation_candidate_id")) == formulation_id]
        owned_measurements = [item for item in measurements if normalize_text(item.get("formulation_candidate_id")) == formulation_id]
        owned_handoffs = best_handoff(handoffs, formulation_id)

        region, locator, snippet = handoff_projection(owned_handoffs)
        row["instance_evidence_region_type"] = region
        row["evidence_section"] = locator
        row["evidence_span_text"] = snippet
        row["supporting_evidence_refs"] = stringify_json(
            [
                {
                    "source_region_type": normalize_text(item.get("source_region_type")),
                    "source_locator_text": normalize_text(item.get("source_locator_text")),
                    "supporting_snippet": normalize_text(item.get("supporting_snippet")),
                    "target_field_name": normalize_text(item.get("target_field_name")),
                }
                for item in owned_handoffs
            ]
        )
        row[TABLE_SCOPE_FIELD] = stringify_json(document.get("table_formulation_scopes"))
        row[TABLE_VARIABLE_ROLE_FIELD] = stringify_json(document.get("table_variable_roles"))
        row[SELECTION_MARKER_FIELD] = stringify_json(
            execution_ready_markers(
                [item for item in ensure_list(document.get("selection_markers")) if isinstance(item, dict)]
            )
        )
        row[INHERITANCE_MARKER_FIELD] = stringify_json(
            execution_ready_markers(
                [item for item in ensure_list(document.get("inheritance_markers")) if isinstance(item, dict)]
            )
        )
        row[BOUNDARY_MARKER_FIELD] = stringify_json(document.get("boundary_markers"))
        identity_variables_payload = build_identity_variables_payload(owned_factors)
        row[IDENTITY_VARIABLES_FIELD] = stringify_json(identity_variables_payload)
        add_trace(
            traces,
            document_key,
            formulation_id,
            IDENTITY_VARIABLES_FIELD,
            [normalize_text(item.get("factor_id")) for item in owned_factors if normalize_text(item.get("factor_id"))],
            DIRECT if identity_variables_payload else UNAVAILABLE,
            DIRECT if identity_variables_payload else UNAVAILABLE,
            "Preserved identity-bearing semantic variables as additive compatibility metadata.",
        )

        polymer_components = choose_components(owned_components, "polymer")
        surfactant_components = choose_components(owned_components, "surfactant")
        solvent_components = choose_components(owned_components, "organic_solvent")
        drug_components = choose_components(owned_components, "drug")

        polymer_refs = [normalize_text(item.get("component_id")) for item in polymer_components]
        surfactant_refs = [normalize_text(item.get("component_id")) for item in surfactant_components]
        solvent_refs = [normalize_text(item.get("component_id")) for item in solvent_components]
        drug_refs = [normalize_text(item.get("component_id")) for item in drug_components]

        _, polymer_name_text, polymer_name_status = project_choice(
            polymer_components,
            lambda item: item.get("component_name_raw"),
        )
        row["polymer_name_raw"] = polymer_name_text
        row["polymer_identity"] = infer_polymer_identity(polymer_name_text) if polymer_name_text else ""
        add_trace(traces, document_key, formulation_id, "polymer_name_raw", polymer_refs, polymer_name_status, polymer_name_status, "Projected from polymer component names.")
        add_trace(traces, document_key, formulation_id, "polymer_identity", polymer_refs, DERIVED if polymer_name_text else UNAVAILABLE, DERIVED if polymer_name_text else UNAVAILABLE, "Derived from polymer component names using deterministic family rules.")

        bundles = {field: field_bundle_empty() for field in CORE_FIELDS}

        def assign_bundle(field: str, value: str, value_text: str, status: str, refs: list[str], note: str) -> None:
            bundle = bundles[field]
            bundle["value"] = value
            bundle["value_text"] = value_text
            bundle["membership_confidence"] = (
                "projected_direct"
                if status == DIRECT
                else "projected_compressed"
                if status == COMPRESSED
                else "projected_derived"
                if status == DERIVED
                else ""
            )
            bundle["evidence_region_type"] = region
            bundle["missing_reason"] = "" if value_text else "not_projectable_from_current_replacement_objects"
            add_trace(traces, document_key, formulation_id, field, refs, status, status, note)

        value, value_text, status = project_choice(
            polymer_components,
            lambda item: find_property(item, "la ga ratio", "ratio"),
        )
        assign_bundle("la_ga_ratio", value, value_text, status, polymer_refs, "Projected from polymer component properties.")

        value, value_text, status = project_choice(
            polymer_components,
            polymer_mw_projection_value,
            polymer_mw_projection_text,
        )
        assign_bundle("polymer_mw_kDa", value, value_text, status, polymer_refs, "Projected from polymer component molecular-weight properties.")

        value, value_text, status = project_choice(
            polymer_components,
            lambda item: first_number(normalize_text(item.get("parsed_value_raw")) or normalize_text(item.get("amount_expression_raw"))),
            lambda item: normalize_text(item.get("amount_expression_raw")) or normalize_text(item.get("parsed_value_raw")),
        )
        assign_bundle("plga_mass_mg", value, value_text, status, polymer_refs, "Projected from polymer component amount expressions.")

        value, value_text, status = project_choice(surfactant_components, lambda item: item.get("component_name_raw"))
        assign_bundle("surfactant_name", value, value_text, status, surfactant_refs, "Projected from surfactant or stabilizer components.")

        value, value_text, status = project_choice(
            surfactant_components,
            lambda item: normalize_text(item.get("parsed_value_raw")),
            lambda item: normalize_text(item.get("amount_expression_raw")) or normalize_text(item.get("parsed_value_raw")),
        )
        assign_bundle("surfactant_concentration_text", value, value_text, status, surfactant_refs, "Projected from surfactant amount expressions.")

        pva_components = [item for item in surfactant_components if "pva" in normalize_token(item.get("component_name_raw"))]
        value, value_text, status = project_choice(
            pva_components,
            lambda item: first_number(normalize_text(item.get("parsed_value_raw")) or normalize_text(item.get("amount_expression_raw"))),
            lambda item: normalize_text(item.get("amount_expression_raw")) or normalize_text(item.get("parsed_value_raw")),
        )
        assign_bundle("pva_conc_percent", value, value_text, status, [normalize_text(item.get("component_id")) for item in pva_components], "Projected only for PVA-labeled surfactant components.")

        value, value_text, status = project_choice(solvent_components, lambda item: item.get("component_name_raw"))
        assign_bundle("organic_solvent", value, value_text, status, solvent_refs, "Projected from solvent components.")

        value, value_text, status = project_choice(drug_components, lambda item: item.get("component_name_raw"))
        assign_bundle("drug_name", value, value_text, status, drug_refs, "Projected from drug components.")

        value, value_text, status = project_choice(
            drug_components,
            lambda item: normalize_text(item.get("parsed_value_raw")),
            lambda item: normalize_text(item.get("amount_expression_raw")) or normalize_text(item.get("parsed_value_raw")),
        )
        assign_bundle("drug_feed_amount_text", value, value_text, status, drug_refs, "Projected from drug amount expressions.")

        process_refs = [normalize_text(item.get("process_step_id")) for item in owned_processes]
        value, value_text, status = project_choice(owned_processes, lambda item: item.get("process_name_raw"))
        assign_bundle("emul_method", value, value_text, status, process_refs, "Projected from process-step names.")
        row["preparation_method"] = value_text
        add_trace(traces, document_key, formulation_id, "preparation_method", process_refs, status, status, "Projected from process-step names.")

        phase_texts = [normalize_text(item.get("phase_code_raw") or item.get("phase_role_hint")) for item in owned_phases if normalize_text(item.get("phase_code_raw") or item.get("phase_role_hint"))]
        phase_refs = [normalize_text(item.get("phase_id")) for item in owned_phases]
        if phase_texts:
            row["emulsion_structure"] = " | ".join(phase_texts)
            add_trace(traces, document_key, formulation_id, "emulsion_structure", phase_refs, DIRECT if len(phase_texts) == 1 else COMPRESSED, DIRECT if len(phase_texts) == 1 else COMPRESSED, "Projected from phase candidates.")
        else:
            add_trace(traces, document_key, formulation_id, "emulsion_structure", [], UNAVAILABLE, UNAVAILABLE, "No phase candidates available.")

        emul_type_text = ""
        if phase_texts:
            joined = " ".join(phase_texts).lower()
            if "w1" in joined and "w2" in joined and "o" in joined:
                emul_type_text = "w1/o/w2"
            elif "o" in joined and "w" in joined:
                emul_type_text = "o/w"
        if not emul_type_text:
            for factor in owned_factors:
                name = normalize_token(factor.get("factor_name_raw"))
                if "emulsion" in name or "phase" in name:
                    emul_type_text = normalize_text(factor.get("factor_expression_raw"))
                    break
        if emul_type_text:
            bundles["emul_type"]["value"] = emul_type_text
            bundles["emul_type"]["value_text"] = emul_type_text
            bundles["emul_type"]["membership_confidence"] = "projected_derived"
            bundles["emul_type"]["evidence_region_type"] = region
            add_trace(traces, document_key, formulation_id, "emul_type", phase_refs, DERIVED, DERIVED, "Derived from phase candidates and factor hints.")
        else:
            bundles["emul_type"]["missing_reason"] = "not_projectable_from_current_replacement_objects"
            add_trace(traces, document_key, formulation_id, "emul_type", [], UNAVAILABLE, UNAVAILABLE, "No phase or factor cues available.")

        for target_field, aliases in MEASUREMENT_ALIASES.items():
            matched = [
                item
                for item in owned_measurements
                if any(alias in measurement_target_name(item) for alias in aliases)
            ]
            value, value_text, status = project_choice(
                matched,
                lambda item: first_number(normalize_text(item.get("measurement_value_raw"))),
                lambda item: " ".join(
                    part
                    for part in [
                        normalize_text(item.get("measurement_value_raw")),
                        normalize_text(item.get("measurement_unit_raw")),
                    ]
                    if part
                ),
            )
            assign_bundle(target_field, value, value_text, status, [normalize_text(item.get("measurement_id")) for item in matched], "Projected from measurement candidates.")

        for field, bundle in bundles.items():
            row[f"{field}_value"] = bundle["value"]
            row[f"{field}_value_text"] = bundle["value_text"]
            row[f"{field}_membership_confidence"] = bundle["membership_confidence"]
            row[f"{field}_evidence_region_type"] = bundle["evidence_region_type"]
            row[f"{field}_missing_reason"] = bundle["missing_reason"]

        rows.append(row)
        jsonl_rows.append({"key": document_key, "doi": doi, "formulation_id": formulation_id, "legacy_row": row})

    semantic_scope = resolve_llm_declared_doe_scope(document)
    recovered_rows, recovered_traces, recovered_jsonl_rows, recovery_summary = run_doe_row_expansion_function_unit(
        document=document,
        model_name=model_name,
        semantic_scope=semantic_scope,
    )
    if recovered_rows:
        rows.extend(recovered_rows)
        traces.extend(recovered_traces)
        jsonl_rows.extend(recovered_jsonl_rows)
        add_trace(
            traces,
            document_key,
            "__recovery__",
            "numbered_doe_row_recovery",
            [normalize_text(item.get("formulation_id")) for item in recovered_jsonl_rows if normalize_text(item.get("formulation_id"))],
            DIRECT,
            DIRECT,
            f"Recovered {recovery_summary.get('candidate_count', 0)} explicit DOE rows from numbered table anchors.",
        )

    table_rows, table_traces, table_jsonl_rows, table_summary = run_table_row_expansion(
        document=document,
        compatibility_columns=compatibility_output_columns(),
        doe_summary=recovery_summary,
    )
    if table_rows:
        rows.extend(table_rows)
        traces.extend(table_traces)
        jsonl_rows.extend(table_jsonl_rows)
        group_hint = normalize_text(table_summary.get("group_hint"))
        if group_hint:
            mark_llm_summary_rows_as_helpers(rows, jsonl_rows, group_hint)

    sequential_summary = {
        "function_unit": SEQUENTIAL_OPTIMIZATION_FUNCTION_UNIT_ID,
        "document_key": document_key,
        "considered": True,
        "authorized": False,
        "called": False,
        "emitted_row_count": 0,
        "retained_row_count": 0,
        "skip_reason": "blocked_by_table_row_expansion" if table_rows else "not_invoked",
        "status": "skipped_due_to_table_row_expansion" if table_rows else "not_invoked",
        "replaced_row_count": 0,
    }
    if not table_rows:
        sequential_rows, sequential_traces, sequential_jsonl_rows, sequential_summary = run_sequential_optimization_interpreter(
            document=document,
            existing_rows=rows,
        )
        if sequential_rows:
            rows.extend(sequential_rows)
            traces.extend(sequential_traces)
            jsonl_rows.extend(sequential_jsonl_rows)
            rows, suppressed_rows = prefer_resolved_sequential_rows(rows)
            if suppressed_rows:
                suppressed_ids = [
                    normalize_text(row.get("formulation_id") or row.get("local_instance_id"))
                    for row in suppressed_rows
                    if normalize_text(row.get("formulation_id") or row.get("local_instance_id"))
                ]
                add_trace(
                    traces,
                    document_key,
                    "__recovery__",
                    "sequential_optimization_overlap_suppression",
                    suppressed_ids,
                    DIRECT,
                    DIRECT,
                    f"Suppressed {len(suppressed_ids)} overlapping LLM rows in favor of governed sequential optimization resolution.",
                )
                kept = {normalize_text(row.get("formulation_id")) for row in rows if normalize_text(row.get("formulation_id"))}
                jsonl_rows = [
                    item
                    for item in jsonl_rows
                    if normalize_text(item.get("formulation_id")) in kept
                ]

    recovery_summary["retained_row_count"] = sum(
        1 for row in rows if is_governed_doe_recovery_candidate_source(normalize_text(row.get("candidate_source")))
    )
    table_summary["retained_row_count"] = sum(
        1 for row in rows if normalize_text(row.get("candidate_source")) == "table_row_expansion_v1"
    )
    sequential_summary = {
        "function_unit": normalize_text(sequential_summary.get("function_unit")) or SEQUENTIAL_OPTIMIZATION_FUNCTION_UNIT_ID,
        "document_key": document_key,
        "considered": bool(sequential_summary.get("considered", True)),
        "authorized": bool(
            sequential_summary.get("authorized")
            if "authorized" in sequential_summary
            else sequential_summary.get("triggered", False)
        ),
        "called": bool(
            sequential_summary.get("called")
            if "called" in sequential_summary
            else sequential_summary.get("triggered", False)
        ),
        "emitted_row_count": int(
            sequential_summary.get("emitted_row_count")
            or sequential_summary.get("candidate_count")
            or 0
        ),
        "retained_row_count": sum(
            1 for row in rows if normalize_text(row.get("candidate_source")) == SEQUENTIAL_OPTIMIZATION_CANDIDATE_SOURCE
        ),
        "skip_reason": normalize_text(sequential_summary.get("skip_reason") or sequential_summary.get("notes")),
        "status": normalize_text(sequential_summary.get("status"))
        or ("emitted_rows" if sequential_summary.get("materialized") else "no_rows_emitted"),
        "replaced_row_count": int(sequential_summary.get("replaced_row_count") or 0),
        "semantic_scope_ref": normalize_text(sequential_summary.get("semantic_scope_ref")),
        "notes": normalize_text(sequential_summary.get("notes")),
    }

    rows, suppressed_rows = prefer_governed_doe_rows_over_llm_numeric_rows(rows)
    if suppressed_rows:
        suppressed_ids = [
            normalize_text(row.get("formulation_id") or row.get("local_instance_id"))
            for row in suppressed_rows
            if normalize_text(row.get("formulation_id") or row.get("local_instance_id"))
        ]
        add_trace(
            traces,
            document_key,
            "__recovery__",
            "numbered_doe_numeric_overlap_suppression",
            suppressed_ids,
            DIRECT,
            DIRECT,
            f"Suppressed {len(suppressed_ids)} overlapping numeric LLM rows in favor of governed DOE recovery rows.",
        )
        kept = {normalize_text(row.get("formulation_id")) for row in rows if normalize_text(row.get("formulation_id"))}
        jsonl_rows = [
            item
            for item in jsonl_rows
            if normalize_text(item.get("formulation_id")) in kept
        ]

    guard_row = None
    if recovery_summary.get("enabled"):
        guard_row = build_governed_numbered_doe_guard_row(
            document=document,
            rows=rows,
            recovery_summary=recovery_summary,
        )

    for field in CORE_FIELDS:
        values = {normalize_text(row.get(f"{field}_value_text")) for row in rows if normalize_text(row.get(f"{field}_value_text"))}
        scope = "global_shared" if field in SHARED_SCOPE_FIELDS and len(values) == 1 and len(rows) > 1 else "instance_specific"
        for row in rows:
            row[f"{field}_scope"] = scope if normalize_text(row.get(f"{field}_value_text")) else ""

    summary = {
        "document_key": document_key,
        "authority_reattachment_summary": reattachment_summary,
        "doe_recovery_summary": recovery_summary,
        "sequential_optimization_summary": sequential_summary,
        "table_row_expansion_summary": table_summary,
    }
    return rows, traces, jsonl_rows, summary, guard_row


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_function_unit_activation_rows(projection_summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in projection_summaries:
        document_key = normalize_text(summary.get("document_key"))
        for unit_key in [
            "doe_recovery_summary",
            "table_row_expansion_summary",
            "sequential_optimization_summary",
        ]:
            unit_summary = summary.get(unit_key)
            if not isinstance(unit_summary, dict):
                continue
            rows.append(
                {
                    "document_key": document_key or normalize_text(unit_summary.get("document_key")),
                    "function_unit": normalize_text(unit_summary.get("function_unit")),
                    "was_unit_considered": "yes" if unit_summary.get("considered") else "no",
                    "was_unit_authorized": "yes" if unit_summary.get("authorized") else "no",
                    "was_unit_called": "yes" if unit_summary.get("called") else "no",
                    "rows_emitted": str(int(unit_summary.get("emitted_row_count") or 0)),
                    "rows_retained_after_projection": str(int(unit_summary.get("retained_row_count") or 0)),
                    "skip_reason": normalize_text(unit_summary.get("skip_reason")),
                    "status": normalize_text(unit_summary.get("status")),
                }
            )
    return rows


def build_execution_ledger_rows(projection_summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in projection_summaries:
        table_summary = summary.get("table_row_expansion_summary")
        if not isinstance(table_summary, dict):
            continue
        for activation_row in ensure_list(table_summary.get("table_activation_rows")):
            if not isinstance(activation_row, dict):
                continue
            rows.append(
                {
                    "document_key": normalize_text(activation_row.get("document_key")),
                    "function_unit": normalize_text(activation_row.get("function_unit")),
                    "table_id": normalize_text(activation_row.get("table_id")),
                    "scope_id": normalize_text(activation_row.get("scope_id")),
                    "table_type": normalize_text(activation_row.get("table_type")),
                    "marker_provenance": normalize_text(activation_row.get("marker_provenance")),
                    "was_unit_considered": normalize_text(activation_row.get("considered")),
                    "was_unit_authorized": normalize_text(activation_row.get("authorized")),
                    "was_unit_called": normalize_text(activation_row.get("called")),
                    "rows_emitted": normalize_text(activation_row.get("rows_emitted")),
                    "rows_retained_after_projection": normalize_text(activation_row.get("rows_retained_after_projection")),
                    "varying_variable_count": normalize_text(activation_row.get("varying_variable_count")),
                    "varying_variables": normalize_text(activation_row.get("varying_variables")),
                    "table_path": normalize_text(activation_row.get("table_path")),
                    "doe_path_attempted": normalize_text(activation_row.get("doe_path_attempted")),
                    "doe_rows_emitted": normalize_text(activation_row.get("doe_rows_emitted")),
                    "fell_back_to_table_expansion": normalize_text(activation_row.get("fell_back_to_table_expansion")),
                    "fallback_reason": normalize_text(activation_row.get("fallback_reason")),
                    "explicit_table_rows_emitted": normalize_text(activation_row.get("explicit_table_rows_emitted")),
                    "non_doe_single_variable_groups_detected": normalize_text(activation_row.get("non_doe_single_variable_groups_detected")),
                    "single_variable_recovery_attempted": normalize_text(activation_row.get("single_variable_recovery_attempted")),
                    "single_variable_rows_emitted": normalize_text(activation_row.get("single_variable_rows_emitted")),
                    "single_variable_recovery_source_type": normalize_text(activation_row.get("single_variable_recovery_source_type")),
                    "single_variable_recovery_failure_reason": normalize_text(activation_row.get("single_variable_recovery_failure_reason")),
                    "held_constant_context_source": normalize_text(activation_row.get("held_constant_context_source")),
                    "variable_axis_detected": normalize_text(activation_row.get("variable_axis_detected")),
                    "skip_reason": normalize_text(activation_row.get("skip_reason")),
                }
            )
    return rows


def build_projection_contract_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        [
            {"legacy_field": "formulation_id", "replacement_object_type": "formulation_identity_candidate", "replacement_field_or_rule": "formulation_candidate_id", "projection_status": "direct", "direct_or_derived": "direct", "notes": "Transitional deterministic projection."},
            {"legacy_field": "raw_formulation_label", "replacement_object_type": "formulation_identity_candidate", "replacement_field_or_rule": "raw_formulation_label", "projection_status": "direct", "direct_or_derived": "direct", "notes": "Transitional deterministic projection."},
            {"legacy_field": "parent_instance_id", "replacement_object_type": "formulation_identity_candidate", "replacement_field_or_rule": "parent_candidate_id", "projection_status": "direct", "direct_or_derived": "direct", "notes": "Transitional deterministic projection."},
            {"legacy_field": "instance_kind", "replacement_object_type": "formulation_identity_candidate", "replacement_field_or_rule": "instance_kind", "projection_status": "direct", "direct_or_derived": "direct", "notes": "Transitional deterministic projection."},
            {"legacy_field": "polymer_identity", "replacement_object_type": "component_candidate", "replacement_field_or_rule": "infer family from polymer component names", "projection_status": "derived", "direct_or_derived": "derived", "notes": "No new LLM arbitration added."},
            {"legacy_field": "polymer_name_raw", "replacement_object_type": "component_candidate", "replacement_field_or_rule": "component_name_raw where role=polymer", "projection_status": "direct_or_compressed", "direct_or_derived": "mixed", "notes": "Multiple polymer components are compressed with delimiter pipes."},
            {"legacy_field": "la_ga_ratio", "replacement_object_type": "component_candidate", "replacement_field_or_rule": "component_properties_raw", "projection_status": "direct_or_unavailable", "direct_or_derived": "mixed", "notes": "Family-specific property is only projected when available."},
            {"legacy_field": "polymer_mw_kDa", "replacement_object_type": "component_candidate", "replacement_field_or_rule": "component_properties_raw.molecular_weight", "projection_status": "direct_or_unavailable", "direct_or_derived": "mixed", "notes": "Uses generic polymer MW properties."},
            {"legacy_field": "plga_mass_mg", "replacement_object_type": "component_candidate", "replacement_field_or_rule": "amount_expression_raw or parsed_value_raw", "projection_status": "direct_or_unavailable", "direct_or_derived": "mixed", "notes": "Legacy naming retained only for downstream compatibility."},
            {"legacy_field": "surfactant_name", "replacement_object_type": "component_candidate", "replacement_field_or_rule": "component_name_raw where role=surfactant/stabilizer", "projection_status": "direct_or_compressed", "direct_or_derived": "mixed", "notes": "Multiple stabilizers are compressed."},
            {"legacy_field": "organic_solvent", "replacement_object_type": "component_candidate", "replacement_field_or_rule": "component_name_raw where role=organic_solvent", "projection_status": "direct_or_compressed", "direct_or_derived": "mixed", "notes": "Supports co-solvent compression during transition."},
            {"legacy_field": "drug_name", "replacement_object_type": "component_candidate", "replacement_field_or_rule": "component_name_raw where role=drug", "projection_status": "direct_or_compressed", "direct_or_derived": "mixed", "notes": "Multiple payloads are compressed if present."},
            {"legacy_field": "emul_method", "replacement_object_type": "process_step_candidate", "replacement_field_or_rule": "process_name_raw", "projection_status": "direct_or_compressed", "direct_or_derived": "mixed", "notes": "General process semantics are projected into the legacy slot."},
            {"legacy_field": "emul_type", "replacement_object_type": "phase_candidate + variable_or_factor_candidate", "replacement_field_or_rule": "derive from phase codes or factor expressions", "projection_status": "derived_or_unavailable", "direct_or_derived": "derived", "notes": "No hidden inference beyond simple deterministic rules."},
            {"legacy_field": "preparation_method", "replacement_object_type": "process_step_candidate", "replacement_field_or_rule": "process_name_raw", "projection_status": "direct_or_compressed", "direct_or_derived": "mixed", "notes": "Transition-friendly generalized process field."},
            {"legacy_field": "emulsion_structure", "replacement_object_type": "phase_candidate", "replacement_field_or_rule": "phase_code_raw", "projection_status": "direct_or_compressed", "direct_or_derived": "mixed", "notes": "Phase-aware field remains supported."},
            {"legacy_field": "size_nm/pdi/zeta_mV/encapsulation_efficiency_percent/loading_content_percent", "replacement_object_type": "measurement_candidate", "replacement_field_or_rule": "measurement_name_raw + measurement_value_raw + measurement_unit_raw", "projection_status": "direct_or_compressed", "direct_or_derived": "mixed", "notes": "Measurements map by deterministic name matching."},
            {"legacy_field": "supporting_evidence_refs", "replacement_object_type": "evidence_handoff", "replacement_field_or_rule": "source_locator_text/supporting_snippet", "projection_status": "coarse_direct_or_unavailable", "direct_or_derived": "direct", "notes": "Coarse evidence handoff only."},
            {"legacy_field": "instance_evidence_region_type/evidence_section/evidence_span_text", "replacement_object_type": "evidence_handoff", "replacement_field_or_rule": "source_region_type/source_locator_text/supporting_snippet", "projection_status": "coarse_direct_or_unavailable", "direct_or_derived": "direct", "notes": "Not audit-grade ownership binding."},
            {"legacy_field": "identity_variables_json", "replacement_object_type": "variable_or_factor_candidate", "replacement_field_or_rule": "preserve normalized factor_name_raw + factor_expression_raw for identity_defining_signal=yes only", "projection_status": "direct_or_unavailable", "direct_or_derived": "direct", "notes": "Additive metadata carrier for downstream identity preservation without changing legacy field bundles."},
            {"legacy_field": "*_scope", "replacement_object_type": "all projected row values", "replacement_field_or_rule": "derive per-document shared vs instance-specific status", "projection_status": "derived", "direct_or_derived": "derived", "notes": "Only a transitional compatibility hint."},
            {"legacy_field": "*_membership_confidence", "replacement_object_type": "projection engine", "replacement_field_or_rule": "projected_direct/projected_compressed/projected_derived", "projection_status": "derived", "direct_or_derived": "derived", "notes": "Does not reintroduce field-level LLM confidence."},
            {"legacy_field": "*_missing_reason", "replacement_object_type": "projection engine", "replacement_field_or_rule": "set when deterministic projection is unavailable", "projection_status": "derived", "direct_or_derived": "derived", "notes": "Audit-friendly transitional metadata."},
        ]
    )
    return rows


def write_projection_contract(path: Path) -> None:
    write_tsv(
        path,
        build_projection_contract_rows(),
        [
            "legacy_field",
            "replacement_object_type",
            "replacement_field_or_rule",
            "projection_status",
            "direct_or_derived",
            "notes",
        ],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project semantic-object Stage2 outputs into the legacy wide-row Stage2 surface."
    )
    parser.add_argument("--input-jsonl", default="", help="Semantic-object Stage2 JSONL input.")
    parser.add_argument("--output-dir", default="", help="Directory for projected compatibility outputs.")
    parser.add_argument("--write-contract-only", action="store_true", help="Write the projection contract TSV and exit.")
    parser.add_argument("--contract-out", default="", help="Optional explicit path for the projection contract TSV.")
    return parser


def run_projection(
    *,
    input_path: Path,
    output_dir: Path,
    contract_path: Path,
    authority_sidecar_path: Path | None = None,
) -> dict[str, Any]:
    documents = load_jsonl_documents(input_path)
    authority_sidecar = load_authority_sidecar(authority_sidecar_path)
    all_rows: list[dict[str, str]] = []
    all_traces: list[dict[str, str]] = []
    all_jsonl_rows: list[dict[str, Any]] = []
    projection_summaries: list[dict[str, Any]] = []
    guard_rows: list[dict[str, str]] = []
    for document in documents:
        document_key = normalize_text(document.get("document_key") or document.get("key"))
        rows, traces, jsonl_rows, projection_summary, guard_row = project_document(
            document,
            authority_sidecar_entry=authority_sidecar.get(document_key),
        )
        all_rows.extend(rows)
        all_traces.extend(traces)
        all_jsonl_rows.extend(jsonl_rows)
        projection_summaries.append(projection_summary)
        if guard_row is not None:
            guard_rows.append(guard_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / LEGACY_TSV_NAME, all_rows, compatibility_output_columns())
    with (output_dir / LEGACY_JSONL_NAME).open("w", encoding="utf-8") as handle:
        for row in all_jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_tsv(
        output_dir / TRACE_TSV_NAME,
        all_traces,
        [
            "document_key",
            "formulation_id",
            "legacy_field",
            "source_replacement_objects",
            "mapping_status",
            "direct_or_derived",
            "notes",
        ],
    )
    write_projection_contract(contract_path)
    guard_stats = write_numbered_doe_guard_artifact(output_dir, guard_rows)
    function_unit_activation_rows = build_function_unit_activation_rows(projection_summaries)
    execution_ledger_rows = build_execution_ledger_rows(projection_summaries)
    write_tsv(
        output_dir / FUNCTION_UNIT_ACTIVATION_NAME,
        function_unit_activation_rows,
        [
            "document_key",
            "function_unit",
            "was_unit_considered",
            "was_unit_authorized",
            "was_unit_called",
            "rows_emitted",
            "rows_retained_after_projection",
            "skip_reason",
            "status",
        ],
    )
    write_tsv(
        output_dir / EXECUTION_LEDGER_NAME,
        execution_ledger_rows,
        [
            "document_key",
            "function_unit",
            "table_id",
            "scope_id",
            "table_type",
            "marker_provenance",
            "was_unit_considered",
            "was_unit_authorized",
            "was_unit_called",
            "rows_emitted",
            "rows_retained_after_projection",
            "varying_variable_count",
            "varying_variables",
            "table_path",
            "doe_path_attempted",
            "doe_rows_emitted",
            "fell_back_to_table_expansion",
            "fallback_reason",
            "explicit_table_rows_emitted",
            "non_doe_single_variable_groups_detected",
            "single_variable_recovery_attempted",
            "single_variable_rows_emitted",
            "single_variable_recovery_source_type",
            "single_variable_recovery_failure_reason",
            "held_constant_context_source",
            "variable_axis_detected",
            "skip_reason",
        ],
    )
    summary = {
        "schema": "stage2_replacement_compatibility_projection_v1",
        "status": "transitional_support",
        "documents": len(documents),
        "projected_rows": len(all_rows),
        "stage2_semantic_source_modes": sorted(
            {
                stage2_semantic_source_mode(document)
                for document in documents
            }
        ),
        "numbered_doe_recovery_enabled": numbered_doe_recovery_enabled(),
        "doe_enumeration_mode": doe_enumeration_mode(),
        "numbered_doe_recovery_min_rows": numbered_doe_recovery_min_rows(),
        "numbered_doe_recovered_rows": sum(
            1 for row in all_rows if is_governed_doe_recovery_candidate_source(normalize_text(row.get("candidate_source")))
        ),
        "numbered_doe_function_unit": FUNCTION_UNIT_ID,
        "sequential_optimization_resolved_rows": sum(
            1 for row in all_rows if normalize_text(row.get("candidate_source")) == SEQUENTIAL_OPTIMIZATION_CANDIDATE_SOURCE
        ),
        "sequential_optimization_function_unit": SEQUENTIAL_OPTIMIZATION_FUNCTION_UNIT_ID,
        "table_row_expansion_rows": sum(
            1 for row in all_rows if normalize_text(row.get("candidate_source")) == "table_row_expansion_v1"
        ),
        "table_row_expansion_function_unit": "table_row_expansion_v1",
        "function_unit_activation_rows": function_unit_activation_rows,
        "execution_ledger_rows": execution_ledger_rows,
        "authority_sidecar_path": str(authority_sidecar_path.resolve()) if authority_sidecar_path is not None and authority_sidecar_path.exists() else "",
        "authority_sidecar_entries": len(authority_sidecar),
        "numbered_doe_guard_tsv": str(Path(guard_stats["guard_path"]).resolve()),
        "numbered_doe_guard_fail_count": int(guard_stats["fail_count"]),
        "numbered_doe_guard_warn_count": int(guard_stats["warn_count"]),
        "projection_summaries": projection_summaries,
        "trace_rows": len(all_traces),
        "legacy_surface_columns": len(compatibility_output_columns()),
        "output_files": [
            str(output_dir / LEGACY_TSV_NAME),
            str(output_dir / LEGACY_JSONL_NAME),
            str(output_dir / TRACE_TSV_NAME),
            str(output_dir / FUNCTION_UNIT_ACTIVATION_NAME),
            str(output_dir / EXECUTION_LEDGER_NAME),
            str(Path(guard_stats["guard_path"]).resolve()),
            str(contract_path),
        ],
    }
    (output_dir / SUMMARY_JSON_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    contract_path = Path(args.contract_out) if args.contract_out else Path("data/db/db_v2") / CONTRACT_TSV_NAME
    if args.write_contract_only:
        write_projection_contract(contract_path)
        print(f"[ok] wrote projection contract -> {contract_path}")
        return

    if not args.input_jsonl or not args.output_dir:
        parser.error("--input-jsonl and --output-dir are required unless --write-contract-only is used.")

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    summary = run_projection(
        input_path=input_path,
        output_dir=output_dir,
        contract_path=contract_path,
    )
    print(f"[ok] projected {summary['projected_rows']} rows from {summary['documents']} document payload(s) -> {output_dir}")


if __name__ == "__main__":
    main()
