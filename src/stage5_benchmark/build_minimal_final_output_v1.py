#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.preparation_method_fields_v1 import PREPARATION_METHOD_FIELDNAMES


DECISION_TRACE_NAME = "final_output_decision_trace_v1.tsv"
FINAL_TABLE_NAME = "final_formulation_table_v1.tsv"
SUMMARY_NAME = "final_output_summary_v1.md"
RELATION_RECORDS_NAME = "formulation_relation_records_v1.tsv"
RESOLVED_RELATION_FIELDS_NAME = "resolved_relation_fields_v1.tsv"
RESOLVED_RELATION_FIELD_NAMES = {
    "polymer_mw_kDa",
    "surfactant_name",
    "organic_solvent",
    "preparation_method",
}

LEGACY_FIELD_ALIASES = {
    "plga_mw_kDa": "polymer_mw_kDa",
}


@dataclass(frozen=True)
class RowDecision:
    decision: str
    target_final_formulation_id: str
    variant_class: str
    variant_signal: str
    equivalence_group_id: str
    family_id: str
    parent_core_row_id: str
    variant_role: str
    payload_state: str
    benchmark_default_include: str
    decision_rule: str
    decision_reason: str
    retention_reason: str
    collapse_reason: str
    review_needed: str
    key_fields_used: str
    confidence_or_rule_scope: str
    notes: str


def row_source_key(row: dict[str, str]) -> str:
    return f"{row.get('key', '').strip()}::{row.get('formulation_id', '').strip()}"


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_token(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9%:/.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def canonical_field_name(field_name: Any) -> str:
    return LEGACY_FIELD_ALIASES.get(str(field_name or "").strip(), str(field_name or "").strip())


def canonicalize_row_columns(row: dict[str, str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for key, value in row.items():
        target_key = str(key)
        for legacy_name, canonical_name in LEGACY_FIELD_ALIASES.items():
            if target_key == legacy_name:
                target_key = canonical_name
                break
            if target_key.startswith(f"{legacy_name}_"):
                target_key = f"{canonical_name}_{target_key[len(legacy_name) + 1:]}"
                break
        if target_key not in canonical or not str(canonical.get(target_key, "")).strip():
            canonical[target_key] = value
    return canonical


def first_number_token(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return ""
    token = match.group(0)
    try:
        num = float(token)
    except ValueError:
        return token
    if num.is_integer():
        return str(int(num))
    return f"{num:.6g}"


def normalize_ratio(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    compact = text.replace(" ", "")
    match = re.match(r"^(\d{1,3})[:/](\d{1,3})$", compact)
    if match:
        return f"{int(match.group(1))}:{int(match.group(2))}"
    return compact


def parse_json_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def extract_paper_local_row_anchor(row: dict[str, str]) -> str:
    candidate_tokens = [
        str(row.get("formulation_id", "") or "").strip(),
        str(row.get("raw_formulation_label", "") or "").strip(),
    ]
    patterns = [
        re.compile(r"doe_row_(\d+)$", re.IGNORECASE),
        re.compile(r"^f[\s_-]*(\d+)\b", re.IGNORECASE),
        re.compile(r"^(\d+)\.?\b"),
    ]
    for token in candidate_tokens:
        for pattern in patterns:
            match = pattern.search(token)
            if match:
                return str(int(match.group(1)))
    return ""


def infer_loaded_state(row: dict[str, str]) -> str:
    raw_bundle = " ".join(
        [
            row.get("raw_formulation_label", ""),
            row.get("drug_name_value", ""),
            row.get("drug_feed_amount_text_value", ""),
        ]
    ).lower()
    if "drug free" in raw_bundle or "empty" in raw_bundle:
        return "empty"
    if row.get("drug_feed_amount_text_value") or row.get("drug_name_value"):
        return "drug_loaded"
    return "unknown"


def infer_polymer_identity(row: dict[str, str]) -> str:
    polymer = normalize_text(row.get("polymer_identity", ""))
    if polymer and polymer != "unknown":
        return polymer.upper()
    raw_bundle = " ".join(
        [
            row.get("polymer_name_raw", ""),
            row.get("raw_formulation_label", ""),
            row.get("la_ga_ratio_value", ""),
        ]
    ).lower()
    if "peg-plga" in raw_bundle or "plga-peg" in raw_bundle:
        return "PEG-PLGA"
    if "plga" in raw_bundle or row.get("la_ga_ratio_value"):
        return "PLGA"
    if "pcl" in raw_bundle:
        return "PCL"
    if "pla" in raw_bundle:
        return "PLA"
    return "unknown"


def normalize_surfactant_concentration(row: dict[str, str]) -> str:
    surf = first_number_token(row.get("surfactant_concentration_text_value"))
    if surf:
        return surf
    return first_number_token(row.get("pva_conc_percent_value"))


def build_core_fields(row: dict[str, str]) -> dict[str, str]:
    return {
        "polymer_identity": infer_polymer_identity(row),
        "polymer_name_raw": str(row.get("polymer_name_raw", "") or "").strip(),
        "la_ga_ratio": normalize_ratio(row.get("la_ga_ratio_value")),
        "loaded_state": infer_loaded_state(row),
        "drug_name": normalize_token(row.get("drug_name_value")),
        "drug_feed_amount_mg": first_number_token(row.get("drug_feed_amount_text_value")),
        "polymer_amount_mg": first_number_token(row.get("plga_mass_mg_value")),
        "surfactant_name": normalize_token(row.get("surfactant_name_value")),
        "surfactant_concentration": normalize_surfactant_concentration(row),
        "organic_solvent": normalize_token(row.get("organic_solvent_value")),
    }


def build_key_fields_used(core_fields: dict[str, str]) -> str:
    return json.dumps(core_fields, ensure_ascii=True, sort_keys=True)


def has_context_tag(row: dict[str, str], target_tags: set[str]) -> bool:
    observed = row_context_tags(row)
    return not observed.isdisjoint(target_tags)


def row_context_tags(row: dict[str, str]) -> set[str]:
    observed = {
        normalize_text(tag)
        for tag in parse_json_list(row.get("instance_context_tags", "[]"))
        + parse_json_list(row.get("change_context_tags", "[]"))
    }
    return observed


def has_commercial_reference_signal(row: dict[str, str]) -> bool:
    tags = row_context_tags(row)
    if "commercial" in tags:
        return True
    signal_blob = " ".join(
        [
            str(row.get("raw_formulation_label", "") or ""),
            str(row.get("evidence_span_text", "") or ""),
            str(row.get("supporting_evidence_refs", "") or ""),
        ]
    ).lower()
    if any(
        phrase in signal_blob
        for phrase in [
            "commercial product",
            "commercial formulation",
            "commercial intravenous formulation",
            "marketed product",
            "marketed formulation",
            "marketed drug product",
            "former commercial",
        ]
    ):
        return True
    for key, value in row.items():
        if not key.endswith("_missing_reason"):
            continue
        if "commercial product" in normalize_text(value):
            return True
    return False


def lacks_internal_preparation_identity(core_fields: dict[str, str]) -> bool:
    if core_fields["polymer_identity"] != "unknown":
        return False
    internal_identity_fields = [
        "la_ga_ratio",
        "drug_feed_amount_mg",
        "polymer_amount_mg",
        "surfactant_name",
        "surfactant_concentration",
        "organic_solvent",
    ]
    return not any(core_fields[field_name] for field_name in internal_identity_fields)


def should_filter_non_formulation(
    row: dict[str, str], core_fields: dict[str, str]
) -> tuple[bool, str, str]:
    if normalize_text(row.get("instance_kind")) == "candidate_non_formulation":
        return (
            True,
            "explicit_candidate_non_formulation",
            "Stage2 explicitly marked this row as candidate_non_formulation.",
        )

    if (
        normalize_text(row.get("formulation_role")) == "characterization_only"
        and normalize_text(row.get("change_role")) == "non_synthesis"
        and has_context_tag(row, {"post_processing", "measurement_context"})
    ):
        return (
            True,
            "characterization_only_post_processing",
            "Row is tagged as post-processing or measurement context only and does not describe a new formulation closure case.",
        )

    if (
        normalize_text(row.get("formulation_role")) == "comparative"
        and has_commercial_reference_signal(row)
        and lacks_internal_preparation_identity(core_fields)
    ):
        return (
            True,
            "external_commercial_reference",
            "Row is a commercial or marketed comparator reference without internal preparation identity and is excluded from benchmark-facing formulation closure.",
        )

    return False, "", ""


def collapse_exclusion_reason(
    row: dict[str, str], core_fields: dict[str, str], allow_context_tags: bool = False
) -> str:
    if normalize_text(row.get("instance_kind")) not in {
        "new_formulation",
        "variant_formulation",
    }:
        return "instance_kind_not_final_output_candidate"
    if core_fields["polymer_identity"] == "unknown":
        return "polymer_identity_unknown"
    if core_fields["loaded_state"] == "unknown":
        return "loaded_state_unknown"
    if (
        not allow_context_tags
        and has_context_tag(row, {"doe", "checkpoint_validation", "center_point", "post_processing"})
    ):
        return "context_tag_excluded_in_phase1"
    completeness = sum(
        1
        for field_name in [
            "polymer_identity",
            "loaded_state",
            "la_ga_ratio",
            "drug_feed_amount_mg",
            "polymer_amount_mg",
            "surfactant_name",
            "surfactant_concentration",
            "organic_solvent",
        ]
        if core_fields[field_name]
    )
    if completeness < 5:
        return "insufficient_core_signature_completeness"
    return ""


def build_collapse_signature(row: dict[str, str], core_fields: dict[str, str]) -> str:
    signature_parts = [
        row.get("key", "").strip(),
        core_fields["polymer_identity"],
        core_fields["la_ga_ratio"],
        core_fields["loaded_state"],
        core_fields["drug_name"],
        core_fields["drug_feed_amount_mg"],
        core_fields["polymer_amount_mg"],
        core_fields["surfactant_name"],
        core_fields["surfactant_concentration"],
        core_fields["organic_solvent"],
    ]
    return "|".join(signature_parts)


def variant_signal_class(row: dict[str, str]) -> str:
    tags = row_context_tags(row)
    if "checkpoint_validation" in tags or "center_point" in tags:
        return "checkpoint_or_validation_variant"
    if (
        "post_processing" in tags
        or "measurement_context" in tags
        or normalize_text(row.get("formulation_role")) == "characterization_only"
    ):
        return "post_processing_or_measurement_variant"
    if "optimized" in tags:
        return "optimized_variant"
    return ""


def is_parent_linked_family_variant(row: dict[str, str]) -> bool:
    return normalize_text(row.get("formulation_role")) in {"characterization_only", "control"} and bool(
        str(row.get("parent_instance_id", "") or "").strip()
    )


def infer_payload_state(row: dict[str, str], core_fields: dict[str, str]) -> str:
    label = normalize_text(row.get("raw_formulation_label"))
    drug_name = normalize_token(row.get("drug_name_value"))
    if "blank" in label or core_fields["loaded_state"] in {"empty", "unknown"} and not drug_name:
        return "blank_control"
    if drug_name == "fitc":
        return "fitc_assay_loaded"
    if core_fields["loaded_state"] == "drug_loaded":
        return "drug_loaded"
    if core_fields["loaded_state"]:
        return core_fields["loaded_state"]
    return "unknown"


def compute_family_labels(
    row: dict[str, str],
    core_fields: dict[str, str],
) -> dict[str, str]:
    key = str(row.get("key", "") or "").strip()
    formulation_id = str(row.get("formulation_id", "") or "").strip()
    parent_instance_id = str(row.get("parent_instance_id", "") or "").strip()
    family_core_id = parent_instance_id if is_parent_linked_family_variant(row) else formulation_id
    variant_role = "true_family_variant" if is_parent_linked_family_variant(row) else "family_core"
    payload_state = infer_payload_state(row, core_fields)
    benchmark_default_include = (
        "yes"
        if variant_role == "family_core" and payload_state == "drug_loaded"
        else "no"
    )
    return {
        "family_id": f"{key}::{family_core_id}" if key and family_core_id else "",
        "parent_core_row_id": family_core_id,
        "variant_role": variant_role,
        "payload_state": payload_state,
        "benchmark_default_include": benchmark_default_include,
    }


def is_non_doe_sweep_row(row: dict[str, str]) -> bool:
    tags = row_context_tags(row)
    return "doe" not in tags and "sweep" not in tags


def populated_core_field_count(core_fields: dict[str, str]) -> int:
    return sum(1 for value in core_fields.values() if value and value != "unknown")


def is_structured_duplicate_representation_row(
    row: dict[str, str], core_fields: dict[str, str]
) -> bool:
    if normalize_text(row.get("candidate_source")) != "doe_numbered_table_row":
        return False
    if not has_context_tag(row, {"doe", "numbered_table_row"}):
        return False
    return (
        core_fields["polymer_identity"] == "unknown"
        or core_fields["loaded_state"] == "unknown"
        or populated_core_field_count(core_fields) <= 3
    )


def can_receive_structured_duplicate_collapse(
    row: dict[str, str], core_fields: dict[str, str]
) -> bool:
    if normalize_text(row.get("instance_kind")) not in {
        "new_formulation",
        "variant_formulation",
    }:
        return False
    if normalize_text(row.get("candidate_source")) == "doe_numbered_table_row":
        return False
    return (
        core_fields["polymer_identity"] != "unknown"
        and core_fields["loaded_state"] != "unknown"
        and populated_core_field_count(core_fields) >= 5
    )


def candidate_priority(row: dict[str, str]) -> int:
    source = normalize_text(row.get("candidate_source"))
    if source == "llm_extracted":
        return 3
    if source == "figure_variable_sweep":
        return 2
    return 1


def confidence_priority(row: dict[str, str]) -> int:
    confidence = normalize_text(row.get("instance_confidence"))
    if confidence == "high":
        return 3
    if confidence == "medium":
        return 2
    if confidence == "low":
        return 1
    return 0


def choose_representative(
    group_rows: list[dict[str, str]],
    core_by_source_id: dict[str, dict[str, str]],
) -> dict[str, str]:
    def sort_key(row: dict[str, str]) -> tuple[int, int, int, int, str]:
        core_fields = core_by_source_id[row_source_key(row)]
        return (
            candidate_priority(row),
            confidence_priority(row),
            populated_core_field_count(core_fields),
            len(str(row.get("evidence_span_text", "") or "")),
            str(row.get("formulation_id", "")),
        )

    return max(group_rows, key=sort_key)


def field_bundle_value(row: dict[str, str], prefix: str) -> str:
    if prefix == "preparation_method":
        value = str(row.get("preparation_method", "") or "").strip()
        return "" if normalize_token(value) in {"", "unknown"} else value
    prefixes = [prefix]
    legacy_prefix = next((legacy for legacy, canonical in LEGACY_FIELD_ALIASES.items() if canonical == prefix), "")
    if legacy_prefix:
        prefixes.append(legacy_prefix)
    for candidate_prefix in prefixes:
        value = str(
            row.get(f"{candidate_prefix}_value", "")
            or row.get(f"{candidate_prefix}_value_text", "")
            or ""
        ).strip()
        if value:
            return value
    return ""


def load_resolved_relation_fields(
    resolved_relation_fields_tsv: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    if not resolved_relation_fields_tsv.exists():
        raise FileNotFoundError(
            f"Resolved relation fields TSV not found: {resolved_relation_fields_tsv}"
        )
    resolved_map: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with resolved_relation_fields_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            candidate_id = str(row.get("formulation_candidate_id", "") or "").strip()
            field_name = canonical_field_name(row.get("field_name", ""))
            if not candidate_id or field_name not in RESOLVED_RELATION_FIELD_NAMES:
                continue
            resolved_map[candidate_id][field_name] = {
                "field_value": str(row.get("field_value", "") or "").strip(),
                "field_value_norm": str(row.get("field_value_norm", "") or "").strip(),
                "scope_type": str(row.get("scope_type", "") or "").strip(),
                "resolution_rule": str(row.get("resolution_rule", "") or "").strip(),
                "source_relation_row_ids": str(row.get("source_relation_row_ids", "") or "").strip(),
                "deterministic_confidence": str(row.get("deterministic_confidence", "") or "").strip(),
            }
    return resolved_map


def apply_resolved_relation_fields(
    *,
    final_row: dict[str, str],
    representative: dict[str, str],
    resolved_field_map: dict[str, dict[str, dict[str, str]]],
) -> tuple[dict[str, str], set[str]]:
    materialized = dict(final_row)
    applied_fields: set[str] = set()
    candidate_id = str(representative.get("formulation_id", "") or "").strip()
    candidate_resolved = resolved_field_map.get(candidate_id, {})
    for field_name, payload in candidate_resolved.items():
        field_value = str(payload.get("field_value", "") or "").strip()
        if not field_value:
            continue
        if field_name == "preparation_method":
            if field_bundle_value(materialized, field_name):
                continue
            materialized["preparation_method"] = field_value
            applied_fields.add(field_name)
            continue
        if field_bundle_value(materialized, field_name):
            continue
        materialized[f"{field_name}_value"] = field_value
        if f"{field_name}_value_text" in materialized:
            materialized[f"{field_name}_value_text"] = field_value
        if f"{field_name}_scope" in materialized:
            materialized[f"{field_name}_scope"] = (
                "instance_specific"
                if str(payload.get("scope_type", "") or "").strip() == "formulation"
                else "global_shared"
            )
        if f"{field_name}_membership_confidence" in materialized:
            materialized[f"{field_name}_membership_confidence"] = str(
                payload.get("deterministic_confidence", "") or "medium"
            ).strip()
        if f"{field_name}_evidence_region_type" in materialized:
            materialized[f"{field_name}_evidence_region_type"] = "relation_resolved"
        if f"{field_name}_missing_reason" in materialized:
            materialized[f"{field_name}_missing_reason"] = ""
        applied_fields.add(field_name)
    return materialized, applied_fields


def group_has_clear_redundancy_signal(group_rows: list[dict[str, str]]) -> bool:
    candidate_sources = {
        normalize_text(row.get("candidate_source", "")) for row in group_rows if row.get("candidate_source")
    }
    return "figure_variable_sweep" in candidate_sources and "llm_extracted" in candidate_sources


def build_structured_duplicate_representation_map(
    rows: list[dict[str, str]],
    core_by_source_id: dict[str, dict[str, str]],
) -> dict[str, str]:
    targets_by_anchor: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        source_key = row_source_key(row)
        core_fields = core_by_source_id[source_key]
        if not can_receive_structured_duplicate_collapse(row, core_fields):
            continue
        row_anchor = extract_paper_local_row_anchor(row)
        if not row_anchor:
            continue
        targets_by_anchor[(row.get("key", "").strip(), row_anchor)].append(source_key)

    alternate_map: dict[str, str] = {}
    for row in rows:
        source_key = row_source_key(row)
        core_fields = core_by_source_id[source_key]
        if not is_structured_duplicate_representation_row(row, core_fields):
            continue
        row_anchor = extract_paper_local_row_anchor(row)
        if not row_anchor:
            continue
        targets = targets_by_anchor.get((row.get("key", "").strip(), row_anchor), [])
        if len(targets) != 1:
            continue
        alternate_map[source_key] = targets[0]
    return alternate_map


def build_variant_governance_target_map(
    rows: list[dict[str, str]],
    core_by_source_id: dict[str, dict[str, str]],
    signature_by_source_id: dict[str, str],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    rows_by_key_signature: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        source_key = row_source_key(row)
        signature = signature_by_source_id.get(source_key, "")
        if not signature:
            continue
        rows_by_key_signature[(row.get("key", "").strip(), signature)].append(row)

    collapse_map: dict[str, dict[str, str]] = {}
    review_map: dict[str, dict[str, str]] = {}

    for row in rows:
        source_key = row_source_key(row)
        if is_parent_linked_family_variant(row):
            continue
        signal = variant_signal_class(row)
        if not signal:
            continue
        signature = signature_by_source_id.get(source_key, "")
        tags = row_context_tags(row)
        if not signature:
            review_map[source_key] = {
                "variant_class": "uncertain_variant",
                "variant_signal": signal,
                "decision_rule": "kept_uncertain_variant_no_signature",
                "decision_reason": (
                    "Potential variant signal detected, but the row lacks a complete conservative "
                    "core signature needed for safe equivalence matching."
                ),
                "notes": f"variant_signal={signal}",
            }
            continue

        candidate_targets: list[str] = []
        for other in rows_by_key_signature[(row.get("key", "").strip(), signature)]:
            target_source_key = row_source_key(other)
            if target_source_key == source_key:
                continue
            if normalize_text(other.get("instance_kind")) not in {
                "new_formulation",
                "variant_formulation",
            }:
                continue
            target_tags = row_context_tags(other)
            if signal == "optimized_variant":
                if "optimized" in target_tags:
                    continue
                if not (is_non_doe_sweep_row(row) and is_non_doe_sweep_row(other)):
                    continue
            elif signal == "checkpoint_or_validation_variant":
                if "checkpoint_validation" in target_tags or "center_point" in target_tags:
                    continue
                if "doe" in tags or "doe" in target_tags:
                    continue
            elif signal == "post_processing_or_measurement_variant":
                if "post_processing" in target_tags or "measurement_context" in target_tags:
                    continue
                if normalize_text(other.get("change_role")) == "non_synthesis":
                    continue
                if not is_non_doe_sweep_row(other):
                    continue
            else:
                continue
            if populated_core_field_count(core_by_source_id[target_source_key]) < populated_core_field_count(
                core_by_source_id[source_key]
            ):
                continue
            candidate_targets.append(target_source_key)

        if len(candidate_targets) == 1:
            collapse_map[source_key] = {
                "target_source_key": candidate_targets[0],
                "variant_class": signal,
                "variant_signal": signal,
                "decision_rule": f"{signal}_same_core_identity",
                "decision_reason": (
                    "Row is classified as a conservative same-core variant and matches exactly one "
                    "stronger retained row in the same paper."
                ),
                "notes": f"collapse_signature={signature}",
            }
        else:
            review_map[source_key] = {
                "variant_class": "uncertain_variant",
                "variant_signal": signal,
                "decision_rule": "kept_uncertain_variant_review",
                "decision_reason": (
                    "Potential variant signal detected, but no unique same-core target was found "
                    "with sufficient evidence for conservative collapse."
                ),
                "notes": (
                    f"variant_signal={signal}; collapse_signature={signature}; "
                    f"candidate_target_count={len(candidate_targets)}"
                ),
            }

    return collapse_map, review_map


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def make_final_formulation_id(
    row: dict[str, str], collapse_signature: str | None
) -> str:
    base = collapse_signature or f"{row.get('key', '')}|{row.get('formulation_id', '')}"
    return f"{row.get('key', '').strip()}__fo__{short_hash(base)}"


def read_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [canonicalize_row_columns(row) for row in reader]


def load_relation_metadata(
    relation_records_tsv: Path,
) -> dict[str, dict[str, Any]]:
    if not relation_records_tsv.exists():
        raise FileNotFoundError(f"Relation records TSV not found: {relation_records_tsv}")

    metadata: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "relation_graph_ids": set(),
            "relation_method_group_ids": set(),
            "relation_parent_candidate_ids": set(),
            "relation_row_count": 0,
        }
    )
    with relation_records_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            candidate = str(row.get("formulation_candidate_id", "") or "").strip()
            if not candidate:
                continue
            item = metadata[candidate]
            item["relation_row_count"] += 1
            graph_id = str(row.get("relation_graph_id", "") or "").strip()
            if graph_id:
                item["relation_graph_ids"].add(graph_id)
            method_group_id = str(row.get("method_group_id", "") or "").strip()
            if method_group_id:
                item["relation_method_group_ids"].add(method_group_id)
            parent_id = str(row.get("parent_entity_id", "") or "").strip()
            if parent_id:
                item["relation_parent_candidate_ids"].add(parent_id)
    return metadata


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_markdown(
    input_path: Path,
    final_rows: list[dict[str, str]],
    decision_rows: list[dict[str, str]],
    summary_path: Path,
    relation_records_tsv: Path | None,
    resolved_relation_fields_tsv: Path | None,
) -> None:
    decision_counts = defaultdict(int)
    variant_class_counts = defaultdict(int)
    review_needed_count = 0
    for row in decision_rows:
        decision_counts[row["decision"]] += 1
        if row.get("variant_class"):
            variant_class_counts[row["variant_class"]] += 1
        if normalize_text(row.get("review_needed", "")) == "yes":
            review_needed_count += 1

    per_key_final = defaultdict(int)
    for row in final_rows:
        per_key_final[row["key"]] += 1

    content = [
        "# Final Output Summary v1",
        "",
        "## Scope",
        "",
        "This summary describes the controlled Stage5 materialization and duplicate/variant governance layer. Stage5 materializes direct-extraction fields and explicit Stage3-resolved relation fields, then applies conservative closure rules.",
        "",
        "## Input",
        "",
        f"- candidate_input_tsv: `{input_path}`",
        (
            f"- relation_records_tsv: `{relation_records_tsv}`"
            if relation_records_tsv is not None
            else "- relation_records_tsv: `not provided`"
        ),
        (
            f"- resolved_relation_fields_tsv: `{resolved_relation_fields_tsv}`"
            if resolved_relation_fields_tsv is not None
            else "- resolved_relation_fields_tsv: `not provided`"
        ),
        "",
        "## What phase 1 currently handles",
        "",
        "- filters rows explicitly marked as non-formulation or characterization-only post-processing rows",
        "- materializes relation-backed descriptive synthesis fields from Stage3 resolved relation outputs",
        "- computes a conservative core-parameter signature from current candidate-row fields",
        "- classifies conservative variant signals into duplicate, optimized, checkpoint/validation, post-processing/measurement, or uncertain review-needed cases",
        "- collapses rows only when signature completeness is high and a unique conservative target is available",
        "- collapses structured DOE/table-derived alternate representations when they clearly duplicate an already richer retained row for the same paper-local row anchor",
        "- preserves provenance by retaining representative-row metadata, collapsed-variant membership, and a row-level decision trace",
        "",
        "## What phase 1 intentionally does not handle",
        "",
        "- semantic inheritance inference beyond Stage3 resolved relation outputs",
        "- generalized DOE coordinate reconciliation when Stage5 lacks a unique deterministic target",
        "- Stage 5B benchmark comparison against GT",
        "- modeling-target-specific filtering such as PLGA-only export subsets",
        "",
        "## Filtering rules applied",
        "",
        "- `explicit_candidate_non_formulation`",
        "- `characterization_only_post_processing`",
        "",
        "## Collapse rules applied",
        "",
        "- collapse only if polymer identity and loaded state are known",
        "- collapse only if the conservative core signature has at least five populated components",
        "- collapse duplicate representations only when a clear mixed-source redundancy signal or a unique same-row-anchor match is present",
        "- collapse optimized, checkpoint/validation, or post-processing/measurement variants only when they resolve to exactly one stronger same-core target under the Stage5 policy",
        "- collapse structured `doe_numbered_table_row` rows when they are weak alternate representations of an already richer same-paper row with the same numeric row anchor",
        "- if uncertain, keep rows separate and mark them review-needed in the decision trace",
        "",
        "## Decision counts",
        "",
        f"- kept: `{decision_counts['kept']}`",
        f"- filtered_non_formulation: `{decision_counts['filtered_non_formulation']}`",
        f"- collapsed_into_existing: `{decision_counts['collapsed_into_existing']}`",
        f"- final_rows: `{len(final_rows)}`",
        f"- review_needed_rows: `{review_needed_count}`",
        "",
        "## Variant class counts",
        "",
    ]
    for variant_class in sorted(variant_class_counts):
        content.append(f"- `{variant_class}`: `{variant_class_counts[variant_class]}`")
    content.extend(
        [
            "",
            "## Final rows by paper",
            "",
        ]
    )
    for key in sorted(per_key_final):
        content.append(f"- `{key}`: `{per_key_final[key]}`")
    content.extend(
        [
            "",
            "## Open questions still visible after phase 1",
            "",
            "- exact core-signature fields for broader collapse remain unresolved",
            "- baseline versus optimized provenance handling is still conservative",
            "- parent/variant collapse policy is still intentionally narrow and unique-target-based",
            "- relation-driven field materialization is limited to explicit Stage3 resolved descriptive synthesis fields",
            "- DOE-aware coordinate closure still needs a later explicit contract when unique deterministic mapping is available",
            "- benchmark comparison still requires the separate Stage 5B comparison step",
        ]
    )
    summary_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def build_minimal_final_output(
    input_tsv: Path,
    out_dir: Path,
    relation_records_tsv: Path,
    resolved_relation_fields_tsv: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_candidate_rows(input_tsv)
    if not rows:
        raise ValueError(f"No candidate rows found in {input_tsv}")
    if relation_records_tsv is None:
        raise ValueError("Stage5 requires --relation-records-tsv; silent bypass is not allowed.")
    if resolved_relation_fields_tsv is None:
        raise ValueError("Stage5 requires --resolved-relation-fields-tsv; silent bypass is not allowed.")
    relation_metadata = load_relation_metadata(relation_records_tsv)
    resolved_relation_field_map = load_resolved_relation_fields(resolved_relation_fields_tsv)
    rows_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_paper[str(row.get("key", "") or "").strip()].append(row)

    original_fieldnames = list(rows[0].keys())
    row_by_source_key = {row_source_key(row): row for row in rows}
    core_by_id: dict[str, dict[str, str]] = {}
    filtered_ids: set[str] = set()
    filter_rules: dict[str, tuple[str, str]] = {}
    eligible_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    collapse_signature_by_id: dict[str, str] = {}
    conservative_signature_by_id: dict[str, str] = {}

    for row in rows:
        source_id = row_source_key(row)
        core_fields = build_core_fields(row)
        core_by_id[source_id] = core_fields
        should_filter, filter_rule, filter_reason = should_filter_non_formulation(row, core_fields)
        if should_filter:
            filtered_ids.add(source_id)
            filter_rules[source_id] = (filter_rule, filter_reason)
            continue

        exclusion = collapse_exclusion_reason(row, core_fields)
        if not exclusion:
            signature = build_collapse_signature(row, core_fields)
            collapse_signature_by_id[source_id] = signature
            eligible_groups[signature].append(row)

        governance_exclusion = collapse_exclusion_reason(
            row,
            core_fields,
            allow_context_tags=True,
        )
        if not governance_exclusion:
            conservative_signature_by_id[source_id] = build_collapse_signature(row, core_fields)

    representative_by_signature: dict[str, dict[str, str]] = {}
    final_id_by_source_id: dict[str, str] = {}
    collapsed_ids: set[str] = set()
    collapse_metadata_by_source_id: dict[str, dict[str, str]] = {}
    review_metadata_by_source_id: dict[str, dict[str, str]] = {}

    for signature, group_rows in eligible_groups.items():
        if len(group_rows) < 2:
            continue
        if not group_has_clear_redundancy_signal(group_rows):
            continue
        representative = choose_representative(group_rows, core_by_id)
        representative_by_signature[signature] = representative
        final_formulation_id = make_final_formulation_id(representative, signature)
        representative_source_key = row_source_key(representative)
        for row in group_rows:
            source_key = row_source_key(row)
            final_id_by_source_id[source_key] = final_formulation_id
            if source_key == representative_source_key:
                continue
            collapsed_ids.add(source_key)
            collapse_metadata_by_source_id[source_key] = {
                "variant_class": "duplicate_representation",
                "variant_signal": "duplicate_representation",
                "decision_rule": "clear_core_signature_overlap",
                "decision_reason": (
                    "Row shares a conservative phase-1 core signature with a higher-priority representative row."
                ),
                "collapse_reason": (
                    "Collapsed as a duplicate representation after a clear mixed-source overlap signal."
                ),
                "review_needed": "no",
                "notes": f"collapse_signature={signature}",
            }

    structured_duplicate_targets = build_structured_duplicate_representation_map(
        rows=rows,
        core_by_source_id=core_by_id,
    )
    variant_governance_targets, variant_review_map = build_variant_governance_target_map(
        rows=rows,
        core_by_source_id=core_by_id,
        signature_by_source_id=conservative_signature_by_id,
    )
    review_metadata_by_source_id.update(variant_review_map)
    representative_source_keys = {
        row_source_key(representative)
        for representative in representative_by_signature.values()
    }
    for source_key, target_source_key in structured_duplicate_targets.items():
        if source_key in collapsed_ids:
            continue
        target_row = row_by_source_key[target_source_key]
        target_signature = (
            collapse_signature_by_id.get(target_source_key)
            if target_source_key in representative_source_keys
            else None
        )
        target_final_formulation_id = final_id_by_source_id.get(
            target_source_key,
            make_final_formulation_id(target_row, target_signature),
        )
        final_id_by_source_id[source_key] = target_final_formulation_id
        final_id_by_source_id[target_source_key] = target_final_formulation_id
        collapsed_ids.add(source_key)
        collapse_metadata_by_source_id[source_key] = {
            "variant_class": "duplicate_representation",
            "variant_signal": "duplicate_representation",
            "decision_rule": "structured_duplicate_representation_same_row_anchor",
            "decision_reason": (
                "Structured DOE/table-derived row matches an already retained richer formulation "
                "representation with the same paper-local row anchor and no additional core identity fields."
            ),
            "collapse_reason": (
                "Collapsed as a duplicate representation because the numbered table row is only an alternate "
                "surface of an already retained formulation."
            ),
            "review_needed": "no",
            "notes": (
                f"matched_source_formulation_id={target_row.get('formulation_id', '')}; "
                f"matched_row_anchor={extract_paper_local_row_anchor(row_by_source_key[source_key])}"
            ),
        }

    for source_key, payload in variant_governance_targets.items():
        if source_key in collapsed_ids:
            continue
        target_source_key = payload["target_source_key"]
        target_row = row_by_source_key[target_source_key]
        target_signature = (
            collapse_signature_by_id.get(target_source_key)
            if target_source_key in representative_source_keys
            else None
        )
        target_final_formulation_id = final_id_by_source_id.get(
            target_source_key,
            make_final_formulation_id(target_row, target_signature),
        )
        final_id_by_source_id[source_key] = target_final_formulation_id
        final_id_by_source_id[target_source_key] = target_final_formulation_id
        collapsed_ids.add(source_key)
        collapse_metadata_by_source_id[source_key] = {
            "variant_class": payload["variant_class"],
            "variant_signal": payload["variant_signal"],
            "decision_rule": payload["decision_rule"],
            "decision_reason": payload["decision_reason"],
            "collapse_reason": (
                "Collapsed as a conservative same-core variant because exactly one stronger retained target was found."
            ),
            "review_needed": "no",
            "notes": (
                f"matched_source_formulation_id={target_row.get('formulation_id', '')}; "
                f"{payload['notes']}"
            ),
        }

    final_rows: list[dict[str, str]] = []
    decision_rows: list[dict[str, str]] = []

    source_rows_by_final_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        source_id = row["formulation_id"]
        source_key = row_source_key(row)
        core_fields = core_by_id[source_key]
        key_fields_used = build_key_fields_used(core_fields)
        family_labels = compute_family_labels(row, core_fields)

        if source_key in filtered_ids:
            rule, reason = filter_rules[source_key]
            variant_signal = variant_signal_class(row)
            variant_class = (
                "post_processing_or_measurement_variant"
                if rule == "characterization_only_post_processing"
                else ""
            )
            decision = RowDecision(
                decision="filtered_non_formulation",
                target_final_formulation_id="",
                variant_class=variant_class,
                variant_signal=variant_signal,
                equivalence_group_id="",
                family_id=family_labels["family_id"],
                parent_core_row_id=family_labels["parent_core_row_id"],
                variant_role=family_labels["variant_role"],
                payload_state=family_labels["payload_state"],
                benchmark_default_include=family_labels["benchmark_default_include"],
                decision_rule=rule,
                decision_reason=reason,
                retention_reason="",
                collapse_reason="Row is excluded from benchmark-facing final formulation closure.",
                review_needed="no",
                key_fields_used=key_fields_used,
                confidence_or_rule_scope="phase1_conservative_filter",
                notes="Row is excluded from final formulation closure.",
            )
        elif source_key in collapsed_ids:
            payload = collapse_metadata_by_source_id[source_key]
            target_final_formulation_id = final_id_by_source_id[source_key]
            decision = RowDecision(
                decision="collapsed_into_existing",
                target_final_formulation_id=target_final_formulation_id,
                variant_class=payload["variant_class"],
                variant_signal=payload["variant_signal"],
                equivalence_group_id=target_final_formulation_id,
                family_id=family_labels["family_id"],
                parent_core_row_id=family_labels["parent_core_row_id"],
                variant_role="duplicate_representation",
                payload_state=family_labels["payload_state"],
                benchmark_default_include="no",
                decision_rule=payload["decision_rule"],
                decision_reason=payload["decision_reason"],
                retention_reason="",
                collapse_reason=payload["collapse_reason"],
                review_needed=payload["review_needed"],
                key_fields_used=key_fields_used,
                confidence_or_rule_scope="phase1_variant_governance",
                notes=payload["notes"],
            )
        else:
            collapse_signature = (
                collapse_signature_by_id.get(source_key)
                if source_key in representative_source_keys
                else None
            )
            target_final_formulation_id = final_id_by_source_id.get(
                source_key,
                make_final_formulation_id(row, collapse_signature),
            )
            final_id_by_source_id[source_key] = target_final_formulation_id
            review_payload = review_metadata_by_source_id.get(source_key)
            if source_key in representative_source_keys:
                decision_rule = "kept_as_representative_after_collapse"
                decision_reason = "Representative row retained for a clear overlap group."
                retention_reason = "Retained as the benchmark-facing representative for a clear overlap group."
                variant_class = "duplicate_representation"
                variant_signal = "duplicate_representation"
                review_needed = "no"
                notes = (
                    f"collapse_signature={collapse_signature}" if collapse_signature else "No collapse signature used."
                )
            elif review_payload:
                decision_rule = review_payload["decision_rule"]
                decision_reason = review_payload["decision_reason"]
                retention_reason = (
                    "Retained as a separate benchmark-facing row because Stage5 did not find a unique safe collapse target."
                )
                variant_class = review_payload["variant_class"]
                variant_signal = review_payload["variant_signal"]
                review_needed = "yes"
                notes = review_payload["notes"]
            else:
                decision_rule = "kept_no_clear_phase1_overlap"
                decision_reason = "No explicit non-formulation rule or clear conservative collapse rule applied."
                retention_reason = "Retained because no conservative duplicate or variant-collapse rule fired."
                variant_class = ""
                variant_signal = variant_signal_class(row)
                review_needed = "no"
                notes = (
                    f"collapse_signature={collapse_signature}" if collapse_signature else "No collapse signature used."
                )
            decision = RowDecision(
                decision="kept",
                target_final_formulation_id=target_final_formulation_id,
                variant_class=variant_class,
                variant_signal=variant_signal,
                equivalence_group_id=target_final_formulation_id,
                family_id=family_labels["family_id"],
                parent_core_row_id=family_labels["parent_core_row_id"],
                variant_role=family_labels["variant_role"],
                payload_state=family_labels["payload_state"],
                benchmark_default_include=family_labels["benchmark_default_include"],
                decision_rule=decision_rule,
                decision_reason=decision_reason,
                retention_reason=retention_reason,
                collapse_reason="",
                review_needed=review_needed,
                key_fields_used=key_fields_used,
                confidence_or_rule_scope="phase1_variant_governance",
                notes=notes,
            )
            source_rows_by_final_id[target_final_formulation_id].append(row)

        decision_rows.append(
            {
                "zotero_key": row.get("key", ""),
                "source_formulation_id": source_id,
                "source_raw_formulation_label": row.get("raw_formulation_label", ""),
                "decision": decision.decision,
                "target_final_formulation_id": decision.target_final_formulation_id,
                "variant_class": decision.variant_class,
                "variant_signal": decision.variant_signal,
                "equivalence_group_id": decision.equivalence_group_id,
                "family_id": decision.family_id,
                "parent_core_row_id": decision.parent_core_row_id,
                "variant_role": decision.variant_role,
                "payload_state": decision.payload_state,
                "benchmark_default_include": decision.benchmark_default_include,
                "decision_rule": decision.decision_rule,
                "decision_reason": decision.decision_reason,
                "retention_reason": decision.retention_reason,
                "collapse_reason": decision.collapse_reason,
                "review_needed": decision.review_needed,
                "key_fields_used": decision.key_fields_used,
                "confidence_or_rule_scope": decision.confidence_or_rule_scope,
                "notes": decision.notes,
            }
        )

    collapsed_variant_members_by_final_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source_key, target_final_formulation_id in final_id_by_source_id.items():
        if source_key not in collapsed_ids:
            continue
        payload = collapse_metadata_by_source_id.get(source_key, {})
        source_row = row_by_source_key[source_key]
        collapsed_variant_members_by_final_id[target_final_formulation_id].append(
            {
                "formulation_id": source_row.get("formulation_id", ""),
                "variant_class": payload.get("variant_class", ""),
                "decision_rule": payload.get("decision_rule", ""),
            }
        )

    review_needed_by_final_id: dict[str, bool] = defaultdict(bool)
    for row in decision_rows:
        if row["decision"] != "kept":
            continue
        if normalize_text(row.get("review_needed", "")) == "yes":
            review_needed_by_final_id[row["target_final_formulation_id"]] = True

    for target_final_formulation_id, source_group in sorted(
        source_rows_by_final_id.items(), key=lambda item: item[0]
    ):
        representative = max(
            source_group,
            key=lambda row: (
                candidate_priority(row),
                confidence_priority(row),
                populated_core_field_count(core_by_id[row_source_key(row)]),
                len(str(row.get("evidence_span_text", "") or "")),
                str(row.get("formulation_id", "")),
            ),
        )
        source_ids = [row["formulation_id"] for row in source_group]
        source_labels = [row.get("raw_formulation_label", "") for row in source_group]
        source_sources = [row.get("candidate_source", "") for row in source_group]
        representative_core = core_by_id[row_source_key(representative)]
        representative_family_labels = compute_family_labels(representative, representative_core)
        source_candidate_ids = [row["formulation_id"] for row in source_group]
        relation_graph_ids = sorted(
            {
                graph_id
                for source_candidate_id in source_candidate_ids
                for graph_id in relation_metadata.get(source_candidate_id, {}).get("relation_graph_ids", set())
            }
        )
        relation_method_group_ids = sorted(
            {
                method_group_id
                for source_candidate_id in source_candidate_ids
                for method_group_id in relation_metadata.get(source_candidate_id, {}).get(
                    "relation_method_group_ids", set()
                )
            }
        )
        relation_parent_candidate_ids = sorted(
            {
                parent_id
                for source_candidate_id in source_candidate_ids
                for parent_id in relation_metadata.get(source_candidate_id, {}).get(
                    "relation_parent_candidate_ids", set()
                )
            }
        )
        relation_row_count = sum(
            int(relation_metadata.get(source_candidate_id, {}).get("relation_row_count", 0))
            for source_candidate_id in source_candidate_ids
        )
        collapsed_members = collapsed_variant_members_by_final_id.get(target_final_formulation_id, [])
        collapsed_variant_ids = [item["formulation_id"] for item in collapsed_members]
        collapsed_variant_classes = sorted(
            {item["variant_class"] for item in collapsed_members if item["variant_class"]}
        )
        representative_trace = next(
            (
                decision_row
                for decision_row in decision_rows
                if decision_row["source_formulation_id"] == representative["formulation_id"]
                and decision_row["target_final_formulation_id"] == target_final_formulation_id
            ),
            {},
        )
        field_source_type = "direct_extraction"

        final_row = {
            "final_formulation_id": target_final_formulation_id,
            "representative_source_formulation_id": representative["formulation_id"],
            "representative_source_raw_formulation_label": representative.get(
                "raw_formulation_label", ""
            ),
            "source_candidate_count": str(len(source_group)),
            "source_candidate_ids": json.dumps(source_ids, ensure_ascii=True),
            "source_candidate_labels": json.dumps(source_labels, ensure_ascii=True),
            "source_candidate_sources": json.dumps(source_sources, ensure_ascii=True),
            "collapsed_variant_count": str(len(collapsed_members)),
            "collapsed_variant_source_ids": json.dumps(collapsed_variant_ids, ensure_ascii=True),
            "collapsed_variant_classes": json.dumps(collapsed_variant_classes, ensure_ascii=True),
            "retention_reason": representative_trace.get("retention_reason", ""),
            "review_needed": "yes" if review_needed_by_final_id.get(target_final_formulation_id, False) else "no",
            "family_id": representative_family_labels["family_id"],
            "parent_core_row_id": representative_family_labels["parent_core_row_id"],
            "variant_role": representative_family_labels["variant_role"],
            "payload_state": representative_family_labels["payload_state"],
            "benchmark_default_include": representative_family_labels["benchmark_default_include"],
            "collapse_signature": collapse_signature_by_id.get(
                row_source_key(representative), ""
            ),
            "loaded_state_final": representative_core["loaded_state"],
            "polymer_identity_final": representative_core["polymer_identity"],
            "final_output_rule": (
                "representative_after_collapse"
                if len(source_group) > 1 or collapsed_members
                else "kept_without_collapse"
            ),
            "relation_graph_ids": json.dumps(relation_graph_ids, ensure_ascii=True),
            "relation_method_group_ids": json.dumps(relation_method_group_ids, ensure_ascii=True),
            "relation_parent_candidate_ids": json.dumps(
                relation_parent_candidate_ids, ensure_ascii=True
            ),
            "relation_record_count": str(relation_row_count),
            "field_source_type": field_source_type,
        }
        for field in original_fieldnames:
            final_row[field] = representative.get(field, "")
        final_row, applied_relation_fields = apply_resolved_relation_fields(
            final_row=final_row,
            representative=representative,
            resolved_field_map=resolved_relation_field_map,
        )
        if applied_relation_fields:
            final_row["field_source_type"] = "relation_resolved"
        elif any(
            not field_bundle_value(final_row, field_name)
            for field_name in RESOLVED_RELATION_FIELD_NAMES
        ):
            final_row["field_source_type"] = "unresolved_blank"
        final_rows.append(final_row)

    decision_trace_path = out_dir / DECISION_TRACE_NAME
    final_table_path = out_dir / FINAL_TABLE_NAME
    summary_path = out_dir / SUMMARY_NAME

    field_source_by_final_id = {
        row["final_formulation_id"]: row.get("field_source_type", "unresolved_blank")
        for row in final_rows
    }
    for row in decision_rows:
        target_final_id = str(row.get("target_final_formulation_id", "") or "").strip()
        row["field_source_type"] = (
            field_source_by_final_id.get(target_final_id, "unresolved_blank")
            if target_final_id
            else "unresolved_blank"
        )

    write_tsv(
        decision_trace_path,
        [
            "zotero_key",
            "source_formulation_id",
            "source_raw_formulation_label",
            "decision",
            "target_final_formulation_id",
            "variant_class",
            "variant_signal",
            "equivalence_group_id",
            "family_id",
            "parent_core_row_id",
            "variant_role",
            "payload_state",
            "benchmark_default_include",
            "decision_rule",
            "decision_reason",
            "retention_reason",
            "collapse_reason",
            "review_needed",
            "key_fields_used",
            "field_source_type",
            "confidence_or_rule_scope",
            "notes",
        ],
        decision_rows,
    )

    write_tsv(
        final_table_path,
        [
            "final_formulation_id",
            "representative_source_formulation_id",
            "representative_source_raw_formulation_label",
            "source_candidate_count",
            "source_candidate_ids",
            "source_candidate_labels",
            "source_candidate_sources",
            "collapsed_variant_count",
            "collapsed_variant_source_ids",
            "collapsed_variant_classes",
            "retention_reason",
            "review_needed",
            "family_id",
            "parent_core_row_id",
            "variant_role",
            "payload_state",
            "benchmark_default_include",
            "collapse_signature",
            "loaded_state_final",
            "polymer_identity_final",
            "final_output_rule",
            "relation_graph_ids",
            "relation_method_group_ids",
            "relation_parent_candidate_ids",
            "relation_record_count",
            "field_source_type",
            *original_fieldnames,
            *[name for name in PREPARATION_METHOD_FIELDNAMES if name not in original_fieldnames],
        ],
        final_rows,
    )

    build_summary_markdown(
        input_tsv,
        final_rows,
        decision_rows,
        summary_path,
        relation_records_tsv,
        resolved_relation_fields_tsv,
    )

    return {
        "input_rows": len(rows),
        "final_rows": len(final_rows),
        "filtered_rows": sum(1 for row in decision_rows if row["decision"] == "filtered_non_formulation"),
        "collapsed_rows": sum(1 for row in decision_rows if row["decision"] == "collapsed_into_existing"),
        "kept_rows": sum(1 for row in decision_rows if row["decision"] == "kept"),
        "final_table_path": final_table_path,
        "decision_trace_path": decision_trace_path,
        "summary_path": summary_path,
        "relation_records_tsv": relation_records_tsv,
        "resolved_relation_fields_tsv": resolved_relation_fields_tsv,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build phase-1 minimal final-output artifacts from Stage2 candidate-instance TSV output."
    )
    parser.add_argument("--input-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--relation-records-tsv", required=True, type=Path)
    parser.add_argument("--resolved-relation-fields-tsv", required=True, type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = build_minimal_final_output(
        args.input_tsv,
        args.out_dir,
        relation_records_tsv=args.relation_records_tsv,
        resolved_relation_fields_tsv=args.resolved_relation_fields_tsv,
    )
    print(
        json.dumps(
            {
                "input_rows": stats["input_rows"],
                "final_rows": stats["final_rows"],
                "filtered_rows": stats["filtered_rows"],
                "collapsed_rows": stats["collapsed_rows"],
                "kept_rows": stats["kept_rows"],
                "relation_records_tsv": (
                    str(stats["relation_records_tsv"]) if stats["relation_records_tsv"] else ""
                ),
                "resolved_relation_fields_tsv": str(stats["resolved_relation_fields_tsv"]),
                "final_table_path": str(stats["final_table_path"]),
                "decision_trace_path": str(stats["decision_trace_path"]),
                "summary_path": str(stats["summary_path"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
