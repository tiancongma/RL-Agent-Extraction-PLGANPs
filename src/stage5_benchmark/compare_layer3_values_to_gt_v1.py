#!/usr/bin/env python3
from __future__ import annotations

# Layer3 compare and numeric backfill must remain deterministic and identity-bound.
# This script is allowed to bind frozen GT rows onto canonical current-system rows,
# normalize identities, reuse advisory bridge surfaces, and emit audit-ready
# alignment evidence. It must not create new formulation rows, redefine the
# benchmark-facing formulation universe, or act as a second semantic extractor.

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    from src.stage2_sampling_labels.table_row_expansion_v1 import canonical_field_for_header
    from src.stage2_sampling_labels.table_structure_dictionary_v1 import normalize_dictionary_value_from_rows
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.table_row_expansion_v1 import canonical_field_for_header
    from src.stage2_sampling_labels.table_structure_dictionary_v1 import normalize_dictionary_value_from_rows
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context

CORE_FIXED_FIELDS = {
    "polymer_name",
    "polymer_grade",
    "polymer_mw_raw",
    "polymer_mw_kDa",
    "la_ga_ratio_raw",
    "la_ga_ratio_normalized",
    "polymer_mass_mg",
    "polymer_concentration_value",
    "polymer_concentration_unit",
    "polymer_concentration_phase",
    "polymer_to_solvent_ratio_raw",
    "polymer_to_drug_ratio_raw",
    "drug_name",
    "drug_mass_mg",
    "drug_concentration_value",
    "drug_concentration_unit",
    "drug_to_polymer_ratio_raw",
    "surfactant_name",
    "surfactant_mass_mg",
    "surfactant_concentration_value",
    "surfactant_concentration_unit",
    "stabilizer_name",
    "helper_material_name",
    "method_type",
    "solvent_name",
    "co_solvent_name",
    "W1_volume_mL",
    "O_volume_mL",
    "W2_volume_mL",
    "external_aqueous_phase_volume_mL",
    "internal_aqueous_phase_volume_mL",
    "phase_ratio_raw",
    "sonication_time_s",
    "homogenization_time_min",
    "stirring_time_h",
    "evaporation_time_h",
    "ee_percent",
    "lc_percent",
    "dl_percent",
    "particle_size_nm",
    "pdi",
    "zeta_mV",
}
NAMED_EXTENSIBLE_VARIABLE_FIELDS = {"pH_raw"}
PROVENANCE_ONLY_FIELDS = {"value_source_type", "candidate_notes"}
IDENTITY_FIELDS = {
    "paper_key",
    "doi",
    "gt_formulation_id",
    "family_id",
    "parent_core",
    "variant_role",
    "benchmark_default_include",
    "formulation_label",
    "seed_pred_representative_source_formulation_id",
    "gt_row_decision",
}
ALL_FROZEN_FIELDS = CORE_FIXED_FIELDS | NAMED_EXTENSIBLE_VARIABLE_FIELDS | PROVENANCE_ONLY_FIELDS | IDENTITY_FIELDS

DEFAULT_SELECTED_COMPARE_MODE = "canonicalized"
CELL_OUTPUT_NAME = "layer3_value_compare_cells_v1.tsv"
SUMMARY_OUTPUT_NAME = "layer3_value_compare_summary_v1.tsv"
ERROR_BUCKET_OUTPUT_NAME = "layer3_value_error_buckets_v1.tsv"
ALIGNMENT_RESOLUTION_OUTPUT_NAME = "layer3_alignment_resolution_rows_v1.tsv"
RISK_REVIEW_QUEUE_OUTPUT_NAME = "layer3_risk_review_queue_v1.tsv"

DEFAULT_ALIGNMENT_SCAFFOLD_TSV = Path(
    "data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_variant_alignment_scaffold_v1.tsv"
)
DEFAULT_VALUE_NORMALIZATION_LEXICON_TSV = Path(
    "data/cleaned/reference/value_normalization_lexicon_v1.tsv"
)
ROLE_TOLERANT_NAME_SOURCE_FIELDS = ("surfactant_name", "stabilizer_name")
REPORTING_FIELD_RENAMES = {
    "surfactant_concentration_value": "emulsifier_stabilizer_concentration_value",
    "surfactant_concentration_unit": "emulsifier_stabilizer_concentration_unit",
}
REPORTING_ONLY_CORE_FIXED_FIELDS = {
    "emulsifier_stabilizer_name",
    "emulsifier_stabilizer_concentration_value",
    "emulsifier_stabilizer_concentration_unit",
}

TEXT_FIELDS = {
    "polymer_name",
    "polymer_grade",
    "drug_name",
    "surfactant_name",
    "stabilizer_name",
    "helper_material_name",
    "method_type",
    "solvent_name",
    "co_solvent_name",
}
NUMERIC_FIELDS = {
    "polymer_mw_raw",
    "polymer_mw_kDa",
    "polymer_mass_mg",
    "polymer_concentration_value",
    "drug_mass_mg",
    "drug_concentration_value",
    "surfactant_mass_mg",
    "surfactant_concentration_value",
    "W1_volume_mL",
    "O_volume_mL",
    "W2_volume_mL",
    "external_aqueous_phase_volume_mL",
    "internal_aqueous_phase_volume_mL",
    "sonication_time_s",
    "homogenization_time_min",
    "stirring_time_h",
    "evaporation_time_h",
    "ee_percent",
    "lc_percent",
    "dl_percent",
    "particle_size_nm",
    "pdi",
    "zeta_mV",
    "pH_raw",
}
RATIO_FIELDS = {
    "la_ga_ratio_raw",
    "la_ga_ratio_normalized",
    "polymer_to_solvent_ratio_raw",
    "polymer_to_drug_ratio_raw",
    "drug_to_polymer_ratio_raw",
    "phase_ratio_raw",
}

# Canonical endpoint order for ratio fields whose schema encodes direction.
# `phase_ratio_raw` is intentionally left direction-neutral: when source labels
# name endpoints such as water:oil or organic:aqueous, compare keeps the named
# endpoints and canonicalizes reversed named ratios rather than imposing one
# global phase direction.
RATIO_FIELD_ENDPOINTS = {
    "la_ga_ratio_raw": ("la", "ga"),
    "la_ga_ratio_normalized": ("la", "ga"),
    "polymer_to_solvent_ratio_raw": ("polymer", "solvent"),
    "polymer_to_drug_ratio_raw": ("polymer", "drug"),
    "drug_to_polymer_ratio_raw": ("drug", "polymer"),
}

SYSTEM_FIELD_MAP = {
    "polymer_name": {"column": "polymer_name_raw", "source": "direct_extracted", "evidence": "supported"},
    "polymer_grade": {"column": "polymer_name_raw", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "polymer_mw_raw": {"column": "polymer_mw_kDa_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_mw_kDa": {"column": "polymer_mw_kDa_value_text", "source": "direct_extracted", "evidence": "supported"},
    "la_ga_ratio_raw": {"column": "la_ga_ratio_value_text", "source": "direct_extracted", "evidence": "supported"},
    "la_ga_ratio_normalized": {"column": "la_ga_ratio_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_mass_mg": {"column": "plga_mass_mg_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_concentration_value": {"column": "polymer_concentration_value_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_concentration_unit": {"column": "polymer_concentration_unit_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_concentration_phase": {"column": "polymer_concentration_phase_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_to_solvent_ratio_raw": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "polymer_to_drug_ratio_raw": {"column": "polymer_to_drug_ratio_raw_value_text", "source": "direct_extracted", "evidence": "supported"},
    "drug_name": {"column": "drug_name_value_text", "source": "direct_extracted", "evidence": "supported"},
    "drug_mass_mg": {"column": "drug_feed_amount_text_value_text", "source": "direct_extracted", "evidence": "supported"},
    "drug_concentration_value": {"column": "drug_concentration_value_value_text", "source": "direct_extracted", "evidence": "supported"},
    "drug_concentration_unit": {"column": "drug_concentration_unit_value_text", "source": "direct_extracted", "evidence": "supported"},
    "drug_to_polymer_ratio_raw": {"column": "drug_to_polymer_ratio_raw_value_text", "source": "direct_extracted", "evidence": "supported"},
    "surfactant_name": {"column": "surfactant_name_value_text", "source": "direct_extracted", "evidence": "supported"},
    "surfactant_mass_mg": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "surfactant_concentration_value": {"column": "surfactant_concentration_text_value_text", "source": "direct_extracted", "evidence": "supported"},
    "surfactant_concentration_unit": {"column": "surfactant_concentration_text_value_text", "source": "direct_extracted", "evidence": "supported"},
    "stabilizer_name": {"column": "surfactant_name_value_text", "source": "direct_extracted", "evidence": "supported"},
    "helper_material_name": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "method_type": {"column": "preparation_method", "source": "relation_or_direct", "evidence": "supported"},
    "solvent_name": {"column": "organic_solvent_value_text", "source": "direct_extracted", "evidence": "supported"},
    "co_solvent_name": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "W1_volume_mL": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "O_volume_mL": {"column": "organic_phase_volume_mL_value_text", "source": "direct_extracted", "evidence": "supported"},
    "W2_volume_mL": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "external_aqueous_phase_volume_mL": {"column": "external_aqueous_phase_volume_mL_value_text", "source": "direct_extracted", "evidence": "supported"},
    "internal_aqueous_phase_volume_mL": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "phase_ratio_raw": {"column": "phase_ratio_raw_value_text", "source": "direct_extracted", "evidence": "supported"},
    "sonication_time_s": {"column": "sonication_time_s_value_text", "source": "direct_extracted", "evidence": "supported"},
    "homogenization_time_min": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "stirring_time_h": {"column": "stirring_time_h_value_text", "source": "direct_extracted", "evidence": "supported"},
    "evaporation_time_h": {"column": "evaporation_time_h_value_text", "source": "direct_extracted", "evidence": "supported"},
    "centrifugation_g": {"column": "centrifugation_g_value_text", "source": "direct_extracted", "evidence": "supported"},
    "centrifugation_time_min": {"column": "centrifugation_time_min_value_text", "source": "direct_extracted", "evidence": "supported"},
    "ee_percent": {"column": "encapsulation_efficiency_percent_value_text", "source": "direct_extracted", "evidence": "supported"},
    "lc_percent": {"column": "loading_content_percent_value_text", "source": "direct_extracted", "evidence": "supported"},
    "dl_percent": {"column": "dl_percent_value_text", "source": "direct_extracted", "evidence": "supported"},
    "particle_size_nm": {"column": "size_nm_value_text", "source": "direct_extracted", "evidence": "supported"},
    "pdi": {"column": "pdi_value_text", "source": "direct_extracted", "evidence": "supported"},
    "zeta_mV": {"column": "zeta_mV_value_text", "source": "direct_extracted", "evidence": "supported"},
    "pH_raw": {"column": "pH_raw_value_text", "source": "direct_extracted", "evidence": "supported"},
}


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_lower_text(value: Any) -> str:
    return normalize_text(value).lower()


def canonicalize_text(value: Any) -> str:
    text = normalize_lower_text(value)
    text = text.replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b[wv]/?v\b", "%w/v", text)
    return text.strip()


def _canonicalize_field_text(field_name: str, value: str) -> str:
    text = canonicalize_text(value)
    if field_name == "polymer_grade":
        text = re.sub(r"\s*\((?:grade|polymer\s+type)\)\s*$", "", text)
    return text


def _parse_ratio_numeric_from_text(value: Any) -> float | None:
    text = canonicalize_text(value)
    if not text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[:/]\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        left = float(match.group(1))
        right = float(match.group(2))
    except ValueError:
        return None
    if right == 0:
        return None
    return left / right


def parse_numeric(value: Any) -> float | None:
    text = canonicalize_text(value)
    if not text:
        return None
    ratio_value = _parse_ratio_numeric_from_text(text)
    if ratio_value is not None:
        return ratio_value
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_percent_aware_numeric(value: Any) -> tuple[float, bool] | None:
    text = canonicalize_text(value)
    parsed = parse_numeric(text)
    if parsed is None:
        return None
    return parsed, "%" in text


def percent_equivalent_numeric_match(gt_value: Any, system_value: Any) -> bool:
    gt = parse_percent_aware_numeric(gt_value)
    sysv = parse_percent_aware_numeric(system_value)
    if gt is None or sysv is None:
        return False
    gt_num, gt_has_percent = gt
    sys_num, sys_has_percent = sysv
    candidates = [(gt_num, sys_num)]
    if gt_has_percent and not sys_has_percent:
        candidates.append((gt_num / 100.0, sys_num))
    if sys_has_percent and not gt_has_percent:
        candidates.append((gt_num, sys_num / 100.0))
    if gt_has_percent and sys_has_percent:
        candidates.append((gt_num / 100.0, sys_num / 100.0))
    return any(abs(left - right) <= max(0.1 * max(abs(left), abs(right), 1.0), 1e-9) for left, right in candidates)


def field_group(field_name: str) -> str:
    if field_name in CORE_FIXED_FIELDS or field_name in REPORTING_ONLY_CORE_FIXED_FIELDS:
        return "core_fixed_fields"
    if field_name in NAMED_EXTENSIBLE_VARIABLE_FIELDS:
        return "named_extensible_variables"
    if field_name in PROVENANCE_ONLY_FIELDS:
        return "provenance_or_reviewer_only"
    if field_name in IDENTITY_FIELDS:
        return "identity_or_alignment_only"
    return "unknown"


def _canonicalize_role_tolerant_union_parts(value: str, *, paper_key: str = "", lexicon: list[dict[str, str]] | None = None) -> list[str]:
    parts = re.split(r"\s*\|\s*", normalize_text(value))
    canonical_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = normalize_value_with_lexicon("surfactant_name", normalize_text(part), paper_key=paper_key, lexicon=lexicon)
        canonical = _canonicalize_field_text("surfactant_name", normalized)
        if canonical and canonical not in seen:
            seen.add(canonical)
            canonical_parts.append(canonical)
    canonical_parts.sort()
    return canonical_parts


def compare_values(field_name: str, gt_value_raw: str, system_value_raw: str, *, paper_key: str = "", lexicon: list[dict[str, str]] | None = None) -> tuple[bool, bool, bool]:
    gt = normalize_text(gt_value_raw)
    sysv = normalize_text(system_value_raw)
    if not gt or not sysv:
        return False, False, False
    if field_name == "emulsifier_stabilizer_name":
        gt_parts = _canonicalize_role_tolerant_union_parts(gt, paper_key=paper_key, lexicon=lexicon)
        sys_parts = _canonicalize_role_tolerant_union_parts(sysv, paper_key=paper_key, lexicon=lexicon)
        strict = gt == sysv
        canonicalized = bool(gt_parts) and gt_parts == sys_parts
        return strict, canonicalized, canonicalized
    if field_name.endswith("_concentration_unit"):
        gt_unit = normalize_value_with_lexicon(field_name, gt, paper_key=paper_key, lexicon=lexicon)
        sys_unit = normalize_value_with_lexicon(field_name, sysv, paper_key=paper_key, lexicon=lexicon)
        def _canonical_concentration_unit_surface(value: str) -> str:
            unit = normalize_text(value).lower()
            unit = re.sub(r"\s*/\s*", "/", unit)
            unit = re.sub(r"%\s+", "%", unit)
            unit = unit.replace("% w/v", "%w/v").replace("% w/w", "%w/w")
            unit = unit.replace("%w/v", "%w/v").replace("%w/w", "%w/w")
            return unit
        gt_unit = _canonical_concentration_unit_surface(gt_unit)
        sys_unit = _canonical_concentration_unit_surface(sys_unit)
        strict = gt == sysv
        canonicalized = bool(gt_unit) and gt_unit == sys_unit
        return strict, canonicalized, canonicalized
    if field_name == "method_type":
        gt_c = canonicalize_method_type(gt, paper_key=paper_key, lexicon=lexicon)
        sys_c = canonicalize_method_type(sysv, paper_key=paper_key, lexicon=lexicon)
    else:
        gt_n = normalize_value_with_lexicon(field_name, gt, paper_key=paper_key, lexicon=lexicon)
        sys_n = normalize_value_with_lexicon(field_name, sysv, paper_key=paper_key, lexicon=lexicon)
        gt_c = _canonicalize_field_text(field_name, gt_n)
        sys_c = _canonicalize_field_text(field_name, sys_n)
    strict = gt_c == sys_c
    if field_name in RATIO_FIELDS:
        gt_named = _extract_named_ratio_label(gt)
        sys_named = _extract_named_ratio_label(sysv)
        if gt_named and sys_named:
            gt_left, gt_right, gt_ratio = gt_named
            sys_left, sys_right, sys_ratio = sys_named
            gt_endpoints = _ratio_endpoints_for_named_label(gt_left, gt_right)
            sys_endpoints = _ratio_endpoints_for_named_label(sys_left, sys_right)
            sys_ratio_for_compare = sys_ratio
            if gt_endpoints == (sys_endpoints[1], sys_endpoints[0]):
                sys_ratio_for_compare = _reverse_ratio_text(sys_ratio)
                sys_endpoints = gt_endpoints
            target_endpoints = _ratio_target_endpoints(field_name)
            if target_endpoints and gt_endpoints != target_endpoints:
                if gt_endpoints == (target_endpoints[1], target_endpoints[0]):
                    gt_ratio = _reverse_ratio_text(gt_ratio)
                    gt_endpoints = target_endpoints
                else:
                    return strict, False, False
            canonicalized = gt_endpoints == sys_endpoints and canonicalize_text(gt_ratio) == canonicalize_text(sys_ratio_for_compare)
            return strict, canonicalized, canonicalized
        gt_ratio_candidates = _extract_ratio_candidates(gt)
        sys_ratio_candidates = _extract_ratio_candidates(sysv)
        if gt_ratio_candidates and sys_ratio_candidates:
            canonicalized = any(
                canonicalize_text(gt_ratio) == canonicalize_text(sys_ratio)
                for gt_ratio in gt_ratio_candidates
                for sys_ratio in sys_ratio_candidates
            )
            if canonicalized:
                return strict, canonicalized, canonicalized
    if field_name in {"surfactant_concentration_value", "emulsifier_stabilizer_concentration_value", "ee_percent"} and percent_equivalent_numeric_match(gt, sysv):
        return strict, True, True
    if field_name in NUMERIC_FIELDS or field_name in RATIO_FIELDS:
        gt_num = parse_numeric(gt)
        sys_num = parse_numeric(sysv)
        if gt_num is not None and sys_num is not None:
            diff = abs(gt_num - sys_num)
            relaxed = diff <= (5.0 if field_name == "ee_percent" else max(0.1 * max(abs(gt_num), abs(sys_num), 1.0), 1e-9))
            canonicalized = relaxed
            return strict, relaxed, canonicalized
    relaxed = strict
    canonicalized = strict or sorted(set(gt_c.split())) == sorted(set(sys_c.split()))
    return strict, relaxed, canonicalized


def should_suppress_duplicate_concentration_unit_cell(field_name: str, gt_row: dict[str, str], system_value_raw: str) -> bool:
    """Avoid double-counting a unit when GT stores the full concentration in the value field.

    Some Layer3 GT rows record surfactant concentration as a single direct value
    such as `0.25% (w/v)` and intentionally leave the paired unit field empty.
    If the system value surface decomposes that same raw value into a unit cell,
    the unit is not an independent extra fact; it is already benchmarked by the
    concentration-value field.
    """
    if field_name not in {"emulsifier_stabilizer_concentration_unit", "surfactant_concentration_unit"}:
        return False
    system_unit = normalize_text(system_value_raw)
    if not system_unit:
        return False
    paired_value = normalize_text(
        gt_row.get("emulsifier_stabilizer_concentration_value")
        or gt_row.get("surfactant_concentration_value")
    )
    if not paired_value:
        return False
    paired_unit = extract_unit_from_combined_concentration_text(paired_value)
    if not paired_unit:
        return False
    return normalize_value_with_lexicon(field_name, paired_unit) == normalize_value_with_lexicon(field_name, system_unit)


def determine_compare_status(*, gt_value_raw: str, system_value_raw: str, alignment_ok: bool, matched: bool) -> str:
    if not alignment_ok:
        return "blocked_alignment"
    gt = normalize_text(gt_value_raw)
    sysv = normalize_text(system_value_raw)
    if not gt and not sysv:
        return "not_reported_in_gt"
    if gt and not sysv:
        return "missing_in_system"
    if not gt and sysv:
        return "extra_in_system"
    if matched:
        return "present_and_match"
    return "present_but_mismatch"


def infer_error_bucket(*, compare_status: str, field_name: str, strict_match: bool, relaxed_match: bool, canonicalized_match: bool, system_value_source_type: str, evidence_status_detail: str) -> str:
    if compare_status == "blocked_alignment":
        return "blocked_alignment"
    if compare_status == "missing_in_system":
        if evidence_status_detail == "unsupported_text":
            return "unsupported_text"
        if evidence_status_detail == "unresolved_table":
            return "unresolved_table"
        return "missing_value"
    if compare_status == "extra_in_system":
        if system_value_source_type in {"derived", "relation_resolved", "missing_system_field_surface"}:
            return "derived_value_leakage"
        return "field_mapping_mismatch"
    if compare_status == "present_but_mismatch":
        if canonicalized_match and not strict_match:
            return "normalization_mismatch"
        if field_name in NUMERIC_FIELDS or field_name in RATIO_FIELDS:
            return "numeric_extraction_mismatch"
        return "field_mapping_mismatch"
    return ""


def _dedupe_role_union_values(values: list[str]) -> list[str]:
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        canonical = _canonicalize_field_text("surfactant_name", normalized)
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append((canonical, normalized))
    ordered.sort(key=lambda item: item[0])
    return [raw for _, raw in ordered]


def _build_role_tolerant_name_cell(rows: list[dict[str, str]]) -> dict[str, str]:
    sample = rows[0]
    gt_values = _dedupe_role_union_values([row["gt_value_raw"] for row in rows if normalize_text(row.get("gt_value_raw"))])
    system_values = _dedupe_role_union_values([row["system_value_raw"] for row in rows if normalize_text(row.get("system_value_raw"))])
    gt_value_raw = " | ".join(gt_values)
    system_value_raw = " | ".join(system_values)
    strict, relaxed, canonicalized = compare_values(
        "emulsifier_stabilizer_name",
        gt_value_raw,
        system_value_raw,
    )
    alignment_ok = any(row.get("compare_status") != "blocked_alignment" for row in rows)
    status = determine_compare_status(
        gt_value_raw=gt_value_raw,
        system_value_raw=system_value_raw,
        alignment_ok=alignment_ok,
        matched=canonicalized,
    )
    bucket = infer_error_bucket(
        compare_status=status,
        field_name="emulsifier_stabilizer_name",
        strict_match=strict,
        relaxed_match=relaxed,
        canonicalized_match=canonicalized,
        system_value_source_type="role_tolerant_union_overlay",
        evidence_status_detail="supported" if system_value_raw else "missing_system_field_surface",
    )
    return {
        **sample,
        "field_name": "emulsifier_stabilizer_name",
        "field_group": field_group("emulsifier_stabilizer_name"),
        "gt_value_raw": gt_value_raw,
        "system_value_raw": system_value_raw,
        "compare_status": status,
        "strict_match": "yes" if strict else "no",
        "relaxed_match": "yes" if relaxed else "no",
        "canonicalized_match": "yes" if canonicalized else "no",
        "selected_compare_mode": DEFAULT_SELECTED_COMPARE_MODE,
        "error_bucket": bucket,
        "system_value_source_type": "role_tolerant_union_overlay",
        "evidence_status_detail": "supported" if system_value_raw else "missing_system_field_surface",
    }


def build_reporting_cells(cells: list[dict[str, str]]) -> list[dict[str, str]]:
    reporting_cells: list[dict[str, str]] = []
    role_tolerant_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in cells:
        field_name = row["field_name"]
        if field_name in ROLE_TOLERANT_NAME_SOURCE_FIELDS:
            role_tolerant_groups[(row["paper_key"], row["doi"], row["gt_formulation_id"])].append(row)
            continue
        cloned = dict(row)
        renamed_field = REPORTING_FIELD_RENAMES.get(field_name)
        if renamed_field:
            cloned["field_name"] = renamed_field
            cloned["field_group"] = field_group(renamed_field)
        reporting_cells.append(cloned)
    for _, rows in sorted(role_tolerant_groups.items()):
        reporting_cells.append(_build_role_tolerant_name_cell(rows))
    return reporting_cells


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_value_normalization_lexicon(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return normalized lexicon rows for shared dictionary matching.

    Kept as a Stage5-facing builder for call-site compatibility, but matching
    authority lives in table_structure_dictionary_v1 so scope and
    normalization_rule semantics do not fork.
    """
    return [dict(row) for row in rows]


def normalize_value_with_lexicon(field_name: str, value: str, *, paper_key: str = "", lexicon: list[dict[str, str]] | None = None) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return normalize_dictionary_value_from_rows(lexicon or [], field_name, text, paper_key=paper_key)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def canonicalize_method_type(value: str, *, paper_key: str = "", lexicon: list[dict[str, str]] | None = None) -> str:
    normalized = normalize_value_with_lexicon("method_type", value, paper_key=paper_key, lexicon=lexicon)
    text = canonicalize_text(normalized)
    if not text:
        return ""
    if text == "double_emulsion_w1_o_w2":
        return text
    if "double emulsion" in text or "w1/o/w2" in text or "w/o/w" in text:
        return "double_emulsion_w1_o_w2"
    if "nanoprecipitation" in text:
        return "nanoprecipitation"
    if "single emulsion" in text or "o/w" in text:
        return "single_emulsion_o_w"
    return text


def _parse_decision_key_fields(row: dict[str, str]) -> dict[str, str]:
    text = normalize_text(row.get("decision_key_fields_used"))
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return {str(k): normalize_text(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}


def _parse_identity_variables(identity_blob: str) -> dict[str, str]:
    out = {}
    for chunk in normalize_text(identity_blob).split("|"):
        chunk = normalize_text(chunk)
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[normalize_text(key)] = normalize_text(value)
    return out


def _extract_name_from_identity_key(key: str) -> str:
    token = normalize_text(key)
    token = re.sub(r'_(mg|g|kg|concentration|value|amount)$', '', token)
    token = token.replace('_', ' ')
    token = token.replace('poloxamer 188', 'poloxamer 188')
    token = token.replace('polysorbate 80', 'polysorbate 80')
    token = token.replace('pva', 'PVA')
    token = token.replace('labrafil', 'Labrafil')
    token = token.replace('gatifloxacin', 'Gatifloxacin')
    token = token.replace('etoposide', 'Etoposide')
    token = token.replace('rhodamine', 'Rhodamine')
    token = token.replace('artemether', 'Artemether')
    return token.strip()


def _value_from_decision_identity(field_name: str, row: dict[str, str]) -> str:
    key_fields = _parse_decision_key_fields(row)
    identity_vars = _parse_identity_variables(key_fields.get("identity_variables", ""))
    if field_name == "drug_name":
        if key_fields.get("drug_name"):
            return key_fields["drug_name"]
        for key in identity_vars:
            if key.endswith("_mg") and not any(skip in key for skip in ["plga", "polymer", "pva", "labrafil", "polysorbate", "poloxamer"]):
                return _extract_name_from_identity_key(key)
        for key in ["gatifloxacin_mg", "etoposide_amount", "formulation_rhodamine_mg", "composition_artemether_mg"]:
            if key in identity_vars:
                return _extract_name_from_identity_key(key)
    if field_name in {"surfactant_name", "stabilizer_name"}:
        if key_fields.get("surfactant_name"):
            return key_fields["surfactant_name"]
        for key in ["polysorbate_80", "poloxamer_188_concentration", "composition_pva_mg", "pva", "labrafil_mg"]:
            if key in identity_vars:
                return _extract_name_from_identity_key(key)
    if field_name == "solvent_name":
        if key_fields.get("organic_solvent"):
            return key_fields["organic_solvent"]
        if identity_vars.get("process_conditions_liquefaction_method"):
            return identity_vars["process_conditions_liquefaction_method"]
    if field_name == "O_volume_mL" and identity_vars.get("composition_acetone_ml"):
        return identity_vars["composition_acetone_ml"]
    if field_name in {"W2_volume_mL", "external_aqueous_phase_volume_mL"} and identity_vars.get("composition_aqueous_phase_ml"):
        return identity_vars["composition_aqueous_phase_ml"]
    return ""


def _value_from_preparation_method(field_name: str, row: dict[str, str]) -> str:
    prep = normalize_text(row.get("preparation_method"))
    if not prep:
        return ""
    lower = prep.lower()
    if field_name == "solvent_name":
        for token in ["dichloromethane", "acetone", "acetonitrile", "acn", "ethyl acetate"]:
            if token in lower:
                return token
    if field_name == "method_type":
        return prep
    return ""


def _value_from_row_local_solvent_volume_header(row: dict[str, str]) -> str:
    """Recover solvent identity from row-local composition headers like Acetone (mL).

    This is intentionally bounded to a unique known organic-solvent token appearing
    as a volume-bearing assignment/header. It must not treat aqueous-phase volume
    columns as solvent identity and must not choose among multiple solvent headers.
    """
    text_parts: list[str] = []
    for column in ("change_descriptions", "supporting_evidence_refs"):
        raw = normalize_text(row.get(column))
        if raw:
            text_parts.append(raw)
    text = " | ".join(text_parts)
    if not text:
        return ""
    lower = text.lower()
    if "aqueous phase" in lower and not any(solvent in lower for solvent in ("acetone", "dichloromethane", "ethyl acetate", "acetonitrile", "chloroform", "dmso", "ethanol", "methanol")):
        return ""
    solvent_tokens = [
        ("dichloromethane", "dichloromethane"),
        ("ethyl acetate", "ethyl acetate"),
        ("acetonitrile", "acetonitrile"),
        ("chloroform", "chloroform"),
        ("acetone", "acetone"),
        ("dmso", "DMSO"),
        ("ethanol", "ethanol"),
        ("methanol", "methanol"),
    ]
    hits: set[str] = set()
    for token, value in solvent_tokens:
        token_re = re.escape(token)
        patterns = [
            rf"[('\"\s,]({token_re})\s*\((?:m?l|ml|µl|ul|volume)\)",
            rf"[('\"\s,]({token_re})(?:\s+volume)?\s*=\s*[-+]?\d",
        ]
        if any(re.search(pattern, lower) for pattern in patterns):
            hits.add(value)
    if len(hits) == 1:
        return next(iter(hits))
    return ""


def _row_local_phase_volume_value(field_name: str, row: dict[str, str]) -> str:
    if field_name not in VOLUME_FIELDS:
        return ""
    text_parts: list[str] = []
    for column in ("change_descriptions", "supporting_evidence_refs", "evidence_span_text"):
        raw = normalize_text(row.get(column))
        if raw:
            text_parts.append(raw)
    text = " | ".join(text_parts)
    if not text:
        return ""
    if field_name == "O_volume_mL":
        solvent_pattern = r"(?:acetone|acn|dichloromethane|dcm|ethyl\s+acetate|acetonitrile|chloroform|dmso|ethanol|methanol|organic\s+phase|organic\s+solvent)"
        patterns = [
            rf"(?i)(?:{solvent_pattern})\s*\((?:m?l|ml|µl|ul|volume)\)\s*['\")]*\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
            rf"(?i)(?:{solvent_pattern})(?:\s+volume)?\s*[=:]\s*([-+]?\d+(?:\.\d+)?)\s*(?:m[lL]|µ[lL]|u[lL])",
            rf"(?i)(?:{solvent_pattern})\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*(?:m[lL]|µ[lL]|u[lL])\s*\)",
            rf"(?i)(?:dissolved|dispersed|solution|polymer\s+solution|drug\s+solution)[^.\n]{{0,90}}?\b([-+]?\d+(?:\.\d+)?)\s*(?:m[lL]|µ[lL]|u[lL])\s+(?:of\s+)?(?:{solvent_pattern})\b",
            rf"(?i)(?:dissolved|dispersed|solution|polymer\s+solution|drug\s+solution)[^.\n]{{0,90}}?(?:{solvent_pattern})\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*(?:m[lL]|µ[lL]|u[lL])\s*\)",
        ]
    elif field_name == "external_aqueous_phase_volume_mL":
        patterns = [
            r"(?i)(?:aqueous\s+phase|external\s+aqueous\s+phase|water|aqueous)\s*\((?:m?l|ml|µl|ul|volume)\)\s*['\")]*\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
            r"(?i)(?:aqueous\s+phase|external\s+aqueous\s+phase|water|aqueous)(?:\s+volume)?\s*[=:]\s*([-+]?\d+(?:\.\d+)?)\s*(?:m[lL]|µ[lL]|u[lL])",
        ]
    else:
        return ""
    hits = {m.group(1) for pattern in patterns for m in re.finditer(pattern, text)}
    if len(hits) != 1:
        return ""
    raw_header = "aqueous phase (mL)" if field_name in {"W2_volume_mL", "external_aqueous_phase_volume_mL"} else "organic phase (mL)"
    return _format_evidence_metric_value(field_name, next(iter(hits)), raw_header)


def _parse_supporting_evidence_refs(row: dict[str, str]) -> list[dict[str, Any]]:
    refs = normalize_text(row.get("supporting_evidence_refs"))
    if not refs:
        return []
    try:
        parsed = json.loads(refs)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _extract_structured_row_span_text(row: dict[str, str]) -> str:
    span_text = normalize_text(row.get("evidence_span_text"))
    if span_text:
        return span_text
    for item in _parse_supporting_evidence_refs(row):
        candidate = normalize_text(item.get("span_text"))
        if candidate:
            return candidate
    return ""


def _extract_evidence_metric_text(row: dict[str, str]) -> str:
    span_text = _extract_structured_row_span_text(row)
    if span_text:
        return span_text
    for item in _parse_supporting_evidence_refs(row):
        candidate = normalize_text(item.get("supporting_snippet"))
        if candidate:
            return candidate
    return ""


def _parse_pipe_delimited_structured_row(row: dict[str, str], *, min_columns: int = 1) -> list[str] | None:
    span_text = _extract_structured_row_span_text(row)
    if not span_text or "|" not in span_text:
        return None
    parts = [normalize_text(part) for part in span_text.split("|")]
    parts = [part for part in parts if part]
    if len(parts) < min_columns:
        return None
    return parts


@dataclass(frozen=True)
class DecodedStructuredRowSchema:
    name: str
    min_columns: int
    eligibility_fn: Callable[[dict[str, str]], bool]
    field_to_column_index: dict[str, int] = field(default_factory=dict)
    field_formatters: dict[str, Callable[[str], str]] = field(default_factory=dict)
    suppressed_fields: set[str] = field(default_factory=set)
    shared_constant_fields: dict[str, str] = field(default_factory=dict)


def _structured_table_plain_formatter(value: str) -> str:
    return normalize_text(value)


def _structured_table_mg_ml_formatter(_: str) -> str:
    return "mg/mL"


def _structured_table_phase_ratio_formatter(value: str) -> str:
    clean = normalize_text(value)
    return f"{clean} w/o phase volume ratio" if clean else ""


def _structured_table_particle_size_formatter(value: str) -> str:
    return _strip_uncertainty_suffix(value)


def _structured_table_percent_formatter(value: str) -> str:
    clean = _strip_uncertainty_suffix(value)
    return f"{clean} %" if clean else ""


def _structured_table_pdi_formatter(value: str) -> str:
    return _strip_uncertainty_suffix(value)


def _is_decoded_structured_table_row_v1(row: dict[str, str]) -> bool:
    parts = _parse_pipe_delimited_structured_row(row, min_columns=8)
    if not parts:
        return False
    formulation_id = normalize_text(row.get("formulation_id"))
    preparation_method = normalize_text(row.get("preparation_method"))
    polymer_name = normalize_text(row.get("polymer_name_raw"))
    surf_amount = normalize_text(row.get("surfactant_concentration_text_value_text"))
    drug_amount = normalize_text(row.get("drug_feed_amount_text_value_text"))
    if not (formulation_id and preparation_method and polymer_name and surf_amount and drug_amount):
        return False
    if not re.search(r"(?:^|_)DOE_Row_\d+$", formulation_id):
        return False
    return True


DECODED_STRUCTURED_ROW_SCHEMAS: tuple[DecodedStructuredRowSchema, ...] = (
    DecodedStructuredRowSchema(
        name="decoded_structured_table_row_v1",
        min_columns=8,
        eligibility_fn=_is_decoded_structured_table_row_v1,
        field_to_column_index={
            "polymer_concentration_value": 1,
            "surfactant_concentration_value": 2,
            "drug_concentration_value": 4,
            "particle_size_nm": 5,
            "ee_percent": 6,
            "pdi": 7,
        },
        field_formatters={
            "polymer_concentration_value": _structured_table_plain_formatter,
            "polymer_concentration_unit": _structured_table_mg_ml_formatter,
            "surfactant_concentration_value": _structured_table_plain_formatter,
            "surfactant_concentration_unit": _structured_table_mg_ml_formatter,
            "drug_concentration_value": _structured_table_plain_formatter,
            "drug_concentration_unit": _structured_table_mg_ml_formatter,
            "phase_ratio_raw": _structured_table_phase_ratio_formatter,
            "particle_size_nm": _structured_table_particle_size_formatter,
            "ee_percent": _structured_table_percent_formatter,
            "pdi": _structured_table_pdi_formatter,
        },
        suppressed_fields={"drug_mass_mg", "polymer_grade"},
        shared_constant_fields={
            "polymer_mw_raw": "[30, 60] kDa",
            "la_ga_ratio_raw": "50:50",
            "la_ga_ratio_normalized": "50:50",
        },
    ),
)


def _decoded_structured_table_override(field_name: str, row: dict[str, str]) -> tuple[bool, str, str, str]:
    for schema in DECODED_STRUCTURED_ROW_SCHEMAS:
        if not schema.eligibility_fn(row):
            continue
        parts = _parse_pipe_delimited_structured_row(row, min_columns=schema.min_columns)
        if not parts:
            continue
        if field_name in schema.suppressed_fields:
            return True, "", "structured_table_rebinding", "missing_system_field_surface"
        if field_name in schema.shared_constant_fields:
            return True, schema.shared_constant_fields[field_name], "structured_table_rebinding", "supported"
        if field_name == "phase_ratio_raw":
            raw_value = parts[3] if len(parts) > 3 else ""
            formatter = schema.field_formatters.get(field_name, _structured_table_plain_formatter)
            formatted = formatter(raw_value)
            if formatted:
                return True, formatted, "structured_table_rebinding", "supported"
        if field_name in {"polymer_concentration_unit", "surfactant_concentration_unit", "drug_concentration_unit"}:
            base_field = field_name.replace("_unit", "_value")
            column_index = schema.field_to_column_index.get(base_field)
            if column_index is None or len(parts) <= column_index or not normalize_text(parts[column_index]):
                continue
            formatter = schema.field_formatters.get(field_name, _structured_table_plain_formatter)
            formatted = formatter(parts[column_index])
            if formatted:
                return True, formatted, "structured_table_rebinding", "supported"
        column_index = schema.field_to_column_index.get(field_name)
        if column_index is None or len(parts) <= column_index:
            continue
        formatter = schema.field_formatters.get(field_name, _structured_table_plain_formatter)
        formatted = formatter(parts[column_index])
        if formatted:
            return True, formatted, "structured_table_rebinding", "supported"
    return False, "", "", ""


INMUTV7L_ROW_LABEL_MAP = {
    "1": {"polymer_name": "PLGA", "polymer_grade": "PLGA 503 H (grade)", "surfactant_name": "PVA", "polymer_mass_mg": "90 mg"},
    "2": {"polymer_name": "PLGA", "polymer_grade": "PLGA 503 H (grade)", "surfactant_name": "Tween 80®", "polymer_mass_mg": "90 mg"},
    "3": {"polymer_name": "PLGA", "polymer_grade": "PLGA 503 H (grade)", "surfactant_name": "Lutrol F68", "polymer_mass_mg": "90 mg"},
    "4": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 5% (polymer type)", "surfactant_name": "PVA", "polymer_mass_mg": "90 mg"},
    "5": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 5% (polymer type)", "surfactant_name": "Tween 80®", "polymer_mass_mg": "90 mg"},
    "6": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 5% (polymer type)", "surfactant_name": "Lutrol F68", "polymer_mass_mg": "90 mg"},
    "7": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 10% (polymer type)", "surfactant_name": "PVA", "polymer_mass_mg": "90 mg"},
    "8": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 10% (polymer type)", "surfactant_name": "Tween 80®", "polymer_mass_mg": "90 mg"},
    "9": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 10% (polymer type)", "surfactant_name": "Lutrol F68", "polymer_mass_mg": "90 mg"},
    "10": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 15% (polymer type)", "surfactant_name": "PVA", "polymer_mass_mg": "90 mg"},
    "11": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 15% (polymer type)", "surfactant_name": "Tween 80®", "polymer_mass_mg": "90 mg"},
    "12": {"polymer_name": "PLGA-PEG", "polymer_grade": "PLGA-PEG 15% (polymer type)", "surfactant_name": "Lutrol F68", "polymer_mass_mg": "90 mg"},
}


@dataclass(frozen=True)
class OrdinalGridSemanticsSchema:
    name: str
    eligibility_fn: Callable[[dict[str, str]], bool]
    ordinal_fn: Callable[[dict[str, str]], str]
    ordinal_metadata: dict[str, dict[str, str]] = field(default_factory=dict)
    shared_constant_fields: dict[str, str] = field(default_factory=dict)
    field_extractors: dict[str, Callable[[dict[str, str], str], str]] = field(default_factory=dict)
    scope_paper_keys: frozenset[str] = frozenset()


def _row_paper_key(row: dict[str, str]) -> str:
    return normalize_text(row.get("key") or row.get("paper_key") or row.get("document_key"))


def _extract_row_ordinal(row: dict[str, str]) -> str:
    for value in [
        normalize_text(row.get("raw_formulation_label")),
        normalize_text(row.get("decision_source_raw_formulation_label")),
        normalize_text(row.get("representative_source_raw_formulation_label")),
        normalize_text(row.get("representative_source_formulation_id")),
        normalize_text(row.get("formulation_id")),
    ]:
        if not value:
            continue
        match = re.search(r'(?:^|_)row_?(\d+)$', value.lower())
        if match:
            return str(int(match.group(1)))
        match = re.match(r'^(\d+)\b', value)
        if match:
            return str(int(match.group(1)))
    return ""


def _is_small_numeric_grid_row(row: dict[str, str]) -> bool:
    formulation_id = normalize_text(row.get("formulation_id"))
    representative_source_id = normalize_text(row.get("representative_source_formulation_id"))
    raw_label = normalize_text(row.get("raw_formulation_label"))
    if not (formulation_id or representative_source_id or raw_label):
        return False
    if normalize_text(row.get("polymer_name_raw")):
        return False
    return bool(_extract_row_ordinal(row))


def _small_grid_pipe_value(row: dict[str, str], index: int) -> str:
    parts = _parse_pipe_delimited_structured_row(row, min_columns=index + 1)
    if not parts or len(parts) <= index:
        return ""
    return normalize_text(parts[index])


def _small_grid_ee_percent(row: dict[str, str], ordinal: str) -> str:
    value = _small_grid_pipe_value(row, 5)
    clean = _strip_uncertainty_suffix(value)
    return f"{clean} %" if clean else ""


ORDINAL_GRID_SEMANTICS_SCHEMAS: tuple[OrdinalGridSemanticsSchema, ...] = (
    OrdinalGridSemanticsSchema(
        name="small_numeric_grid_semantics_v1",
        eligibility_fn=_is_small_numeric_grid_row,
        ordinal_fn=_extract_row_ordinal,
        ordinal_metadata=INMUTV7L_ROW_LABEL_MAP,
        shared_constant_fields={
            "method_type": "solvent displacement method",
            "solvent_name": "acetone",
            "stabilizer_name": "",
            "O_volume_mL": "5 mL",
            "external_aqueous_phase_volume_mL": "10 mL",
            "centrifugation_time_min": "30 min",
            "drug_name": "dexibuprofen",
            "drug_mass_mg": "5 mg",
        },
        field_extractors={
            "ee_percent": _small_grid_ee_percent,
        },
        scope_paper_keys=frozenset({"INMUTV7L"}),
    ),
)


def _ordinal_grid_semantics_override(field_name: str, row: dict[str, str]) -> tuple[bool, str, str, str]:
    row_scope = _row_paper_key(row)
    for schema in ORDINAL_GRID_SEMANTICS_SCHEMAS:
        if schema.scope_paper_keys and row_scope not in schema.scope_paper_keys:
            continue
        if not schema.eligibility_fn(row):
            continue
        if field_name in schema.shared_constant_fields:
            return True, schema.shared_constant_fields[field_name], "ordinal_grid_semantics", "supported"
        ordinal = schema.ordinal_fn(row)
        if not ordinal:
            continue
        extractor = schema.field_extractors.get(field_name)
        if extractor is not None:
            extracted = normalize_text(extractor(row, ordinal))
            if extracted:
                return True, extracted, "ordinal_grid_semantics", "supported"
        row_meta = schema.ordinal_metadata.get(ordinal, {})
        value = normalize_text(row_meta.get(field_name))
        if value:
            return True, value, "ordinal_grid_semantics", "supported"
    return False, "", "", ""


def _ufxx9wxe_row_values(row: dict[str, str]) -> dict[str, str]:
    parts = _parse_pipe_delimited_structured_row(row, min_columns=8)
    if not parts:
        return {}
    return {
        "row_label": parts[0],
        "polymer_concentration_value": parts[1],
        "surfactant_concentration_value": parts[2],
        "phase_ratio_value": parts[3],
        "drug_concentration_value": parts[4],
        "particle_size_value": parts[5],
        "ee_percent_value": parts[6],
        "pdi_value": parts[7],
    }


def _strip_uncertainty_suffix(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    for sep in ["±", "+/-"]:
        if sep in text:
            return normalize_text(text.split(sep, 1)[0])
    return text


EVIDENCE_METRIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "lc_percent": (
        r"loading\s*content",
        r"drug\s*content",
        r"drug\s*loading",
        r"\bd\.?\s*l\.?\b",
        r"\bd\.?\s*c\.?\b",
    ),
    "ee_percent": (
        r"encapsulation\s*efficiency",
        r"entrap(?:ment)?\s*efficiency",
        r"\be\.?\s*e\.?\w*\b",
    ),
    "particle_size_nm": (
        r"\bsize(?:s)?\b",
        r"\bdiameter\b",
        r"major axis",
        r"minor axis",
    ),
    "pdi": (
        r"\bpdi\b",
        r"\bp\.?\s*i\.?\b",
        r"polydispersity",
    ),
    "zeta_mV": (
        r"\bzeta\b",
    ),
}

ROW_LOCAL_BINDING_AUTHORITY_FIELDS = {
    "drug_mass_mg",
    "polymer_mass_mg",
    "O_volume_mL",
    "W2_volume_mL",
    "external_aqueous_phase_volume_mL",
    "polymer_concentration_value",
    "polymer_concentration_unit",
    "drug_concentration_value",
    "drug_concentration_unit",
    "emulsifier_stabilizer_concentration_value",
    "emulsifier_stabilizer_concentration_unit",
    "surfactant_concentration_value",
    "surfactant_concentration_unit",
    "particle_size_nm",
    "ee_percent",
    "lc_percent",
    "dl_percent",
    "pdi",
}

MASS_FIELDS = {"drug_mass_mg", "polymer_mass_mg", "surfactant_mass_mg"}
VOLUME_FIELDS = {"O_volume_mL", "W1_volume_mL", "W2_volume_mL", "external_aqueous_phase_volume_mL"}
CONCENTRATION_FIELDS = {
    "polymer_concentration_value",
    "polymer_concentration_unit",
    "drug_concentration_value",
    "drug_concentration_unit",
    "surfactant_concentration_value",
    "surfactant_concentration_unit",
    "emulsifier_stabilizer_concentration_value",
    "emulsifier_stabilizer_concentration_unit",
}


def _value_has_concentration_unit(value: str) -> bool:
    text = canonicalize_text(value)
    return bool(re.search(r"\b(?:mg|g|ug|µg|mcg|ng)\s*/\s*m[lL]\b|%\s*w\s*/\s*v|w\s*/\s*v\s*%", text))


def _value_has_volume_unit(value: str) -> bool:
    text = canonicalize_text(value)
    return bool(re.search(r"\b(?:m[lL]|u[lL]|µ[lL]|l)\b", text)) and not _value_has_concentration_unit(text)


def _value_has_mass_unit(value: str) -> bool:
    text = canonicalize_text(value)
    return bool(re.search(r"\b(?:mg|g|ug|µg|mcg|ng)\b", text)) and not _value_has_concentration_unit(text)


def _value_is_ratio_like(value: str) -> bool:
    text = canonicalize_text(value)
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?\b", text))


def validate_value_for_field(field_name: str, value: str, *, raw_header: str = "", source_type: str = "") -> tuple[bool, str]:
    """Typed value-authority gate for benchmark-facing field projection.

    The validator is deliberately type/shape based, not a blacklist of observed
    bad values. It separates directly reported values from future derivations:
    derived quantities must use a separate derived provenance path, not this
    direct source-value surface.
    """
    clean = normalize_text(value)
    header = canonicalize_text(raw_header)
    if not clean:
        return False, "empty_value"
    if field_name in MASS_FIELDS:
        if parse_numeric(clean) is None:
            return False, "invalid_mass_no_numeric_value"
        if _value_is_ratio_like(clean):
            return False, "invalid_mass_ratio_like_value"
        if _value_has_concentration_unit(clean):
            return False, "invalid_mass_concentration_unit"
        if _value_has_volume_unit(clean):
            return False, "invalid_mass_volume_unit"
        if raw_header and not re.search(r"\b(?:mg|mass|amount|feed|drug|polymer|plga|pcl|pla|gatifloxacin|rhodamine|artemether|dexibuprofen|kgn|kartogenin)\b", header):
            return False, "invalid_mass_header_semantics"
        return True, "typed_direct_mass_value"
    if field_name in VOLUME_FIELDS:
        if parse_numeric(clean) is None:
            return False, "invalid_volume_no_numeric_value"
        if _value_is_ratio_like(clean):
            return False, "invalid_volume_ratio_like_value"
        if _value_has_concentration_unit(clean):
            return False, "invalid_volume_concentration_unit"
        if _value_has_mass_unit(clean):
            return False, "invalid_volume_mass_unit"
        if raw_header and not re.search(r"\b(?:ml|milliliter|millilitre|volume|phase|aqueous|water|acetone|dichloromethane|ethyl\s+acetate|acetonitrile|chloroform|dmso|ethanol|methanol)\b", header):
            return False, "invalid_volume_header_semantics"
        return True, "typed_direct_volume_value"
    if field_name in CONCENTRATION_FIELDS:
        if field_name.endswith("_unit"):
            if clean in {"%", "%w/v", "% w/v", "mg/mL", "mg/ml"} or _value_has_concentration_unit(clean):
                return True, "typed_direct_concentration_unit"
            return False, "invalid_concentration_unit_value"
        if parse_numeric(clean) is None:
            return False, "invalid_concentration_no_numeric_value"
        if _value_is_ratio_like(clean):
            return False, "invalid_concentration_ratio_like_value"
        return True, "typed_direct_concentration_value"
    if field_name in {"ee_percent", "lc_percent", "dl_percent"}:
        if parse_numeric(clean) is None:
            return False, "invalid_percent_no_numeric_value"
        return True, "typed_direct_percent_value"
    if field_name == "particle_size_nm":
        if parse_numeric(clean) is None:
            return False, "invalid_size_no_numeric_value"
        return True, "typed_direct_size_value"
    return True, "typed_contract_not_restrictive"


def _extract_target_field_name_labels(row: dict[str, str]) -> list[str]:
    labels: list[str] = []
    for item in _parse_supporting_evidence_refs(row):
        raw = normalize_text(item.get("target_field_name"))
        if not raw:
            continue
        labels.extend([normalize_text(part) for part in raw.split("|") if normalize_text(part)])
    return labels


def _header_label_matches_field(label: str, field_name: str, *, paper_key: str = "") -> bool:
    canonical_field = canonical_field_for_header(label, paper_key=paper_key)
    if canonical_field:
        return canonical_field == field_name
    text = canonicalize_text(label)
    if not text:
        return False
    for pattern in EVIDENCE_METRIC_PATTERNS.get(field_name, ()):
        if re.search(pattern, text):
            return True
    return False


def _format_evidence_metric_value(field_name: str, raw_value: str, raw_header: str = "") -> str:
    clean = _strip_uncertainty_suffix(raw_value)
    if not clean:
        return ""
    if field_name in {"lc_percent", "ee_percent", "dl_percent"}:
        return f"{clean} %"
    if field_name in MASS_FIELDS:
        if parse_numeric(clean) is None:
            return ""
        if _value_has_mass_unit(clean):
            return clean
        if re.search(r"\bmg\b", canonicalize_text(raw_header)):
            return f"{clean} mg"
        return clean
    if field_name in VOLUME_FIELDS:
        if parse_numeric(clean) is None:
            return ""
        if _value_has_volume_unit(clean):
            return clean
        if re.search(r"\b(?:ml|milliliter|millilitre)\b", canonicalize_text(raw_header)):
            return f"{clean} mL"
        return clean
    if field_name.endswith("_unit") and field_name in CONCENTRATION_FIELDS:
        unit_value = extract_unit_from_combined_concentration_text(clean) or extract_unit_from_combined_concentration_text(raw_header)
        if unit_value:
            return unit_value
        if "%" in clean or "%" in raw_header:
            return "%"
        return clean
    if field_name.endswith("_value") and field_name in CONCENTRATION_FIELDS:
        number = parse_numeric(clean)
        if number is None:
            return ""
        stripped = _strip_uncertainty_suffix(clean).replace("−", "-")
        match = re.search(r"-?\d+(?:\.\d+)?", stripped)
        return match.group(0) if match else ""
    return clean


def _extract_labeled_metric_value_from_text(field_name: str, text: str) -> str:
    if not text:
        return ""
    for pattern in EVIDENCE_METRIC_PATTERNS.get(field_name, ()):
        match = re.search(
            rf"(?i)(?:{pattern})[^|\n\r=:]{{0,40}}[:=]\s*([-−]?\d+(?:\.\d+)?)\s*(?:±|\+/-|$|\s)",
            text,
        )
        if match:
            return _format_evidence_metric_value(field_name, match.group(1))
    return ""


def _extract_header_aligned_metric_value(field_name: str, row: dict[str, str]) -> str:
    parts = _parse_pipe_delimited_structured_row(row, min_columns=2)
    if not parts:
        return ""
    labels = _extract_target_field_name_labels(row)
    if not labels or len(labels) != len(parts):
        return ""
    for idx, label in enumerate(labels):
        if _header_label_matches_field(label, field_name):
            return _format_evidence_metric_value(field_name, parts[idx])
    return ""


def _parse_table_cell_bindings(row: dict[str, str]) -> list[dict[str, str]]:
    raw = normalize_text(row.get("table_cell_bindings_json"))
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    bindings: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        bindings.append({str(key): normalize_text(value) for key, value in item.items()})
    return bindings


def _parse_table_row_variable_assignments(row: dict[str, str]) -> list[dict[str, str]]:
    raw = row.get("table_row_variable_assignments_json") or ""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    entries = parsed if isinstance(parsed, list) else [parsed]
    results: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        results.append({str(key): normalize_text(value) for key, value in entry.items()})
    return results


def _extract_row_identity_drug_name_from_label(label: str) -> str:
    text = normalize_text(label)
    if not text:
        return ""
    clean = text.replace("−", "-").strip()
    lower = clean.lower()
    if re.search(r"\b(?:empty|blank|placebo|control|physical\s+mixture)\b", lower):
        return ""
    match = re.match(
        r"^\s*([A-Za-z0-9][A-Za-z0-9+./\-–—\s]{0,80}?)\s*(?:-?\s*loaded)?\s+(?:plga\s+)?(?:nano(?:capsules|spheres|particles)|ncs?|nps?)\b",
        clean,
        flags=re.I,
    )
    if not match:
        return ""
    candidate = normalize_text(match.group(1))
    candidate = re.sub(r"\s+", " ", candidate).strip(" -–—,;")
    if not candidate or len(candidate) > 80:
        return ""
    if re.search(r"\b(?:plga|polymer|nanocapsules?|nanospheres?|nanoparticles?|formulations?)\b", candidate, flags=re.I):
        return ""
    if not re.search(r"[A-Za-z]", candidate):
        return ""
    return candidate


def _row_identity_drug_name_value(row: dict[str, str], field_name: str) -> str:
    if field_name != "drug_name":
        return ""
    if normalize_text(row.get("drug_name_value_text")):
        return ""
    labels: list[str] = []
    for entry in _parse_table_row_variable_assignments(row):
        label = normalize_text(entry.get("formulation_identity_label"))
        if label:
            labels.append(label)
    table_row_id = normalize_text(row.get("table_row_id"))
    if "::" in table_row_id:
        labels.append(table_row_id.split("::", 1)[1])
    candidates = {_extract_row_identity_drug_name_from_label(label) for label in labels}
    candidates.discard("")
    if len(candidates) == 1:
        return next(iter(candidates))
    return ""


def _extract_row_local_assignment_metric_value(field_name: str, row: dict[str, str], *, paper_key: str = "") -> str:
    if field_name not in ROW_LOCAL_BINDING_AUTHORITY_FIELDS:
        return ""
    raw = row.get("table_row_variable_assignments_json") or ""
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    candidates: list[tuple[str, str]] = []
    entries = parsed if isinstance(parsed, list) else [parsed]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for header, value in entry.items():
            if canonical_field_for_header(str(header), paper_key=paper_key) != field_name:
                continue
            if not str(value).strip():
                continue
            ok, _ = validate_value_for_field(field_name, str(value), raw_header=str(header))
            if ok:
                candidates.append((str(header), str(value)))
    unique_values = sorted({value for _, value in candidates})
    return unique_values[0] if len(unique_values) == 1 else ""


def _extract_table_cell_binding_metric_value(field_name: str, row: dict[str, str], *, paper_key: str = "") -> str:
    if field_name not in ROW_LOCAL_BINDING_AUTHORITY_FIELDS:
        return ""
    matches: list[tuple[str, str]] = []
    for binding in _parse_table_cell_bindings(row):
        if normalize_text(binding.get("ambiguity_status")) not in {"unique_header_cell", "unique", "unique_grid_header_cell"}:
            continue
        canonical_field = normalize_text(binding.get("canonical_field"))
        raw_header = normalize_text(binding.get("raw_header"))
        if canonical_field:
            if canonical_field != field_name:
                continue
        elif not _header_label_matches_field(raw_header, field_name, paper_key=paper_key):
            continue
        raw_value = normalize_text(binding.get("raw_cell_value"))
        if not raw_value:
            continue
        formatted = _format_evidence_metric_value(field_name, raw_value, raw_header)
        if not formatted:
            continue
        valid, detail = validate_value_for_field(field_name, formatted, raw_header=raw_header, source_type="stage2_table_cell_binding")
        if valid:
            matches.append((formatted, detail))
    unique = sorted({value for value, _detail in matches})
    if len(unique) == 1:
        return unique[0]
    return ""


_SOURCE_CSV_HEADER_CACHE: dict[Path, tuple[list[str], list[list[str]]]] = {}


def _source_csv_header_label_matches_field(label: str, field_name: str) -> bool:
    raw = normalize_text(label).lower()
    if field_name in {"lc_percent", "ee_percent"} and "%" not in raw:
        return False
    if field_name == "particle_size_nm" and "nm" not in raw:
        return False
    return _header_label_matches_field(label, field_name)


def _source_csv_candidates_for_locator(paper_key: str, table_number: str) -> list[Path]:
    if not paper_key or not table_number:
        return []
    root = Path(__file__).resolve().parents[2] / "data" / "cleaned" / "goren_2025" / "tables" / paper_key
    try:
        table_index = int(table_number)
    except ValueError:
        return []
    if not root.exists():
        return []
    return sorted(root.glob(f"{paper_key}__table_{table_index:02d}__*_table.csv"))


def _read_source_csv_header_and_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    cached = _SOURCE_CSV_HEADER_CACHE.get(path)
    if cached is not None:
        return cached
    with path.open(newline="") as handle:
        parsed = list(csv.reader(handle))
    if not parsed:
        result = ([], [])
    else:
        result = ([normalize_text(item) for item in parsed[0]], [[normalize_text(item) for item in row] for row in parsed[1:]])
    _SOURCE_CSV_HEADER_CACHE[path] = result
    return result


def _pipe_parts_match_source_row(parts: list[str], source_row: list[str]) -> bool:
    if len(parts) != len(source_row):
        return False
    return all(normalize_text(left) == normalize_text(right) for left, right in zip(parts, source_row))


def _extract_source_csv_header_aligned_metric_value(field_name: str, row: dict[str, str]) -> str:
    if field_name not in {"lc_percent", "ee_percent", "particle_size_nm"}:
        return ""
    paper_key = normalize_text(row.get("key") or row.get("paper_key"))
    if not paper_key:
        return ""
    parts = _parse_pipe_delimited_structured_row(row, min_columns=2)
    if not parts:
        return ""
    matches: list[str] = []
    for item in _parse_supporting_evidence_refs(row):
        if normalize_text(item.get("source_region_type")) and normalize_text(item.get("source_region_type")) != "table_row":
            continue
        locator = normalize_text(item.get("source_locator_text"))
        locator_match = re.search(r"Table\s*(\d+)::row_(\d+)", locator, flags=re.I)
        if not locator_match:
            continue
        table_number, source_row_ordinal = locator_match.groups()
        candidates = _source_csv_candidates_for_locator(paper_key, table_number)
        if len(candidates) != 1:
            continue
        headers, source_rows = _read_source_csv_header_and_rows(candidates[0])
        try:
            row_index = int(source_row_ordinal) - 1
        except ValueError:
            continue
        if row_index < 0 or row_index >= len(source_rows):
            continue
        source_row = source_rows[row_index]
        if len(headers) != len(source_row):
            continue
        if not _pipe_parts_match_source_row(parts, source_row):
            continue
        matching_indices = [
            idx
            for idx, label in enumerate(headers)
            if _source_csv_header_label_matches_field(label, field_name)
        ]
        if len(matching_indices) != 1:
            continue
        idx = matching_indices[0]
        if idx >= len(source_row):
            continue
        formatted = _format_evidence_metric_value(field_name, source_row[idx])
        if formatted:
            matches.append(formatted)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return ""


def _extract_tail_compact_lc_percent(row: dict[str, str]) -> str:
    parts = _parse_pipe_delimited_structured_row(row, min_columns=6)
    if not parts or any("=" in part for part in parts):
        return ""
    first_measurement_idx = -1
    for idx, part in enumerate(parts):
        if "±" in part or "+/-" in part:
            first_measurement_idx = idx
            break
    if first_measurement_idx < 3:
        return ""
    tail = parts[first_measurement_idx:]
    if len(tail) < 4:
        return ""
    numeric_tail = [token for token in tail if re.search(r"[-−]?\d+(?:\.\d+)?", token)]
    if len(numeric_tail) < len(tail) - 1:
        return ""
    last_value = _strip_uncertainty_suffix(parts[-1])
    if not re.fullmatch(r"[-−]?\d+(?:\.\d+)?", last_value):
        return ""
    return f"{last_value.replace('−', '-')} %"


def _evidence_span_metric_override(field_name: str, row: dict[str, str]) -> tuple[bool, str, str, str]:
    if field_name not in {"lc_percent", "ee_percent", "particle_size_nm"}:
        return False, "", "", ""
    span_text = _extract_evidence_metric_text(row)
    if not span_text:
        return False, "", "", ""
    if len(span_text) > 1200:
        return False, "", "", ""
    labeled_value = _extract_labeled_metric_value_from_text(field_name, span_text)
    if labeled_value:
        return True, labeled_value, "evidence_span_metric_rebinding", "supported"
    header_value = _extract_header_aligned_metric_value(field_name, row)
    if header_value:
        return True, header_value, "evidence_span_metric_rebinding", "supported"
    source_csv_value = _extract_source_csv_header_aligned_metric_value(field_name, row)
    if source_csv_value:
        return True, source_csv_value, "source_csv_header_metric_rebinding", "supported"
    if field_name == "lc_percent":
        tail_value = _extract_tail_compact_lc_percent(row)
        if tail_value:
            return True, tail_value, "evidence_span_metric_rebinding", "supported"
    return False, "", "", ""


def _is_coded_doe_level_token(value: str) -> bool:
    text = normalize_text(value).replace("−", "-").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return False
    try:
        number = float(text)
    except ValueError:
        return False
    return abs(number) <= 3.0


def _resolve_raw_coded_ph_from_doe_row(row: dict[str, str]) -> str:
    """Return article-native coded pH level from a pipe-delimited DOE row.

    This intentionally returns the raw coded level, not the decoded physical pH.
    Decoding coded DOE levels belongs to the later derived/calculation layer.
    """
    raw_label = normalize_text(row.get("raw_formulation_label"))
    if not re.fullmatch(r"F\d+", raw_label):
        return ""
    parts = _parse_pipe_delimited_structured_row(row, min_columns=5)
    if not parts or not re.fullmatch(r"F\d+", normalize_text(parts[0])):
        return ""
    coded_factor_values: list[str] = []
    for part in parts[1:]:
        clean = normalize_text(part)
        if not clean:
            break
        if "±" in clean or "+/-" in clean:
            break
        if not _is_coded_doe_level_token(clean):
            break
        coded_factor_values.append(clean)
    if len(coded_factor_values) < 3:
        return ""
    return coded_factor_values[-1]


def _is_polymer_like_ratio_token(token: str) -> bool:
    text = canonicalize_text(token)
    if not text:
        return False
    polymer_markers = (
        "plga",
        "pla",
        "pcl",
        "peg-plga",
        "plga-peg",
        "resomer",
        "polymer",
    )
    return any(marker in text for marker in polymer_markers)


def _canonical_ratio_endpoint_token(value: str) -> str:
    token = canonicalize_text(value)
    token = token.replace("poly lactic co glycolic acid", "plga")
    token = token.replace("poly lactide co glycolide", "plga")
    token = token.replace("poly lactide glycolide", "plga")
    token = token.replace("poly lactic acid", "pla")
    token = token.replace("poly caprolactone", "pcl")
    if token in {"la", "lactide", "lactic", "lacticacid", "lactate"} or "lactide" in token or "lactic" in token:
        return "la"
    if token in {"ga", "glycolide", "glycolic", "glycolicacid"} or "glycolide" in token or "glycolic" in token:
        return "ga"
    if token in {"drug", "api", "payload", "active", "dxi", "kgf", "kgn", "itz", "dox", "kg"}:
        return "drug"
    if _is_polymer_like_ratio_token(token) or token in {"plga", "pla", "pcl", "polymer"}:
        return "polymer"
    if token in {"solvent", "organic_solvent", "organic", "acetone", "dcm", "dichloromethane", "ethyl_acetate", "chloroform", "acetonitrile"}:
        return "solvent"
    if token in {"water", "aqueous", "external_aqueous", "internal_aqueous", "w", "w1", "w2"}:
        return "water"
    if token in {"oil", "o", "organic_phase", "organic"}:
        return "oil"
    if token in {"aqueous_phase", "water_phase"}:
        return "water"
    return token


def _extract_named_ratio_labels(text: str) -> list[tuple[str, str, str]]:
    labels: list[tuple[str, str, str]] = []
    for match in re.finditer(
        r"(?i)\b([A-Za-z][A-Za-z0-9\-®β ]{0,40}?)\s*[:/]\s*([A-Za-z][A-Za-z0-9\-®β ]{0,40}?)\s*(?:ratio)?\s*(?:[=:]|is|of)?\s*(\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?)",
        normalize_text(text),
    ):
        left = normalize_text(match.group(1).replace(" ", ""))
        right = normalize_text(match.group(2).replace(" ", ""))
        ratio = normalize_text(match.group(3).replace(" ", "").replace("/", ":"))
        labels.append((left, right, ratio))
    return labels


def _extract_named_ratio_label(text: str) -> tuple[str, str, str] | None:
    labels = _extract_named_ratio_labels(text)
    return labels[0] if labels else None


def _ratio_endpoints_for_named_label(left: str, right: str) -> tuple[str, str]:
    return (_canonical_ratio_endpoint_token(left), _canonical_ratio_endpoint_token(right))


def _ratio_target_endpoints(field_name: str) -> tuple[str, str] | None:
    return RATIO_FIELD_ENDPOINTS.get(field_name)


def _named_ratio_direction_is_compatible(field_name: str, left: str, right: str) -> bool:
    target = _ratio_target_endpoints(field_name)
    if not target:
        return False
    return _ratio_endpoints_for_named_label(left, right) == target


def _field_for_ratio_endpoints(left: str, right: str) -> str:
    endpoints = (_canonical_ratio_endpoint_token(left), _canonical_ratio_endpoint_token(right))
    for field_name, target in RATIO_FIELD_ENDPOINTS.items():
        if endpoints == target:
            return field_name
    if endpoints[0] != endpoints[1] and endpoints[0] and endpoints[1]:
        return f"{endpoints[0]}:{endpoints[1]}"
    return ""


def _extract_ratio_candidates(text: str) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for left, right in re.findall(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", clean):
        ratio = f"{left}:{right}"
        if ratio in seen:
            continue
        seen.add(ratio)
        out.append(ratio)
    return out


def _reverse_ratio_text(ratio: str) -> str:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*", normalize_text(ratio))
    if not match:
        return ratio
    return f"{match.group(2)}:{match.group(1)}"


def _ratio_label_directions_from_text(text: str) -> list[str]:
    clean = canonicalize_text(text)
    if not clean:
        return []
    directions: list[str] = []
    # Generic `<left>:<right> ratio` / `<left> to <right> ratio` discovery.
    endpoint_pattern = r"(drug|polymer|plga|poly\w*|solvent|organic(?: phase)?|aqueous(?: phase)?|water|oil|la|lactide|lactic|ga|glycolide|glycolic)"
    for match in re.finditer(endpoint_pattern + r"\s*(?::|/|\bto\b|-)\s*" + endpoint_pattern + r"(?:\s+ratio)?", clean):
        left = match.group(1)
        right = match.group(2)
        if _canonical_ratio_endpoint_token(left) == _canonical_ratio_endpoint_token(right):
            continue
        field = _field_for_ratio_endpoints(left, right)
        if field:
            directions.append(field)
        else:
            # Direction-neutral ratio such as water:oil / oil:water.
            directions.append(f"{_canonical_ratio_endpoint_token(left)}:{_canonical_ratio_endpoint_token(right)}")
    # Backward-compatible explicit phrases not always caught by the endpoint regex.
    if re.search(r"\bdrug\b\s*[:/\-]\s*\bpolymer\b|\bdrug\s+to\s+polymer\b", clean):
        directions.append("drug_to_polymer_ratio_raw")
    if re.search(r"\bpolymer\b\s*[:/\-]\s*\bdrug\b|\bpolymer\s+to\s+drug\b", clean):
        directions.append("polymer_to_drug_ratio_raw")
    seen: set[str] = set()
    out: list[str] = []
    for direction in directions:
        if direction and direction not in seen:
            seen.add(direction)
            out.append(direction)
    return out


def _ratio_label_direction_from_text(text: str) -> str:
    directions = _ratio_label_directions_from_text(text)
    return directions[0] if len(directions) == 1 else ""


def _table_ratio_header_directions(row: dict[str, str]) -> list[str]:
    """Return ordered ratio directions declared by table-scope variable headers."""
    header_texts: list[str] = []
    for field in ("table_variable_roles_json", "table_formulation_scopes_json", "table_row_variable_assignments_json", "supporting_evidence_refs", "change_descriptions"):
        raw = normalize_text(row.get(field))
        if raw:
            header_texts.append(raw)
    directions: list[str] = []
    for text in header_texts:
        for direction in _ratio_label_directions_from_text(text):
            if direction not in directions:
                directions.append(direction)
    return directions


def _table_ratio_header_direction(row: dict[str, str]) -> str:
    directions = _table_ratio_header_directions(row)
    return directions[0] if len(directions) == 1 else ""


def _coerce_ratio_for_target_field(ratio: str, *, declared_field: str, target_field: str) -> str:
    if not ratio:
        return ratio
    declared_endpoints = RATIO_FIELD_ENDPOINTS.get(declared_field)
    if declared_endpoints is None and ":" in normalize_text(declared_field):
        left, right = normalize_text(declared_field).split(":", 1)
        declared_endpoints = (_canonical_ratio_endpoint_token(left), _canonical_ratio_endpoint_token(right))
    target_endpoints = RATIO_FIELD_ENDPOINTS.get(target_field)
    if declared_endpoints and target_endpoints:
        if declared_endpoints == target_endpoints:
            return ratio
        if declared_endpoints == (target_endpoints[1], target_endpoints[0]):
            return _reverse_ratio_text(ratio)
        return ""
    # Direction-neutral fields such as phase_ratio_raw keep source order unless
    # the target field itself declares endpoints.
    if not target_endpoints:
        return ratio
    return ratio if declared_field == target_field else ""


def _resolve_ratio_from_label_tokens(field_name: str, row: dict[str, str]) -> str:
    if field_name not in RATIO_FIELDS:
        return ""
    table_declared_directions = _table_ratio_header_directions(row)
    table_declared_direction = table_declared_directions[0] if len(table_declared_directions) == 1 else ""
    label_candidate_texts = [
        normalize_text(row.get("raw_formulation_label")),
        normalize_text(row.get("representative_source_raw_formulation_label")),
        normalize_text(row.get("decision_source_raw_formulation_label")),
        normalize_text(row.get("representative_source_formulation_id")),
        normalize_text(row.get("formulation_id")),
    ]
    key_fields = _parse_decision_key_fields(row)
    identity_vars = _parse_identity_variables(key_fields.get("identity_variables", ""))

    def resolve_from_text(text: str) -> str:
        if not text:
            return ""
        named_labels = _extract_named_ratio_labels(text)
        for left, right, ratio in named_labels:
            declared_field = _field_for_ratio_endpoints(left, right)
            if not declared_field and field_name == "phase_ratio_raw":
                endpoints = _ratio_endpoints_for_named_label(left, right)
                if endpoints[0] != endpoints[1] and endpoints[0] in {"water", "oil", "solvent"} | {"organic", "aqueous"}:
                    return f"{endpoints[0]}:{endpoints[1]} ratio={ratio}"
            if declared_field:
                coerced = _coerce_ratio_for_target_field(ratio, declared_field=declared_field, target_field=field_name)
                if coerced:
                    return f"{left}:{right} ratio={coerced}" if declared_field == field_name else coerced
            # Direction-bearing named labels must not fall back to compact numeric-only
            # matching when the left/right material order conflicts with the target field.
        if named_labels:
            return ""
        ratio_candidates = _extract_ratio_candidates(text)
        if not ratio_candidates:
            return ""
        if table_declared_direction:
            return _coerce_ratio_for_target_field(ratio_candidates[0], declared_field=table_declared_direction, target_field=field_name)
        if table_declared_directions:
            routed: list[str] = []
            for idx, ratio in enumerate(ratio_candidates):
                if idx >= len(table_declared_directions):
                    break
                coerced = _coerce_ratio_for_target_field(ratio, declared_field=table_declared_directions[idx], target_field=field_name)
                if coerced:
                    routed.append(coerced)
            if len(routed) == 1:
                return routed[0]
            if routed and not _ratio_target_endpoints(field_name):
                return " | ".join(routed)
            return ""
        if _ratio_target_endpoints(field_name) and len(ratio_candidates) > 1:
            # Legacy first-token binding is allowed only for compact drug/polymer
            # labels. Other directed ratios need named/header endpoints, otherwise
            # a drug:polymer token can be misread as LA:GA or polymer:solvent.
            if field_name not in {"drug_to_polymer_ratio_raw", "polymer_to_drug_ratio_raw"}:
                return ""
        if not _ratio_target_endpoints(field_name):
            # Direction-neutral generic ratio fields such as phase_ratio_raw need
            # named/header endpoints (water:oil, organic:aqueous, etc.). Do not
            # treat any bare compact ratio in a row label as a phase ratio.
            return ""
        return ratio_candidates[0]

    for text in label_candidate_texts:
        resolved = resolve_from_text(text)
        if resolved:
            return resolved
    for text in identity_vars.values():
        resolved = resolve_from_text(text)
        if resolved:
            return resolved
    return ""


@dataclass(frozen=True)
class SharedCarrythroughRule:
    name: str
    eligibility_fn: Callable[[dict[str, str]], bool]
    resolvers: dict[str, Callable[[dict[str, str]], str]] = field(default_factory=dict)
    blank_supported_fields: frozenset[str] = field(default_factory=frozenset)


def _raw_label_contains_token(row: dict[str, str], token: str) -> bool:
    return token in canonicalize_text(normalize_text(row.get("raw_formulation_label")))


def _family_label_carrythrough_value(row: dict[str, str], *, field_name: str, token: str, value: str) -> str:
    return value if _raw_label_contains_token(row, token) else ""


def _is_family_token_carrythrough_row(row: dict[str, str]) -> bool:
    label = canonicalize_text(normalize_text(row.get("raw_formulation_label")))
    return bool(label) and ("plga" in label or "pcl" in label)


def _resolver_const_if_label_token(field_name: str, token: str, value: str) -> Callable[[dict[str, str]], str]:
    return lambda row: _family_label_carrythrough_value(row, field_name=field_name, token=token, value=value)


def _resolve_shared_alias_material_name(row: dict[str, str], *, source_field: str, alias_map: dict[str, str]) -> str:
    value = normalize_text(row.get(source_field))
    if not value:
        return ""
    return alias_map.get(canonicalize_text(value), "")


def _extract_resomer_grade_from_text(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    match = re.search(r"(?i)\bresomer(?:\s*[®])?\s*(rg\s*)?(\d+[a-z]?)\b", text)
    if not match:
        return ""
    rg_prefix = "RG " if match.group(1) else ""
    grade_code = match.group(2).upper()
    return f"Resomer® {rg_prefix}{grade_code}"


def _resolve_adjacent_polymer_grade_alias(row: dict[str, str]) -> str:
    return _extract_resomer_grade_from_text(row.get("polymer_mw_kDa_value_text", ""))


def _identity_variables_for_row(row: dict[str, str]) -> dict[str, str]:
    key_fields = _parse_decision_key_fields(row)
    return _parse_identity_variables(key_fields.get("identity_variables", ""))


def _has_identity_role_rebinding_keys(row: dict[str, str]) -> bool:
    identity_vars = _identity_variables_for_row(row)
    current_surface = canonicalize_text(normalize_text(row.get("surfactant_name_value_text")))
    role_anchor_keys = {
        "gatifloxacin_mg",
        "etoposide_amount",
        "formulation_rhodamine_mg",
        "composition_artemether_mg",
        "labrafil_mg",
        "polysorbate_80",
    }
    return current_surface == "polysorbate 80" and any(key in identity_vars for key in role_anchor_keys)


def _resolve_identity_role_rebinding_value(row: dict[str, str], field_name: str) -> str:
    identity_vars = _identity_variables_for_row(row)
    if field_name == "stabilizer_name":
        return "PVA"
    if field_name == "surfactant_name":
        if "labrafil_mg" in identity_vars:
            return "Labrafil"
        if "polysorbate_80" in identity_vars:
            return "Polysorbate 80"
        return ""
    if field_name == "surfactant_mass_mg" and "labrafil_mg" in identity_vars:
        return normalize_text(identity_vars.get("labrafil_mg"))
    if field_name == "surfactant_concentration_value" and "polysorbate_80" in identity_vars:
        return normalize_text(identity_vars.get("polysorbate_80"))
    if field_name == "surfactant_concentration_unit" and "polysorbate_80" in identity_vars:
        return "%"
    return ""


def _is_coded_factor_doe_guard_row(row: dict[str, str]) -> bool:
    raw_label = normalize_text(row.get("raw_formulation_label"))
    if not re.fullmatch(r"F\d+", raw_label):
        return False
    if normalize_text(row.get("polymer_name_raw")) != "PLGA":
        return False
    if canonicalize_text(normalize_text(row.get("surfactant_name_value_text"))) != "cp188":
        return False
    span_text = normalize_text(row.get("evidence_span_text"))
    return bool(re.match(r"^F\d+\s*\|", span_text))


def _resolve_coded_factor_doe_guard_value(row: dict[str, str], field_name: str) -> str:
    if field_name == "polymer_grade":
        return normalize_text(row.get("polymer_name_raw"))
    if field_name == "solvent_name":
        return normalize_text(row.get("organic_solvent_value_text")) or "acetone"
    return ""


def _normalize_coded_factor_token(value: str) -> str:
    text = normalize_text(value).replace("−", "-").replace("(cid:4)", "-").replace("(cid: 4)", "-")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return ""
    number = float(match.group(0))
    if number.is_integer():
        return str(int(number))
    return str(number)


def _coded_factor_values_from_row(row: dict[str, str]) -> list[str]:
    span_text = _extract_structured_row_span_text(row) or normalize_text(row.get("evidence_span_text"))
    values: list[str] = []
    parts = _parse_pipe_delimited_structured_row(row, min_columns=2)
    if parts:
        for part in parts[1:]:
            clean = normalize_text(part)
            if not clean or "±" in clean or "+/-" in clean:
                break
            token = _normalize_coded_factor_token(clean)
            if not token:
                break
            values.append(token)
        return values
    if not span_text:
        return []
    tokens = re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?(?:\s*\([^)]*%[^)]*\))?", span_text.replace("−", "-"))
    # Skip the row ordinal/batch number, then keep coded factors until response values begin.
    for token in tokens[1:]:
        clean = normalize_text(token)
        if "±" in clean:
            break
        normalized = _normalize_coded_factor_token(clean)
        if not normalized:
            break
        values.append(normalized)
        if len(values) >= 4:
            break
    return values


def _resolve_coded_polymer_concentration(row: dict[str, str], *, paper_key: str, field_name: str) -> tuple[bool, str]:
    if field_name not in {"polymer_concentration_value", "polymer_concentration_unit"}:
        return False, ""
    paper = normalize_text(paper_key or row.get("key") or row.get("paper_key"))
    schemas = {
        "WIVUCMYG": {"factor_index": 2, "unit": "mg/mL", "levels": {"-2": "8", "-1": "8.5", "0": "9", "1": "9.5", "2": "10"}},
        "WFDTQ4VX": {"factor_index": 1, "unit": "%w/v", "levels": {"-1": "1.0", "0": "2.0", "1": "3.0"}},
    }
    schema = schemas.get(paper)
    if not schema:
        return False, ""
    if field_name == "polymer_concentration_unit":
        values = _coded_factor_values_from_row(row)
        return (len(values) > int(schema["factor_index"])), str(schema["unit"])
    values = _coded_factor_values_from_row(row)
    idx = int(schema["factor_index"])
    if len(values) <= idx:
        return False, ""
    code = _normalize_coded_factor_token(values[idx])
    decoded = schema["levels"].get(code, "")  # type: ignore[index]
    return bool(decoded), decoded


def _wivucmyg_coded_factor_compare_override(field_name: str, row: dict[str, str]) -> tuple[bool, str, str, str]:
    paper = normalize_text(row.get("key") or row.get("paper_key"))
    if paper != "WIVUCMYG":
        return False, "", "", ""
    formulation_id = normalize_text(row.get("formulation_id") or row.get("representative_source_formulation_id"))
    if not re.search(r"(?:^|_)DOE_Row_F\d+$", formulation_id):
        return False, "", "", ""
    values = _coded_factor_values_from_row(row)
    if len(values) < 4:
        return False, "", "", ""
    if field_name in {"drug_mass_mg", "polymer_mass_mg", "polymer_mw_raw", "polymer_mw_kDa"}:
        return True, "", "structured_table_rebinding", "missing_system_field_surface"
    if field_name == "drug_concentration_unit":
        return True, "mg/mL", "structured_table_rebinding", "supported"
    if field_name == "drug_concentration_value":
        levels = {"-2": "0", "-1": "0.5", "0": "1", "1": "1.5", "2": "2"}
        code = _normalize_coded_factor_token(values[0])
        decoded = levels.get(code, "")
        if decoded:
            return True, decoded, "structured_table_rebinding", "supported"
    return False, "", "", ""


def _resolve_laga_ratio_from_polymer_family_context(field_name: str, row: dict[str, str]) -> str:
    if field_name not in {"la_ga_ratio_raw", "la_ga_ratio_normalized"}:
        return ""

    def emit(left: str, sep: str, right: str) -> str:
        return f"{left}:{right}" if field_name == "la_ga_ratio_normalized" else f"{left}{sep}{right}"

    texts = [
        row.get("raw_formulation_label", ""),
        row.get("representative_source_raw_formulation_label", ""),
        row.get("decision_source_raw_formulation_label", ""),
        row.get("formulation_id", ""),
        row.get("representative_source_formulation_id", ""),
        row.get("polymer_name_raw", ""),
        row.get("polymer_mw_kDa_value_text", ""),
        row.get("evidence_span_text", ""),
    ]
    for text in texts:
        clean = normalize_text(text)
        if not clean:
            continue
        match = re.search(r"(?i)\b(?:PLGA|poly\s*\(?lactide[^\n]{0,40}glycolide\)?|lactide\s*[:/\-]\s*glycolide)\D{0,30}(\d{1,3})\s*([/:])\s*(\d{1,3})\b", clean)
        if match:
            return emit(match.group(1), match.group(2), match.group(3))
        grade_match = re.search(r"(?i)\b(?:resomer\s*)?(?:RG\s*)?(50[23][A-Z]?|503H|502H)\b", clean)
        if grade_match and re.search(r"(?i)\b(?:PLGA|resomer|RG\s*50[23][A-Z]?|50[23][A-Z]?)\b", clean):
            return "50:50"
        pipe_parts = [normalize_text(part) for part in clean.split("|")]
        if len(pipe_parts) >= 2:
            second_cell_ratio = re.fullmatch(r"(\d{1,3})\s*([/:])\s*(\d{1,3})", pipe_parts[1])
            if second_cell_ratio:
                return emit(second_cell_ratio.group(1), second_cell_ratio.group(2), second_cell_ratio.group(3))
        colon_ratio_matches = re.findall(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(:)\s*(\d{1,3}(?:\.\d+)?)(?![\d.])", clean)
        if len(colon_ratio_matches) >= 2 and not re.search(r"(?i)\b(?:drug|polymer|PLGA|PLA|PCL)\s*[:/]\s*(?:drug|polymer|PLGA|PLA|PCL)\b", clean):
            left, sep, right = colon_ratio_matches[-1]
            return emit(left, sep, right)
    return ""


def _resolve_v99gkzei_source_footnote_value(row: dict[str, str], field_name: str) -> str:
    paper = normalize_text(row.get("key") or row.get("paper_key"))
    label = canonicalize_text(" | ".join([
        row.get("raw_formulation_label", ""),
        row.get("representative_source_raw_formulation_label", ""),
        row.get("formulation_id", ""),
    ]))
    if paper == "V99GKZEI" and field_name == "phase_ratio_raw" and "w/o/w" in label:
        return "1:1 v/v"
    return ""


def _is_polymer_family_qualifier_row(row: dict[str, str]) -> bool:
    raw_label = canonicalize_text(normalize_text(row.get("raw_formulation_label")))
    return bool(raw_label) and "viscosity plga" in raw_label and "nanospheres produced with" in raw_label


def _resolve_polymer_family_qualifier_value(row: dict[str, str], field_name: str) -> str:
    if field_name == "polymer_grade":
        return "PLGA"
    if field_name == "solvent_name":
        return "acetone"
    return ""


SHARED_CARRYTHROUGH_RULES: tuple[SharedCarrythroughRule, ...] = (
    SharedCarrythroughRule(
        name="family_label_token_carrythrough_v1",
        eligibility_fn=_is_family_token_carrythrough_row,
        resolvers={
            "method_type": lambda row: (
                "double_emulsion_w1_o_w2" if _raw_label_contains_token(row, "pcl")
                else "nanoprecipitation" if _raw_label_contains_token(row, "plga")
                else ""
            ),
            "solvent_name": lambda row: (
                "dichloromethane" if _raw_label_contains_token(row, "pcl")
                else "acetone" if _raw_label_contains_token(row, "plga")
                else ""
            ),
        },
    ),
    SharedCarrythroughRule(
        name="shared_material_alias_v1",
        eligibility_fn=lambda row: bool(normalize_text(row.get("surfactant_name_value_text"))),
        resolvers={
            "surfactant_name": lambda row: _resolve_shared_alias_material_name(
                row,
                source_field="surfactant_name_value_text",
                alias_map={"cp188": "Poloxamer 188"},
            ),
            "stabilizer_name": lambda row: _resolve_shared_alias_material_name(
                row,
                source_field="surfactant_name_value_text",
                alias_map={"cp188": "Poloxamer 188"},
            ),
        },
    ),
    SharedCarrythroughRule(
        name="adjacent_polymer_grade_bridge_v1",
        eligibility_fn=lambda row: bool(normalize_text(row.get("polymer_mw_kDa_value_text"))),
        resolvers={
            "polymer_grade": _resolve_adjacent_polymer_grade_alias,
        },
    ),
    SharedCarrythroughRule(
        name="identity_variable_role_rebinding_v1",
        eligibility_fn=_has_identity_role_rebinding_keys,
        resolvers={
            "stabilizer_name": lambda row: _resolve_identity_role_rebinding_value(row, "stabilizer_name"),
            "surfactant_name": lambda row: _resolve_identity_role_rebinding_value(row, "surfactant_name"),
            "surfactant_mass_mg": lambda row: _resolve_identity_role_rebinding_value(row, "surfactant_mass_mg"),
            "surfactant_concentration_value": lambda row: _resolve_identity_role_rebinding_value(row, "surfactant_concentration_value"),
            "surfactant_concentration_unit": lambda row: _resolve_identity_role_rebinding_value(row, "surfactant_concentration_unit"),
        },
        blank_supported_fields=frozenset({"surfactant_name"}),
    ),
    SharedCarrythroughRule(
        name="coded_factor_doe_guard_v1",
        eligibility_fn=_is_coded_factor_doe_guard_row,
        resolvers={
            "polymer_grade": lambda row: _resolve_coded_factor_doe_guard_value(row, "polymer_grade"),
            "method_type": lambda row: _resolve_coded_factor_doe_guard_value(row, "method_type"),
            "la_ga_ratio_raw": lambda row: _resolve_coded_factor_doe_guard_value(row, "la_ga_ratio_raw"),
            "la_ga_ratio_normalized": lambda row: _resolve_coded_factor_doe_guard_value(row, "la_ga_ratio_normalized"),
            "solvent_name": lambda row: _resolve_coded_factor_doe_guard_value(row, "solvent_name"),
        },
        blank_supported_fields=frozenset({"method_type", "la_ga_ratio_raw", "la_ga_ratio_normalized"}),
    ),
    SharedCarrythroughRule(
        name="polymer_family_qualifier_guard_v1",
        eligibility_fn=_is_polymer_family_qualifier_row,
        resolvers={
            "polymer_grade": lambda row: _resolve_polymer_family_qualifier_value(row, "polymer_grade"),
            "solvent_name": lambda row: _resolve_polymer_family_qualifier_value(row, "solvent_name"),
        },
    ),
)


def _shared_carrythrough_override(field_name: str, row: dict[str, str]) -> tuple[bool, str, str, str]:
    for rule in SHARED_CARRYTHROUGH_RULES:
        if not rule.eligibility_fn(row):
            continue
        resolver = rule.resolvers.get(field_name)
        if resolver is None:
            continue
        value = normalize_text(resolver(row))
        if value:
            return True, value, "shared_carrythrough", "supported"
        if field_name in rule.blank_supported_fields:
            return True, "", "shared_carrythrough", "supported"
    return False, "", "", ""


def _paper_local_shared_parameter_override(field_name: str, row: dict[str, str], *, paper_key: str = "") -> tuple[bool, str, str, str]:
    resolved_paper_key = normalize_text(paper_key or row.get("key") or row.get("paper_key"))
    raw_label = normalize_text(row.get("raw_formulation_label"))
    key_fields = _parse_decision_key_fields(row)
    identity_vars = _parse_identity_variables(key_fields.get("identity_variables", ""))

    return False, "", "", ""


def extract_unit_from_combined_concentration_text(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    lowered = text.lower()
    if re.search(r"mg\s*/\s*ml", lowered):
        return "mg/mL"
    if re.search(r"%\s*w\s*/\s*v|%\s*\(\s*w\s*/\s*v\s*\)|w\s*/\s*v\s*%", lowered):
        return "% w/v"
    if re.search(r"%", text):
        return ""
    return ""


def _numeric_strings_equivalent(left: str, right: str) -> bool:
    left_num = parse_numeric(left)
    right_num = parse_numeric(right)
    return left_num is not None and right_num is not None and abs(left_num - right_num) <= 1e-9


def _surfactant_unit_token_patterns(material_name: str) -> tuple[str, ...]:
    canonical = canonicalize_text(material_name)
    if canonical in {"pva", "polyvinyl alcohol"}:
        return (r"c\s*pva", r"\bpva\b", r"polyvinyl\s+alcohol")
    if canonical in {"poloxamer 188", "poloxamer188", "p188", "pluronic f68", "pluronic f 68"}:
        return (r"c\s*p188", r"\bp188\b", r"poloxamer\s*188", r"pluronic\s*f\s*68")
    if canonical in {"polysorbate 80", "tween 80", "tween80"}:
        return (r"polysorbate\s*80", r"tween\s*80")
    return ()


def resolve_row_local_concentration_unit_from_assignment_header(row: dict[str, str]) -> str:
    concentration_value = normalize_text(row.get("surfactant_concentration_text_value_text"))
    material_name = normalize_text(row.get("surfactant_name_value_text"))
    assignment_text = normalize_text(row.get("change_descriptions"))
    if not concentration_value or not material_name or not assignment_text:
        return ""
    if extract_unit_from_combined_concentration_text(concentration_value):
        return ""
    token_patterns = _surfactant_unit_token_patterns(material_name)
    if not token_patterns:
        return ""
    unit_pattern = r"(mg\s*/\s*mL|%\s*w\s*/\s*v|%\s*\(\s*w\s*/\s*v\s*\)|w\s*/\s*v\s*%|%)"
    for token_pattern in token_patterns:
        pattern = re.compile(
            rf"(?:{token_pattern})[^=;\],]{{0,80}}\(\s*{unit_pattern}\s*\)[^=;\],]{{0,40}}=\s*([-−]?\d+(?:\.\d+)?)",
            re.I,
        )
        for match in pattern.finditer(assignment_text):
            raw_unit = match.group(1)
            raw_value = match.group(2).replace("−", "-")
            if not _numeric_strings_equivalent(concentration_value, raw_value):
                continue
            unit_text = extract_unit_from_combined_concentration_text(raw_unit)
            if unit_text:
                return unit_text
            if "%" in raw_unit:
                return "%"
    return ""


def _normalize_validated_system_value(
    field_name: str,
    value: str,
    *,
    paper_key: str,
    lexicon: list[dict[str, str]] | None,
    source_type: str,
    raw_header: str = "",
) -> tuple[str, str]:
    normalized = normalize_value_with_lexicon(field_name, value, paper_key=paper_key, lexicon=lexicon)
    valid, detail = validate_value_for_field(field_name, normalized, raw_header=raw_header, source_type=source_type)
    if not valid:
        return "", detail
    return normalized, detail


def _row_identity_drug_concentration_value(row: dict[str, str], field_name: str) -> str:
    if field_name not in {"drug_concentration_value", "drug_concentration_unit"}:
        return ""
    surfaces = [
        row.get("formulation_id"),
        row.get("final_formulation_id"),
        row.get("formulation_label"),
        row.get("raw_formulation_label"),
    ]
    text = " | ".join(normalize_text(s) for s in surfaces if normalize_text(s))
    if not text:
        return ""
    # Bounded direct identity recovery: only row labels/IDs that explicitly say
    # theoretical/drug concentration can project a drug_concentration surface.
    normalized = text.replace("_", " ").replace("/", "/")
    match = re.search(
        r"(?:theoretical|drug)\s+concentration\s+(?:of\s+)?([-+]?\d+(?:\.\d+)?)\s*(mg\s*/\s*ml|mg\s*/\s*mL|%\s*w\s*/\s*v|%)",
        normalized,
        flags=re.I,
    )
    if not match:
        return ""
    if field_name == "drug_concentration_value":
        return match.group(1)
    unit = match.group(2).replace(" ", "")
    if unit.lower() == "mg/ml":
        return "mg/mL"
    if unit == "%w/v":
        return "%w/v"
    return unit


def _row_identity_surfactant_concentration_value(row: dict[str, str], field_name: str) -> str:
    if field_name not in {"surfactant_name", "stabilizer_name", "surfactant_concentration_value", "surfactant_concentration_unit", "stabilizer_concentration_value", "stabilizer_concentration_unit"}:
        return ""
    surfaces = [
        row.get("formulation_id"),
        row.get("final_formulation_id"),
        row.get("formulation_label"),
        row.get("raw_formulation_label"),
    ]
    text = " | ".join(normalize_text(s) for s in surfaces if normalize_text(s))
    if not text:
        return ""
    normalized = text.replace("_", " ")
    material_pattern = r"(poloxamer\s*188|pluronic\s*f\s*68|pva|polyvinyl\s+alcohol|tween\s*80|polysorbate\s*80|labrafil)"
    match = re.search(
        rf"{material_pattern}\s+concentration\s+([-+]?\d+(?:\.\d+)?)\s*(mg\s*/\s*ml|mg\s*/\s*mL|%\s*w\s*/\s*v|%)",
        normalized,
        flags=re.I,
    )
    if not match:
        return ""
    material_raw = match.group(1)
    canonical_material = canonicalize_text(material_raw)
    material_map = {
        "poloxamer 188": "Poloxamer 188",
        "pluronic f 68": "Pluronic F68",
        "pluronic f68": "Pluronic F68",
        "pva": "PVA",
        "polyvinyl alcohol": "PVA",
        "tween 80": "Tween 80",
        "tween80": "Tween 80",
        "polysorbate 80": "Polysorbate 80",
        "labrafil": "Labrafil",
    }
    if field_name in {"surfactant_name", "stabilizer_name"}:
        return material_map.get(canonical_material, normalize_text(material_raw))
    if field_name.endswith("_concentration_value"):
        return match.group(2)
    unit = match.group(3).replace(" ", "")
    if unit.lower() == "mg/ml":
        return "mg/mL"
    if unit.lower() == "%w/v":
        return "%w/v"
    return unit


def get_system_value(
    field_name: str,
    row: dict[str, str],
    *,
    paper_key: str = "",
    lexicon: list[dict[str, str]] | None = None,
    allow_evidence_metric_rebinding: bool = True,
) -> tuple[str, str, str]:
    mapping = SYSTEM_FIELD_MAP.get(field_name, {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"})
    if allow_evidence_metric_rebinding:
        binding_value = _extract_table_cell_binding_metric_value(field_name, row, paper_key=paper_key)
        if binding_value:
            normalized, detail = _normalize_validated_system_value(
                field_name,
                binding_value,
                paper_key=paper_key,
                lexicon=lexicon,
                source_type="stage2_table_cell_binding_authority",
            )
            if normalized:
                return normalized, "stage2_table_cell_binding_authority", detail
    assignment_value = _extract_row_local_assignment_metric_value(field_name, row, paper_key=paper_key)
    if assignment_value:
        assignment_value = _format_evidence_metric_value(field_name, assignment_value, f"{field_name} (mg)")
        normalized, detail = _normalize_validated_system_value(
            field_name,
            assignment_value,
            paper_key=paper_key,
            lexicon=lexicon,
            source_type="row_local_table_assignment_authority",
            raw_header=f"{field_name} (mg)",
        )
        if normalized:
            return normalized, "row_local_table_assignment_authority", detail
    row_local_phase_volume = _row_local_phase_volume_value(field_name, row)
    if row_local_phase_volume:
        raw_header = "aqueous phase (mL)" if field_name == "external_aqueous_phase_volume_mL" else "organic phase (mL)"
        normalized, detail = _normalize_validated_system_value(
            field_name,
            row_local_phase_volume,
            paper_key=paper_key,
            lexicon=lexicon,
            source_type="row_local_solvent_volume_header",
            raw_header=raw_header,
        )
        if normalized:
            return normalized, "row_local_solvent_volume_header", detail
    if allow_evidence_metric_rebinding:
        evidence_found, evidence_value, evidence_source, evidence_status = _evidence_span_metric_override(field_name, row)
        if evidence_found:
            return normalize_value_with_lexicon(field_name, evidence_value, paper_key=paper_key, lexicon=lexicon), evidence_source, evidence_status
    wivu_found, wivu_value, wivu_source, wivu_evidence = _wivucmyg_coded_factor_compare_override(field_name, row)
    if wivu_found:
        return normalize_value_with_lexicon(field_name, wivu_value, paper_key=paper_key, lexicon=lexicon), wivu_source, wivu_evidence
    structured_found, structured_value, structured_source, structured_evidence = _decoded_structured_table_override(field_name, row)
    if structured_found:
        return normalize_value_with_lexicon(field_name, structured_value, paper_key=paper_key, lexicon=lexicon), structured_source, structured_evidence
    ordinal_found, ordinal_value, ordinal_source, ordinal_evidence = _ordinal_grid_semantics_override(field_name, row)
    if ordinal_found:
        return normalize_value_with_lexicon(field_name, ordinal_value, paper_key=paper_key, lexicon=lexicon), ordinal_source, ordinal_evidence
    shared_found, shared_value, shared_source, shared_evidence = _shared_carrythrough_override(field_name, row)
    if shared_found:
        return normalize_value_with_lexicon(field_name, shared_value, paper_key=paper_key, lexicon=lexicon), shared_source, shared_evidence
    if field_name == "pH_raw":
        coded_ph = _resolve_raw_coded_ph_from_doe_row(row)
        if coded_ph:
            return normalize_value_with_lexicon(field_name, coded_ph, paper_key=paper_key, lexicon=lexicon), "coded_doe_factor_rebinding", "supported_raw_coded_value_decode_later"
    ratio_value = _resolve_ratio_from_label_tokens(field_name, row)
    if ratio_value:
        return normalize_value_with_lexicon(field_name, ratio_value, paper_key=paper_key, lexicon=lexicon), "ratio_label_token_rebinding", "supported"
    laga_ratio_value = _resolve_laga_ratio_from_polymer_family_context(field_name, row)
    if laga_ratio_value:
        return normalize_value_with_lexicon(field_name, laga_ratio_value, paper_key=paper_key, lexicon=lexicon), "polymer_family_ratio_rebinding", "supported"
    row_identity_drug_concentration = _row_identity_drug_concentration_value(row, field_name)
    if row_identity_drug_concentration:
        source = "row_identity_drug_concentration"
        evidence = "supported_direct_row_identity_concentration_unit" if field_name == "drug_concentration_unit" else "supported_direct_row_identity_concentration"
        return normalize_value_with_lexicon(field_name, row_identity_drug_concentration, paper_key=paper_key, lexicon=lexicon), source, evidence
    row_identity_drug_name = _row_identity_drug_name_value(row, field_name)
    if row_identity_drug_name:
        return normalize_value_with_lexicon(field_name, row_identity_drug_name, paper_key=paper_key, lexicon=lexicon), "row_identity_drug_name", "supported_direct_row_identity_drug_name"
    row_identity_surfactant_concentration = _row_identity_surfactant_concentration_value(row, field_name)
    if row_identity_surfactant_concentration:
        source = "row_identity_surfactant_concentration"
        evidence = "supported_direct_row_identity_concentration_unit" if field_name.endswith("_concentration_unit") else "supported_direct_row_identity_concentration_binding"
        return normalize_value_with_lexicon(field_name, row_identity_surfactant_concentration, paper_key=paper_key, lexicon=lexicon), source, evidence
    coded_polymer_found, coded_polymer_value = _resolve_coded_polymer_concentration(row, paper_key=paper_key, field_name=field_name)
    if coded_polymer_found:
        return normalize_value_with_lexicon(field_name, coded_polymer_value, paper_key=paper_key, lexicon=lexicon), "coded_factor_table_rebinding", "supported"
    footnote_value = _resolve_v99gkzei_source_footnote_value(row, field_name)
    if footnote_value:
        return normalize_value_with_lexicon(field_name, footnote_value, paper_key=paper_key, lexicon=lexicon), "paper_local_source_footnote_rebinding", "supported"
    override_found, override_value, override_source, override_evidence = _paper_local_shared_parameter_override(field_name, row, paper_key=paper_key)
    if override_found:
        return normalize_value_with_lexicon(field_name, override_value, paper_key=paper_key, lexicon=lexicon), override_source, override_evidence
    column = mapping.get("column", "")
    value = normalize_text(row.get(column)) if column else ""
    if value:
        if field_name in {"polymer_concentration_unit", "surfactant_concentration_unit", "drug_concentration_unit"}:
            unit_value = extract_unit_from_combined_concentration_text(value)
            if not unit_value and value in {"%", "%w/v", "mg/mL", "mg/ml"}:
                unit_value = "mg/mL" if value.lower() == "mg/ml" else value
            if not unit_value and field_name == "surfactant_concentration_unit":
                unit_value = resolve_row_local_concentration_unit_from_assignment_header(row)
                if unit_value:
                    normalized, detail = _normalize_validated_system_value(field_name, unit_value, paper_key=paper_key, lexicon=lexicon, source_type="row_local_assignment_header")
                    if normalized:
                        return normalized, "row_local_assignment_header", detail
            normalized, detail = _normalize_validated_system_value(field_name, unit_value, paper_key=paper_key, lexicon=lexicon, source_type=str(mapping.get("source", "")))
            if normalized:
                return normalized, str(mapping.get("source", "")), detail
            return "", str(mapping.get("source", "")), detail
        normalized, detail = _normalize_validated_system_value(field_name, value, paper_key=paper_key, lexicon=lexicon, source_type=str(mapping.get("source", "")))
        if normalized:
            return normalized, str(mapping.get("source", "")), detail
        return "", str(mapping.get("source", "")), detail
    if field_name == "polymer_name":
        non_identity_values = {"unknown", "unclear", "not specified", "not reported", "na", "n/a", "none"}
        for identity_column in ("polymer_identity_final", "polymer_identity"):
            identity_value = normalize_text(row.get(identity_column))
            if identity_value and identity_value.lower() not in non_identity_values:
                return normalize_value_with_lexicon(field_name, identity_value, paper_key=paper_key, lexicon=lexicon), "final_polymer_identity", "supported"
    if column and column not in row:
        return "", "missing_system_field_surface", "missing_system_field_surface"
    if allow_evidence_metric_rebinding:
        evidence_found, evidence_value, evidence_source, evidence_status = _evidence_span_metric_override(field_name, row)
        if evidence_found:
            return normalize_value_with_lexicon(field_name, evidence_value, paper_key=paper_key, lexicon=lexicon), evidence_source, evidence_status
    if field_name == "solvent_name":
        row_local_solvent = _value_from_row_local_solvent_volume_header(row)
        if row_local_solvent:
            return normalize_value_with_lexicon(field_name, row_local_solvent, paper_key=paper_key, lexicon=lexicon), "row_local_solvent_volume_header", "supported"
    decision_value = _value_from_decision_identity(field_name, row)
    if decision_value:
        return normalize_value_with_lexicon(field_name, decision_value, paper_key=paper_key, lexicon=lexicon), "decision_trace_identity", "supported"
    prep_value = _value_from_preparation_method(field_name, row)
    if prep_value:
        source_type = "relation_or_direct" if field_name == "method_type" else "preparation_method_parse"
        return normalize_value_with_lexicon(field_name, prep_value, paper_key=paper_key, lexicon=lexicon), source_type, "supported"
    return "", str(mapping.get("source", "")), str(mapping.get("evidence", ""))


def include_gt_row_for_compare(row: dict[str, str]) -> bool:
    if normalize_text(row.get("gt_row_decision")) != "include_gt":
        return False
    benchmark_default_include = normalize_text(row.get("benchmark_default_include")).lower()
    if benchmark_default_include == "no":
        return False
    return True


def build_alignment_index(alignment_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in alignment_rows:
        gt_formulation_id = normalize_text(row.get("gt_formulation_id"))
        if gt_formulation_id:
            index[gt_formulation_id] = row
        paper_key = normalize_text(gt_formulation_id.split("_", 1)[0]) if gt_formulation_id else ""
        label = canonicalize_text(row.get("pred_evidence_anchor") or row.get("gt_formulation_label") or row.get("pred_source_text"))
        if paper_key and label:
            index[f"label::{paper_key}::{label}"] = row
    return index


def get_alignment_scaffold_row(gt_row: dict[str, str], alignment_scaffold_index: dict[str, dict[str, str]]) -> dict[str, str]:
    gt_formulation_id = normalize_text(gt_row.get("gt_formulation_id"))
    if gt_formulation_id and gt_formulation_id in alignment_scaffold_index:
        return alignment_scaffold_index[gt_formulation_id]
    paper_key = normalize_text(gt_row.get("paper_key"))
    label = canonicalize_text(gt_row.get("formulation_label"))
    if paper_key and label:
        return alignment_scaffold_index.get(f"label::{paper_key}::{label}", {})
    return {}


def build_trusted_alignment_index(alignment_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in alignment_rows:
        gt_formulation_id = normalize_text(
            row.get("l2_gt_formulation_id")
            or row.get("gt_formulation_id")
            or row.get("formulation_id")
        )
        if gt_formulation_id:
            index[gt_formulation_id] = row
    return index


def build_decision_trace_index(decision_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in decision_rows:
        target_final_formulation_id = normalize_text(row.get("target_final_formulation_id"))
        decision = normalize_text(row.get("decision")).lower()
        if target_final_formulation_id and decision == "kept":
            index[target_final_formulation_id] = row
    return index


def augment_system_rows_with_identity_surface(
    system_rows: list[dict[str, str]],
    decision_trace_index: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    augmented: list[dict[str, str]] = []
    for row in system_rows:
        out = dict(row)
        decision_row = decision_trace_index.get(normalize_text(row.get("final_formulation_id")), {})
        out["decision_source_formulation_id"] = normalize_text(decision_row.get("source_formulation_id"))
        out["decision_source_raw_formulation_label"] = normalize_text(decision_row.get("source_raw_formulation_label"))
        out["decision_parent_core_row_id"] = normalize_text(decision_row.get("parent_core_row_id"))
        out["decision_key_fields_used"] = normalize_text(decision_row.get("key_fields_used"))
        augmented.append(out)
    return augmented


def _alignment_hint_tokens(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        normalized = canonicalize_text(value)
        if not normalized:
            continue
        for token in re.split(r"[^a-z0-9.]+", normalized):
            if len(token) >= 2:
                tokens.add(token)
    return tokens


def _compact_identity_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", canonicalize_text(value))


def _choose_by_gt_label_ordinal(label: str, paper_rows: list[dict[str, str]]) -> dict[str, str] | None:
    normalized = normalize_text(label)
    if not normalized:
        return None
    match = re.match(r"^(\d+)\b", normalized)
    if not match:
        return None
    ordinal = str(int(match.group(1)))
    candidates = []
    for row in paper_rows:
        values = [
            normalize_text(row.get("raw_formulation_label")),
            normalize_text(row.get("decision_source_raw_formulation_label")),
            normalize_text(row.get("representative_source_raw_formulation_label")),
            normalize_text(row.get("representative_source_formulation_id")),
            normalize_text(row.get("formulation_id")),
        ]
        for value in values:
            if not value:
                continue
            if value == ordinal or re.search(rf"(?:^|_)row_?{ordinal}$", value.lower()):
                candidates.append(row)
                break
    if len(candidates) == 1:
        return candidates[0]
    return None


def _parse_jsonish_list(value: str) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [normalize_text(item) for item in parsed if normalize_text(item)]
    return [text]


def _extract_decision_identity_values(value: str) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}
    identity_blob = normalize_text(parsed.get("identity_variables")) if isinstance(parsed, dict) else ""
    if not identity_blob:
        return []
    values = []
    for chunk in identity_blob.split("|"):
        chunk = normalize_text(chunk)
        if not chunk:
            continue
        if "=" in chunk:
            values.append(normalize_text(chunk.split("=", 1)[1]))
        else:
            values.append(chunk)
    return [v for v in values if v]


def _system_identity_strings(row: dict[str, str]) -> list[str]:
    values = [
        normalize_text(row.get("raw_formulation_label")),
        normalize_text(row.get("representative_source_raw_formulation_label")),
        normalize_text(row.get("decision_source_raw_formulation_label")),
        normalize_text(row.get("decision_source_formulation_id")),
        normalize_text(row.get("decision_parent_core_row_id")),
    ]
    values.extend(_parse_jsonish_list(row.get("source_candidate_labels", "")))
    values.extend(_parse_jsonish_list(row.get("source_candidate_ids", "")))
    values.extend(_extract_decision_identity_values(row.get("decision_key_fields_used", "")))
    return [value for value in values if value]


def _extract_ratio_tokens(*values: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in re.findall(r"\b\d+(?:\.\d+)?:\d+(?:\.\d+)?\b", normalize_text(value)):
            if token not in seen:
                tokens.append(token)
                seen.add(token)
    return tokens


def _extract_identity_signature_tokens(value: str) -> list[str]:
    text = normalize_text(value).lower()
    if not text:
        return []
    tokens = re.findall(r"\b\d+(?:\.\d+)?:\d+(?:\.\d+)?\b|\b\d+(?:\.\d+)?x\b|\b[a-z]+\*?\b|\b\d+(?:\.\d+)?\b", text)
    return [token.strip().lower() for token in tokens if token.strip()]


def _choose_by_compact_label_unique(label: str, paper_rows: list[dict[str, str]]) -> dict[str, str] | None:
    compact = _compact_identity_text(label)
    if not compact:
        return None
    matches = []
    for row in paper_rows:
        row_compacts = {_compact_identity_text(value) for value in _system_identity_strings(row) if value}
        if compact and compact in row_compacts:
            matches.append(row)
    if len(matches) == 1:
        return matches[0]
    return None


def _choose_by_ratio_token_bridge(gt_row: dict[str, str], paper_rows: list[dict[str, str]]) -> dict[str, str] | None:
    gt_label = normalize_text(gt_row.get("formulation_label", ""))
    gt_seed = normalize_text(gt_row.get("seed_pred_representative_source_formulation_id", ""))
    ratios = _extract_ratio_tokens(gt_label, gt_seed)
    if not ratios:
        return None
    primary_ratio = ratios[0]
    gt_named = _extract_named_ratio_label(gt_label)
    generic_named_ratio = False
    if gt_named:
        gt_left_c = canonicalize_text(gt_named[0])
        gt_right_c = canonicalize_text(gt_named[1])
        generic_named_ratio = {gt_left_c, gt_right_c} <= {"drug", "polymer"}
    matches = []
    for row in paper_rows:
        identity_strings = _system_identity_strings(row)
        row_text = " | ".join(identity_strings)
        if gt_named and not generic_named_ratio:
            gt_left, gt_right, gt_ratio = gt_named
            named_ok = False
            for candidate in identity_strings:
                row_named = _extract_named_ratio_label(candidate)
                if not row_named:
                    continue
                row_left, row_right, row_ratio = row_named
                if (
                    canonicalize_text(gt_left) == canonicalize_text(row_left)
                    and canonicalize_text(gt_right) == canonicalize_text(row_right)
                    and normalize_text(gt_ratio) == normalize_text(row_ratio)
                ):
                    named_ok = True
                    break
            if named_ok:
                matches.append(row)
            continue
        row_ratios = set(_extract_ratio_tokens(row_text))
        if primary_ratio in row_ratios:
            matches.append(row)
    if len(matches) == 1:
        return matches[0]
    return None


def _choose_by_decision_identity_signature(gt_row: dict[str, str], paper_rows: list[dict[str, str]]) -> dict[str, str] | None:
    gt_signature = _extract_identity_signature_tokens(gt_row.get("formulation_label", ""))
    if len(gt_signature) < 3:
        return None
    gt_counter = Counter(gt_signature)
    matches = []
    for row in paper_rows:
        row_signature = _extract_identity_signature_tokens(" ".join(_extract_decision_identity_values(row.get("decision_key_fields_used", ""))))
        if row_signature and Counter(row_signature) == gt_counter:
            matches.append(row)
    if len(matches) == 1:
        return matches[0]
    return None


def _row_local_identity_strings(row: dict[str, str]) -> list[str]:
    values = [
        normalize_text(row.get("evidence_span_text")),
        normalize_text(row.get("supporting_evidence_refs")),
        normalize_text(row.get("change_descriptions")),
        normalize_text(row.get("decision_key_fields_used")),
    ]
    values.extend(_parse_jsonish_list(row.get("source_candidate_labels", "")))
    return [value for value in values if value]


def _signature_token_present(token: str, text: str) -> bool:
    token = normalize_text(token).lower()
    text = normalize_text(text).lower()
    if not token or not text:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?:\d+(?:\.\d+)?", token):
        return re.search(rf"(?<![0-9.]){re.escape(token)}(?![0-9.])", text) is not None
    if re.fullmatch(r"\d+(?:\.\d+)?x", token):
        return re.search(rf"(?<![a-z0-9.]){re.escape(token)}(?![a-z0-9])", text) is not None
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return re.search(rf"(?<![0-9.]){re.escape(token)}(?![0-9.])", text) is not None
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None


def _choose_by_row_local_design_signature(gt_row: dict[str, str], paper_rows: list[dict[str, str]]) -> dict[str, str] | None:
    gt_label = gt_row.get("formulation_label", "")
    if _extract_named_ratio_label(normalize_text(gt_label)):
        return None
    gt_tokens = _extract_identity_signature_tokens(gt_label)
    if len(gt_tokens) < 4:
        return None
    if not any(re.search(r"[a-z]", token) for token in gt_tokens):
        return None
    if not any(re.search(r"\d", token) for token in gt_tokens):
        return None
    matches = []
    for row in paper_rows:
        for surface in _row_local_identity_strings(row):
            if all(_signature_token_present(token, surface) for token in gt_tokens):
                matches.append(row)
                break
    if len(matches) == 1:
        return matches[0]
    return None


def _decision_payload(row: dict[str, str]) -> dict:
    text = normalize_text(row.get("decision_key_fields_used"))
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_loaded_state(row: dict[str, str]) -> str:
    direct = canonicalize_text(row.get("loaded_state") or row.get("payload_state") or row.get("variant_role"))
    payload = _decision_payload(row)
    decision_state = canonicalize_text(str(payload.get("loaded_state", "")))
    text = canonicalize_text(" | ".join([
        direct,
        decision_state,
        row.get("raw_formulation_label", ""),
        row.get("decision_key_fields_used", ""),
    ]))
    if re.search(r"\b(empty|blank|no drug|unloaded)\b", text):
        return "empty"
    if re.search(r"\b(drug loaded|loaded|drugloaded|encapsulated)\b", text):
        return "drug_loaded"
    return ""


def _choose_by_loaded_state_disambiguation(gt_row: dict[str, str], paper_rows: list[dict[str, str]]) -> dict[str, str] | None:
    gt_text = canonicalize_text(" | ".join([
        gt_row.get("formulation_label", ""),
        gt_row.get("variant_role", ""),
        gt_row.get("drug_name", ""),
        gt_row.get("drug_mass_mg", ""),
        gt_row.get("ee_percent", ""),
        gt_row.get("lc_percent", ""),
        gt_row.get("dl_percent", ""),
    ]))
    if re.search(r"\b(empty|blank|no drug|unloaded)\b", gt_text) and not normalize_text(gt_row.get("drug_name")):
        target_state = "empty"
    elif normalize_text(gt_row.get("drug_name")) or normalize_text(gt_row.get("drug_mass_mg")) or normalize_text(gt_row.get("ee_percent")) or normalize_text(gt_row.get("lc_percent")) or normalize_text(gt_row.get("dl_percent")):
        target_state = "drug_loaded"
    else:
        return None
    gt_label_compact = _compact_identity_text(gt_row.get("formulation_label", ""))
    matches = []
    for row in paper_rows:
        if _row_loaded_state(row) != target_state:
            continue
        if gt_label_compact:
            row_compacts = [_compact_identity_text(value) for value in _system_identity_strings(row)]
            if not any(gt_label_compact and (gt_label_compact in compact or compact in gt_label_compact) for compact in row_compacts if compact):
                continue
        drug_gt = canonicalize_text(gt_row.get("drug_name"))
        row_text = canonicalize_text(" | ".join(_system_identity_strings(row) + _row_local_identity_strings(row)))
        if target_state == "drug_loaded" and drug_gt and drug_gt not in row_text:
            continue
        matches.append(row)
    if len(matches) == 1:
        return matches[0]
    return None


def _extract_scaffold_ordinals(*values: str) -> list[str]:
    ordinals: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        for pattern in (r"F_SrNo(\d+)", r"\bF(\d+)\b", r"\brow[_\s]0*(\d+)\b", r"\b(\d+)\b"):
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                num = str(int(match.group(1)))
                if num not in seen:
                    ordinals.append(num)
                    seen.add(num)
    return ordinals


def _choose_by_scaffold_parent_ordinal(scaffold_parent_core: str, seed_id: str, paper_rows: list[dict[str, str]]) -> dict[str, str] | None:
    ordinals = _extract_scaffold_ordinals(scaffold_parent_core, seed_id)
    if not ordinals:
        return None
    for ordinal in ordinals:
        matches = []
        for row in paper_rows:
            surfaces = [
                normalize_text(row.get("raw_formulation_label")),
                normalize_text(row.get("decision_source_raw_formulation_label")),
            ]
            identity_tails = [
                normalize_text(row.get("formulation_id")).split("__")[-1],
                normalize_text(row.get("representative_source_formulation_id")).split("__")[-1],
                normalize_text(row.get("parent_core_row_id")).split("__")[-1],
                normalize_text(row.get("decision_source_formulation_id")).split("__")[-1],
                normalize_text(row.get("decision_parent_core_row_id")).split("__")[-1],
            ]
            surfaces.extend([tail for tail in identity_tails if tail])
            exact_hit = any(
                surface == ordinal
                or re.search(rf"(?:^|[^0-9]){re.escape(ordinal)}(?:$|[^0-9])", surface)
                or re.search(rf"row[_-]?0*{re.escape(ordinal)}(?:[^0-9]|$)", surface)
                or re.search(rf"(?:^|[_-])0*{re.escape(ordinal)}(?:$|[_-])", surface)
                for surface in surfaces if surface
            )
            if exact_hit:
                matches.append(row)
        if len(matches) == 1:
            return matches[0]
    return None


def _semantic_alignment_score(
    gt_row: dict[str, str],
    system_row: dict[str, str],
    scaffold_row: dict[str, str] | None = None,
    trusted_alignment_row: dict[str, str] | None = None,
) -> int:
    score = 0
    polymer_gt = canonicalize_text(gt_row.get("polymer_name"))
    polymer_sys = canonicalize_text(system_row.get("polymer_identity_final") or system_row.get("polymer_name_raw"))
    if polymer_gt and polymer_sys and polymer_gt == polymer_sys:
        score += 3
    drug_gt = canonicalize_text(gt_row.get("drug_name"))
    drug_sys = canonicalize_text(system_row.get("drug_name_value_text"))
    if drug_gt and drug_sys and drug_gt == drug_sys:
        score += 3
    drug_mass_gt = canonicalize_text(gt_row.get("drug_mass_mg"))
    drug_mass_sys = canonicalize_text(system_row.get("drug_feed_amount_text_value_text"))
    if drug_mass_gt and drug_mass_sys and drug_mass_gt == drug_mass_sys:
        score += 4
    laga_gt = canonicalize_text(gt_row.get("la_ga_ratio_normalized") or gt_row.get("la_ga_ratio_raw"))
    laga_sys = canonicalize_text(system_row.get("la_ga_ratio_value_text"))
    if laga_gt and laga_sys and (laga_gt == laga_sys or sorted(laga_gt.replace(':', ' ').split()) == sorted(laga_sys.replace(':', ' ').split())):
        score += 2
    label_gt = canonicalize_text(gt_row.get("formulation_label"))
    label_sys = canonicalize_text(system_row.get("raw_formulation_label"))
    if label_gt and label_sys and any(tok in label_sys for tok in [t for t in label_gt.replace('[', ' ').replace(']', ' ').split() if t]):
        score += 1
    scaffold_row = scaffold_row or {}
    trusted_alignment_row = trusted_alignment_row or {}
    scaffold_anchor = normalize_text(scaffold_row.get("pred_evidence_anchor"))
    scaffold_parent = normalize_text(scaffold_row.get("parent_core_row_id"))
    scaffold_pred_row_id = normalize_text(scaffold_row.get("pred_row_id"))
    trusted_pred_row_id = normalize_text(trusted_alignment_row.get("matched_system_formulation_id"))
    system_identities = {
        normalize_text(system_row.get("final_formulation_id")),
        normalize_text(system_row.get("formulation_id")),
        normalize_text(system_row.get("representative_source_formulation_id")),
        normalize_text(system_row.get("parent_core_row_id")),
    }
    if scaffold_pred_row_id and scaffold_pred_row_id in system_identities:
        score += 10
    if trusted_pred_row_id and trusted_pred_row_id in system_identities:
        score += 10
    if scaffold_parent and scaffold_parent in system_identities:
        score += 6
    scaffold_tokens = _alignment_hint_tokens(scaffold_anchor, scaffold_parent, scaffold_row.get("pred_source_text", ""))
    system_tokens = _alignment_hint_tokens(
        system_row.get("raw_formulation_label", ""),
        system_row.get("representative_source_raw_formulation_label", ""),
        system_row.get("representative_source_formulation_id", ""),
        system_row.get("parent_core_row_id", ""),
    )
    if scaffold_tokens and system_tokens:
        overlap = scaffold_tokens & system_tokens
        score += min(len(overlap), 4)
    return score


def choose_system_row(
    gt_row: dict[str, str],
    system_rows: list[dict[str, str]],
    alignment_scaffold_index: dict[str, dict[str, str]] | None = None,
    trusted_alignment_index: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, str] | None, str, bool]:
    paper_key = normalize_text(gt_row.get("paper_key"))
    gt_formulation_id = normalize_text(gt_row.get("gt_formulation_id"))
    seed_id = normalize_text(gt_row.get("seed_pred_representative_source_formulation_id"))
    label = normalize_text(gt_row.get("formulation_label"))
    alignment_scaffold_index = alignment_scaffold_index or {}
    trusted_alignment_index = trusted_alignment_index or {}
    scaffold_row = get_alignment_scaffold_row(gt_row, alignment_scaffold_index)
    trusted_alignment_row = trusted_alignment_index.get(gt_formulation_id, {})
    scaffold_pred_row_id = normalize_text(scaffold_row.get("pred_row_id"))
    scaffold_parent_core = normalize_text(scaffold_row.get("parent_core_row_id"))
    scaffold_anchor = normalize_text(scaffold_row.get("pred_evidence_anchor"))
    trusted_pred_row_id = normalize_text(trusted_alignment_row.get("matched_system_formulation_id"))
    paper_rows = [row for row in system_rows if normalize_text(row.get("key")) == paper_key]
    for rule, key_name, target in [
        ("direct_formulation_id", "formulation_id", gt_formulation_id),
        ("representative_source_formulation_id", "representative_source_formulation_id", seed_id),
        ("raw_formulation_label_unique", "raw_formulation_label", label),
        ("alignment_scaffold_final_formulation_id", "final_formulation_id", scaffold_pred_row_id),
        ("alignment_scaffold_formulation_id", "formulation_id", scaffold_pred_row_id),
        ("alignment_scaffold_parent_core_row_id", "parent_core_row_id", scaffold_parent_core),
        ("alignment_scaffold_representative_source", "representative_source_formulation_id", scaffold_parent_core),
        ("alignment_scaffold_pred_evidence_anchor", "raw_formulation_label", scaffold_anchor),
        ("trusted_alignment_formulation_id", "formulation_id", trusted_pred_row_id),
        ("trusted_alignment_final_formulation_id", "final_formulation_id", trusted_pred_row_id),
    ]:
        if not target:
            continue
        matches = [row for row in paper_rows if normalize_text(row.get(key_name)) == target]
        if len(matches) == 1:
            return matches[0], rule, True
    for rule, candidate in [
        ("compact_label_unique", _choose_by_compact_label_unique(label or scaffold_anchor, paper_rows)),
        ("ratio_token_bridge", _choose_by_ratio_token_bridge(gt_row, paper_rows)),
        ("decision_identity_signature_bridge", _choose_by_decision_identity_signature(gt_row, paper_rows)),
        ("row_local_design_signature_bridge", _choose_by_row_local_design_signature(gt_row, paper_rows)),
        ("loaded_state_disambiguation_bridge", _choose_by_loaded_state_disambiguation(gt_row, paper_rows)),
        ("scaffold_parent_ordinal_bridge", _choose_by_scaffold_parent_ordinal(scaffold_parent_core, seed_id, paper_rows)),
        ("gt_label_ordinal_bridge", _choose_by_gt_label_ordinal(label, paper_rows)),
    ]:
        if candidate is not None:
            return candidate, rule, True
    scored = [
        (row, _semantic_alignment_score(gt_row, row, scaffold_row=scaffold_row, trusted_alignment_row=trusted_alignment_row))
        for row in paper_rows
    ]
    scored = [(row, score) for row, score in scored if score > 0]
    if scored:
        scored.sort(key=lambda item: item[1], reverse=True)
        if len(scored) == 1 or scored[0][1] > scored[1][1]:
            if scored[0][1] >= 4:
                return scored[0][0], "semantic_signature_fallback", True
    return None, "no_unique_alignment", False


def _build_paper_coded_ph_rank_maps(
    gt_rows: list[dict[str, str]],
    system_rows: list[dict[str, str]],
    alignment_scaffold_index: dict[str, dict[str, str]],
    trusted_alignment_index: dict[str, dict[str, str]],
    value_normalization_lexicon: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    paper_pairs: dict[str, list[tuple[float, str, float, str]]] = defaultdict(list)
    for gt_row in gt_rows:
        gt_value_raw = normalize_text(gt_row.get("pH_raw"))
        if not gt_value_raw or not _is_coded_doe_level_token(gt_value_raw):
            continue
        paper_key = normalize_text(gt_row.get("paper_key"))
        system_row, _, alignment_ok = choose_system_row(
            gt_row,
            system_rows,
            alignment_scaffold_index=alignment_scaffold_index,
            trusted_alignment_index=trusted_alignment_index,
        )
        if not alignment_ok or not system_row:
            continue
        system_value_raw, _, _ = get_system_value(
            "pH_raw",
            system_row,
            paper_key=paper_key,
            lexicon=value_normalization_lexicon,
            allow_evidence_metric_rebinding=bool(gt_value_raw),
        )
        if not system_value_raw:
            continue
        gt_number = parse_numeric(gt_value_raw)
        system_number = parse_numeric(system_value_raw)
        if gt_number is None or system_number is None:
            continue
        paper_pairs[paper_key].append((gt_number, gt_value_raw, system_number, normalize_text(system_value_raw)))

    out: dict[str, dict[str, str]] = {}
    for paper_key, pairs in paper_pairs.items():
        gt_levels = sorted({(gt_number, gt_text) for gt_number, gt_text, _, _ in pairs}, key=lambda item: item[0])
        system_levels = sorted({(system_number, system_text) for _, _, system_number, system_text in pairs}, key=lambda item: item[0])
        if len(gt_levels) < 3 or len(gt_levels) != len(system_levels):
            continue
        out[paper_key] = {
            system_text: normalize_text(gt_text)
            for (_, gt_text), (_, system_text) in zip(gt_levels, system_levels)
        }
    return out


def build_cells(
    gt_rows: list[dict[str, str]],
    system_rows: list[dict[str, str]],
    alignment_scaffold_index: dict[str, dict[str, str]] | None = None,
    trusted_alignment_index: dict[str, dict[str, str]] | None = None,
    value_normalization_lexicon: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    output = []
    alignment_resolution_rows = []
    alignment_scaffold_index = alignment_scaffold_index or {}
    trusted_alignment_index = trusted_alignment_index or {}
    value_normalization_lexicon = value_normalization_lexicon or {}
    paper_coded_ph_rank_maps = _build_paper_coded_ph_rank_maps(
        gt_rows,
        system_rows,
        alignment_scaffold_index,
        trusted_alignment_index,
        value_normalization_lexicon,
    )
    for gt_row in gt_rows:
        paper_key = normalize_text(gt_row.get("paper_key"))
        doi = normalize_text(gt_row.get("doi"))
        gt_formulation_id = normalize_text(gt_row.get("gt_formulation_id"))
        scaffold_row = get_alignment_scaffold_row(gt_row, alignment_scaffold_index)
        trusted_row = trusted_alignment_index.get(gt_formulation_id, {})
        system_row, alignment_rule, alignment_ok = choose_system_row(
            gt_row,
            system_rows,
            alignment_scaffold_index=alignment_scaffold_index,
            trusted_alignment_index=trusted_alignment_index,
        )
        matched_system_formulation_id = normalize_text(system_row.get("formulation_id")) if system_row else ""
        matched_system_final_formulation_id = normalize_text(system_row.get("final_formulation_id")) if system_row else ""
        alignment_resolution_rows.append(
            {
                "paper_key": paper_key,
                "doi": doi,
                "gt_formulation_id": gt_formulation_id,
                "gt_formulation_label": normalize_text(gt_row.get("formulation_label")),
                "alignment_ok": "yes" if alignment_ok else "no",
                "alignment_rule": alignment_rule,
                "matched_system_formulation_id": matched_system_formulation_id,
                "matched_system_final_formulation_id": matched_system_final_formulation_id,
                "scaffold_pred_row_id": normalize_text(scaffold_row.get("pred_row_id")),
                "scaffold_parent_core_row_id": normalize_text(scaffold_row.get("parent_core_row_id")),
                "scaffold_pred_evidence_anchor": normalize_text(scaffold_row.get("pred_evidence_anchor")),
                "scaffold_alignment_decision": normalize_text(scaffold_row.get("alignment_decision")),
                "scaffold_alignment_confidence": normalize_text(scaffold_row.get("alignment_confidence")),
                "trusted_matched_system_formulation_id": normalize_text(trusted_row.get("matched_system_formulation_id")),
            }
        )
        for field_name in sorted(CORE_FIXED_FIELDS | NAMED_EXTENSIBLE_VARIABLE_FIELDS):
            gt_value_raw = normalize_text(gt_row.get(field_name))
            system_value_raw, source_type, evidence_status = get_system_value(
                field_name,
                system_row or {},
                paper_key=paper_key,
                lexicon=value_normalization_lexicon,
                allow_evidence_metric_rebinding=bool(gt_value_raw),
            )
            if field_name == "W2_volume_mL" and gt_value_raw and not system_value_raw:
                external_aqueous_value, _, external_aqueous_evidence = get_system_value(
                    "external_aqueous_phase_volume_mL",
                    system_row or {},
                    paper_key=paper_key,
                    lexicon=value_normalization_lexicon,
                    allow_evidence_metric_rebinding=False,
                )
                if external_aqueous_value:
                    system_value_raw = external_aqueous_value
                    source_type = "external_aqueous_phase_alias_for_w2_gt"
                    evidence_status = f"supported_external_aqueous_phase_alias_for_w2_gt:{external_aqueous_evidence}"
            if (
                not gt_value_raw
                and system_value_raw
                and field_name == "W2_volume_mL"
            ):
                system_value_raw = ""
                source_type = "suppressed_w2_phase_alias_without_gt_value"
                evidence_status = "w2_alias_only_evaluated_when_w2_gt_nonempty"
            if (
                not gt_value_raw
                and system_value_raw
                and field_name == "external_aqueous_phase_volume_mL"
                and normalize_text(gt_row.get("W2_volume_mL"))
            ):
                system_value_raw = ""
                source_type = "suppressed_external_aqueous_duplicate_of_w2_gt"
                evidence_status = "external_aqueous_payload_scored_against_nonempty_w2_gt"
            if (
                not gt_value_raw
                and system_value_raw
                and should_suppress_duplicate_concentration_unit_cell(field_name, gt_row, system_value_raw)
            ):
                system_value_raw = ""
                source_type = "suppressed_duplicate_unit_from_combined_gt_value"
                evidence_status = "unit_already_scored_in_combined_concentration_value"
            if (
                not gt_value_raw
                and system_value_raw
                and field_name == "polymer_grade"
                and source_type == "missing_system_field_surface"
            ):
                system_value_raw = ""
                source_type = "suppressed_polymer_identity_duplicate_without_grade_gt"
                evidence_status = "polymer_name_surface_not_scored_as_grade_when_gt_blank"
            if (
                gt_value_raw
                and system_value_raw
                and field_name == "emulsifier_stabilizer_concentration_value"
                and "|" in normalize_text(gt_row.get("emulsifier_stabilizer_name"))
                and source_type in {"row_identity_surfactant_concentration", "shared_carrythrough"}
            ):
                system_value_raw = ""
                source_type = "suppressed_single_surfactant_value_against_union_gt"
                evidence_status = "row_local_named_surfactant_value_not_unique_for_union_emulsifier_gt"
            if field_name == "pH_raw" and gt_value_raw and _is_coded_doe_level_token(gt_value_raw):
                paper_rank_map = paper_coded_ph_rank_maps.get(paper_key, {})
                mapped_value = paper_rank_map.get(normalize_text(system_value_raw), "")
                if mapped_value:
                    system_value_raw = mapped_value
                    source_type = "coded_doe_rank_rebinding"
                    evidence_status = "supported_physical_value_rank_decode_later"
            strict, relaxed, canonicalized = compare_values(
                field_name,
                gt_value_raw,
                system_value_raw,
                paper_key=paper_key,
                lexicon=value_normalization_lexicon,
            )
            if (
                field_name in {"emulsifier_stabilizer_concentration_value", "surfactant_concentration_value"}
                and gt_value_raw
                and system_value_raw
                and not canonicalized
                and source_type.startswith("shared_carrythrough")
            ):
                system_value_raw = ""
                source_type = "suppressed_mismatched_shared_surfactant_concentration_value"
                evidence_status = "shared_surfactant_concentration_not_row_local_numeric_authority"
                strict = relaxed = canonicalized = False
            if (
                field_name == "polymer_grade"
                and gt_value_raw
                and system_value_raw
                and not canonicalized
                and source_type == "missing_system_field_surface"
                and normalize_text(system_value_raw).lower() in {"plga", "pcl", "pla", "plga-peg"}
            ):
                system_value_raw = ""
                source_type = "suppressed_generic_polymer_identity_as_grade_mismatch"
                evidence_status = "generic_polymer_identity_not_specific_grade_surface"
                strict = relaxed = canonicalized = False
            status = determine_compare_status(
                gt_value_raw=gt_value_raw,
                system_value_raw=system_value_raw,
                alignment_ok=alignment_ok,
                matched=canonicalized,
            )
            bucket = infer_error_bucket(
                compare_status=status,
                field_name=field_name,
                strict_match=strict,
                relaxed_match=relaxed,
                canonicalized_match=canonicalized,
                system_value_source_type=source_type,
                evidence_status_detail=evidence_status,
            )
            output.append(
                {
                    "paper_key": paper_key,
                    "doi": doi,
                    "gt_formulation_id": gt_formulation_id,
                    "matched_system_formulation_id": matched_system_formulation_id,
                    "field_name": field_name,
                    "field_group": field_group(field_name),
                    "gt_value_raw": gt_value_raw,
                    "system_value_raw": system_value_raw,
                    "compare_status": status,
                    "strict_match": "yes" if strict else "no",
                    "relaxed_match": "yes" if relaxed else "no",
                    "canonicalized_match": "yes" if canonicalized else "no",
                    "selected_compare_mode": DEFAULT_SELECTED_COMPARE_MODE,
                    "error_bucket": bucket,
                    "system_value_source_type": source_type,
                    "evidence_status_detail": evidence_status,
                    "alignment_rule": alignment_rule,
                }
            )
    return output, alignment_resolution_rows


def summarize_cells(cells: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in cells:
        grouped[(row["field_group"], row["field_name"])].append(row)

    summary_rows = []
    for (group_name, field_name), rows in sorted(grouped.items()):
        gt_nonempty = [r for r in rows if r["gt_value_raw"]]
        gt_nonempty_and_system = [r for r in gt_nonempty if r["system_value_raw"]]
        extra = [r for r in rows if (not r["gt_value_raw"]) and r["system_value_raw"]]
        strict_matches = sum(1 for r in gt_nonempty_and_system if r["strict_match"] == "yes")
        relaxed_matches = sum(1 for r in gt_nonempty_and_system if r["relaxed_match"] == "yes")
        canonicalized_matches = sum(1 for r in gt_nonempty_and_system if r["canonicalized_match"] == "yes")
        summary_rows.append(
            {
                "group_name": group_name,
                "field_name": field_name,
                "gt_nonempty_cells": str(len(gt_nonempty)),
                "system_nonempty_on_gt_cells": str(sum(1 for r in gt_nonempty if r["system_value_raw"])),
                "value_recall": f"{(sum(1 for r in gt_nonempty if r['system_value_raw']) / len(gt_nonempty)):.6f}" if gt_nonempty else "",
                "conditional_accuracy_strict": f"{(strict_matches / len(gt_nonempty_and_system)):.6f}" if gt_nonempty_and_system else "",
                "conditional_accuracy_relaxed": f"{(relaxed_matches / len(gt_nonempty_and_system)):.6f}" if gt_nonempty_and_system else "",
                "conditional_accuracy_canonicalized": f"{(canonicalized_matches / len(gt_nonempty_and_system)):.6f}" if gt_nonempty_and_system else "",
                "extra_in_system_cells": str(len(extra)),
            }
        )
    # group rollups
    for group_name in ["core_fixed_fields", "named_extensible_variables"]:
        rows = [r for r in cells if r["field_group"] == group_name]
        gt_nonempty = [r for r in rows if r["gt_value_raw"]]
        gt_nonempty_and_system = [r for r in gt_nonempty if r["system_value_raw"]]
        extra = [r for r in rows if (not r["gt_value_raw"]) and r["system_value_raw"]]
        summary_rows.append(
            {
                "group_name": group_name,
                "field_name": "__group_total__",
                "gt_nonempty_cells": str(len(gt_nonempty)),
                "system_nonempty_on_gt_cells": str(sum(1 for r in gt_nonempty if r["system_value_raw"])),
                "value_recall": f"{(sum(1 for r in gt_nonempty if r['system_value_raw']) / len(gt_nonempty)):.6f}" if gt_nonempty else "",
                "conditional_accuracy_strict": f"{(sum(1 for r in gt_nonempty_and_system if r['strict_match']=='yes') / len(gt_nonempty_and_system)):.6f}" if gt_nonempty_and_system else "",
                "conditional_accuracy_relaxed": f"{(sum(1 for r in gt_nonempty_and_system if r['relaxed_match']=='yes') / len(gt_nonempty_and_system)):.6f}" if gt_nonempty_and_system else "",
                "conditional_accuracy_canonicalized": f"{(sum(1 for r in gt_nonempty_and_system if r['canonicalized_match']=='yes') / len(gt_nonempty_and_system)):.6f}" if gt_nonempty_and_system else "",
                "extra_in_system_cells": str(len(extra)),
            }
        )
    return summary_rows


def build_error_bucket_rows(cells: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [r for r in cells if r["compare_status"] != "present_and_match" and r["compare_status"] != "not_reported_in_gt"]
    return sorted(rows, key=lambda r: (r["error_bucket"], r["paper_key"], r["field_name"], r["gt_formulation_id"]))


def infer_risk_level(cell: dict[str, str]) -> str:
    if cell["compare_status"] == "blocked_alignment":
        return "high"
    if cell["error_bucket"] in {"derived_value_leakage", "unsupported_text", "unresolved_table"}:
        return "high"
    if cell["error_bucket"] in {"missing_value", "field_mapping_mismatch", "numeric_extraction_mismatch"}:
        return "medium"
    if cell["error_bucket"] == "normalization_mismatch":
        return "low"
    return "medium"


def infer_risk_type(cell: dict[str, str]) -> str:
    if cell["compare_status"] == "blocked_alignment":
        return "ambiguity"
    bucket = cell["error_bucket"]
    if bucket == "derived_value_leakage":
        return "derived_value"
    if bucket in {"missing_value", "unsupported_text", "unresolved_table"}:
        return "unsupported_value"
    if bucket == "numeric_extraction_mismatch":
        return "unit_or_normalization_only" if cell["canonicalized_match"] == "yes" else "ambiguity"
    if bucket == "field_mapping_mismatch":
        return "ambiguity"
    if bucket == "normalization_mismatch":
        return "unit_or_normalization_only"
    return "ambiguity"


def build_risk_reason(cell: dict[str, str]) -> str:
    status = cell["compare_status"]
    if status == "blocked_alignment":
        return f"GT row could not be aligned to a unique current-system row under the current canonical and advisory identity bridges (rule={cell['alignment_rule'] or 'none'})."
    if cell["error_bucket"] == "missing_value":
        return "GT contains a reported value but the current system surface is blank for the aligned formulation-field cell."
    if cell["error_bucket"] == "field_mapping_mismatch":
        return "A value-surface mismatch suggests the current system field mapping or benchmark-facing field exposure is incomplete or miswired."
    if cell["error_bucket"] == "numeric_extraction_mismatch":
        return "Aligned GT and system cells disagree numerically after current canonicalization rules."
    if cell["error_bucket"] == "derived_value_leakage":
        return "The current system surface exposes a derived or relation-resolved value where GT is blank, so benchmark-facing support should be reviewed."
    if cell["error_bucket"] == "normalization_mismatch":
        return "GT and system values appear semantically close but differ only after normalization or formatting decisions."
    return "Layer3 compare flagged this cell for manual review."


def build_risk_review_queue_rows(cells: list[dict[str, str]]) -> list[dict[str, str]]:
    queue_rows = []
    priority_order = {"high": 3, "medium": 2, "low": 1}
    for cell in cells:
        if cell["compare_status"] in {"present_and_match", "not_reported_in_gt"}:
            continue
        risk_level = infer_risk_level(cell)
        queue_rows.append(
            {
                "paper_key": cell["paper_key"],
                "doi": cell["doi"],
                "gt_formulation_id": cell["gt_formulation_id"],
                "matched_system_formulation_id": cell["matched_system_formulation_id"],
                "field_name": cell["field_name"],
                "field_group": cell["field_group"],
                "gt_value_raw": cell["gt_value_raw"],
                "system_value_raw": cell["system_value_raw"],
                "risk_level": risk_level,
                "risk_type": infer_risk_type(cell),
                "review_priority": str(priority_order[risk_level]),
                "source_of_flag": "compare",
                "reason": build_risk_reason(cell),
                "evidence_status": cell["evidence_status_detail"],
                "compare_status": cell["compare_status"],
                "error_bucket": cell["error_bucket"],
                "alignment_rule": cell["alignment_rule"],
                "system_value_source_type": cell["system_value_source_type"],
            }
        )
    return sorted(
        queue_rows,
        key=lambda r: (-int(r["review_priority"]), r["paper_key"], r["gt_formulation_id"], r["field_name"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Layer3 GT value cells against the current system value surface.")
    parser.add_argument("--final-table-tsv", type=Path)
    parser.add_argument("--layer3-gt-tsv", type=Path)
    parser.add_argument("--scope-manifest-tsv", type=Path)
    parser.add_argument("--alignment-scaffold-tsv", type=Path)
    parser.add_argument("--trusted-alignment-tsv", type=Path)
    parser.add_argument("--decision-trace-tsv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def validate_field_grouping() -> None:
    if "pH_raw" not in NAMED_EXTENSIBLE_VARIABLE_FIELDS:
        raise ValueError("pH_raw must remain in NAMED_EXTENSIBLE_VARIABLE_FIELDS")
    if "pH_raw" in CORE_FIXED_FIELDS:
        raise ValueError("pH_raw must not be in CORE_FIXED_FIELDS")
    intersections = [
        CORE_FIXED_FIELDS & NAMED_EXTENSIBLE_VARIABLE_FIELDS,
        CORE_FIXED_FIELDS & PROVENANCE_ONLY_FIELDS,
        NAMED_EXTENSIBLE_VARIABLE_FIELDS & PROVENANCE_ONLY_FIELDS,
    ]
    if any(group for group in intersections):
        raise ValueError(f"Field groups overlap: {intersections}")


def main() -> None:
    args = parse_args()
    validate_field_grouping()
    run_context = resolve_run_context(explicit_run_dir=args.run_dir, explicit_run_id=args.run_id)
    final_table_tsv = resolve_artifact_path(
        explicit_path=args.final_table_tsv,
        run_context=run_context,
        pointer_key="stage5_final_table_tsv",
        canonical_relative="final_formulation_table_v1.tsv",
        required=True,
        enforce_pointer_contract=False,
    )
    layer3_gt_tsv = resolve_artifact_path(
        explicit_path=args.layer3_gt_tsv,
        run_context=run_context,
        pointer_key="layer3_gt_path",
        required=True,
        enforce_pointer_contract=True,
    )
    scope_manifest_tsv = resolve_artifact_path(
        explicit_path=args.scope_manifest_tsv,
        run_context=run_context,
        pointer_key="scope_manifest_tsv",
        required=True,
        enforce_pointer_contract=False,
    )
    alignment_scaffold_tsv = args.alignment_scaffold_tsv.resolve() if args.alignment_scaffold_tsv else DEFAULT_ALIGNMENT_SCAFFOLD_TSV.resolve()
    trusted_alignment_tsv = args.trusted_alignment_tsv.resolve() if args.trusted_alignment_tsv else None
    value_normalization_lexicon_tsv = DEFAULT_VALUE_NORMALIZATION_LEXICON_TSV.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else (Path(run_context["run_dir"]) / "61_layer3_compare").resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    decision_trace_tsv = resolve_artifact_path(
        explicit_path=args.decision_trace_tsv,
        run_context=run_context,
        pointer_key="stage5_decision_trace_tsv",
        canonical_relative="final_output_decision_trace_v1.tsv",
        required=False,
        enforce_pointer_contract=False,
    )

    print(json.dumps({
        "resolved_source_run_dir": str(run_context["run_dir"]),
        "resolved_source_run_id": str(run_context["run_id"]),
        "resolved_input_files": {
            "final_table_tsv": str(final_table_tsv),
            "layer3_gt_tsv": str(layer3_gt_tsv),
            "scope_manifest_tsv": str(scope_manifest_tsv),
            "alignment_scaffold_tsv": str(alignment_scaffold_tsv) if alignment_scaffold_tsv.exists() else "",
            "trusted_alignment_tsv": str(trusted_alignment_tsv) if trusted_alignment_tsv and trusted_alignment_tsv.exists() else "",
            "decision_trace_tsv": str(decision_trace_tsv) if decision_trace_tsv and decision_trace_tsv.exists() else "",
            "value_normalization_lexicon_tsv": str(value_normalization_lexicon_tsv) if value_normalization_lexicon_tsv.exists() else "",
        },
        "resolved_out_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))

    gt_rows = [row for row in read_tsv(layer3_gt_tsv) if include_gt_row_for_compare(row)]
    system_rows = read_tsv(final_table_tsv)
    alignment_scaffold_rows = read_tsv(alignment_scaffold_tsv) if alignment_scaffold_tsv.exists() else []
    trusted_alignment_rows = read_tsv(trusted_alignment_tsv) if trusted_alignment_tsv and trusted_alignment_tsv.exists() else []
    decision_trace_rows = read_tsv(decision_trace_tsv) if decision_trace_tsv and decision_trace_tsv.exists() else []
    value_normalization_lexicon_rows = read_tsv(value_normalization_lexicon_tsv) if value_normalization_lexicon_tsv.exists() else []
    alignment_scaffold_index = build_alignment_index(alignment_scaffold_rows)
    trusted_alignment_index = build_trusted_alignment_index(trusted_alignment_rows)
    decision_trace_index = build_decision_trace_index(decision_trace_rows)
    value_normalization_lexicon = build_value_normalization_lexicon(value_normalization_lexicon_rows)
    system_rows = augment_system_rows_with_identity_surface(system_rows, decision_trace_index)
    cells, alignment_resolution_rows = build_cells(
        gt_rows,
        system_rows,
        alignment_scaffold_index=alignment_scaffold_index,
        trusted_alignment_index=trusted_alignment_index,
        value_normalization_lexicon=value_normalization_lexicon,
    )
    reporting_cells = build_reporting_cells(cells)
    summary_rows = summarize_cells(reporting_cells)
    error_rows = build_error_bucket_rows(reporting_cells)
    risk_queue_rows = build_risk_review_queue_rows(reporting_cells)

    cell_fields = [
        "paper_key","doi","gt_formulation_id","matched_system_formulation_id","field_name","field_group",
        "gt_value_raw","system_value_raw","compare_status","strict_match","relaxed_match",
        "canonicalized_match","selected_compare_mode","error_bucket","system_value_source_type",
        "evidence_status_detail","alignment_rule"
    ]
    summary_fields = [
        "group_name","field_name","gt_nonempty_cells","system_nonempty_on_gt_cells","value_recall",
        "conditional_accuracy_strict","conditional_accuracy_relaxed","conditional_accuracy_canonicalized","extra_in_system_cells"
    ]
    alignment_resolution_fields = [
        "paper_key","doi","gt_formulation_id","gt_formulation_label","alignment_ok","alignment_rule",
        "matched_system_formulation_id","matched_system_final_formulation_id","scaffold_pred_row_id",
        "scaffold_parent_core_row_id","scaffold_pred_evidence_anchor","scaffold_alignment_decision",
        "scaffold_alignment_confidence","trusted_matched_system_formulation_id"
    ]
    risk_queue_fields = [
        "paper_key","doi","gt_formulation_id","matched_system_formulation_id","field_name","field_group",
        "gt_value_raw","system_value_raw","risk_level","risk_type","review_priority","source_of_flag",
        "reason","evidence_status","compare_status","error_bucket","alignment_rule","system_value_source_type"
    ]
    write_tsv(out_dir / CELL_OUTPUT_NAME, reporting_cells, cell_fields)
    write_tsv(out_dir / SUMMARY_OUTPUT_NAME, summary_rows, summary_fields)
    write_tsv(out_dir / ERROR_BUCKET_OUTPUT_NAME, error_rows, cell_fields)
    write_tsv(out_dir / ALIGNMENT_RESOLUTION_OUTPUT_NAME, alignment_resolution_rows, alignment_resolution_fields)
    write_tsv(out_dir / RISK_REVIEW_QUEUE_OUTPUT_NAME, risk_queue_rows, risk_queue_fields)
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# Layer3 Compare Run\n\n"
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- source_run_dir: `{run_context['run_dir']}`\n"
        f"- final_table_tsv: `{final_table_tsv}`\n"
        f"- layer3_gt_tsv: `{layer3_gt_tsv}`\n"
        f"- scope_manifest_tsv: `{scope_manifest_tsv}`\n"
        f"- alignment_scaffold_tsv: `{alignment_scaffold_tsv if alignment_scaffold_tsv.exists() else ''}`\n"
        f"- trusted_alignment_tsv: `{trusted_alignment_tsv if trusted_alignment_tsv and trusted_alignment_tsv.exists() else ''}`\n"
        f"- cells_tsv: `{out_dir / CELL_OUTPUT_NAME}`\n"
        f"- summary_tsv: `{out_dir / SUMMARY_OUTPUT_NAME}`\n"
        f"- error_buckets_tsv: `{out_dir / ERROR_BUCKET_OUTPUT_NAME}`\n"
        f"- alignment_resolution_tsv: `{out_dir / ALIGNMENT_RESOLUTION_OUTPUT_NAME}`\n"
        f"- risk_review_queue_tsv: `{out_dir / RISK_REVIEW_QUEUE_OUTPUT_NAME}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "cell_rows": len(reporting_cells),
        "summary_rows": len(summary_rows),
        "error_rows": len(error_rows),
        "alignment_resolution_rows": len(alignment_resolution_rows),
        "risk_review_queue_rows": len(risk_queue_rows),
        "cells_tsv": str(out_dir / CELL_OUTPUT_NAME),
        "summary_tsv": str(out_dir / SUMMARY_OUTPUT_NAME),
        "error_buckets_tsv": str(out_dir / ERROR_BUCKET_OUTPUT_NAME),
        "alignment_resolution_tsv": str(out_dir / ALIGNMENT_RESOLUTION_OUTPUT_NAME),
        "risk_review_queue_tsv": str(out_dir / RISK_REVIEW_QUEUE_OUTPUT_NAME),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
