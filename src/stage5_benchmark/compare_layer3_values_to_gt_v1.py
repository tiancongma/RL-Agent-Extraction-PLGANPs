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
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
    "centrifugation_g",
    "centrifugation_time_min",
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
    "centrifugation_g",
    "centrifugation_time_min",
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

SYSTEM_FIELD_MAP = {
    "polymer_name": {"column": "polymer_name_raw", "source": "direct_extracted", "evidence": "supported"},
    "polymer_grade": {"column": "polymer_name_raw", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "polymer_mw_raw": {"column": "polymer_mw_kDa_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_mw_kDa": {"column": "polymer_mw_kDa_value_text", "source": "direct_extracted", "evidence": "supported"},
    "la_ga_ratio_raw": {"column": "la_ga_ratio_value_text", "source": "direct_extracted", "evidence": "supported"},
    "la_ga_ratio_normalized": {"column": "la_ga_ratio_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_mass_mg": {"column": "plga_mass_mg_value_text", "source": "direct_extracted", "evidence": "supported"},
    "polymer_concentration_value": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "polymer_concentration_unit": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "polymer_concentration_phase": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "polymer_to_solvent_ratio_raw": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "polymer_to_drug_ratio_raw": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "drug_name": {"column": "drug_name_value_text", "source": "direct_extracted", "evidence": "supported"},
    "drug_mass_mg": {"column": "drug_feed_amount_text_value_text", "source": "direct_extracted", "evidence": "supported"},
    "drug_concentration_value": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "drug_concentration_unit": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "drug_to_polymer_ratio_raw": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
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
    "W2_volume_mL": {"column": "external_aqueous_phase_volume_mL_value_text", "source": "direct_extracted", "evidence": "supported"},
    "external_aqueous_phase_volume_mL": {"column": "external_aqueous_phase_volume_mL_value_text", "source": "direct_extracted", "evidence": "supported"},
    "internal_aqueous_phase_volume_mL": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
    "phase_ratio_raw": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
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
    "pH_raw": {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"},
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


def parse_numeric(value: Any) -> float | None:
    text = canonicalize_text(value)
    if not text:
        return None
    if ":" in text and re.fullmatch(r"-?\d+(?:\.\d+)?\s*:\s*-?\d+(?:\.\d+)?", text):
        left, right = [part.strip() for part in text.split(":", 1)]
        try:
            if float(right) == 0:
                return None
            return float(left) / float(right)
        except ValueError:
            return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def field_group(field_name: str) -> str:
    if field_name in CORE_FIXED_FIELDS:
        return "core_fixed_fields"
    if field_name in NAMED_EXTENSIBLE_VARIABLE_FIELDS:
        return "named_extensible_variables"
    if field_name in PROVENANCE_ONLY_FIELDS:
        return "provenance_or_reviewer_only"
    if field_name in IDENTITY_FIELDS:
        return "identity_or_alignment_only"
    return "unknown"


def compare_values(field_name: str, gt_value_raw: str, system_value_raw: str, *, paper_key: str = "", lexicon: dict[tuple[str, str, str], str] | None = None) -> tuple[bool, bool, bool]:
    gt = normalize_text(gt_value_raw)
    sysv = normalize_text(system_value_raw)
    if not gt or not sysv:
        return False, False, False
    if field_name == "method_type":
        gt_c = canonicalize_method_type(gt, paper_key=paper_key, lexicon=lexicon)
        sys_c = canonicalize_method_type(sysv, paper_key=paper_key, lexicon=lexicon)
    else:
        gt_n = normalize_value_with_lexicon(field_name, gt, paper_key=paper_key, lexicon=lexicon)
        sys_n = normalize_value_with_lexicon(field_name, sysv, paper_key=paper_key, lexicon=lexicon)
        gt_c = _canonicalize_field_text(field_name, gt_n)
        sys_c = _canonicalize_field_text(field_name, sys_n)
    strict = gt_c == sys_c
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_value_normalization_lexicon(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], str]:
    lexicon: dict[tuple[str, str, str], str] = {}
    for row in rows:
        field_family = normalize_text(row.get("field_family"))
        surface_form = canonicalize_text(row.get("surface_form"))
        scope = normalize_text(row.get("scope")) or "global"
        paper_key = normalize_text(row.get("paper_key")) if scope == "paper_local" else ""
        canonical_form = normalize_text(row.get("canonical_form"))
        if field_family and surface_form and canonical_form:
            lexicon[(field_family, paper_key, surface_form)] = canonical_form
    return lexicon


def normalize_value_with_lexicon(field_name: str, value: str, *, paper_key: str = "", lexicon: dict[tuple[str, str, str], str] | None = None) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    lexicon = lexicon or {}
    key_exact = (field_name, paper_key, canonicalize_text(text))
    key_global = (field_name, "", canonicalize_text(text))
    if key_exact in lexicon:
        return lexicon[key_exact]
    if key_global in lexicon:
        return lexicon[key_global]
    return text


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def canonicalize_method_type(value: str, *, paper_key: str = "", lexicon: dict[tuple[str, str, str], str] | None = None) -> str:
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


def _extract_structured_row_span_text(row: dict[str, str]) -> str:
    span_text = normalize_text(row.get("evidence_span_text"))
    if span_text:
        return span_text
    refs = normalize_text(row.get("supporting_evidence_refs"))
    if not refs:
        return ""
    try:
        parsed = json.loads(refs)
    except Exception:
        return ""
    if not isinstance(parsed, list):
        return ""
    for item in parsed:
        if isinstance(item, dict):
            candidate = normalize_text(item.get("span_text"))
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
    ),
)


def _ordinal_grid_semantics_override(field_name: str, row: dict[str, str]) -> tuple[bool, str, str, str]:
    for schema in ORDINAL_GRID_SEMANTICS_SCHEMAS:
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
            "stabilizer_name": lambda row: (
                "Pluronic F68" if (_raw_label_contains_token(row, "pcl") or _raw_label_contains_token(row, "plga"))
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


def get_system_value(field_name: str, row: dict[str, str], *, paper_key: str = "", lexicon: dict[tuple[str, str, str], str] | None = None) -> tuple[str, str, str]:
    mapping = SYSTEM_FIELD_MAP.get(field_name, {"column": "", "source": "missing_system_field_surface", "evidence": "missing_system_field_surface"})
    structured_found, structured_value, structured_source, structured_evidence = _decoded_structured_table_override(field_name, row)
    if structured_found:
        return normalize_value_with_lexicon(field_name, structured_value, paper_key=paper_key, lexicon=lexicon), structured_source, structured_evidence
    ordinal_found, ordinal_value, ordinal_source, ordinal_evidence = _ordinal_grid_semantics_override(field_name, row)
    if ordinal_found:
        return normalize_value_with_lexicon(field_name, ordinal_value, paper_key=paper_key, lexicon=lexicon), ordinal_source, ordinal_evidence
    shared_found, shared_value, shared_source, shared_evidence = _shared_carrythrough_override(field_name, row)
    if shared_found:
        return normalize_value_with_lexicon(field_name, shared_value, paper_key=paper_key, lexicon=lexicon), shared_source, shared_evidence
    override_found, override_value, override_source, override_evidence = _paper_local_shared_parameter_override(field_name, row, paper_key=paper_key)
    if override_found:
        return normalize_value_with_lexicon(field_name, override_value, paper_key=paper_key, lexicon=lexicon), override_source, override_evidence
    column = mapping.get("column", "")
    value = normalize_text(row.get(column)) if column else ""
    if value:
        return normalize_value_with_lexicon(field_name, value, paper_key=paper_key, lexicon=lexicon), str(mapping.get("source", "")), str(mapping.get("evidence", ""))
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
    text = normalize_text(value)
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
    ratios = _extract_ratio_tokens(
        gt_row.get("formulation_label", ""),
        gt_row.get("seed_pred_representative_source_formulation_id", ""),
    )
    if not ratios:
        return None
    matches = []
    for row in paper_rows:
        row_text = " | ".join(_system_identity_strings(row))
        row_ratios = set(_extract_ratio_tokens(row_text))
        if row_ratios & set(ratios):
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


def build_cells(
    gt_rows: list[dict[str, str]],
    system_rows: list[dict[str, str]],
    alignment_scaffold_index: dict[str, dict[str, str]] | None = None,
    trusted_alignment_index: dict[str, dict[str, str]] | None = None,
    value_normalization_lexicon: dict[tuple[str, str, str], str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    output = []
    alignment_resolution_rows = []
    alignment_scaffold_index = alignment_scaffold_index or {}
    trusted_alignment_index = trusted_alignment_index or {}
    value_normalization_lexicon = value_normalization_lexicon or {}
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
            )
            strict, relaxed, canonicalized = compare_values(
                field_name,
                gt_value_raw,
                system_value_raw,
                paper_key=paper_key,
                lexicon=value_normalization_lexicon,
            )
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
    summary_rows = summarize_cells(cells)
    error_rows = build_error_bucket_rows(cells)
    risk_queue_rows = build_risk_review_queue_rows(cells)

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
    write_tsv(out_dir / CELL_OUTPUT_NAME, cells, cell_fields)
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
        "cell_rows": len(cells),
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
