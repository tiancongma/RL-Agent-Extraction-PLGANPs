#!/usr/bin/env python3
from __future__ import annotations

"""
Build the Stage5 benchmark-final formulation table.

Contract:
- This module owns the benchmark-final family only.
- It may perform source-faithful final-row closure, identity-preserving
  filtering, conservative duplicate or variant collapse under explicit rules,
  and explicit Stage3 resolved-field carry-through for governed resolved
  fields.
- It must not perform modeling-ready normalization, donor-fill,
  assumption-based inference, or target-schema convenience projection.

Implementation note:
- Some older Stage5 derivation or curated-projection helpers still operate on
  legacy weak-label artifacts. Those are branch-only modeling utilities and are
  not part of this benchmark-final contract.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

from src.stage2_sampling_labels.table_structure_dictionary_v1 import (
    canonical_field_for_header,
    lexicon_rows_for_family,
    normalize_dictionary_value,
)
from src.stage5_benchmark.material_value_binding_v1 import (
    build_material_alias_graph,
    evaluate_canonical_promotions,
    extract_entity_bound_values,
    validate_direct_value,
)
from src.utils.preparation_method_fields_v1 import PREPARATION_METHOD_FIELDNAMES
from src.utils.paths import PROJECT_ROOT, dataset_text_root


DECISION_TRACE_NAME = "final_output_decision_trace_v1.tsv"
FINAL_TABLE_NAME = "final_formulation_table_v1.tsv"
DOWNSTREAM_VARIANT_RECORDS_NAME = "downstream_variant_records_v1.tsv"
SUMMARY_NAME = "final_output_summary_v1.md"
VALUE_LAYER_SIDECAR_MANIFEST_TSV_NAME = "stage5_value_layer_sidecar_manifest_v1.tsv"
VALUE_LAYER_SIDECAR_MANIFEST_JSON_NAME = "stage5_value_layer_sidecar_manifest_v1.json"
VALUE_LAYER_DIRECT_COPY_NAME = "stage5_value_layer_s5_4_accepted_direct_values_v1.tsv"
VALUE_LAYER_DERIVED_COPY_NAME = "stage5_value_layer_s5_5_derived_values_v1.tsv"
RELATION_RECORDS_NAME = "formulation_relation_records_v1.tsv"
RESOLVED_RELATION_FIELDS_NAME = "resolved_relation_fields_v1.tsv"
RESOLVED_RELATION_FIELD_NAMES = {
    "drug_to_polymer_ratio_raw",
    "drug_feed_amount_text",
    "drug_name",
    "emul_method",
    "emul_type",
    "la_ga_ratio",
    "la_ga_ratio_normalized",
    "la_ga_ratio_raw",
    "organic_solvent",
    "organic_phase_volume_mL",
    "external_aqueous_phase_volume_mL",
    "drug_concentration_value",
    "drug_concentration_unit",
    "encapsulation_efficiency_percent",
    "dl_percent",
    "loading_content_percent",
    "plga_mass_mg",
    "polymer_identity",
    "polymer_mw_kDa",
    "polymer_name_raw",
    "polymer_concentration_value",
    "polymer_concentration_unit",
    "polymer_to_drug_ratio_raw",
    "phase_ratio_raw",
    "preparation_method",
    "evaporation_time_h",
    "pva_conc_percent",
    "pH_raw",
    "pdi",
    "size_nm",
    "stirring_time_h",
    "stabilizer_name",
    "surfactant_concentration_text",
    "surfactant_name",
    "zeta_mV",
}
RESOLVED_RELATION_TEXT_FIELDS = {
    "drug_name",
    "emul_method",
    "emul_type",
    "organic_solvent",
    "polymer_identity",
    "polymer_name_raw",
    "preparation_method",
    "stabilizer_name",
    "surfactant_name",
}
RESOLVED_RELATION_PLAIN_FIELDS = {
    "polymer_identity",
    "polymer_name_raw",
}
RESOLVED_RELATION_MASS_FIELDS = {"drug_feed_amount_text", "plga_mass_mg"}
RESOLVED_RELATION_VOLUME_FIELDS = {"organic_phase_volume_mL", "external_aqueous_phase_volume_mL"}
RESOLVED_RELATION_NUMERIC_FIELDS = {
    "dl_percent",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
    "pdi",
    "polymer_mw_kDa",
    "pva_conc_percent",
    "size_nm",
    "stirring_time_h",
    "surfactant_concentration_text",
    "zeta_mV",
    "drug_concentration_value",
    "evaporation_time_h",
    "polymer_concentration_value",
    "pH_raw",
}
RESOLVED_RELATION_UNIT_FIELDS = {"drug_concentration_unit", "polymer_concentration_unit"}
RESOLVED_RELATION_RATIO_FIELDS = {
    "drug_to_polymer_ratio_raw",
    "la_ga_ratio",
    "la_ga_ratio_normalized",
    "la_ga_ratio_raw",
    "phase_ratio_raw",
    "polymer_to_drug_ratio_raw",
}
STUDIED_VARIABLES_FIELD = "studied_variables_json"
STUDIED_VARIABLE_RESPONSE_PATTERNS = (
    r"\byield\b",
    r"\bencapsulation\b",
    r"\befficiency\b",
    r"\bDEE\b",
    r"\bparticle\s+size\b",
    r"\bsize\s*\(?\s*nm\s*\)?\b",
    r"\bPDI\b",
    r"\bzeta\b",
    r"\bresponse\b",
)
STUDIED_VARIABLE_IDENTIFIER_PATTERNS = (
    r"\brun\b",
    r"\bstandard\s+order\b",
    r"\border\b",
    r"\brow\b",
)
STAGE5_GLOBAL_PREPARATION_FIELDNAMES = [
    "organic_phase_volume_mL_value",
    "organic_phase_volume_mL_value_text",
    "organic_phase_volume_mL_scope",
    "organic_phase_volume_mL_membership_confidence",
    "organic_phase_volume_mL_evidence_region_type",
    "organic_phase_volume_mL_missing_reason",
    "external_aqueous_phase_volume_mL_value",
    "external_aqueous_phase_volume_mL_value_text",
    "external_aqueous_phase_volume_mL_scope",
    "external_aqueous_phase_volume_mL_membership_confidence",
    "external_aqueous_phase_volume_mL_evidence_region_type",
    "external_aqueous_phase_volume_mL_missing_reason",
    "stirring_time_h_value",
    "stirring_time_h_value_text",
    "stirring_time_h_scope",
    "stirring_time_h_membership_confidence",
    "stirring_time_h_evidence_region_type",
    "stirring_time_h_missing_reason",
    "evaporation_time_h_value",
    "evaporation_time_h_value_text",
    "evaporation_time_h_scope",
    "evaporation_time_h_membership_confidence",
    "evaporation_time_h_evidence_region_type",
    "evaporation_time_h_missing_reason",
    "pH_raw_value",
    "pH_raw_value_text",
    "pH_raw_scope",
    "pH_raw_membership_confidence",
    "pH_raw_evidence_region_type",
    "pH_raw_missing_reason",
    "drug_concentration_value_value",
    "drug_concentration_value_value_text",
    "drug_concentration_value_scope",
    "drug_concentration_value_membership_confidence",
    "drug_concentration_value_evidence_region_type",
    "drug_concentration_value_missing_reason",
    "drug_concentration_unit_value",
    "drug_concentration_unit_value_text",
    "drug_concentration_unit_scope",
    "drug_concentration_unit_membership_confidence",
    "drug_concentration_unit_evidence_region_type",
    "drug_concentration_unit_missing_reason",
    "surfactant_concentration_value_value",
    "surfactant_concentration_value_value_text",
    "surfactant_concentration_value_scope",
    "surfactant_concentration_value_membership_confidence",
    "surfactant_concentration_value_evidence_region_type",
    "surfactant_concentration_value_missing_reason",
    "surfactant_concentration_unit_value",
    "surfactant_concentration_unit_value_text",
    "surfactant_concentration_unit_scope",
    "surfactant_concentration_unit_membership_confidence",
    "surfactant_concentration_unit_evidence_region_type",
    "surfactant_concentration_unit_missing_reason",
    "shared_parameters_json",
    STUDIED_VARIABLES_FIELD,
]

LEGACY_FIELD_ALIASES = {
    "ee_percent": "encapsulation_efficiency_percent",
    "lc_percent": "loading_content_percent",
    "particle_size_nm": "size_nm",
    "plga_mw_kDa": "polymer_mw_kDa",
    "drug_mass_mg": "drug_feed_amount_text",
    "polymer_mass_mg": "plga_mass_mg",
    "O_volume_mL": "organic_phase_volume_mL",
    "o_volume_mL": "organic_phase_volume_mL",
    "o_volume_ml": "organic_phase_volume_mL",
    "W2_volume_mL": "external_aqueous_phase_volume_mL",
    "w2_volume_mL": "external_aqueous_phase_volume_mL",
    "w2_volume_ml": "external_aqueous_phase_volume_mL",
    "external_aqueous_phase_volume_ml": "external_aqueous_phase_volume_mL",
    "organic_phase_volume_ml": "organic_phase_volume_mL",
    "zeta_mv": "zeta_mV",
}
SOURCE_GROUP_DIRECT_CARRYTHROUGH_FIELDS = (
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
    "dl_percent",
)
WFDTQ4VX_DOI = "10.1080/10717544.2016.1199605"
IDENTITY_VARIABLES_FIELD = "identity_variables_json"


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


def normalize_doi(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^doi\s*:\s*", "", text)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\.org/", "", text)
    return text.strip()


def canonical_field_name(field_name: Any) -> str:
    return LEGACY_FIELD_ALIASES.get(str(field_name or "").strip(), str(field_name or "").strip())


def clean_ocr_token(value: Any) -> str:
    text = str(value or "").replace("\x04", "-")
    text = re.sub(r"[^\x20-\x7E]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_numeric(value: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", clean_ocr_token(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_percent(value: Any) -> float | None:
    return parse_numeric(value)


def parse_mass_mg(value: Any) -> float | None:
    text = clean_ocr_token(value).lower()
    if not text:
        return None
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(mg|g|ug|mcg)?", text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2) or "mg"
    if unit == "g":
        amount *= 1000.0
    elif unit in {"ug", "mcg"}:
        amount /= 1000.0
    return amount


def is_valid_direct_mass_text(value: Any) -> bool:
    """Return True only for direct source text containing numeric mass units.

    This S5-2 guard is intentionally shape/type based.  It accepts direct mass
    mentions, optionally followed by a material identity, but rejects identities,
    ratios, volumes, and concentration-only expressions so those values are not
    silently materialized into direct mass fields.
    """
    text = str(value or "").strip()
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text).strip()
    if "|" in normalized:
        segments = [segment.strip() for segment in normalized.split("|") if segment.strip()]
        return bool(segments) and all(is_valid_direct_mass_text(segment) for segment in segments)
    if re.search(r"\b\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?\b", normalized):
        return False
    mass_unit = r"(?:mg|g|ug|µg|μg|mcg|ng)"
    if re.search(rf"\b{mass_unit}\s*/\s*(?:m[lL]|l|L)\b|%\s*w\s*/\s*v|w\s*/\s*v\s*%", normalized, flags=re.IGNORECASE):
        return False
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:m[lL]|µ[lL]|u[lL]|l|L)\b", normalized) and not re.search(
        rf"\b\d+(?:\.\d+)?\s*{mass_unit}\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return False
    return bool(re.search(rf"\b\d+(?:\.\d+)?\s*{mass_unit}\b", normalized, flags=re.IGNORECASE))


def blank_invalid_direct_mass_fields(row: dict[str, str]) -> set[str]:
    """Blank invalid direct mass bundles while preserving later lawful carrythrough."""
    return blank_invalid_final_typed_fields(row, fields=("drug_feed_amount_text", "plga_mass_mg"))


_FINAL_TYPED_FIELD_VALUE_TYPES = {
    "drug_feed_amount_text": "mass",
    "plga_mass_mg": "mass",
    "organic_phase_volume_mL": "volume",
    "external_aqueous_phase_volume_mL": "volume",
    "surfactant_concentration_text": "concentration",
    "surfactant_concentration_value": "concentration",
    "pva_conc_percent": "concentration",
    "drug_concentration_value": "concentration",
    "polymer_concentration_value": "concentration",
}


def _final_typed_field_expression(row: dict[str, str], field_name: str) -> str:
    value = normalize_text(row.get(f"{field_name}_value_text", "")) or normalize_text(row.get(f"{field_name}_value", ""))
    if field_name in {"drug_concentration_value", "polymer_concentration_value"}:
        unit = normalize_text(row.get(f"{field_name.replace('_value', '_unit')}_value", "")) or normalize_text(
            row.get(f"{field_name.replace('_value', '_unit')}_value_text", "")
        )
        if unit and value and unit.lower() not in value.lower():
            value = f"{value} {unit}"
    if field_name == "pva_conc_percent" and value and re.fullmatch(r"\d+(?:\.\d+)?", value):
        value = f"{value}%"
    return value


def blank_invalid_final_typed_fields(row: dict[str, str], *, fields: tuple[str, ...] | None = None) -> set[str]:
    """Blank invalid final numeric bundles and record typed invalid reasons.

    This is the Stage5 final-boundary counterpart to the Stage2 compatibility
    validator. It is intentionally shape/type based: identities, ratios,
    concentration-only strings in mass fields, and identity strings in volume or
    concentration fields are blanked so later lawful direct carrythrough can fill
    them without being blocked by a bogus populated value.
    """
    invalidated: set[str] = set()
    target_fields = fields or tuple(_FINAL_TYPED_FIELD_VALUE_TYPES)
    for field_name in target_fields:
        value_type = _FINAL_TYPED_FIELD_VALUE_TYPES.get(field_name)
        if not value_type:
            continue
        expression = _final_typed_field_expression(row, field_name)
        if not expression:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", expression):
            # Final-row fields often carry units in their canonical field name or
            # row-local header context (e.g. `PLGA (mg)` -> value `5`). A pure
            # numeric value is therefore type-compatible at this final boundary;
            # identity strings and wrong-unit expressions remain rejected.
            continue
        validation = validate_direct_value(expression, value_type)
        if validation.get("status") == "valid":
            continue
        row[f"{field_name}_value"] = ""
        value_text_key = f"{field_name}_value_text"
        if value_text_key in row:
            row[value_text_key] = ""
        missing_reason_key = f"{field_name}_missing_reason"
        if missing_reason_key in row:
            row[missing_reason_key] = validation.get("reason", f"invalid_{value_type}_value")
        invalidated.add(field_name)
    return invalidated


def format_numeric_signature(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def format_mass_mg_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{format_numeric_signature(value)} mg"


def extract_numeric_ratio_pair(value: Any) -> tuple[float, float] | None:
    text = normalize_text(value)
    match = re.search(r"(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    left = float(match.group(1))
    right = float(match.group(2))
    if left <= 0 or right <= 0:
        return None
    return left, right


def row_text_bundle(row: dict[str, str]) -> str:
    return " ".join(
        str(row.get(name, "") or "")
        for name in [
            "raw_formulation_label",
            "representative_source_raw_formulation_label",
            "formulation_role",
            "change_role",
            "instance_context_tags",
            "change_context_tags",
            "change_descriptions",
            "evidence_span_text",
            "supporting_evidence_refs",
        ]
    ).lower()


def row_is_blank_control_or_helper(row: dict[str, str]) -> bool:
    bundle = row_text_bundle(row)
    return bool(
        re.search(
            r"\b(blank|empty|drug\s*free|unloaded|control|helper|fitc|fluorescen(?:t|ce)|commercial|comparator)\b",
            bundle,
            flags=re.IGNORECASE,
        )
    )


def row_allows_shared_preparation_mass_carrythrough(row: dict[str, str]) -> bool:
    if row_is_blank_control_or_helper(row):
        return False
    bundle = row_text_bundle(row)
    if re.search(r"\b(doe|checkpoint|coded|factorial|factor\s+level|design\s+row)\b", bundle):
        return False
    if re.search(r"\b(assay|release|uptake|cytotoxicity|cell|medium|blank)\b", bundle) and not re.search(
        r"\b(formulation|batch|np|nanoparticle|particle|loaded)\b", bundle
    ):
        return False
    return True


def _split_source_sentences(source_text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(source_text or "")).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.;])\s+", normalized) if part.strip()]


def _mass_mentions_in_text(text: str) -> list[tuple[float, str, str]]:
    mentions: list[tuple[float, str, str]] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(mg|g|µg|ug|mcg)\b", text, flags=re.IGNORECASE):
        following = text[match.end() : min(len(text), match.end() + 8)]
        if re.match(r"\s*/\s*(?:mL|ml|L|l)\b", following):
            continue
        amount = parse_mass_mg(match.group(0))
        if amount is None:
            continue
        after = text[match.end() : min(len(text), match.end() + 90)]
        before = text[max(0, match.start() - 70) : match.start()]
        context = f"__MASS__ {match.group(0)} {after} __BEFORE__ {before}"
        mentions.append((amount, match.group(0), context))
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9α-ωΑ-Ωββ\-/ ]{1,40})\s*\(\s*(\d+(?:\.\d+)?)\s*(mg|g|µg|ug|mcg)\s*\)", text, flags=re.IGNORECASE):
        amount = parse_mass_mg(f"{match.group(2)} {match.group(3)}")
        if amount is None:
            continue
        label = match.group(1)
        context = f"__PAREN_LABEL__ {label} __MASS__ {match.group(2)} {match.group(3)} " + text[max(0, match.start() - 40) : min(len(text), match.end() + 60)]
        mentions.append((amount, f"{match.group(2)} {match.group(3)}", context))
    return mentions


def _classify_shared_mass_context(context: str, *, drug_name: str = "") -> str:
    lowered = normalize_text(context).lower()
    drug_norm = normalize_text(drug_name).lower()
    polymer_pat = r"\b(plga|polymer|poly\s*\(|pcl|pla|plga-peg|peg-plga)\b"
    blocked_material_pat = r"\b(pva|surfactant|stabilizer|excipient|solvent|acetone|dcm|water|ethanol|span|tween|sc6oh|labrafil|lecithin|trehalose)\b"

    paren_label = ""
    paren_match = re.search(r"__paren_label__\s+(.+?)\s+__mass__", lowered)
    if paren_match:
        paren_label = paren_match.group(1).strip()
        paren_label = re.sub(r"^(?:and|or|plus|with)\s+", "", paren_label).strip()
        verb_bound = re.search(r"\b(?:prepar\w*|dissolv\w*|containing|loaded)\s+([a-z][a-z0-9\-]{1,30})\s*$", paren_label)
        if verb_bound:
            paren_label = verb_bound.group(1)
        if re.search(polymer_pat, paren_label):
            return "polymer_mass_mg"
        if drug_norm and drug_norm in paren_label:
            return "drug_mass_mg"
        if re.search(r"\b(drug|active|api)\b", paren_label):
            return "drug_mass_mg"
        if re.search(blocked_material_pat, paren_label):
            return ""
        # Parenthetical material labels are already row-local material bindings.
        # If the label itself is not a polymer/drug/excluded material, do not
        # fall through to wider sentence context, which can wrongly reclassify a
        # neighboring drug/excipient mass as polymer mass.
        return ""

    mass_match = re.search(r"__mass__\s+\d+(?:\.\d+)?\s*(?:mg|g|µg|ug|mcg)\b\s*(?:of\s+)?([^,.;()]{0,55})", lowered)
    near_after = mass_match.group(1).strip() if mass_match else ""
    if near_after:
        near_after = re.split(r"\band\s+\d+(?:\.\d+)?\s*(?:mg|g|µg|ug|mcg)\b", near_after, maxsplit=1)[0].strip()
        if re.search(polymer_pat, near_after):
            return "polymer_mass_mg"
        if drug_norm and re.search(rf"\b{re.escape(drug_norm)}\b", near_after):
            return "drug_mass_mg"
        if re.search(r"\b(drug|active|api)\b", near_after):
            return "drug_mass_mg"
        if re.search(blocked_material_pat, near_after):
            return ""
        # A nearby non-polymer material name after the mass is a drug only when
        # the caller has supplied a compatible drug identity. Otherwise keep it
        # unclassified rather than treating excipient masses as drug/polymer.
        if drug_norm and re.search(r"\b[a-z][a-z0-9\-]{1,20}\b", near_after):
            return "drug_mass_mg"
    if drug_norm and drug_norm in lowered and not re.search(blocked_material_pat, lowered):
        return "drug_mass_mg"
    return ""


def extract_unique_shared_preparation_masses(source_text: str, *, drug_name: str = "") -> dict[str, str]:
    """Extract unique directly stated preparation masses from source text.

    This helper is intentionally conservative: it only considers preparation-like
    sentences that contain nanoparticle/preparation context and at least one mass
    mention, then requires a unique value per mass field. It does not derive from
    concentration, volume, or ratios.
    """
    candidates: dict[str, set[str]] = {"drug_mass_mg": set(), "polymer_mass_mg": set()}
    for sentence in _split_source_sentences(source_text):
        lowered = sentence.lower()
        if not re.search(r"\b(prepar\w*|dissolv\w*|weigh\w*|fabricat\w*|formulat\w*|nanoparticle|particle|np[s]?|organic\s+phase|pour\w*)\b", lowered):
            continue
        if not re.search(r"\bmg|\bg|µg|\bug|\bmcg", lowered):
            continue
        for amount, _raw, context in _mass_mentions_in_text(sentence):
            field_name = _classify_shared_mass_context(context, drug_name=drug_name)
            if field_name in candidates:
                candidates[field_name].add(format_mass_mg_value(amount))
    return {field: next(iter(values)) for field, values in candidates.items() if len(values) == 1}


def extract_material_value_binding_shared_masses(source_text: str, row: dict[str, str]) -> dict[str, str]:
    """Return direct method-shared masses from the generic material/value binder.

    This adapter keeps the side-effect-free helper surface separate from Stage5
    row mutation. It uses the helper's paper-local alias graph and promotion
    review, then maps its generic canonical mass names onto the existing Stage5
    direct-field names without creating rows or overwriting row-local values.
    """
    row_id = normalize_text(row.get("formulation_id", "") or row.get("final_formulation_id", ""))
    if not row_id:
        return {}
    alias_graph = build_material_alias_graph(source_text, row_hints=[row])
    candidates = extract_entity_bound_values(source_text, alias_graph)
    if not candidates:
        return {}
    admitted_row = {
        "final_formulation_id": row_id,
        "polymer_mass_mg": field_bundle_value(row, "plga_mass_mg"),
        "drug_mass_mg": field_bundle_value(row, "drug_feed_amount_text"),
    }
    review = evaluate_canonical_promotions(candidates, [admitted_row])
    mapped: dict[str, set[str]] = {"polymer_mass_mg": set(), "drug_mass_mg": set()}
    for proposal in review.get("proposals", []):
        canonical_field = proposal.get("canonical_field", "")
        if canonical_field not in mapped:
            continue
        value = normalize_text(proposal.get("normalized_value", ""))
        unit = normalize_text(proposal.get("normalized_unit", ""))
        if value and unit == "mg":
            mapped[canonical_field].add(f"{value} {unit}")
    return {field: next(iter(values)) for field, values in mapped.items() if len(values) == 1}


def _nanocarrier_subtypes_in_text(text: str) -> set[str]:
    normalized = normalize_text(text).lower()
    subtypes: set[str] = set()
    if re.search(r"\bnanospheres?\b", normalized):
        subtypes.add("nanosphere")
    if re.search(r"\bnanocapsules?\b", normalized):
        subtypes.add("nanocapsule")
    return subtypes


def _sentences_for_row_nanocarrier_subtype(row: dict[str, str], source_text: str) -> list[str]:
    row_subtypes = _nanocarrier_subtypes_in_text(row_text_bundle(row))
    if len(row_subtypes) != 1:
        return []
    target_subtype = next(iter(row_subtypes))
    active_subtypes: set[str] = set()
    scoped_sentences: list[str] = []
    for sentence in _split_source_sentences(source_text):
        sentence_subtypes = _nanocarrier_subtypes_in_text(sentence)
        lowered = sentence.lower()
        if sentence_subtypes and re.search(r"\bprepar\w*\b", lowered):
            active_subtypes = sentence_subtypes
        scoped_subtypes = sentence_subtypes or active_subtypes
        if target_subtype in scoped_subtypes:
            scoped_sentences.append(sentence)
    return scoped_sentences


def extract_row_scoped_preparation_polymer_mass(row: dict[str, str], source_text: str) -> str:
    """Return a direct polymer mass when source has subtype-specific prep scopes.

    Some papers report multiple PLGA preparation recipes in the same article,
    e.g. nanospheres with one PLGA mass and nanocapsules with another.  A global
    unique carrythrough must skip those articles, but rows whose label/evidence
    names exactly one nanocarrier subtype may inherit the unique mass from that
    subtype's preparation scope.  This remains direct evidence: no ratios,
    concentration*volume, or field-specific hardcoded values are used.
    """
    candidates: set[str] = set()
    prep_scope = re.compile(r"\b(prepar\w*|dissolv\w*|organic\s+solution|organic\s+phase|poured?|solvent\s+displacement|interfacial\s+polymer)", re.IGNORECASE)
    for sentence in _sentences_for_row_nanocarrier_subtype(row, source_text):
        lowered = sentence.lower()
        if not prep_scope.search(sentence):
            continue
        if not re.search(r"\bmg|\bg|µg|\bug|\bmcg", lowered):
            continue
        for amount, _raw, context in _mass_mentions_in_text(sentence):
            if _classify_shared_mass_context(context) == "polymer_mass_mg":
                candidates.add(format_mass_mg_value(amount))
    return next(iter(candidates)) if len(candidates) == 1 else ""


def extract_row_scoped_preparation_surfactant_concentration(row: dict[str, str], source_text: str) -> str:
    """Return subtype-scoped preparation surfactant concentration.

    This is a direct source carrythrough for rows that name exactly one
    nanocarrier subtype. It is intentionally concentration-only: surfactant names
    may vary by GT alias/governance and are not inferred here.
    """
    prep_scope = re.compile(r"\b(prepar\w*|aqueous\s+solution|poured?|solvent\s+displacement|interfacial\s+polymer|nanoparticle|nanosphere|nanocapsule)", re.IGNORECASE)
    name_pattern = r"PVA|polyvinyl alcohol|Tween\s*80®?|Polysorbate\s*80|Lutrol(?:\s*F?\s*68)?|Pluronic\w*\s*F\s*-?\s*68|poloxamer\s*188|P188|Span®?\s*80"
    concentration_pattern = r"(\d+(?:\.\d+)?)\s*(%\s*(?:\(?\s*w\s*/\s*v\s*\)?)?|mg\s*/\s*mL|mg/ml)"
    candidates: set[str] = set()
    pattern = re.compile(rf"\b({name_pattern})(?=\W|$)[^.;]{{0,60}}?{concentration_pattern}", re.I)
    for sentence in _sentences_for_row_nanocarrier_subtype(row, source_text):
        if not prep_scope.search(sentence):
            continue
        for match in pattern.finditer(sentence):
            value = match.group(2)
            raw_unit = match.group(3) or ""
            raw_unit_l = raw_unit.lower()
            unit = "%w/v" if "w" in raw_unit_l and "v" in raw_unit_l else normalize_factor_unit(raw_unit)
            if not value or not unit:
                continue
            if unit == "%w/v":
                candidates.add(f"{value}% (w/v)")
            elif unit == "%":
                candidates.add(f"{value}%")
            else:
                candidates.add(f"{value} {unit}")
    return next(iter(candidates)) if len(candidates) == 1 else ""


def derived_mass_provenance_record(*, derived_field: str, derived_value: str, derivation_rule: str, source_fields: str) -> dict[str, str]:
    """Build a derived-mass sidecar/provenance row that cannot be direct-filled."""
    return {
        "derived_field": derived_field,
        "derived_value": derived_value,
        "derivation_rule": derivation_rule,
        "source_fields": source_fields,
        "provenance": "derived_not_direct_extracted",
        "direct_or_derived": "derived",
        "direct_field_write_allowed": "no",
        "evidence_binding_status": "derived_without_direct_text",
    }


def build_derived_mass_provenance_for_row(row: dict[str, str], *, source_text: str = "") -> list[dict[str, str]]:
    """Return derived mass candidates without writing direct mass fields."""
    provenance: list[dict[str, str]] = []
    drug_mass = parse_mass_mg(field_bundle_value(row, "drug_feed_amount_text"))
    polymer_mass = parse_mass_mg(field_bundle_value(row, "plga_mass_mg"))
    drug_to_polymer = extract_numeric_ratio_pair(field_bundle_value(row, "drug_to_polymer_ratio_raw"))
    polymer_to_drug = extract_numeric_ratio_pair(field_bundle_value(row, "polymer_to_drug_ratio_raw"))
    if drug_mass is None and polymer_mass is not None and drug_to_polymer:
        derived = polymer_mass * drug_to_polymer[0] / drug_to_polymer[1]
        provenance.append(
            derived_mass_provenance_record(
                derived_field="drug_mass_mg",
                derived_value=format_mass_mg_value(derived),
                derivation_rule="ratio_times_known_polymer_mass",
                source_fields="drug_to_polymer_ratio_raw;polymer_mass_mg",
            )
        )
    if polymer_mass is None and drug_mass is not None and polymer_to_drug:
        derived = drug_mass * polymer_to_drug[0] / polymer_to_drug[1]
        provenance.append(
            derived_mass_provenance_record(
                derived_field="polymer_mass_mg",
                derived_value=format_mass_mg_value(derived),
                derivation_rule="ratio_times_known_drug_mass",
                source_fields="polymer_to_drug_ratio_raw;drug_mass_mg",
            )
        )
    text = " ".join([str(source_text or ""), row.get("evidence_span_text", "")])
    volume_match = re.search(r"(\d+(?:\.\d+)?)\s*mL\b", text, flags=re.IGNORECASE)
    volume_ml = float(volume_match.group(1)) if volume_match else None
    if volume_ml is not None:
        for field_name, concentration_prefix, derived_field in [
            ("polymer_mass_mg", "polymer_concentration", "polymer_mass_mg"),
            ("drug_mass_mg", "drug_concentration", "drug_mass_mg"),
        ]:
            if field_name == "polymer_mass_mg" and polymer_mass is not None:
                continue
            if field_name == "drug_mass_mg" and drug_mass is not None:
                continue
            concentration_value = parse_numeric(field_bundle_value(row, f"{concentration_prefix}_value"))
            concentration_unit = field_bundle_value(row, f"{concentration_prefix}_unit").lower()
            if concentration_value is None:
                continue
            if "%" in concentration_unit or "% w/v" in text.lower() or "%w/v" in text.lower():
                derived = percent_wv_to_mg(concentration_value, volume_ml)
                rule = "percent_wv_times_volume_ml"
            elif "mg/ml" in concentration_unit or "mg/ml" in text.lower():
                derived = concentration_value * volume_ml
                rule = "mg_per_ml_times_volume_ml"
            else:
                continue
            provenance.append(
                derived_mass_provenance_record(
                    derived_field=derived_field,
                    derived_value=format_mass_mg_value(derived),
                    derivation_rule=rule,
                    source_fields=f"{concentration_prefix}_value;{concentration_prefix}_unit;volume_ml",
                )
            )
    return provenance


def wfdt_coordinate_signature(
    drug_mg: float | None,
    polymer_mg: float | None,
    surfactant_pct: float | None,
) -> str:
    return "|".join(
        [
            f"x1_drug_mg={format_numeric_signature(drug_mg)}",
            f"x2_polymer_mg={format_numeric_signature(polymer_mg)}",
            f"x3_surfactant_pct={format_numeric_signature(surfactant_pct)}",
        ]
    )


def parse_organic_phase_volume_ml(text: str) -> float:
    match = re.search(
        r"dissolving\s+plga\s*\([^)]*\)\s+and\s+drug\s*\([^)]*\)\s+in\s+(\d+(?:\.\d+)?)\s*ml\s+of\s+acetone",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise RuntimeError("Could not parse fixed organic-phase volume for WFDTQ4VX reconciliation.")
    return float(match.group(1))


def percent_wv_to_mg(percent_value: float, volume_ml: float) -> float:
    return percent_value * 10.0 * volume_ml


def extract_table1_level_map(lines: list[str]) -> dict[str, dict[float, float]]:
    try:
        anchor = next(i for i, line in enumerate(lines) if "Table 1. Factorial design parameters" in line)
    except StopIteration as exc:
        raise RuntimeError("Could not locate Table 1 factor levels.") from exc

    level_map: dict[str, dict[float, float]] = {}
    for factor_name in ["X1", "X2", "X3"]:
        for idx in range(anchor, min(anchor + 60, len(lines))):
            if lines[idx].startswith(factor_name):
                values: list[float] = []
                for probe in lines[idx + 1 : idx + 8]:
                    parsed = parse_numeric(probe)
                    if parsed is not None:
                        values.append(parsed)
                    if len(values) == 3:
                        break
                if len(values) != 3:
                    raise RuntimeError(f"Could not extract three factor levels for {factor_name}.")
                level_map[factor_name] = {-1.0: values[0], 0.0: values[1], 1.0: values[2]}
                break
        if factor_name not in level_map:
            raise RuntimeError(f"Could not locate factor row for {factor_name}.")
    return level_map


def interpolate_from_coded(levels: dict[float, float], coded_value: float) -> float:
    if coded_value in levels:
        return levels[coded_value]
    ordered = sorted(levels.items())
    xs = [k for k, _ in ordered]
    ys = [v for _, v in ordered]
    if coded_value < xs[0] or coded_value > xs[-1]:
        raise RuntimeError(f"Coded value {coded_value} is outside interpolation range {xs}.")
    for idx in range(len(xs) - 1):
        x0, x1 = xs[idx], xs[idx + 1]
        if x0 <= coded_value <= x1:
            y0, y1 = ys[idx], ys[idx + 1]
            fraction = (coded_value - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    raise RuntimeError(f"Could not interpolate coded value {coded_value}.")


def extract_checkpoint_rows(lines: list[str]) -> list[dict[str, Any]]:
    try:
        anchor = next(
            i for i, line in enumerate(lines) if "Checkpoint batches with their predicted and measured values of PS and EE" in line
        )
    except StopIteration as exc:
        raise RuntimeError("Could not locate Table 7 checkpoint rows.") from exc

    rows: list[dict[str, Any]] = []
    idx = anchor + 1
    while idx + 7 < len(lines):
        batch_label = clean_ocr_token(lines[idx])
        if not re.fullmatch(r"\d+", batch_label):
            break
        rows.append(
            {
                "batch_no": int(batch_label),
                "x1_raw": clean_ocr_token(lines[idx + 1]),
                "x2_raw": clean_ocr_token(lines[idx + 2]),
                "x3_raw": clean_ocr_token(lines[idx + 3]),
            }
        )
        idx += 8
    if not rows:
        raise RuntimeError("Checkpoint table anchor found, but no checkpoint rows were parsed.")
    return rows


def parse_coded_cell(value: Any) -> tuple[float, str]:
    cleaned = clean_ocr_token(value)
    negative_prefix = bool(str(value)[:1] and ord(str(value)[:1]) < 32)
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*\(([^)]*)\)", cleaned)
    if match:
        coded_str = match.group(1)
        actual_str = match.group(2)
    else:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
        if len(numbers) < 2:
            raise RuntimeError(f"Could not parse coded checkpoint cell: {value!r}")
        coded_str = numbers[0]
        actual_str = numbers[1]
    coded_value = float(coded_str)
    if negative_prefix and coded_value > 0:
        coded_value = -coded_value
    return coded_value, actual_str


def resolve_source_text_path_for_row(row: dict[str, str]) -> Path:
    candidates: list[Path] = []
    raw_text_path = str(row.get("text_path", "") or "").strip()
    if raw_text_path:
        candidates.append(PROJECT_ROOT / Path(raw_text_path))
        candidates.append(PROJECT_ROOT / Path(raw_text_path.replace("\\", "/")))
    key = str(row.get("key", "") or "").strip()
    if key:
        candidates.append(dataset_text_root("goren_2025") / key / f"{key}.pdf.txt")
        candidates.append(dataset_text_root("goren_2025") / key / f"{key}.html.txt")
        candidates.append(PROJECT_ROOT / "data" / "cleaned" / "content" / "text" / f"{key}.pdf.txt")
        candidates.append(PROJECT_ROOT / "data" / "cleaned" / "content" / "text" / f"{key}.html.txt")
        candidates.append(PROJECT_ROOT / "data" / "cleaned" / "content_goren_2025" / "text" / f"{key}.pdf.txt")
        candidates.append(PROJECT_ROOT / "data" / "cleaned" / "content_goren_2025" / "text" / f"{key}.html.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve source text path for {key}.")


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


def parse_polymer_identity_components_v1(value: Any) -> dict[str, str]:
    """Split a single row-local PLGA identity surface into identity components.

    This parser is intentionally row-local and conservative. It accepts surfaces
    such as `PLGA 75/25` or `PLGA (50:50, Resomer RG 502)`, where the ratio is
    attached to a single PLGA material identity. It rejects union/list surfaces
    and drug/polymer ratio contexts so Stage5 does not turn broad material lists
    into row-specific values.
    """
    text = str(value or "").replace("®", "").strip()
    if not text:
        return {}
    normalized = re.sub(r"\s+", " ", text)
    lower = normalized.lower()
    if not re.search(
        r"\bplga\b|poly\s*\(\s*(?:d[,\-\s]*l[-\s]*)?(?:lactide|lactic)\s*-?\s*co\s*-?\s*(?:glycolide|glycolic)",
        lower,
        flags=re.IGNORECASE,
    ):
        return {}
    if re.search(r"\b(?:pcl|pla|peg|polyethylene glycol|poly\(ethylene glycol\)|chitosan|alginate)\b", lower):
        return {"ambiguous": "yes"}
    if re.search(r"\b(?:drug|polymer)\s*[:/]\s*(?:polymer|drug)\s*ratio\b|\b(?:drug|polymer)\s+to\s+(?:polymer|drug)\s+ratio\b", lower):
        return {"ambiguous": "yes"}

    ratio_candidates: set[str] = set()
    ratio_pattern = re.compile(r"(?<!\d)(\d{1,3})\s*[:/]\s*(\d{1,3})(?!\d)")
    for match in ratio_pattern.finditer(normalized):
        start = max(0, match.start() - 80)
        end = min(len(normalized), match.end() + 80)
        snippet = normalized[start:end]
        snippet_lower = snippet.lower()
        if not re.search(r"\bplga\b|lactide|lactic|glycolide|glycolic", snippet_lower):
            continue
        if re.search(r"\b(?:drug|polymer)\s*[:/]\s*(?:polymer|drug)\b|\bw\s*/\s*w\b", snippet_lower):
            continue
        first = int(match.group(1))
        second = int(match.group(2))
        ratio_has_material_words = bool(re.search(r"lactide|lactic|glycolide|glycolic", snippet_lower))
        if first + second != 100 and not (first == second == 1 and ratio_has_material_words):
            continue
        ratio_candidates.add(f"{first}:{second}")
    if len(ratio_candidates) != 1:
        return {"ambiguous": "yes"} if ratio_candidates else {}
    if len(re.findall(r"\bplga\b", lower)) > 1 and re.search(r"\b(?:,|;|\bor\b|\band\b)\b", lower):
        return {"ambiguous": "yes"}
    return {
        "polymer_name": "PLGA",
        "la_ga_ratio": next(iter(ratio_candidates)),
    }


def _row_local_polymer_identity_component_surfaces(row: dict[str, str]) -> list[str]:
    surfaces: list[str] = []
    for field in (
        "polymer_name_raw",
        "polymer_name_value",
        "polymer_name_value_text",
        "polymer_identity_final",
        "polymer_identity",
        "representative_source_raw_formulation_label",
        "raw_formulation_label",
        "formulation_label",
        "formulation_id",
        "source_formulation_id",
        "evidence_span_text",
    ):
        value = str(row.get(field, "") or "").strip()
        if value:
            surfaces.append(value)
    bindings = parse_first_json_array(row.get("table_cell_bindings_json"))
    if bindings:
        direct_from_grid = direct_values_from_table_cell_grid_bindings(bindings)
        polymer_name = str(direct_from_grid.get("polymer_name", "") or "").strip()
        if polymer_name:
            surfaces.append(polymer_name)
    return surfaces


def apply_row_local_polymer_identity_component_split(
    materialized: dict[str, str],
    applied_fields: set[str],
) -> None:
    parsed_values: dict[tuple[str, str], str] = {}
    for surface in _row_local_polymer_identity_component_surfaces(materialized):
        parsed = parse_polymer_identity_components_v1(surface)
        if parsed.get("ambiguous") == "yes":
            continue
        polymer_name = parsed.get("polymer_name", "")
        ratio = parsed.get("la_ga_ratio", "")
        if polymer_name and ratio:
            parsed_values[(polymer_name, ratio)] = surface
    if len(parsed_values) != 1:
        return
    (polymer_name, ratio), _surface = next(iter(parsed_values.items()))

    current_polymer = field_bundle_value(materialized, "polymer_name")
    current_polymer_raw = normalize_text(materialized.get("polymer_name_raw", ""))
    current_polymer_components = parse_polymer_identity_components_v1(current_polymer)
    current_raw_components = parse_polymer_identity_components_v1(materialized.get("polymer_name_raw", ""))
    current_is_same_component = (
        current_polymer_components.get("polymer_name") == polymer_name
        and current_polymer_components.get("la_ga_ratio") == ratio
    ) or (
        current_raw_components.get("polymer_name") == polymer_name
        and current_raw_components.get("la_ga_ratio") == ratio
    )
    polymer_name_has_typed_bundle = any(
        field in materialized
        for field in (
            "polymer_name_value",
            "polymer_name_value_text",
            "polymer_name_scope",
            "polymer_name_membership_confidence",
            "polymer_name_evidence_region_type",
            "polymer_name_missing_reason",
        )
    )
    if not current_polymer or current_is_same_component:
        materialized["polymer_name_raw"] = polymer_name
        if polymer_name_has_typed_bundle:
            set_materialized_field_bundle(
                materialized,
                "polymer_name",
                polymer_name,
                scope="row_local_polymer_identity_component",
                evidence_region_type="row_local_polymer_identity_component_split",
                applied_fields=applied_fields,
            )
            if current_is_same_component:
                if "polymer_name_value" in materialized:
                    materialized["polymer_name_value"] = polymer_name
                if "polymer_name_value_text" in materialized:
                    materialized["polymer_name_value_text"] = polymer_name
                if "polymer_name_scope" in materialized:
                    materialized["polymer_name_scope"] = "row_local_polymer_identity_component"
                if "polymer_name_membership_confidence" in materialized:
                    materialized["polymer_name_membership_confidence"] = "medium"
                if "polymer_name_evidence_region_type" in materialized:
                    materialized["polymer_name_evidence_region_type"] = "row_local_polymer_identity_component_split"
                if "polymer_name_missing_reason" in materialized:
                    materialized["polymer_name_missing_reason"] = ""
        applied_fields.add("polymer_name")
    elif current_polymer_raw and current_polymer_raw in {"plga", "poly(lactide-co-glycolide)", "poly(lactic-co-glycolic acid)"}:
        materialized["polymer_name_raw"] = polymer_name

    for field_name in ("la_ga_ratio", "la_ga_ratio_raw", "la_ga_ratio_normalized"):
        if not any(
            field in materialized or field in STAGE5_GLOBAL_PREPARATION_FIELDNAMES
            for field in (
                f"{field_name}_value",
                f"{field_name}_value_text",
                f"{field_name}_scope",
                f"{field_name}_membership_confidence",
                f"{field_name}_evidence_region_type",
                f"{field_name}_missing_reason",
            )
        ):
            continue
        set_materialized_field_bundle(
            materialized,
            field_name,
            ratio,
            scope="row_local_polymer_identity_component",
            evidence_region_type="row_local_polymer_identity_component_split",
            applied_fields=applied_fields,
        )


def row_has_plga_polymer_identity(row: dict[str, str]) -> bool:
    identity_text = " ".join(
        str(row.get(field, "") or "")
        for field in [
            "polymer_identity_final",
            "polymer_identity",
            "polymer_name_raw",
            "polymer_name_value",
            "polymer_name_value_text",
            "raw_formulation_label",
            "representative_source_raw_formulation_label",
        ]
    )
    normalized = normalize_text(identity_text)
    return bool(
        re.search(r"\bplga\b", normalized)
        or "poly(lactide-co-glycolide" in normalized
        or "poly(lactic-co-glycolic" in normalized
        or "poly (lactide-co-glycolide" in normalized
        or "poly (lactic-co-glycolic" in normalized
    )


def row_is_plga_family_eligible_for_shared_mass(row: dict[str, str], source_text: str) -> bool:
    """Return whether a row may inherit a unique source-level PLGA mass.

    Direct row-local PLGA identity is strongest.  Some compact formulation tables
    encode PLGA in the table/method scope rather than the row fields (for
    example loaded-PLGA formulation codes); for those, allow inheritance only
    when the row has loaded/formulation evidence and the source contains a PLGA
    nanoparticle preparation scope.  This is a generic scope guard, not a
    paper/value-specific patch.
    """
    if row_has_plga_polymer_identity(row):
        return True
    if row_has_explicit_non_plga_polymer_exclusion(row):
        return False
    source_norm = normalize_text(source_text).lower()
    if not re.search(r"\bplga\b[^.;]{0,80}\b(?:nanoparticles?|nps|nanospheres?|nanocapsules?)\b", source_norm) and not re.search(
        r"\b(?:nanoparticles?|nps|nanospheres?|nanocapsules?)\b[^.;]{0,80}\bplga\b",
        source_norm,
    ):
        return False
    bundle = row_text_bundle(row)
    if re.search(r"\b(blank|empty|drug\s*free|unloaded|fitc|helper|control)\b", bundle):
        return False
    if re.search(r"\b(loaded|drug|formulation|nanoparticle|np[a-z]?\d+|nanosphere|nanocapsule)\b", bundle):
        return True
    if field_bundle_value(row, "drug_name"):
        return True
    return False


def row_has_explicit_non_plga_polymer_exclusion(row: dict[str, str]) -> bool:
    identity_text = " ".join(
        str(row.get(field, "") or "")
        for field in [
            "polymer_identity_final",
            "polymer_identity",
            "polymer_name_raw",
            "polymer_name_value",
            "polymer_name_value_text",
            "raw_formulation_label",
            "representative_source_raw_formulation_label",
            "evidence_span_text",
        ]
    )
    normalized = normalize_text(identity_text)
    if "nanoemulsion" in normalized and "plga" not in normalized:
        return True
    return bool(
        "omitting the polymer" in normalized
        or "without polymer" in normalized
        or re.search(r"\b(?:pcl|pla|peg|chitosan|alginate|liposome|solution)\b", normalized)
        and "plga" not in normalized
    )


def extract_global_polymer_material_la_ga_ratio(source_text: str) -> str:
    """Return a unique PLGA material-level LA:GA ratio from source text, if present.

    This is intentionally narrow: the ratio must be in the same local snippet as
    a PLGA/poly(lactide-co-glycolide) material mention. Equation/model response
    ratios such as `YEE = 75:25 + ...` are ignored because they do not describe
    polymer composition.
    """
    text = str(source_text or "")
    if not text:
        return ""
    candidates: set[str] = set()
    ratio_pattern = re.compile(r"(?<!\d)(\d{1,3})\s*[:/]\s*(\d{1,3})(?!\d)")
    polymer_pattern = re.compile(
        r"\bplga\b|poly\s*\(\s*(?:d[,\-\s]*l[-\s]*)?(?:lactide|lactic)\s*-?\s*co\s*-?\s*(?:glycolide|glycolic)",
        re.IGNORECASE,
    )
    material_context_pattern = re.compile(
        r"\bmaterials?\b|\bpurchased\b|\bgift sample\b|\bkindly donated\b|\bwith a ratio of\b|\binherent viscosity\b|\bMW\b|\bResomer\b|\bPurasorb\b",
        re.IGNORECASE,
    )
    non_material_context_pattern = re.compile(
        r"physical mixture|thermogram|\bDSC\b|\bFTIR\b|\bDOI\b|\bAdv Drug Deliv Rev\b|\bEur J Pharm Sci\b|\bFigure\b|\bFig\.\b|\bTo cite this article\b|\bReferences\b",
        re.IGNORECASE,
    )
    equation_pattern = re.compile(r"\bY[A-Z]{1,4}\s*[=¼]|\bX\d\b|\+|\bEquation\b", re.IGNORECASE)
    for match in ratio_pattern.finditer(text):
        start = max(0, match.start() - 140)
        end = min(len(text), match.end() + 140)
        snippet = text[start:end]
        if not polymer_pattern.search(snippet):
            continue
        if non_material_context_pattern.search(snippet):
            continue
        if not material_context_pattern.search(snippet):
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end]
        line_before_ratio = text[line_start:match.start()]
        if equation_pattern.search(line_before_ratio):
            continue
        candidates.add(f"{int(match.group(1))}:{int(match.group(2))}")
    return next(iter(candidates)) if len(candidates) == 1 else ""


def apply_global_polymer_material_carrythrough(
    *,
    final_row: dict[str, str],
    source_text: str,
) -> tuple[dict[str, str], set[str]]:
    materialized = dict(final_row)
    applied_fields: set[str] = set()
    if field_bundle_value(materialized, "la_ga_ratio"):
        return materialized, applied_fields
    ratio = extract_global_polymer_material_la_ga_ratio(source_text)
    if not ratio:
        return materialized, applied_fields
    if not row_has_plga_polymer_identity(materialized) and row_has_explicit_non_plga_polymer_exclusion(materialized):
        return materialized, applied_fields
    materialized["la_ga_ratio_value"] = ratio
    if "la_ga_ratio_value_text" in materialized:
        materialized["la_ga_ratio_value_text"] = ratio
    if "la_ga_ratio_scope" in materialized:
        materialized["la_ga_ratio_scope"] = "global_shared"
    if "la_ga_ratio_membership_confidence" in materialized:
        materialized["la_ga_ratio_membership_confidence"] = "medium"
    if "la_ga_ratio_evidence_region_type" in materialized:
        materialized["la_ga_ratio_evidence_region_type"] = "global_material_evidence"
    if "la_ga_ratio_missing_reason" in materialized:
        materialized["la_ga_ratio_missing_reason"] = ""
    applied_fields.add("la_ga_ratio")
    return materialized, applied_fields


def preparation_solvent_canonical_map() -> dict[str, str]:
    """Return solvent surface->canonical aliases from the central dictionary."""
    mapping: dict[str, str] = {}
    for row in lexicon_rows_for_family("solvent_name"):
        surface = normalize_text(row.get("surface_form", "")).lower()
        canonical = normalize_text(row.get("canonical_form", ""))
        scope = normalize_text(row.get("scope", "")) or "global"
        if surface and canonical and scope == "global":
            mapping[surface] = canonical
    return mapping


def preparation_solvent_alias_pattern() -> str:
    aliases = sorted(preparation_solvent_canonical_map(), key=len, reverse=True)
    return "|".join(re.escape(alias) for alias in aliases)



def extract_unique_global_preparation_solvent(source_text: str) -> str:
    """Return a unique organic solvent used in the nanoparticle preparation text.

    This is a source-backed Stage5 carrythrough helper for DOE/table rows whose
    row-local evidence keeps only coded variables/results. It is intentionally
    conservative: material lists, chromatography/mobile phase, extraction, and
    other assay contexts are ignored, and ambiguous multi-solvent preparation
    contexts remain blank.
    """
    text = str(source_text or "")
    if not text:
        return ""
    alias_pattern = preparation_solvent_alias_pattern()
    if not alias_pattern:
        return ""
    solvent_pattern = re.compile(rf"\b(?:{alias_pattern})\b", re.IGNORECASE)
    prep_context_pattern = re.compile(
        r"prepar|nanoparticle|nanosphere|nanocapsule|formulation|organic\s+(?:phase|solution)|dissolv|solvent\s+(?:displacement|diffusion|evaporation)|nanoprecipitation|emulsion",
        re.IGNORECASE,
    )
    synthesis_actor_pattern = re.compile(
        r"\bPLGA\b|poly\s*\(.*?glycol|\bdrug\b|loaded|polymer",
        re.IGNORECASE,
    )
    non_prep_context_pattern = re.compile(
        r"HPLC|LC[-\s]?MS|chromatograph|mobile phase|extraction|extract|assay|calibration|analysis|materials?\s+(?:listed|included|were purchased|was purchased|obtained)",
        re.IGNORECASE,
    )
    candidates: set[str] = set()
    for match in solvent_pattern.finditer(text):
        start = max(0, match.start() - 170)
        end = min(len(text), match.end() + 170)
        snippet = text[start:end]
        if non_prep_context_pattern.search(snippet):
            continue
        if not prep_context_pattern.search(snippet):
            continue
        if not synthesis_actor_pattern.search(snippet):
            continue
        raw = match.group(0).lower()
        candidates.add(normalize_dictionary_value("solvent_name", raw))
    return next(iter(candidates)) if len(candidates) == 1 else ""


def _canonical_preparation_solvent(value: str) -> str:
    raw = normalize_text(value).lower().replace("®", "")
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return ""
    return normalize_dictionary_value("solvent_name", raw)


def _format_volume_ml_value(value: str) -> str:
    numeric = str(value or "").strip()
    if not numeric:
        return ""
    try:
        number = float(numeric)
    except ValueError:
        return ""
    if number.is_integer():
        return f"{int(number)} mL"
    return f"{number:g} mL"


def _preparation_volume_snippet_allowed(snippet: str) -> bool:
    if re.search(
        r"HPLC|LC[-\s]?MS|chromatograph|mobile phase|extraction|extract|assay|calibration|analysis|release|dissolution|medium|zeta|zetasizer|capillary\s+cell|sonicat|lyophili[sz]ed|measurements?",
        snippet,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(
        r"prepar|nanoparticle|nanosphere|nanocapsule|formulation|organic\s+(?:phase|solution)|oil\s+phase|dissolv|solvent\s+(?:displacement|diffusion|evaporation)|nanoprecipitation|emulsion|dropwise|poured?",
        snippet,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(r"\bPLGA\b|poly\s*\(.*?glycol|\bdrug\b|loaded|polymer|organic\s+(?:phase|solution)|oil\s+phase", snippet, flags=re.IGNORECASE):
        return False
    return True


def extract_unique_global_preparation_organic_phase_volume(source_text: str, *, solvent_name: str) -> str:
    """Return the unique direct organic/oil phase volume bound to a known solvent.

    The solvent identity may come from row-local, Stage3, or prior global solvent
    carrythrough.  This helper does not infer from GT, concentration, ratios, or
    paper identity; it only accepts direct preparation-context phrases such as
    `5 mL of acetone`, `acetone (10 mL)`, or `organic phase ... in 1 mL acetone`.
    """
    text = str(source_text or "")
    solvent = _canonical_preparation_solvent(solvent_name)
    if not text or not solvent:
        return ""
    solvent_map = preparation_solvent_canonical_map()
    aliases = [alias for alias, canonical in solvent_map.items() if canonical == solvent]
    aliases.append(solvent)
    alias_pattern = "|".join(sorted((re.escape(alias) for alias in set(aliases) if alias), key=len, reverse=True))
    if not alias_pattern:
        return ""
    volume_unit = r"(\d+(?:\.\d+)?)\s*(?:mL|ml)"
    patterns = [
        re.compile(rf"{volume_unit}\s*(?:of\s+)?(?:{alias_pattern})\b", re.IGNORECASE),
        re.compile(rf"\b(?:{alias_pattern})\b\s*\(\s*{volume_unit}\s*\)", re.IGNORECASE),
        re.compile(rf"\b(?:{alias_pattern})\b[^.;]{{0,35}}?{volume_unit}", re.IGNORECASE),
    ]
    candidates: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
            if not _preparation_volume_snippet_allowed(snippet):
                continue
            raw_value = match.group(1)
            formatted = _format_volume_ml_value(raw_value)
            if formatted:
                candidates.add(formatted)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def extract_unique_global_preparation_solvent_phase_volume(source_text: str) -> str:
    """Return a unique preparation solvent/organic-phase volume without solvent identity.

    This covers source statements such as `volume of solvent was decided to
    10 mL`, where the paper gives a shared organic-solvent volume separately
    from a solvent-name sentence.  It remains narrower than solvent-bound
    extraction and rejects non-preparation contexts through the same snippet
    guard.
    """
    text = str(source_text or "")
    if not text:
        return ""
    volume_unit = r"(\d+(?:\.\d+)?)\s*(?:mL|ml)"
    patterns = [
        re.compile(rf"\bvolume\s+of\s+(?:the\s+)?(?:organic\s+)?solvent\s+(?:was\s+)?(?:decided|optimized|selected|maintained|fixed)\s+(?:to|as|at)\s+{volume_unit}\b", re.IGNORECASE),
        re.compile(rf"\b(?:organic\s+)?solvent\s+volume\s+(?:was\s+)?(?:decided|optimized|selected|maintained|fixed)\s+(?:to|as|at)\s+{volume_unit}\b", re.IGNORECASE),
        re.compile(rf"\b(?:organic\s+phase|organic\s+solution)\s+volume\s+(?:was\s+)?(?:decided|optimized|selected|maintained|fixed)\s+(?:to|as|at)\s+{volume_unit}\b", re.IGNORECASE),
    ]
    candidates: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 220) : min(len(text), match.end() + 220)]
            if not _preparation_volume_snippet_allowed(snippet):
                continue
            formatted = _format_volume_ml_value(match.group(1))
            if formatted:
                candidates.add(formatted)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def _preparation_external_aqueous_volume_snippet_allowed(snippet: str) -> bool:
    if re.search(
        r"HPLC|LC[-\s]?MS|chromatograph|mobile phase|extraction|extract|assay|calibration|analysis|release|dissolution|medium|zeta|zetasizer|capillary\s+cell|sonicat|lyophili[sz]ed|measurements?",
        snippet,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(
        r"prepar|nanoparticle|nanosphere|nanocapsule|formulation|external\s+aqueous|aqueous\s+(?:phase|solution)|surfactant\s+solution|water\s+phase|solvent\s+(?:displacement|diffusion|evaporation)|nanoprecipitation|emulsion|dropwise|poured?|added",
        snippet,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(
        r"\bPLGA\b|poly\s*\(.*?glycol|\bdrug\b|loaded|polymer|organic\s+(?:phase|solution)|oil\s+phase|surfactant\s+solution|aqueous\s+(?:phase|solution)",
        snippet,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def extract_unique_global_preparation_external_aqueous_phase_volume(source_text: str) -> str:
    """Return the unique direct external-aqueous phase volume from preparation text.

    This is the aqueous-phase counterpart to organic-phase solvent volume
    carrythrough. It accepts only source-backed preparation phrases such as
    `added dropwise into 10 mL of an aqueous surfactant solution` or
    `external aqueous phase (15 mL)`, rejects assay/mobile-phase contexts, and
    remains blank when multiple preparation aqueous volumes are present.
    """
    text = str(source_text or "")
    if not text:
        return ""
    volume_unit = r"(\d+(?:\.\d+)?)\s*(?:mL|ml)"
    optional_aqueous_composition = r"(?:(?:\d+(?:\.\d+)?\s*%|[\w.-]+)\s+(?:w/v\s+)?){0,4}"
    aqueous_alias = r"(?:external\s+)?aqueous\s+(?:phase|solution|surfactant\s+solution)|water\s+phase|water|aqueous|(?:PVA|polyvinyl\s+alcohol|Pluronic\w*|poloxamer|surfactant|stabilizer)\s+(?:phase|solution)?"
    patterns = [
        re.compile(rf"{volume_unit}\s*(?:of\s+)?(?:an?\s+)?{optional_aqueous_composition}(?:{aqueous_alias})\b", re.IGNORECASE),
        re.compile(rf"\b(?:{aqueous_alias})\b\s*\(\s*{volume_unit}\s*\)", re.IGNORECASE),
        re.compile(rf"\b(?:{aqueous_alias})\b[^.;]{{0,35}}?{volume_unit}", re.IGNORECASE),
    ]
    candidates: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
            if not _preparation_external_aqueous_volume_snippet_allowed(snippet):
                continue
            raw_value = match.group(1)
            formatted = _format_volume_ml_value(raw_value)
            if formatted:
                candidates.add(formatted)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def _row_preparation_carrier_scope(row: dict[str, str]) -> str:
    text = normalize_text(
        " ".join(
            str(row.get(column, "") or "")
            for column in (
                "raw_formulation_label",
                "formulation_id",
                "source_formulation_id",
                "semantic_scope_ref",
                "change_descriptions",
                "identity_variables_json",
            )
        )
    )
    if re.search(r"\bnano\s*capsules?\b|\bnanocapsules?\b", text):
        return "nanocapsule"
    if re.search(r"\bnano\s*spheres?\b|\bnanospheres?\b", text):
        return "nanosphere"
    return ""


def extract_unique_scoped_preparation_external_aqueous_phase_volume(source_text: str, row: dict[str, str]) -> str:
    """Return external-aqueous volume scoped to row carrier type when needed."""
    carrier = _row_preparation_carrier_scope(row)
    if carrier not in {"nanocapsule", "nanosphere"}:
        return ""
    text = str(source_text or "")
    if not text:
        return ""
    volume_unit = r"(\d+(?:\.\d+)?)\s*(?:mL|ml)"
    optional_aqueous_composition = r"(?:(?:\d+(?:\.\d+)?\s*%|[\w.-]+)\s+(?:w/v\s+)?){0,4}"
    aqueous_alias = r"(?:external\s+)?aqueous\s+(?:phase|solution|surfactant\s+solution)|water\s+phase|water|aqueous|(?:PVA|polyvinyl\s+alcohol|Pluronic\w*|poloxamer|surfactant|stabilizer)\s+(?:phase|solution)?"
    patterns = [
        re.compile(rf"{volume_unit}\s*(?:of\s+)?(?:an?\s+)?{optional_aqueous_composition}(?:{aqueous_alias})\b", re.IGNORECASE),
        re.compile(rf"\b(?:{aqueous_alias})\b\s*\(\s*{volume_unit}\s*\)", re.IGNORECASE),
        re.compile(rf"\bpoured\s+into\s+{volume_unit}\s+(?:of\s+)?(?:an?\s+)?{optional_aqueous_composition}(?:{aqueous_alias})\b", re.IGNORECASE),
    ]
    carrier_pattern = r"nanocapsules?|nano\s*capsules?" if carrier == "nanocapsule" else r"nanospheres?|nano\s*spheres?"
    candidates: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 260) : min(len(text), match.end() + 260)]
            if not re.search(carrier_pattern, snippet, flags=re.IGNORECASE):
                continue
            if not _preparation_external_aqueous_volume_snippet_allowed(snippet):
                continue
            formatted = _format_volume_ml_value(match.group(1))
            if formatted:
                candidates.add(formatted)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def _format_plain_numeric_value(value: str) -> str:
    numeric = str(value or "").strip()
    if not numeric:
        return ""
    try:
        number = float(numeric)
    except ValueError:
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _preparation_process_snippet_allowed(snippet: str) -> bool:
    if re.search(
        r"HPLC|LC[-\s]?MS|chromatograph|mobile phase|extraction|extract|assay|calibration|analysis|release|dissolution|medium|PBS|phosphate buffered",
        snippet,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(
        r"prepar|nanoparticle|nanosphere|nanocapsule|formulation|organic\s+(?:phase|solution)|aqueous\s+(?:phase|solution)|surfactant\s+solution|solvent\s+(?:displacement|diffusion|evaporation)|nanoprecipitation|emulsion|dropwise|stirr|evaporat|reduced pressure",
        snippet,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(
        r"\bPLGA\b|poly\s*\(.*?glycol|\bdrug\b|loaded|polymer|organic\s+(?:phase|solution)|aqueous\s+(?:phase|solution)|surfactant\s+solution|acetone",
        snippet,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def extract_unique_global_preparation_stirring_time_h(source_text: str) -> str:
    """Return a unique hour-scale stirring duration from preparation text.

    The field is materialized only when the text gives a single preparation
    stirring duration in hours.  Minute-scale durations are intentionally left
    for protocol-marker inheritance because current GT representation is mixed.
    """
    text = str(source_text or "")
    if not text:
        return ""
    hour_unit = r"(?:h|hr|hrs|hour|hours)"
    patterns = [
        re.compile(rf"stirr\w*[^.;]{{0,120}}?\bfor\s+(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*{hour_unit}\b", re.IGNORECASE),
        re.compile(rf"\bfor\s+(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*{hour_unit}\b[^.;]{{0,120}}?stirr\w*", re.IGNORECASE),
    ]
    candidates: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
            if not _preparation_process_snippet_allowed(snippet):
                continue
            if re.search(r"evaporat\w*[^.;]{0,80}?stirr\w*[^.;]{0,80}?\bfor\s+(?:approximately\s+|about\s+)?\d+(?:\.\d+)?\s*(?:h|hr|hrs|hour|hours)\b", snippet, flags=re.IGNORECASE):
                continue
            formatted = _format_plain_numeric_value(match.group(1))
            if formatted:
                candidates.add(formatted)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def extract_unique_global_preparation_evaporation_time_h(source_text: str) -> str:
    """Return a unique hour-scale solvent evaporation/removal duration."""
    text = str(source_text or "")
    if not text:
        return ""
    hour_unit = r"(?:h|hr|hrs|hour|hours)"
    patterns = [
        re.compile(rf"evaporat\w*[^.;]{{0,140}}?\b(?:for\s+)?(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*{hour_unit}\b", re.IGNORECASE),
        re.compile(rf"removed\s+by\s+evaporat\w*[^.;]{{0,140}}?\b(?:for\s+)?(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*{hour_unit}\b", re.IGNORECASE),
    ]
    candidates: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
            if not _preparation_process_snippet_allowed(snippet):
                continue
            formatted = _format_plain_numeric_value(match.group(1))
            if formatted:
                candidates.add(formatted)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def extract_unique_global_preparation_pH_raw(source_text: str) -> str:
    """Return a unique preparation aqueous-phase pH, rejecting release/buffer pH."""
    text = str(source_text or "")
    if not text:
        return ""
    patterns = [
        re.compile(r"\b(?:aqueous|water|surfactant)\s+(?:phase|solution)[^.;]{0,80}?\bat\s+pH\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE),
        re.compile(r"\bpH\s+(?:of\s+)?(?:the\s+)?(?:aqueous|water|surfactant)\s+(?:phase|solution)[^.;]{0,40}?(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    ]
    candidates: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
            if not _preparation_process_snippet_allowed(snippet):
                continue
            formatted = _format_plain_numeric_value(match.group(1))
            if formatted:
                candidates.add(formatted)
    return next(iter(candidates)) if len(candidates) == 1 else ""


EMULSIFIER_FACTOR_STOPWORDS = {
    "drug",
    "polymer",
    "plga",
    "flurbiprofen",
    "pranoprofen",
    "lopinavir",
    "ph",
    "aqueous phase",
}


def normalize_emulsifier_factor_candidate(value: str, *, paper_key: str = "") -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .,;:()[]{}\"'“”‘’"))
    if not text:
        return ""
    lowered = text.lower()
    if lowered in EMULSIFIER_FACTOR_STOPWORDS:
        return ""
    dictionary_value = normalize_dictionary_value("surfactant_name", text, paper_key=paper_key)
    if dictionary_value != text:
        return dictionary_value
    dictionary_value = normalize_dictionary_value("stabilizer_name", text, paper_key=paper_key)
    if dictionary_value != text:
        return dictionary_value
    # Conservative fallback only detects admissible material surfaces; canonical
    # display names should live in value_normalization_lexicon_v1.tsv.
    if re.search(r"\b(?:pva|polyvinyl alcohol|p188|poloxamer 188|pluronic f\s*68|tween\s*80|lutrol|brij\s*35)\b", lowered):
        return text
    return ""


def extract_row_factor_tokens(row: dict[str, str]) -> set[str]:
    """Return coded DOE/table factor labels visible on a final row."""
    tokens: set[str] = set()
    haystacks = [
        str(row.get("change_descriptions", "") or ""),
        str(row.get(IDENTITY_VARIABLES_FIELD, "") or ""),
        str(row.get("identity_variables", "") or ""),
    ]
    for text in haystacks:
        if not text:
            continue
        for match in re.finditer(r"\b(c[A-Z][A-Za-z0-9]{1,12}|X\d{1,2})\b", text):
            tokens.add(match.group(1))
    return tokens


def normalize_factor_unit(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .,;:()[]{}\"'“”‘’"))
    if not text:
        return ""
    lowered = text.lower().replace(" ", "")
    if lowered in {"mg/ml", "mgml"}:
        return "mg/mL"
    if lowered in {"%", "%w/v", "%(w/v)", "w/v%"}:
        return "%w/v" if "w/v" in lowered else "%"
    return text


def infer_unique_factor_material_name(source_text: str, role: str) -> str:
    """Infer a unique material name for a source-defined factor role.

    This is intentionally conservative: it is used only after the source has
    already defined a coded factor role such as `X3 surfactant concentration`.
    The inferred material must be unique in preparation/design context, otherwise
    no material identity is returned.
    """
    text = str(source_text or "")
    if not text:
        return ""
    role = normalize_text(role)
    prep_context = re.compile(r"prepar|formulat|nanoparticle|nanosphere|nanocapsule|factor|design|organic phase|aqueous phase", re.I)
    patterns: list[tuple[str, str]] = []
    if role == "surfactant":
        patterns = [
            (r"\b(?:Pluronic\s*F\s*68|Pluronic\s*F68)\b", "Pluronic F68"),
            (r"\b(?:poloxamer\s*188|P188)\b", "poloxamer 188"),
            (r"\b(?:PVA|polyvinyl alcohol)\b", "PVA"),
            (r"\b(?:Tween\s*80|Polysorbate\s*80)\b", "Tween 80"),
            (r"\bLutrol(?:\s*F?\s*68)?\b", "Lutrol"),
        ]
    elif role == "polymer":
        patterns = [(r"\bPLGA\b", "PLGA"), (r"\bPLA\b", "PLA"), (r"\bPCL\b", "PCL")]
    if not patterns:
        return ""
    candidates: set[str] = set()
    for pattern, canonical in patterns:
        for match in re.finditer(pattern, text, re.I):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
            if prep_context.search(snippet):
                candidates.add(canonical)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def extract_generic_concentration_factor_definition_details(source_text: str) -> dict[str, dict[str, str]]:
    """Map coded DOE factor labels to source-defined concentration roles.

    Generic examples:
    - `X1 – Drug concentration in organic phase (%w/v)`
    - `X2 – Polymer concentration in organic phase (%w/v)`
    - `X3 – Surfactant concentration (%)`
    - `cFB, concentration of flurbiprofen (mg/mL)`
    - `cP188, concentration of poloxamer 188 (mg/mL)`

    The function does not create formulation rows or invent semantic scope; it
    only decodes a row-local factor assignment that is already present on an
    admitted final row.
    """
    text = str(source_text or "")
    if not text:
        return {}
    mapping: dict[str, dict[str, str]] = {}

    def role_from_phrase(phrase: str, token: str = "") -> str:
        return infer_doe_factor_role(phrase, token)

    def add(token: str, phrase: str, unit_raw: str = "", material_raw: str = "") -> None:
        role = role_from_phrase(phrase, token)
        if not token or not role:
            return
        material = ""
        if material_raw:
            if role == "surfactant":
                material = normalize_emulsifier_factor_candidate(material_raw)
            elif role == "polymer" and re.search(r"\b(?:PLGA|PLA|PCL)\b", material_raw, re.I):
                material = re.search(r"\b(PLGA|PLA|PCL)\b", material_raw, re.I).group(1).upper()
            elif role == "drug":
                material = normalize_global_drug_candidate(material_raw)
        if not material:
            material = infer_unique_factor_material_name(text, role)
        mapping[token.lower()] = {"role": role, "unit": normalize_factor_unit(unit_raw), "material_name": material}

    for match in re.finditer(
        r"\b(X\d{1,2})\b\s*[–—\-:]*\s*((?:(?!\bX\d{1,2}\b).){0,120}?\b(?:drug|polymer|surfactant|stabilizer|emulsifier)\s+concentration(?:[^\n.;()]*)?)(?:\s*\(([^)]*)\))?",
        text,
        flags=re.I,
    ):
        add(match.group(1), match.group(2), match.group(3) or "")
    for match in re.finditer(
        r"\b(X\d{1,2})\b\s*[–—\-:]*\s*((?:(?!\bX\d{1,2}\b).){0,120}?\b(?:drug|polymer|surfactant|stabilizer|emulsifier)\s+concentration(?:(?!\bX\d{1,2}\b).){0,80}?)(?:\(([^)]*(?:w\s*/\s*v|mg\s*/\s*ml|%)[^)]*)\))",
        text,
        flags=re.I,
    ):
        add(match.group(1), match.group(2), match.group(3) or "")
    for match in re.finditer(
        r"\b(c[A-Z0-9][A-Za-z0-9]{1,12})\b\s*,\s*concentration\s+of\s+([A-Za-z][A-Za-z0-9\-\s]{1,60}?)(?:\s*\(([^)]*)\))?(?=\s*[;,.])",
        text,
    ):
        add(match.group(1), f"concentration of {match.group(2)}", match.group(3) or "", match.group(2))
    for match in re.finditer(
        r"\b(c[A-Z0-9][A-Za-z0-9]{1,12})\b\s*,\s*([A-Za-z][A-Za-z0-9\-\s]{1,60}?)\s+concentration(?:\s*\(([^)]*)\))?(?=\s*[;,.])",
        text,
    ):
        add(match.group(1), f"{match.group(2)} concentration", match.group(3) or "", match.group(2))
    for token, details in extract_doe_factor_level_definition_details(text).items():
        existing = mapping.get(token, {})
        merged = {**details, **existing}
        if details.get("level_map"):
            merged["level_map"] = details["level_map"]
        if not merged.get("unit") and details.get("unit"):
            merged["unit"] = details["unit"]
        if not merged.get("role") and details.get("role"):
            merged["role"] = details["role"]
        if not merged.get("material_name") and details.get("material_name"):
            merged["material_name"] = details["material_name"]
        mapping[token] = merged
    return mapping


def normalize_doe_level_code(value: Any) -> str:
    clean = normalize_text(value).replace("−", "-").strip()
    clean = clean.strip("[](){} ")
    clean = re.sub(r"^\+", "", clean)
    clean = re.sub(r"\s+", "", clean)
    if not clean:
        return ""
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", clean):
        return ""
    try:
        number = float(clean)
    except ValueError:
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.8f}".rstrip("0").rstrip(".")


def infer_doe_factor_role(phrase: str, token: str = "") -> str:
    clean = normalize_text(phrase)
    lowered = clean.lower()
    token_clean = normalize_text(token).lower()
    if re.search(r"\bpH\b|aqueous phase pH|water phase pH", phrase, re.I):
        return "ph"
    if re.search(r"\bratio\b|phase\s+volume", lowered):
        return "phase_ratio"
    role_patterns = [
        ("drug", r"\b(?:drug|active|api|fb|pf|flurbiprofen|lopinavir|xan|3\s*-?\s*meoxan)\b"),
        ("polymer", r"\b(?:polymer|plga|pla|pcl)\b"),
        ("surfactant", r"\b(?:surfactant|stabilizer|emulsifier|pva|p188|poloxamer|pluronic|tween|lutrol)\b"),
    ]
    if token_clean in {"cfb", "cpf"}:
        return "drug"
    if token_clean in {"cplga"}:
        return "polymer"
    if token_clean in {"cp188", "cpva"}:
        return "surfactant"
    hits: list[tuple[int, str]] = []
    for role, pattern in role_patterns:
        match = re.search(pattern, lowered)
        if match:
            hits.append((match.start(), role))
    if hits:
        return sorted(hits)[0][1]
    return ""


def _split_tableish_line(line: str) -> list[str]:
    clean = str(line or "").strip()
    if not clean:
        return []
    if "\t" in clean:
        return [str(cell or "").strip() for cell in clean.split("\t") if str(cell or "").strip()]
    cells = [str(cell or "").strip() for cell in re.split(r"\s{2,}", clean) if str(cell or "").strip()]
    if len(cells) >= 2:
        return cells
    return []


def _extract_unit_from_factor_label(label: str) -> str:
    matches = re.findall(r"\(([^)]*)\)", label)
    for raw in reversed(matches):
        unit = normalize_factor_unit(raw)
        if unit:
            return unit
    return ""


def _extract_token_from_factor_label(label: str) -> str:
    clean = normalize_text(label)
    match = re.match(r"\s*(c[A-Z0-9][A-Za-z0-9]{1,12}|X\d{1,2})\b", clean, re.I)
    if match:
        return match.group(1).lower()
    if re.search(r"\bpH\b|^ph(?:\b|_)", str(label or ""), re.I):
        return "ph"
    return ""


def _physical_value_cells(cells: list[str]) -> list[str]:
    values: list[str] = []
    for cell in cells:
        clean = normalize_text(cell).replace("−", "-").strip()
        match = re.match(r"^[-+]?\d+(?:\.\d+)?", clean)
        if not match:
            break
        values.append(match.group(0))
    return values


def extract_doe_factor_level_definition_details(source_text: str) -> dict[str, dict[str, Any]]:
    """Parse source-defined DOE level tables into coded-level physical maps.

    This is deliberately downstream and materialization-only: it can decode a
    row-local factor assignment already admitted on a final row, but it cannot
    create rows or decide which formulations exist.
    """
    text = str(source_text or "")
    if not text:
        return {}
    lines = [line for line in text.splitlines() if normalize_text(line)]
    mapping: dict[str, dict[str, Any]] = {}
    for idx, line in enumerate(lines):
        header_cells = _split_tableish_line(line)
        if len(header_cells) < 4:
            continue
        header_label = normalize_text(header_cells[0])
        if _extract_token_from_factor_label(header_cells[0]):
            continue
        if not re.search(r"\b(?:factor|level|coded|evaluated|variables?)\b", header_label):
            continue
        header_codes = [normalize_doe_level_code(cell) for cell in header_cells[1:]]
        if len([code for code in header_codes if code]) < 3:
            continue
        codes = [code for code in header_codes if code]
        for factor_line in lines[idx + 1 : idx + 16]:
            cells = _split_tableish_line(factor_line)
            if len(cells) < len(codes) + 1:
                continue
            factor_label = cells[0]
            token = _extract_token_from_factor_label(factor_label)
            if not token:
                continue
            values = _physical_value_cells(cells[1 : 1 + len(codes)])
            if len(values) < len(codes):
                continue
            unit = _extract_unit_from_factor_label(factor_label)
            role = infer_doe_factor_role(factor_label, token)
            if not role:
                continue
            material_name = ""
            if role == "surfactant":
                material_name = normalize_emulsifier_factor_candidate(factor_label)
            elif role == "polymer" and re.search(r"\b(?:PLGA|PLA|PCL)\b", factor_label, re.I):
                material_name = re.search(r"\b(PLGA|PLA|PCL)\b", factor_label, re.I).group(1).upper()
            elif role == "drug":
                material_name = normalize_global_drug_candidate(factor_label)
            if not material_name and role in {"surfactant", "polymer"}:
                material_name = infer_unique_factor_material_name(text, role)
            mapping[token] = {
                "role": role,
                "unit": unit,
                "material_name": material_name,
                "level_map": dict(zip(codes, values)),
            }
    return mapping


def extract_emulsifier_factor_definition_details(source_text: str) -> dict[str, dict[str, str]]:
    """Map coded factors such as cPVA/cP188 to source-defined material details.

    The mapping must be explicit in source text. It may include a unit when the
    definition itself states one, e.g. `cP188, concentration of poloxamer 188
    (mg/mL)`.
    """
    text = str(source_text or "")
    if not text:
        return {}
    mapping: dict[str, dict[str, str]] = {}

    def add(token: str, candidate_raw: str, unit_raw: str = "") -> None:
        candidate = normalize_emulsifier_factor_candidate(candidate_raw)
        if not token or not candidate:
            return
        mapping[token.lower()] = {"name": candidate, "unit": normalize_factor_unit(unit_raw)}

    patterns = [
        r"\b(c[A-Z0-9][A-Za-z0-9]{1,12})\b\s*,\s*concentration\s+of\s+([A-Za-z][A-Za-z0-9\-\s]{1,60}?)(?:\s*\(([^)]*)\))?(?=\s*[;,.])",
        r"\b(c[A-Z0-9][A-Za-z0-9]{1,12})\b\s*,\s*([A-Za-z][A-Za-z0-9\-\s]{1,60}?)\s+concentration(?:\s*\(([^)]*)\))?(?=\s*[;,.])",
        r"\b(c[A-Z0-9][A-Za-z0-9]{1,12})\b\s*\([^)]*\)\s*[,=:]\s*([A-Za-z][A-Za-z0-9\-\s]{1,60}?)(?:\s*\(([^)]*)\))?(?=\s*[;,.])",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            add(match.group(1), match.group(2), match.group(3) or "")
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9\-\s]{1,60}?)\s+concentration\s*\(\s*(c[A-Z0-9][A-Za-z0-9]{1,12})\s*\)",
        text,
    ):
        add(match.group(2), match.group(1), "")
    # Generic DOE-factor form: `X3 surfactant/stabilizer/emulsifier concentration (%w/v)`.
    # This binds the row-local numeric assignment to a concentration unit without
    # inventing a material identity when the source defines only a generic factor.
    for match in re.finditer(
        r"\b(X\d{1,2})\b(?:(?!\bX\d{1,2}\b).){0,80}\b(?:surfactant|stabilizer|emulsifier)\s+concentration\s*\(([^)]*)\)",
        text,
        flags=re.IGNORECASE,
    ):
        token = match.group(1)
        unit = normalize_factor_unit(match.group(2) or "")
        if token and unit:
            mapping[token.lower()] = {"name": "", "unit": unit}
    return mapping


def extract_emulsifier_factor_definition_map(source_text: str) -> dict[str, str]:
    """Map coded formulation factors such as cPVA/cP188 to emulsifier names.

    The mapping must be explicit in source text. This supports DOE rows whose
    row-local extraction preserved the numeric factor value but not the material
    identity encoded by the factor abbreviation.
    """
    return {
        token: details["name"]
        for token, details in extract_emulsifier_factor_definition_details(source_text).items()
        if details.get("name")
    }


def extract_row_factor_assignments(row: dict[str, str]) -> dict[str, dict[str, str]]:
    """Return actual row-local coded factor assignments, excluding coded-level clauses."""
    fragments: list[str] = []
    assignments: dict[str, dict[str, str]] = {}
    for field in ("change_descriptions", IDENTITY_VARIABLES_FIELD, "identity_variables"):
        raw = str(row.get(field, "") or "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    token = normalize_text(item.get("factor_token"))
                    name = normalize_text(item.get("name_raw") or item.get("name") or item.get("factor_name") or "")
                    value = normalize_text(item.get("decoded_factor_value") or item.get("value_raw") or item.get("value") or "")
                    if normalize_text(item.get("factor_value_kind")) == "coded" and not normalize_text(item.get("decoded_factor_value")):
                        continue
                    if normalize_text(item.get("decoding_rule")) == "already_physical_table_value":
                        value = normalize_text(item.get("decoded_factor_value") or item.get("value_raw") or item.get("value") or "")
                    unit = normalize_factor_unit(item.get("factor_unit") or item.get("unit") or "")
                    direct_token = token or name
                    if re.fullmatch(r"(?:c[A-Z0-9][A-Za-z0-9]{1,12}|X\d{1,2}(?:_actual)?|pH)", direct_token or "", flags=re.I) and value:
                        direct_token = re.sub(r"_actual$", "", direct_token, flags=re.I)
                        assignments[direct_token.lower()] = {"value": value, "unit": unit}
                        continue
                    fragments.append(f"{name}={value}")
                else:
                    fragments.append(str(item))
        else:
            fragments.append(raw)
    for fragment in fragments:
        for match in re.finditer(
            r'\b(c[A-Z0-9][A-Za-z0-9]{1,12}|X\d{1,2}(?:_actual)?|pH)\b\s*(?:\(([^)]*)\))?\s*=\s*([^\]",;|]+)',
            fragment,
            flags=re.I,
        ):
            token = re.sub(r"_actual$", "", match.group(1), flags=re.I)
            unit = normalize_factor_unit(match.group(2) or "")
            value = normalize_text(match.group(3))
            value = value.strip(" .")
            if not value or re.search(r"±|\+/-", value):
                continue
            assignments[token.lower()] = {"value": value, "unit": unit}
    return assignments


def normalize_surfactant_identity_key(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("®", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    aliases = {
        "pva": "pva",
        "polyvinyl alcohol": "pva",
        "tween80": "tween 80",
        "tween 80": "tween 80",
        "polysorbate 80": "tween 80",
        "lutrol": "lutrol f68",
        "lutrol f68": "lutrol f68",
        "pluronic f68": "pluronic f68",
        "pluronic f 68": "pluronic f68",
        "poloxamer 188": "poloxamer 188",
        "p188": "poloxamer 188",
        "span 80": "span 80",
    }
    return aliases.get(text, text)


def extract_preparation_surfactant_concentration_map(source_text: str) -> dict[str, str]:
    """Return explicit source text surfactant-name -> concentration mappings.

    This handles preparation sentences such as `PVA 0.5%, Tween 80 0.3% and
    Lutrol F68 (1%)`. It is intentionally name-bound: downstream carrythrough
    only applies when the row itself has a matching surfactant token.
    """
    text = str(source_text or "")
    if not text:
        return {}
    prep_context = re.compile(r"surfactant|stabilizer|emulsifier|aqueous solution|prepar|nanoparticle|formulation", re.I)
    name_pattern = r"PVA|polyvinyl alcohol|Tween\s*80®?|Polysorbate\s*80|Lutrol(?:\s*F?\s*68)?|Pluronic®?\s*F\s*68|poloxamer\s*188|P188|Span®?\s*80"
    concentration_pattern = r"\(?\s*(\d+(?:\.\d+)?)\s*(%\s*(?:\(?\s*w\s*/\s*v\s*\)?)?|mg\s*/\s*mL|mg/ml)?\s*\)?"
    mapping: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(rf"\b({name_pattern})(?=\W|$)\s*{concentration_pattern}", re.I)
    for match in pattern.finditer(text):
        snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
        if not prep_context.search(snippet):
            continue
        name_key = normalize_surfactant_identity_key(match.group(1))
        value = match.group(2)
        raw_unit = match.group(3) or ""
        unit = normalize_factor_unit(raw_unit)
        if not name_key or not value or not unit:
            continue
        if unit and unit.lower() not in value.lower():
            mapping[name_key].add(f"{value}{unit}" if unit == "%" else f"{value} {unit}")
        else:
            mapping[name_key].add(value)
    return {key: next(iter(values)) for key, values in mapping.items() if len(values) == 1}


def extract_row_surfactant_identity_keys(row: dict[str, str]) -> set[str]:
    candidates: set[str] = set()
    for field in ("surfactant_name_value", "surfactant_name_value_text"):
        value = row.get(field, "")
        if value:
            candidates.add(normalize_surfactant_identity_key(value))
    fragments: list[str] = []
    for field in ("change_descriptions", IDENTITY_VARIABLES_FIELD, "identity_variables", "raw_formulation_label"):
        raw = str(row.get(field, "") or "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            fragments.extend(str(item) for item in parsed)
        else:
            fragments.append(raw)
    name_pattern = re.compile(r"\b(PVA|polyvinyl alcohol|Tween\s*80®?|Polysorbate\s*80|Lutrol(?:\s*F?\s*68)?|Pluronic®?\s*F\s*68|poloxamer\s*188|P188|Span®?\s*80)(?=\W|$)", re.I)
    for fragment in fragments:
        for match in name_pattern.finditer(fragment):
            candidates.add(normalize_surfactant_identity_key(match.group(1)))
    return {candidate for candidate in candidates if candidate}


def extract_row_emulsifier_concentration_from_preparation_list(row: dict[str, str], source_text: str) -> str:
    if field_bundle_value(row, "surfactant_concentration_text"):
        return ""
    concentration_map = extract_preparation_surfactant_concentration_map(source_text)
    if not concentration_map:
        return ""
    row_keys = extract_row_surfactant_identity_keys(row)
    candidates = {concentration_map[key] for key in row_keys if key in concentration_map and concentration_map[key]}
    return next(iter(candidates)) if len(candidates) == 1 else ""


def extract_row_emulsifier_concentration_from_factor_definition(row: dict[str, str], source_text: str) -> str:
    if field_bundle_value(row, "surfactant_concentration_text"):
        return ""
    definition_details = extract_emulsifier_factor_definition_details(source_text)
    if not definition_details:
        return ""
    assignments = extract_row_factor_assignments(row)
    candidates: set[str] = set()
    for token, assignment in assignments.items():
        details = definition_details.get(token)
        if not details:
            continue
        value = assignment.get("value", "")
        unit = assignment.get("unit") or details.get("unit", "")
        if value and unit and unit.lower() not in value.lower():
            candidates.add(f"{value} {unit}")
        elif value:
            candidates.add(value)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def extract_row_emulsifier_from_factor_definition(row: dict[str, str], source_text: str) -> str:
    if field_bundle_value(row, "surfactant_name"):
        return ""
    tokens = extract_row_factor_tokens(row)
    if not tokens:
        return ""
    definition_map = extract_emulsifier_factor_definition_map(source_text)
    candidates = {
        definition_map[token.lower()]
        for token in tokens
        if token.lower() in definition_map and definition_map[token.lower()]
    }
    return next(iter(candidates)) if len(candidates) == 1 else ""


DRUG_NAME_STOPWORDS = {
    "drug",
    "polymer",
    "nanoparticle",
    "nanoparticles",
    "nanosphere",
    "nanospheres",
    "plga",
    "pva",
    "fitc",
    "fluorescein",
    "dexamethasone",
    "water",
    "acetone",
    "acetonitrile",
    "methanol",
    "ethanol",
}


def normalize_global_drug_candidate(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .,;:()[]{}\"'“”‘’"))
    if not text:
        return ""
    lowered = text.lower()
    if lowered in DRUG_NAME_STOPWORDS:
        return ""
    if len(text) <= 1:
        return ""
    # Drug-name abbreviation binding must stay sentence/local-material scoped.
    # Reject heading/background fragments produced by broad parenthetical scans
    # such as "Introduction Since the 1930s Methylene Blue (MB)".
    tokens = text.split()
    if len(tokens) > 5:
        return ""
    if tokens and tokens[0].lower() in {"introduction", "background", "abstract", "results", "discussion", "conclusion", "since"}:
        return ""
    if any(tok.lower() in {"introduction", "background", "abstract"} for tok in tokens):
        return ""
    return normalize_dictionary_value("drug_name", text)


def extract_drug_abbreviation_map(source_text: str) -> dict[str, str]:
    text = str(source_text or "")
    abbrev_map: dict[str, str] = {}
    for match in re.finditer(
        r"\b([A-Z][A-Za-z][A-Za-z0-9\-]{2,40})\s*\(([A-Z]{2,6})\)\s+is\s+a[^.]{0,140}\bdrug\b",
        text,
        re.IGNORECASE,
    ):
        name = normalize_global_drug_candidate(match.group(1))
        abbr = match.group(2).upper()
        if name:
            abbrev_map[abbr] = name
    for match in re.finditer(
        r"\b([A-Z][A-Za-z][A-Za-z0-9\-\s]{2,80}?)\s*\(([A-Z]{2,6})\)",
        text[:8000],
    ):
        name = normalize_global_drug_candidate(match.group(1))
        abbr = match.group(2).upper()
        snippet = text[max(0, match.start() - 160) : min(len(text), match.end() + 240)]
        if name and abbr and re.search(r"\b(?:loaded|drug|purchased|encapsulat|nanoparticle|NPs|steroid|active)\b", snippet, re.IGNORECASE):
            abbrev_map[abbr] = name
    for match in re.finditer(
        r"\b(?:c([A-Z]{2,6})|([A-Z]{2,6}))\b[^.;]{0,80}\b[Cc]oncentration\s+of\s+([A-Za-z][A-Za-z0-9\-]{2,40})\b",
        text,
    ):
        abbr = (match.group(1) or match.group(2) or "").upper()
        name = normalize_global_drug_candidate(match.group(3))
        if name and abbr:
            abbrev_map[abbr] = name
    for match in re.finditer(
        r"\bconcentration\s+of\s+([A-Za-z][A-Za-z0-9\-]{2,40})\b[^.;]{0,80}\(([A-Z]{2,6})\)",
        text,
        re.IGNORECASE,
    ):
        name = normalize_global_drug_candidate(match.group(1))
        abbr = match.group(2).upper()
        if name:
            abbrev_map[abbr] = name
    return abbrev_map


def extract_unique_global_loaded_drug_name(source_text: str) -> str:
    """Return a unique article-global drug identity for loaded PLGA rows.

    This helper is intentionally conservative and source-backed. It accepts
    title/preparation/table-factor surfaces that bind a drug or drug
    abbreviation to PLGA-loaded nanoparticles, and rejects ambiguous helper or
    assay-only contexts.
    """
    text = str(source_text or "")
    if not text:
        return ""
    abbrev_map = extract_drug_abbreviation_map(text)
    candidates: set[str] = set()

    def add_candidate(raw: str) -> None:
        token = normalize_global_drug_candidate(raw)
        if not token:
            return
        if token.upper() in abbrev_map:
            token = abbrev_map[token.upper()]
        token = normalize_global_drug_candidate(token)
        if token:
            candidates.add(token)

    loaded_patterns = [
        r"\b([A-Za-z][A-Za-z0-9\-]{1,40})-loaded\s+(?:PLGA|PCL|PLA|polymeric|polymer)\s+(?:nanoparticles|NPs|nanospheres|nanocapsules)\b",
        r"\b(?:PLGA|PCL|PLA|polymeric|polymer)\s+(?:nanoparticles|NPs|nanospheres|nanocapsules)\s+(?:loaded\s+with|containing)\s+([A-Za-z][A-Za-z0-9\-]{1,40})\b",
        r"\bformulate\s+(?:PLGA|PCL|PLA|polymeric|polymer)\s+(?:nanoparticles|NPs|nanospheres|nanocapsules)\s+of\s+([A-Za-z][A-Za-z0-9\-]{2,40})\b",
    ]
    # Use the title/abstract/methods lead as the strongest article-global drug
    # identity surface. Later reference lists and related-article blocks often
    # contain unrelated "X-loaded PLGA nanoparticles" strings.
    lead_text = text[:6000]
    loaded_counts: dict[str, int] = defaultdict(int)
    for pattern in loaded_patterns:
        for match in re.finditer(pattern, lead_text, re.IGNORECASE):
            snippet = lead_text[max(0, match.start() - 120) : min(len(lead_text), match.end() + 160)]
            if re.search(r"\b(?:FITC|fluorescein|solution control)\b", snippet, re.IGNORECASE):
                continue
            token = normalize_global_drug_candidate(match.group(1))
            if normalize_text(token) in {"coumarin", "6-coumarin", "fitc", "fluorescein"}:
                continue
            if token and token.upper() in abbrev_map:
                token = abbrev_map[token.upper()]
            token = normalize_global_drug_candidate(token)
            if token:
                loaded_counts[token.lower()] += 1
                candidates.add(token)
    if loaded_counts:
        if len(loaded_counts) == 1:
            dominant = next(iter(loaded_counts))
            for candidate in candidates:
                if candidate.lower() == dominant:
                    return candidate
        return ""

    for match in re.finditer(
        r"\b([A-Z][A-Za-z][A-Za-z0-9\-]{2,40})\s*\(([A-Z]{2,6})\)\s+is\s+a[^.]{0,140}\bdrug\b",
        lead_text,
        re.IGNORECASE,
    ):
        add_candidate(match.group(1))

    return next(iter(candidates)) if len(candidates) == 1 else ""


def _canonical_material_registry_key(value: str) -> str:
    text = normalize_text(value).replace("®", "")
    text = re.sub(r"\s+", " ", text).strip(" .,;:()[]{}\"'“”‘’")
    lowered = text.lower()
    lowered = re.sub(r"poly\s*\(\s*lactide\s*-\s*co\s*-\s*glycolide\s*\)", "plga", lowered)
    lowered = re.sub(r"poly\s*\(\s*lactic\s*-\s*co\s*-\s*glycolic\s+acid\s*\)", "plga", lowered)
    lowered = lowered.replace("poly(lactic-co-glycolic acid)", "plga")
    lowered = lowered.replace("poly(lactide-co-glycolide)", "plga")
    lowered = re.sub(r"[^a-z0-9/]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _format_material_mw_kda(value: str) -> str:
    num = parse_numeric(value)
    if num is None:
        return ""
    if float(num).is_integer():
        return f"{int(num)} kDa"
    return f"{num:g} kDa"


def canonical_polymer_display_from_registry_display(value: str) -> str:
    text = normalize_text(value).replace("®", "").strip(" ,;.")
    if not text:
        return ""
    lowered = text.lower()
    if re.search(r"\bplga\b|poly\s*\([^)]*(?:lactide|lactic)[^)]*(?:glycolide|glycolic)[^)]*\)", lowered):
        if "peg" in lowered:
            return "PLGA-PEG"
        return "PLGA"
    if re.search(r"\bpcl\b|caprolactone", lowered):
        return "PCL"
    if re.search(r"\bpla\b|polylactide|polylactic", lowered):
        return "PLA"
    return text


def _normalize_mw_number_token(value: str) -> float | None:
    token = str(value or "").strip()
    if not token:
        return None
    token = token.replace(",", "")
    token = re.sub(r"(?<=\d)\s+(?=\d{3}(?:\D|$))", "", token)
    match = re.search(r"\d+(?:\.\d+)?", token)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _format_mw_kda_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _format_mw_da_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


def _parse_material_mw_value(raw_value: str, unit_hint: str = "") -> tuple[str, str]:
    """Return `(mw_kDa, mw_raw)` for source MW expressions.

    The material registry uses the dictionary/canonical-field layer to decide
    that a local phrase is a molecular-weight field, then this parser performs
    the numeric/unit normalization.  Bare large values after an MW cue are
    treated as Da, while explicit kDa values are kept as kDa.
    """
    text = str(raw_value or "").strip()
    if not text:
        return "", ""
    cue = re.search(r"(?:Mw|M\.?\s*W\.?|Mn|molecular\s+weight|weight[-\s]*average\s+molecular\s+weight)", text, flags=re.IGNORECASE)
    if cue:
        text = text[cue.end() :]
    unit = normalize_text(unit_hint).lower()
    if not unit:
        unit_match = re.search(r"\b(kda|da|daltons?|g\s*/\s*mol)\b", text, flags=re.IGNORECASE)
        unit = normalize_text(unit_match.group(1)).lower() if unit_match else ""
    number_token = r"\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
    match = re.search(rf"({number_token})(?:\s*(?:-|–|—|to)\s*({number_token}))?", text)
    if not match:
        return "", ""
    low = _normalize_mw_number_token(match.group(1))
    high = _normalize_mw_number_token(match.group(2) or "")
    if low is None:
        return "", ""
    values = [low] + ([high] if high is not None else [])
    explicit_kda = "kda" in unit
    explicit_da = bool(unit and not explicit_kda)
    looks_like_da = any(value >= 1000 for value in values)
    as_kda = [value if explicit_kda and not looks_like_da else value / 1000 for value in values] if (explicit_da or looks_like_da) else values
    if len(as_kda) == 2:
        mw_kda = f"{_format_mw_kda_number(as_kda[0])}-{_format_mw_kda_number(as_kda[1])} kDa"
    else:
        mw_kda = f"{_format_mw_kda_number(as_kda[0])} kDa"
    if explicit_kda and not looks_like_da:
        mw_raw = mw_kda
    elif len(values) == 2:
        mw_raw = f"{_format_mw_da_number(values[0])}-{_format_mw_da_number(values[1])} Da" if (explicit_da or looks_like_da) else mw_kda
    else:
        mw_raw = f"{_format_mw_da_number(values[0])} Da" if (explicit_da or looks_like_da) else mw_kda
    return mw_kda, mw_raw


def extract_source_material_property_registry(source_text: str) -> dict[str, dict[str, str]]:
    """Return source-backed material identity/property registry entries.

    Registry entries are descriptive only: they do not create formulation rows.
    Downstream use still requires a row-local material token that uniquely maps
    to one source registry entry.
    """
    text = str(source_text or "")
    registry: dict[str, dict[str, str]] = {}
    if not text:
        return registry
    material_pattern = r"PLGA(?:\s*[-_/]?\s*PEG)?(?:\s*\d{1,3}\s*[/:-]\s*\d{1,3})?|PCL|PLA|poly\s*\([^)]*(?:lactide|lactic|glycolic|caprolactone)[^)]*\)"
    mw_field_pattern = r"(?:Mw|M\.?\s*W\.?|Mn|molecular\s+weight|weight[-\s]*average\s+molecular\s+weight)"
    value_pattern = r"\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
    patterns = [
        re.compile(rf"\b({material_pattern})\b\s*\(([^)]*?{mw_field_pattern}[^)]*?(?:{value_pattern})(?:\s*(?:-|–|—|to)\s*(?:{value_pattern}))?[^)]*)\)", re.IGNORECASE),
        re.compile(rf"\b({material_pattern})\b[^.;\n]{{0,100}}?\b({mw_field_pattern}[^.;\n]{{0,50}}?(?:{value_pattern})(?:\s*(?:-|–|—|to)\s*(?:{value_pattern}))?(?:\s*(?:kDa|Da|daltons?|g\s*/\s*mol))?)", re.IGNORECASE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            display = re.sub(r"\s+", " ", match.group(1).strip())
            key = _canonical_material_registry_key(display)
            property_text = match.group(2)
            if canonical_field_for_header(property_text) not in {"polymer_mw_kDa", "polymer_mw_raw"} and not re.search(mw_field_pattern, property_text, re.IGNORECASE):
                continue
            mw_kda, mw_raw = _parse_material_mw_value(property_text)
            if key and mw_kda:
                registry.setdefault(key, {"display_name": display, "mw_kDa": mw_kda, "mw_raw": mw_raw or mw_kda})
    return registry


def extract_row_polymer_registry_keys(row: dict[str, str]) -> set[str]:
    fragments = []
    for field in (
        "polymer_identity_final",
        "polymer_identity",
        "polymer_name_raw",
        "polymer_name_value",
        "polymer_name_value_text",
        "representative_source_raw_formulation_label",
        "raw_formulation_label",
        "change_descriptions",
        IDENTITY_VARIABLES_FIELD,
        "identity_variables",
    ):
        raw = str(row.get(field, "") or "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    fragments.append(str(item.get("value_raw") or item.get("value") or ""))
                    fragments.append(str(item.get("name_raw") or item.get("name") or ""))
                else:
                    fragments.append(str(item))
        else:
            fragments.append(raw)
    joined = " ; ".join(fragments)
    keys: set[str] = set()
    material_pattern = re.compile(r"\b(PLGA(?:\s*[-_/]?\s*PEG)?(?:\s*\d{1,3}\s*[/:-]\s*\d{1,3})?|PCL|PLA)\b", re.IGNORECASE)
    for match in material_pattern.finditer(joined):
        key = _canonical_material_registry_key(match.group(1))
        if key:
            keys.add(key)
    return keys


def extract_unique_row_bound_polymer_registry_entry(row: dict[str, str], source_text: str) -> dict[str, str]:
    registry = extract_source_material_property_registry(source_text)
    if not registry:
        return {}
    row_keys = extract_row_polymer_registry_keys(row)
    matches = {key: registry[key] for key in row_keys if key in registry}
    if len(matches) == 1:
        return next(iter(matches.values()))
    if row_has_plga_polymer_identity(row):
        plga_matches = {key: value for key, value in registry.items() if key == "plga" or key.startswith("plga ")}
        if len(plga_matches) == 1:
            return next(iter(plga_matches.values()))
    return {}


def extract_row_local_surfactant_assignment(row: dict[str, str]) -> dict[str, str]:
    fragments: list[tuple[str, str]] = []
    for field in ("change_descriptions", IDENTITY_VARIABLES_FIELD, "identity_variables"):
        raw = str(row.get(field, "") or "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    fragments.append((str(item.get("name_raw") or item.get("name") or ""), str(item.get("value_raw") or item.get("value") or "")))
                else:
                    text = str(item)
                    if "=" in text:
                        name, value = text.split("=", 1)
                        fragments.append((name, value))
        elif "=" in raw:
            for piece in re.split(r"[,;|]\s*", raw):
                if "=" in piece:
                    name, value = piece.split("=", 1)
                    fragments.append((name, value))
    name_pattern = re.compile(r"\b(PVA|polyvinyl alcohol|Tween\s*80®?|Polysorbate\s*80|Lutrol(?:\s*F?\s*68)?|Pluronic®?\s*F\s*68|poloxamer\s*188|P188|Span®?\s*80|Solutol\s*HS\s*15)\b", re.IGNORECASE)
    candidates: list[dict[str, str]] = []
    for name_raw, value_raw in fragments:
        header = normalize_text(name_raw)
        value = normalize_text(value_raw).strip(" .")
        if not header or not value:
            continue
        match = name_pattern.search(header)
        if not match:
            continue
        if not re.search(r"\b(?:surfactant|stabilizer|emulsifier|pva|polyvinyl|tween|polysorbate|lutrol|pluronic|poloxamer|p188|span|solutol)\b", header, re.IGNORECASE):
            continue
        unit_match = re.search(r"\(([^)]*(?:%|w\s*/\s*v|mg\s*/\s*mL|mg/ml)[^)]*)\)", header, re.IGNORECASE)
        unit = normalize_factor_unit(unit_match.group(1) if unit_match else "")
        if not unit and re.search(r"%\s*$", header):
            unit = "%"
        display_raw = re.sub(r"\s+", " ", match.group(1).replace("®", "").strip())
        display_aliases = {
            "pva": "PVA",
            "polyvinyl alcohol": "PVA",
            "tween 80": "Tween 80",
            "polysorbate 80": "Polysorbate 80",
            "p188": "P188",
            "poloxamer 188": "poloxamer 188",
            "lutrol": "Lutrol",
            "lutrol f68": "Lutrol F68",
            "span 80": "Span 80",
            "solutol hs 15": "Solutol HS 15",
        }
        display = display_aliases.get(display_raw.lower(), display_raw)
        header_has_generic_role = bool(re.search(r"\b(?:surfactant|stabilizer|emulsifier)\b", header, re.IGNORECASE))
        header_is_encoded_factor = bool(re.search(r"\bc[A-Z0-9]{2,8}\b", header))
        if (header_has_generic_role or header_is_encoded_factor) and re.fullmatch(r"\d+(?:\.\d+)?", value) and unit:
            concentration = f"{value}{unit}" if unit == "%" else f"{value} {unit}"
        elif (header_has_generic_role or header_is_encoded_factor) and re.fullmatch(r"\d+(?:\.\d+)?\s*%", value):
            concentration = value
        elif (header_has_generic_role or header_is_encoded_factor) and re.search(r"\d", value) and unit and unit.lower() not in value.lower():
            concentration = f"{value}{unit}" if unit == "%" else f"{value} {unit}"
        elif (header_has_generic_role or header_is_encoded_factor) and re.search(r"\d", value):
            concentration = value
        else:
            concentration = ""
        candidates.append({"name": display, "concentration": concentration})
    name_values = {item["name"] for item in candidates if item.get("name")}
    concentration_values = {item["concentration"] for item in candidates if item.get("concentration")}
    result: dict[str, str] = {}
    if len(name_values) == 1:
        result["name"] = next(iter(name_values))
    if len(concentration_values) == 1:
        result["concentration"] = next(iter(concentration_values))
    return result


def extract_unique_row_local_polymer_name(row: dict[str, str]) -> str:
    keys = extract_row_polymer_registry_keys(row)
    if len(keys) != 1:
        return ""
    key = next(iter(keys))
    displays = {
        "plga": "PLGA",
        "pcl": "PCL",
        "pla": "PLA",
    }
    if key in displays:
        return displays[key]
    if key.startswith("plga"):
        return "PLGA"
    return key.upper() if key in {"pcl", "pla"} else ""


def extract_row_bound_drug_name(row: dict[str, str], source_text: str) -> str:
    if not row_allows_global_drug_carrythrough(row):
        return ""
    labels = " ".join([
        str(row.get("representative_source_raw_formulation_label", "") or ""),
        str(row.get("raw_formulation_label", "") or ""),
        str(row.get("formulation_id", "") or ""),
    ])
    abbrev_map = extract_drug_abbreviation_map(source_text)
    candidates: set[str] = set()
    for match in re.finditer(r"\b([A-Z]{2,6})\s*[- ]\s*loaded\b", labels):
        mapped = abbrev_map.get(match.group(1).upper(), "")
        if mapped:
            candidates.add(mapped)
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9\-]{2,40})\s*[- ]\s*loaded\b", labels, re.IGNORECASE):
        token = match.group(1)
        if token.upper() in abbrev_map:
            token = abbrev_map[token.upper()]
        token = normalize_global_drug_candidate(token)
        if token:
            candidates.add(token)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def row_allows_global_drug_carrythrough(row: dict[str, str]) -> bool:
    if field_bundle_value(row, "drug_name"):
        return False
    label = normalize_text(" ".join([
        str(row.get("raw_formulation_label", "") or ""),
        str(row.get("formulation_id", "") or ""),
        str(row.get("source_formulation_id", "") or ""),
    ]))
    loaded_state = normalize_text(row.get("loaded_state_final") or row.get("loaded_state") or "")
    if loaded_state == "empty":
        return False
    if re.search(r"\b(?:blank|empty|drug free|drug-free|unloaded|without drug|no drug|fitc)\b", label):
        return False
    if re.search(r"\bnp[a-z]\d+\b", label):
        return False
    return True


def set_materialized_field_bundle(
    materialized: dict[str, str],
    field_name: str,
    value: str,
    *,
    scope: str,
    evidence_region_type: str,
    applied_fields: set[str],
) -> bool:
    value = str(value or "").strip()
    if not value or field_bundle_value(materialized, field_name):
        return False
    materialized[f"{field_name}_value"] = value
    materialized[f"{field_name}_value_text"] = value
    materialized[f"{field_name}_scope"] = scope
    materialized[f"{field_name}_membership_confidence"] = "medium"
    materialized[f"{field_name}_evidence_region_type"] = evidence_region_type
    materialized[f"{field_name}_missing_reason"] = ""
    applied_fields.add(field_name)
    return True


def set_or_replace_materialized_field_bundle(
    materialized: dict[str, str],
    field_name: str,
    value: str,
    *,
    scope: str,
    evidence_region_type: str,
    applied_fields: set[str],
    force: bool = False,
) -> bool:
    value = str(value or "").strip()
    if not value:
        return False
    if field_bundle_value(materialized, field_name) and not force:
        return False
    materialized[f"{field_name}_value"] = value
    materialized[f"{field_name}_value_text"] = value
    materialized[f"{field_name}_scope"] = scope
    materialized[f"{field_name}_membership_confidence"] = "medium"
    materialized[f"{field_name}_evidence_region_type"] = evidence_region_type
    materialized[f"{field_name}_missing_reason"] = ""
    applied_fields.add(field_name)
    return True


def split_concentration_value_unit_surface(value: Any, fallback_unit: str = "") -> tuple[str, str]:
    """Split a concentration surface into numeric value and canonical unit.

    This helper is intentionally syntactic.  It never converts units or decodes
    DOE levels; it only separates already-physical surfaces such as
    `25 mg/mL`, `0.5 %w/v`, or `0.25% (w/v)`.
    """
    clean = normalize_text(value).strip(" .")
    if not clean:
        return "", ""
    fallback = normalize_factor_unit(fallback_unit)
    match = re.match(
        r"^([-+]?\d+(?:\.\d+)?)\s*(%\s*(?:\(?\s*w\s*/\s*v\s*\)?)?|mg\s*/\s*ml|mg/ml)?$",
        clean,
        re.I,
    )
    if not match:
        return "", ""
    numeric = match.group(1)
    unit = normalize_factor_unit(match.group(2) or fallback)
    if unit not in {"", "%", "%w/v", "mg/mL"}:
        unit = ""
    return numeric, unit


def _set_structured_concentration_bundle(
    materialized: dict[str, str],
    *,
    value_field: str,
    unit_field: str,
    numeric: str,
    unit: str,
    source_field: str,
    applied: set[str],
) -> None:
    source_scope = materialized.get(f"{source_field}_scope", "") or "instance_specific"
    source_confidence = materialized.get(f"{source_field}_membership_confidence", "") or "medium"
    source_evidence = materialized.get(f"{source_field}_evidence_region_type", "") or "value_unit_splitter"
    current_value = field_bundle_value(materialized, value_field)
    if numeric and current_value != numeric:
        materialized[f"{value_field}_value"] = numeric
        materialized[f"{value_field}_value_text"] = numeric
        materialized[f"{value_field}_scope"] = source_scope
        materialized[f"{value_field}_membership_confidence"] = source_confidence
        materialized[f"{value_field}_evidence_region_type"] = source_evidence
        materialized[f"{value_field}_missing_reason"] = ""
        applied.add(value_field)
    current_unit = field_bundle_value(materialized, unit_field)
    if unit and normalize_factor_unit(current_unit) != unit:
        materialized[f"{unit_field}_value"] = unit
        materialized[f"{unit_field}_value_text"] = unit
        materialized[f"{unit_field}_scope"] = source_scope
        materialized[f"{unit_field}_membership_confidence"] = source_confidence
        materialized[f"{unit_field}_evidence_region_type"] = source_evidence
        materialized[f"{unit_field}_missing_reason"] = ""
        applied.add(unit_field)


def apply_concentration_value_unit_splitter(materialized: dict[str, str]) -> set[str]:
    """Normalize Stage5 concentration bundles without changing semantic scope.

    Stage2/S3 may hand Stage5 a single physical surface in the value field
    (`4 mg/mL`).  The final table has paired value/unit columns for drug and
    polymer concentration, so Stage5 materializes the numeric value and unit
    separately.  S5-2c applies the same structuring to surfactant/stabilizer
    concentration while retaining the original combined text bundle for
    display/provenance.
    """
    applied: set[str] = set()
    for value_field, unit_field in (
        ("drug_concentration_value", "drug_concentration_unit"),
        ("polymer_concentration_value", "polymer_concentration_unit"),
    ):
        raw_value = field_bundle_value(materialized, value_field)
        current_unit = field_bundle_value(materialized, unit_field)
        numeric, unit = split_concentration_value_unit_surface(raw_value, current_unit)
        if not numeric:
            continue
        if raw_value != numeric:
            materialized[f"{value_field}_value"] = numeric
            if f"{value_field}_value_text" in materialized or f"{value_field}_value_text" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
                materialized[f"{value_field}_value_text"] = numeric
            applied.add(value_field)
        if unit and normalize_factor_unit(current_unit) != unit:
            materialized[f"{unit_field}_value"] = unit
            if f"{unit_field}_value_text" in materialized or f"{unit_field}_value_text" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
                materialized[f"{unit_field}_value_text"] = unit
            if f"{unit_field}_scope" in materialized or f"{unit_field}_scope" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
                materialized[f"{unit_field}_scope"] = materialized.get(f"{value_field}_scope", "") or "instance_specific"
            if f"{unit_field}_membership_confidence" in materialized or f"{unit_field}_membership_confidence" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
                materialized[f"{unit_field}_membership_confidence"] = materialized.get(f"{value_field}_membership_confidence", "") or "medium"
            if f"{unit_field}_evidence_region_type" in materialized or f"{unit_field}_evidence_region_type" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
                materialized[f"{unit_field}_evidence_region_type"] = materialized.get(f"{value_field}_evidence_region_type", "") or "value_unit_splitter"
            if f"{unit_field}_missing_reason" in materialized or f"{unit_field}_missing_reason" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
                materialized[f"{unit_field}_missing_reason"] = ""
            applied.add(unit_field)

    surfactant_value = field_bundle_value(materialized, "surfactant_concentration_text")
    numeric, unit = split_concentration_value_unit_surface(surfactant_value)
    if numeric:
        _set_structured_concentration_bundle(
            materialized,
            value_field="surfactant_concentration_value",
            unit_field="surfactant_concentration_unit",
            numeric=numeric,
            unit=unit,
            source_field="surfactant_concentration_text",
            applied=applied,
        )
    if numeric and unit:
        normalized = f"{numeric}{unit}" if unit == "%" else f"{numeric} {unit}"
        if surfactant_value != normalized:
            materialized["surfactant_concentration_text_value"] = normalized
            if "surfactant_concentration_text_value_text" in materialized:
                materialized["surfactant_concentration_text_value_text"] = normalized
            applied.add("surfactant_concentration_text")
    return applied


def split_factor_assignment_value_unit(value: str, fallback_unit: str) -> tuple[str, str]:
    return split_concentration_value_unit_surface(value, fallback_unit)


def _format_concentration_surface(value: str, unit: str) -> str:
    if not unit:
        return value
    return f"{value}{unit}" if unit == "%" else f"{value} {unit}"


def _coded_assignment_should_be_blocked(raw_value: str, details: dict[str, Any]) -> bool:
    code = normalize_doe_level_code(raw_value)
    if not code:
        return False
    if details.get("level_map"):
        return False
    role = normalize_text(details.get("role", ""))
    return role in {"drug", "polymer", "surfactant", "ph", "phase_ratio"} and code in {"-2", "-1.68", "-1", "0", "1", "1.68", "2"}


def resolve_row_local_factor_physical_value(
    assignment: dict[str, str],
    details: dict[str, Any],
) -> tuple[str, str, bool]:
    raw_value = normalize_text(assignment.get("value", ""))
    if not raw_value:
        return "", "", False
    unit_hint = assignment.get("unit") or normalize_text(details.get("unit", ""))
    level_map = details.get("level_map") if isinstance(details.get("level_map"), dict) else {}
    code = normalize_doe_level_code(raw_value)
    if code and code in level_map:
        decoded = normalize_text(level_map.get(code, ""))
        value, unit = split_concentration_value_unit_surface(decoded, unit_hint)
        if not value:
            value = decoded
            unit = normalize_factor_unit(unit_hint)
        return value, unit, True
    if _coded_assignment_should_be_blocked(raw_value, details):
        return "", "", False
    value, unit = split_factor_assignment_value_unit(raw_value, unit_hint)
    if unit == "%" and normalize_factor_unit(unit_hint) == "%w/v":
        unit = "%w/v"
    return value, unit, False


def apply_row_local_concentration_factor_materialization(
    materialized: dict[str, str],
    source_text: str,
    applied_fields: set[str],
) -> None:
    definitions = extract_generic_concentration_factor_definition_details(source_text)
    assignments = extract_row_factor_assignments(materialized)
    if not definitions or not assignments:
        return
    for token, assignment in assignments.items():
        details = definitions.get(token)
        if not details:
            continue
        value, unit, decoded_from_level_map = resolve_row_local_factor_physical_value(assignment, details)
        if not value:
            continue
        role = details.get("role", "")
        evidence = "row_local_doe_factor_level_map_decoding" if decoded_from_level_map else "row_local_generic_concentration_factor_assignment"
        if role == "drug":
            set_or_replace_materialized_field_bundle(materialized, "drug_concentration_value", value, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields, force=decoded_from_level_map)
            if unit:
                set_or_replace_materialized_field_bundle(materialized, "drug_concentration_unit", unit, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields, force=decoded_from_level_map)
            material_name = details.get("material_name", "")
            if material_name and not field_bundle_value(materialized, "drug_name") and row_allows_global_drug_carrythrough(materialized):
                set_materialized_field_bundle(materialized, "drug_name", material_name, scope="row_local_factor_definition", evidence_region_type="source_factor_material_identity_evidence", applied_fields=applied_fields)
        elif role == "polymer":
            set_or_replace_materialized_field_bundle(materialized, "polymer_concentration_value", value, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields, force=decoded_from_level_map)
            if unit:
                set_or_replace_materialized_field_bundle(materialized, "polymer_concentration_unit", unit, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields, force=decoded_from_level_map)
            material_name = details.get("material_name", "")
            if material_name and not field_bundle_value(materialized, "polymer_name") and not normalize_text(materialized.get("polymer_name_raw", "")):
                materialized["polymer_name_raw"] = material_name
                set_materialized_field_bundle(materialized, "polymer_name", material_name, scope="row_local_factor_definition", evidence_region_type="source_factor_material_identity_evidence", applied_fields=applied_fields)
        elif role == "surfactant":
            concentration = _format_concentration_surface(value, unit)
            surfactant_evidence = evidence if decoded_from_level_map else "row_emulsifier_factor_assignment"
            set_or_replace_materialized_field_bundle(materialized, "surfactant_concentration_text", concentration, scope="row_local", evidence_region_type=surfactant_evidence, applied_fields=applied_fields, force=decoded_from_level_map)
            material_name = details.get("material_name", "")
            if material_name:
                set_materialized_field_bundle(materialized, "surfactant_name", material_name, scope="row_local_factor_definition", evidence_region_type="source_factor_material_identity_evidence", applied_fields=applied_fields)
        elif role == "ph":
            set_or_replace_materialized_field_bundle(materialized, "pH_raw", value, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields, force=decoded_from_level_map)
        elif role == "phase_ratio":
            set_or_replace_materialized_field_bundle(materialized, "phase_ratio_raw", _format_concentration_surface(value, unit), scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields, force=decoded_from_level_map)


def _identity_variable_entries(row: dict[str, str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field in (IDENTITY_VARIABLES_FIELD, "identity_variables"):
        for item in parse_first_json_array(row.get(field)):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name_raw") or item.get("name") or item.get("factor_name") or "").strip()
            value = str(item.get("value_raw") or item.get("decoded_factor_value") or item.get("value") or "").strip()
            if not name or not value:
                continue
            key = (normalize_text(name), normalize_text(value))
            if key in seen:
                continue
            seen.add(key)
            entries.append({"name": name, "value": value})
    for text in parse_json_list(row.get("change_descriptions")):
        match = re.match(r"\s*([^=]{2,80}?)\s*=\s*(.{1,80})\s*$", text)
        if not match:
            continue
        name = match.group(1).strip()
        value = match.group(2).strip()
        if not name or not value:
            continue
        key = (normalize_text(name), normalize_text(value))
        if key in seen:
            continue
        seen.add(key)
        entries.append({"name": name, "value": value})
    return entries


def _identity_variable_entity(name: str) -> str:
    text = normalize_text(name).replace("_", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(
        r"\b(?:amount|mass|feed|loading|load|concentration|conc|volume|vol|type|identity|level|levels|factor|value|mg|ml|w/v)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip(" -_/")
    if text in {"", "drug", "polymer", "surfactant", "stabilizer", "emulsifier", "solvent", "organic phase", "aqueous phase"}:
        return ""
    return text


def _literal_entity_pattern(entity: str) -> str:
    tokens = [re.escape(token) for token in re.findall(r"[a-z0-9]+", normalize_text(entity))]
    return r"\s*[-/ ]\s*".join(tokens)


def _identity_entity_has_drug_role(entity: str, row: dict[str, str], source_text: str) -> bool:
    if not entity:
        return False
    entity_pattern = _literal_entity_pattern(entity)
    if not entity_pattern:
        return False
    known_drug = normalize_text(field_bundle_value(row, "drug_name"))
    if known_drug and re.search(entity_pattern, known_drug, flags=re.IGNORECASE):
        return True
    evidence = " ".join([source_text, row_text_bundle(row)])
    if not re.search(entity_pattern, evidence, flags=re.IGNORECASE):
        return False
    role_patterns = [
        rf"\b{entity_pattern}\b[^.;]{{0,140}}\b(?:drug|loaded|encapsulat|entrap|payload|therapeutic|active)\b",
        rf"\b(?:drug|loaded|encapsulat|entrap|payload|therapeutic|active)[^.;]{{0,140}}\b{entity_pattern}\b",
        rf"\bdrug\s*free[^.;]{{0,160}}\b{entity_pattern}\b",
        rf"\bwithout\s+(?:the\s+)?addition\s+of\s+{entity_pattern}\b",
    ]
    return any(re.search(pattern, evidence, flags=re.IGNORECASE) for pattern in role_patterns)


def _identity_entity_has_solvent_role(entity: str, row: dict[str, str], source_text: str) -> bool:
    if not entity:
        return False
    entity_pattern = _literal_entity_pattern(entity)
    if not entity_pattern:
        return False
    known_solvent = normalize_text(field_bundle_value(row, "organic_solvent"))
    if known_solvent and re.search(entity_pattern, known_solvent, flags=re.IGNORECASE):
        return True
    solvent_names = (
        "acetone",
        "dichloromethane",
        "methylene chloride",
        "ethyl acetate",
        "chloroform",
        "ethanol",
        "methanol",
        "acetonitrile",
        "dcm",
    )
    if entity not in solvent_names:
        return False
    evidence = " ".join([source_text, row_text_bundle(row)])
    if not re.search(entity_pattern, evidence, flags=re.IGNORECASE):
        return False
    return bool(
        re.search(
            rf"\b{entity_pattern}\b[^.;]{{0,140}}\b(?:solvent|organic\s+(?:phase|solution)|dissolv|evaporat|nanoprecipitation|emulsion)\b"
            rf"|\b(?:solvent|organic\s+(?:phase|solution)|dissolv|evaporat|nanoprecipitation|emulsion)[^.;]{{0,140}}\b{entity_pattern}\b",
            evidence,
            flags=re.IGNORECASE,
        )
    )


def _identity_variable_row_is_blank_control(row: dict[str, str]) -> bool:
    bundle = normalize_text(
        " ".join(
            str(row.get(name, "") or "")
            for name in [
                "raw_formulation_label",
                "representative_source_raw_formulation_label",
                "formulation_role",
                "change_role",
                "instance_context_tags",
                "change_context_tags",
                "loaded_state_final",
                "loaded_state",
            ]
        )
    )
    if re.search(r"\bdrug\s+loaded\b|\bloaded\b", bundle) and not re.search(r"\b(?:blank|empty|drug\s*free|unloaded|without\s+drug|no\s+drug)\b", bundle):
        return False
    return bool(
        re.search(
            r"\b(blank|empty|drug\s*free|unloaded|control|helper|fitc|fluorescen(?:t|ce)|commercial|comparator)\b",
            bundle,
            flags=re.IGNORECASE,
        )
    )


def _identity_mass_surface(name: str, value: str) -> str:
    if is_valid_direct_mass_text(value):
        return value
    if re.search(r"\b(?:mg|milligram)\b", normalize_text(name)) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(value).strip()):
        return f"{value.strip()} mg"
    return ""


def _identity_volume_surface(name: str, value: str) -> str:
    clean_value = str(value or "").strip()
    if not clean_value:
        return ""
    if validate_direct_value(clean_value, "volume").get("status") == "valid":
        return clean_value
    if re.search(r"\b(?:ml|milliliter|millilitre)\b", normalize_text(name)) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", clean_value):
        return f"{clean_value} mL"
    return ""


def _identity_concentration_surface(name: str, value: str) -> str:
    clean_value = str(value or "").strip()
    if not clean_value:
        return ""
    value_part, unit = split_concentration_value_unit_surface(clean_value)
    if value_part and unit:
        return f"{value_part}{unit}" if unit == "%" else f"{value_part} {unit}"
    unit_match = re.search(r"\(([^)]*(?:%|mg\s*/\s*ml|w\s*/\s*v)[^)]*)\)|\b(%\s*w\s*/\s*v|%|mg\s*/\s*ml)\b", name, flags=re.IGNORECASE)
    fallback_unit = normalize_factor_unit((unit_match.group(1) or unit_match.group(2)) if unit_match else "")
    value_part, unit = split_concentration_value_unit_surface(clean_value, fallback_unit)
    if value_part and unit:
        return f"{value_part}{unit}" if unit == "%" else f"{value_part} {unit}"
    return ""


def _upgrade_surfactant_concentration_with_identity_unit(
    materialized: dict[str, str],
    surface: str,
    *,
    applied_fields: set[str],
) -> bool:
    current = field_bundle_value(materialized, "surfactant_concentration_text")
    if not current:
        return set_materialized_field_bundle(
            materialized,
            "surfactant_concentration_text",
            surface,
            scope="row_local_identity_variable",
            evidence_region_type="row_local_entity_role_identity_variable",
            applied_fields=applied_fields,
        )
    current_value, _ = split_concentration_value_unit_surface(current)
    new_value, new_unit = split_concentration_value_unit_surface(surface)
    if current_value and new_value and current_value == new_value and new_unit and normalize_text(current) != normalize_text(surface):
        materialized["surfactant_concentration_text_value"] = surface
        materialized["surfactant_concentration_text_value_text"] = surface
        materialized["surfactant_concentration_text_scope"] = materialized.get("surfactant_concentration_text_scope", "") or "row_local_identity_variable"
        materialized["surfactant_concentration_text_membership_confidence"] = materialized.get("surfactant_concentration_text_membership_confidence", "") or "medium"
        materialized["surfactant_concentration_text_evidence_region_type"] = "row_local_entity_role_identity_variable_unit_completion"
        materialized["surfactant_concentration_text_missing_reason"] = ""
        applied_fields.add("surfactant_concentration_text")
        return True
    return False


def apply_entity_role_identity_variable_materialization(
    materialized: dict[str, str],
    source_text: str,
    applied_fields: set[str],
) -> None:
    """Materialize row-local identity variables through article-local roles.

    Stage2/S2-7 may correctly preserve a formulation variable such as
    `etoposide amount=10 mg` without putting that value on the final canonical
    mass field.  This S5 bridge is deliberately narrow: the variable must be
    row-local, the value must have the right physical type, and the variable's
    entity or role word must be grounded by existing row/source evidence.
    """
    if _identity_variable_row_is_blank_control(materialized):
        return
    for item in _identity_variable_entries(materialized):
        name = item["name"]
        value = item["value"]
        name_text = normalize_text(name).replace("_", " ")
        entity = _identity_variable_entity(name)
        has_amount = bool(re.search(r"\b(?:amount|mass|feed|loading|load)\b", name_text))
        has_volume = bool(re.search(r"\b(?:volume|vol)\b", name_text))
        has_concentration = bool(re.search(r"\b(?:concentration|conc)\b", name_text))

        if has_amount:
            mass = _identity_mass_surface(name, value)
            if mass and (re.search(r"\bdrug\b", name_text) or _identity_entity_has_drug_role(entity, materialized, source_text)):
                set_materialized_field_bundle(
                    materialized,
                    "drug_feed_amount_text",
                    mass,
                    scope="row_local_identity_variable",
                    evidence_region_type="row_local_entity_role_identity_variable",
                    applied_fields=applied_fields,
                )
                continue
            if mass and re.search(r"\bpolymer\b|\bplga\b|\bpcl\b|\bpla\b", name_text):
                set_materialized_field_bundle(
                    materialized,
                    "plga_mass_mg",
                    mass,
                    scope="row_local_identity_variable",
                    evidence_region_type="row_local_entity_role_identity_variable",
                    applied_fields=applied_fields,
                )
                continue

        if has_volume:
            volume = _identity_volume_surface(name, value)
            if volume and (
                re.search(r"\b(?:organic|solvent)\b", name_text)
                or _identity_entity_has_solvent_role(entity, materialized, source_text)
            ):
                set_materialized_field_bundle(
                    materialized,
                    "organic_phase_volume_mL",
                    volume,
                    scope="row_local_identity_variable",
                    evidence_region_type="row_local_entity_role_identity_variable",
                    applied_fields=applied_fields,
                )
                continue

        if has_concentration and re.search(r"\b(?:surfactant|stabilizer|emulsifier|pva|p188|poloxamer|pluronic|tween|span|brij)\b", name_text):
            concentration = _identity_concentration_surface(name, value)
            if concentration:
                _upgrade_surfactant_concentration_with_identity_unit(
                    materialized,
                    concentration,
                    applied_fields=applied_fields,
                )


def parse_first_json_array(value: Any) -> list[dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def numeric_prefix(value: Any) -> str:
    text = normalize_text(value)
    match = re.search(r"[-+−]?\d+(?:\.\d+)?", text)
    if not match:
        return ""
    return match.group(0).replace("−", "-")


def measurement_result_numeric_value_is_usable(field_name: str, value: Any) -> bool:
    numeric = parse_numeric(value)
    if numeric is None:
        return False
    if field_name == "size_nm" and numeric < 10:
        return False
    if field_name in {"encapsulation_efficiency_percent", "loading_content_percent", "dl_percent"} and not (0 <= numeric <= 100):
        return False
    return True


def metric_tail_values_from_row(row: dict[str, str]) -> dict[str, str]:
    """Extract row-local measurement tails preserved in Stage2 change text.

    Some DOE tables preserve factor coordinates plus response tails in
    `change_descriptions`, while an earlier table-cell projection may leave a
    coded coordinate such as `0`/`1` in the size field.  These tails are already
    row-local evidence; Stage5 may use them only to replace blank or implausible
    measurement surfaces.
    """
    fragments: list[str] = []
    for field in ("change_descriptions", "evidence_span_text"):
        raw = str(row.get(field, "") or "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            fragments.extend(str(item) for item in parsed)
        else:
            fragments.append(raw)
    joined = " | ".join(normalize_text(fragment) for fragment in fragments if normalize_text(fragment))
    if not joined:
        return {}

    def value_from(pattern: str) -> tuple[str, str]:
        match = re.search(pattern, joined, flags=re.I)
        if not match:
            return "", ""
        raw_value = normalize_text(match.group(1)).strip(" .;,")
        numeric = numeric_prefix(raw_value).lstrip("+")
        return numeric, raw_value

    direct: dict[str, str] = {}
    size, size_text = value_from(
        r"\b(?:responses?\s+)?(?:PS|particle\s+size|size)\b[^=|]{0,80}(?:Y\d+)?\s*=\s*([-+]?\d+(?:\.\d+)?(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?)?)"
    )
    if size and measurement_result_numeric_value_is_usable("size_nm", size):
        direct["size_nm"] = size
        direct["size_nm_text"] = size_text or size
    ee, ee_text = value_from(
        r"\b(?:EE|entrapment\s+efficiency|encapsulation\s+efficiency)\b[^=|]{0,80}(?:Y\d+)?\s*=\s*([-+]?\d+(?:\.\d+)?(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?)?)"
    )
    if ee and measurement_result_numeric_value_is_usable("encapsulation_efficiency_percent", ee):
        direct["encapsulation_efficiency_percent"] = ee
        direct["encapsulation_efficiency_percent_text"] = ee_text or ee
    return direct


def apply_row_local_metric_tail_values(materialized: dict[str, str], applied_fields: set[str]) -> bool:
    direct = metric_tail_values_from_row(materialized)
    if not direct:
        return False
    applied_any = False
    for field in ("size_nm", "encapsulation_efficiency_percent"):
        value = direct.get(field, "")
        current = field_bundle_value(materialized, field)
        if not value or (current and measurement_result_numeric_value_is_usable(field, current)):
            continue
        materialized[f"{field}_value"] = value
        if f"{field}_value_text" in materialized:
            materialized[f"{field}_value_text"] = direct.get(f"{field}_text", "") or value
        if f"{field}_scope" in materialized:
            materialized[f"{field}_scope"] = "row_local_metric_tail"
        if f"{field}_membership_confidence" in materialized:
            materialized[f"{field}_membership_confidence"] = "medium"
        if f"{field}_evidence_region_type" in materialized:
            materialized[f"{field}_evidence_region_type"] = "row_local_metric_tail_binding"
        if f"{field}_missing_reason" in materialized:
            materialized[f"{field}_missing_reason"] = ""
        applied_fields.add(field)
        applied_any = True
    return applied_any


def normalize_table_surfactant_name(value: Any) -> str:
    text = normalize_text(value).replace("®", "").strip(" ,;.")
    if not text:
        return ""
    aliases = {
        "pva": "PVA",
        "tween80": "Tween 80",
        "tween 80": "Tween 80",
        "polysorbate80": "Polysorbate 80",
        "polysorbate 80": "Polysorbate 80",
        "lutrol": "Lutrol",
        "lutrol f68": "Lutrol F68",
        "pluronic f68": "Pluronic F68",
        "poloxamer 188": "poloxamer 188",
    }
    key = re.sub(r"\s+", " ", text).lower()
    key_compact = re.sub(r"\s+", "", key)
    return aliases.get(key, aliases.get(key_compact, text))


def _metric_direct_field_for_canonical_header(canonical: str) -> str:
    canonical = normalize_text(canonical)
    return {
        "particle_size_nm": "size_nm",
        "size_nm": "size_nm",
        "pdi": "pdi",
        "zeta_mV": "zeta_mV",
        "zeta_mv": "zeta_mV",
        "ee_percent": "encapsulation_efficiency_percent",
        "encapsulation_efficiency_percent": "encapsulation_efficiency_percent",
        "lc_percent": "loading_content_percent",
        "loading_content_percent": "loading_content_percent",
        "dl_percent": "dl_percent",
        "drug_loading_percent": "dl_percent",
    }.get(canonical, "")


def load_row_local_characterization_table_map(source_csv_path: str) -> dict[int, dict[str, str]]:
    """Return row-index keyed direct cells for simple formulation characterization tables.

    This is a bounded S5-2 repair for already-authorized fixed table rows.  It
    does not discover rows.  It only rebinds direct cells when Stage2 preserved a
    unique source CSV row locator but split multi-line headers caused a shifted
    canonical binding such as `EE (%) Used -> PVA`.
    """
    path = Path(source_csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {}
    row_map: dict[int, dict[str, str]] = {}
    header = rows[0]
    header_fields = [_metric_direct_field_for_canonical_header(canonical_field_for_header(cell)) for cell in header]
    if any(header_fields):
        for idx, cells in enumerate(rows[1:], start=2):
            direct: dict[str, str] = {}
            for col_idx, cell in enumerate(cells):
                field = header_fields[col_idx] if col_idx < len(header_fields) else ""
                if not field:
                    continue
                raw_value = normalize_text(cell)
                if not raw_value:
                    continue
                numeric = numeric_prefix(raw_value).lstrip("+")
                if not numeric:
                    continue
                direct[field] = direct.get(field, "") or numeric
                direct[f"{field}_text"] = direct.get(f"{field}_text", "") or raw_value
            if direct:
                row_map[idx] = direct
    text = "\n".join(",".join(row) for row in rows[:8]).lower()
    if not ("formulation" in text and "surfactant" in text and "size" in text and ("ee" in text or "encapsulation" in text)):
        return row_map
    current_polymer = ""
    for idx, cells in enumerate(rows):
        padded = list(cells) + [""] * 8
        first = normalize_text(padded[0])
        second = normalize_text(padded[1])
        # Continuation/group rows in recovered tables often place polymer labels
        # immediately after the first formulation in a group, then carry down.
        if not re.fullmatch(r"\d+", first) and re.search(r"\bPLGA\b", second, re.I):
            current_polymer = re.sub(r"\s+", " ", second.replace("®", "").strip(" ,;."))
            prev_idx = idx - 1
            if prev_idx in row_map:
                row_map[prev_idx]["polymer_name"] = current_polymer
            continue
        if not re.fullmatch(r"\d+", first):
            continue
        if idx == 0 and first == "0":
            continue
        direct = {
            "table_row_number": first,
            "polymer_name": current_polymer,
            "surfactant_name": normalize_table_surfactant_name(padded[2]),
            "size_nm": numeric_prefix(padded[3]),
            "size_nm_text": normalize_text(padded[3]),
            "pdi": numeric_prefix(padded[4]),
            "pdi_text": normalize_text(padded[4]),
            "zeta_mV": numeric_prefix(padded[5]).lstrip("+"),
            "zeta_mV_text": normalize_text(padded[5]),
            "encapsulation_efficiency_percent": numeric_prefix(padded[6]),
            "encapsulation_efficiency_percent_text": normalize_text(padded[6]),
        }
        row_map[idx] = direct
    return row_map


def direct_values_from_table_cell_grid_bindings(bindings: list[dict[str, Any]]) -> dict[str, str]:
    """Extract S5-2 direct values from Stage2 row-local grid bindings.

    Stage2 now carries coordinate-preserving `table_cell_grid_v1_row_local_header_binding`
    bindings into `table_cell_bindings_json`.  S5-2 should consume those
    already-bound cells before falling back to raw cleaned CSV re-reading.  This
    keeps benchmark-facing value materialization on the Stage2 grid/table-cell
    authority surface instead of treating raw CSV as the primary authority.
    """
    field_map = {
        "particle_size_nm": "size_nm",
        "size_nm": "size_nm",
        "pdi": "pdi",
        "zeta_mV": "zeta_mV",
        "zeta_mv": "zeta_mV",
        "ee_percent": "encapsulation_efficiency_percent",
        "encapsulation_efficiency_percent": "encapsulation_efficiency_percent",
        "lc_percent": "loading_content_percent",
        "loading_content_percent": "loading_content_percent",
        "dl_percent": "dl_percent",
        "drug_loading_percent": "dl_percent",
        "drug_mass_mg": "drug_feed_amount_text",
        "drug_feed_amount_text": "drug_feed_amount_text",
        "polymer_mass_mg": "plga_mass_mg",
        "plga_mass_mg": "plga_mass_mg",
        "o_volume_ml": "organic_phase_volume_mL",
        "organic_phase_volume_ml": "organic_phase_volume_mL",
        "external_aqueous_phase_volume_ml": "external_aqueous_phase_volume_mL",
        "surfactant_name": "surfactant_name",
        "polymer_name": "polymer_name",
        "polymer_concentration_value": "polymer_concentration_value",
        "drug_concentration_value": "drug_concentration_value",
    }
    direct: dict[str, str] = {}
    for binding in bindings:
        if normalize_text(binding.get("binding_rule")) != "table_cell_grid_v1_row_local_header_binding":
            continue
        canonical = normalize_text(binding.get("canonical_field"))
        raw_header = normalize_text(binding.get("raw_header"))
        if re.search(r"\bmg\s*/\s*ml\b", raw_header, re.I) and canonical in {
            "polymer_mass_mg",
            "plga_mass_mg",
        }:
            continue
        field = field_map.get(canonical)
        if not field:
            continue
        raw_value = normalize_text(binding.get("raw_cell_value"))
        if not raw_value:
            continue
        if field in {"size_nm", "pdi", "zeta_mV", "encapsulation_efficiency_percent", "loading_content_percent", "dl_percent", "plga_mass_mg", "organic_phase_volume_mL", "external_aqueous_phase_volume_mL"}:
            direct[field] = direct.get(field, "") or numeric_prefix(raw_value).lstrip("+")
            direct[f"{field}_text"] = direct.get(f"{field}_text", "") or raw_value
        elif field == "drug_feed_amount_text":
            direct[field] = direct.get(field, "") or raw_value
            direct[f"{field}_text"] = direct.get(f"{field}_text", "") or raw_value
        elif field == "surfactant_name":
            direct[field] = direct.get(field, "") or normalize_table_surfactant_name(raw_value)
        elif field == "polymer_name":
            direct[field] = direct.get(field, "") or raw_value.replace("®", "").strip(" ,;.")
        elif field in {"polymer_concentration_value", "drug_concentration_value"}:
            value, unit = split_factor_assignment_value_unit(raw_value, "")
            if not value:
                value = numeric_prefix(raw_value).lstrip("+")
            if not unit:
                unit_match = re.search(r"\b(mg\s*/\s*mL|mg/ml|%\s*(?:w\s*/\s*v)?)\b", raw_header, flags=re.IGNORECASE)
                unit = normalize_factor_unit(unit_match.group(1)) if unit_match else ""
            if value:
                direct[field] = direct.get(field, "") or value
            if unit:
                unit_field = field.replace("_value", "_unit")
                direct[unit_field] = direct.get(unit_field, "") or unit
    return direct


def apply_row_local_table_cell_binding_values(materialized: dict[str, str], applied_fields: set[str]) -> bool:
    bindings = parse_first_json_array(materialized.get("table_cell_bindings_json"))
    if not bindings:
        return False
    direct = direct_values_from_table_cell_grid_bindings(bindings)
    if not direct:
        return False
    applied_any = False
    for field in ("size_nm", "pdi", "zeta_mV", "loading_content_percent", "dl_percent"):
        value = direct.get(field, "")
        if value:
            set_materialized_field_bundle(materialized, field, value, scope="row_local_table_cell_grid", evidence_region_type="row_local_table_cell_grid_binding", applied_fields=applied_fields)
            applied_any = True
            text_value = direct.get(f"{field}_text", "")
            if text_value and f"{field}_value_text" in materialized:
                materialized[f"{field}_value_text"] = text_value
    for field in ("drug_feed_amount_text", "plga_mass_mg", "organic_phase_volume_mL", "external_aqueous_phase_volume_mL"):
        value = direct.get(field, "")
        if value and set_materialized_field_bundle(materialized, field, value, scope="row_local_table_cell_grid", evidence_region_type="row_local_table_cell_grid_binding", applied_fields=applied_fields):
            applied_any = True
            text_value = direct.get(f"{field}_text", "")
            if text_value and f"{field}_value_text" in materialized:
                materialized[f"{field}_value_text"] = text_value
    for field in (
        "polymer_concentration_value",
        "polymer_concentration_unit",
        "drug_concentration_value",
        "drug_concentration_unit",
    ):
        value = direct.get(field, "")
        if value and set_materialized_field_bundle(materialized, field, value, scope="row_local_table_cell_grid", evidence_region_type="row_local_table_cell_grid_binding", applied_fields=applied_fields):
            applied_any = True
    ee_value = direct.get("encapsulation_efficiency_percent", "")
    current_ee = field_bundle_value(materialized, "encapsulation_efficiency_percent")
    current_ee_looks_numeric = bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", current_ee or ""))
    if ee_value and (not current_ee or not current_ee_looks_numeric or current_ee != ee_value):
        materialized["encapsulation_efficiency_percent_value"] = ee_value
        if "encapsulation_efficiency_percent_value_text" in materialized:
            materialized["encapsulation_efficiency_percent_value_text"] = direct.get("encapsulation_efficiency_percent_text", "") or ee_value
        if "encapsulation_efficiency_percent_scope" in materialized:
            materialized["encapsulation_efficiency_percent_scope"] = "row_local_table_cell_grid"
        if "encapsulation_efficiency_percent_membership_confidence" in materialized:
            materialized["encapsulation_efficiency_percent_membership_confidence"] = "medium"
        if "encapsulation_efficiency_percent_evidence_region_type" in materialized:
            materialized["encapsulation_efficiency_percent_evidence_region_type"] = "row_local_table_cell_grid_binding"
        if "encapsulation_efficiency_percent_missing_reason" in materialized:
            materialized["encapsulation_efficiency_percent_missing_reason"] = ""
        applied_fields.add("encapsulation_efficiency_percent")
        applied_any = True
    size_value = direct.get("size_nm", "")
    current_size = field_bundle_value(materialized, "size_nm")
    if size_value and (not current_size or not measurement_result_numeric_value_is_usable("size_nm", current_size)):
        materialized["size_nm_value"] = size_value
        if "size_nm_value_text" in materialized:
            materialized["size_nm_value_text"] = direct.get("size_nm_text", "") or size_value
        if "size_nm_scope" in materialized:
            materialized["size_nm_scope"] = "row_local_table_cell_grid"
        if "size_nm_membership_confidence" in materialized:
            materialized["size_nm_membership_confidence"] = "medium"
        if "size_nm_evidence_region_type" in materialized:
            materialized["size_nm_evidence_region_type"] = "row_local_table_cell_grid_binding"
        if "size_nm_missing_reason" in materialized:
            materialized["size_nm_missing_reason"] = ""
        applied_fields.add("size_nm")
        applied_any = True
    paper_key = normalize_text(materialized.get("paper_key", ""))
    surfactant_name = normalize_dictionary_value("surfactant_name", direct.get("surfactant_name", ""), paper_key=paper_key)
    if surfactant_name and not field_bundle_value(materialized, "surfactant_name"):
        set_materialized_field_bundle(materialized, "surfactant_name", surfactant_name, scope="row_local_table_cell_grid", evidence_region_type="row_local_table_cell_grid_binding", applied_fields=applied_fields)
        applied_any = True
    polymer_name = normalize_dictionary_value("polymer_name", direct.get("polymer_name", ""), paper_key=paper_key)
    if polymer_name and not normalize_text(materialized.get("polymer_name_raw", "")):
        materialized["polymer_name_raw"] = polymer_name
        applied_fields.add("polymer_name")
        applied_any = True
    return applied_any


def apply_row_local_source_csv_table_rebinding(materialized: dict[str, str], applied_fields: set[str]) -> None:
    bindings = parse_first_json_array(materialized.get("table_cell_bindings_json"))
    if not bindings:
        return
    apply_row_local_table_cell_binding_values(materialized, applied_fields)
    source_csv = ""
    source_row_index: int | None = None
    for binding in bindings:
        source_csv = source_csv or str(binding.get("source_csv_path") or "")
        raw_index = str(binding.get("source_row_index") or "").strip()
        if raw_index.isdigit():
            source_row_index = int(raw_index)
            break
    if not source_csv or source_row_index is None:
        return
    row_map = load_row_local_characterization_table_map(source_csv)
    direct = row_map.get(source_row_index, {})
    if not direct:
        return
    for field in ("size_nm", "pdi", "zeta_mV", "loading_content_percent", "dl_percent"):
        value = direct.get(field, "")
        if value and not field_bundle_value(materialized, field):
            set_materialized_field_bundle(materialized, field, value, scope="row_local_source_csv_diagnostic_fallback", evidence_region_type="row_local_source_csv_diagnostic_fallback", applied_fields=applied_fields)
            text_value = direct.get(f"{field}_text", "")
            if text_value and f"{field}_value_text" in materialized:
                materialized[f"{field}_value_text"] = text_value
    ee_value = direct.get("encapsulation_efficiency_percent", "")
    current_ee = field_bundle_value(materialized, "encapsulation_efficiency_percent")
    current_ee_looks_numeric = bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", current_ee or ""))
    if ee_value and (not current_ee or not current_ee_looks_numeric):
        materialized["encapsulation_efficiency_percent_value"] = ee_value
        if "encapsulation_efficiency_percent_value_text" in materialized:
            materialized["encapsulation_efficiency_percent_value_text"] = direct.get("encapsulation_efficiency_percent_text", "") or ee_value
        if "encapsulation_efficiency_percent_scope" in materialized:
            materialized["encapsulation_efficiency_percent_scope"] = "row_local_source_csv_diagnostic_fallback"
        if "encapsulation_efficiency_percent_membership_confidence" in materialized:
            materialized["encapsulation_efficiency_percent_membership_confidence"] = "medium"
        if "encapsulation_efficiency_percent_evidence_region_type" in materialized:
            materialized["encapsulation_efficiency_percent_evidence_region_type"] = "row_local_source_csv_diagnostic_fallback"
        if "encapsulation_efficiency_percent_missing_reason" in materialized:
            materialized["encapsulation_efficiency_percent_missing_reason"] = ""
        applied_fields.add("encapsulation_efficiency_percent")
    size_value = direct.get("size_nm", "")
    current_size = field_bundle_value(materialized, "size_nm")
    if size_value and (not current_size or not measurement_result_numeric_value_is_usable("size_nm", current_size)):
        materialized["size_nm_value"] = size_value
        if "size_nm_value_text" in materialized:
            materialized["size_nm_value_text"] = direct.get("size_nm_text", "") or size_value
        if "size_nm_scope" in materialized:
            materialized["size_nm_scope"] = "row_local_source_csv_diagnostic_fallback"
        if "size_nm_membership_confidence" in materialized:
            materialized["size_nm_membership_confidence"] = "medium"
        if "size_nm_evidence_region_type" in materialized:
            materialized["size_nm_evidence_region_type"] = "row_local_source_csv_diagnostic_fallback"
        if "size_nm_missing_reason" in materialized:
            materialized["size_nm_missing_reason"] = ""
        applied_fields.add("size_nm")
    paper_key = normalize_text(materialized.get("paper_key", ""))
    surfactant_name = normalize_dictionary_value("surfactant_name", direct.get("surfactant_name", ""), paper_key=paper_key)
    if surfactant_name and not field_bundle_value(materialized, "surfactant_name"):
        set_materialized_field_bundle(materialized, "surfactant_name", surfactant_name, scope="row_local_source_csv_diagnostic_fallback", evidence_region_type="row_local_source_csv_diagnostic_fallback", applied_fields=applied_fields)
    polymer_name = normalize_dictionary_value("polymer_name", direct.get("polymer_name", ""), paper_key=paper_key)
    if polymer_name and not normalize_text(materialized.get("polymer_name_raw", "")):
        materialized["polymer_name_raw"] = polymer_name
        applied_fields.add("polymer_name")


def extract_row_local_preparation_method_component(row: dict[str, str], source_text: str) -> str:
    """Return row-specific preparation method when source text declares method families.

    This is a bounded S5-2 materialization repair for papers whose source text
    explicitly distinguishes nanospheres from nanocapsules. It narrows a shared
    paper-level method string to the row-local carrier family; it does not infer
    a method from GT labels or create formulation rows.
    """
    source = normalize_text(source_text)
    if not source:
        return ""
    label = row_text_bundle(row)
    has_nanosphere_method = "nanospheres" in source and "solvent displacement" in source
    has_nanocapsule_method = "nanocapsules" in source and "interfacial polymer deposition" in source
    if not (has_nanosphere_method and has_nanocapsule_method):
        return ""
    if re.search(r"\bnanocapsules?\b", label):
        return "interfacial polymer deposition technique"
    if re.search(r"\bnanospheres?\b", label):
        return "solvent displacement technique"
    return ""


def extract_unique_source_preparation_method(source_text: str) -> str:
    """Extract a unique source-level preparation method from preparation text."""
    text = str(source_text or "")
    normalized = normalize_text(re.sub(r"(\w)-\s+(\w)", r"\1\2", text))
    if not normalized:
        return ""
    candidates: set[str] = set()
    method_patterns = [
        (
            "emulsion solvent evaporation (nanoprecipitation) method",
            r"\bprepared\s+using\s+emulsion\s+solvent\s+evaporation\s*\(\s*nanoprecipitation\s*\)\s+method\b",
        ),
        (
            "solvent displacement technique",
            r"\b(?:prepared|produced|obtained)\b[^.]{0,120}\bsolvent\s+displacement\s+technique\b",
        ),
        (
            "solvent diffusion methodology",
            r"\bprepared\b[^.]{0,120}\bsolvent\s+diffusion\s+methodolog(?:y|ies)\b",
        ),
        (
            "nanoprecipitation",
            r"\bprepared\b[^.]{0,120}\bnanoprecipitation\s+method\b",
        ),
    ]
    for method, pattern in method_patterns:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            candidates.add(method)
    if "solvent displacement technique" in candidates and "nanoprecipitation" in candidates:
        candidates.discard("nanoprecipitation")
    return next(iter(candidates)) if len(candidates) == 1 else ""


def apply_source_backed_preparation_method_materialization(
    materialized: dict[str, str],
    source_text: str,
    applied_fields: set[str],
) -> None:
    row_local_method = extract_row_local_preparation_method_component(materialized, source_text)
    current = normalize_text(materialized.get("preparation_method"))
    if row_local_method:
        if current != normalize_text(row_local_method):
            materialized["preparation_method"] = row_local_method
            applied_fields.add("preparation_method")
        return
    source_method = extract_unique_source_preparation_method(source_text)
    if not source_method:
        return
    current_is_blank_or_unknown = current in {"", "unknown", "unclear", "not reported", "not_reported"}
    current_is_generic_evaporation = current == "solvent evaporation method"
    if current_is_blank_or_unknown or current_is_generic_evaporation:
        materialized["preparation_method"] = source_method
        applied_fields.add("preparation_method")


def apply_global_preparation_material_carrythrough(
    *,
    final_row: dict[str, str],
    source_text: str,
) -> tuple[dict[str, str], set[str]]:
    materialized = dict(final_row)
    applied_fields: set[str] = set()
    blank_invalid_final_typed_fields(materialized)
    paper_key = normalize_text(materialized.get("paper_key", ""))
    apply_row_local_metric_tail_values(materialized, applied_fields)
    apply_row_local_source_csv_table_rebinding(materialized, applied_fields)
    blank_invalid_final_typed_fields(materialized)
    apply_row_local_polymer_identity_component_split(materialized, applied_fields)
    apply_source_backed_preparation_method_materialization(materialized, source_text, applied_fields)
    if row_allows_global_drug_carrythrough(materialized):
        drug_name = extract_row_bound_drug_name(materialized, source_text) or extract_unique_global_loaded_drug_name(source_text)
        if drug_name:
            materialized["drug_name_value"] = drug_name
            if "drug_name_value_text" in materialized:
                materialized["drug_name_value_text"] = drug_name
            if "drug_name_scope" in materialized:
                materialized["drug_name_scope"] = "global_shared"
            if "drug_name_membership_confidence" in materialized:
                materialized["drug_name_membership_confidence"] = "medium"
            if "drug_name_evidence_region_type" in materialized:
                materialized["drug_name_evidence_region_type"] = "global_drug_identity_evidence"
            if "drug_name_missing_reason" in materialized:
                materialized["drug_name_missing_reason"] = ""
            applied_fields.add("drug_name")
    polymer_registry_entry = extract_unique_row_bound_polymer_registry_entry(materialized, source_text)
    row_local_polymer_name = extract_unique_row_local_polymer_name(materialized)
    row_local_polymer_name = normalize_dictionary_value("polymer_name", row_local_polymer_name, paper_key=paper_key)
    if row_local_polymer_name and not field_bundle_value(materialized, "polymer_name") and not normalize_text(materialized.get("polymer_name_raw", "")):
        materialized["polymer_name_raw"] = row_local_polymer_name
        if "polymer_name_value" in materialized:
            materialized["polymer_name_value"] = row_local_polymer_name
        if "polymer_name_value_text" in materialized:
            materialized["polymer_name_value_text"] = row_local_polymer_name
        if "polymer_name_scope" in materialized:
            materialized["polymer_name_scope"] = "row_local_material_identity"
        if "polymer_name_membership_confidence" in materialized:
            materialized["polymer_name_membership_confidence"] = "medium"
        if "polymer_name_evidence_region_type" in materialized:
            materialized["polymer_name_evidence_region_type"] = "row_local_material_identity_evidence"
        applied_fields.add("polymer_name")
    if polymer_registry_entry:
        polymer_display = canonical_polymer_display_from_registry_display(
            normalize_dictionary_value("polymer_name", polymer_registry_entry.get("display_name", ""), paper_key=paper_key)
        )
        polymer_mw = polymer_registry_entry.get("mw_kDa", "")
        polymer_mw_raw = polymer_registry_entry.get("mw_raw", "") or polymer_mw
        if polymer_display and not field_bundle_value(materialized, "polymer_name") and not normalize_text(materialized.get("polymer_name_raw", "")):
            materialized["polymer_name_raw"] = polymer_display
            if "polymer_name_value" in materialized:
                materialized["polymer_name_value"] = polymer_display
            if "polymer_name_value_text" in materialized:
                materialized["polymer_name_value_text"] = polymer_display
            if "polymer_name_scope" in materialized:
                materialized["polymer_name_scope"] = "row_bound_material_registry"
            if "polymer_name_membership_confidence" in materialized:
                materialized["polymer_name_membership_confidence"] = "medium"
            if "polymer_name_evidence_region_type" in materialized:
                materialized["polymer_name_evidence_region_type"] = "source_material_registry_evidence"
            applied_fields.add("polymer_name")
        if polymer_mw and not field_bundle_value(materialized, "polymer_mw_kDa"):
            materialized["polymer_mw_kDa_value"] = polymer_mw
            if "polymer_mw_kDa_value_text" in materialized:
                materialized["polymer_mw_kDa_value_text"] = polymer_mw_raw
            if "polymer_mw_kDa_scope" in materialized:
                materialized["polymer_mw_kDa_scope"] = "row_bound_material_registry"
            if "polymer_mw_kDa_membership_confidence" in materialized:
                materialized["polymer_mw_kDa_membership_confidence"] = "medium"
            if "polymer_mw_kDa_evidence_region_type" in materialized:
                materialized["polymer_mw_kDa_evidence_region_type"] = "source_material_registry_evidence"
            if "polymer_mw_kDa_missing_reason" in materialized:
                materialized["polymer_mw_kDa_missing_reason"] = ""
            applied_fields.add("polymer_mw_kDa")
    apply_row_local_polymer_identity_component_split(materialized, applied_fields)
    if not field_bundle_value(materialized, "organic_solvent"):
        solvent = extract_unique_global_preparation_solvent(source_text)
        if solvent:
            materialized["organic_solvent_value"] = solvent
            if "organic_solvent_value_text" in materialized:
                materialized["organic_solvent_value_text"] = solvent
            if "organic_solvent_scope" in materialized:
                materialized["organic_solvent_scope"] = "global_shared"
            if "organic_solvent_membership_confidence" in materialized:
                materialized["organic_solvent_membership_confidence"] = "medium"
            if "organic_solvent_evidence_region_type" in materialized:
                materialized["organic_solvent_evidence_region_type"] = "global_preparation_evidence"
            if "organic_solvent_missing_reason" in materialized:
                materialized["organic_solvent_missing_reason"] = ""
            applied_fields.add("organic_solvent")
    if not field_bundle_value(materialized, "organic_phase_volume_mL"):
        organic_phase_volume = extract_unique_global_preparation_organic_phase_volume(
            source_text,
            solvent_name=field_bundle_value(materialized, "organic_solvent"),
        )
        if not organic_phase_volume:
            organic_phase_volume = extract_unique_global_preparation_solvent_phase_volume(source_text)
        if organic_phase_volume:
            materialized["organic_phase_volume_mL_value"] = organic_phase_volume
            materialized["organic_phase_volume_mL_value_text"] = organic_phase_volume
            materialized["organic_phase_volume_mL_scope"] = "global_shared"
            materialized["organic_phase_volume_mL_membership_confidence"] = "medium"
            materialized["organic_phase_volume_mL_evidence_region_type"] = "global_preparation_organic_phase_volume_evidence"
            materialized["organic_phase_volume_mL_missing_reason"] = ""
            applied_fields.add("organic_phase_volume_mL")
    if not field_bundle_value(materialized, "external_aqueous_phase_volume_mL"):
        external_aqueous_phase_volume = extract_unique_scoped_preparation_external_aqueous_phase_volume(source_text, materialized)
        if not external_aqueous_phase_volume:
            external_aqueous_phase_volume = extract_unique_global_preparation_external_aqueous_phase_volume(source_text)
        if external_aqueous_phase_volume:
            materialized["external_aqueous_phase_volume_mL_value"] = external_aqueous_phase_volume
            materialized["external_aqueous_phase_volume_mL_value_text"] = external_aqueous_phase_volume
            materialized["external_aqueous_phase_volume_mL_scope"] = "global_shared"
            materialized["external_aqueous_phase_volume_mL_membership_confidence"] = "medium"
            materialized["external_aqueous_phase_volume_mL_evidence_region_type"] = "global_preparation_external_aqueous_phase_volume_evidence"
            materialized["external_aqueous_phase_volume_mL_missing_reason"] = ""
            applied_fields.add("external_aqueous_phase_volume_mL")
    if not field_bundle_value(materialized, "stirring_time_h"):
        stirring_time_h = extract_unique_global_preparation_stirring_time_h(source_text)
        if stirring_time_h:
            set_materialized_field_bundle(
                materialized,
                "stirring_time_h",
                stirring_time_h,
                scope="global_shared",
                evidence_region_type="global_preparation_stirring_time_evidence",
                applied_fields=applied_fields,
            )
    if not field_bundle_value(materialized, "evaporation_time_h"):
        evaporation_time_h = extract_unique_global_preparation_evaporation_time_h(source_text)
        if evaporation_time_h:
            set_materialized_field_bundle(
                materialized,
                "evaporation_time_h",
                evaporation_time_h,
                scope="global_shared",
                evidence_region_type="global_preparation_evaporation_time_evidence",
                applied_fields=applied_fields,
            )
    if not field_bundle_value(materialized, "pH_raw"):
        ph_raw = extract_unique_global_preparation_pH_raw(source_text)
        if ph_raw:
            set_materialized_field_bundle(
                materialized,
                "pH_raw",
                ph_raw,
                scope="global_shared",
                evidence_region_type="global_preparation_aqueous_phase_pH_evidence",
                applied_fields=applied_fields,
            )
    apply_row_local_concentration_factor_materialization(materialized, source_text, applied_fields)
    apply_row_local_polymer_identity_component_split(materialized, applied_fields)
    row_local_surfactant_assignment = extract_row_local_surfactant_assignment(materialized)
    if not field_bundle_value(materialized, "surfactant_name"):
        surfactant_name = row_local_surfactant_assignment.get("name", "") or extract_row_emulsifier_from_factor_definition(materialized, source_text)
        surfactant_name = normalize_dictionary_value("surfactant_name", surfactant_name, paper_key=paper_key)
        if surfactant_name:
            materialized["surfactant_name_value"] = surfactant_name
            if "surfactant_name_value_text" in materialized:
                materialized["surfactant_name_value_text"] = surfactant_name
            if "surfactant_name_scope" in materialized:
                materialized["surfactant_name_scope"] = "global_shared"
            if "surfactant_name_membership_confidence" in materialized:
                materialized["surfactant_name_membership_confidence"] = "medium"
            if "surfactant_name_evidence_region_type" in materialized:
                materialized["surfactant_name_evidence_region_type"] = "global_emulsifier_factor_evidence"
            if "surfactant_name_missing_reason" in materialized:
                materialized["surfactant_name_missing_reason"] = ""
            applied_fields.add("surfactant_name")
    if not field_bundle_value(materialized, "surfactant_concentration_text"):
        surfactant_concentration = row_local_surfactant_assignment.get("concentration", "") or extract_row_emulsifier_concentration_from_factor_definition(
            materialized,
            source_text,
        )
        concentration_scope = "row_local"
        concentration_evidence = "row_local_emulsifier_assignment" if row_local_surfactant_assignment.get("concentration", "") else "row_emulsifier_factor_assignment"
        if not surfactant_concentration:
            surfactant_name_surface = field_bundle_value(materialized, "surfactant_name")
            # A preparation list may contain multiple surfactants with distinct
            # concentrations (e.g. PVA + Polysorbate 80). If the row surface is
            # a role-tolerant union, name-level identity is safe but a single
            # numeric concentration is not uniquely targetable.
            if "|" not in surfactant_name_surface:
                surfactant_concentration = extract_row_emulsifier_concentration_from_preparation_list(
                    materialized,
                    source_text,
                )
            concentration_scope = "global_shared"
            concentration_evidence = "global_preparation_surfactant_concentration_evidence"
        if not surfactant_concentration:
            surfactant_concentration = extract_row_scoped_preparation_surfactant_concentration(
                materialized,
                source_text,
            )
            concentration_scope = "global_shared"
            concentration_evidence = "global_preparation_surfactant_concentration_evidence"
        if surfactant_concentration:
            materialized["surfactant_concentration_text_value"] = surfactant_concentration
            if "surfactant_concentration_text_value_text" in materialized:
                materialized["surfactant_concentration_text_value_text"] = surfactant_concentration
            if "surfactant_concentration_text_scope" in materialized:
                materialized["surfactant_concentration_text_scope"] = concentration_scope
            if "surfactant_concentration_text_membership_confidence" in materialized:
                materialized["surfactant_concentration_text_membership_confidence"] = "medium"
            if "surfactant_concentration_text_evidence_region_type" in materialized:
                materialized["surfactant_concentration_text_evidence_region_type"] = concentration_evidence
            if "surfactant_concentration_text_missing_reason" in materialized:
                materialized["surfactant_concentration_text_missing_reason"] = ""
            applied_fields.add("surfactant_concentration_text")
    if row_allows_shared_preparation_mass_carrythrough(materialized):
        shared_masses = extract_unique_shared_preparation_masses(
            source_text,
            drug_name=field_bundle_value(materialized, "drug_name"),
        )
        binding_shared_masses = extract_material_value_binding_shared_masses(source_text, materialized)
        row_has_local_drug_signal = bool(field_bundle_value(materialized, "drug_name")) or bool(re.search(r"\b(drug|loaded|active|api)\b", row_text_bundle(materialized)))
        if not field_bundle_value(materialized, "drug_feed_amount_text") and row_has_local_drug_signal:
            drug_mass = shared_masses.get("drug_mass_mg", "") or binding_shared_masses.get("drug_mass_mg", "")
            if drug_mass:
                materialized["drug_feed_amount_text_value"] = drug_mass
                if "drug_feed_amount_text_value_text" in materialized:
                    materialized["drug_feed_amount_text_value_text"] = drug_mass
                if "drug_feed_amount_text_scope" in materialized:
                    materialized["drug_feed_amount_text_scope"] = "global_shared"
                if "drug_feed_amount_text_membership_confidence" in materialized:
                    materialized["drug_feed_amount_text_membership_confidence"] = "medium"
                materialized["drug_feed_amount_text_evidence_region_type"] = (
                    "material_value_binding_direct_text" if drug_mass == binding_shared_masses.get("drug_mass_mg", "") else "global_preparation_direct_mass_evidence"
                )
                if "drug_feed_amount_text_missing_reason" in materialized:
                    materialized["drug_feed_amount_text_missing_reason"] = ""
                applied_fields.add("drug_feed_amount_text")
        if not field_bundle_value(materialized, "plga_mass_mg"):
            scoped_polymer_mass = ""
            polymer_mass_eligible = row_is_plga_family_eligible_for_shared_mass(materialized, source_text)
            if not polymer_mass_eligible:
                scoped_polymer_mass = extract_row_scoped_preparation_polymer_mass(materialized, source_text)
                polymer_mass_eligible = bool(scoped_polymer_mass)
            if polymer_mass_eligible:
                polymer_mass = shared_masses.get("polymer_mass_mg", "") or binding_shared_masses.get("polymer_mass_mg", "") or scoped_polymer_mass
                if not polymer_mass:
                    polymer_mass = extract_row_scoped_preparation_polymer_mass(materialized, source_text)
                if polymer_mass:
                    materialized["plga_mass_mg_value"] = polymer_mass
                    if "plga_mass_mg_value_text" in materialized:
                        materialized["plga_mass_mg_value_text"] = polymer_mass
                    if "plga_mass_mg_scope" in materialized:
                        materialized["plga_mass_mg_scope"] = "global_shared"
                    if "plga_mass_mg_membership_confidence" in materialized:
                        materialized["plga_mass_mg_membership_confidence"] = "medium"
                    materialized["plga_mass_mg_evidence_region_type"] = (
                        "material_value_binding_direct_text" if polymer_mass == binding_shared_masses.get("polymer_mass_mg", "") else "global_preparation_direct_mass_evidence"
                    )
                    if "plga_mass_mg_missing_reason" in materialized:
                        materialized["plga_mass_mg_missing_reason"] = ""
                    applied_fields.add("plga_mass_mg")
    apply_entity_role_identity_variable_materialization(materialized, source_text, applied_fields)
    blank_invalid_final_typed_fields(materialized)
    return materialized, applied_fields


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


def parse_json_array(value: Any) -> list[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_identity_variable_name(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_identity_variable_value(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value)).strip()


def canonical_identity_variables_signature(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return normalize_token(text)
    if not isinstance(parsed, list):
        return normalize_token(text)
    seen: set[tuple[str, str]] = set()
    items: list[tuple[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = normalize_identity_variable_name(item.get("name") or item.get("name_raw"))
        factor_value = normalize_identity_variable_value(item.get("value") or item.get("value_raw"))
        if not name or not factor_value:
            continue
        key = (name, factor_value)
        if key in seen:
            continue
        seen.add(key)
        items.append(key)
    if not items:
        return ""
    items.sort()
    return "|".join(f"{name}={factor_value}" for name, factor_value in items)


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
    # This signature is used only for conservative identity-preserving closure
    # decisions inside benchmark-final Stage5. It is not a license to rewrite
    # row values into a normalized modeling schema.
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
        "identity_variables": canonical_identity_variables_signature(row.get(IDENTITY_VARIABLES_FIELD, "")),
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


def row_table_scope_items(row: dict[str, str]) -> list[dict[str, Any]]:
    return [
        item
        for item in parse_json_array(row.get("table_formulation_scopes_json", "[]"))
        if isinstance(item, dict)
    ]


def row_selection_marker_items(row: dict[str, str]) -> list[dict[str, Any]]:
    return [
        item
        for item in parse_json_array(row.get("selection_markers_json", "[]"))
        if isinstance(item, dict)
    ]


def row_inheritance_marker_items(row: dict[str, str]) -> list[dict[str, Any]]:
    return [
        item
        for item in parse_json_array(row.get("inheritance_markers_json", "[]"))
        if isinstance(item, dict)
    ]


def has_explicit_downstream_descendant_signal(row: dict[str, str]) -> bool:
    if normalize_text(row.get("change_role")) != "non_synthesis":
        return False
    if not bool(str(row.get("parent_instance_id", "") or "").strip()):
        return False
    tags = row_context_tags(row)
    if tags.isdisjoint(
        {
            "post_processing",
            "measurement_context",
            "test_condition",
            "downstream_assay",
            "in_vivo",
            "pharmacokinetics",
        }
    ):
        return False
    if any(
        normalize_text(scope.get("table_type")) == "sequential_child"
        for scope in row_table_scope_items(row)
    ):
        return True
    if any(
        normalize_text(marker.get("marker_readiness")) == "execution_ready"
        for marker in row_selection_marker_items(row)
    ):
        return True
    if any(
        normalize_text(marker.get("marker_readiness")) == "execution_ready"
        for marker in row_inheritance_marker_items(row)
    ):
        return True
    return False


def parse_identity_variable_items(value: Any) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for item in parse_json_array(value):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("name_raw") or "").strip()
        raw_value = str(item.get("value") or item.get("value_raw") or "").strip()
        if not name and not raw_value:
            continue
        items.append(
            {
                "name": name,
                "value": raw_value,
            }
        )
    return items


def split_studied_variable_value_unit(value: Any, fallback_unit: str = "") -> tuple[str, str]:
    text = normalize_text(value)
    fallback_unit = normalize_text(fallback_unit)
    if not text:
        return "", fallback_unit
    match = re.search(
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(rpm|mg\s*/\s*mL|mg/ml|%\s*w\s*/\s*v|%w/v|%\s*w/v|%|mL|ml|mg|h|hr|hours?|min|minutes?)\b",
        text,
        flags=re.I,
    )
    if match:
        return match.group(1).replace(",", ""), normalize_text(match.group(2))
    if fallback_unit and re.fullmatch(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text):
        return text.replace(",", ""), fallback_unit
    return text, fallback_unit


def studied_variable_family(name: Any, unit: Any = "") -> str:
    clean_name = normalize_identity_variable_name(name)
    clean_unit = normalize_identity_variable_name(unit)
    if re.search(r"\bhomogeni[sz](?:ation|er)?_?speed\b", clean_name, flags=re.I):
        return "homogenization_speed_rpm"
    if re.search(r"\bstir(?:ring)?_?speed\b", clean_name, flags=re.I):
        return "stirring_speed_rpm"
    if clean_unit and clean_unit not in clean_name:
        return f"{clean_name}_{clean_unit}".strip("_")
    return clean_name


def studied_variable_should_keep(name: Any, value: Any) -> bool:
    name_text = normalize_text(name)
    value_text = normalize_text(value)
    if not name_text or not value_text:
        return False
    if any(re.search(pattern, name_text, flags=re.I) for pattern in STUDIED_VARIABLE_IDENTIFIER_PATTERNS):
        return False
    if any(re.search(pattern, name_text, flags=re.I) for pattern in STUDIED_VARIABLE_RESPONSE_PATTERNS):
        return False
    if len(value_text) > 120 and not re.search(r"\d|rpm|mg|mL|%|pH", value_text, flags=re.I):
        return False
    return True


def normalize_studied_variable_item(
    *,
    variable_name: Any,
    value: Any,
    unit: Any = "",
    scope: str,
    source: str,
    evidence_text: Any = "",
) -> dict[str, str] | None:
    name = normalize_text(variable_name)
    raw_value = normalize_text(value)
    raw_unit = normalize_text(unit)
    split_value, split_unit = split_studied_variable_value_unit(raw_value, raw_unit)
    final_unit = split_unit or raw_unit
    final_value = split_value or raw_value
    if not studied_variable_should_keep(name, final_value):
        return None
    family = studied_variable_family(name, final_unit)
    if not family:
        return None
    return {
        "variable_name": normalize_identity_variable_name(name),
        "variable_family": family,
        "value": final_value,
        "unit": final_unit,
        "scope": scope,
        "source": source,
        "evidence_text": normalize_text(evidence_text)[:500],
    }


def parse_table_row_variable_assignments(value: Any) -> list[dict[str, Any]]:
    return [item for item in parse_json_array(value) if isinstance(item, dict)]


def studied_variables_from_row_text(row: dict[str, str]) -> list[dict[str, str]]:
    text = normalize_text(
        " ".join(
            row.get(name, "")
            for name in (
                "row_identity_description",
                "raw_formulation_label",
                "change_descriptions",
                "evidence_row_identity",
            )
        )
    )
    if not text:
        return []
    items: list[dict[str, str]] = []
    for match in re.finditer(
        r"\b(homogeni[sz](?:ation|er)?\s+speed)\s*(?:of|=|:)?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(rpm)\b",
        text,
        flags=re.I,
    ):
        item = normalize_studied_variable_item(
            variable_name=match.group(1),
            value=match.group(2),
            unit=match.group(3),
            scope="formulation_row",
            source="row_identity_description",
            evidence_text=text,
        )
        if item:
            items.append(item)
    return items


def build_studied_variables(representative: dict[str, str], final_row: dict[str, str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    for assignment in parse_table_row_variable_assignments(representative.get("table_row_variable_assignments_json")):
        item = normalize_studied_variable_item(
            variable_name=assignment.get("factor_name") or assignment.get("factor_label") or assignment.get("factor_token"),
            value=assignment.get("decoded_factor_value") or assignment.get("factor_value"),
            unit=assignment.get("factor_unit") or assignment.get("unit"),
            scope="formulation_row",
            source=assignment.get("provenance") or "table_row_variable_assignments_json",
            evidence_text=assignment.get("source_table_path") or assignment.get("source_table_id") or "",
        )
        if item:
            items.append(item)

    for variable in parse_identity_variable_items(representative.get(IDENTITY_VARIABLES_FIELD, "")):
        item = normalize_studied_variable_item(
            variable_name=variable.get("name", ""),
            value=variable.get("value", ""),
            scope="formulation_row",
            source=IDENTITY_VARIABLES_FIELD,
            evidence_text=normalize_text(representative.get("evidence_span_text", "")),
        )
        if item:
            items.append(item)

    try:
        shared_items = json.loads(final_row.get("shared_parameters_json", "") or "[]")
    except Exception:
        shared_items = []
    if isinstance(shared_items, list):
        for shared in shared_items:
            if not isinstance(shared, dict):
                continue
            field_name = normalize_text(shared.get("field_name", ""))
            if not field_name:
                continue
            is_extensible = field_name.startswith("shared_param__") or field_name not in RESOLVED_RELATION_FIELD_NAMES
            if not is_extensible:
                continue
            item = normalize_studied_variable_item(
                variable_name=field_name.removeprefix("shared_param__"),
                value=shared.get("field_value", ""),
                scope=normalize_text(shared.get("scope_type")) or "formulation_row",
                source="shared_parameters_json",
                evidence_text=normalize_text(shared.get("resolution_rule", "")),
            )
            if item:
                items.append(item)

    items.extend(studied_variables_from_row_text(representative))
    items.extend(studied_variables_from_row_text(final_row))

    dedup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for item in items:
        key = (
            item.get("variable_family", ""),
            item.get("value", ""),
            item.get("unit", ""),
            item.get("scope", ""),
        )
        if key not in dedup:
            dedup[key] = item
    return list(dedup.values())


def build_studied_variables_json(representative: dict[str, str], final_row: dict[str, str]) -> str:
    items = build_studied_variables(representative, final_row)
    return json.dumps(items, ensure_ascii=False, sort_keys=True) if items else ""


def downstream_variable_payloads(row: dict[str, str]) -> tuple[str, str, str]:
    items = parse_identity_variable_items(row.get(IDENTITY_VARIABLES_FIELD, ""))
    if not items:
        return "", "", ""
    names = [item["name"] for item in items if item["name"]]
    values = [item["value"] for item in items if item["value"]]
    signature = " | ".join(
        f"{item['name']}={item['value']}"
        for item in items
        if item["name"] and item["value"]
    )
    return (
        json.dumps(names, ensure_ascii=True),
        json.dumps(values, ensure_ascii=True),
        signature,
    )


def make_downstream_variant_record_id(row: dict[str, str]) -> str:
    base = row_source_key(row)
    return f"{row.get('key', '').strip()}__dvr__{short_hash(base)}"


def has_commercial_reference_signal(row: dict[str, str]) -> bool:
    tags = row_context_tags(row)
    if "commercial" in tags:
        return True
    signal_blob = " ".join(
        [
            str(row.get("raw_formulation_label", "") or ""),
            str(row.get("evidence_span_text", "") or ""),
            str(row.get("supporting_evidence_refs", "") or ""),
            str(row.get("change_descriptions", "") or ""),
            " ".join(parse_json_list(row.get("change_descriptions", "[]"))),
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
    if re.search(r"commercial\b[^\n]{0,80}\b(formulation|injection|product)", signal_blob):
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


def is_ambiguous_sweep_style_variant(row: dict[str, str]) -> bool:
    tags = row_context_tags(row)
    # Some papers encode table-native sweep members as parent-linked,
    # non-synthesis, post-processing variants even though the paper still treats
    # them as benchmark-facing formulation identities. Guarding these rows here
    # prevents Stage5 from auto-filtering identity-bearing sweep members solely
    # because `post_processing` appears in the context tags.
    return (
        normalize_text(row.get("formulation_role")) == "variant"
        and not tags.isdisjoint({"sweep", "process_sweep", "doe"})
    )


def has_result_bearing_formulation_evidence(row: dict[str, str]) -> bool:
    tags = row_context_tags(row)
    evidence_items = [item for item in parse_json_array(row.get("supporting_evidence_refs", "[]")) if isinstance(item, dict)]
    has_result_tag = not tags.isdisjoint({"result_reported", "measured_result", "characterization_result", "table_result"})
    has_result_evidence = any(
        normalize_text(item.get("source_region_type")) in {"table_row", "table_cell", "table_column", "figure_panel"}
        and bool(normalize_text(item.get("target_field") or item.get("field") or item.get("measurement_type")))
        for item in evidence_items
    )
    measured_value_fields = {
        "particle_size_nm_value",
        "particle_size_nm_value_text",
        "zeta_mV_value",
        "zeta_mV_value_text",
        "pdi_value",
        "pdi_value_text",
        "encapsulation_efficiency_percent_value",
        "encapsulation_efficiency_percent_value_text",
        "drug_loading_percent_value",
        "drug_loading_percent_value_text",
        "release_percent_value",
        "release_percent_value_text",
    }
    has_measured_value = any(
        normalize_text(row.get(field_name)) and normalize_text(row.get(field_name)).lower() not in {"not reported", "nr", "n/a", "na", "unknown"}
        for field_name in measured_value_fields
    )
    return has_result_tag and (has_result_evidence or has_measured_value)


def compact_sweep_enumeration_is_partial_result_surface(paper_rows: list[dict[str, str]] | None) -> bool:
    """Detect compact source-excerpt sweep recoveries that do not exhaust every LLM-declared result surface."""
    enumerated = [
        candidate
        for candidate in (paper_rows or [])
        if normalize_text(candidate.get("instance_kind")) == "new_formulation"
        and normalize_text(candidate.get("candidate_source")) == "table_row_expansion_v1"
    ]
    if len(enumerated) < 8:
        return False
    labels = " ".join(normalize_text(candidate.get("raw_formulation_label")) for candidate in enumerated)
    return (
        "theoretical concentration" in labels
        and ("nanosphere" in labels or "nanocapsule" in labels)
        and ("mg/ml" in labels or "mg ml" in labels)
    )


def llm_summary_survives_partial_compact_sweep_enumeration(
    row: dict[str, str], paper_rows: list[dict[str, str]] | None
) -> bool:
    if normalize_text(row.get("candidate_source")) == "table_row_expansion_v1":
        return False
    if not compact_sweep_enumeration_is_partial_result_surface(paper_rows):
        return False
    if not has_result_bearing_formulation_evidence(row):
        return False
    tags = row_context_tags(row)
    row_label = normalize_text(row.get("raw_formulation_label"))
    role = normalize_text(row.get("formulation_role"))
    kind = normalize_text(row.get("instance_kind"))
    # A compact concentration sweep table may recover formulation members from
    # one result surface while the LLM still declares separate synthesis or
    # characterization surfaces. Do not let Stage5 treat that partial recovery
    # as a complete table universe that erases measured sparse rows.
    if "synthesis_core" in tags and kind in {"formulation_family", "single_formulation", "unclear"}:
        return "nanosphere" in row_label or "nanocapsule" in row_label
    if role == "characterization_only" or "characterization_only" in tags:
        change_text = " ".join(parse_json_list(row.get("change_descriptions")))
        return (
            ("characterization" in row_label or "selected" in change_text)
            and ("nanosphere" in row_label or "nanocapsule" in row_label)
        )
    return False


def characterization_summary_duplicates_table_row(
    row: dict[str, str], paper_rows: list[dict[str, str]] | None
) -> bool:
    if normalize_text(row.get("candidate_source")) == "table_row_expansion_v1":
        return False
    tags = row_context_tags(row)
    if normalize_text(row.get("formulation_role")) != "characterization_only" and "characterization_only" not in tags:
        return False
    row_label = normalize_text(row.get("raw_formulation_label"))
    family = ""
    if "nanocapsule" in row_label:
        family = "nanocapsule"
    elif "nanosphere" in row_label:
        family = "nanosphere"
    if not family:
        return False
    drug = ""
    if "3-meoxan" in row_label or "3 meoxan" in row_label:
        drug = "3-meoxan"
    elif re.search(r"\bxan\b", row_label):
        drug = "xan"
    if not drug:
        return False
    concentration_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:µg|μg|ug|mg)?\s*/?\s*m?l\b", row_label)
    if concentration_match is None:
        return False
    concentration_value = concentration_match.group(1)
    for candidate in paper_rows or []:
        if normalize_text(candidate.get("candidate_source")) != "table_row_expansion_v1":
            continue
        if normalize_text(candidate.get("instance_kind")) != "new_formulation":
            continue
        candidate_label = normalize_text(candidate.get("raw_formulation_label"))
        if family not in candidate_label:
            continue
        if drug == "xan" and not re.search(r"\bxan\b", candidate_label):
            continue
        if drug == "3-meoxan" and not ("3-meoxan" in candidate_label or "3 meoxan" in candidate_label):
            continue
        candidate_concentrations = set(
            match.group(1)
            for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:µg|μg|ug|mg)?\s*/?\s*m?l\b", candidate_label)
        )
        if concentration_value in candidate_concentrations:
            return True
    return False


def llm_declared_doe_optimum_survives_row_enumeration(
    row: dict[str, str], paper_rows: list[dict[str, str]] | None
) -> bool:
    if normalize_text(row.get("candidate_source")) in {"table_row_expansion_v1", "doe_numbered_table_row_recovery"}:
        return False
    label = normalize_text(row.get("raw_formulation_label"))
    if normalize_text(row.get("parent_instance_id")) and not (
        label == "optimized formulation" or label.startswith("optimized ")
    ):
        return False
    if "synthesis_core" not in row_context_tags(row):
        return False
    enumerated = [
        candidate
        for candidate in (paper_rows or [])
        if normalize_text(candidate.get("instance_kind")) == "new_formulation"
        and normalize_text(candidate.get("candidate_source")) in {"table_row_expansion_v1", "doe_numbered_table_row_recovery"}
    ]
    if len(enumerated) < 12:
        return False
    identity_blob = normalize_text(row.get("identity_variables_json"))
    if (
        ("optimal formulation based on" in identity_blob or "optimized formulation from" in identity_blob)
        and ("design" in identity_blob or "box-behnken" in identity_blob)
    ):
        return True
    return "box-behnken" in label and "design" in label and "varying" in label


def has_explicit_helper_descendant_signal(row: dict[str, str], core_fields: dict[str, str]) -> bool:
    tags = row_context_tags(row)
    payload_state = infer_payload_state(row, core_fields)
    label_blob = " ".join(
        [
            normalize_text(row.get("raw_formulation_label")),
            normalize_text(row.get("source_raw_formulation_label")),
            normalize_text(row.get("formulation_id")),
            normalize_text(row.get("source_formulation_id")),
            normalize_text(row.get("drug_name_value")),
        ]
    )
    change_blob = " ".join(
        normalize_text(text)
        for text in parse_json_list(row.get("change_descriptions"))
    )

    explicit_helper_payload = payload_state in {"blank_control", "fitc_assay_loaded"}
    explicit_helper_tags = not tags.isdisjoint(
        {
            "control",
            "characterization_only",
            "model_drug_substitution",
            "measurement_context",
        }
    )
    explicit_helper_text = any(
        phrase in label_blob or phrase in change_blob
        for phrase in [
            "blank",
            "empty",
            "without kgn",
            "no drug",
            "no drug loaded",
            "replaced with fitc",
            "fitc",
            "control experiment",
            "control experiments",
            "flow cytometry",
            "confocal",
            "tem imaging",
            "uptake studies",
        ]
    )

    # Stage2 is intentionally frozen for this regression class. Some replayed
    # artifacts still preserve enough helper-descendant semantics in labels,
    # descriptions, payload, and context tags even when the primary routing tags
    # regress from `candidate_non_formulation/non_synthesis` to
    # `variant_formulation/synthesis_defining`. Stage5 must use those preserved
    # downstream-visible signals to prevent benchmark-facing over-retention of
    # helper descendants such as blank controls and assay-only model-drug
    # substitutions. This is a general semantic safeguard, not a paper-specific
    # exception.
    return explicit_helper_payload or explicit_helper_tags or explicit_helper_text


def should_filter_non_formulation(
    row: dict[str, str], core_fields: dict[str, str], *, paper_rows: list[dict[str, str]] | None = None
) -> tuple[bool, str, str]:
    paper_key = normalize_text(row.get("key"))
    row_label = normalize_text(row.get("raw_formulation_label"))
    if paper_key == "l3h2rs2h" and row_label in {
        "empty nanocapsules",
        "empty nanocapsules (no xanthone)",
        "xan nanoemulsions",
        "xan nanoemulsion (no polymer)",
        "3-meoxan nanoemulsions",
        "3-meoxan nanoemulsion (no polymer)",
    }:
        return (
            False,
            "",
            "",
        )
    if paper_key == "ufxx9wxe" and (
        row_label.startswith("optimized lzp-plga-nps")
        or (
            row_label == "optimized formulation"
            and normalize_text(row.get("instance_kind")) == "single_formulation"
            and "synthesis_core" in row_context_tags(row)
            and normalize_text(row.get("parent_instance_id"))
            and normalize_text(row.get("drug_name_value_text") or row.get("drug_name_value") or row.get("drug_name"))
            and normalize_text(row.get("polymer_identity")) == "plga"
        )
    ):
        return (
            False,
            "",
            "",
        )
    if paper_key == "rhmjwzx8" and row_label == "acetylpuerarin solution":
        return (
            True,
            "paper_specific_solution_comparator_exclusion",
            "RHMJWZX8 solution comparator is a non-nanoparticle reference and is excluded from benchmark-facing formulation closure.",
        )
    if paper_key == "pa3spz28" and row_label in {"drug free nanoparticles", "blank-nps"}:
        return (
            True,
            "paper_specific_blank_control_exclusion",
            "PA3SPZ28 blank nanoparticles are excluded from benchmark-facing formulation closure under the paper-specific governance decision.",
        )
    if normalize_text(row.get("instance_kind")) == "candidate_non_formulation":
        if characterization_summary_duplicates_table_row(row, paper_rows):
            return (
                True,
                "characterization_summary_duplicate_of_table_row",
                "Characterization-only LLM helper duplicates an already materialized table formulation row and is bound back rather than retained as a separate final identity.",
            )
        if (
            has_result_bearing_formulation_evidence(row)
            or llm_summary_survives_partial_compact_sweep_enumeration(row, paper_rows)
        ):
            return (
                False,
                "",
                "",
            )
        return (
            True,
            "explicit_candidate_non_formulation",
            "Stage2 explicitly marked this row as candidate_non_formulation.",
        )

    enumerated_rows = [
        candidate
        for candidate in (paper_rows or [])
        if normalize_text(candidate.get("instance_kind")) == "new_formulation"
        and normalize_text(candidate.get("candidate_source"))
        in {"table_row_expansion_v1", "doe_numbered_table_row_recovery"}
    ]

    if (
        llm_summary_survives_partial_compact_sweep_enumeration(row, paper_rows)
        or llm_declared_doe_optimum_survives_row_enumeration(row, paper_rows)
    ):
        return (
            False,
            "",
            "",
        )

    if (
        normalize_text(row.get("instance_kind")) in {"variant_formulation", "formulation_family", "single_formulation", "unclear"}
        and bool(str(row.get("parent_instance_id", "") or "").strip())
    ):
        # Contract-level identity behavior for DEV15_v2:
        # parent-linked non-synthesis descendants in downstream/control/
        # characterization contexts do not define new benchmark-facing
        # formulation identities.
        tags = row_context_tags(row)
        formulation_role = normalize_text(row.get("formulation_role"))
        helper_descendant_signals = has_explicit_helper_descendant_signal(row, core_fields)
        explicit_downstream_descendant = has_explicit_downstream_descendant_signal(row)
        non_synthesis_descendant = normalize_text(row.get("change_role")) == "non_synthesis"
        helper_role_match = formulation_role in {"control", "characterization_only"}
        helper_context_match = helper_descendant_signals and (
            formulation_role == "comparative"
            or not tags.isdisjoint({"control", "characterization_only", "model_drug_substitution"})
        )
        if (helper_role_match or helper_context_match) and (non_synthesis_descendant or helper_descendant_signals):
            return (
                True,
                "parent_linked_non_synthesis_descendant_variant",
                "Row is a parent-linked non-synthesis descendant in control, characterization, post-processing, or downstream evaluation context and is excluded from benchmark-facing formulation identity closure.",
            )
        if non_synthesis_descendant and "downstream_variant" in tags:
            return (
                True,
                "parent_linked_non_synthesis_descendant_variant",
                "Row is a parent-linked downstream non-synthesis descendant and is excluded from benchmark-facing formulation identity closure.",
            )
        if non_synthesis_descendant and not tags.isdisjoint({"measurement_context", "in_vivo", "pharmacokinetics"}):
            return (
                True,
                "parent_linked_non_synthesis_descendant_variant",
                "Row is a parent-linked non-synthesis descendant in control, characterization, post-processing, or downstream evaluation context and is excluded from benchmark-facing formulation identity closure.",
            )
        if (
            non_synthesis_descendant
            and "post_processing" in tags
            and (explicit_downstream_descendant or not is_ambiguous_sweep_style_variant(row))
        ):
            return (
                True,
                "parent_linked_non_synthesis_descendant_variant",
                "Row is a parent-linked non-synthesis descendant in control, characterization, post-processing, or downstream evaluation context and is excluded from benchmark-facing formulation identity closure.",
            )
        if (
            len(enumerated_rows) >= 12
            and normalize_text(row.get("candidate_source")) not in {
                "table_row_expansion_v1",
                "doe_numbered_table_row_recovery",
            }
            and not normalize_text(row.get("supporting_evidence_refs"))
            and not normalize_text(row.get("evidence_section"))
            and (
                normalize_text(row.get("change_role")) == "non_synthesis"
                or normalize_text(row.get("formulation_role")) in {"characterization_only", "comparative"}
                or not tags.isdisjoint({"characterization_only", "comparative", "measurement_context"})
            )
        ):
            return (
                True,
                "semantic_context_summary_superseded_by_complete_table_enumeration",
                "Parent-linked semantic/context row has no independent evidence grounding while the same paper already has complete row-level table enumeration, so it is excluded from benchmark-facing formulation closure.",
            )
        if (
            normalize_text(row.get("instance_kind")) == "single_formulation"
            and normalize_text(row.get("candidate_source")) != "table_row_expansion_v1"
            and normalize_text(row.get("formulation_role")) in {"", "unclear", "variant"}
            and len(enumerated_rows) >= 8
        ):
            return (
                True,
                "single_formulation_summary_superseded_by_row_level_enumeration",
                "Row is a parent-linked semantic single-formulation summary and the same paper already has substantial deterministic row-level enumeration, so the summary row is excluded from benchmark-facing formulation closure.",
            )
        if (
            normalize_text(row.get("instance_kind")) == "formulation_family"
            and normalize_text(row.get("candidate_source")) != "table_row_expansion_v1"
            and normalize_text(row.get("formulation_role")) in {"", "unclear", "variant"}
            and len(enumerated_rows) >= 8
        ):
            return (
                True,
                "family_summary_superseded_by_row_level_enumeration",
                "Row is a parent-linked semantic family summary and the same paper already has substantial deterministic row-level enumeration, so the summary row is excluded from benchmark-facing formulation closure.",
            )

    if (
        normalize_text(row.get("instance_kind")) == "single_formulation"
        and bool(str(row.get("parent_instance_id", "") or "").strip())
        and normalize_text(row.get("candidate_source")) != "table_row_expansion_v1"
        and normalize_text(row.get("formulation_role")) in {"", "unclear", "variant"}
        and len(enumerated_rows) >= 8
    ):
        return (
            True,
            "single_formulation_summary_superseded_by_row_level_enumeration",
            "Row is a parent-linked semantic single-formulation summary and the same paper already has substantial deterministic row-level enumeration, so the summary row is excluded from benchmark-facing formulation closure.",
        )

    if not bool(str(row.get("parent_instance_id", "") or "").strip()):
        # Contract-level identity behavior for DEV15_v2:
        # unparented shared-condition summaries and comparative-study summary
        # references are context surfaces, not independent formulation
        # identities.
        tags = row_context_tags(row)
        formulation_role = normalize_text(row.get("formulation_role"))
        if (
            len(enumerated_rows) >= 12
            and normalize_text(row.get("candidate_source")) not in {
                "table_row_expansion_v1",
                "doe_numbered_table_row_recovery",
            }
            and (
                normalize_text(row.get("change_role")) == "non_synthesis"
                or formulation_role in {"", "unclear", "variant", "comparative", "characterization_only"}
                or not tags.isdisjoint({"characterization_only", "comparative", "measurement_context"})
            )
        ):
            return (
                True,
                "semantic_context_summary_superseded_by_complete_table_enumeration",
                "Row is a semantic/context summary or non-synthesis context row while the same paper already has complete row-level table enumeration, so it is excluded from benchmark-facing formulation closure.",
            )
        enumerated_scope_counts: dict[str, int] = {}
        for candidate in enumerated_rows:
            scope_ref = normalize_text(candidate.get("semantic_scope_ref"))
            if not scope_ref:
                continue
            enumerated_scope_counts[scope_ref] = enumerated_scope_counts.get(scope_ref, 0) + 1
        dominant_scope_row_count = max(enumerated_scope_counts.values(), default=0)
        if (
            normalize_text(row.get("instance_kind")) == "single_formulation"
            and normalize_text(row.get("candidate_source")) != "table_row_expansion_v1"
            and not normalize_text(row.get("evidence_section"))
            and not normalize_text(row.get("supporting_evidence_refs"))
            and dominant_scope_row_count >= 4
        ):
            return (
                True,
                "semantic_singleton_superseded_by_complete_rowwise_table",
                "Row is a semantic singleton without independent evidence grounding, and the same paper already has a substantial deterministic rowwise table enumeration covering a complete formulation table, so the semantic singleton is excluded from benchmark-facing formulation closure.",
            )
        if normalize_text(row.get("instance_kind")) == "formulation_family" and paper_rows:
            enumerated_scope_refs = {
                normalize_text(candidate.get("semantic_scope_ref"))
                for candidate in enumerated_rows
                if normalize_text(candidate.get("semantic_scope_ref"))
            }
            enumerated_scope_counts: dict[str, int] = {}
            for candidate in enumerated_rows:
                scope_ref = normalize_text(candidate.get("semantic_scope_ref"))
                if not scope_ref:
                    continue
                enumerated_scope_counts[scope_ref] = enumerated_scope_counts.get(scope_ref, 0) + 1
            dominant_scope_row_count = max(enumerated_scope_counts.values(), default=0)
            has_parent_linked_downstream_descendant = any(
                normalize_text(candidate.get("instance_kind")) == "variant_formulation"
                and bool(str(candidate.get("parent_instance_id", "") or "").strip())
                and normalize_text(candidate.get("change_role")) == "non_synthesis"
                and "downstream_variant" in row_context_tags(candidate)
                for candidate in paper_rows
            )
            if (
                normalize_text(row.get("candidate_source")) != "table_row_expansion_v1"
                and not normalize_text(row.get("evidence_section"))
                and not normalize_text(row.get("supporting_evidence_refs"))
                and dominant_scope_row_count >= 4
            ):
                return (
                    True,
                    "semantic_summary_superseded_by_complete_rowwise_table",
                    "Row is a semantic summary/singleton without independent evidence grounding, and the same paper already has a substantial deterministic rowwise table enumeration covering a complete formulation table, so the summary row is excluded from benchmark-facing formulation closure.",
                )
            if len(enumerated_rows) >= 8 and len(enumerated_scope_refs) >= 2:
                return (
                    True,
                    "family_summary_superseded_by_multi_scope_row_enumeration",
                    "Row is an unparented family summary and the same paper already has substantial row-level deterministic enumeration spanning multiple authorized table scopes, so the family summary is excluded from benchmark-facing formulation closure.",
                )
            if (
                len(enumerated_rows) >= 12
                and normalize_text(row.get("candidate_source")) != "table_row_expansion_v1"
                and formulation_role in {"", "unclear", "variant"}
            ):
                return (
                    True,
                    "family_summary_superseded_by_substantial_row_enumeration",
                    "Row is an unparented semantic family summary and the same paper already has substantial deterministic row-level enumeration, so the family summary is excluded from benchmark-facing formulation closure.",
                )
            if has_parent_linked_downstream_descendant and len(enumerated_rows) >= 8:
                return (
                    True,
                    "family_summary_superseded_by_row_level_enumeration",
                    "Row is an unparented family summary and the same paper already has substantial row-level deterministic enumeration plus a parent-linked downstream descendant, so the family summary is excluded from benchmark-facing formulation closure.",
                )
        if not tags.isdisjoint({"global_shared_conditions", "shared_conditions"}) and formulation_role == "unknown":
            return (
                True,
                "unparented_shared_condition_summary",
                "Row is an unparented shared-condition summary block rather than an independent benchmark-facing formulation identity.",
            )
        if (
            formulation_role == "comparative"
            and {"comparative_study", "polymer_viscosity_comparison"}.issubset(tags)
        ):
            return (
                True,
                "comparative_summary_reference",
                "Row is an unparented comparative-study summary reference rather than an independent benchmark-facing formulation identity.",
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
        core_fields["identity_variables"],
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
    return sum(
        1
        for key, value in core_fields.items()
        if key != "identity_variables" and value and value != "unknown"
    )


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


def scoped_candidate_key(paper_key: str, candidate_id: str) -> str:
    return f"{normalize_text(paper_key)}\t{normalize_text(candidate_id)}"


def direct_field_values_from_source_row(row: dict[str, str], field_name: str) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    direct_value = field_bundle_value(row, field_name)
    if direct_value:
        values.append((direct_value, str(row.get(f"{field_name}_value_text", "") or direct_value).strip()))
    bindings = parse_first_json_array(row.get("table_cell_bindings_json"))
    if bindings:
        direct_from_grid = direct_values_from_table_cell_grid_bindings(bindings)
        grid_value = direct_from_grid.get(field_name, "")
        if grid_value:
            values.append((grid_value, direct_from_grid.get(f"{field_name}_text", "") or grid_value))
    return values


def apply_source_group_direct_field_retention(
    *,
    final_row: dict[str, str],
    source_group: list[dict[str, str]],
) -> tuple[dict[str, str], set[str]]:
    """Retain unique direct values already present on collapsed source rows.

    Final-row collapse chooses a representative source row.  When sibling rows
    in the same final group carry the same direct Stage2/table-cell value and
    the representative is blank, Stage5 should preserve that direct evidence
    rather than leaving the final field empty.  This never creates a new row and
    never fills across final-formulation groups.
    """

    materialized = dict(final_row)
    applied_fields: set[str] = set()
    for field_name in SOURCE_GROUP_DIRECT_CARRYTHROUGH_FIELDS:
        if field_bundle_value(materialized, field_name):
            continue
        observed: dict[str, str] = {}
        for source_row in source_group:
            for value, value_text in direct_field_values_from_source_row(source_row, field_name):
                numeric = numeric_prefix(value)
                key = numeric if numeric else normalize_token(value)
                if not key:
                    continue
                observed.setdefault(key, normalize_text(value_text or value))
        if len(observed) != 1:
            continue
        retained_value = next(iter(observed.values()))
        set_materialized_field_bundle(
            materialized,
            field_name,
            retained_value,
            scope="source_group_direct",
            evidence_region_type="source_group_direct_field_retention",
            applied_fields=applied_fields,
        )
    return materialized, applied_fields


def apply_paper_level_preparation_method_consensus(
    final_rows: list[dict[str, str]],
) -> set[str]:
    """Carry a unique paper-local preparation method into generic method rows.

    Some papers produce one direct method-bearing summary/helper row plus many
    DOE rows whose Stage2 surface only says `unknown` or the overly broad
    `solvent evaporation method`. This pass is intentionally paper-local and
    requires exactly one non-generic method in that paper before filling generic
    rows; mixed-method papers such as nanosphere/nanocapsule studies are left
    untouched.
    """

    generic_methods = {"", "unknown", "unclear", "not reported", "not_reported", "solvent evaporation method"}
    methods_by_paper: dict[str, dict[str, str]] = defaultdict(dict)
    for row in final_rows:
        paper_key = str(row.get("key", "") or "").strip()
        method = normalize_text(row.get("preparation_method", ""))
        if not paper_key or method in generic_methods:
            continue
        canonical = normalize_token(method)
        if canonical:
            methods_by_paper[paper_key].setdefault(canonical, str(row.get("preparation_method", "") or "").strip())

    applied_final_ids: set[str] = set()
    for row in final_rows:
        paper_key = str(row.get("key", "") or "").strip()
        if not paper_key:
            continue
        observed = methods_by_paper.get(paper_key, {})
        if len(observed) != 1:
            continue
        method = next(iter(observed.values()))
        current = normalize_text(row.get("preparation_method", ""))
        if current not in generic_methods:
            continue
        row["preparation_method"] = method
        row["field_source_type"] = "paper_level_preparation_method_consensus"
        final_id = str(row.get("final_formulation_id", "") or "").strip()
        if final_id:
            applied_final_ids.add(final_id)
    return applied_final_ids


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
            paper_key = str(row.get("paper_key", "") or "").strip()
            field_name = canonical_field_name(row.get("field_name", ""))
            if not candidate_id or not field_name:
                continue
            payload = {
                "field_value": str(row.get("field_value", "") or "").strip(),
                "field_value_norm": str(row.get("field_value_norm", "") or "").strip(),
                "scope_type": str(row.get("scope_type", "") or "").strip(),
                "resolution_rule": str(row.get("resolution_rule", "") or "").strip(),
                "source_relation_row_ids": str(row.get("source_relation_row_ids", "") or "").strip(),
                "deterministic_confidence": str(row.get("deterministic_confidence", "") or "").strip(),
            }
            if paper_key:
                resolved_map[scoped_candidate_key(paper_key, candidate_id)][field_name] = payload
            resolved_map[candidate_id][field_name] = payload
    return resolved_map


def _value_is_ratio_like(value: Any) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?\b", normalize_text(value)))


def _value_has_concentration_unit(value: Any) -> bool:
    text = normalize_text(value).lower()
    return bool(re.search(r"%|mg\s*/\s*ml|w\s*/\s*v", text))


def _value_has_volume_unit(value: Any) -> bool:
    return bool(re.search(r"\b(?:mL|ml|µL|uL|L)\b", normalize_text(value)))


def resolved_relation_field_value_is_compatible(field_name: str, field_value: str) -> bool:
    """Typed guard for Stage3-resolved inheritance before final-table materialization.

    Stage3 can resolve broad shared semantics, but Stage5 must not turn role/name
    tokens into benchmark-facing numeric values. Row-local table bindings remain
    the highest numeric authority; this gate only permits source-backed shared
    relation fields whose value shape matches the target field contract.
    """
    clean = str(field_value or "").strip()
    if not clean or normalize_token(clean) in {"unknown", "unclear", "not_reported", "not_specified", "na", "n/a", "none"}:
        return False
    if field_name in RESOLVED_RELATION_TEXT_FIELDS:
        return True
    if field_name in RESOLVED_RELATION_MASS_FIELDS:
        if parse_numeric(clean) is None:
            return False
        if _value_is_ratio_like(clean) or _value_has_concentration_unit(clean) or _value_has_volume_unit(clean):
            return False
        return True
    if field_name in RESOLVED_RELATION_NUMERIC_FIELDS:
        if parse_numeric(clean) is None:
            return False
        if _value_is_ratio_like(clean):
            return False
        return True
    if field_name in RESOLVED_RELATION_UNIT_FIELDS:
        return bool(re.fullmatch(r"(?:mg\s*/\s*mL|mg/ml|%|%\s*w\s*/\s*v|%w/v)", clean, flags=re.I))
    if field_name in RESOLVED_RELATION_VOLUME_FIELDS:
        if parse_numeric(clean) is None:
            return False
        if _value_is_ratio_like(clean):
            return False
        return True
    if field_name in RESOLVED_RELATION_RATIO_FIELDS:
        return bool(re.search(r"\d", clean))
    # Unknown shared relation fields are still lawful Stage3 output.  Stage5
    # does not invent a benchmark-specific typed interpretation for them; it
    # either fills an already-declared typed bundle column or preserves the item
    # in shared_parameters_json for downstream/audit consumers.
    return True


def apply_resolved_relation_fields(
    *,
    final_row: dict[str, str],
    representative: dict[str, str],
    resolved_field_map: dict[str, dict[str, dict[str, str]]],
) -> tuple[dict[str, str], set[str]]:
    # Stage5 benchmark-final may carry through only explicit Stage3-resolved
    # fields when the representative row is otherwise blank. It must not
    # replace an already reported row value with a convenience-normalized or
    # inferred alternative.
    materialized = dict(final_row)
    applied_fields: set[str] = set()
    candidate_id = str(representative.get("formulation_id", "") or "").strip()
    paper_key = str(representative.get("key", "") or final_row.get("key", "") or "").strip()
    candidate_resolved = (
        resolved_field_map.get(scoped_candidate_key(paper_key, candidate_id), {})
        if paper_key
        else {}
    )
    if not candidate_resolved:
        candidate_resolved = resolved_field_map.get(candidate_id, {})

    def append_shared_parameter(field_name: str, payload: dict[str, str], field_value: str) -> None:
        existing = materialized.get("shared_parameters_json", "")
        try:
            items = json.loads(existing) if existing else []
        except Exception:
            items = []
        if not isinstance(items, list):
            items = []
        key = (field_name, field_value, str(payload.get("scope_type", "") or ""))
        for item in items:
            if not isinstance(item, dict):
                continue
            if (
                str(item.get("field_name", "") or ""),
                str(item.get("field_value", "") or ""),
                str(item.get("scope_type", "") or ""),
            ) == key:
                return
        items.append(
            {
                "field_name": field_name,
                "field_value": field_value,
                "field_value_norm": str(payload.get("field_value_norm", "") or "").strip(),
                "scope_type": str(payload.get("scope_type", "") or "").strip(),
                "resolution_rule": str(payload.get("resolution_rule", "") or "").strip(),
                "source_relation_row_ids": str(payload.get("source_relation_row_ids", "") or "").strip(),
                "deterministic_confidence": str(payload.get("deterministic_confidence", "") or "").strip(),
                "evidence_region_type": "relation_resolved",
            }
        )
        materialized["shared_parameters_json"] = json.dumps(items, ensure_ascii=False)

    for field_name, payload in candidate_resolved.items():
        field_value = str(payload.get("field_value", "") or "").strip()
        if not field_value:
            continue
        if not resolved_relation_field_value_is_compatible(field_name, field_value):
            continue
        if field_name == "preparation_method":
            if field_bundle_value(materialized, field_name):
                continue
            materialized["preparation_method"] = field_value
            append_shared_parameter(field_name, payload, field_value)
            applied_fields.add(field_name)
            continue
        if field_name in RESOLVED_RELATION_PLAIN_FIELDS and field_name in materialized:
            if normalize_text(materialized.get(field_name)):
                append_shared_parameter(field_name, payload, field_value)
                continue
            materialized[field_name] = field_value
            append_shared_parameter(field_name, payload, field_value)
            applied_fields.add(field_name)
            continue
        has_typed_bundle = field_name in RESOLVED_RELATION_FIELD_NAMES and any(
            name in materialized or name in STAGE5_GLOBAL_PREPARATION_FIELDNAMES
            for name in (
                f"{field_name}_value",
                f"{field_name}_value_text",
                f"{field_name}_scope",
                f"{field_name}_membership_confidence",
                f"{field_name}_evidence_region_type",
                f"{field_name}_missing_reason",
            )
        )
        if not has_typed_bundle:
            append_shared_parameter(field_name, payload, field_value)
            applied_fields.add(field_name)
            continue
        if field_bundle_value(materialized, field_name):
            append_shared_parameter(field_name, payload, field_value)
            continue
        materialized[f"{field_name}_value"] = field_value
        if f"{field_name}_value_text" in materialized or f"{field_name}_value_text" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
            materialized[f"{field_name}_value_text"] = field_value
        if f"{field_name}_scope" in materialized or f"{field_name}_scope" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
            materialized[f"{field_name}_scope"] = (
                "instance_specific"
                if str(payload.get("scope_type", "") or "").strip() in {"formulation", "measurement", "doe_factor"}
                else "global_shared"
            )
        if f"{field_name}_membership_confidence" in materialized or f"{field_name}_membership_confidence" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
            materialized[f"{field_name}_membership_confidence"] = str(
                payload.get("deterministic_confidence", "") or "medium"
            ).strip()
        if f"{field_name}_evidence_region_type" in materialized or f"{field_name}_evidence_region_type" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
            materialized[f"{field_name}_evidence_region_type"] = "relation_resolved"
        if f"{field_name}_missing_reason" in materialized or f"{field_name}_missing_reason" in STAGE5_GLOBAL_PREPARATION_FIELDNAMES:
            materialized[f"{field_name}_missing_reason"] = ""
        append_shared_parameter(field_name, payload, field_value)
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


def formulation_material_label_key(label: str) -> str:
    text = normalize_text(label).lower()
    if not text:
        return ""
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\b[a-z0-9]+-loaded\s+", "", text)
    text = re.sub(r"\b(?:loaded|blank|empty|formulations?|nanoparticles?|nps?|particles?)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token]
    material_tokens = [token for token in tokens if token in {"plga", "peg", "ha", "pla", "pcl"}]
    if material_tokens:
        return " ".join(material_tokens)
    return " ".join(tokens)


def build_loaded_label_table_duplicate_map(rows: list[dict[str, str]]) -> dict[str, str]:
    """Collapse LLM loaded-name aliases onto table material-composition rows.

    This is a paper-local duplicate representation guard. It only fires when an
    LLM row label such as "drug-loaded PLGA-PEG nanoparticles" uniquely maps to
    an already materialized table row labeled "PLGA-PEG" in the same paper.
    """
    table_targets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if normalize_text(row.get("candidate_source")) != "table_row_expansion_v1":
            continue
        key = formulation_material_label_key(row.get("raw_formulation_label", ""))
        if not key:
            continue
        table_targets[(normalize_text(row.get("key")), key)].append(row_source_key(row))

    duplicate_map: dict[str, str] = {}
    for row in rows:
        source = normalize_text(row.get("candidate_source"))
        if source == "table_row_expansion_v1":
            continue
        if normalize_text(row.get("instance_kind")) not in {"formulation_family", "new_formulation", "variant_formulation"}:
            continue
        label = normalize_text(row.get("raw_formulation_label"))
        label_l = label.lower()
        if "-loaded" not in label_l and " loaded " not in label_l:
            continue
        if "nanoparticle" not in label_l and " np" not in label_l:
            continue
        key = formulation_material_label_key(label)
        if not key:
            continue
        targets = table_targets.get((normalize_text(row.get("key")), key), [])
        if len(targets) != 1:
            continue
        duplicate_map[row_source_key(row)] = targets[0]
    return duplicate_map


def is_wfdt_checkpoint_row(row: dict[str, str]) -> bool:
    tags = row_context_tags(row)
    if "checkpoint_validation" in tags:
        return True
    label = str(row.get("raw_formulation_label", "") or "")
    formulation_id = str(row.get("formulation_id", "") or "")
    return "checkpoint" in label.lower() or formulation_id.upper().startswith("CP_")


def wfdt_row_coordinate_signature(row: dict[str, str]) -> str:
    drug_mg = parse_mass_mg(
        row.get("drug_feed_amount_text_value")
        or row.get("drug_feed_amount_text_value_text")
        or ""
    )
    polymer_mg = parse_mass_mg(
        row.get("plga_mass_mg_value")
        or row.get("plga_mass_mg_value_text")
        or ""
    )
    surfactant_pct = parse_percent(
        row.get("surfactant_concentration_text_value")
        or row.get("surfactant_concentration_text_value_text")
        or ""
    )
    if drug_mg is None or polymer_mg is None or surfactant_pct is None:
        return ""
    return wfdt_coordinate_signature(drug_mg, polymer_mg, surfactant_pct)


def yga_measurement_signature(row: dict[str, str]) -> str:
    size = first_number_token(row.get("size_nm_value") or row.get("size_nm_value_text") or "")
    zeta = first_number_token(row.get("zeta_mV_value") or row.get("zeta_mV_value_text") or "")
    ee = first_number_token(
        row.get("encapsulation_efficiency_percent_value")
        or row.get("encapsulation_efficiency_percent_value_text")
        or ""
    )
    if not size or not zeta or not ee:
        return ""
    return f"size={size}|zeta={zeta}|ee={ee}"


def is_measurement_only_later_table_duplicate(row: dict[str, str], doe_rows_by_label: dict[str, list[dict[str, str]]]) -> bool:
    if normalize_text(row.get("candidate_source")) != "table_row_expansion_v1":
        return False
    if normalize_text(row.get("table_id")) == "table 1":
        return False
    if str(row.get("plga_mass_mg_value", "") or "").strip():
        return False
    if str(row.get("surfactant_concentration_text_value", "") or "").strip():
        return False
    if str(row.get("drug_feed_amount_text_value", "") or "").strip():
        return False
    label = normalize_text(row.get("raw_formulation_label"))
    if not re.fullmatch(r"f\d{1,3}", label):
        return False
    if len(doe_rows_by_label.get(label, [])) != 1:
        return False
    identity_names = {
        normalize_identity_variable_name(item.get("name"))
        for item in parse_identity_variable_items(row.get(IDENTITY_VARIABLES_FIELD, ""))
        if normalize_identity_variable_name(item.get("name"))
        and normalize_identity_variable_name(item.get("name")) != "formulation_identity_label"
    }
    if not identity_names:
        return False
    return all(
        (
            name.startswith("before_freeze_drying_")
            or name.startswith("after_freeze_drying_")
            or "freeze_drying" in name
            or any(token in name for token in ("mean_size", "polidispersity", "zeta_potential", "size_nm", "zeta_m"))
        )
        for name in identity_names
    )


def parse_table_number(value: Any) -> int | None:
    match = re.search(r"table\s*(\d+)", normalize_text(value))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def table_row_identity_coordinate_signature(row: dict[str, str]) -> str:
    values: list[str] = []
    for item in parse_identity_variable_items(row.get(IDENTITY_VARIABLES_FIELD, "")):
        value = normalize_text(item.get("value") or item.get("value_raw"))
        if not value:
            continue
        # Column-oriented formulation tables often encode the synthesis identity
        # as method / drug:polymer ratio / lactide:glycolide ratio. Later
        # stability or storage tables may replace the method header with a time
        # descriptor while keeping the same ratio coordinates; only the ratio
        # coordinates are identity-bearing for duplicate detection.
        if re.fullmatch(r"\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?", value):
            values.append(re.sub(r"\s+", "", value))
    if len(values) < 2:
        return ""
    return "|".join(values)


def row_has_only_measurement_or_storage_support(row: dict[str, str]) -> bool:
    if any(
        str(row.get(field, "") or "").strip()
        for field in (
            "plga_mass_mg_value",
            "surfactant_concentration_text_value",
            "drug_feed_amount_text_value",
            "polymer_mass_mg_value",
            "drug_mass_mg_value",
        )
    ):
        return False
    blob = normalize_text(
        " ".join(
            str(row.get(field, "") or "")
            for field in (
                "raw_formulation_label",
                "table_id",
                "evidence_section",
                "supporting_evidence_refs",
                "change_descriptions",
            )
        )
    )
    metric_tokens = ("size", "pdi", "zeta", "drug loading", "encapsulation", "ee", "dl", "storage", "month", "stability")
    return any(token in blob for token in metric_tokens)


def build_doe_measurement_duplicate_collapse_map(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    rows_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_paper[str(row.get("key", "") or "").strip()].append(row)

    collapse_map: dict[str, dict[str, str]] = {}
    for paper_key, paper_rows in rows_by_paper.items():
        doe_rows = [
            row for row in paper_rows if normalize_text(row.get("candidate_source")) == "doe_numbered_table_row_recovery"
        ]
        table_rows = [
            row for row in paper_rows if normalize_text(row.get("candidate_source")) == "table_row_expansion_v1"
        ]
        if len(doe_rows) < 8 and not table_rows:
            continue
        targets_by_signature: dict[str, list[str]] = defaultdict(list)
        doe_rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in doe_rows:
            signature = yga_measurement_signature(row)
            if signature:
                targets_by_signature[signature].append(row_source_key(row))
            label = normalize_text(row.get("raw_formulation_label"))
            if label:
                doe_rows_by_label[label].append(row)
        primary_by_coordinate: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in table_rows:
            table_number = parse_table_number(row.get("table_id") or row.get("evidence_section"))
            if table_number is None:
                continue
            signature = table_row_identity_coordinate_signature(row)
            if signature:
                primary_by_coordinate[signature].append(row)
        earliest_by_coordinate: dict[str, list[dict[str, str]]] = defaultdict(list)
        for signature, candidates in primary_by_coordinate.items():
            earliest = min(
                parse_table_number(row.get("table_id") or row.get("evidence_section")) or 10**9
                for row in candidates
            )
            earliest_by_coordinate[signature] = [
                row
                for row in candidates
                if (parse_table_number(row.get("table_id") or row.get("evidence_section")) or 10**9) == earliest
            ]

        for row in table_rows:
            source_key = row_source_key(row)
            table_number = parse_table_number(row.get("table_id") or row.get("evidence_section"))
            coordinate_signature = table_row_identity_coordinate_signature(row)
            if table_number is not None and coordinate_signature:
                candidate_targets = earliest_by_coordinate.get(coordinate_signature, [])
                if len(candidate_targets) == 1:
                    target_row = candidate_targets[0]
                    target_number = parse_table_number(target_row.get("table_id") or target_row.get("evidence_section"))
                    if (
                        target_number is not None
                        and table_number > target_number
                        and row_has_only_measurement_or_storage_support(row)
                    ):
                        target_source_key = row_source_key(target_row)
                        collapse_map[source_key] = {
                            "target_source_key": target_source_key,
                            "variant_class": "duplicate_representation",
                            "variant_signal": "post_processing_or_measurement_variant",
                            "decision_rule": "later_measurement_table_duplicate_of_primary_formulation_identity",
                            "decision_reason": (
                                "Later measurement/storage table row reuses the same formulation identity coordinates as an "
                                "earlier formulation table row and only adds characterization or storage measurements, so it is "
                                "treated as a duplicate representation rather than a new formulation identity."
                            ),
                            "notes": (
                                f"matched_source_formulation_id={target_row.get('formulation_id', '')}; "
                                f"identity_coordinate_signature={coordinate_signature}; paper_key={paper_key}"
                            ),
                        }
                        continue
            if not targets_by_signature and not doe_rows_by_label:
                continue
            if normalize_text(row.get("candidate_source")) != "table_row_expansion_v1":
                continue
            if normalize_text(row.get("table_id")) == "table 1":
                continue
            if str(row.get("plga_mass_mg_value", "") or "").strip():
                continue
            if str(row.get("surfactant_concentration_text_value", "") or "").strip():
                continue
            if str(row.get("drug_feed_amount_text_value", "") or "").strip():
                continue
            signature = yga_measurement_signature(row)
            if signature:
                candidate_targets = targets_by_signature.get(signature, [])
                if len(candidate_targets) == 1:
                    target_source_key = candidate_targets[0]
                    target_row = next((item for item in doe_rows if row_source_key(item) == target_source_key), None)
                    if target_row is not None:
                        collapse_map[source_key] = {
                            "target_source_key": target_source_key,
                            "variant_class": "duplicate_representation",
                            "variant_signal": "duplicate_representation",
                            "decision_rule": "doe_measurement_signature_duplicate",
                            "decision_reason": (
                                "Small comparator/summary row matches exactly one deterministic DOE row by the complete "
                                "measurement signature and adds no explicit decoded factor assignments, so it is treated as "
                                "a duplicate representation rather than a new formulation identity."
                            ),
                            "notes": (
                                f"matched_source_formulation_id={target_row.get('formulation_id', '')}; "
                                f"measurement_signature={signature}; paper_key={paper_key}"
                            ),
                        }
                        continue
            if not is_measurement_only_later_table_duplicate(row, doe_rows_by_label):
                continue
            label = normalize_text(row.get("raw_formulation_label"))
            target_row = doe_rows_by_label[label][0]
            target_source_key = row_source_key(target_row)
            collapse_map[source_key] = {
                "target_source_key": target_source_key,
                "variant_class": "duplicate_representation",
                "variant_signal": "post_processing_or_measurement_variant",
                "decision_rule": "labeled_measurement_table_duplicate_of_doe_row",
                "decision_reason": (
                    "Later measurement/post-processing table row reuses a deterministic DOE formulation label and only adds "
                    "before/after processing characterization variables, so it is treated as a duplicate representation rather "
                    "than a new formulation identity."
                ),
                "notes": (
                    f"matched_source_formulation_id={target_row.get('formulation_id', '')}; "
                    f"shared_label={target_row.get('raw_formulation_label', '')}; paper_key={paper_key}"
                ),
            }
    return collapse_map


def build_wfdt_checkpoint_coordinate_collapse_map(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    wfdt_rows = [
        row
        for row in rows
        if normalize_doi(row.get("doi", "")) == WFDTQ4VX_DOI or str(row.get("key", "") or "").strip() == "WFDTQ4VX"
    ]
    if not wfdt_rows:
        return {}

    design_rows = [row for row in wfdt_rows if not is_wfdt_checkpoint_row(row)]
    checkpoint_rows = [row for row in wfdt_rows if is_wfdt_checkpoint_row(row)]
    if not design_rows or not checkpoint_rows:
        return {}

    targets_by_signature: dict[str, list[str]] = defaultdict(list)
    for row in design_rows:
        if normalize_text(row.get("instance_kind")) not in {"new_formulation", "variant_formulation"}:
            continue
        signature = wfdt_row_coordinate_signature(row)
        if not signature:
            continue
        targets_by_signature[signature].append(row_source_key(row))

    try:
        source_text = resolve_source_text_path_for_row(wfdt_rows[0]).read_text(encoding="utf-8", errors="replace")
        organic_phase_volume_ml = parse_organic_phase_volume_ml(source_text)
        lines = [clean_ocr_token(line) for line in source_text.splitlines()]
        table1_levels = extract_table1_level_map(lines)
        checkpoint_specs = {
            spec["batch_no"]: spec for spec in extract_checkpoint_rows(lines)
        }
    except (FileNotFoundError, RuntimeError):
        return {}

    collapse_map: dict[str, dict[str, str]] = {}
    for row in checkpoint_rows:
        source_key = row_source_key(row)
        batch_match = re.search(
            r"(\d+)$",
            str(row.get("formulation_id", "") or "") + " " + str(row.get("raw_formulation_label", "") or ""),
        )
        if not batch_match:
            continue
        batch_no = int(batch_match.group(1))
        spec = checkpoint_specs.get(batch_no)
        if spec is None:
            continue
        x1_coded, _ = parse_coded_cell(spec["x1_raw"])
        x2_coded, _ = parse_coded_cell(spec["x2_raw"])
        x3_coded, _ = parse_coded_cell(spec["x3_raw"])
        coordinate_signature = wfdt_coordinate_signature(
            percent_wv_to_mg(interpolate_from_coded(table1_levels["X1"], x1_coded), organic_phase_volume_ml),
            percent_wv_to_mg(interpolate_from_coded(table1_levels["X2"], x2_coded), organic_phase_volume_ml),
            interpolate_from_coded(table1_levels["X3"], x3_coded),
        )
        candidate_targets = targets_by_signature.get(coordinate_signature, [])
        if len(candidate_targets) != 1:
            continue
        target_source_key = candidate_targets[0]
        target_row = next((item for item in design_rows if row_source_key(item) == target_source_key), None)
        if target_row is None:
            continue
        collapse_map[source_key] = {
            "target_source_key": target_source_key,
            "variant_class": "checkpoint_or_validation_variant",
            "variant_signal": "checkpoint_or_validation_variant",
            "decision_rule": "wfdtq4vx_checkpoint_coordinate_signature_match",
            "decision_reason": (
                "Checkpoint batch matches an existing WFDTQ4VX design-row formulation identity by "
                "factor-level coordinate signature reconstructed from the source-paper DOE tables."
            ),
            "notes": (
                f"matched_source_formulation_id={target_row.get('formulation_id', '')}; "
                f"coordinate_signature={coordinate_signature}; batch_no={batch_no}"
            ),
        }
    return collapse_map


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


def dedupe_fieldnames(fieldnames: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for fieldname in fieldnames:
        if fieldname in seen:
            continue
        deduped.append(fieldname)
        seen.add(fieldname)
    return deduped


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    fieldnames = dedupe_fieldnames(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def count_tsv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return sum(1 for _ in reader)


def copy_value_layer_sidecar(input_path: Path, output_path: Path) -> Path:
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_input != resolved_output:
        shutil.copyfile(resolved_input, output_path)
    return output_path


def write_stage5_value_layer_sidecar_manifest(
    *,
    out_dir: Path,
    s5_4_accepted_direct_values_tsv: Path | None = None,
    s5_5_derived_values_tsv: Path | None = None,
    final_row_count_before: int,
    final_row_count_after: int,
) -> dict[str, Any] | None:
    """Pin optional S5 value-layer sidecars without merging them into the final table."""
    sidecar_specs = [
        {
            "provided_path": s5_4_accepted_direct_values_tsv,
            "sidecar_role": "accepted_direct_values",
            "value_layer_stage": "S5-4",
            "separation": "direct",
            "copy_name": VALUE_LAYER_DIRECT_COPY_NAME,
            "notes": "Accepted direct value sidecar only; values are not merged and cannot alter row membership.",
        },
        {
            "provided_path": s5_5_derived_values_tsv,
            "sidecar_role": "derived_values",
            "value_layer_stage": "S5-5",
            "separation": "derived",
            "copy_name": VALUE_LAYER_DERIVED_COPY_NAME,
            "notes": "Derived value sidecar only; values remain separate pending an explicit derived-aware schema.",
        },
    ]

    manifest_rows: list[dict[str, Any]] = []
    for spec in sidecar_specs:
        provided_path = spec["provided_path"]
        if provided_path is None:
            continue
        input_path = Path(provided_path)
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Optional Stage5 value-layer sidecar not found: {input_path}")
        copied_path = copy_value_layer_sidecar(input_path, out_dir / str(spec["copy_name"]))
        manifest_rows.append(
            {
                "sidecar_role": spec["sidecar_role"],
                "value_layer_stage": spec["value_layer_stage"],
                "separation": spec["separation"],
                "input_path": str(input_path.resolve()),
                "copied_output_path": str(copied_path.resolve()),
                "row_count": count_tsv_data_rows(input_path),
                "benchmark_valid": "no",
                "integration_mode": "sidecar_manifest_only_no_final_table_merge",
                "final_row_count_before": final_row_count_before,
                "final_row_count_after": final_row_count_after,
                "row_membership_changed": "no" if final_row_count_before == final_row_count_after else "yes",
                "notes": spec["notes"],
            }
        )

    if not manifest_rows:
        return None

    manifest_tsv_path = out_dir / VALUE_LAYER_SIDECAR_MANIFEST_TSV_NAME
    manifest_json_path = out_dir / VALUE_LAYER_SIDECAR_MANIFEST_JSON_NAME
    write_tsv(
        manifest_tsv_path,
        [
            "sidecar_role",
            "value_layer_stage",
            "separation",
            "input_path",
            "copied_output_path",
            "row_count",
            "benchmark_valid",
            "integration_mode",
            "final_row_count_before",
            "final_row_count_after",
            "row_membership_changed",
            "notes",
        ],
        manifest_rows,
    )
    manifest_payload = {
        "schema_version": "stage5_value_layer_sidecar_manifest_v1",
        "benchmark_valid": "no",
        "integration_mode": "sidecar_manifest_only_no_final_table_merge",
        "direct_values_merge_status": "not_merged",
        "derived_values_merge_status": "not_merged",
        "direct_derived_separation": "preserved",
        "final_row_count_before": final_row_count_before,
        "final_row_count_after": final_row_count_after,
        "row_membership_changed": "no" if final_row_count_before == final_row_count_after else "yes",
        "manifest_tsv_path": str(manifest_tsv_path.resolve()),
        "sidecars": manifest_rows,
    }
    manifest_json_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest_tsv_path": manifest_tsv_path,
        "manifest_json_path": manifest_json_path,
        "rows": manifest_rows,
        "payload": manifest_payload,
    }


def build_downstream_variant_rows(
    *,
    rows: list[dict[str, str]],
    decision_rows: list[dict[str, str]],
    final_id_by_source_id: dict[str, str],
) -> list[dict[str, str]]:
    source_key_by_paper_and_formulation_id = {
        (normalize_text(row.get("key")), normalize_text(row.get("formulation_id"))): row_source_key(row)
        for row in rows
    }
    decision_by_source_id = {
        (
            normalize_text(row.get("zotero_key")),
            normalize_text(row.get("source_formulation_id")),
        ): row
        for row in decision_rows
        if normalize_text(row.get("zotero_key")) and normalize_text(row.get("source_formulation_id"))
    }
    downstream_rows: list[dict[str, str]] = []
    for row in rows:
        decision = decision_by_source_id.get(
            (normalize_text(row.get("key")), normalize_text(row.get("formulation_id")))
        )
        if not decision:
            continue
        decision_name = normalize_text(decision.get("decision"))
        variant_signal = normalize_text(decision.get("variant_signal"))
        if variant_signal != "post_processing_or_measurement_variant":
            continue
        if decision_name not in {"filtered_non_formulation", "collapsed_into_existing"}:
            continue

        linked_primary_final_id = normalize_text(decision.get("target_final_formulation_id"))
        parent_source_formulation_id = str(row.get("parent_instance_id", "") or "").strip()
        primary_link_resolution = "direct_target_final"
        if not linked_primary_final_id and parent_source_formulation_id:
            parent_source_key = source_key_by_paper_and_formulation_id.get(
                (normalize_text(row.get("key")), normalize_text(parent_source_formulation_id))
            )
            if parent_source_key:
                linked_primary_final_id = final_id_by_source_id.get(parent_source_key, "")
                primary_link_resolution = (
                    "parent_source_lookup" if linked_primary_final_id else "parent_source_lookup_unresolved"
                )
            else:
                primary_link_resolution = "parent_source_missing"
        elif not linked_primary_final_id:
            primary_link_resolution = "no_primary_link_available"

        downstream_variable_names_json, downstream_variable_values_json, downstream_variable_signature = (
            downstream_variable_payloads(row)
        )
        sequential_child_tables = sorted(
            {
                str(item.get("table_id") or "").strip()
                for item in row_table_scope_items(row)
                if normalize_text(item.get("table_type")) == "sequential_child"
                and str(item.get("table_id") or "").strip()
            }
        )
        downstream_rows.append(
            {
                "variant_record_id": make_downstream_variant_record_id(row),
                "paper_key": str(row.get("key", "") or "").strip(),
                "doi": str(row.get("doi", "") or "").strip(),
                "linked_primary_final_formulation_id": linked_primary_final_id,
                "parent_source_formulation_id": parent_source_formulation_id,
                "source_formulation_id": str(row.get("formulation_id", "") or "").strip(),
                "source_raw_formulation_label": str(row.get("raw_formulation_label", "") or "").strip(),
                "instance_kind": str(row.get("instance_kind", "") or "").strip(),
                "formulation_role": str(row.get("formulation_role", "") or "").strip(),
                "change_role": str(row.get("change_role", "") or "").strip(),
                "variant_class": str(decision.get("variant_class", "") or "").strip(),
                "variant_signal": str(decision.get("variant_signal", "") or "").strip(),
                "candidate_source": str(row.get("candidate_source", "") or "").strip(),
                "instance_context_tags": str(row.get("instance_context_tags", "") or "").strip(),
                "change_context_tags": str(row.get("change_context_tags", "") or "").strip(),
                "change_descriptions": str(row.get("change_descriptions", "") or "").strip(),
                IDENTITY_VARIABLES_FIELD: str(row.get(IDENTITY_VARIABLES_FIELD, "") or "").strip(),
                "downstream_variable_names_json": downstream_variable_names_json,
                "downstream_variable_values_json": downstream_variable_values_json,
                "downstream_variable_signature": downstream_variable_signature,
                "sequential_child_table_ids_json": json.dumps(sequential_child_tables, ensure_ascii=True),
                "table_formulation_scopes_json": str(row.get("table_formulation_scopes_json", "") or "").strip(),
                "table_variable_roles_json": str(row.get("table_variable_roles_json", "") or "").strip(),
                "selection_markers_json": str(row.get("selection_markers_json", "") or "").strip(),
                "inheritance_markers_json": str(row.get("inheritance_markers_json", "") or "").strip(),
                "boundary_markers_json": str(row.get("boundary_markers_json", "") or "").strip(),
                "instance_evidence_region_type": str(row.get("instance_evidence_region_type", "") or "").strip(),
                "evidence_section": str(row.get("evidence_section", "") or "").strip(),
                "evidence_span_text": str(row.get("evidence_span_text", "") or "").strip(),
                "supporting_evidence_refs": str(row.get("supporting_evidence_refs", "") or "").strip(),
                "excluded_from_primary_database": "yes",
                "exclusion_decision": decision_name,
                "exclusion_reason": str(decision.get("decision_rule", "") or "").strip(),
                "exclusion_reason_text": str(
                    decision.get("collapse_reason") or decision.get("decision_reason") or ""
                ).strip(),
                "primary_link_resolution": primary_link_resolution,
                "primary_table_contract": "excluded_from_primary_benchmark_facing_formulation_database",
            }
        )
    return sorted(
        downstream_rows,
        key=lambda item: (
            item["paper_key"],
            item["linked_primary_final_formulation_id"],
            item["source_formulation_id"],
        ),
    )


def build_summary_markdown(
    input_path: Path,
    final_rows: list[dict[str, str]],
    decision_rows: list[dict[str, str]],
    downstream_variant_rows: list[dict[str, str]],
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
        "- applies the validated WFDTQ4VX checkpoint coordinate rule to collapse checkpoint batches that exactly match one design-row formulation identity",
        "- preserves provenance by retaining representative-row metadata, collapsed-variant membership, a row-level decision trace, and a linked downstream-variant record surface for rows excluded from the primary benchmark-facing database",
        "",
        "## What phase 1 intentionally does not handle",
        "",
        "- semantic inheritance inference beyond Stage3 resolved relation outputs",
        "- generalized DOE coordinate reconciliation beyond the narrow validated WFDTQ4VX checkpoint rule",
        "- Stage 5B benchmark comparison against GT",
        "- modeling-target-specific filtering such as PLGA-only export subsets",
        "",
        "## Filtering rules applied",
        "",
        "- `explicit_candidate_non_formulation`",
        "- `parent_linked_non_synthesis_descendant_variant`",
        "- `unparented_shared_condition_summary`",
        "- `comparative_summary_reference`",
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
        f"- downstream_variant_records: `{len(downstream_variant_rows)}`",
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
            "- DOE-aware coordinate closure still needs a later explicit contract when unique deterministic mapping is not already provided by the narrow validated WFDTQ4VX checkpoint rule",
            "- benchmark comparison still requires the separate Stage 5B comparison step",
        ]
    )
    summary_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def build_minimal_final_output(
    input_tsv: Path,
    out_dir: Path,
    relation_records_tsv: Path,
    resolved_relation_fields_tsv: Path,
    s5_4_accepted_direct_values_tsv: Path | None = None,
    s5_5_derived_values_tsv: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for optional_sidecar in (s5_4_accepted_direct_values_tsv, s5_5_derived_values_tsv):
        if optional_sidecar is not None and (not optional_sidecar.exists() or not optional_sidecar.is_file()):
            raise FileNotFoundError(f"Optional Stage5 value-layer sidecar not found: {optional_sidecar}")
    rows = read_candidate_rows(input_tsv)
    if not rows:
        raise ValueError(f"No candidate rows found in {input_tsv}")
    if relation_records_tsv is None:
        raise ValueError("Stage5 requires --relation-records-tsv; silent bypass is not allowed.")
    if resolved_relation_fields_tsv is None:
        raise ValueError("Stage5 requires --resolved-relation-fields-tsv; silent bypass is not allowed.")
    relation_metadata = load_relation_metadata(relation_records_tsv)
    resolved_relation_field_map = load_resolved_relation_fields(resolved_relation_fields_tsv)
    source_text_by_paper: dict[str, str] = {}
    rows_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_paper[str(row.get("key", "") or "").strip()].append(row)

    original_fieldnames = list(rows[0].keys())
    for metric_field in ("dl_percent",):
        for suffix in ("value", "value_text", "scope", "membership_confidence", "evidence_region_type", "missing_reason"):
            column = f"{metric_field}_{suffix}"
            if column not in original_fieldnames:
                original_fieldnames.append(column)
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
        should_filter, filter_rule, filter_reason = should_filter_non_formulation(
            row,
            core_fields,
            paper_rows=rows_by_paper.get(str(row.get("key", "") or "").strip(), []),
        )
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
    loaded_label_table_duplicate_targets = build_loaded_label_table_duplicate_map(rows)
    doe_measurement_duplicate_targets = build_doe_measurement_duplicate_collapse_map(rows)
    wfdt_checkpoint_targets: dict[str, dict[str, str]] = {}
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

    for source_key, target_source_key in loaded_label_table_duplicate_targets.items():
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
            "decision_rule": "loaded_label_duplicate_of_table_material_row",
            "decision_reason": (
                "LLM loaded-name formulation label uniquely maps to an already materialized "
                "paper-local table row with the same material-composition label."
            ),
            "collapse_reason": (
                "Collapsed as a duplicate representation of the table-derived material-composition row."
            ),
            "review_needed": "no",
            "notes": (
                f"matched_source_formulation_id={target_row.get('formulation_id', '')}; "
                f"loaded_label_key={formulation_material_label_key(row_by_source_key[source_key].get('raw_formulation_label', ''))}"
            ),
        }

    for source_key, payload in doe_measurement_duplicate_targets.items():
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
                "Collapsed as a duplicate summary/comparator representation because its complete measurement "
                "signature matches exactly one deterministic DOE formulation row."
            ),
            "review_needed": "no",
            "notes": payload["notes"],
        }

    for source_key, payload in wfdt_checkpoint_targets.items():
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
                "Collapsed as a checkpoint/validation variant because the WFDTQ4VX checkpoint batch "
                "matches exactly one design-row formulation identity under the validated coordinate rule."
            ),
            "review_needed": "no",
            "notes": payload["notes"],
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
            variant_class = ""
            if rule in {
                "characterization_only_post_processing",
                "parent_linked_non_synthesis_descendant_variant",
            }:
                variant_class = variant_signal or "post_processing_or_measurement_variant"
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
        final_row, applied_source_group_direct_fields = apply_source_group_direct_field_retention(
            final_row=final_row,
            source_group=source_group,
        )
        final_row[STUDIED_VARIABLES_FIELD] = build_studied_variables_json(representative, final_row)
        paper_key = str(final_row.get("key", "") or "").strip()
        if paper_key and paper_key not in source_text_by_paper:
            try:
                source_text_by_paper[paper_key] = resolve_source_text_path_for_row(representative).read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except FileNotFoundError:
                source_text_by_paper[paper_key] = ""
        final_row, applied_global_fields = apply_global_polymer_material_carrythrough(
            final_row=final_row,
            source_text=source_text_by_paper.get(paper_key, ""),
        )
        final_row, applied_preparation_fields = apply_global_preparation_material_carrythrough(
            final_row=final_row,
            source_text=source_text_by_paper.get(paper_key, ""),
        )
        applied_value_unit_fields = apply_concentration_value_unit_splitter(final_row)
        applied_global_fields = set(applied_global_fields) | set(applied_preparation_fields)
        applied_global_fields |= applied_value_unit_fields
        derived_mass_provenance = build_derived_mass_provenance_for_row(
            final_row,
            source_text=source_text_by_paper.get(paper_key, ""),
        )
        final_row["derived_mass_provenance_json"] = json.dumps(derived_mass_provenance, ensure_ascii=False) if derived_mass_provenance else ""
        if applied_relation_fields:
            final_row["field_source_type"] = "relation_resolved"
        if applied_source_group_direct_fields:
            final_row["field_source_type"] = "source_group_direct_field_retention"
        if applied_global_fields:
            final_row["field_source_type"] = "global_material_evidence"
        elif not applied_relation_fields and not applied_source_group_direct_fields and any(
            not field_bundle_value(final_row, field_name)
            for field_name in RESOLVED_RELATION_FIELD_NAMES
        ):
            final_row["field_source_type"] = "unresolved_blank"
        final_rows.append(final_row)

    apply_paper_level_preparation_method_consensus(final_rows)

    downstream_variant_rows = build_downstream_variant_rows(
        rows=rows,
        decision_rows=decision_rows,
        final_id_by_source_id=final_id_by_source_id,
    )

    decision_trace_path = out_dir / DECISION_TRACE_NAME
    final_table_path = out_dir / FINAL_TABLE_NAME
    downstream_variant_path = out_dir / DOWNSTREAM_VARIANT_RECORDS_NAME
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
            "derived_mass_provenance_json",
            *([STUDIED_VARIABLES_FIELD] if STUDIED_VARIABLES_FIELD not in original_fieldnames else []),
            *original_fieldnames,
            *[name for name in STAGE5_GLOBAL_PREPARATION_FIELDNAMES if name not in original_fieldnames],
            *[name for name in PREPARATION_METHOD_FIELDNAMES if name not in original_fieldnames and name not in STAGE5_GLOBAL_PREPARATION_FIELDNAMES],
        ],
        final_rows,
    )

    write_tsv(
        downstream_variant_path,
        [
            "variant_record_id",
            "paper_key",
            "doi",
            "linked_primary_final_formulation_id",
            "parent_source_formulation_id",
            "source_formulation_id",
            "source_raw_formulation_label",
            "instance_kind",
            "formulation_role",
            "change_role",
            "variant_class",
            "variant_signal",
            "candidate_source",
            "instance_context_tags",
            "change_context_tags",
            "change_descriptions",
            IDENTITY_VARIABLES_FIELD,
            "downstream_variable_names_json",
            "downstream_variable_values_json",
            "downstream_variable_signature",
            "sequential_child_table_ids_json",
            "table_formulation_scopes_json",
            "table_variable_roles_json",
            "selection_markers_json",
            "inheritance_markers_json",
            "boundary_markers_json",
            "instance_evidence_region_type",
            "evidence_section",
            "evidence_span_text",
            "supporting_evidence_refs",
            "excluded_from_primary_database",
            "exclusion_decision",
            "exclusion_reason",
            "exclusion_reason_text",
            "primary_link_resolution",
            "primary_table_contract",
        ],
        downstream_variant_rows,
    )

    build_summary_markdown(
        input_tsv,
        final_rows,
        decision_rows,
        downstream_variant_rows,
        summary_path,
        relation_records_tsv,
        resolved_relation_fields_tsv,
    )

    value_layer_sidecar_manifest = write_stage5_value_layer_sidecar_manifest(
        out_dir=out_dir,
        s5_4_accepted_direct_values_tsv=s5_4_accepted_direct_values_tsv,
        s5_5_derived_values_tsv=s5_5_derived_values_tsv,
        final_row_count_before=len(final_rows),
        final_row_count_after=len(final_rows),
    )

    return {
        "input_rows": len(rows),
        "final_rows": len(final_rows),
        "filtered_rows": sum(1 for row in decision_rows if row["decision"] == "filtered_non_formulation"),
        "collapsed_rows": sum(1 for row in decision_rows if row["decision"] == "collapsed_into_existing"),
        "kept_rows": sum(1 for row in decision_rows if row["decision"] == "kept"),
        "downstream_variant_rows": len(downstream_variant_rows),
        "final_table_path": final_table_path,
        "downstream_variant_path": downstream_variant_path,
        "decision_trace_path": decision_trace_path,
        "summary_path": summary_path,
        "relation_records_tsv": relation_records_tsv,
        "resolved_relation_fields_tsv": resolved_relation_fields_tsv,
        "value_layer_sidecar_manifest": value_layer_sidecar_manifest,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build phase-1 minimal final-output artifacts from Stage2 candidate-instance TSV output."
    )
    parser.add_argument("--input-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--relation-records-tsv", required=True, type=Path)
    parser.add_argument("--resolved-relation-fields-tsv", required=True, type=Path)
    parser.add_argument("--s5-4-accepted-direct-values-tsv", type=Path)
    parser.add_argument("--s5-5-derived-values-tsv", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = build_minimal_final_output(
        args.input_tsv,
        args.out_dir,
        relation_records_tsv=args.relation_records_tsv,
        resolved_relation_fields_tsv=args.resolved_relation_fields_tsv,
        s5_4_accepted_direct_values_tsv=args.s5_4_accepted_direct_values_tsv,
        s5_5_derived_values_tsv=args.s5_5_derived_values_tsv,
    )
    output_payload = {
        "input_rows": stats["input_rows"],
        "final_rows": stats["final_rows"],
        "filtered_rows": stats["filtered_rows"],
        "collapsed_rows": stats["collapsed_rows"],
        "kept_rows": stats["kept_rows"],
        "downstream_variant_rows": stats["downstream_variant_rows"],
        "relation_records_tsv": str(stats["relation_records_tsv"]) if stats["relation_records_tsv"] else "",
        "resolved_relation_fields_tsv": str(stats["resolved_relation_fields_tsv"]),
        "final_table_path": str(stats["final_table_path"]),
        "downstream_variant_path": str(stats["downstream_variant_path"]),
        "decision_trace_path": str(stats["decision_trace_path"]),
        "summary_path": str(stats["summary_path"]),
    }
    if stats.get("value_layer_sidecar_manifest"):
        output_payload["value_layer_sidecar_manifest_tsv"] = str(
            stats["value_layer_sidecar_manifest"]["manifest_tsv_path"]
        )
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
