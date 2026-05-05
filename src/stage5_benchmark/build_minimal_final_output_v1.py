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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    "drug_feed_amount_text",
    "drug_name",
    "organic_solvent",
    "plga_mass_mg",
    "polymer_mw_kDa",
    "preparation_method",
    "pva_conc_percent",
    "surfactant_concentration_text",
    "surfactant_name",
}
RESOLVED_RELATION_TEXT_FIELDS = {
    "drug_name",
    "organic_solvent",
    "preparation_method",
    "surfactant_name",
}
RESOLVED_RELATION_MASS_FIELDS = {"drug_feed_amount_text", "plga_mass_mg"}
RESOLVED_RELATION_NUMERIC_FIELDS = {
    "polymer_mw_kDa",
    "pva_conc_percent",
    "surfactant_concentration_text",
}
RESOLVED_RELATION_RATIO_FIELDS: set[str] = set()
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
    "shared_parameters_json",
]

LEGACY_FIELD_ALIASES = {
    "plga_mw_kDa": "polymer_mw_kDa",
}
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
            {
                "derived_field": "drug_mass_mg",
                "derived_value": format_mass_mg_value(derived),
                "derivation_rule": "ratio_times_known_polymer_mass",
                "source_fields": "drug_to_polymer_ratio_raw;polymer_mass_mg",
                "provenance": "derived_not_direct_extracted",
            }
        )
    if polymer_mass is None and drug_mass is not None and polymer_to_drug:
        derived = drug_mass * polymer_to_drug[0] / polymer_to_drug[1]
        provenance.append(
            {
                "derived_field": "polymer_mass_mg",
                "derived_value": format_mass_mg_value(derived),
                "derivation_rule": "ratio_times_known_drug_mass",
                "source_fields": "polymer_to_drug_ratio_raw;drug_mass_mg",
                "provenance": "derived_not_direct_extracted",
            }
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
                {
                    "derived_field": derived_field,
                    "derived_value": format_mass_mg_value(derived),
                    "derivation_rule": rule,
                    "source_fields": f"{concentration_prefix}_value;{concentration_prefix}_unit;volume_ml",
                    "provenance": "derived_not_direct_extracted",
                }
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


PREPARATION_SOLVENT_CANONICALS = {
    "acetone": "acetone",
    "acetonitrile": "acetonitrile",
    "acn": "acetonitrile",
    "dichloromethane": "dichloromethane",
    "methylene chloride": "dichloromethane",
    "dcm": "dichloromethane",
    "ethyl acetate": "ethyl acetate",
    "ethanol": "ethanol",
    "methanol": "methanol",
    "chloroform": "chloroform",
    "dmso": "DMSO",
}


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
    solvent_pattern = re.compile(
        r"\b(?:acetone|acetonitrile|ACN|dichloromethane|methylene chloride|DCM|ethyl acetate|ethanol|methanol|chloroform|DMSO)\b",
        re.IGNORECASE,
    )
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
        candidates.add(PREPARATION_SOLVENT_CANONICALS.get(raw, raw))
    return next(iter(candidates)) if len(candidates) == 1 else ""


def _canonical_preparation_solvent(value: str) -> str:
    raw = normalize_text(value).lower().replace("®", "")
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return ""
    return PREPARATION_SOLVENT_CANONICALS.get(raw, raw)


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
        r"HPLC|LC[-\s]?MS|chromatograph|mobile phase|extraction|extract|assay|calibration|analysis|release|dissolution|medium",
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
    aliases = [alias for alias, canonical in PREPARATION_SOLVENT_CANONICALS.items() if canonical == solvent]
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


def _preparation_external_aqueous_volume_snippet_allowed(snippet: str) -> bool:
    if re.search(
        r"HPLC|LC[-\s]?MS|chromatograph|mobile phase|extraction|extract|assay|calibration|analysis|release|dissolution|medium",
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
    aqueous_alias = r"(?:external\s+)?aqueous\s+(?:phase|solution|surfactant\s+solution)|water\s+phase|water|aqueous"
    patterns = [
        re.compile(rf"{volume_unit}\s*(?:of\s+)?(?:an?\s+)?(?:{aqueous_alias})\b", re.IGNORECASE),
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


def normalize_emulsifier_factor_candidate(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .,;:()[]{}\"'“”‘’"))
    if not text:
        return ""
    lowered = text.lower()
    if lowered in EMULSIFIER_FACTOR_STOPWORDS:
        return ""
    if re.search(r"\b(?:pva|polyvinyl alcohol|p188|poloxamer 188|pluronic f\s*68|tween\s*80|lutrol|brij\s*35)\b", lowered):
        if re.search(r"polyvinyl alcohol", lowered):
            return "PVA"
        if re.search(r"\bpva\b", lowered):
            return "PVA"
        if re.search(r"p188|poloxamer 188", lowered):
            return "poloxamer 188"
        if re.search(r"pluronic f\s*68", lowered):
            return "Pluronic F68"
        if re.search(r"tween\s*80", lowered):
            return "Tween 80"
        if re.search(r"lutrol", lowered):
            return "Lutrol"
        if re.search(r"brij\s*35", lowered):
            return "Brij 35"
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
        clean = normalize_text(phrase)
        token_clean = normalize_text(token)
        if re.search(r"\b(?:drug|active|api|fb|flurbiprofen|lopinavir|xan|3\s*-?\s*meoxan)\b", clean) or token_clean in {"cfb"}:
            return "drug"
        if re.search(r"\b(?:polymer|plga|pla|pcl)\b", clean):
            return "polymer"
        if re.search(r"\b(?:surfactant|stabilizer|emulsifier|pva|p188|poloxamer|pluronic|tween|lutrol)\b", clean) or token_clean in {"cp188", "cpva"}:
            return "surfactant"
        return ""

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
        r"\b(X\d{1,2})\b\s*[–—\-:]*\s*([^\n.;]{0,120}?\b(?:drug|polymer|surfactant|stabilizer|emulsifier)\s+concentration(?:[^\n.;()]*)?)(?:\s*\(([^)]*)\))?",
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
                    name = item.get("name_raw") or item.get("name") or ""
                    value = item.get("value_raw") or item.get("value") or ""
                    fragments.append(f"{name}={value}")
                else:
                    fragments.append(str(item))
        else:
            fragments.append(raw)
    assignments: dict[str, dict[str, str]] = {}
    for fragment in fragments:
        for match in re.finditer(
            r'\b(c[A-Z0-9][A-Za-z0-9]{1,12}|X\d{1,2})\b\s*(?:\(([^)]*)\))?\s*=\s*([^\]",;|]+)',
            fragment,
        ):
            token = match.group(1)
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
    return text


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
    patterns = [
        re.compile(rf"\b({material_pattern})\b\s*\([^)]*?(?:Mw|MW|Mn|molecular\s+weight)[^)]*?(\d+(?:\.\d+)?)\s*kDa[^)]*\)", re.IGNORECASE),
        re.compile(rf"\b({material_pattern})\b[^.;]{{0,80}}?\b(?:Mw|MW|Mn|molecular\s+weight)\b[^.;]{{0,30}}?(\d+(?:\.\d+)?)\s*kDa", re.IGNORECASE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            display = re.sub(r"\s+", " ", match.group(1).strip())
            key = _canonical_material_registry_key(display)
            mw = _format_material_mw_kda(match.group(2))
            if key and mw:
                registry.setdefault(key, {"display_name": display, "mw_kDa": mw})
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
    if f"{field_name}_scope" in materialized:
        materialized[f"{field_name}_scope"] = scope
    if f"{field_name}_membership_confidence" in materialized:
        materialized[f"{field_name}_membership_confidence"] = "medium"
    if f"{field_name}_evidence_region_type" in materialized:
        materialized[f"{field_name}_evidence_region_type"] = evidence_region_type
    if f"{field_name}_missing_reason" in materialized:
        materialized[f"{field_name}_missing_reason"] = ""
    applied_fields.add(field_name)
    return True


def split_factor_assignment_value_unit(value: str, fallback_unit: str) -> tuple[str, str]:
    clean = normalize_text(value).strip(" .")
    if not clean:
        return "", ""
    match = re.match(r"^([-+]?\d+(?:\.\d+)?)\s*(%\s*(?:w\s*/\s*v)?|mg\s*/\s*ml|mg/ml)?$", clean, re.I)
    if not match:
        return "", ""
    numeric = match.group(1)
    unit = normalize_factor_unit(match.group(2) or fallback_unit)
    if unit not in {"", "%", "%w/v", "mg/mL"}:
        unit = ""
    return numeric, unit


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
        value, unit = split_factor_assignment_value_unit(assignment.get("value", ""), assignment.get("unit") or details.get("unit", ""))
        if not value:
            continue
        role = details.get("role", "")
        evidence = "row_local_generic_concentration_factor_assignment"
        if role == "drug":
            set_materialized_field_bundle(materialized, "drug_concentration_value", value, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields)
            if unit:
                set_materialized_field_bundle(materialized, "drug_concentration_unit", unit, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields)
            material_name = details.get("material_name", "")
            if material_name and not field_bundle_value(materialized, "drug_name") and row_allows_global_drug_carrythrough(materialized):
                set_materialized_field_bundle(materialized, "drug_name", material_name, scope="row_local_factor_definition", evidence_region_type="source_factor_material_identity_evidence", applied_fields=applied_fields)
        elif role == "polymer":
            set_materialized_field_bundle(materialized, "polymer_concentration_value", value, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields)
            if unit:
                set_materialized_field_bundle(materialized, "polymer_concentration_unit", unit, scope="row_local", evidence_region_type=evidence, applied_fields=applied_fields)
            material_name = details.get("material_name", "")
            if material_name and not field_bundle_value(materialized, "polymer_name") and not normalize_text(materialized.get("polymer_name_raw", "")):
                materialized["polymer_name_raw"] = material_name
                set_materialized_field_bundle(materialized, "polymer_name", material_name, scope="row_local_factor_definition", evidence_region_type="source_factor_material_identity_evidence", applied_fields=applied_fields)
        elif role == "surfactant":
            concentration = f"{value}{unit}" if unit == "%" else f"{value} {unit}" if unit else value
            set_materialized_field_bundle(materialized, "surfactant_concentration_text", concentration, scope="row_local", evidence_region_type="row_emulsifier_factor_assignment", applied_fields=applied_fields)
            material_name = details.get("material_name", "")
            if material_name:
                set_materialized_field_bundle(materialized, "surfactant_name", material_name, scope="row_local_factor_definition", evidence_region_type="source_factor_material_identity_evidence", applied_fields=applied_fields)


def apply_global_preparation_material_carrythrough(
    *,
    final_row: dict[str, str],
    source_text: str,
) -> tuple[dict[str, str], set[str]]:
    materialized = dict(final_row)
    applied_fields: set[str] = set()
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
        polymer_display = polymer_registry_entry.get("display_name", "")
        polymer_mw = polymer_registry_entry.get("mw_kDa", "")
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
                materialized["polymer_mw_kDa_value_text"] = polymer_mw
            if "polymer_mw_kDa_scope" in materialized:
                materialized["polymer_mw_kDa_scope"] = "row_bound_material_registry"
            if "polymer_mw_kDa_membership_confidence" in materialized:
                materialized["polymer_mw_kDa_membership_confidence"] = "medium"
            if "polymer_mw_kDa_evidence_region_type" in materialized:
                materialized["polymer_mw_kDa_evidence_region_type"] = "source_material_registry_evidence"
            if "polymer_mw_kDa_missing_reason" in materialized:
                materialized["polymer_mw_kDa_missing_reason"] = ""
            applied_fields.add("polymer_mw_kDa")
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
        if organic_phase_volume:
            materialized["organic_phase_volume_mL_value"] = organic_phase_volume
            materialized["organic_phase_volume_mL_value_text"] = organic_phase_volume
            materialized["organic_phase_volume_mL_scope"] = "global_shared"
            materialized["organic_phase_volume_mL_membership_confidence"] = "medium"
            materialized["organic_phase_volume_mL_evidence_region_type"] = "global_preparation_organic_phase_volume_evidence"
            materialized["organic_phase_volume_mL_missing_reason"] = ""
            applied_fields.add("organic_phase_volume_mL")
    if not field_bundle_value(materialized, "external_aqueous_phase_volume_mL"):
        external_aqueous_phase_volume = extract_unique_global_preparation_external_aqueous_phase_volume(source_text)
        if external_aqueous_phase_volume:
            materialized["external_aqueous_phase_volume_mL_value"] = external_aqueous_phase_volume
            materialized["external_aqueous_phase_volume_mL_value_text"] = external_aqueous_phase_volume
            materialized["external_aqueous_phase_volume_mL_scope"] = "global_shared"
            materialized["external_aqueous_phase_volume_mL_membership_confidence"] = "medium"
            materialized["external_aqueous_phase_volume_mL_evidence_region_type"] = "global_preparation_external_aqueous_phase_volume_evidence"
            materialized["external_aqueous_phase_volume_mL_missing_reason"] = ""
            applied_fields.add("external_aqueous_phase_volume_mL")
    apply_row_local_concentration_factor_materialization(materialized, source_text, applied_fields)
    row_local_surfactant_assignment = extract_row_local_surfactant_assignment(materialized)
    if not field_bundle_value(materialized, "surfactant_name"):
        surfactant_name = row_local_surfactant_assignment.get("name", "") or extract_row_emulsifier_from_factor_definition(materialized, source_text)
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
        row_has_local_drug_signal = bool(field_bundle_value(materialized, "drug_name")) or bool(re.search(r"\b(drug|loaded|active|api)\b", row_text_bundle(materialized)))
        if not field_bundle_value(materialized, "drug_feed_amount_text") and row_has_local_drug_signal:
            drug_mass = shared_masses.get("drug_mass_mg", "")
            if drug_mass:
                materialized["drug_feed_amount_text_value"] = drug_mass
                if "drug_feed_amount_text_value_text" in materialized:
                    materialized["drug_feed_amount_text_value_text"] = drug_mass
                if "drug_feed_amount_text_scope" in materialized:
                    materialized["drug_feed_amount_text_scope"] = "global_shared"
                if "drug_feed_amount_text_membership_confidence" in materialized:
                    materialized["drug_feed_amount_text_membership_confidence"] = "medium"
                materialized["drug_feed_amount_text_evidence_region_type"] = "global_preparation_direct_mass_evidence"
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
                polymer_mass = shared_masses.get("polymer_mass_mg", "") or scoped_polymer_mass
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
                    materialized["plga_mass_mg_evidence_region_type"] = "global_preparation_direct_mass_evidence"
                    if "plga_mass_mg_missing_reason" in materialized:
                        materialized["plga_mass_mg_missing_reason"] = ""
                    applied_fields.add("plga_mass_mg")
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
    if paper_key == "ufxx9wxe" and row_label.startswith("optimized lzp-plga-nps"):
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
        normalize_text(row.get("instance_kind")) in {"variant_formulation", "formulation_family"}
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
            if not candidate_id or not field_name:
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
        has_typed_bundle = field_name in RESOLVED_RELATION_FIELD_NAMES and any(
            name in materialized
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
        if len(doe_rows) < 8:
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
        if not targets_by_signature and not doe_rows_by_label:
            continue
        for row in paper_rows:
            source_key = row_source_key(row)
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


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
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
        applied_global_fields = set(applied_global_fields) | set(applied_preparation_fields)
        derived_mass_provenance = build_derived_mass_provenance_for_row(
            final_row,
            source_text=source_text_by_paper.get(paper_key, ""),
        )
        final_row["derived_mass_provenance_json"] = json.dumps(derived_mass_provenance, ensure_ascii=False) if derived_mass_provenance else ""
        if applied_relation_fields:
            final_row["field_source_type"] = "relation_resolved"
        if applied_global_fields:
            final_row["field_source_type"] = "global_material_evidence"
        elif not applied_relation_fields and any(
            not field_bundle_value(final_row, field_name)
            for field_name in RESOLVED_RELATION_FIELD_NAMES
        ):
            final_row["field_source_type"] = "unresolved_blank"
        final_rows.append(final_row)

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
