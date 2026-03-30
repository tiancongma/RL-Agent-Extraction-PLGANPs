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

from src.stage2_sampling_labels.auto_extract_weak_labels_v7pilot_r3_fixparse import (
    CORE_FIELDS,
    build_output_columns,
)


LEGACY_TSV_NAME = "weak_labels__v7pilot_r3_fixparse.tsv"
LEGACY_JSONL_NAME = "weak_labels__v7pilot_r3_fixparse.jsonl"
TRACE_TSV_NAME = "compatibility_projection_trace_v1.tsv"
SUMMARY_JSON_NAME = "compatibility_projection_summary_v1.json"
CONTRACT_TSV_NAME = "stage2_replacement_compatibility_projection_contract.tsv"
IDENTITY_VARIABLES_FIELD = "identity_variables_json"

DIRECT = "direct"
DERIVED = "derived"
COMPRESSED = "compressed"
UNAVAILABLE = "unavailable"

OBJECT_KEYS = {
    "formulation_identity_candidate": "formulation_identity_candidates",
    "component_candidate": "component_candidates",
    "phase_candidate": "phase_candidates",
    "process_step_candidate": "process_step_candidates",
    "variable_or_factor_candidate": "variable_or_factor_candidates",
    "measurement_candidate": "measurement_candidates",
    "relation_cue": "relation_cues",
    "evidence_handoff": "evidence_handoffs",
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
    return columns


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


def load_jsonl_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
    return documents


def object_rows(document: dict[str, Any], object_type: str) -> list[dict[str, Any]]:
    value = document.get(OBJECT_KEYS[object_type], [])
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
            "change_role": "unclear",
            "instance_context_tags": stringify_json(instance_context_tags),
            "change_context_tags": stringify_json(change_context_tags),
            "formulation_role": normalize_text(identity.get("formulation_role")),
            "instance_confidence": normalize_text(identity.get("identity_confidence")) or "projected",
            "candidate_source": normalize_text(identity.get("candidate_source")) or "compatibility_projection",
        }
    )
    return row


def project_document(document: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    doi = normalize_text(document.get("doi"))
    model_name = normalize_text(document.get("model_name")) or "stage2_replacement_compatibility_projection_v1"
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
            lambda item: first_number(find_property(item, "molecular weight", "mw", "kda")),
            lambda item: find_property(item, "molecular weight", "mw", "kda"),
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

    for field in CORE_FIELDS:
        values = {normalize_text(row.get(f"{field}_value_text")) for row in rows if normalize_text(row.get(f"{field}_value_text"))}
        scope = "global_shared" if field in SHARED_SCOPE_FIELDS and len(values) == 1 and len(rows) > 1 else "instance_specific"
        for row in rows:
            row[f"{field}_scope"] = scope if normalize_text(row.get(f"{field}_value_text")) else ""

    return rows, traces, jsonl_rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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
    documents = load_jsonl_documents(input_path)
    all_rows: list[dict[str, str]] = []
    all_traces: list[dict[str, str]] = []
    all_jsonl_rows: list[dict[str, Any]] = []
    for document in documents:
        rows, traces, jsonl_rows = project_document(document)
        all_rows.extend(rows)
        all_traces.extend(traces)
        all_jsonl_rows.extend(jsonl_rows)

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
    summary = {
        "schema": "stage2_replacement_compatibility_projection_v1",
        "status": "transitional_support",
        "documents": len(documents),
        "projected_rows": len(all_rows),
        "trace_rows": len(all_traces),
        "legacy_surface_columns": len(compatibility_output_columns()),
        "output_files": [
            str(output_dir / LEGACY_TSV_NAME),
            str(output_dir / LEGACY_JSONL_NAME),
            str(output_dir / TRACE_TSV_NAME),
            str(contract_path),
        ],
    }
    (output_dir / SUMMARY_JSON_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] projected {len(all_rows)} rows from {len(documents)} document payload(s) -> {output_dir}")


if __name__ == "__main__":
    main()
