#!/usr/bin/env python3
from __future__ import annotations

"""
Deterministically lift legacy Stage2 wide-row weak labels into semantic-object payloads.

Purpose:
- Support governed replacement-path validation before a true paper-driven semantic
  Stage2 emitter exists.
- Reuse real paper-driven legacy Stage2 outputs as the source surface.
- Emit auditable semantic-object JSONL without adding any new semantic inference.

This script does not:
- call any LLM or external API,
- replace the long-term semantic Stage2 emitter,
- overwrite historical run artifacts.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stage2_sampling_labels.auto_extract_weak_labels_v7pilot_r3_fixparse import (
    CORE_FIELDS,
    add_or_update_component,
    build_component_properties,
    infer_phase_context,
    parse_json_string_list,
    top_level_split,
)


SEMANTIC_JSONL_NAME = "semantic_stage2_objects_v1.jsonl"
SUMMARY_TSV_NAME = "semantic_stage2_object_summary_v1.tsv"
MANIFEST_JSON_NAME = "semantic_stage2_object_manifest_v1.json"

MEASUREMENT_FIELDS = {
    "size_nm": ("size", "nm"),
    "pdi": ("pdi", ""),
    "zeta_mV": ("zeta potential", "mV"),
    "encapsulation_efficiency_percent": ("encapsulation efficiency", "%"),
    "loading_content_percent": ("loading content", "%"),
}

COMPONENT_FIELD_SPECS = [
    {
        "name_field": "polymer_name_raw",
        "fallback_role": "polymer",
        "source_field": "polymer_name_raw",
        "amount_field": "plga_mass_mg_value_text",
        "parsed_value_field": "plga_mass_mg_value",
    },
    {
        "name_field": "surfactant_name_value_text",
        "fallback_role": "surfactant",
        "source_field": "surfactant_name",
        "amount_field": "surfactant_concentration_text_value_text",
        "parsed_value_field": "surfactant_concentration_text_value",
    },
    {
        "name_field": "organic_solvent_value_text",
        "fallback_role": "organic_solvent",
        "source_field": "organic_solvent",
        "amount_field": "",
        "parsed_value_field": "",
    },
    {
        "name_field": "drug_name_value_text",
        "fallback_role": "drug",
        "source_field": "drug_name",
        "amount_field": "drug_feed_amount_text_value_text",
        "parsed_value_field": "drug_feed_amount_text_value",
    },
]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_token(value: Any) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


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


def parse_delimited_names(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = top_level_split(cleaned, (" | ", ";", ", and ", " and "))
    return [part.strip() for part in parts if part.strip()]


def choose_raw_names(row: dict[str, str], field_name: str) -> list[str]:
    text = normalize_text(row.get(field_name))
    if not text and field_name == "polymer_name_raw":
        fallback = normalize_text(row.get("polymer_identity"))
        return [fallback] if fallback else []
    return parse_delimited_names(text)


def choose_unit_from_expression(text: str) -> str:
    lowered = normalize_text(text).lower()
    for unit in ["mg/mL", "% w/v", "%", "mg", "kDa", "mV", "nm", "mL", "uL"]:
        if unit.lower() in lowered:
            return unit
    return ""


def component_properties_json(row: dict[str, str], component_role: str, component_name_raw: str) -> str:
    props = build_component_properties(row, component_role=component_role, component_name_raw=component_name_raw)
    if not props:
        return "[]"
    converted = [
        {
            "name": normalize_text(item.get("property_name")),
            "value": normalize_text(item.get("property_value_raw")),
            "raw_value": normalize_text(item.get("property_value_raw")),
            "unit": normalize_text(item.get("property_unit_raw")),
        }
        for item in props
        if normalize_text(item.get("property_name")) and normalize_text(item.get("property_value_raw"))
    ]
    return json.dumps(converted, ensure_ascii=False)


def identity_object(row: dict[str, str]) -> dict[str, Any]:
    return {
        "document_key": normalize_text(row.get("key")),
        "doi": normalize_text(row.get("doi")),
        "formulation_candidate_id": normalize_text(row.get("formulation_id")),
        "raw_formulation_label": normalize_text(row.get("raw_formulation_label")),
        "parent_candidate_id": normalize_text(row.get("parent_instance_id")),
        "instance_kind": normalize_text(row.get("instance_kind")),
        "formulation_role": normalize_text(row.get("formulation_role")),
        "identity_confidence": normalize_text(row.get("instance_confidence")),
        "candidate_source": "legacy_stage2_lift",
        "change_descriptions": parse_json_string_list(row.get("change_descriptions")),
        "instance_context_tags": parse_json_string_list(row.get("instance_context_tags")),
        "change_context_tags": parse_json_string_list(row.get("change_context_tags")),
    }


def build_components_for_row(row: dict[str, str]) -> list[dict[str, Any]]:
    formulation_id = normalize_text(row.get("formulation_id"))
    components: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    for spec in COMPONENT_FIELD_SPECS:
        names = choose_raw_names(row, spec["name_field"])
        amount_expression = normalize_text(row.get(spec["amount_field"])) if spec["amount_field"] else ""
        parsed_value = normalize_text(row.get(spec["parsed_value_field"])) if spec["parsed_value_field"] else ""
        unit_raw = choose_unit_from_expression(amount_expression)
        for raw_name in names:
            add_or_update_component(
                components,
                ordered_keys,
                formulation_row_id=formulation_id,
                raw_name=raw_name,
                fallback_role=spec["fallback_role"],
                source_field=spec["source_field"],
                amount_expression_raw=amount_expression,
                value_raw=parsed_value,
                unit_raw=unit_raw,
                extra_context_text=normalize_text(row.get("raw_formulation_label")),
            )

    pva_amount = normalize_text(row.get("pva_conc_percent_value_text"))
    if pva_amount:
        add_or_update_component(
            components,
            ordered_keys,
            formulation_row_id=formulation_id,
            raw_name="PVA",
            fallback_role="surfactant",
            source_field="pva_conc_percent",
            amount_expression_raw=pva_amount,
            value_raw=normalize_text(row.get("pva_conc_percent_value")),
            unit_raw=choose_unit_from_expression(pva_amount),
            extra_context_text=normalize_text(row.get("raw_formulation_label")),
        )

    rows: list[dict[str, Any]] = []
    for idx, key in enumerate(ordered_keys, start=1):
        component = components[key]
        component_name_raw = normalize_text(component.get("component_name_raw"))
        component_role = normalize_text(component.get("component_role_normalized") or component.get("component_role_raw"))
        phase_context_raw, phase_code, phase_confidence = infer_phase_context(
            row,
            source_field=normalize_text(component.get("_source_field")),
            component_role=component_role,
            component_name_raw=component_name_raw,
        )
        rows.append(
            {
                "formulation_candidate_id": formulation_id,
                "component_id": f"{formulation_id}__component_{idx:02d}",
                "component_name_raw": component_name_raw,
                "component_role_raw": component_role,
                "phase_hint_raw": phase_code if phase_code != "unspecified" else normalize_text(phase_context_raw),
                "amount_expression_raw": normalize_text(component.get("amount_expression_raw")),
                "amount_kind_hint": normalize_text(component.get("amount_kind")),
                "parsed_value_raw": normalize_text(component.get("value_raw")),
                "parsed_unit_raw": normalize_text(component.get("unit_raw")),
                "component_properties_raw": component_properties_json(row, component_role, component_name_raw),
                "notes_raw": normalize_text(component.get("context_text")),
                "phase_confidence": phase_confidence,
            }
        )
    return rows


def build_phase_candidates(row: dict[str, str], components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formulation_id = normalize_text(row.get("formulation_id"))
    phases: dict[str, dict[str, Any]] = {}

    emulsion_structure = normalize_text(row.get("emulsion_structure"))
    if emulsion_structure:
        phase_tokens = [token.strip() for token in re.split(r"[|/]", emulsion_structure) if token.strip()]
        for idx, token in enumerate(phase_tokens, start=1):
            token_norm = normalize_text(token)
            key = normalize_token(token_norm) or f"phase_{idx:02d}"
            phases[key] = {
                "formulation_candidate_id": formulation_id,
                "phase_id": f"{formulation_id}__phase_{idx:02d}",
                "phase_code_raw": token_norm,
                "phase_role_hint": token_norm,
                "phase_order_hint": str(idx),
            }

    for component in components:
        phase_hint = normalize_text(component.get("phase_hint_raw"))
        if not phase_hint:
            continue
        key = normalize_token(phase_hint)
        if key in phases:
            continue
        phases[key] = {
            "formulation_candidate_id": formulation_id,
            "phase_id": f"{formulation_id}__phase_{len(phases) + 1:02d}",
            "phase_code_raw": phase_hint,
            "phase_role_hint": phase_hint,
            "phase_order_hint": str(len(phases) + 1),
        }

    return list(phases.values())


def build_process_candidates(row: dict[str, str]) -> list[dict[str, Any]]:
    formulation_id = normalize_text(row.get("formulation_id"))
    processes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, text in enumerate(
        [
            normalize_text(row.get("preparation_method")),
            normalize_text(row.get("emul_method_value_text")),
            normalize_text(row.get("emul_type_value_text")),
        ],
        start=1,
    ):
        if not text:
            continue
        token = normalize_token(text)
        if token in seen:
            continue
        seen.add(token)
        processes.append(
            {
                "formulation_candidate_id": formulation_id,
                "process_step_id": f"{formulation_id}__process_{len(processes) + 1:02d}",
                "process_name_raw": text,
                "process_step_order_hint": str(len(processes) + 1),
                "parameter_name_raw": "",
                "parameter_expression_raw": "",
            }
        )
    return processes


def build_variable_candidates(row: dict[str, str]) -> list[dict[str, Any]]:
    formulation_id = normalize_text(row.get("formulation_id"))
    change_tags = set(parse_json_string_list(row.get("change_context_tags")))
    instance_tags = set(parse_json_string_list(row.get("instance_context_tags")))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in CORE_FIELDS:
        value_text = normalize_text(row.get(f"{field}_value_text"))
        scope = normalize_text(row.get(f"{field}_scope"))
        if not value_text:
            continue
        if scope == "global_shared" and field not in change_tags:
            continue
        identity_signal = "yes" if field in change_tags or {"doe", "sweep"} & instance_tags else "unclear"
        factor_id = f"{formulation_id}__factor_{len(out) + 1:02d}"
        seen.add(field)
        out.append(
            {
                "formulation_candidate_id": formulation_id,
                "factor_id": factor_id,
                "factor_name_raw": field,
                "factor_expression_raw": value_text,
                "factor_entity_hint": "component" if field in {"la_ga_ratio", "polymer_mw_kDa", "plga_mass_mg", "surfactant_name", "surfactant_concentration_text", "pva_conc_percent", "organic_solvent", "drug_name", "drug_feed_amount_text"} else "process",
                "identity_defining_signal": identity_signal,
            }
        )

    for tag in sorted(change_tags - seen):
        out.append(
            {
                "formulation_candidate_id": formulation_id,
                "factor_id": f"{formulation_id}__factor_{len(out) + 1:02d}",
                "factor_name_raw": tag,
                "factor_expression_raw": "",
                "factor_entity_hint": "unknown",
                "identity_defining_signal": "yes",
            }
        )
    return out


def build_measurements(row: dict[str, str]) -> list[dict[str, Any]]:
    formulation_id = normalize_text(row.get("formulation_id"))
    measurements: list[dict[str, Any]] = []
    for field_name, (label, unit) in MEASUREMENT_FIELDS.items():
        value_text = normalize_text(row.get(f"{field_name}_value_text"))
        if not value_text:
            continue
        measurements.append(
            {
                "formulation_candidate_id": formulation_id,
                "measurement_id": f"{formulation_id}__measurement_{len(measurements) + 1:02d}",
                "measurement_name_raw": label,
                "measurement_value_raw": value_text,
                "measurement_unit_raw": unit,
                "statistic_qualifier_raw": "",
                "measurement_context_raw": normalize_text(row.get("evidence_section")),
            }
        )
    return measurements


def build_relation_cues(row: dict[str, str], variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formulation_id = normalize_text(row.get("formulation_id"))
    cues: list[dict[str, Any]] = []
    parent_id = normalize_text(row.get("parent_instance_id"))
    if parent_id:
        cues.append(
            {
                "cue_id": f"{formulation_id}__cue_{len(cues) + 1:02d}",
                "cue_type": "inherits_from",
                "source_object_ref": formulation_id,
                "target_object_ref": parent_id,
                "cue_text": normalize_text(row.get("change_descriptions")),
                "cue_confidence": "direct_from_legacy_stage2",
            }
        )
    for variable in variables:
        cues.append(
            {
                "cue_id": f"{formulation_id}__cue_{len(cues) + 1:02d}",
                "cue_type": "varies_on",
                "source_object_ref": formulation_id,
                "target_object_ref": normalize_text(variable.get("factor_id")),
                "cue_text": normalize_text(variable.get("factor_name_raw")),
                "cue_confidence": "deterministic_lift",
            }
        )
    return cues


def supporting_refs(row: dict[str, str]) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(row.get("supporting_evidence_refs"))
    return [item for item in ensure_list(parsed) if isinstance(item, dict)]


def build_evidence_handoffs(row: dict[str, str], components: list[dict[str, Any]], measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formulation_id = normalize_text(row.get("formulation_id"))
    handoffs: list[dict[str, Any]] = []

    def add_handoff(target_object_ref: str, target_field_name: str, region_type: str, locator: str, snippet: str, support_role: str = "direct_support") -> None:
        if not (target_object_ref and (region_type or locator or snippet)):
            return
        handoffs.append(
            {
                "handoff_id": f"{formulation_id}__handoff_{len(handoffs) + 1:02d}",
                "target_object_ref": target_object_ref,
                "target_field_name": target_field_name,
                "source_region_type": region_type,
                "source_locator_text": locator,
                "supporting_snippet": snippet,
                "support_role": support_role,
            }
        )

    add_handoff(
        formulation_id,
        "instance",
        normalize_text(row.get("instance_evidence_region_type")),
        normalize_text(row.get("evidence_section")),
        normalize_text(row.get("evidence_span_text")),
    )

    for field in CORE_FIELDS:
        value_text = normalize_text(row.get(f"{field}_value_text"))
        if not value_text:
            continue
        add_handoff(
            formulation_id,
            field,
            normalize_text(row.get(f"{field}_evidence_region_type")),
            normalize_text(row.get("evidence_section")),
            value_text,
        )

    for ref in supporting_refs(row):
        add_handoff(
            formulation_id,
            "supporting_evidence_refs",
            normalize_text(ref.get("region_type") or ref.get("evidence_region_type")),
            normalize_text(ref.get("section") or ref.get("evidence_section")),
            normalize_text(ref.get("span_text") or ref.get("evidence_span_text")),
            support_role="contextual_support",
        )

    for component in components:
        if normalize_text(component.get("amount_expression_raw")):
            add_handoff(
                normalize_text(component.get("component_id")),
                "amount_expression_raw",
                normalize_text(row.get("instance_evidence_region_type")),
                normalize_text(row.get("evidence_section")),
                normalize_text(component.get("amount_expression_raw")),
            )

    for measurement in measurements:
        add_handoff(
            normalize_text(measurement.get("measurement_id")),
            "measurement_value_raw",
            normalize_text(row.get("instance_evidence_region_type")),
            normalize_text(row.get("evidence_section")),
            normalize_text(measurement.get("measurement_value_raw")),
        )

    return handoffs


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_document_payload(document_key: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    first = rows[0]
    identities = [identity_object(row) for row in rows]
    components: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    variables: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    cues: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []

    for row in rows:
        row_components = build_components_for_row(row)
        row_phases = build_phase_candidates(row, row_components)
        row_processes = build_process_candidates(row)
        row_variables = build_variable_candidates(row)
        row_measurements = build_measurements(row)
        row_cues = build_relation_cues(row, row_variables)
        row_handoffs = build_evidence_handoffs(row, row_components, row_measurements)
        components.extend(row_components)
        phases.extend(row_phases)
        processes.extend(row_processes)
        variables.extend(row_variables)
        measurements.extend(row_measurements)
        cues.extend(row_cues)
        handoffs.extend(row_handoffs)

    return {
        "document_key": document_key,
        "doi": normalize_text(first.get("doi")),
        "model_name": normalize_text(first.get("model")),
        "source_schema": "weak_labels_v7pilot_r3_fixparse",
        "source_mode": "deterministic_lift_from_legacy_stage2",
        "replacement_emitter_status": "semantic_emitter_missing__legacy_lift_validation_only",
        "formulation_identity_candidates": identities,
        "component_candidates": components,
        "phase_candidates": phases,
        "process_step_candidates": processes,
        "variable_or_factor_candidates": variables,
        "measurement_candidates": measurements,
        "relation_cues": cues,
        "evidence_handoffs": handoffs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically lift legacy Stage2 weak labels into semantic-object payloads."
    )
    parser.add_argument("--legacy-weak-labels-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--paper-keys", nargs="*", default=[])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = load_rows(args.legacy_weak_labels_tsv)
    selected = set(args.paper_keys or [])
    if selected:
        rows = [row for row in rows if normalize_text(row.get("key")) in selected]
    if not rows:
        raise SystemExit("No legacy Stage2 rows selected for lifting.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(row.get("key"))].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    semantic_jsonl = args.out_dir / SEMANTIC_JSONL_NAME
    summary_rows: list[dict[str, Any]] = []

    with semantic_jsonl.open("w", encoding="utf-8") as handle:
        for key in sorted(grouped):
            document = build_document_payload(key, grouped[key])
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
            summary_rows.append(
                {
                    "document_key": key,
                    "doi": document["doi"],
                    "legacy_row_count": len(grouped[key]),
                    "identity_count": len(document["formulation_identity_candidates"]),
                    "component_count": len(document["component_candidates"]),
                    "phase_count": len(document["phase_candidates"]),
                    "process_count": len(document["process_step_candidates"]),
                    "variable_count": len(document["variable_or_factor_candidates"]),
                    "measurement_count": len(document["measurement_candidates"]),
                    "relation_cue_count": len(document["relation_cues"]),
                    "evidence_handoff_count": len(document["evidence_handoffs"]),
                }
            )

    write_tsv(
        args.out_dir / SUMMARY_TSV_NAME,
        summary_rows,
        [
            "document_key",
            "doi",
            "legacy_row_count",
            "identity_count",
            "component_count",
            "phase_count",
            "process_count",
            "variable_count",
            "measurement_count",
            "relation_cue_count",
            "evidence_handoff_count",
        ],
    )
    manifest = {
        "schema": "stage2_semantic_objects_v1",
        "source_surface": str(args.legacy_weak_labels_tsv),
        "source_mode": "deterministic_lift_from_legacy_stage2",
        "documents": len(grouped),
        "output_jsonl": str(semantic_jsonl),
        "output_summary_tsv": str(args.out_dir / SUMMARY_TSV_NAME),
        "validation_note": "True paper-driven semantic emitter is not yet implemented; this artifact is a deterministic lift used only for governed replacement-path validation.",
    }
    (args.out_dir / MANIFEST_JSON_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
