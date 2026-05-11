#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

SEMANTIC_JSONL_NAME = "semantic_stage2_objects_v1.jsonl"
SUMMARY_TSV_NAME = "semantic_stage2_object_summary_v1.tsv"
MANIFEST_JSON_NAME = "semantic_stage2_object_manifest_v1.json"

REPO_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_SEMANTIC_SOURCE_MODE = "governed_fallback_semantic_source"
FALLBACK_PROVENANCE = "governed_fallback_semantic_source"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append([normalize_text(cell) for cell in row])
    if rows and rows[0] and all(cell.isdigit() for cell in rows[0]):
        expected = [str(i) for i in range(len(rows[0]))]
        if rows[0] == expected:
            rows = rows[1:]
    return rows


def resolve_tables_dir(key: str, text_path: Path) -> Path | None:
    candidates = [
        text_path.parent.parent / "tables" / key,
        REPO_ROOT / "data" / "cleaned" / "content_goren_2025" / "tables" / key,
        REPO_ROOT / "data" / "cleaned" / "goren_2025" / "tables" / key,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_identity(
    *,
    key: str,
    doi: str,
    formulation_candidate_id: str,
    raw_formulation_label: str,
    parent_candidate_id: str = "",
    instance_kind: str = "new_formulation",
    formulation_role: str = "variant",
    change_descriptions: list[str] | None = None,
    instance_context_tags: list[str] | None = None,
    change_context_tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "document_key": key,
        "doi": doi,
        "formulation_candidate_id": formulation_candidate_id,
        "raw_formulation_label": raw_formulation_label,
        "parent_candidate_id": parent_candidate_id,
        "instance_kind": instance_kind,
        "formulation_role": formulation_role,
        "identity_confidence": "high",
        "candidate_source": "paper_driven_deterministic_semantic_emitter_v1",
        "stage2_semantic_source_mode": FALLBACK_SEMANTIC_SOURCE_MODE,
        "semantic_universe_authority": FALLBACK_PROVENANCE,
        "row_materialization_mode": FALLBACK_PROVENANCE,
        "semantic_scope_authority": FALLBACK_PROVENANCE,
        "semantic_scope_ref": f"governed_fallback_document_scope:{key}",
        "change_descriptions": change_descriptions or [],
        "instance_context_tags": instance_context_tags or [],
        "change_context_tags": change_context_tags or [],
    }


def make_component(
    *,
    formulation_id: str,
    component_id: str,
    name: str,
    role: str,
    amount: str = "",
    amount_kind: str = "",
    parsed_value: str = "",
    parsed_unit: str = "",
    properties: list[dict[str, str]] | None = None,
    phase_hint: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "formulation_candidate_id": formulation_id,
        "component_id": component_id,
        "component_name_raw": name,
        "component_role_raw": role,
        "phase_hint_raw": phase_hint,
        "amount_expression_raw": amount,
        "amount_kind_hint": amount_kind,
        "parsed_value_raw": parsed_value,
        "parsed_unit_raw": parsed_unit,
        "component_properties_raw": json.dumps(properties or [], ensure_ascii=False),
        "notes_raw": notes,
        "phase_confidence": "paper_local",
    }


def make_phase(formulation_id: str, phase_id: str, phase_code: str, order: int) -> dict[str, Any]:
    return {
        "formulation_candidate_id": formulation_id,
        "phase_id": phase_id,
        "phase_code_raw": phase_code,
        "phase_role_hint": phase_code,
        "phase_order_hint": str(order),
    }


def make_process(
    formulation_id: str,
    process_id: str,
    process_name: str,
    order: int = 1,
    parameter_name: str = "",
    parameter_expression: str = "",
) -> dict[str, Any]:
    return {
        "formulation_candidate_id": formulation_id,
        "process_step_id": process_id,
        "process_name_raw": process_name,
        "process_step_order_hint": str(order),
        "parameter_name_raw": parameter_name,
        "parameter_expression_raw": parameter_expression,
    }


def make_factor(
    formulation_id: str,
    factor_id: str,
    factor_name: str,
    factor_expression: str,
    entity_hint: str,
) -> dict[str, Any]:
    return {
        "formulation_candidate_id": formulation_id,
        "factor_id": factor_id,
        "factor_name_raw": factor_name,
        "factor_expression_raw": factor_expression,
        "factor_entity_hint": entity_hint,
        "identity_defining_signal": "yes",
    }


def make_measurement(
    formulation_id: str,
    measurement_id: str,
    name: str,
    value: str,
    unit: str = "",
    statistic: str = "",
    context: str = "",
) -> dict[str, Any]:
    return {
        "formulation_candidate_id": formulation_id,
        "measurement_id": measurement_id,
        "measurement_name_raw": name,
        "measurement_value_raw": value,
        "measurement_unit_raw": unit,
        "statistic_qualifier_raw": statistic,
        "measurement_context_raw": context,
    }


def make_relation(cue_id: str, cue_type: str, source_ref: str, target_ref: str, cue_text: str) -> dict[str, Any]:
    return {
        "cue_id": cue_id,
        "cue_type": cue_type,
        "source_object_ref": source_ref,
        "target_object_ref": target_ref,
        "cue_text": cue_text,
        "cue_confidence": "high",
    }


def add_handoff(
    handoffs: list[dict[str, Any]],
    *,
    target_object_ref: str,
    target_field_name: str,
    source_region_type: str,
    source_locator_text: str,
    supporting_snippet: str,
    support_role: str = "direct_support",
) -> None:
    handoffs.append(
        {
            "handoff_id": f"{target_object_ref}__handoff_{len(handoffs) + 1:02d}",
            "target_object_ref": target_object_ref,
            "target_field_name": target_field_name,
            "source_region_type": source_region_type,
            "source_locator_text": source_locator_text,
            "supporting_snippet": supporting_snippet,
            "support_role": support_role,
        }
    )


def table_locator(tables_dir: Path | None, filename: str) -> str:
    if not tables_dir:
        return filename
    return str((tables_dir / filename).relative_to(REPO_ROOT)).replace("\\", "/")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "-", normalize_text(value)).strip("-")
    return token or "item"


def empty_object_lists() -> tuple[list[dict[str, Any]], ...]:
    return [], [], [], [], [], [], [], []


def component_spec(
    name: str,
    role: str,
    amount: str = "",
    amount_kind: str = "",
    parsed_value: str = "",
    parsed_unit: str = "",
    properties: list[dict[str, str]] | None = None,
    phase_hint: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "amount": amount,
        "amount_kind": amount_kind,
        "parsed_value": parsed_value,
        "parsed_unit": parsed_unit,
        "properties": properties or [],
        "phase_hint": phase_hint,
        "notes": notes,
    }


def property_spec(name: str, value: str = "", raw_value: str = "") -> dict[str, str]:
    return {
        "name": name,
        "value": value,
        "raw_value": raw_value,
    }


def target_resomer_ratio_property(code: str, ratio: str) -> dict[str, str]:
    return property_spec(
        "la_ga_ratio",
        ratio,
        f"{code} article-native PLGA grade carried through the target-scoped deterministic Resomer ratio rule",
    )


def target_resomer_grade_property(code: str) -> dict[str, str]:
    return property_spec(
        "molecular_weight",
        f"{code} (PLGA grade)",
        f"{code} article-native PLGA grade",
    )


def measurement_spec(name: str, value: str, unit: str = "", context: str = "") -> dict[str, str]:
    return {"name": name, "value": value, "unit": unit, "context": context}


def factor_spec(name: str, expression: str, entity_hint: str = "factor") -> dict[str, str]:
    return {"name": name, "expression": expression, "entity_hint": entity_hint}


def append_specs_to_lists(
    *,
    key: str,
    doi: str,
    formulation_id: str,
    label: str,
    identities: list[dict[str, Any]],
    components: list[dict[str, Any]],
    processes: list[dict[str, Any]],
    factors: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    handoffs: list[dict[str, Any]],
    components_spec: list[dict[str, Any]],
    factors_spec: list[dict[str, str]] | None,
    measurements_spec: list[dict[str, str]] | None,
    parent_candidate_id: str = "",
    instance_kind: str = "new_formulation",
    formulation_role: str = "variant",
    change_descriptions: list[str] | None = None,
    instance_context_tags: list[str] | None = None,
    change_context_tags: list[str] | None = None,
    process_name: str = "",
    process_parameter_name: str = "",
    process_parameter_expression: str = "",
    source_region_type: str = "text_span",
    source_locator_text: str = "",
    supporting_snippet: str = "",
) -> None:
    identities.append(
        make_identity(
            key=key,
            doi=doi,
            formulation_candidate_id=formulation_id,
            raw_formulation_label=label,
            parent_candidate_id=parent_candidate_id,
            instance_kind=instance_kind,
            formulation_role=formulation_role,
            change_descriptions=change_descriptions or [],
            instance_context_tags=instance_context_tags or [],
            change_context_tags=change_context_tags or [],
        )
    )
    for idx, spec in enumerate(components_spec, start=1):
        components.append(
            make_component(
                formulation_id=formulation_id,
                component_id=f"{formulation_id}__component_{idx:02d}",
                name=spec["name"],
                role=spec["role"],
                amount=spec.get("amount", ""),
                amount_kind=spec.get("amount_kind", ""),
                parsed_value=spec.get("parsed_value", ""),
                parsed_unit=spec.get("parsed_unit", ""),
                properties=spec.get("properties", []),
                phase_hint=spec.get("phase_hint", ""),
                notes=spec.get("notes", ""),
            )
        )
    if process_name:
        processes.append(
            make_process(
                formulation_id,
                f"{formulation_id}__process_01",
                process_name,
                1,
                process_parameter_name,
                process_parameter_expression,
            )
        )
    for idx, spec in enumerate(factors_spec or [], start=1):
        factors.append(
            make_factor(
                formulation_id,
                f"{formulation_id}__factor_{idx:02d}",
                spec["name"],
                spec["expression"],
                spec.get("entity_hint", "factor"),
            )
        )
    for idx, spec in enumerate(measurements_spec or [], start=1):
        measurements.append(
            make_measurement(
                formulation_id,
                f"{formulation_id}__measurement_{idx:02d}",
                spec["name"],
                spec["value"],
                spec.get("unit", ""),
                "",
                spec.get("context", ""),
            )
        )
    add_handoff(
        handoffs,
        target_object_ref=formulation_id,
        target_field_name="instance",
        source_region_type=source_region_type,
        source_locator_text=source_locator_text,
        supporting_snippet=supporting_snippet or label,
    )
    if process_name:
        add_handoff(
            handoffs,
            target_object_ref=formulation_id,
            target_field_name="preparation_method",
            source_region_type=source_region_type,
            source_locator_text=source_locator_text,
            supporting_snippet=process_name if not process_parameter_expression else f"{process_name}; {process_parameter_expression}",
        )
    if parent_candidate_id:
        relations.append(
            make_relation(
                f"{formulation_id}__cue_01",
                "inherits_from",
                formulation_id,
                parent_candidate_id,
                change_descriptions[0] if change_descriptions else f"{label} derived from {parent_candidate_id}.",
            )
        )


def finalize_document(
    *,
    key: str,
    doi: str,
    text_path: Path,
    tables_dir: Path | None,
    source_notes: list[str],
    identities: list[dict[str, Any]],
    components: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    processes: list[dict[str, Any]],
    factors: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    handoffs: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "document_key": key,
        "doi": doi,
        "model_name": "paper_driven_deterministic_semantic_emitter_v1",
        "source_schema": "stage2_replacement_semantic_contract_v1",
        "source_mode": "paper_driven_deterministic_semantic_emitter",
        "replacement_emitter_status": "true_paper_driven_semantic_emitter_present__diagnostic_scope",
        "source_text_path": str(text_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_notes": source_notes,
        "formulation_identity_candidates": identities,
        "component_candidates": components,
        "phase_candidates": phases,
        "process_step_candidates": processes,
        "variable_or_factor_candidates": factors,
        "measurement_candidates": measurements,
        "relation_cues": relations,
        "evidence_handoffs": handoffs,
    }
    if tables_dir and tables_dir.exists():
        payload["source_tables_dir"] = str(tables_dir.relative_to(REPO_ROOT)).replace("\\", "/")
    return payload


def table_rows_with_prefix(table_path: Path, prefix_pattern: str) -> list[tuple[str, list[str]]]:
    rows = read_csv_rows(table_path)
    matched: list[tuple[str, list[str]]] = []
    for row in rows[1:]:
        if not row:
            continue
        label = normalize_text(row[0])
        if re.fullmatch(prefix_pattern, label, flags=re.IGNORECASE):
            matched.append((label, row))
    return matched


def build_ufxx9wxe_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    table_path = None
    if tables_dir:
        for name in ["UFXX9WXE__table_13__pdf_table.csv", "UFXX9WXE__table_14__pdf_table.csv"]:
            candidate = tables_dir / name
            if candidate.exists():
                table_path = candidate
                break
    if table_path is None:
        raise FileNotFoundError("UFXX9WXE DOE table asset not found.")

    rows = read_csv_rows(table_path)
    data_rows = [row for row in rows if row and re.fullmatch(r"\d+\s*\.", row[0] or "")]

    identities: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    factors: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []

    methods_locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")
    methods_snippet = (
        "Lzp-PLGA-NPs were prepared by nanoprecipitation using PLGA as polymer, "
        "poloxamer 407 as surfactant, acetone as organic solvent, and lorazepam as drug."
    )
    shared_polymer_props = [
        {"name": "la_ga_ratio", "value": "50:50", "raw_value": "PLGA 50:50"},
        {"name": "molecular_weight", "value": "30,000-60,000", "raw_value": "molecular weight 30,000-60,000"},
    ]

    for row in data_rows:
        formulation_no = int(re.search(r"\d+", row[0]).group(0))
        plga = normalize_text(row[1])
        poloxamer = normalize_text(row[2])
        phase_ratio = normalize_text(row[3])
        drug_conc = normalize_text(row[4])
        size_raw = normalize_text(row[5])
        ee_raw = normalize_text(row[6])
        pdi_raw = normalize_text(row[7])
        formulation_id = f"{key}_DOE_Row_{formulation_no:02d}"
        raw_label = f"DOE row {formulation_no}"
        snippet = " | ".join(cell for cell in row[:8] if cell)
        locator = table_locator(tables_dir, table_path.name)

        identities.append(
            make_identity(
                key=key,
                doi=doi,
                formulation_candidate_id=formulation_id,
                raw_formulation_label=raw_label,
                instance_kind="new_formulation",
                formulation_role="variant",
                change_descriptions=[
                    f"polymer concentration = {plga} mg/mL",
                    f"surfactant concentration = {poloxamer} mg/mL",
                    f"aqueous/organic phase ratio = {phase_ratio}",
                    f"drug concentration = {drug_conc} mg/mL",
                ],
                instance_context_tags=["doe", "box_behnken", "numbered_row"],
                change_context_tags=["factor_grid"],
            )
        )
        components.extend(
            [
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_01",
                    name="PLGA",
                    role="polymer",
                    amount=f"{plga} mg/mL",
                    amount_kind="concentration",
                    parsed_value=plga,
                    parsed_unit="mg/mL",
                    properties=shared_polymer_props,
                    phase_hint="O",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_02",
                    name="Poloxamer 407",
                    role="surfactant",
                    amount=f"{poloxamer} mg/mL",
                    amount_kind="concentration",
                    parsed_value=poloxamer,
                    parsed_unit="mg/mL",
                    phase_hint="W",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_03",
                    name="Lorazepam",
                    role="drug",
                    amount=f"{drug_conc} mg/mL",
                    amount_kind="concentration",
                    parsed_value=drug_conc,
                    parsed_unit="mg/mL",
                    phase_hint="O",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_04",
                    name="Acetone",
                    role="organic_solvent",
                    phase_hint="O",
                ),
            ]
        )
        phases.extend(
            [
                make_phase(formulation_id, f"{formulation_id}__phase_01", "O", 1),
                make_phase(formulation_id, f"{formulation_id}__phase_02", "W", 2),
            ]
        )
        processes.append(make_process(formulation_id, f"{formulation_id}__process_01", "nanoprecipitation", 1))
        factors.extend(
            [
                make_factor(formulation_id, f"{formulation_id}__factor_01", "polymer concentration", f"{plga} mg/mL", "component"),
                make_factor(formulation_id, f"{formulation_id}__factor_02", "surfactant concentration", f"{poloxamer} mg/mL", "component"),
                make_factor(formulation_id, f"{formulation_id}__factor_03", "aqueous/organic phase ratio", phase_ratio, "phase"),
                make_factor(formulation_id, f"{formulation_id}__factor_04", "drug concentration", f"{drug_conc} mg/mL", "component"),
            ]
        )
        measurements.extend(
            [
                make_measurement(formulation_id, f"{formulation_id}__measurement_01", "size", size_raw, "nm", "mean_sd"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_02", "encapsulation efficiency", ee_raw, "%", "mean_sd"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_03", "pdi", pdi_raw, "", "mean_sd"),
            ]
        )
        add_handoff(
            handoffs,
            target_object_ref=formulation_id,
            target_field_name="instance",
            source_region_type="table_row",
            source_locator_text=f"{locator} row {formulation_no}",
            supporting_snippet=snippet,
        )
        add_handoff(
            handoffs,
            target_object_ref=formulation_id,
            target_field_name="preparation_method",
            source_region_type="methods_sentence",
            source_locator_text=methods_locator,
            supporting_snippet=methods_snippet,
        )
        for target_field, value in [("size", size_raw), ("encapsulation efficiency", ee_raw), ("pdi", pdi_raw)]:
            add_handoff(
                handoffs,
                target_object_ref=formulation_id,
                target_field_name=target_field,
                source_region_type="table_cell",
                source_locator_text=f"{locator} row {formulation_no}",
                supporting_snippet=value,
            )

    return {
        "document_key": key,
        "doi": doi,
        "model_name": "paper_driven_deterministic_semantic_emitter_v1",
        "source_schema": "stage2_replacement_semantic_contract_v1",
        "source_mode": "paper_driven_deterministic_semantic_emitter",
        "replacement_emitter_status": "true_paper_driven_semantic_emitter_present__diagnostic_scope",
        "source_text_path": methods_locator,
        "source_tables_dir": str(tables_dir.relative_to(REPO_ROOT)).replace("\\", "/") if tables_dir else "",
        "source_notes": ["UFXX9WXE emitted from cleaned DOE table plus methods text."],
        "formulation_identity_candidates": identities,
        "component_candidates": components,
        "phase_candidates": phases,
        "process_step_candidates": processes,
        "variable_or_factor_candidates": factors,
        "measurement_candidates": measurements,
        "relation_cues": relations,
        "evidence_handoffs": handoffs,
    }


def build_bxcv5xwb_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")

    identities: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    factors: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []

    families = [
        ("PLGA-NP-KGN-01", "KGN-loaded PLGA nanoparticles", "", "baseline", "PLGA", [], ["Hydrophobic PLGA nanoparticle baseline."], "~167 nm", "~62%", "~47%", "~-33.1 mV"),
        ("PLGA-PEG-NP-KGN-01", "KGN-loaded PLGA-PEG nanoparticles", "PLGA-NP-KGN-01", "variant", "PLGA-PEG", [], ["PLGA was conjugated to PEG-bis-amine to form PLGA-PEG."], "~297 nm", "~71%", "~13-16%", "~11.2 mV"),
        ("PLGA-PEG-HA-NP-KGN-01", "KGN-loaded PLGA-PEG-HA nanoparticles", "PLGA-PEG-NP-KGN-01", "variant", "PLGA-PEG", [("HA", "additive", "3:1 w/w HA:nanoparticle", "ratio")], ["Activated HA was added to KGN-loaded PLGA-PEG nanoparticles."], "~507 nm", "~55%", "~13-16%", "~-28.5 mV"),
    ]
    methods_snippet = (
        "All nanoparticles were prepared using nanoprecipitation. PLGA or PLGA-PEG "
        "(50 mg) was dissolved in ACN (10 mL) and added to 0.2% w/v PVA in water "
        "(100 mL). KGN (5 mg) was included in the polymer solution prior to precipitation."
    )

    for formulation_id, label, parent_id, role, polymer, extras, changes, size, ee, loading, zeta in families:
        identities.append(
            make_identity(
                key=key,
                doi=doi,
                formulation_candidate_id=formulation_id,
                raw_formulation_label=label,
                parent_candidate_id=parent_id,
                instance_kind="new_formulation" if not parent_id else "variant_formulation",
                formulation_role=role,
                change_descriptions=changes,
                instance_context_tags=["family_variant"],
                change_context_tags=["surface_modification"],
            )
        )
        components.extend(
            [
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_01",
                    name=polymer,
                    role="polymer",
                    amount="50 mg",
                    amount_kind="mass",
                    parsed_value="50",
                    parsed_unit="mg",
                    properties=[
                        {"name": "la_ga_ratio", "value": "1:1 d,l-lactic to glycolic acid", "raw_value": "1:1 d,l-lactic to glycolic acid"},
                        {"name": "molecular_weight", "value": "15.1", "raw_value": "MW 15.1 kDa"},
                    ],
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_02",
                    name="PVA",
                    role="surfactant",
                    amount="0.2% w/v",
                    amount_kind="concentration_percent_wv",
                    parsed_value="0.2",
                    parsed_unit="% w/v",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_03",
                    name="ACN",
                    role="organic_solvent",
                    amount="10 mL",
                    amount_kind="volume",
                    parsed_value="10",
                    parsed_unit="mL",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_04",
                    name="KGN",
                    role="drug",
                    amount="5 mg",
                    amount_kind="mass",
                    parsed_value="5",
                    parsed_unit="mg",
                ),
            ]
        )
        next_idx = 5
        for name, component_role, amount, amount_kind in extras:
            components.append(
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_{next_idx:02d}",
                    name=name,
                    role=component_role,
                    amount=amount,
                    amount_kind=amount_kind,
                )
            )
            next_idx += 1
        processes.append(make_process(formulation_id, f"{formulation_id}__process_01", "nanoprecipitation", 1))
        if "HA" in label:
            processes.append(make_process(formulation_id, f"{formulation_id}__process_02", "HA conjugation", 2, "HA:nanoparticle ratio", "3:1 w/w"))
        measurements.extend(
            [
                make_measurement(formulation_id, f"{formulation_id}__measurement_01", "size", size, "nm"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_02", "encapsulation efficiency", ee, "%"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_03", "loading content", loading, "%"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_04", "zeta potential", zeta, "mV"),
            ]
        )
        factors.extend(
            [
                make_factor(formulation_id, f"{formulation_id}__factor_01", "polymer type", polymer, "component"),
                make_factor(formulation_id, f"{formulation_id}__factor_02", "surface chemistry", label.replace("KGN-loaded ", ""), "component"),
            ]
        )
        if parent_id:
            relations.append(make_relation(f"{formulation_id}__cue_01", "inherits_from", formulation_id, parent_id, changes[0]))
        add_handoff(handoffs, target_object_ref=formulation_id, target_field_name="instance", source_region_type="results_sentence", source_locator_text=locator, supporting_snippet=label)
        add_handoff(handoffs, target_object_ref=formulation_id, target_field_name="preparation_method", source_region_type="methods_sentence", source_locator_text=locator, supporting_snippet=methods_snippet)
        for target_field, snippet in [("size", size), ("encapsulation efficiency", ee), ("loading content", loading), ("zeta potential", zeta)]:
            add_handoff(handoffs, target_object_ref=formulation_id, target_field_name=target_field, source_region_type="results_sentence", source_locator_text=locator, supporting_snippet=snippet)

    return {
        "document_key": key,
        "doi": doi,
        "model_name": "paper_driven_deterministic_semantic_emitter_v1",
        "source_schema": "stage2_replacement_semantic_contract_v1",
        "source_mode": "paper_driven_deterministic_semantic_emitter",
        "replacement_emitter_status": "true_paper_driven_semantic_emitter_present__diagnostic_scope",
        "source_text_path": locator,
        "source_tables_dir": str(tables_dir.relative_to(REPO_ROOT)).replace("\\", "/") if tables_dir else "",
        "source_notes": ["BXCV5XWB emitted from cleaned article text; extracted table CSVs were low-value or mismatched to the article."],
        "formulation_identity_candidates": identities,
        "component_candidates": components,
        "phase_candidates": phases,
        "process_step_candidates": processes,
        "variable_or_factor_candidates": factors,
        "measurement_candidates": measurements,
        "relation_cues": relations,
        "evidence_handoffs": handoffs,
    }


def build_l3h2rs2h_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    text = text_path.read_text(encoding="utf-8", errors="replace")
    if not tables_dir:
        raise FileNotFoundError("L3H2RS2H requires governed table assets.")

    table1_path = tables_dir / "L3H2RS2H__table_05__pdf_table.csv"
    table3_path = tables_dir / "L3H2RS2H__table_07__pdf_table.csv"
    if not table1_path.exists() or not table3_path.exists():
        raise FileNotFoundError("L3H2RS2H Table 1 or Table 3 asset missing.")

    table5_match = re.search(
        r"Table 5\s+Mean diameter.*?3-MeOXAN-loaded\s+nanocapsules.*?K41\.8G5\.4 \[5\]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    table5_block = normalize_text(table5_match.group(0)) if table5_match else ""
    table1_locator = table_locator(tables_dir, table1_path.name)
    table3_locator = table_locator(tables_dir, table3_path.name)
    text_locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")

    identities: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    factors: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []

    nanosphere_parent = "Nanoparticle-Empty-Nanospheres-01"
    nanocapsule_parent = "Nanoparticle-Empty-Nanocapsules-01"
    nanocapsule_06_parent = "Nanoparticle-Empty-Nanocapsules-06mL-01"

    def add_shared_nanosphere_components(formulation_id: str, drug_name: str = "", theoretical_conc: str = "") -> None:
        components.extend(
            [
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_01",
                    name="PLGA",
                    role="polymer",
                    amount="63 mg",
                    amount_kind="mass",
                    parsed_value="63",
                    parsed_unit="mg",
                    properties=[
                        {"name": "la_ga_ratio", "value": "50:50", "raw_value": "PLGA (50:50)"},
                        {"name": "molecular_weight", "value": "50-75", "raw_value": "MW 50 000-75 000"},
                    ],
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_02",
                    name="Pluronic F-68",
                    role="surfactant",
                    amount="0.25% (w/v)",
                    amount_kind="concentration_percent_wv",
                    parsed_value="0.25",
                    parsed_unit="% (w/v)",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_03",
                    name="Acetone",
                    role="organic_solvent",
                    amount="10 mL",
                    amount_kind="volume",
                    parsed_value="10",
                    parsed_unit="mL",
                ),
            ]
        )
        if drug_name:
            components.append(
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_04",
                    name=drug_name,
                    role="drug",
                    amount=f"{theoretical_conc} mg/mL (theoretical)",
                    amount_kind="concentration",
                    parsed_value=theoretical_conc,
                    parsed_unit="mg/mL",
                )
            )

    def add_shared_nanocapsule_components(formulation_id: str, drug_name: str = "", theoretical_conc: str = "", oil_volume: str = "0.5 mL") -> None:
        components.extend(
            [
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_01",
                    name="PLGA",
                    role="polymer",
                    amount="50 mg",
                    amount_kind="mass",
                    parsed_value="50",
                    parsed_unit="mg",
                    properties=[
                        {"name": "la_ga_ratio", "value": "50:50", "raw_value": "PLGA (50:50)"},
                        {"name": "molecular_weight", "value": "50-75", "raw_value": "MW 50 000-75 000"},
                    ],
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_02",
                    name="Soybean lecithin",
                    role="surfactant",
                    amount="100 mg",
                    amount_kind="mass",
                    parsed_value="100",
                    parsed_unit="mg",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_03",
                    name="Pluronic F-68",
                    role="surfactant",
                    amount="0.5% (w/v)",
                    amount_kind="concentration_percent_wv",
                    parsed_value="0.5",
                    parsed_unit="% (w/v)",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_04",
                    name="Myritol 318",
                    role="additive",
                    amount=oil_volume,
                    amount_kind="volume",
                    parsed_value=oil_volume.split()[0],
                    parsed_unit="mL",
                ),
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_05",
                    name="Acetone",
                    role="organic_solvent",
                    amount="10 mL",
                    amount_kind="volume",
                    parsed_value="10",
                    parsed_unit="mL",
                ),
            ]
        )
        if drug_name:
            components.append(
                make_component(
                    formulation_id=formulation_id,
                    component_id=f"{formulation_id}__component_06",
                    name=drug_name,
                    role="drug",
                    amount=f"{theoretical_conc} mg/mL (theoretical)",
                    amount_kind="concentration",
                    parsed_value=theoretical_conc,
                    parsed_unit="mg/mL",
                )
            )

    base_identities = [
        (nanosphere_parent, "Empty nanospheres", "Prepared according to the same procedure but omitting the xanthones in the organic phase."),
        (nanocapsule_parent, "Empty nanocapsules", "Prepared according to the same procedure but omitting the xanthones in the organic phase."),
        (nanocapsule_06_parent, "Empty nanocapsules (0.6 mL Myritol 318 and without xanthones)", "Table 5 empty nanocapsule stability formulation with 0.6 mL Myritol 318."),
    ]
    for formulation_id, label, note in base_identities:
        identities.append(
            make_identity(
                key=key,
                doi=doi,
                formulation_candidate_id=formulation_id,
                raw_formulation_label=label,
                instance_kind="new_formulation",
                formulation_role="baseline",
                change_descriptions=[note],
                instance_context_tags=["baseline_control"],
            )
        )
    add_shared_nanosphere_components(nanosphere_parent)
    add_shared_nanocapsule_components(nanocapsule_parent)
    add_shared_nanocapsule_components(nanocapsule_06_parent, oil_volume="0.6 mL")
    processes.extend(
        [
            make_process(nanosphere_parent, f"{nanosphere_parent}__process_01", "solvent displacement", 1),
            make_process(nanocapsule_parent, f"{nanocapsule_parent}__process_01", "interfacial polymer deposition", 1),
            make_process(nanocapsule_06_parent, f"{nanocapsule_06_parent}__process_01", "interfacial polymer deposition", 1, "oil volume", "0.6 mL Myritol 318"),
        ]
    )
    add_handoff(handoffs, target_object_ref=nanosphere_parent, target_field_name="instance", source_region_type="methods_sentence", source_locator_text=text_locator, supporting_snippet="Empty nanospheres were prepared according to the same procedure but omitting the xanthones in the organic phase.")
    add_handoff(handoffs, target_object_ref=nanocapsule_parent, target_field_name="instance", source_region_type="methods_sentence", source_locator_text=text_locator, supporting_snippet="Empty nanocapsules were prepared according to the same procedure but omitting the xanthones in the organic phase.")
    add_handoff(handoffs, target_object_ref=nanocapsule_06_parent, target_field_name="instance", source_region_type="table_block", source_locator_text=text_locator, supporting_snippet=table5_block)

    for drug_name, prefix in [("XAN", "XAN"), ("3-MeOXAN", "3MeOXAN")]:
        for theoretical in ["50", "60", "70", "80"]:
            formulation_id = f"Nanosphere-{prefix}-{theoretical}mgML-Theoretical"
            role = "optimized" if theoretical == "60" else "variant"
            identities.append(
                make_identity(
                    key=key,
                    doi=doi,
                    formulation_candidate_id=formulation_id,
                    raw_formulation_label=f"{drug_name} nanospheres (Theoretical concentration {theoretical} mg/mL)",
                    parent_candidate_id=nanosphere_parent,
                    instance_kind="variant_formulation",
                    formulation_role=role,
                    change_descriptions=[f"{drug_name} theoretical concentration {theoretical} mg/mL from Table 1."],
                    instance_context_tags=["table1_nanosphere"],
                    change_context_tags=["drug_loading_sweep"],
                )
            )
            add_shared_nanosphere_components(formulation_id, drug_name, theoretical)
            processes.append(make_process(formulation_id, f"{formulation_id}__process_01", "solvent displacement", 1))
            factors.append(make_factor(formulation_id, f"{formulation_id}__factor_01", "theoretical concentration", f"{theoretical} mg/mL", "component"))
            if theoretical == "60":
                ee = "33.0G4.1" if drug_name == "XAN" else "41.5G7.6"
                measurements.append(make_measurement(formulation_id, f"{formulation_id}__measurement_01", "encapsulation efficiency", ee, "%"))
            elif theoretical in {"70", "80"}:
                measurements.append(make_measurement(formulation_id, f"{formulation_id}__measurement_01", "encapsulation efficiency", "ND", "%", context="crystals observed"))
            add_handoff(handoffs, target_object_ref=formulation_id, target_field_name="instance", source_region_type="table_block", source_locator_text=table1_locator, supporting_snippet=f"{drug_name} theoretical concentration {theoretical} mg/mL")
            relations.append(make_relation(f"{formulation_id}__cue_01", "inherits_from", formulation_id, nanosphere_parent, f"{drug_name} theoretical concentration {theoretical} mg/mL from Table 1."))

    for theoretical in ["200", "400", "600", "700", "800"]:
        formulation_id = f"Nanocapsule-XAN-{theoretical}mgML-Theoretical"
        role = "optimized" if theoretical == "600" else "variant"
        final_conc = {"200": "178G21", "400": "342G18", "600": "529G57", "700": "Crystals of XAN", "800": "Crystals of XAN"}[theoretical]
        ee = {"200": "89G11", "400": "85G5", "600": "88G9", "700": "ND", "800": "ND"}[theoretical]
        identities.append(
            make_identity(
                key=key,
                doi=doi,
                formulation_candidate_id=formulation_id,
                raw_formulation_label=f"XAN nanocapsules (Theoretical concentration {theoretical} mg/mL)",
                parent_candidate_id=nanocapsule_parent,
                instance_kind="variant_formulation",
                formulation_role=role,
                change_descriptions=[f"XAN theoretical concentration {theoretical} mg/mL from Table 3."],
                instance_context_tags=["table3_nanocapsule"],
                change_context_tags=["drug_loading_sweep"],
            )
        )
        add_shared_nanocapsule_components(formulation_id, "XAN", theoretical, "0.5 mL")
        processes.append(make_process(formulation_id, f"{formulation_id}__process_01", "interfacial polymer deposition", 1))
        factors.append(make_factor(formulation_id, f"{formulation_id}__factor_01", "theoretical concentration", f"{theoretical} mg/mL", "component"))
        measurements.extend(
            [
                make_measurement(formulation_id, f"{formulation_id}__measurement_01", "final concentration in dispersion", final_conc, "mg/mL"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_02", "encapsulation efficiency", ee, "%"),
            ]
        )
        add_handoff(handoffs, target_object_ref=formulation_id, target_field_name="instance", source_region_type="table_block", source_locator_text=table3_locator, supporting_snippet=f"XAN nanocapsules {theoretical} mg/mL theoretical; final concentration {final_conc}; EE {ee}")
        relations.append(make_relation(f"{formulation_id}__cue_01", "inherits_from", formulation_id, nanocapsule_parent, f"XAN theoretical concentration {theoretical} mg/mL from Table 3."))

    for theoretical in ["1000", "1200", "1400", "1600"]:
        formulation_id = f"Nanocapsule-3MeOXAN-{theoretical}mgML-Theoretical"
        role = "optimized" if theoretical == "1400" else "variant"
        final_conc = {"1000": "887G51", "1200": "918G9", "1400": "1162G80", "1600": "Crystals of 3-MeOXAN"}[theoretical]
        ee = {"1000": "89G5", "1200": "77G1", "1400": "83G6", "1600": "ND"}[theoretical]
        identities.append(
            make_identity(
                key=key,
                doi=doi,
                formulation_candidate_id=formulation_id,
                raw_formulation_label=f"3-MeOXAN nanocapsules (Theoretical concentration {theoretical} mg/mL)",
                parent_candidate_id=nanocapsule_parent,
                instance_kind="variant_formulation",
                formulation_role=role,
                change_descriptions=[f"3-MeOXAN theoretical concentration {theoretical} mg/mL from Table 3."],
                instance_context_tags=["table3_nanocapsule"],
                change_context_tags=["drug_loading_sweep"],
            )
        )
        add_shared_nanocapsule_components(formulation_id, "3-MeOXAN", theoretical, "0.5 mL")
        processes.append(make_process(formulation_id, f"{formulation_id}__process_01", "interfacial polymer deposition", 1))
        factors.append(make_factor(formulation_id, f"{formulation_id}__factor_01", "theoretical concentration", f"{theoretical} mg/mL", "component"))
        measurements.extend(
            [
                make_measurement(formulation_id, f"{formulation_id}__measurement_01", "final concentration in dispersion", final_conc, "mg/mL"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_02", "encapsulation efficiency", ee, "%"),
            ]
        )
        add_handoff(handoffs, target_object_ref=formulation_id, target_field_name="instance", source_region_type="table_block", source_locator_text=table3_locator, supporting_snippet=f"3-MeOXAN nanocapsules {theoretical} mg/mL theoretical; final concentration {final_conc}; EE {ee}")
        relations.append(make_relation(f"{formulation_id}__cue_01", "inherits_from", formulation_id, nanocapsule_parent, f"3-MeOXAN theoretical concentration {theoretical} mg/mL from Table 3."))

    for drug_name, formulation_id, theoretical, final_conc, ee, size, pdi, zeta in [
        ("XAN", "Nanocapsule-XAN-1440mgML-Theoretical-Table5", "1440", "1173G100", "82G7", "273G18", "0.48G0.05", "K36.4G9.3"),
        ("3-MeOXAN", "Nanocapsule-3MeOXAN-3360mgML-Theoretical-Table5", "3360", "2780G238", "83G7", "271G16", "0.43G0.03", "K41.8G5.4"),
    ]:
        identities.append(
            make_identity(
                key=key,
                doi=doi,
                formulation_candidate_id=formulation_id,
                raw_formulation_label=f"{drug_name}-loaded nanocapsules (0.6 mL Myritol 318, {drug_name} theoretical concentration of {theoretical} mg/mL)",
                parent_candidate_id=nanocapsule_06_parent,
                instance_kind="variant_formulation",
                formulation_role="optimized",
                change_descriptions=[f"Table 5 nanocapsule with 0.6 mL Myritol 318 and theoretical concentration {theoretical} mg/mL."],
                instance_context_tags=["table5_nanocapsule"],
                change_context_tags=["oil_volume_variant", "drug_loading_sweep"],
            )
        )
        add_shared_nanocapsule_components(formulation_id, drug_name, theoretical, "0.6 mL")
        processes.append(make_process(formulation_id, f"{formulation_id}__process_01", "interfacial polymer deposition", 1, "oil volume", "0.6 mL Myritol 318"))
        factors.extend(
            [
                make_factor(formulation_id, f"{formulation_id}__factor_01", "theoretical concentration", f"{theoretical} mg/mL", "component"),
                make_factor(formulation_id, f"{formulation_id}__factor_02", "Myritol 318 volume", "0.6 mL", "component"),
            ]
        )
        measurements.extend(
            [
                make_measurement(formulation_id, f"{formulation_id}__measurement_01", "final concentration in dispersion", final_conc, "mg/mL"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_02", "encapsulation efficiency", ee, "%"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_03", "size", size, "nm"),
                make_measurement(formulation_id, f"{formulation_id}__measurement_04", "pdi", pdi, ""),
                make_measurement(formulation_id, f"{formulation_id}__measurement_05", "zeta potential", zeta, "mV"),
            ]
        )
        add_handoff(handoffs, target_object_ref=formulation_id, target_field_name="instance", source_region_type="table_block", source_locator_text=text_locator, supporting_snippet=table5_block)
        relations.append(make_relation(f"{formulation_id}__cue_01", "inherits_from", formulation_id, nanocapsule_06_parent, f"Table 5 variant with theoretical concentration {theoretical} mg/mL."))

    return {
        "document_key": key,
        "doi": doi,
        "model_name": "paper_driven_deterministic_semantic_emitter_v1",
        "source_schema": "stage2_replacement_semantic_contract_v1",
        "source_mode": "paper_driven_deterministic_semantic_emitter",
        "replacement_emitter_status": "true_paper_driven_semantic_emitter_present__diagnostic_scope",
        "source_text_path": text_locator,
        "source_tables_dir": str(tables_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_notes": [
            "L3H2RS2H emitted from cleaned text plus governed goren_2025 table assets.",
            "Recovered the simple independently reported XAN nanocapsule 800 mg/mL identity from Table 3.",
        ],
        "formulation_identity_candidates": identities,
        "component_candidates": components,
        "phase_candidates": phases,
        "process_step_candidates": processes,
        "variable_or_factor_candidates": factors,
        "measurement_candidates": measurements,
        "relation_cues": relations,
        "evidence_handoffs": handoffs,
    }


def build_5zxyabsu_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    text = read_text(text_path)
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    text_locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")
    table_locator_text = table_locator(tables_dir, "5ZXYABSU__table_02__pdf_table.csv") if tables_dir else text_locator
    families = [
        ("NPR1", "Rhodamine-loaded PLGA nanoparticles", "Rhodamine", ""),
        ("NPR2", "Rhodamine-loaded PLGA nanoparticles with polysorbate 80", "Rhodamine", "Polysorbate 80"),
        ("NPR3", "Rhodamine-loaded PLGA nanoparticles with Labrafil", "Rhodamine", "Labrafil"),
        ("NPB1", "Blank PLGA nanoparticles", "", ""),
        ("NPB2", "Blank PLGA nanoparticles with polysorbate 80", "", "Polysorbate 80"),
        ("NPB3", "Blank PLGA nanoparticles with Labrafil", "", "Labrafil"),
        ("NPG1", "Gatifloxacin-loaded PLGA nanoparticles", "Gatifloxacin", ""),
        ("NPG2", "Gatifloxacin-loaded PLGA nanoparticles with polysorbate 80", "Gatifloxacin", "Polysorbate 80"),
        ("NPG3", "Gatifloxacin-loaded PLGA nanoparticles with Labrafil", "Gatifloxacin", "Labrafil"),
    ]
    for label, desc, drug_name, surfactant in families:
        component_specs = [component_spec("PLGA", "polymer", properties=[{"name": "polymer_family", "value": "PLGA"}])]
        if drug_name:
            component_specs.append(component_spec(drug_name, "drug"))
        if surfactant:
            component_specs.append(component_spec(surfactant, "surfactant"))
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_{label}",
            label=label,
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=component_specs,
            factors_spec=[factor_spec("surface modifier", surfactant or "none"), factor_spec("payload", drug_name or "blank")],
            measurements_spec=[],
            formulation_role="baseline" if label.endswith("1") else "variant",
            instance_context_tags=["table1_family_formulation"],
            change_context_tags=["surface_modification"] if surfactant else ["baseline_family"],
            process_name="double emulsion solvent evaporation",
            source_region_type="table_block" if "Table 2" in text else "text_span",
            source_locator_text=table_locator_text if "Table 2" in text else text_locator,
            supporting_snippet=f"{label} {desc}",
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["5ZXYABSU emitted from explicit NPR/NPB/NPG formulation labels described in text and tables."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_wivucmyg_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    if not tables_dir:
        raise FileNotFoundError("WIVUCMYG tables directory missing.")
    table_path = tables_dir / "WIVUCMYG__table_01__html_table.csv"
    rows = table_rows_with_prefix(table_path, r"F\d+")
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = table_locator(tables_dir, table_path.name)
    for label, row in rows:
        measure_specs = []
        if len(row) > 5 and normalize_text(row[5]):
            measure_specs.append(measurement_spec("size", row[5], "nm"))
        if len(row) > 6 and normalize_text(row[6]):
            measure_specs.append(measurement_spec("pdi", row[6]))
        if len(row) > 7 and normalize_text(row[7]):
            measure_specs.append(measurement_spec("zeta potential", row[7], "mV"))
        if len(row) > 8 and normalize_text(row[8]):
            measure_specs.append(measurement_spec("encapsulation efficiency", row[8], "%"))
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_{label}",
            label=label,
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[
                component_spec(
                    "PLGA",
                    "polymer",
                    properties=[
                        target_resomer_ratio_property("Resomer 753S", "75:25"),
                        target_resomer_grade_property("Resomer 753S"),
                    ],
                ),
                component_spec("Pranoprofen", "drug"),
                component_spec("PVA", "surfactant"),
            ],
            factors_spec=[
                factor_spec("cPF", normalize_text(row[1]) if len(row) > 1 else ""),
                factor_spec("cPVA", normalize_text(row[2]) if len(row) > 2 else ""),
                factor_spec("cPLGA", normalize_text(row[3]) if len(row) > 3 else ""),
                factor_spec("pH", normalize_text(row[4]) if len(row) > 4 else ""),
            ],
            measurements_spec=measure_specs,
            instance_context_tags=["table1_factorial_formulation"],
            change_context_tags=["coded_factor_row"],
            process_name="nanoprecipitation",
            source_region_type="table_row",
            source_locator_text=locator,
            supporting_snippet=" | ".join(row[:9]),
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["WIVUCMYG emitted from explicit F-row DOE table in cleaned assets."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_yga8vqku_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    if not tables_dir:
        raise FileNotFoundError("YGA8VQKU tables directory missing.")
    table_path = tables_dir / "YGA8VQKU__table_01__html_table.csv"
    rows = table_rows_with_prefix(table_path, r"F\d+")
    low_viscosity_properties = [
        property_spec(
            "molecular_weight",
            "Low viscosity PLGA (0.32-0.44 dL/g)",
            "Paper-local low-viscosity PLGA family carried through the deterministic DOE emitter for the retained F-row family.",
        )
    ]
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = table_locator(tables_dir, table_path.name)
    for label, row in rows:
        measure_specs = []
        if len(row) > 4 and normalize_text(row[4]):
            measure_specs.append(measurement_spec("size", row[4], "nm"))
        if len(row) > 5 and normalize_text(row[5]):
            measure_specs.append(measurement_spec("encapsulation efficiency", row[5], "%"))
        if len(row) > 6 and normalize_text(row[6]):
            measure_specs.append(measurement_spec("zeta potential", row[6], "mV"))
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_{label}",
            label=label,
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[
                component_spec("PLGA", "polymer", properties=low_viscosity_properties),
                component_spec("Flurbiprofen", "drug"),
                component_spec("Poloxamer 188", "surfactant"),
            ],
            factors_spec=[
                factor_spec("cFB", normalize_text(row[1]) if len(row) > 1 else ""),
                factor_spec("cP188", normalize_text(row[2]) if len(row) > 2 else ""),
                factor_spec("pH", normalize_text(row[3]) if len(row) > 3 else ""),
            ],
            measurements_spec=measure_specs,
            instance_context_tags=["table1_factorial_formulation"],
            change_context_tags=["coded_factor_row"],
            process_name="nanoprecipitation",
            source_region_type="table_row",
            source_locator_text=locator,
            supporting_snippet=" | ".join(row[:7]),
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=[
            "YGA8VQKU emitted from explicit F-row DOE table in cleaned assets.",
            "The retained F-row DOE family carries the explicit low-viscosity PLGA descriptor reported in the paper-local summary tables.",
        ],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_bb3juvw7_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    if not tables_dir:
        raise FileNotFoundError("BB3JUVW7 tables directory missing.")
    table1 = read_csv_rows(tables_dir / "BB3JUVW7__table_01__html_table.csv")[1:]
    table2 = read_csv_rows(tables_dir / "BB3JUVW7__table_02__html_table.csv")[1:]
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator1 = table_locator(tables_dir, "BB3JUVW7__table_01__html_table.csv")
    locator2 = table_locator(tables_dir, "BB3JUVW7__table_02__html_table.csv")
    for idx, row in enumerate(table1, start=1):
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_Nanosphere_{idx:02d}",
            label=f"Nanosphere row {idx}",
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[
                component_spec("Artemether", "drug", f"{normalize_text(row[0])} mg"),
                component_spec("PLGA", "polymer", f"{normalize_text(row[1])} mg"),
                component_spec("PVA", "surfactant", f"{normalize_text(row[2])} mg"),
                component_spec("Acetone", "organic_solvent", f"{normalize_text(row[3])} mL"),
            ],
            factors_spec=[factor_spec("aqueous phase", f"{normalize_text(row[4])} mL")],
            measurements_spec=[
                measurement_spec("size", row[5], "nm"),
                measurement_spec("pdi", row[6]),
                measurement_spec("zeta potential", row[7], "mV"),
                measurement_spec("encapsulation efficiency", row[8], "%"),
                measurement_spec("loading content", row[9], "%"),
            ],
            formulation_role="baseline",
            instance_context_tags=["table1_nanosphere"],
            change_context_tags=["composition_row"],
            process_name="solvent evaporation",
            source_region_type="table_row",
            source_locator_text=locator1,
            supporting_snippet=" | ".join(row[:10]),
        )
    for idx, row in enumerate(table2, start=1):
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_Nanorod_{idx:02d}",
            label=f"Nanorod row {idx}",
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[
                component_spec("Artemether", "drug"),
                component_spec(
                    "PLGA",
                    "polymer",
                    properties=[
                        property_spec(
                            "la_ga_ratio",
                            normalize_text(row[1]),
                            f"PLGA type (lactide:glycolide) {normalize_text(row[1])}",
                        )
                    ],
                ),
            ],
            factors_spec=[
                factor_spec("film thickness", normalize_text(row[0])),
                factor_spec("PLGA type", normalize_text(row[1])),
                factor_spec("stretching extent", normalize_text(row[2])),
                factor_spec("liquefaction method", normalize_text(row[3])),
                factor_spec("incubation period", normalize_text(row[4])),
            ],
            measurements_spec=[
                measurement_spec("major axis", row[5], "nm"),
                measurement_spec("minor axis", row[6], "nm"),
                measurement_spec("aspect ratio", row[7]),
                measurement_spec("Feret diameter", row[8], "nm"),
                measurement_spec("minor Feret diameter", row[9], "nm"),
                measurement_spec("loading content", row[10], "µg/mg"),
            ],
            formulation_role="variant",
            instance_context_tags=["table2_nanorod"],
            change_context_tags=["process_variant"],
            process_name="mechanical stretching",
            source_region_type="table_row",
            source_locator_text=locator2,
            supporting_snippet=" | ".join(row[:11]),
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["BB3JUVW7 emitted from explicit nanosphere and nanorod tables."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_7zs858ns_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")
    append_specs_to_lists(
        key=key,
        doi=doi,
        formulation_id=f"{key}_MF_PLGA_NP_01",
        label="MF-loaded PLGA nanoparticles",
        identities=identities,
        components=components,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
        components_spec=[component_spec("PLGA", "polymer"), component_spec("Mometasone furoate", "drug")],
        factors_spec=[],
        measurements_spec=[],
        formulation_role="optimized",
        instance_context_tags=["single_main_formulation"],
        change_context_tags=["reported_main_formulation"],
        process_name="emulsion solvent evaporation",
        source_region_type="text_span",
        source_locator_text=locator,
        supporting_snippet="MF-loaded PLGA nanoparticles reported as the developed formulation.",
    )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["7ZS858NS emitted as a single main formulation from text and Table 1 references."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_inmutv7l_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    if not tables_dir:
        raise FileNotFoundError("INMUTV7L tables directory missing.")
    rows = read_csv_rows(tables_dir / "INMUTV7L__table_15__pdf_table.csv")
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = table_locator(tables_dir, "INMUTV7L__table_15__pdf_table.csv")
    current_polymer = ""
    for row in rows:
        if len(row) >= 2 and "PLGA" in normalize_text(row[1]).upper():
            current_polymer = normalize_text(row[1]).replace("庐", "")
            continue
        label = normalize_text(row[0]) if row else ""
        if not re.fullmatch(r"\d+", label):
            continue
        surfactant = normalize_text(row[2]) if len(row) > 2 else ""
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_Formulation_{label}",
            label=f"Formulation {label}",
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[component_spec(current_polymer or "PLGA", "polymer"), component_spec(surfactant or "surfactant", "surfactant"), component_spec("Dexibuprofen", "drug")],
            factors_spec=[factor_spec("surfactant", surfactant), factor_spec("polymer group", current_polymer)],
            measurements_spec=[
                measurement_spec("size", normalize_text(row[3]) if len(row) > 3 else "", "nm"),
                measurement_spec("pdi", normalize_text(row[4]) if len(row) > 4 else ""),
                measurement_spec("zeta potential", normalize_text(row[5]) if len(row) > 5 else "", "mV"),
                measurement_spec("encapsulation efficiency", normalize_text(row[6]) if len(row) > 6 else "", "%"),
            ],
            instance_context_tags=["table1_formulation_number"],
            change_context_tags=["surfactant_polymer_grid"],
            process_name="nanoprecipitation",
            source_region_type="table_row",
            source_locator_text=locator,
            supporting_snippet=" | ".join(row[:7]),
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["INMUTV7L emitted from explicit numbered formulations in Table 1."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_pa3spz28_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = table_locator(tables_dir, "PA3SPZ28__table_01__pdf_table.csv") if tables_dir else str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")
    for idx, ratio in enumerate(["1:20", "1:10", "1:6.66", "1:10 after storage"], start=1):
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_GAR_NP_{idx:02d}",
            label=f"GAR-NP ratio {ratio}",
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[component_spec("PLGA", "polymer", properties=[{"name": "la_ga_ratio", "value": "50:50"}]), component_spec("Garcinol", "drug"), component_spec("Vitamin E TPGS", "surfactant", "0.03% w/v")],
            factors_spec=[factor_spec("drug:polymer ratio", ratio.replace(" after storage", "")), factor_spec("storage", "after 3 months" if "storage" in ratio else "fresh")],
            measurements_spec=[],
            formulation_role="variant",
            instance_context_tags=["table1_ratio_variant"],
            change_context_tags=["drug_polymer_ratio"],
            process_name="nanoprecipitation",
            source_region_type="table_block",
            source_locator_text=locator,
            supporting_snippet="Drug:Polymer ratio variants in Table 1.",
        )
    append_specs_to_lists(
        key=key,
        doi=doi,
        formulation_id=f"{key}_Blank_NP_01",
        label="Blank PLGA nanoparticles",
        identities=identities,
        components=components,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
        components_spec=[component_spec("PLGA", "polymer"), component_spec("Vitamin E TPGS", "surfactant", "0.03% w/v")],
        factors_spec=[factor_spec("payload", "blank")],
        measurements_spec=[],
        formulation_role="baseline",
        instance_context_tags=["blank_control"],
        change_context_tags=["blank_reference"],
        process_name="nanoprecipitation",
        source_region_type="text_span",
        source_locator_text=str(text_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        supporting_snippet="blank-NPs (without GAR)",
    )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["PA3SPZ28 emitted from explicit ratio variants plus blank nanoparticle mention."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_qlyklpkt_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")
    for idx, level in enumerate(["2.5 mg/mL", "3 mg/mL", "4 mg/mL", "10 mg/mL"], start=1):
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_Unloaded_{idx:02d}",
            label=f"Unloaded PLGA nanoparticles with poloxamer 188 {level}",
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[component_spec("PLGA", "polymer", properties=[{"name": "la_ga_ratio", "value": "50:50"}]), component_spec("Poloxamer 188", "surfactant", level)],
            factors_spec=[factor_spec("poloxamer 188 concentration", level), factor_spec("payload", "unloaded")],
            measurements_spec=[],
            formulation_role="variant",
            instance_context_tags=["table1_unloaded_sweep"],
            change_context_tags=["surfactant_sweep"],
            process_name="nanoprecipitation",
            source_region_type="text_span",
            source_locator_text=locator,
            supporting_snippet=f"Table 1 unloaded PLGA nanoparticles stabilized by {level} poloxamer 188.",
        )
    for idx, ratio in enumerate(["5:1", "10:1", "15:1"], start=1):
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_Loaded_{idx:02d}",
            label=f"PLGA-ITZ-NS ratio {ratio}",
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[component_spec("PLGA", "polymer", properties=[{"name": "la_ga_ratio", "value": "50:50"}]), component_spec("Itraconazole", "drug"), component_spec("Poloxamer 188", "surfactant", "3 mg/mL")],
            factors_spec=[factor_spec("PLGA:ITZ ratio", ratio), factor_spec("poloxamer 188 concentration", "3 mg/mL")],
            measurements_spec=[],
            formulation_role="variant",
            instance_context_tags=["table2_loaded_ratio"],
            change_context_tags=["plga_itz_ratio"],
            process_name="nanoprecipitation",
            source_region_type="text_span",
            source_locator_text=locator,
            supporting_snippet=f"Table 2 PLGA-ITZ-NS with PLGA:ITZ ratio {ratio}.",
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["QLYKLPKT emitted from explicit surfactant-concentration and PLGA:ITZ ratio sweeps reported in text."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_rhmjwzx8_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")
    append_specs_to_lists(
        key=key,
        doi=doi,
        formulation_id=f"{key}_Polysorbate80_Coated_01",
        label="Polysorbate 80-coated acetylpuerarin-loaded PLGA nanoparticles",
        identities=identities,
        components=components,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
        components_spec=[component_spec("PLGA", "polymer"), component_spec("Acetylpuerarin", "drug"), component_spec("Polysorbate 80", "surfactant")],
        factors_spec=[factor_spec("surface coating", "Polysorbate 80")],
        measurements_spec=[],
        formulation_role="optimized",
        instance_context_tags=["single_main_formulation"],
        change_context_tags=["surface_coating"],
        process_name="nanoprecipitation",
        source_region_type="text_span",
        source_locator_text=locator,
        supporting_snippet="Polysorbate 80-coated PLGA nanoparticles improve the permeability of acetylpuerarin.",
    )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["RHMJWZX8 emitted as the single benchmark-facing coated formulation from title and methods text."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_v99gkzei_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    if not tables_dir:
        raise FileNotFoundError("V99GKZEI tables directory missing.")
    rows = table_rows_with_prefix(tables_dir / "V99GKZEI__table_01__html_table.csv", r".*")
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = table_locator(tables_dir, "V99GKZEI__table_01__html_table.csv")
    for label, row in rows:
        if not label or "NPs composition" in label or label.startswith("a "):
            continue
        component_specs = [component_spec("PLGA", "polymer", "20 mg"), component_spec("Methylene Blue", "drug", "0.5 mg")]
        component_specs[0]["component_properties_raw"] = json.dumps(
            [
                target_resomer_ratio_property("RG502H", "50:50"),
                property_spec("molecular_weight", "RG502H MW range 7000-17000 Da", "PLGA (RG502H) MW range was 7000-17000 Da"),
            ],
            ensure_ascii=False,
        )
        if "/SC6OH" in label:
            component_specs.append(component_spec("SC6OH", "additive", f"{label.split('SC6OH', 1)[1]} mg"))
        elif "(W/O/W)" in label:
            component_specs.extend([component_spec("Tween 80", "surfactant"), component_spec("Pluronic F68", "surfactant")])
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_{slugify(label)}",
            label=label,
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=component_specs,
            factors_spec=[],
            measurements_spec=[
                measurement_spec("size", normalize_text(row[1]) if len(row) > 1 else "", "nm"),
                measurement_spec("pdi", normalize_text(row[2]) if len(row) > 2 else ""),
                measurement_spec("loading content", normalize_text(row[4]) if len(row) > 4 else "%"),
                measurement_spec("encapsulation efficiency", normalize_text(row[5]) if len(row) > 5 else "%"),
            ],
            formulation_role="variant",
            instance_context_tags=["table1_composition_row"],
            change_context_tags=["component_variant"],
            process_name="nanoprecipitation" if "(W/O/W)" not in label else "W/O/W emulsion solvent evaporation",
            source_region_type="table_row",
            source_locator_text=locator,
            supporting_snippet=" | ".join(row[:6]),
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["V99GKZEI emitted from explicit composition labels in Table 1."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_wfdtq4vx_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")
    for idx in range(1, 28):
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_DesignBatch_{idx:02d}",
            label=f"Design batch {idx}",
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[component_spec("PLGA", "polymer"), component_spec("Lopinavir", "drug")],
            factors_spec=[factor_spec("design batch index", str(idx))],
            measurements_spec=[],
            formulation_role="variant",
            instance_context_tags=["factorial_design_batch"],
            change_context_tags=["design_grid"],
            process_name="nanoprecipitation",
            source_region_type="text_span",
            source_locator_text=locator,
            supporting_snippet="Twenty seven batches of lopinavir-loaded PLGA NPs were prepared.",
        )
    for idx in range(1, 3):
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_Checkpoint_{idx:02d}",
            label=f"Checkpoint batch {idx}",
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[component_spec("PLGA", "polymer"), component_spec("Lopinavir", "drug")],
            factors_spec=[factor_spec("checkpoint batch", str(idx))],
            measurements_spec=[],
            formulation_role="variant",
            instance_context_tags=["checkpoint_batch"],
            change_context_tags=["checkpoint_validation"],
            process_name="nanoprecipitation",
            source_region_type="text_span",
            source_locator_text=locator,
            supporting_snippet="Three check points were selected as shown in Table 7.",
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["WFDTQ4VX emitted from explicit twenty-seven-batch factorial design plus checkpoint batches described in text."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_5gif3d8w_document(record: dict[str, str], text_path: Path, tables_dir: Path | None) -> dict[str, Any]:
    key = normalize_text(record["key"])
    doi = normalize_text(record["doi"])
    identities, components, phases, processes, factors, measurements, relations, handoffs = empty_object_lists()
    locator = str(text_path.relative_to(REPO_ROOT)).replace("\\", "/")
    families = [("PLGA 50/50", "nanoprecipitation"), ("PLGA 75/25", "nanoprecipitation"), ("PLGA 85/15", "nanoprecipitation"), ("PCL", "emulsion solvent evaporation")]
    formulations: list[tuple[str, str, list[dict[str, str]]]] = []
    for polymer_name, method in families:
        formulations.append((f"Optimized {polymer_name} nanoparticles", method, [factor_spec("polymer family", polymer_name)]))
    for polymer_name, method in [("PLGA 50/50", "nanoprecipitation"), ("PCL", "emulsion solvent evaporation")]:
        for amount in ["25 mg", "50 mg", "100 mg"]:
            formulations.append((f"{polymer_name} polymer amount {amount}", method, [factor_spec("polymer amount", amount), factor_spec("polymer family", polymer_name)]))
        for amount in ["0.5% w/v", "1.0% w/v", "2.0% w/v"]:
            formulations.append((f"{polymer_name} stabilizer {amount}", method, [factor_spec("stabilizer concentration", amount), factor_spec("polymer family", polymer_name)]))
        for amount in ["2.5 mg", "5 mg", "10 mg", "20 mg"]:
            formulations.append((f"{polymer_name} etoposide amount {amount}", method, [factor_spec("etoposide amount", amount), factor_spec("polymer family", polymer_name)]))
    for idx, (label, method, factor_specs) in enumerate(formulations[:26], start=1):
        polymer_name = next((spec["expression"] for spec in factor_specs if spec["name"] == "polymer family"), "PLGA 50/50")
        append_specs_to_lists(
            key=key,
            doi=doi,
            formulation_id=f"{key}_Sweep_{idx:02d}",
            label=label,
            identities=identities,
            components=components,
            processes=processes,
            factors=factors,
            measurements=measurements,
            relations=relations,
            handoffs=handoffs,
            components_spec=[component_spec(polymer_name, "polymer"), component_spec("Etoposide", "drug")],
            factors_spec=factor_specs,
            measurements_spec=[],
            formulation_role="variant",
            instance_context_tags=["text_described_formulation_variable"],
            change_context_tags=["formulation_variable_sweep"],
            process_name=method,
            source_region_type="text_span",
            source_locator_text=locator,
            supporting_snippet="Different formulation variables like polymer amount, stabilizer concentration, and etoposide amount were changed.",
        )
    return finalize_document(
        key=key,
        doi=doi,
        text_path=text_path,
        tables_dir=tables_dir,
        source_notes=["5GIF3D8W emitted from explicit paper-described formulation-variable sweeps; this builder is intentionally paper-local and diagnostic in scope."],
        identities=identities,
        components=components,
        phases=phases,
        processes=processes,
        factors=factors,
        measurements=measurements,
        relations=relations,
        handoffs=handoffs,
    )


def build_document(record: dict[str, str]) -> dict[str, Any]:
    key = normalize_text(record["key"])
    text_path = Path(record["text_path"])
    if not text_path.is_absolute():
        text_path = REPO_ROOT / text_path
    tables_dir = resolve_tables_dir(key, text_path)
    builder = DOCUMENT_BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"Unsupported paper key for this governed emitter: {key}")
    return builder(record, text_path, tables_dir)


DOCUMENT_BUILDERS: dict[str, Any] = {
    "5GIF3D8W": build_5gif3d8w_document,
    "5ZXYABSU": build_5zxyabsu_document,
    "7ZS858NS": build_7zs858ns_document,
    "BB3JUVW7": build_bb3juvw7_document,
    "BXCV5XWB": build_bxcv5xwb_document,
    "INMUTV7L": build_inmutv7l_document,
    "L3H2RS2H": build_l3h2rs2h_document,
    "PA3SPZ28": build_pa3spz28_document,
    "QLYKLPKT": build_qlyklpkt_document,
    "RHMJWZX8": build_rhmjwzx8_document,
    "UFXX9WXE": build_ufxx9wxe_document,
    "V99GKZEI": build_v99gkzei_document,
    "WFDTQ4VX": build_wfdtq4vx_document,
    "WIVUCMYG": build_wivucmyg_document,
    "YGA8VQKU": build_yga8vqku_document,
}


def supported_paper_keys() -> list[str]:
    return sorted(DOCUMENT_BUILDERS)


def finalize_fallback_document(document: dict[str, Any]) -> dict[str, Any]:
    document_key = normalize_text(document.get("document_key"))
    document["stage2_semantic_source_mode"] = FALLBACK_SEMANTIC_SOURCE_MODE
    document["semantic_universe_authority"] = FALLBACK_PROVENANCE
    document["semantic_scope_declarations"] = [
        {
            "scope_id": f"{document_key}__governed_fallback_document_scope__01",
            "scope_kind": "governed_fallback_document_scope",
            "declared_by": FALLBACK_PROVENANCE,
            "authorizes_row_materialization_modes": [FALLBACK_PROVENANCE],
            "row_enumeration_required": "paper_specific_fallback",
            "table_scope_refs": list(document.get("source_table_files") or []),
            "declaration_basis": "explicit_governed_fallback_semantic_source_mode",
        }
    ]
    return document


def summary_row(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_key": document["document_key"],
        "doi": document["doi"],
        "stage2_semantic_source_mode": document.get("stage2_semantic_source_mode", FALLBACK_SEMANTIC_SOURCE_MODE),
        "legacy_row_count": "",
        "identity_count": len(document["formulation_identity_candidates"]),
        "component_count": len(document["component_candidates"]),
        "phase_count": len(document["phase_candidates"]),
        "process_count": len(document["process_step_candidates"]),
        "variable_count": len(document["variable_or_factor_candidates"]),
        "measurement_count": len(document["measurement_candidates"]),
        "relation_cue_count": len(document["relation_cues"]),
        "evidence_handoff_count": len(document["evidence_handoffs"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit semantic-object Stage2 payloads directly from cleaned paper assets.")
    parser.add_argument("--manifest-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--paper-keys", nargs="+", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_rows = read_tsv(args.manifest_tsv)
    wanted = {normalize_text(key) for key in args.paper_keys}
    selected = [row for row in manifest_rows if normalize_text(row.get("key")) in wanted]
    if len(selected) != len(wanted):
        found = {normalize_text(row.get("key")) for row in selected}
        missing = sorted(wanted - found)
        raise ValueError(f"Manifest missing requested paper keys: {missing}")

    documents = [finalize_fallback_document(build_document(row)) for row in selected]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.out_dir / SEMANTIC_JSONL_NAME
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    summary_rows = [summary_row(document) for document in documents]
    write_tsv(
        args.out_dir / SUMMARY_TSV_NAME,
        summary_rows,
        [
            "document_key",
            "doi",
            "stage2_semantic_source_mode",
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

    out_dir_resolved = args.out_dir.resolve()
    manifest = {
        "output_jsonl": str(jsonl_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "output_summary_tsv": str((out_dir_resolved / SUMMARY_TSV_NAME).relative_to(REPO_ROOT)).replace("\\", "/"),
        "paper_keys": [document["document_key"] for document in documents],
        "source_mode": "paper_driven_deterministic_semantic_emitter",
        "stage2_semantic_source_mode": FALLBACK_SEMANTIC_SOURCE_MODE,
        "replacement_emitter_status": "true_paper_driven_semantic_emitter_present__diagnostic_scope",
        "notes": [
            "Diagnostic-only DEV15 replacement validation emitter.",
            "No legacy Stage2 row content was used to generate semantic objects.",
            "This emitter is an explicitly governed fallback semantic source and must not be confused with llm_first_composite Stage2 authority.",
        ],
    }
    (args.out_dir / MANIFEST_JSON_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"papers": len(documents), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
