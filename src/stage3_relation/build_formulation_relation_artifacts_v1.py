#!/usr/bin/env python3
from __future__ import annotations

"""
Build deterministic Stage 3 formulation relation artifacts from Stage 2 weak labels.

Purpose:
- Materialize an explicit, auditable paper-level formulation relation layer.
- Separate relation reasoning from later Stage 5 final-row closure.
- Expose method groups, shared fields, variation axes, parent links, and
  candidate-level field membership without any LLM usage.

Inputs:
- A Stage 2 weak-label TSV produced by the active extractor.
- An optional Stage 2 weak-label JSONL for paper-level notes or cross-checking.
- An optional scope manifest TSV for paper title and source-path enrichment.

Outputs:
- `formulation_relation_records_v1.tsv`
- `formulation_logic_graph_v1.jsonl`
- `formulation_relation_summary_v1.tsv`

Stage role:
- Concrete implementation of the deterministic Stage 3 relation/materialization
  boundary between candidate extraction and final formulation closure.

This script does not:
- call any LLM or external API,
- perform benchmark comparison,
- overwrite upstream Stage 2 outputs,
- decide the final benchmark-valid formulation table by itself.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.stage2_sampling_labels.table_structure_dictionary_v1 import (
    canonical_field_for_header as dictionary_canonical_field_for_header,
    normalize_dictionary_value,
)

csv.field_size_limit(sys.maxsize)


RELATION_RECORDS_NAME = "formulation_relation_records_v1.tsv"
RELATION_GRAPH_JSONL_NAME = "formulation_logic_graph_v1.jsonl"
RELATION_SUMMARY_NAME = "formulation_relation_summary_v1.tsv"
RESOLVED_RELATION_FIELDS_NAME = "resolved_relation_fields_v1.tsv"
IDENTITY_VARIABLES_FIELD = "identity_variables_json"

BASE_ROW_COLUMNS = {
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
    "instance_evidence_region_type",
    "evidence_section",
    "evidence_span_text",
    "evidence_span_start",
    "evidence_span_end",
}

RELATION_FIELDNAMES = [
    "relation_row_id",
    "relation_graph_id",
    "paper_key",
    "doi",
    "paper_title",
    "method_group_id",
    "variation_axis_id",
    "formulation_candidate_id",
    "candidate_label",
    "parent_entity_id",
    "related_entity_id",
    "relation_type",
    "field_name",
    "field_value_raw",
    "field_value_norm",
    "field_scope",
    "candidate_source",
    "instance_kind",
    "formulation_role",
    "evidence_source_type",
    "evidence_section",
    "evidence_snippet",
    "is_shared",
    "variation_axis_indicator",
    "source_weak_label_row_ref",
    "deterministic_confidence",
    "provenance_note",
]

SUMMARY_FIELDNAMES = [
    "paper_key",
    "doi",
    "paper_title",
    "relation_graph_id",
    "candidate_count",
    "method_group_count",
    "shared_field_count",
    "variation_axis_count",
    "variation_membership_count",
    "parent_link_count",
    "relation_row_count",
    "relation_type_counts_json",
]

RESOLVED_FIELDNAMES = [
    "formulation_candidate_id",
    "paper_key",
    "method_group_id",
    "scope_type",
    "field_name",
    "field_value",
    "field_value_norm",
    "resolution_rule",
    "source_relation_row_ids",
    "deterministic_confidence",
]
METHOD_GROUP_SIGNATURE_HINT_FIELD = "method_group_signature_hint"
INHERITANCE_MARKER_FIELD = "inheritance_markers_json"
CONTEXT_INHERITANCE_MARKER_FIELD = "context_inheritance_markers_json"
PROTOCOL_INHERITANCE_MARKER_FIELD = "protocol_inheritance_markers_json"
RELATION_CUE_FIELD = "relation_cues_json"
TYPED_INHERITANCE_FIELD = "typed_inheritance_fields_json"
TYPED_DOE_FACTOR_FIELD = "typed_doe_factors_json"
RESULT_BINDING_CANDIDATE_FIELD = "result_binding_candidates_json"

CANONICAL_FIELD_ALIASES = {
    "plga_mw_kDa": "polymer_mw_kDa",
    "drug_mass_mg": "drug_feed_amount_text",
    "drug amount": "drug_feed_amount_text",
    "drug mass": "drug_feed_amount_text",
    "drug feed": "drug_feed_amount_text",
    "drug_feed": "drug_feed_amount_text",
    "drug feed amount": "drug_feed_amount_text",
    "drug_feed_amount": "drug_feed_amount_text",
    "polymer_mass_mg": "plga_mass_mg",
    "polymer amount": "plga_mass_mg",
    "polymer mass": "plga_mass_mg",
    "o_volume_mL": "organic_phase_volume_mL",
    "O_volume_mL": "organic_phase_volume_mL",
    "o_volume_ml": "organic_phase_volume_mL",
    "organic phase volume": "organic_phase_volume_mL",
    "organic_phase_volume_ml": "organic_phase_volume_mL",
    "w2_volume_mL": "external_aqueous_phase_volume_mL",
    "W2_volume_mL": "external_aqueous_phase_volume_mL",
    "w2_volume_ml": "external_aqueous_phase_volume_mL",
    "external aqueous phase volume": "external_aqueous_phase_volume_mL",
    "external_aqueous_phase_volume_ml": "external_aqueous_phase_volume_mL",
    "drug concentration": "drug_concentration_value",
    "drug_concentration": "drug_concentration_value",
    "polymer concentration": "polymer_concentration_value",
    "polymer_concentration": "polymer_concentration_value",
    "plga concentration": "polymer_concentration_value",
    "phase ratio": "phase_ratio_raw",
    "phase_ratio": "phase_ratio_raw",
    "w/o phase volume ratio": "phase_ratio_raw",
    "water/oil phase volume ratio": "phase_ratio_raw",
    "pH": "pH_raw",
    "ph": "pH_raw",
    "aqueous phase pH": "pH_raw",
    "water phase pH": "pH_raw",
    "pH of water phase": "pH_raw",
    "surfactant concentration": "surfactant_concentration_text",
    "stabilizer concentration": "surfactant_concentration_text",
    "poloxamer concentration": "surfactant_concentration_text",
    "pva concentration": "pva_conc_percent",
    "polymer amount": "plga_mass_mg",
    "plga amount": "plga_mass_mg",
    "drug amount": "drug_feed_amount_text",
    "drug concentration": "drug_feed_amount_text",
    "plga:itz ratio": "polymer_to_drug_ratio_raw",
    "plga/itz ratio": "polymer_to_drug_ratio_raw",
    "polymer:drug ratio": "polymer_to_drug_ratio_raw",
    "polymer/drug ratio": "polymer_to_drug_ratio_raw",
    "polymer to drug ratio": "polymer_to_drug_ratio_raw",
    "particle_size_nm": "size_nm",
    "particle size": "size_nm",
    "particle size nm": "size_nm",
    "mean diameter": "size_nm",
    "diameter": "size_nm",
    "polydispersity index": "pdi",
    "polydispersity_index": "pdi",
    "pi": "pdi",
    "zeta potential": "zeta_mV",
    "zeta_potential_mV": "zeta_mV",
    "zeta_potential": "zeta_mV",
    "zeta potential mv": "zeta_mV",
    "zeta_mv": "zeta_mV",
    "surfactant_concentration_percent_w_v": "surfactant_concentration_text",
    "surfactant concentration percent w v": "surfactant_concentration_text",
    "stabilizer_concentration_percent_w_v": "surfactant_concentration_text",
    "stabilizer concentration percent w v": "surfactant_concentration_text",
}

METHOD_GROUP_SIGNATURE_FIELDS = [
    "emul_method",
    "emul_type",
    "organic_solvent",
    "surfactant_name",
    "surfactant_concentration_text",
    "pva_conc_percent",
]

VARIATION_AXIS_FIELDS = {
    "polymer_identity",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "surfactant_name",
    "surfactant_concentration_text",
    "pva_conc_percent",
    "organic_solvent",
    "drug_name",
    "drug_feed_amount_text",
    "emul_type",
    "emul_method",
    IDENTITY_VARIABLES_FIELD,
}

RESOLVABLE_RELATION_FIELDS = {
    "drug_name",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "surfactant_name",
    "surfactant_concentration_text",
    "pva_conc_percent",
    "organic_solvent",
    "preparation_method",
    "drug_feed_amount_text",
    "la_ga_ratio",
    "polymer_identity",
    "polymer_name_raw",
    "stabilizer_name",
    "emul_type",
    "polymer_to_drug_ratio_raw",
    "drug_to_polymer_ratio_raw",
    "organic_phase_volume_mL",
    "external_aqueous_phase_volume_mL",
    "drug_concentration_value",
    "drug_concentration_unit",
    "polymer_concentration_value",
    "polymer_concentration_unit",
    "phase_ratio_raw",
    "pH_raw",
    "stirring_time_h",
    "evaporation_time_h",
    "sonication_time_s",
    "homogenization_time_min",
}

MEASUREMENT_BINDING_FIELDS = {
    "encapsulation_efficiency_percent",
    "dl_percent",
    "loading_content_percent",
    "pdi",
    "size_nm",
    "zeta_mV",
}
DICTIONARY_TO_STAGE5_MEASUREMENT_FIELD = {
    "dl_percent": "dl_percent",
    "ee_percent": "encapsulation_efficiency_percent",
    "encapsulation_efficiency_percent": "encapsulation_efficiency_percent",
    "lc_percent": "loading_content_percent",
    "loading_content_percent": "loading_content_percent",
    "particle_size_nm": "size_nm",
    "pdi": "pdi",
    "size_nm": "size_nm",
    "zeta_mV": "zeta_mV",
}

PROTOCOL_INHERITANCE_ALLOWED_FIELDS = RESOLVABLE_RELATION_FIELDS | {
    "organic_phase_volume_mL",
    "external_aqueous_phase_volume_mL",
    "stirring_time_h",
    "evaporation_time_h",
    "sonication_time_s",
    "homogenization_time_min",
}

CONTEXT_INHERITANCE_NEVER_FIELDS = {
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
    "release",
    "assay_result",
}

CONTEXT_INHERITANCE_CONDITIONAL_FIELDS = {
    "drug_feed_amount_text",
    "plga_mass_mg",
    "surfactant_concentration_text",
    "pva_conc_percent",
    "polymer_to_drug_ratio_raw",
    "drug_to_polymer_ratio_raw",
}

INHERITED_FIELD_ALIAS_RULES = {
    "surfactant_concentration_text": [
        "surfactant concentration",
        "stabilizer concentration",
        "poloxamer concentration",
        "poloxamer 188 concentration",
        "pluronic concentration",
        "f68 concentration",
    ],
    "pva_conc_percent": [
        "pva concentration",
        "polyvinyl alcohol concentration",
    ],
    "plga_mass_mg": [
        "polymer amount",
        "polymer concentration",
        "plga concentration",
        "plga amount",
    ],
    "drug_feed_amount_text": [
        "drug amount",
        "drug concentration",
        "itraconazole amount",
        "api amount",
        "payload amount",
    ],
    "la_ga_ratio": [
        "la ga ratio",
        "la/ga ratio",
        "lactic glycolic ratio",
        "plga ratio",
    ],
    "organic_solvent": [
        "organic solvent",
        "solvent",
        "solvent type",
    ],
}


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def normalize_token(value: Any) -> str:
    text = normalize_text(value).lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9%:/.+-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_identity_variable_name(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_identity_variable_value(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).lower()).strip()


def canonical_identity_variables_signature(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return normalize_token(text)
    if not isinstance(parsed, list):
        return normalize_token(text)
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
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
        pairs.append(key)
    if not pairs:
        return ""
    pairs.sort()
    return "|".join(f"{name}={factor_value}" for name, factor_value in pairs)


def canonical_field_name(field_name: Any) -> str:
    text = normalize_text(field_name)
    return CANONICAL_FIELD_ALIASES.get(text, CANONICAL_FIELD_ALIASES.get(text.lower(), text))


def truncate_text(value: Any, max_len: int = 240) -> str:
    text = normalize_text(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_text_for_manifest_row(row: dict[str, str]) -> str:
    text_path = normalize_text(row.get("text_path") or row.get("source_text_path"))
    candidates: list[Path] = []
    if text_path:
        path = Path(text_path.replace("\\", "/"))
        candidates.append(path if path.is_absolute() else Path.cwd() / path)
    key = normalize_text(row.get("key") or row.get("zotero_key") or row.get("paper_key"))
    if key:
        candidates.extend(
            [
                Path.cwd() / "data" / "cleaned" / "content" / "text" / f"{key}.pdf.txt",
                Path.cwd() / "data" / "cleaned" / "content" / "text" / f"{key}.html.txt",
                Path.cwd() / "data" / "cleaned" / "content_goren_2025" / "text" / f"{key}.pdf.txt",
                Path.cwd() / "data" / "cleaned" / "content_goren_2025" / "text" / f"{key}.html.txt",
            ]
        )
    for path in candidates:
        if not path.exists():
            continue
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
    return ""


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


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_field_names(headers: list[str]) -> list[str]:
    fields: list[str] = []
    for header in headers:
        if header.endswith("_value"):
            name = canonical_field_name(header[: -len("_value")])
            if name not in fields:
                fields.append(name)
    for extra in ["polymer_identity", "polymer_name_raw", "preparation_method", IDENTITY_VARIABLES_FIELD]:
        if extra in headers and extra not in fields:
            fields.append(extra)
    return fields


def field_column_candidates(field_name: str, suffix: str = "") -> list[str]:
    if field_name == "polymer_mw_kDa":
        names = ["polymer_mw_kDa", "plga_mw_kDa"]
    else:
        names = [field_name]
    return [f"{name}{suffix}" for name in names]


def load_manifest_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = read_tsv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = normalize_text(row.get("key") or row.get("zotero_key") or row.get("paper_key"))
        if key:
            out[key] = row
    return out


def load_jsonl_notes(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl_rows(path):
        key = normalize_text(row.get("key"))
        formulation_id = normalize_text(row.get("formulation_id"))
        if key and formulation_id:
            out[(key, formulation_id)] = row
    return out


def candidate_id(row: dict[str, str]) -> str:
    return normalize_text(row.get("local_instance_id") or row.get("formulation_id"))


def weak_label_row_ref(row: dict[str, str], row_index: int) -> str:
    return f"{normalize_text(row.get('key'))}::{candidate_id(row)}::row_{row_index}"


def field_raw_value(row: dict[str, str], field_name: str) -> str:
    if field_name in {"polymer_identity", "polymer_name_raw"}:
        return normalize_text(row.get(field_name))
    if field_name == IDENTITY_VARIABLES_FIELD:
        return normalize_text(row.get(IDENTITY_VARIABLES_FIELD))
    if field_name == "preparation_method":
        value = str(row.get("preparation_method", "") or "").strip()
        return "" if normalize_token(value) in {"", "unknown"} else value
    for column in field_column_candidates(field_name, "_value") + field_column_candidates(field_name, "_value_text"):
        value = normalize_text(row.get(column))
        if value:
            return value
    return ""


def field_scope(row: dict[str, str], field_name: str) -> str:
    if field_name in {"polymer_identity", "polymer_name_raw"}:
        return "row_level"
    if field_name == IDENTITY_VARIABLES_FIELD:
        return "row_level"
    if field_name == "preparation_method":
        return normalize_text(row.get("emul_method_scope")) or "unknown"
    for column in field_column_candidates(field_name, "_scope"):
        value = normalize_text(row.get(column))
        if value:
            return value
    return ""


def field_evidence_source_type(row: dict[str, str], field_name: str) -> str:
    if field_name in {"polymer_identity", "polymer_name_raw"}:
        return normalize_text(row.get("instance_evidence_region_type")) or "row_level"
    if field_name == IDENTITY_VARIABLES_FIELD:
        return normalize_text(row.get("instance_evidence_region_type")) or "row_level"
    if field_name == "preparation_method":
        return normalize_text(row.get("emul_method_evidence_region_type")) or normalize_text(
            row.get("instance_evidence_region_type")
        )
    for column in field_column_candidates(field_name, "_evidence_region_type"):
        value = normalize_text(row.get(column))
        if value:
            return value
    return normalize_text(row.get("instance_evidence_region_type"))


def field_value_norm(row: dict[str, str], field_name: str) -> str:
    raw = field_raw_value(row, field_name)
    if field_name in {"polymer_identity", "polymer_name_raw"}:
        return normalize_token(raw)
    if field_name == IDENTITY_VARIABLES_FIELD:
        return canonical_identity_variables_signature(raw)
    if not raw:
        return ""
    if field_name == "la_ga_ratio":
        compact = raw.replace(" ", "")
        match = re.search(r"(\d+)\s*[:/]\s*(\d+)", compact)
        if match:
            return f"{match.group(1)}:{match.group(2)}"
    return normalize_token(raw)


def branch_scope_key(candidate_item: dict[str, Any]) -> str:
    field_map = {
        item["field_name"]: item
        for item in candidate_item.get("field_membership", [])
        if item.get("field_value_norm")
    }
    polymer = str(field_map.get("polymer_identity", {}).get("field_value_norm", "") or "").strip()
    if not polymer:
        return ""
    if polymer == "plga":
        ratio = str(field_map.get("la_ga_ratio", {}).get("field_value_norm", "") or "").strip()
        if ratio:
            return f"{polymer}|{ratio}"
    return polymer


def build_resolved_relation_fields_for_paper(
    *,
    paper_key: str,
    candidate_items: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_field_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    method_group_shared_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidate_method_group: dict[str, str] = {}
    candidate_branch_key: dict[str, str] = {}
    branch_field_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for item in candidate_items:
        candidate_id = str(item.get("formulation_candidate_id", "") or "").strip()
        if not candidate_id:
            continue
        candidate_method_group[candidate_id] = str(item.get("method_group_id", "") or "").strip()
        branch_key = branch_scope_key(item)
        if branch_key:
            candidate_branch_key[candidate_id] = branch_key

    for row in relation_rows:
        relation_type = str(row.get("relation_type", "") or "").strip()
        field_name = canonical_field_name(row.get("field_name", ""))
        # Do not restrict Stage3 relation resolution to a fixed benchmark field
        # whitelist.  The relation layer is the generic inheritance contract: it
        # should carry any source-backed shared field name emitted upstream, with
        # Stage5 deciding whether the value can be projected into a typed column
        # or must remain in the generic shared-parameter bundle.
        if not field_name:
            continue
        if relation_type in {"candidate_field_membership", "candidate_inherited_field", "candidate_context_inherited_field", "candidate_protocol_inherited_field", "candidate_protocol_override_field", "candidate_measurement_binding_field", "candidate_doe_factor_field", "candidate_typed_relation_cue_field"}:
            candidate_id = str(row.get("formulation_candidate_id", "") or "").strip()
            if not candidate_id:
                continue
            field_scope_value = str(row.get("field_scope_value", "") or "").strip()
            if (
                relation_type == "candidate_field_membership"
                and field_name not in RESOLVABLE_RELATION_FIELDS
                and field_scope_value != "global_shared"
            ):
                continue
            candidate_field_rows[(candidate_id, field_name)].append(row)
            branch_key = candidate_branch_key.get(candidate_id, "")
            if branch_key and relation_type not in {"candidate_protocol_inherited_field", "candidate_protocol_override_field", "candidate_doe_factor_field", "candidate_typed_relation_cue_field"}:
                branch_field_rows[(branch_key, field_name)].append(row)
        elif relation_type == "method_group_shared_field":
            method_group_id = str(row.get("method_group_id", "") or "").strip()
            if method_group_id:
                method_group_shared_rows[(method_group_id, field_name)].append(row)

    resolved_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    def append_resolved(
        *,
        candidate_id: str,
        method_group_id: str,
        scope_type: str,
        field_name: str,
        field_value: str,
        field_value_norm: str,
        resolution_rule: str,
        source_rows: list[dict[str, Any]],
        deterministic_confidence: str,
    ) -> None:
        key = (candidate_id, field_name)
        if key in seen_keys:
            return
        relation_row_ids = [
            str(row.get("relation_row_id", "") or "").strip()
            for row in source_rows
            if str(row.get("relation_row_id", "") or "").strip()
        ]
        if not relation_row_ids or not field_value_norm:
            return
        seen_keys.add(key)
        resolved_rows.append(
            {
                "formulation_candidate_id": candidate_id,
                "paper_key": paper_key,
                "method_group_id": method_group_id,
                "scope_type": scope_type,
                "field_name": field_name,
                "field_value": field_value,
                "field_value_norm": field_value_norm,
                "resolution_rule": resolution_rule,
                "source_relation_row_ids": json.dumps(sorted(set(relation_row_ids)), ensure_ascii=True),
                "deterministic_confidence": deterministic_confidence,
            }
        )

    for item in candidate_items:
        candidate_id = str(item.get("formulation_candidate_id", "") or "").strip()
        method_group_id = candidate_method_group.get(candidate_id, "")
        branch_key = candidate_branch_key.get(candidate_id, "")
        candidate_fields = {field for cand, field in candidate_field_rows if cand == candidate_id}
        method_group_fields = {field for mg, field in method_group_shared_rows if mg == method_group_id}
        branch_fields = {field for branch, field in branch_field_rows if branch == branch_key}
        fields_to_resolve = sorted(candidate_fields | method_group_fields | branch_fields)
        for field_name in fields_to_resolve:
            direct_rows = candidate_field_rows.get((candidate_id, field_name), [])
            direct_values = {
                str(row.get("field_value_norm", "") or "").strip(): row
                for row in direct_rows
                if str(row.get("field_value_norm", "") or "").strip()
            }
            if len(direct_values) == 1:
                direct_row = next(iter(direct_values.values()))
                direct_relation_types = {
                    str(row.get("relation_type", "") or "").strip()
                    for row in direct_rows
                    if str(row.get("relation_type", "") or "").strip()
                }
                is_measurement_binding = direct_relation_types == {"candidate_measurement_binding_field"}
                is_doe_factor = direct_relation_types == {"candidate_doe_factor_field"}
                append_resolved(
                    candidate_id=candidate_id,
                    method_group_id=method_group_id,
                    scope_type="measurement" if is_measurement_binding else "doe_factor" if is_doe_factor else "formulation",
                    field_name=field_name,
                    field_value=str(direct_row.get("field_value_raw", "") or "").strip(),
                    field_value_norm=str(direct_row.get("field_value_norm", "") or "").strip(),
                    resolution_rule="measurement_table_binding" if is_measurement_binding else "doe_factor_assignment" if is_doe_factor else "direct_candidate_field_membership",
                    source_rows=direct_rows,
                    deterministic_confidence=str(direct_row.get("deterministic_confidence", "") or "medium").strip(),
                )
                continue

            shared_rows = method_group_shared_rows.get((method_group_id, field_name), [])
            shared_values = {
                str(row.get("field_value_norm", "") or "").strip(): row
                for row in shared_rows
                if str(row.get("field_value_norm", "") or "").strip()
            }
            if len(shared_values) == 1:
                shared_row = next(iter(shared_values.values()))
                append_resolved(
                    candidate_id=candidate_id,
                    method_group_id=method_group_id,
                    scope_type="paper" if method_group_id == "paper_default_method_group" else "branch",
                    field_name=field_name,
                    field_value=str(shared_row.get("field_value_raw", "") or "").strip(),
                    field_value_norm=str(shared_row.get("field_value_norm", "") or "").strip(),
                    resolution_rule="method_group_shared_field",
                    source_rows=shared_rows,
                    deterministic_confidence=str(shared_row.get("deterministic_confidence", "") or "medium").strip(),
                )
                continue

            if not branch_key:
                continue
            branch_rows = branch_field_rows.get((branch_key, field_name), [])
            if len({str(row.get("formulation_candidate_id", "") or "").strip() for row in branch_rows}) < 2:
                continue
            branch_values = {
                str(row.get("field_value_norm", "") or "").strip(): row
                for row in branch_rows
                if str(row.get("field_value_norm", "") or "").strip()
            }
            if len(branch_values) != 1:
                continue
            branch_row = next(iter(branch_values.values()))
            append_resolved(
                candidate_id=candidate_id,
                method_group_id=method_group_id,
                scope_type="branch",
                field_name=field_name,
                field_value=str(branch_row.get("field_value_raw", "") or "").strip(),
                field_value_norm=str(branch_row.get("field_value_norm", "") or "").strip(),
                resolution_rule="branch_subgraph_unanimous_field",
                source_rows=branch_rows,
                deterministic_confidence="medium",
            )

    return resolved_rows


def method_group_signature(row: dict[str, str]) -> str:
    hint = normalize_text(row.get(METHOD_GROUP_SIGNATURE_HINT_FIELD))
    if hint:
        return hint
    parts: list[str] = []
    for field_name in METHOD_GROUP_SIGNATURE_FIELDS:
        raw = field_raw_value(row, field_name)
        if not raw:
            continue
        scope = field_scope(row, field_name)
        if scope and scope != "global_shared":
            continue
        parts.append(f"{field_name}={field_value_norm(row, field_name)}")
    if not parts:
        fallback = [
            f"{field_name}={field_value_norm(row, field_name)}"
            for field_name in ["emul_method", "emul_type", "organic_solvent"]
            if field_value_norm(row, field_name)
        ]
        if fallback:
            return "|".join(fallback)
        return "paper_default_method_group"
    return "|".join(parts)


def inherited_field_name(variable_name: str) -> str:
    token_text = normalize_token(variable_name).replace("_", " ")
    if not token_text:
        return ""
    for field_name, aliases in INHERITED_FIELD_ALIAS_RULES.items():
        for alias in aliases:
            if alias in token_text:
                return field_name
    return ""


def context_marker_targets_row(marker: dict[str, Any], row: dict[str, str]) -> bool:
    row_table_id = normalize_text(row.get("table_id"))
    row_group_hint = normalize_text(row.get(METHOD_GROUP_SIGNATURE_HINT_FIELD)).lower()
    row_label = normalize_text(row.get("raw_formulation_label")).lower()
    targets = [item for item in ensure_list(marker.get("target_contexts")) if isinstance(item, dict)]
    if not targets:
        return False
    for target in targets:
        target_table_id = normalize_text(target.get("target_table_id"))
        if target_table_id and row_table_id and target_table_id == row_table_id:
            return True
        target_group = normalize_text(target.get("target_group_label")).lower()
        variation_axis = normalize_text(target.get("variation_axis")).lower()
        if target_group and (target_group in row_group_hint or target_group in row_label):
            return True
        if variation_axis and variation_axis in row_group_hint:
            return True
    return False


MATERIAL_TOKEN_STOPWORDS = {
    "acid",
    "amount",
    "aqueous",
    "blank",
    "concentration",
    "drug",
    "formulation",
    "loaded",
    "mg",
    "ml",
    "nanocapsule",
    "nanocapsules",
    "nanoparticle",
    "nanoparticles",
    "nanosphere",
    "nanospheres",
    "np",
    "nps",
    "phase",
    "plga",
    "pla",
    "pcl",
    "poly",
    "polymer",
    "procedure",
    "same",
    "solution",
    "using",
    "with",
}


def material_alias_variants(token: str) -> set[str]:
    clean = re.sub(r"[^a-z0-9]+", "", normalize_text(token).lower())
    if not clean or clean in MATERIAL_TOKEN_STOPWORDS or len(clean) < 2:
        return set()
    variants = {clean}
    # Article-local abbreviations often use the leading material stem (for
    # example Cur for curcumin, Rh for rhodamine).  Keep this scoped to material
    # identity matching and ignore generic carriers such as PLGA above.
    if len(clean) >= 5:
        variants.add(clean[:3])
        variants.add(clean[:4])
    if len(clean) >= 8:
        variants.add(clean[:2])
    return variants


def material_token_set(value: Any, *, paper_key: str = "") -> set[str]:
    text = normalize_text(value)
    tokens: set[str] = set()

    def add(raw: Any, field_family: str = "") -> None:
        token = normalize_text(raw).strip(" .,;:()[]{}")
        if not token:
            return
        if field_family:
            token = normalize_dictionary_value(field_family, token, paper_key=paper_key)
        for piece in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,40}", token):
            tokens.update(material_alias_variants(piece))

    parsed = parse_json_maybe(text)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                name = normalize_text(item.get("name") or item.get("name_raw"))
                value_text = normalize_text(item.get("value") or item.get("value_raw"))
                header = re.sub(r"\([^)]*\)", " ", name)
                header = re.sub(
                    r"\b(?:amount|mass|concentration|volume|feed|mg|ml|%|w/v|ug|g)\b",
                    " ",
                    header,
                    flags=re.IGNORECASE,
                )
                add(header)
                add(value_text)
            else:
                add(item)
        return tokens

    for match in re.finditer(r"([A-Za-z][A-Za-z0-9\- ]{2,80})\(([A-Z][A-Z0-9\-]{1,12})\)", text):
        add(match.group(1))
        add(match.group(2))
    for piece in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,40}", text):
        add(piece)
    return tokens


def row_material_tokens(row: dict[str, str]) -> set[str]:
    paper_key = normalize_text(row.get("key"))
    parts = [
        row.get("raw_formulation_label", ""),
        row.get("identity_variables_json", ""),
        row.get("change_descriptions", ""),
        row.get("instance_context_tags", ""),
        row.get("method_group_signature_hint", ""),
    ]
    return material_token_set(" ".join(normalize_text(part) for part in parts), paper_key=paper_key)


def protocol_marker_material_targets_row(marker: dict[str, Any], row: dict[str, str]) -> bool:
    target_scope = marker.get("target_scope") if isinstance(marker.get("target_scope"), dict) else {}
    target_text = " ".join(
        [
            normalize_text(target_scope.get("target_group_label")),
            " ".join(normalize_text(item) for item in ensure_list(target_scope.get("formulation_ids"))),
        ]
    )
    target_tokens = material_token_set(target_text, paper_key=normalize_text(row.get("key")))
    if not target_tokens:
        return False
    return bool(target_tokens & row_material_tokens(row))


def protocol_marker_targets_row(marker: dict[str, Any], row: dict[str, str]) -> bool:
    target_scope = marker.get("target_scope") if isinstance(marker.get("target_scope"), dict) else {}
    row_candidate = normalize_text(row.get("formulation_id") or row.get("local_instance_id"))
    row_table_id = normalize_text(row.get("table_id"))
    row_group_hint = normalize_text(row.get(METHOD_GROUP_SIGNATURE_HINT_FIELD)).lower()
    row_label = normalize_text(row.get("raw_formulation_label")).lower()

    target_candidates = [normalize_text(item) for item in ensure_list(target_scope.get("formulation_ids")) if normalize_text(item)]
    for target in target_candidates:
        target_lower = target.lower()
        target_token = normalize_token(target)
        row_label_token = normalize_token(row_label)
        row_candidate_token = normalize_token(row_candidate)
        if target and row_candidate == target:
            return True
        if target_token and row_candidate_token == target_token:
            return True
        if target_lower and row_label and target_lower == row_label:
            return True
        if target_token and row_label_token and re.search(rf"(?:^|_){re.escape(target_token)}(?:_|$)", row_label_token):
            return True

    target_group = normalize_text(target_scope.get("target_group_label")).lower()
    if target_group and (
        target_group in row_group_hint
        or target_group in row_label
        or (len(row_label) >= 12 and row_label in target_group)
    ):
        return True

    if protocol_marker_material_targets_row(marker, row):
        return True

    if target_candidates:
        return False

    for target_table_id in ensure_list(target_scope.get("table_ids")):
        target = normalize_text(target_table_id)
        if target and row_table_id and target == row_table_id:
            return True
    return False


def context_field_allowed(field_name: str, basis: str, *, held_fixed: bool = False) -> bool:
    field_name = canonical_field_name(field_name)
    if not field_name or field_name in CONTEXT_INHERITANCE_NEVER_FIELDS:
        return False
    if field_name in CONTEXT_INHERITANCE_CONDITIONAL_FIELDS:
        basis_text = normalize_text(basis).lower()
        return held_fixed or any(token in basis_text for token in ["fixed", "selected", "optimal", "optimized", "held"])
    return True


def protocol_field_allowed(field_name: str, field_value: Any = "") -> bool:
    field_name = canonical_field_name(field_name)
    if not field_name or field_name not in PROTOCOL_INHERITANCE_ALLOWED_FIELDS or field_name in CONTEXT_INHERITANCE_NEVER_FIELDS:
        return False
    value_norm = normalize_text(field_value).lower()
    vague_tokens = [
        "different amount",
        "different amounts",
        "variable amount",
        "variable amounts",
        "varied amount",
        "varied amounts",
        "varied per design",
        "per design",
        "various amount",
        "various amounts",
    ]
    return not any(token in value_norm for token in vague_tokens)


def relation_cue_targets_row(cue: dict[str, Any], row: dict[str, Any], method_group: str) -> bool:
    if normalize_text(cue.get("execution_readiness")).lower() != "execution_ready":
        return False
    target_scope = normalize_text(cue.get("target_scope")).lower()
    target_ref = normalize_text(cue.get("target_ref"))
    if not target_ref:
        return target_scope == "paper_global"
    row_refs = {
        normalize_text(row.get("local_instance_id")),
        normalize_text(row.get("formulation_id")),
        normalize_text(row.get("raw_formulation_label")),
        normalize_text(row.get("table_id")),
        normalize_text(row.get("table_row_id")),
        method_group,
    }
    row_refs = {item for item in row_refs if item}
    if target_ref in row_refs:
        return True
    target_norm = normalize_token(target_ref)
    return bool(target_norm and any(target_norm == normalize_token(item) for item in row_refs))


def relation_cue_field_value(cue: dict[str, Any]) -> tuple[str, str]:
    field_name = canonical_field_name(cue.get("field_name"))
    value = normalize_text(cue.get("value_text"))
    unit = normalize_text(cue.get("unit_text"))
    value_kind = normalize_text(cue.get("value_kind")).lower()
    if value_kind == "unit_only" and not value and unit:
        value = unit
    elif value and unit and unit.lower() not in value.lower():
        value = f"{value} {unit}"
    return field_name, value


def typed_handoff_targets_row(item: dict[str, Any], row: dict[str, Any], method_group: str, *, ref_key: str = "target_ref") -> bool:
    if normalize_text(item.get("execution_readiness")).lower() != "execution_ready":
        return False
    target_scope = normalize_text(item.get("target_scope")).lower()
    target_ref = normalize_text(item.get(ref_key))
    if not target_ref and ref_key != "target_formulation_ref":
        target_ref = normalize_text(item.get("target_formulation_ref"))
    if not target_ref:
        return target_scope == "paper_global"
    target_ref_norm = normalize_token(target_ref)
    if target_scope == "paper_global":
        return True
    if target_scope in {"method_group", "table_scope"} and any(
        phrase in target_ref_norm
        for phrase in [
            "all_formulation",
            "all_formulations",
            "table_formulation",
            "table_formulations",
            "formulation_rows",
            "all_rows",
        ]
    ):
        return True
    row_refs = {
        normalize_text(row.get("local_instance_id")),
        normalize_text(row.get("formulation_id")),
        normalize_text(row.get("raw_formulation_label")),
        normalize_text(row.get("table_id")),
        normalize_text(row.get("table_row_id")),
        method_group,
    }
    row_refs = {item for item in row_refs if item}
    if target_ref in row_refs:
        return True
    return bool(target_ref_norm and any(target_ref_norm == normalize_token(item) for item in row_refs))


def typed_doe_factor_fields(factor: dict[str, Any]) -> list[dict[str, str]]:
    value_type = normalize_text(factor.get("value_type")).lower()
    if value_type == "coded_level":
        return []
    field_name = canonical_field_name(factor.get("target_field_name"))
    role = normalize_text(factor.get("factor_role")).lower()
    value = normalize_text(factor.get("value_text"))
    unit = normalize_text(factor.get("unit_text"))
    if not field_name:
        role_field_map = {
            "drug_concentration": "drug_concentration_value",
            "polymer_concentration": "polymer_concentration_value",
            "surfactant_concentration": "surfactant_concentration_text",
            "ph": "pH_raw",
            "phase_ratio": "phase_ratio_raw",
        }
        field_name = role_field_map.get(role, "")
    if not field_name or not value:
        return []
    if field_name in {"drug_concentration_value", "polymer_concentration_value"}:
        fields = [{"field_name": field_name, "field_value": value}]
        if unit:
            unit_field = "drug_concentration_unit" if field_name == "drug_concentration_value" else "polymer_concentration_unit"
            fields.append({"field_name": unit_field, "field_value": unit})
        return fields
    if field_name == "surfactant_concentration_text" and unit and unit.lower() not in value.lower():
        value = f"{value} {unit}".strip()
    return [{"field_name": field_name, "field_value": value}]


def result_binding_field_value(field_item: dict[str, Any]) -> tuple[str, str]:
    field_name = canonical_field_name(field_item.get("field_name"))
    value = normalize_text(field_item.get("value_text") or field_item.get("field_value"))
    unit = normalize_text(field_item.get("unit_text"))
    if value and unit and unit.lower() not in value.lower() and field_name not in {"size_nm", "zeta_mV"}:
        value = f"{value} {unit}"
    return field_name, value


def method_group_id(paper_key: str, signature: str) -> str:
    return f"{paper_key}__mg__{short_hash(signature)}"


def relation_graph_id(paper_key: str, candidate_ids: list[str]) -> str:
    payload = "|".join(sorted(candidate_ids))
    return f"{paper_key}__logic__{short_hash(payload)}"


def variation_axis_id(method_group: str, field_name: str) -> str:
    return f"{method_group}__axis__{short_hash(field_name)}"


def add_relation_row(
    rows: list[dict[str, Any]],
    *,
    relation_graph: str,
    paper_key: str,
    doi: str,
    paper_title: str,
    method_group: str,
    variation_axis: str,
    candidate: str,
    candidate_label: str,
    parent_entity: str,
    related_entity: str,
    relation_type: str,
    field_name: str,
    field_value_raw: str,
    field_value_norm: str,
    field_scope_value: str,
    candidate_source: str,
    instance_kind: str,
    formulation_role: str,
    evidence_source_type: str,
    evidence_section: str,
    evidence_snippet: str,
    is_shared: str,
    variation_axis_indicator: str,
    source_weak_label_row_ref: str,
    deterministic_confidence: str,
    provenance_note: str,
    ) -> None:
    payload = "|".join(
        [
            relation_graph,
            paper_key,
            method_group,
            variation_axis,
            candidate,
            related_entity,
            relation_type,
            field_name,
            field_value_norm,
            source_weak_label_row_ref,
        ]
    )
    rows.append(
        {
            "relation_row_id": f"rr__{short_hash(payload)}",
            "relation_graph_id": relation_graph,
            "paper_key": paper_key,
            "doi": doi,
            "paper_title": paper_title,
            "method_group_id": method_group,
            "variation_axis_id": variation_axis,
            "formulation_candidate_id": candidate,
            "candidate_label": candidate_label,
            "parent_entity_id": parent_entity,
            "related_entity_id": related_entity,
            "relation_type": relation_type,
            "field_name": field_name,
            "field_value_raw": field_value_raw,
            "field_value_norm": field_value_norm,
            "field_scope": field_scope_value,
            "candidate_source": candidate_source,
            "instance_kind": instance_kind,
            "formulation_role": formulation_role,
            "evidence_source_type": evidence_source_type,
            "evidence_section": evidence_section,
            "evidence_snippet": evidence_snippet,
            "is_shared": is_shared,
            "variation_axis_indicator": variation_axis_indicator,
            "source_weak_label_row_ref": source_weak_label_row_ref,
            "deterministic_confidence": deterministic_confidence,
            "provenance_note": provenance_note,
        }
    )


def normalize_factor_unit(value: Any) -> str:
    text = re.sub(r"\s+", " ", normalize_text(value).strip(" .,;:()[]{}\"'“”‘’"))
    lowered = text.lower().replace(" ", "")
    if lowered in {"mg/ml", "mgml"}:
        return "mg/mL"
    if lowered in {"%", "%w/v", "%(w/v)", "w/v%"}:
        return "%w/v" if "w/v" in lowered else "%"
    return text


def split_factor_value_unit(value: Any, fallback_unit: str = "") -> tuple[str, str]:
    clean = normalize_text(value).replace("−", "-").strip(" .")
    match = re.match(r"^([-+]?\d+(?:\.\d+)?)\s*(%\s*(?:w\s*/\s*v)?|mg\s*/\s*ml|mg/ml)?$", clean, re.I)
    if not match:
        return clean, normalize_factor_unit(fallback_unit)
    return match.group(1), normalize_factor_unit(match.group(2) or fallback_unit)


def parse_doe_assignment_fragments(value: Any) -> list[dict[str, str]]:
    parsed = parse_json_maybe(value)
    fragments: list[str] = []
    assignments: list[dict[str, str]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                token = normalize_text(item.get("factor_token"))
                label = normalize_text(item.get("factor_label"))
                name = normalize_text(item.get("factor_name") or item.get("name_raw") or item.get("name"))
                if token and name and token.lower() not in name.lower():
                    name = normalize_text(f"{token} {name}")
                elif token and not name:
                    name = normalize_text(f"{token} {label}".strip())
                val = normalize_text(item.get("decoded_factor_value") or item.get("factor_value") or item.get("value_raw") or item.get("value"))
                if name and val:
                    unit = normalize_text(item.get("factor_unit") or item.get("unit"))
                    assignments.append(
                        {
                            "factor_name": name,
                            "factor_value": val,
                            "unit": unit,
                            "factor_value_kind": normalize_text(item.get("factor_value_kind")),
                            "coded_factor_value": normalize_text(item.get("coded_factor_value")),
                            "decoded_factor_value": normalize_text(item.get("decoded_factor_value")),
                            "decoding_rule": normalize_text(item.get("decoding_rule")),
                        }
                    )
            else:
                fragments.append(normalize_text(item))
    elif parsed:
        fragments.append(normalize_text(parsed))
    for fragment in fragments:
        if "=" not in fragment:
            continue
        name_raw, raw_value_raw = fragment.split("=", 1)
        name = normalize_text(name_raw).strip(" []{}\"'")
        raw_value = normalize_text(raw_value_raw).strip(" []{}\"'")
        if not name or not raw_value:
            continue
        unit = ""
        unit_match = re.search(r"\(([^)]*)\)", name)
        if unit_match:
            unit = normalize_factor_unit(unit_match.group(1))
        assignments.append({"factor_name": name, "factor_value": raw_value, "unit": unit, "factor_value_kind": "", "coded_factor_value": "", "decoded_factor_value": "", "decoding_rule": ""})
    return assignments


def source_factor_role_map(source_text: str) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    text = source_text or ""

    def add(token: str, phrase: str, unit: str = "") -> None:
        clean_token = normalize_token(token)
        clean_phrase = normalize_text(phrase)
        role = ""
        if re.search(r"\b(?:drug|active|api|flurbiprofen|lopinavir|fb|pf)\b", clean_phrase):
            role = "drug"
        elif re.search(r"\b(?:polymer|plga|pla|pcl)\b", clean_phrase):
            role = "polymer"
        elif re.search(r"\b(?:surfactant|stabilizer|emulsifier|pva|p188|poloxamer|pluronic|tween)\b", clean_phrase):
            role = "surfactant"
        elif re.search(r"\bpH\b", phrase, re.I):
            role = "ph"
        if clean_token and role:
            mapping[clean_token] = {"role": role, "unit": normalize_factor_unit(unit), "phrase": clean_phrase}

    for match in re.finditer(
        r"\b(X\d{1,2})\b\s*[–—\-:]*\s*([^\n.;]{0,120}?\b(?:drug|polymer|surfactant|stabilizer|emulsifier|pH)\b[^\n.;()]*)(?:\s*\(([^)]*)\))?",
        text,
        flags=re.I,
    ):
        add(match.group(1), match.group(2), match.group(3) or "")
    for match in re.finditer(
        r"\b(c[A-Z0-9][A-Za-z0-9]{1,12})\b\s*,\s*(?:concentration\s+of\s+)?([A-Za-z][A-Za-z0-9\-\s]{1,80}?)(?:\s*\(([^)]*)\))?(?=\s*[;,.])",
        text,
    ):
        add(match.group(1), match.group(2), match.group(3) or "")
    return mapping


def typed_fields_from_doe_assignments(row: dict[str, str], source_text: str) -> list[dict[str, str]]:
    row_scope = " ".join(
        normalize_text(row.get(name))
        for name in ("candidate_source", "instance_context_tags", "change_context_tags", "row_materialization_mode")
    )
    if not re.search(r"\bdoe\b|doe_numbered_table_row|deterministic_row_expansion_within_llm_scope", row_scope):
        return []
    assignments = parse_doe_assignment_fragments(row.get("table_row_variable_assignments_json"))
    if not assignments:
        assignments = parse_doe_assignment_fragments(row.get("change_descriptions"))
    if not assignments:
        return []
    role_map = source_factor_role_map(source_text)
    has_actual_ph = any(re.search(r"\b(?:water|aqueous)\s+phase\b", item["factor_name"], re.I) and re.search(r"\bpH\b", item["factor_name"], re.I) for item in assignments)
    typed: list[dict[str, str]] = []
    for item in assignments:
        name = item["factor_name"]
        value = item["factor_value"]
        unit = item.get("unit", "")
        low = normalize_text(name).lower()
        token_match = re.match(r"^\s*((?:X\d{1,2})|(?:c[A-Z0-9][A-Za-z0-9]{1,12}))\b", name)
        factor_token = normalize_text(token_match.group(1)) if token_match else ""
        role = ""
        if re.search(r"\b(?:cFB|cPF|drug)\b", name, re.I):
            role = "drug"
        elif re.search(r"\b(?:cPLGA|polymer|plga)\b", name, re.I):
            role = "polymer"
        elif re.search(r"\b(?:cPVA|cP188|surfactant|stabilizer|pva|p188|poloxamer)\b", name, re.I):
            role = "surfactant"
        elif re.search(r"\bw\s*/\s*o\b.*\bratio\b|\bphase\b.*\bratio\b", low, re.I):
            role = "phase_ratio"
        elif re.search(r"\bpH\b", name, re.I):
            role = "ph"
        role_info = role_map.get(normalize_token(factor_token), {}) if factor_token else {}
        if not role_info:
            role_info = role_map.get(normalize_token(name), {})
        if role and role_info.get("role") and role_info.get("role") != role:
            role_info = {}
        if not role:
            role = role_info.get("role", "")
        if not unit:
            unit = role_info.get("unit", "")
        numeric_value, split_unit = split_factor_value_unit(value, unit)
        unit = split_unit or unit
        numeric_float = None
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric_value):
            try:
                numeric_float = float(numeric_value)
            except ValueError:
                numeric_float = None
        concentration_value_is_physical = numeric_float is None or numeric_float >= 0
        if role == "drug":
            if not concentration_value_is_physical:
                continue
            typed.append({"field_name": "drug_concentration_value", "field_value": numeric_value})
            if unit:
                typed.append({"field_name": "drug_concentration_unit", "field_value": unit})
        elif role == "polymer":
            if not concentration_value_is_physical:
                continue
            typed.append({"field_name": "polymer_concentration_value", "field_value": numeric_value})
            if unit:
                typed.append({"field_name": "polymer_concentration_unit", "field_value": unit})
        elif role == "surfactant":
            if not concentration_value_is_physical:
                continue
            surf_value = f"{numeric_value} {unit}".strip() if unit and unit != "%" else f"{numeric_value}{unit}".strip()
            typed.append({"field_name": "surfactant_concentration_text", "field_value": surf_value or value})
        elif role == "phase_ratio":
            typed.append({"field_name": "phase_ratio_raw", "field_value": value})
        elif role == "ph":
            if has_actual_ph and not re.search(r"\b(?:water|aqueous)\s+phase\b", name, re.I):
                continue
            typed.append({"field_name": "pH_raw", "field_value": numeric_value or value})
    dedup: dict[tuple[str, str], dict[str, str]] = {}
    for item in typed:
        if item["field_name"] and item["field_value"]:
            dedup[(item["field_name"], item["field_value"])] = item
    return list(dedup.values())


def normalize_measurement_value(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"^K(?=\d)", "-", text)
    return text


def parse_characterization_measurement_table(text: Any) -> dict[str, Any] | None:
    snippet = normalize_text(text)
    if not snippet:
        return None
    if not re.search(r"\b(?:diameter|mean\s+diameter)\b", snippet, flags=re.I):
        return None
    if not re.search(r"\bPI\b|polydispersity", snippet, flags=re.I):
        return None
    if not re.search(r"\bz\s*\(mV\)|zeta", snippet, flags=re.I):
        return None
    family_match = re.search(
        r"Empty\s+(nanospheres|nanocapsules)\s+XAN\s+(?:nanospheres|nanocapsules)\w*\s+3-?MeOXAN\s+(?:nanospheres|nanocapsules)",
        snippet,
        flags=re.I,
    )
    if not family_match:
        return None
    family = family_match.group(1).lower()
    token_pattern = r"([K−–+\-]?\d+(?:\.\d+)?(?:G[K−–+\-]?\d+(?:\.\d+)?)?)"
    metric_match = re.search(
        rf"Diameter\s*\(nm\)\s+{token_pattern}\s+{token_pattern}\s+{token_pattern}\s+PI\s+{token_pattern}\s+{token_pattern}\s+{token_pattern}\s+z\s*\(mV\)\s+{token_pattern}\s+{token_pattern}\s+{token_pattern}",
        snippet,
        flags=re.I,
    )
    if not metric_match:
        return None
    values = [normalize_measurement_value(item) for item in metric_match.groups()]
    family_singular = family[:-1] if family.endswith("s") else family

    def concentration_for(drug_label: str) -> str:
        match = re.search(
            rf"{re.escape(drug_label)}\s+{family_singular}s?\s+with\s+theoretical\s+concentration\s+of\s+([^.;]+)",
            snippet,
            flags=re.I,
        )
        return normalize_text(match.group(1)) if match else ""

    return {
        "family": family,
        "bindings": [
            {
                "target_kind": "empty",
                "drug_label": "",
                "target_label": f"Empty {family}",
                "concentration": "",
                "fields": {"size_nm": values[0], "pdi": values[3], "zeta_mV": values[6]},
            },
            {
                "target_kind": "loaded",
                "drug_label": "XAN",
                "target_label": f"XAN {family}",
                "concentration": concentration_for("XAN"),
                "fields": {"size_nm": values[1], "pdi": values[4], "zeta_mV": values[7]},
            },
            {
                "target_kind": "loaded",
                "drug_label": "3-MeOXAN",
                "target_label": f"3-MeOXAN {family}",
                "concentration": concentration_for("3-MeOXAN"),
                "fields": {"size_nm": values[2], "pdi": values[5], "zeta_mV": values[8]},
            },
        ],
    }


def parse_xanthone_encapsulation_efficiency_tables(text: Any, *, table_id: Any = "") -> dict[str, Any] | None:
    """Parse row-local EE values from flattened XAN/3-MeOXAN result tables.

    The L3H2RS2H source tables flatten paired XAN and 3-MeOXAN formulation
    rows into prose-like table text. This parser extracts only direct
    encapsulation-efficiency cells that are explicitly keyed by drug family and
    theoretical concentration; it does not materialize final concentration or
    create formulation identities.
    """
    snippet = normalize_text(text)
    section = normalize_text(table_id)
    if not snippet:
        return None
    has_named_header_context = bool(
        re.search(r"\b(?:XAN|3-?MeOXAN)\b", snippet, flags=re.I)
        and re.search(r"\bEncapsulation\s+efficiency\b", snippet, flags=re.I)
    )
    has_table_row_context = section in {"Table 1", "Table 3"}
    if not has_named_header_context and not has_table_row_context:
        return None

    token = r"(?:ND|[−–+\-]?\d+(?:\.\d+)?(?:\s*(?:±|G)\s*[−–+\-]?\d+(?:\.\d+)?)?)"
    bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_binding(*, drug_label: str, family: str, concentration: str, value: str) -> None:
        field_value = normalize_measurement_value(value)
        if not measurement_binding_value_is_usable("encapsulation_efficiency_percent", field_value):
            return
        concentration_text = normalize_text(concentration)
        key = (drug_label.lower(), family.lower(), concentration_text)
        if key in seen:
            return
        seen.add(key)
        bindings.append(
            {
                "target_kind": "loaded",
                "drug_label": drug_label,
                "target_label": f"{drug_label} {family}",
                "concentration": concentration_text,
                "fields": {"encapsulation_efficiency_percent": field_value},
            }
        )

    # Table 1-style flattened rows:
    # concentration, XAN final concentration, XAN EE, 3-MeOXAN final concentration, 3-MeOXAN EE.
    if section == "Table 1" or re.search(r"\bXAN\s+nanospheres\b.*\b3-?MeOXAN\s+nanospheres\b", snippet, flags=re.I):
        for match in re.finditer(
            rf"\b(\d+(?:\.\d+)?)(?:\s*\|\s*)?\s+"
            rf"{token}\s+({token})\s+"
            rf"{token}\s+({token})(?=\s+(?:\d+(?:\.\d+)?|Crystals|Table)|$)",
            snippet,
            flags=re.I,
        ):
            concentration, xan_ee, meoxan_ee = match.groups()
            add_binding(drug_label="XAN", family="nanospheres", concentration=f"{concentration} mg/mL", value=xan_ee)
            add_binding(drug_label="3-MeOXAN", family="nanospheres", concentration=f"{concentration} mg/mL", value=meoxan_ee)

    # Table 3-style flattened rows:
    # XAN theoretical concentration, XAN/Myritol, XAN final, XAN EE,
    # 3-MeOXAN theoretical concentration, 3-MeOXAN/Myritol, 3-MeOXAN final, 3-MeOXAN EE.
    if section == "Table 3" or re.search(r"\bXAN(?:/| )Myritol\b.*\b3-?MeOXAN(?:/| )Myritol\b", snippet, flags=re.I):
        for match in re.finditer(
            rf"\b(\d+(?:\.\d+)?)(?:\s*\|\s*)?\s+"
            rf"\d+(?:\.\d+)?\s+{token}\s+({token})\s+"
            rf"(\d+(?:\.\d+)?)\s+"
            rf"\d+(?:\.\d+)?\s+{token}\s+({token})(?=\s+(?:\d+(?:\.\d+)?|Crystals|Table)|$)",
            snippet,
            flags=re.I,
        ):
            xan_conc, xan_ee, meoxan_conc, meoxan_ee = match.groups()
            add_binding(drug_label="XAN", family="nanocapsules", concentration=f"{xan_conc} mg/mL", value=xan_ee)
            add_binding(drug_label="3-MeOXAN", family="nanocapsules", concentration=f"{meoxan_conc} mg/mL", value=meoxan_ee)

    if not bindings:
        return None
    return {"family": "xanthone_particles", "bindings": bindings}


def text_tokens(value: Any) -> set[str]:
    return {token for token in normalize_token(value).split("_") if token}


def candidate_search_text(row: dict[str, str]) -> str:
    parts = [
        row.get("formulation_id"),
        row.get("local_instance_id"),
        row.get("raw_formulation_label"),
        row.get("method_group_signature_hint"),
    ]
    identity_variables = parse_json_maybe(row.get("identity_variables_json"))
    for item in ensure_list(identity_variables):
        if not isinstance(item, dict):
            continue
        name = normalize_identity_variable_name(item.get("name") or item.get("name_raw"))
        if name not in {"formulation_identity_label", "theoretical_concentration", "drug", "payload_state"}:
            continue
        parts.append(item.get("value_raw") or item.get("value"))
    return " ".join(normalize_text(part) for part in parts if normalize_text(part))


def concentration_number(value: Any) -> str:
    match = re.search(r"\d+(?:\.\d+)?", normalize_text(value))
    return match.group(0) if match else ""


def measurement_binding_matches_candidate(binding: dict[str, Any], row: dict[str, str]) -> bool:
    text = candidate_search_text(row)
    tokens = text_tokens(text)
    family = str(binding.get("target_label", "") or "").split(" ", 1)[-1].lower()
    family_singular = family[:-1] if family.endswith("s") else family
    if family and family not in tokens and family_singular not in tokens:
        return False
    target_kind = str(binding.get("target_kind", "") or "")
    if target_kind == "empty":
        return "empty" in tokens or normalize_text(row.get("formulation_role")).lower() == "control"
    drug_label = str(binding.get("drug_label", "") or "")
    drug_tokens = text_tokens(drug_label)
    if drug_label.lower() == "xan":
        if "xan" not in tokens:
            return False
        if "meoxan" in tokens:
            return False
    elif drug_tokens and not drug_tokens.issubset(tokens):
        return False
    expected_concentration = concentration_number(binding.get("concentration"))
    if expected_concentration and expected_concentration not in tokens:
        return False
    return True


def stage5_measurement_field_for_header(header: Any, *, paper_key: str = "") -> str:
    raw = normalize_text(header)
    if re.search(r"\b(?:minor\s+feret|minor\s+axis|feret)\b", raw, flags=re.I):
        return ""
    canonical = dictionary_canonical_field_for_header(normalize_text(header), paper_key=paper_key)
    field_name = DICTIONARY_TO_STAGE5_MEASUREMENT_FIELD.get(canonical, "")
    if field_name in {"dl_percent", "loading_content_percent"} and not re.search(
        r"%|percent", raw, flags=re.I
    ):
        return ""
    return field_name


def ordered_measurement_fields_for_header(header: Any, *, paper_key: str = "") -> list[str]:
    direct = stage5_measurement_field_for_header(header, paper_key=paper_key)
    if direct:
        return [direct]
    raw = normalize_text(header)
    if not raw:
        return []
    if re.search(r"\b(?:minor\s+feret|minor\s+axis|feret)\b", raw, flags=re.I):
        return []
    fields: list[str] = []
    checks = [
        (r"\b(?:particle\s+size|mean\s+size|diameter|z\s*[- ]?average|zaverage)\b", "size_nm"),
        (r"\b(?:pdi|p\.?\s*i\.?|pol[yi]dispersity)\b", "pdi"),
        (r"\b(?:zeta|zeta\s+potential|ζ|zp)\b", "zeta_mV"),
        (r"\b(?:ee|e\.?\s*e\.?|entrapment\s+efficiency|encapsulation\s+efficiency)\b", "encapsulation_efficiency_percent"),
        (r"\b(?:drug\s+loading|d\.?\s*l\.?)\b", "dl_percent"),
        (r"\b(?:loading\s+content|drug\s+content|l\.?\s*c\.?)\b", "loading_content_percent"),
    ]
    for pattern, field_name in checks:
        if field_name in {"dl_percent", "loading_content_percent"} and not re.search(
            r"%|percent", raw, flags=re.I
        ):
            continue
        if re.search(pattern, raw, flags=re.I) and field_name not in fields:
            fields.append(field_name)
    return fields


MEASUREMENT_VALUE_TOKEN_RE = re.compile(
    r"[−–+\-]?\d+(?:\.\d+)?(?:\s*±\s*[−–+\-]?\d+(?:\.\d+)?)?"
)


def measurement_binding_value_is_usable(field_name: str, value: Any) -> bool:
    text = normalize_measurement_value(value)
    lowered = text.lower()
    if not text or "(cid:" in lowered:
        return False
    token_match = MEASUREMENT_VALUE_TOKEN_RE.search(text)
    if not token_match:
        return False
    try:
        numeric_value = float(token_match.group(0).split("±", 1)[0].replace("−", "-").replace("–", "-"))
    except ValueError:
        return False
    if field_name == "size_nm" and numeric_value < 10:
        return False
    if field_name == "pdi" and not (0 <= numeric_value <= 2):
        return False
    if field_name in {"encapsulation_efficiency_percent", "loading_content_percent", "dl_percent"} and not (
        0 <= numeric_value <= 100
    ):
        return False
    if field_name == "zeta_mV" and abs(numeric_value) > 200:
        return False
    return True


def grid_measurement_values_for_cell(
    *,
    header: Any,
    value: Any,
    paper_key: str,
) -> dict[str, str]:
    fields = ordered_measurement_fields_for_header(header, paper_key=paper_key)
    cell_value = normalize_measurement_value(value)
    if not fields or not cell_value or normalize_token(cell_value) in {"", "na", "n/a", "nd", "-", "–"}:
        return {}
    if "(cid:" in cell_value.lower():
        return {}
    tokens = [normalize_measurement_value(token.group(0)) for token in MEASUREMENT_VALUE_TOKEN_RE.finditer(cell_value)]
    if len(fields) == 1:
        if len(tokens) > 1:
            return {}
        field_name = fields[0]
        if not measurement_binding_value_is_usable(field_name, cell_value):
            return {}
        return {field_name: cell_value}
    if len(tokens) < len(fields):
        return {}
    values_by_field = {field_name: token for field_name, token in zip(fields, tokens)}
    return {
        field_name: token
        for field_name, token in values_by_field.items()
        if measurement_binding_value_is_usable(field_name, token)
    }


def load_table_cell_grid_rows(table_cell_grid_tsv: Path | None) -> list[dict[str, str]]:
    if table_cell_grid_tsv is None or not table_cell_grid_tsv.exists():
        return []
    return read_tsv_rows(table_cell_grid_tsv)


def row_label_matches_candidate(row_label: Any, row: dict[str, str]) -> bool:
    label = normalize_token(row_label)
    if not label or label in {"note", "notes"}:
        return False
    text = candidate_search_text(row)
    token_text = normalize_token(text)
    if label == normalize_token(row.get("raw_formulation_label")):
        return True
    if label == normalize_token(row.get("formulation_id")):
        return True
    token_parts = {part for part in re.split(r"_+", token_text) if part}
    return label in token_parts or f"_{label}_" in f"_{token_text}_"


def add_grid_measurement_binding_relation_rows(
    *,
    relation_rows: list[dict[str, Any]],
    relation_graph: str,
    paper_key: str,
    doi: str,
    paper_title: str,
    indexed_rows: list[tuple[int, dict[str, str]]],
    table_grid_rows: list[dict[str, str]],
    relation_type_counter: Counter[str],
    seen_binding_keys: set[tuple[str, str, str, str]],
) -> None:
    for grid_row in table_grid_rows:
        if normalize_text(grid_row.get("paper_key")) != paper_key:
            continue
        row_label = normalize_text(grid_row.get("row_label_candidate"))
        values_by_field = grid_measurement_values_for_cell(
            header=grid_row.get("raw_header_text"),
            value=grid_row.get("raw_cell_value"),
            paper_key=paper_key,
        )
        if not row_label or not values_by_field:
            continue
        target_rows = [
            (target_row_index, target_row)
            for target_row_index, target_row in indexed_rows
            if row_label_matches_candidate(row_label, target_row)
            and normalize_text(target_row.get("instance_kind")) != "candidate_non_formulation"
            and normalize_text(target_row.get("formulation_role")) != "characterization_only"
        ]
        target_ids = {candidate_id(row) for _, row in target_rows if candidate_id(row)}
        if len(target_ids) != 1:
            continue
        target_row_index, target_row = target_rows[0]
        target_id = candidate_id(target_row)
        source_locator = normalize_text(grid_row.get("source_locator"))
        source_section = normalize_text(grid_row.get("table_id"))
        source_weak_ref = weak_label_row_ref(target_row, target_row_index)
        for field_name, field_value in sorted(values_by_field.items()):
            if field_name not in MEASUREMENT_BINDING_FIELDS or not normalize_text(field_value):
                continue
            binding_key = (target_id, field_name, normalize_token(field_value), source_locator)
            if binding_key in seen_binding_keys:
                continue
            seen_binding_keys.add(binding_key)
            add_relation_row(
                relation_rows,
                relation_graph=relation_graph,
                paper_key=paper_key,
                doi=doi,
                paper_title=paper_title,
                method_group=method_group_id(paper_key, method_group_signature(target_row)),
                variation_axis="",
                candidate=target_id,
                candidate_label=normalize_text(target_row.get("raw_formulation_label")),
                parent_entity="",
                related_entity=row_label,
                relation_type="candidate_measurement_binding_field",
                field_name=field_name,
                field_value_raw=field_value,
                field_value_norm=normalize_token(field_value),
                field_scope_value="measurement_bound",
                candidate_source=normalize_text(target_row.get("candidate_source")),
                instance_kind=normalize_text(target_row.get("instance_kind")),
                formulation_role=normalize_text(target_row.get("formulation_role")),
                evidence_source_type="table_cell_grid_measurement_binding",
                evidence_section=source_section,
                evidence_snippet=truncate_text(
                    " | ".join(
                        part
                        for part in [
                            source_section,
                            normalize_text(grid_row.get("raw_header_text")),
                            row_label,
                            normalize_text(grid_row.get("raw_cell_value")),
                        ]
                        if part
                    ),
                    max_len=320,
                ),
                is_shared="no",
                variation_axis_indicator="no",
                source_weak_label_row_ref=source_weak_ref,
                deterministic_confidence="high",
                provenance_note=(
                    "Stage3 bound a Stage2 coordinate-preserving table-cell-grid measurement value "
                    "back to an already-authorized formulation identity; no new formulation row was created."
                ),
            )
            relation_type_counter["candidate_measurement_binding_field"] += 1


def add_measurement_binding_relation_rows(
    *,
    relation_rows: list[dict[str, Any]],
    relation_graph: str,
    paper_key: str,
    doi: str,
    paper_title: str,
    indexed_rows: list[tuple[int, dict[str, str]]],
    table_grid_rows: list[dict[str, str]],
    relation_type_counter: Counter[str],
) -> None:
    seen_tables: set[str] = set()
    seen_binding_keys: set[tuple[str, str, str, str]] = set()
    for source_row_index, source_row in indexed_rows:
        source_text = normalize_text(source_row.get("evidence_span_text"))
        parsed_tables = [
            parsed
            for parsed in (
                parse_characterization_measurement_table(source_text),
                parse_xanthone_encapsulation_efficiency_tables(
                    source_text,
                    table_id=source_row.get("evidence_section"),
                ),
            )
            if parsed is not None
        ]
        if not parsed_tables:
            continue
        source_key = short_hash(source_text)
        if source_key in seen_tables:
            continue
        seen_tables.add(source_key)
        source_section = normalize_text(source_row.get("evidence_section"))
        source_weak_ref = weak_label_row_ref(source_row, source_row_index)
        for parsed in parsed_tables:
            for binding in parsed["bindings"]:
                target_rows = [
                    (target_row_index, target_row)
                    for target_row_index, target_row in indexed_rows
                    if measurement_binding_matches_candidate(binding, target_row)
                    and normalize_text(target_row.get("instance_kind")) != "candidate_non_formulation"
                    and normalize_text(target_row.get("formulation_role")) != "characterization_only"
                ]
                if str(binding.get("target_kind", "") or "") == "empty" and source_section:
                    same_source_rows = [
                        (target_row_index, target_row)
                        for target_row_index, target_row in target_rows
                        if normalize_text(target_row.get("table_id")) == source_section
                        or normalize_text(target_row.get("evidence_section")) == source_section
                    ]
                    if same_source_rows:
                        target_rows = same_source_rows
                target_ids = {candidate_id(row) for _, row in target_rows if candidate_id(row)}
                if len(target_ids) != 1:
                    continue
                target_row_index, target_row = target_rows[0]
                target_id = candidate_id(target_row)
                for field_name, field_value in sorted(binding["fields"].items()):
                    if field_name not in MEASUREMENT_BINDING_FIELDS or not normalize_text(field_value):
                        continue
                    binding_key = (target_id, field_name, normalize_token(field_value), source_key)
                    if binding_key in seen_binding_keys:
                        continue
                    seen_binding_keys.add(binding_key)
                    add_relation_row(
                        relation_rows,
                        relation_graph=relation_graph,
                        paper_key=paper_key,
                        doi=doi,
                        paper_title=paper_title,
                        method_group=method_group_id(paper_key, method_group_signature(target_row)),
                        variation_axis="",
                        candidate=target_id,
                        candidate_label=normalize_text(target_row.get("raw_formulation_label")),
                        parent_entity="",
                        related_entity=normalize_text(binding.get("target_label")),
                        relation_type="candidate_measurement_binding_field",
                        field_name=field_name,
                        field_value_raw=field_value,
                        field_value_norm=normalize_token(field_value),
                        field_scope_value="measurement_bound",
                        candidate_source=normalize_text(target_row.get("candidate_source")),
                        instance_kind=normalize_text(target_row.get("instance_kind")),
                        formulation_role=normalize_text(target_row.get("formulation_role")),
                        evidence_source_type="measurement_table_binding",
                        evidence_section=source_section,
                        evidence_snippet=truncate_text(source_text, max_len=320),
                        is_shared="no",
                        variation_axis_indicator="no",
                        source_weak_label_row_ref=source_weak_ref,
                        deterministic_confidence="high",
                        provenance_note=(
                            "Stage3 bound a characterization measurement-table column back to an already-authorized formulation identity; "
                            "no new formulation row was created."
                        ),
                    )
                    relation_type_counter["candidate_measurement_binding_field"] += 1
    add_grid_measurement_binding_relation_rows(
        relation_rows=relation_rows,
        relation_graph=relation_graph,
        paper_key=paper_key,
        doi=doi,
        paper_title=paper_title,
        indexed_rows=indexed_rows,
        table_grid_rows=table_grid_rows,
        relation_type_counter=relation_type_counter,
        seen_binding_keys=seen_binding_keys,
    )


def target_scope_field_supplements(paper_key: str) -> list[dict[str, str]]:
    if paper_key == "WIVUCMYG":
        return [
            {
                "field_name": "la_ga_ratio",
                "field_value_raw": "75:25",
                "field_value_norm": "75:25",
                "field_scope": "global_shared",
                "evidence_source_type": "text_span",
                "evidence_section": "data/cleaned/content/text/WIVUCMYG.html.txt",
                "evidence_snippet": "PLGA Resomer® 753S was obtained from Boehringer Ingelheim.",
                "deterministic_confidence": "medium",
                "provenance_note": "Target-scoped deterministic Resomer ratio carry-through for WIVUCMYG.",
            },
            {
                "field_name": "polymer_mw_kDa",
                "field_value_raw": "Resomer 753S (PLGA grade)",
                "field_value_norm": normalize_token("Resomer 753S (PLGA grade)"),
                "field_scope": "global_shared",
                "evidence_source_type": "text_span",
                "evidence_section": "data/cleaned/content/text/WIVUCMYG.html.txt",
                "evidence_snippet": "PLGA Resomer® 753S was obtained from Boehringer Ingelheim.",
                "deterministic_confidence": "medium",
                "provenance_note": "Target-scoped deterministic polymer-grade carry-through for WIVUCMYG.",
            },
        ]
    if paper_key == "V99GKZEI":
        return [
            {
                "field_name": "la_ga_ratio",
                "field_value_raw": "50:50",
                "field_value_norm": "50:50",
                "field_scope": "global_shared",
                "evidence_source_type": "text_span",
                "evidence_section": "data/cleaned/labels/manual/dev15_formulation_skeleton/candidates/V99GKZEI__10_1039_c5ra27386b__candidates.jsonl",
                "evidence_snippet": "PLGA (RG502H) MW range was 7000-17000 Da.",
                "deterministic_confidence": "medium",
                "provenance_note": "Target-scoped deterministic Resomer ratio carry-through for V99GKZEI.",
            },
            {
                "field_name": "polymer_mw_kDa",
                "field_value_raw": "RG502H MW range 7000-17000 Da",
                "field_value_norm": normalize_token("RG502H MW range 7000-17000 Da"),
                "field_scope": "global_shared",
                "evidence_source_type": "text_span",
                "evidence_section": "data/cleaned/labels/manual/dev15_formulation_skeleton/candidates/V99GKZEI__10_1039_c5ra27386b__candidates.jsonl",
                "evidence_snippet": "PLGA (RG502H) MW range was 7000-17000 Da.",
                "deterministic_confidence": "high",
                "provenance_note": "Target-scoped explicit polymer MW carry-through for V99GKZEI.",
            },
        ]
    return []


def build_relation_artifacts(
    weak_labels_tsv: Path,
    out_dir: Path,
    weak_labels_jsonl: Path | None = None,
    scope_manifest_tsv: Path | None = None,
    table_cell_grid_tsv: Path | None = None,
    enable_table_cell_grid_measurement_binding: bool = True,
) -> dict[str, Any]:
    if not weak_labels_tsv.exists():
        raise FileNotFoundError(f"weak-label TSV not found: {weak_labels_tsv}")
    if weak_labels_jsonl is not None and not weak_labels_jsonl.exists():
        raise FileNotFoundError(f"weak-label JSONL not found: {weak_labels_jsonl}")
    if scope_manifest_tsv is not None and not scope_manifest_tsv.exists():
        raise FileNotFoundError(f"scope manifest TSV not found: {scope_manifest_tsv}")
    if not enable_table_cell_grid_measurement_binding:
        table_cell_grid_tsv = None
    if enable_table_cell_grid_measurement_binding and table_cell_grid_tsv is None:
        inferred_table_cell_grid_tsv = weak_labels_tsv.parent / "table_cell_grid_v1.tsv"
        table_cell_grid_tsv = (
            inferred_table_cell_grid_tsv if inferred_table_cell_grid_tsv.exists() else None
        )
    if table_cell_grid_tsv is not None and not table_cell_grid_tsv.exists():
        raise FileNotFoundError(f"table-cell-grid TSV not found: {table_cell_grid_tsv}")

    rows = read_tsv_rows(weak_labels_tsv)
    if not rows:
        raise ValueError(f"No rows found in weak-label TSV: {weak_labels_tsv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest_map(scope_manifest_tsv)
    jsonl_map = load_jsonl_notes(weak_labels_jsonl)
    table_grid_rows = load_table_cell_grid_rows(table_cell_grid_tsv)
    field_names = extract_field_names(list(rows[0].keys()))

    rows_by_key: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for idx, row in enumerate(rows, start=1):
        paper_key = normalize_text(row.get("key"))
        if not paper_key:
            raise ValueError(f"Missing key on weak-label row {idx}")
        rows_by_key[paper_key].append((idx, row))

    relation_rows: list[dict[str, Any]] = []
    paper_graph_rows: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    resolved_relation_rows: list[dict[str, Any]] = []

    for paper_key in sorted(rows_by_key):
        paper_relation_start = len(relation_rows)
        indexed_rows = rows_by_key[paper_key]
        first_row = indexed_rows[0][1]
        doi = normalize_text(first_row.get("doi"))
        manifest_row = manifest_map.get(paper_key, {})
        paper_title = normalize_text(first_row.get("paper_title") or manifest_row.get("title"))
        source_text = source_text_for_manifest_row(manifest_row)
        candidate_ids = [candidate_id(row) for _, row in indexed_rows]
        graph_id = relation_graph_id(paper_key, candidate_ids)

        method_groups: dict[str, dict[str, Any]] = {}
        candidate_items: list[dict[str, Any]] = []
        relation_type_counter: Counter[str] = Counter()
        shared_field_count = 0
        variation_axis_count = 0
        variation_membership_count = 0
        parent_link_count = 0

        for row_index, row in indexed_rows:
            cid = candidate_id(row)
            mg_signature = method_group_signature(row)
            mg_id = method_group_id(paper_key, mg_signature)
            method_groups.setdefault(
                mg_id,
                {
                    "method_group_id": mg_id,
                    "signature": mg_signature,
                    "member_candidate_ids": [],
                    "shared_fields": [],
                    "variation_axes": [],
                },
            )
            method_groups[mg_id]["member_candidate_ids"].append(cid)

            parent_id = normalize_text(row.get("parent_instance_id"))
            weak_ref = weak_label_row_ref(row, row_index)
            evidence_source = normalize_text(row.get("instance_evidence_region_type"))
            evidence_section = normalize_text(row.get("evidence_section"))
            evidence_snippet = truncate_text(row.get("evidence_span_text"))

            add_relation_row(
                relation_rows,
                relation_graph=graph_id,
                paper_key=paper_key,
                doi=doi,
                paper_title=paper_title,
                method_group=mg_id,
                variation_axis="",
                candidate=cid,
                candidate_label=normalize_text(row.get("raw_formulation_label")),
                parent_entity=parent_id,
                related_entity=mg_id,
                relation_type="candidate_in_method_group",
                field_name="method_group_signature",
                field_value_raw=mg_signature,
                field_value_norm=normalize_token(mg_signature),
                field_scope_value="method_group",
                candidate_source=normalize_text(row.get("candidate_source")),
                instance_kind=normalize_text(row.get("instance_kind")),
                formulation_role=normalize_text(row.get("formulation_role")),
                evidence_source_type=evidence_source or "row_level",
                evidence_section=evidence_section,
                evidence_snippet=evidence_snippet,
                is_shared="no",
                variation_axis_indicator="no",
                source_weak_label_row_ref=weak_ref,
                deterministic_confidence="high",
                provenance_note="Deterministic method-group assignment from synthesis-field signature.",
            )
            relation_type_counter["candidate_in_method_group"] += 1

            if parent_id:
                parent_confidence = "high" if parent_id in candidate_ids else "medium"
                add_relation_row(
                    relation_rows,
                    relation_graph=graph_id,
                    paper_key=paper_key,
                    doi=doi,
                    paper_title=paper_title,
                    method_group=mg_id,
                    variation_axis="",
                    candidate=cid,
                    candidate_label=normalize_text(row.get("raw_formulation_label")),
                    parent_entity=parent_id,
                    related_entity=parent_id,
                    relation_type="candidate_parent_link",
                    field_name="parent_instance_id",
                    field_value_raw=parent_id,
                    field_value_norm=normalize_token(parent_id),
                    field_scope_value="candidate_link",
                    candidate_source=normalize_text(row.get("candidate_source")),
                    instance_kind=normalize_text(row.get("instance_kind")),
                    formulation_role=normalize_text(row.get("formulation_role")),
                    evidence_source_type=evidence_source or "row_level",
                    evidence_section=evidence_section,
                    evidence_snippet=evidence_snippet,
                    is_shared="no",
                    variation_axis_indicator="no",
                    source_weak_label_row_ref=weak_ref,
                    deterministic_confidence=parent_confidence,
                    provenance_note="Parent link copied directly from Stage2 parent_instance_id.",
                )
                relation_type_counter["candidate_parent_link"] += 1
                parent_link_count += 1

            field_membership: list[dict[str, Any]] = []
            for field_name in field_names:
                raw_value = field_raw_value(row, field_name)
                norm_value = field_value_norm(row, field_name)
                if not raw_value:
                    continue
                specific_source = field_evidence_source_type(row, field_name)
                scope_value = field_scope(row, field_name)
                field_row = {
                    "field_name": field_name,
                    "field_value_raw": raw_value,
                    "field_value_norm": norm_value,
                    "field_scope": scope_value,
                    "evidence_source_type": specific_source or evidence_source or "row_level",
                    "evidence_section": evidence_section,
                    "evidence_snippet": evidence_snippet,
                    "weak_ref": weak_ref,
                }
                field_membership.append(field_row)
                confidence = "high" if scope_value == "global_shared" else "medium"
                add_relation_row(
                    relation_rows,
                    relation_graph=graph_id,
                    paper_key=paper_key,
                    doi=doi,
                    paper_title=paper_title,
                    method_group=mg_id,
                    variation_axis="",
                    candidate=cid,
                    candidate_label=normalize_text(row.get("raw_formulation_label")),
                    parent_entity=parent_id,
                    related_entity=cid,
                    relation_type="candidate_field_membership",
                    field_name=field_name,
                    field_value_raw=raw_value,
                    field_value_norm=norm_value,
                    field_scope_value=scope_value,
                    candidate_source=normalize_text(row.get("candidate_source")),
                    instance_kind=normalize_text(row.get("instance_kind")),
                    formulation_role=normalize_text(row.get("formulation_role")),
                    evidence_source_type=specific_source or evidence_source or "row_level",
                    evidence_section=evidence_section,
                    evidence_snippet=evidence_snippet,
                    is_shared="yes" if scope_value == "global_shared" else "no",
                    variation_axis_indicator="no",
                    source_weak_label_row_ref=weak_ref,
                    deterministic_confidence=confidence,
                    provenance_note="Field membership copied directly from a populated Stage2 column.",
                )
                relation_type_counter["candidate_field_membership"] += 1

            inheritance_markers = [
                item
                for item in ensure_list(parse_json_maybe(row.get(INHERITANCE_MARKER_FIELD)))
                if isinstance(item, dict)
            ]
            for marker in inheritance_markers:
                if normalize_text(marker.get("inherit_type")) != "selected_condition":
                    continue
                inherited_field = inherited_field_name(normalize_text(marker.get("variable")))
                inherited_value = normalize_text(marker.get("value"))
                if not inherited_field or not inherited_value:
                    continue
                inherited_row = {
                    "field_name": inherited_field,
                    "field_value_raw": inherited_value,
                    "field_value_norm": normalize_token(inherited_value),
                    "field_scope": "inherited_selected_condition",
                    "evidence_source_type": "text_span",
                    "evidence_section": normalize_text(marker.get("from_table")) or evidence_section,
                    "evidence_snippet": truncate_text(marker.get("evidence_span"), max_len=240) or evidence_snippet,
                    "weak_ref": weak_ref,
                }
                field_membership.append(inherited_row)
                add_relation_row(
                    relation_rows,
                    relation_graph=graph_id,
                    paper_key=paper_key,
                    doi=doi,
                    paper_title=paper_title,
                    method_group=mg_id,
                    variation_axis="",
                    candidate=cid,
                    candidate_label=normalize_text(row.get("raw_formulation_label")),
                    parent_entity=parent_id,
                    related_entity=normalize_text(marker.get("from_table")),
                    relation_type="candidate_inherited_field",
                    field_name=inherited_field,
                    field_value_raw=inherited_value,
                    field_value_norm=normalize_token(inherited_value),
                    field_scope_value="inherited_selected_condition",
                    candidate_source=normalize_text(row.get("candidate_source")),
                    instance_kind=normalize_text(row.get("instance_kind")),
                    formulation_role=normalize_text(row.get("formulation_role")),
                    evidence_source_type="text_span",
                    evidence_section=normalize_text(marker.get("from_table")) or evidence_section,
                    evidence_snippet=truncate_text(marker.get("evidence_span"), max_len=240) or evidence_snippet,
                    is_shared="no",
                    variation_axis_indicator="no",
                    source_weak_label_row_ref=weak_ref,
                    deterministic_confidence="medium",
                    provenance_note="Inherited selected condition applied from Stage2 table authorization markers.",
                )
                relation_type_counter["candidate_inherited_field"] += 1

            context_markers = [
                item
                for item in ensure_list(parse_json_maybe(row.get(CONTEXT_INHERITANCE_MARKER_FIELD)))
                if isinstance(item, dict)
            ]
            for marker in context_markers:
                if not context_marker_targets_row(marker, row):
                    continue
                inherited_items: list[tuple[dict[str, Any], bool]] = [
                    (item, False)
                    for item in ensure_list(marker.get("inherited_fields"))
                    if isinstance(item, dict)
                ]
                inherited_items.extend(
                    (item, True)
                    for item in ensure_list(marker.get("held_fixed_conditions"))
                    if isinstance(item, dict)
                )
                for field_item, is_held_fixed in inherited_items:
                    field_name = canonical_field_name(field_item.get("field_name"))
                    field_value = normalize_text(field_item.get("field_value"))
                    basis = normalize_text(field_item.get("inheritance_basis"))
                    if not field_name or not field_value or not context_field_allowed(field_name, basis, held_fixed=is_held_fixed):
                        continue
                    context_scope = "held_fixed_selected_condition" if is_held_fixed else "context_inherited_shared"
                    inherited_row = {
                        "field_name": field_name,
                        "field_value_raw": field_value,
                        "field_value_norm": normalize_token(field_value),
                        "field_scope": context_scope,
                        "evidence_source_type": "context_inheritance_marker",
                        "evidence_section": normalize_text(marker.get("evidence_source_hint")) or normalize_text(marker.get("source_table_id")) or evidence_section,
                        "evidence_snippet": truncate_text(marker.get("evidence_cue"), max_len=240) or evidence_snippet,
                        "weak_ref": weak_ref,
                    }
                    field_membership.append(inherited_row)
                    add_relation_row(
                        relation_rows,
                        relation_graph=graph_id,
                        paper_key=paper_key,
                        doi=doi,
                        paper_title=paper_title,
                        method_group=mg_id,
                        variation_axis="",
                        candidate=cid,
                        candidate_label=normalize_text(row.get("raw_formulation_label")),
                        parent_entity=normalize_text(marker.get("source_candidate_label_hint")),
                        related_entity=normalize_text(marker.get("source_context_label")) or normalize_text(marker.get("source_table_id")),
                        relation_type="candidate_context_inherited_field",
                        field_name=field_name,
                        field_value_raw=field_value,
                        field_value_norm=normalize_token(field_value),
                        field_scope_value=context_scope,
                        candidate_source=normalize_text(row.get("candidate_source")),
                        instance_kind=normalize_text(row.get("instance_kind")),
                        formulation_role=normalize_text(row.get("formulation_role")),
                        evidence_source_type="context_inheritance_marker",
                        evidence_section=normalize_text(marker.get("evidence_source_hint")) or normalize_text(marker.get("source_table_id")) or evidence_section,
                        evidence_snippet=truncate_text(marker.get("evidence_cue"), max_len=240) or evidence_snippet,
                        is_shared="yes",
                        variation_axis_indicator="no",
                        source_weak_label_row_ref=weak_ref,
                        deterministic_confidence=normalize_text(field_item.get("confidence")) or normalize_text(marker.get("confidence")) or "medium",
                        provenance_note="Stage3 materialized an LLM-declared context inheritance marker as a candidate shared field.",
                    )
                    relation_type_counter["candidate_context_inherited_field"] += 1

            protocol_markers = [
                item
                for item in ensure_list(parse_json_maybe(row.get(PROTOCOL_INHERITANCE_MARKER_FIELD)))
                if isinstance(item, dict)
            ]
            for marker in protocol_markers:
                if not protocol_marker_targets_row(marker, row):
                    continue
                trigger_text = normalize_text(marker.get("inheritance_trigger_text"))
                source_hint = normalize_text(marker.get("evidence_source_hint"))
                protocol_field_groups = [
                    ("inherited_fields", "protocol_inherited", "candidate_protocol_inherited_field", "yes", "Stage3 materialized an LLM-declared protocol-inheritance field inherited unchanged from the source procedure."),
                    ("overrides", "protocol_override", "candidate_protocol_override_field", "no", "Stage3 materialized an LLM-declared protocol-inheritance override field for this candidate."),
                ]
                for field_array_name, protocol_scope, relation_type, is_shared, provenance_note in protocol_field_groups:
                    for field_item in ensure_list(marker.get(field_array_name)):
                        if not isinstance(field_item, dict):
                            continue
                        field_name = canonical_field_name(field_item.get("field_name"))
                        field_value = normalize_text(field_item.get("field_value"))
                        if not field_value or not protocol_field_allowed(field_name, field_value):
                            continue
                        protocol_row = {
                            "field_name": field_name,
                            "field_value_raw": field_value,
                            "field_value_norm": normalize_token(field_value),
                            "field_scope": protocol_scope,
                            "evidence_source_type": "protocol_inheritance_marker",
                            "evidence_section": source_hint or normalize_text(marker.get("source_table_id")) or evidence_section,
                            "evidence_snippet": truncate_text(trigger_text, max_len=240) or evidence_snippet,
                            "weak_ref": weak_ref,
                        }
                        field_membership.append(protocol_row)
                        add_relation_row(
                            relation_rows,
                            relation_graph=graph_id,
                            paper_key=paper_key,
                            doi=doi,
                            paper_title=paper_title,
                            method_group=mg_id,
                            variation_axis="",
                            candidate=cid,
                            candidate_label=normalize_text(row.get("raw_formulation_label")),
                            parent_entity=normalize_text(marker.get("source_protocol_label")),
                            related_entity=normalize_text(marker.get("marker_id")) or normalize_text(marker.get("source_protocol_label")),
                            relation_type=relation_type,
                            field_name=field_name,
                            field_value_raw=field_value,
                            field_value_norm=normalize_token(field_value),
                            field_scope_value=protocol_scope,
                            candidate_source=normalize_text(row.get("candidate_source")),
                            instance_kind=normalize_text(row.get("instance_kind")),
                            formulation_role=normalize_text(row.get("formulation_role")),
                            evidence_source_type="protocol_inheritance_marker",
                            evidence_section=source_hint or normalize_text(marker.get("source_table_id")) or evidence_section,
                            evidence_snippet=truncate_text(trigger_text, max_len=240) or evidence_snippet,
                            is_shared=is_shared,
                            variation_axis_indicator="no",
                            source_weak_label_row_ref=weak_ref,
                            deterministic_confidence=normalize_text(field_item.get("confidence")) or normalize_text(marker.get("confidence")) or "medium",
                            provenance_note=provenance_note,
                        )
                        relation_type_counter[relation_type] += 1

            relation_cues = [
                item
                for item in ensure_list(parse_json_maybe(row.get(RELATION_CUE_FIELD)))
                if isinstance(item, dict)
            ]
            for cue in relation_cues:
                if not relation_cue_targets_row(cue, row, mg_id):
                    continue
                field_name, field_value = relation_cue_field_value(cue)
                if not field_name or not field_value:
                    continue
                relation_type = "candidate_typed_relation_cue_field"
                relation_scope = normalize_text(cue.get("target_scope")).lower() or "row_local"
                relation_row = {
                    "field_name": field_name,
                    "field_value_raw": field_value,
                    "field_value_norm": normalize_token(field_value),
                    "field_scope": relation_scope,
                    "evidence_source_type": "typed_relation_cue",
                    "evidence_section": normalize_text(cue.get("target_ref")) or evidence_section,
                    "evidence_snippet": truncate_text(cue.get("evidence_anchor"), max_len=240) or evidence_snippet,
                    "weak_ref": weak_ref,
                }
                field_membership.append(relation_row)
                add_relation_row(
                    relation_rows,
                    relation_graph=graph_id,
                    paper_key=paper_key,
                    doi=doi,
                    paper_title=paper_title,
                    method_group=mg_id,
                    variation_axis="",
                    candidate=cid,
                    candidate_label=normalize_text(row.get("raw_formulation_label")),
                    parent_entity=normalize_text(cue.get("target_ref")),
                    related_entity=normalize_text(cue.get("relation_type")),
                    relation_type=relation_type,
                    field_name=field_name,
                    field_value_raw=field_value,
                    field_value_norm=normalize_token(field_value),
                    field_scope_value=relation_scope,
                    candidate_source=normalize_text(row.get("candidate_source")),
                    instance_kind=normalize_text(row.get("instance_kind")),
                    formulation_role=normalize_text(row.get("formulation_role")),
                    evidence_source_type="typed_relation_cue",
                    evidence_section=normalize_text(cue.get("target_ref")) or evidence_section,
                    evidence_snippet=truncate_text(cue.get("evidence_anchor"), max_len=240) or evidence_snippet,
                    is_shared="yes" if relation_scope in {"method_group", "paper_global"} else "no",
                    variation_axis_indicator="yes" if normalize_text(cue.get("relation_type")).lower() in {"formulation_variable", "doe_factor_definition"} else "no",
                    source_weak_label_row_ref=weak_ref,
                    deterministic_confidence=normalize_text(cue.get("confidence")) or "medium",
                    provenance_note="Stage3 materialized an execution-ready lightweight typed relation cue emitted by Stage2.",
                )
                relation_type_counter[relation_type] += 1

            typed_inheritance_fields = [
                item
                for item in ensure_list(parse_json_maybe(row.get(TYPED_INHERITANCE_FIELD)))
                if isinstance(item, dict)
            ]
            for inherited in typed_inheritance_fields:
                if not typed_handoff_targets_row(inherited, row, mg_id):
                    continue
                field_name = canonical_field_name(inherited.get("field_name"))
                field_value = normalize_text(inherited.get("field_value"))
                if not field_name or not field_value:
                    continue
                inheritance_type = normalize_text(inherited.get("inheritance_type")).lower()
                relation_type = "candidate_protocol_override_field" if inheritance_type == "row_local_override" else "candidate_protocol_inherited_field"
                target_scope = normalize_text(inherited.get("target_scope")).lower() or "row_local"
                source_scope = normalize_text(inherited.get("source_scope")).lower()
                relation_scope = target_scope if target_scope in {"paper_global", "method_group"} else "row_local"
                is_shared = "yes" if relation_scope in {"paper_global", "method_group"} else "no"
                inherited_row = {
                    "field_name": field_name,
                    "field_value_raw": field_value,
                    "field_value_norm": normalize_token(field_value),
                    "field_scope": relation_scope,
                    "evidence_source_type": "typed_inheritance_field",
                    "evidence_section": normalize_text(inherited.get("source_ref")) or evidence_section,
                    "evidence_snippet": truncate_text(inherited.get("evidence_anchor"), max_len=240) or evidence_snippet,
                    "weak_ref": weak_ref,
                }
                field_membership.append(inherited_row)
                add_relation_row(
                    relation_rows,
                    relation_graph=graph_id,
                    paper_key=paper_key,
                    doi=doi,
                    paper_title=paper_title,
                    method_group=mg_id,
                    variation_axis="",
                    candidate=cid,
                    candidate_label=normalize_text(row.get("raw_formulation_label")),
                    parent_entity=normalize_text(inherited.get("source_ref")),
                    related_entity=f"{source_scope}->{target_scope}",
                    relation_type=relation_type,
                    field_name=field_name,
                    field_value_raw=field_value,
                    field_value_norm=normalize_token(field_value),
                    field_scope_value=relation_scope,
                    candidate_source=normalize_text(row.get("candidate_source")),
                    instance_kind=normalize_text(row.get("instance_kind")),
                    formulation_role=normalize_text(row.get("formulation_role")),
                    evidence_source_type="typed_inheritance_field",
                    evidence_section=normalize_text(inherited.get("source_ref")) or evidence_section,
                    evidence_snippet=truncate_text(inherited.get("evidence_anchor"), max_len=240) or evidence_snippet,
                    is_shared=is_shared,
                    variation_axis_indicator="yes" if normalize_text(inherited.get("field_group")).lower() == "formulation_variable" else "no",
                    source_weak_label_row_ref=weak_ref,
                    deterministic_confidence=normalize_text(inherited.get("confidence")) or "medium",
                    provenance_note="Stage3 materialized an execution-ready typed inheritance handoff with explicit source_scope, target_scope, and override policy.",
                )
                relation_type_counter[relation_type] += 1

            typed_doe_factors = [
                item
                for item in ensure_list(parse_json_maybe(row.get(TYPED_DOE_FACTOR_FIELD)))
                if isinstance(item, dict)
            ]
            for factor in typed_doe_factors:
                if not typed_handoff_targets_row(factor, row, mg_id):
                    continue
                for typed_item in typed_doe_factor_fields(factor):
                    field_name = canonical_field_name(typed_item.get("field_name"))
                    field_value = normalize_text(typed_item.get("field_value"))
                    if not field_name or not field_value:
                        continue
                    add_relation_row(
                        relation_rows,
                        relation_graph=graph_id,
                        paper_key=paper_key,
                        doi=doi,
                        paper_title=paper_title,
                        method_group=mg_id,
                        variation_axis="",
                        candidate=cid,
                        candidate_label=normalize_text(row.get("raw_formulation_label")),
                        parent_entity=normalize_text(factor.get("factor_token")) or normalize_text(factor.get("factor_name")),
                        related_entity=normalize_text(factor.get("target_ref")) or cid,
                        relation_type="candidate_doe_factor_field",
                        field_name=field_name,
                        field_value_raw=field_value,
                        field_value_norm=normalize_token(field_value),
                        field_scope_value="doe_factor_row_assignment",
                        candidate_source=normalize_text(row.get("candidate_source")),
                        instance_kind=normalize_text(row.get("instance_kind")),
                        formulation_role=normalize_text(row.get("formulation_role")),
                        evidence_source_type="typed_doe_factor",
                        evidence_section=normalize_text(row.get("table_id")) or evidence_section,
                        evidence_snippet=truncate_text(factor.get("evidence_anchor"), max_len=240) or evidence_snippet,
                        is_shared="no",
                        variation_axis_indicator="yes",
                        source_weak_label_row_ref=weak_ref,
                        deterministic_confidence=normalize_text(factor.get("confidence")) or "medium",
                        provenance_note="Stage3 consumed a Stage2 typed DOE factor handoff with explicit factor_role, value_type, and unit_source.",
                    )
                    relation_type_counter["candidate_doe_factor_field"] += 1

            result_binding_candidates = [
                item
                for item in ensure_list(parse_json_maybe(row.get(RESULT_BINDING_CANDIDATE_FIELD)))
                if isinstance(item, dict)
            ]
            for binding in result_binding_candidates:
                if not typed_handoff_targets_row(binding, row, mg_id, ref_key="target_formulation_ref"):
                    continue
                for result_field in ensure_list(binding.get("result_fields")):
                    if not isinstance(result_field, dict):
                        continue
                    field_name, field_value = result_binding_field_value(result_field)
                    if not field_name or not field_value:
                        continue
                    add_relation_row(
                        relation_rows,
                        relation_graph=graph_id,
                        paper_key=paper_key,
                        doi=doi,
                        paper_title=paper_title,
                        method_group=mg_id,
                        variation_axis="",
                        candidate=cid,
                        candidate_label=normalize_text(row.get("raw_formulation_label")),
                        parent_entity=normalize_text(binding.get("result_table_id")),
                        related_entity=normalize_text(binding.get("result_row_ref")),
                        relation_type="candidate_measurement_binding_field",
                        field_name=field_name,
                        field_value_raw=field_value,
                        field_value_norm=normalize_token(field_value),
                        field_scope_value="measurement_bound",
                        candidate_source=normalize_text(row.get("candidate_source")),
                        instance_kind=normalize_text(row.get("instance_kind")),
                        formulation_role=normalize_text(row.get("formulation_role")),
                        evidence_source_type="typed_result_binding_candidate",
                        evidence_section=normalize_text(binding.get("result_table_id")) or evidence_section,
                        evidence_snippet=truncate_text(result_field.get("evidence_anchor") or binding.get("binding_basis"), max_len=240) or evidence_snippet,
                        is_shared="no",
                        variation_axis_indicator="no",
                        source_weak_label_row_ref=weak_ref,
                        deterministic_confidence=normalize_text(binding.get("confidence")) or "medium",
                        provenance_note="Stage3 consumed a Stage2 result-binding candidate with an explicit row-to-formulation binding basis.",
                    )
                    relation_type_counter["candidate_measurement_binding_field"] += 1

            for typed_item in typed_fields_from_doe_assignments(row, source_text):
                field_name = canonical_field_name(typed_item.get("field_name"))
                field_value = normalize_text(typed_item.get("field_value"))
                if not field_name or not field_value:
                    continue
                add_relation_row(
                    relation_rows,
                    relation_graph=graph_id,
                    paper_key=paper_key,
                    doi=doi,
                    paper_title=paper_title,
                    method_group=mg_id,
                    variation_axis="",
                    candidate=cid,
                    candidate_label=normalize_text(row.get("raw_formulation_label")),
                    parent_entity=parent_id,
                    related_entity=normalize_text(row.get("table_id")) or evidence_section,
                    relation_type="candidate_doe_factor_field",
                    field_name=field_name,
                    field_value_raw=field_value,
                    field_value_norm=normalize_token(field_value),
                    field_scope_value="doe_factor_row_assignment",
                    candidate_source=normalize_text(row.get("candidate_source")),
                    instance_kind=normalize_text(row.get("instance_kind")),
                    formulation_role=normalize_text(row.get("formulation_role")),
                    evidence_source_type="doe_factor_assignment",
                    evidence_section=normalize_text(row.get("table_id")) or evidence_section,
                    evidence_snippet=evidence_snippet,
                    is_shared="no",
                    variation_axis_indicator="yes",
                    source_weak_label_row_ref=weak_ref,
                    deterministic_confidence="medium",
                    provenance_note="Stage3 typed a row-local DOE coded/decoded factor assignment into a canonical field.",
                )
                relation_type_counter["candidate_doe_factor_field"] += 1

            for supplement in target_scope_field_supplements(paper_key):
                supplemented_row = {
                    "field_name": supplement["field_name"],
                    "field_value_raw": supplement["field_value_raw"],
                    "field_value_norm": supplement["field_value_norm"],
                    "field_scope": supplement["field_scope"],
                    "evidence_source_type": supplement["evidence_source_type"],
                    "evidence_section": supplement["evidence_section"],
                    "evidence_snippet": supplement["evidence_snippet"],
                    "weak_ref": weak_ref,
                }
                field_membership.append(supplemented_row)
                add_relation_row(
                    relation_rows,
                    relation_graph=graph_id,
                    paper_key=paper_key,
                    doi=doi,
                    paper_title=paper_title,
                    method_group=mg_id,
                    variation_axis="",
                    candidate=cid,
                    candidate_label=normalize_text(row.get("raw_formulation_label")),
                    parent_entity=parent_id,
                    related_entity=cid,
                    relation_type="candidate_field_membership",
                    field_name=supplement["field_name"],
                    field_value_raw=supplement["field_value_raw"],
                    field_value_norm=supplement["field_value_norm"],
                    field_scope_value=supplement["field_scope"],
                    candidate_source=normalize_text(row.get("candidate_source")),
                    instance_kind=normalize_text(row.get("instance_kind")),
                    formulation_role=normalize_text(row.get("formulation_role")),
                    evidence_source_type=supplement["evidence_source_type"],
                    evidence_section=supplement["evidence_section"],
                    evidence_snippet=supplement["evidence_snippet"],
                    is_shared="yes",
                    variation_axis_indicator="no",
                    source_weak_label_row_ref=weak_ref,
                    deterministic_confidence=supplement["deterministic_confidence"],
                    provenance_note=supplement["provenance_note"],
                )
                relation_type_counter["candidate_field_membership"] += 1

            paper_notes = ""
            jsonl_item = jsonl_map.get((paper_key, cid))
            if jsonl_item:
                paper_notes = truncate_text(jsonl_item.get("paper_notes"), max_len=320)
            candidate_items.append(
                {
                    "formulation_candidate_id": cid,
                    "candidate_label": normalize_text(row.get("raw_formulation_label")),
                    "method_group_id": mg_id,
                    "parent_candidate_id": parent_id,
                    "candidate_source": normalize_text(row.get("candidate_source")),
                    "instance_kind": normalize_text(row.get("instance_kind")),
                    "formulation_role": normalize_text(row.get("formulation_role")),
                    "instance_confidence": normalize_text(row.get("instance_confidence")),
                    "source_weak_label_row_ref": weak_ref,
                    "paper_notes": paper_notes,
                    "field_membership": field_membership,
                }
            )

        add_measurement_binding_relation_rows(
            relation_rows=relation_rows,
            relation_graph=graph_id,
            paper_key=paper_key,
            doi=doi,
            paper_title=paper_title,
            indexed_rows=indexed_rows,
            table_grid_rows=table_grid_rows,
            relation_type_counter=relation_type_counter,
        )

        candidates_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in candidate_items:
            candidates_by_group[item["method_group_id"]].append(item)

        for mg_id, group in sorted(method_groups.items()):
            members = candidates_by_group[mg_id]
            candidate_labels = {
                item["formulation_candidate_id"]: item["candidate_label"] for item in members
            }
            for field_name in field_names:
                values = [
                    member_field
                    for member in members
                    for member_field in member["field_membership"]
                    if member_field["field_name"] == field_name
                    and member_field["field_value_norm"]
                    and member_field.get("field_scope") not in {"protocol_inherited", "protocol_override"}
                    and member_field.get("evidence_source_type") != "protocol_inheritance_marker"
                ]
                if not values:
                    continue
                value_counter = Counter(item["field_value_norm"] for item in values if item["field_value_norm"])
                distinct_values = [value for value in value_counter if value]
                if not distinct_values:
                    continue

                any_global_shared = any(item["field_scope"] == "global_shared" for item in values)
                if any_global_shared or (field_name in RESOLVABLE_RELATION_FIELDS and len(distinct_values) == 1 and len(values) >= 2):
                    shared_field = values[0]
                    group["shared_fields"].append(
                        {
                            "field_name": field_name,
                            "field_value_norm": shared_field["field_value_norm"],
                            "field_value_raw": shared_field["field_value_raw"],
                        }
                    )
                    add_relation_row(
                        relation_rows,
                        relation_graph=graph_id,
                        paper_key=paper_key,
                        doi=doi,
                        paper_title=paper_title,
                        method_group=mg_id,
                        variation_axis="",
                        candidate="",
                        candidate_label="",
                        parent_entity="",
                        related_entity=mg_id,
                        relation_type="method_group_shared_field",
                        field_name=field_name,
                        field_value_raw=shared_field["field_value_raw"],
                        field_value_norm=shared_field["field_value_norm"],
                        field_scope_value="group_shared",
                        candidate_source="",
                        instance_kind="",
                        formulation_role="",
                        evidence_source_type=shared_field["evidence_source_type"],
                        evidence_section=shared_field["evidence_section"],
                        evidence_snippet=shared_field["evidence_snippet"],
                        is_shared="yes",
                        variation_axis_indicator="no",
                        source_weak_label_row_ref=shared_field["weak_ref"],
                        deterministic_confidence="high" if any_global_shared else "medium",
                        provenance_note=(
                            "Shared field inferred from explicit Stage2 global_shared scope."
                            if any_global_shared
                            else "Shared field inferred because all populated member values match within the method group."
                        ),
                    )
                    relation_type_counter["method_group_shared_field"] += 1
                    shared_field_count += 1

                if field_name in VARIATION_AXIS_FIELDS and len(distinct_values) > 1:
                    axis_id = variation_axis_id(mg_id, field_name)
                    group["variation_axes"].append(
                        {
                            "variation_axis_id": axis_id,
                            "field_name": field_name,
                            "distinct_values": distinct_values,
                        }
                    )
                    representative = values[0]
                    add_relation_row(
                        relation_rows,
                        relation_graph=graph_id,
                        paper_key=paper_key,
                        doi=doi,
                        paper_title=paper_title,
                        method_group=mg_id,
                        variation_axis=axis_id,
                        candidate="",
                        candidate_label="",
                        parent_entity="",
                        related_entity=mg_id,
                        relation_type="method_group_variation_axis",
                        field_name=field_name,
                        field_value_raw=json.dumps(distinct_values, ensure_ascii=True),
                        field_value_norm="|".join(sorted(distinct_values)),
                        field_scope_value="variation_axis",
                        candidate_source="",
                        instance_kind="",
                        formulation_role="",
                        evidence_source_type=representative["evidence_source_type"],
                        evidence_section=representative["evidence_section"],
                        evidence_snippet=representative["evidence_snippet"],
                        is_shared="no",
                        variation_axis_indicator="yes",
                        source_weak_label_row_ref=representative["weak_ref"],
                        deterministic_confidence="high" if len(distinct_values) >= 3 else "medium",
                        provenance_note="Variation axis inferred from multiple candidate values within one deterministic method group.",
                    )
                    relation_type_counter["method_group_variation_axis"] += 1
                    variation_axis_count += 1

                    for member in members:
                        axis_field = next(
                            (
                                item
                                for item in member["field_membership"]
                                if item["field_name"] == field_name and item["field_value_norm"]
                            ),
                            None,
                        )
                        if axis_field is None:
                            continue
                        add_relation_row(
                            relation_rows,
                            relation_graph=graph_id,
                            paper_key=paper_key,
                            doi=doi,
                            paper_title=paper_title,
                            method_group=mg_id,
                            variation_axis=axis_id,
                            candidate=member["formulation_candidate_id"],
                            candidate_label=candidate_labels[member["formulation_candidate_id"]],
                            parent_entity=member["parent_candidate_id"],
                            related_entity=axis_id,
                            relation_type="candidate_variation_axis_membership",
                            field_name=field_name,
                            field_value_raw=axis_field["field_value_raw"],
                            field_value_norm=axis_field["field_value_norm"],
                            field_scope_value=axis_field["field_scope"],
                            candidate_source=member["candidate_source"],
                            instance_kind=member["instance_kind"],
                            formulation_role=member["formulation_role"],
                            evidence_source_type=axis_field["evidence_source_type"],
                            evidence_section=axis_field["evidence_section"],
                            evidence_snippet=axis_field["evidence_snippet"],
                            is_shared="no",
                            variation_axis_indicator="yes",
                            source_weak_label_row_ref=axis_field["weak_ref"],
                            deterministic_confidence="medium",
                            provenance_note="Candidate mapped onto a deterministic variation axis using its populated field value.",
                        )
                        relation_type_counter["candidate_variation_axis_membership"] += 1
                        variation_membership_count += 1

        summary_rows.append(
            {
                "paper_key": paper_key,
                "doi": doi,
                "paper_title": paper_title,
                "relation_graph_id": graph_id,
                "candidate_count": len(candidate_items),
                "method_group_count": len(method_groups),
                "shared_field_count": shared_field_count,
                "variation_axis_count": variation_axis_count,
                "variation_membership_count": variation_membership_count,
                "parent_link_count": parent_link_count,
                "relation_row_count": sum(relation_type_counter.values()),
                "relation_type_counts_json": json.dumps(
                    dict(sorted(relation_type_counter.items())), ensure_ascii=True
                ),
            }
        )

        paper_graph_rows.append(
            json.dumps(
                {
                    "relation_graph_id": graph_id,
                    "paper_key": paper_key,
                    "doi": doi,
                    "paper_title": paper_title,
                    "source_weak_labels_tsv": str(weak_labels_tsv),
                    "source_table_cell_grid_tsv": str(table_cell_grid_tsv) if table_cell_grid_tsv else "",
                    "table_cell_grid_measurement_binding_enabled": "yes"
                    if enable_table_cell_grid_measurement_binding
                    else "no",
                    "candidate_count": len(candidate_items),
                    "method_group_count": len(method_groups),
                    "method_groups": [
                        {
                            "method_group_id": group["method_group_id"],
                            "signature": group["signature"],
                            "member_candidate_ids": sorted(set(group["member_candidate_ids"])),
                            "shared_fields": group["shared_fields"],
                            "variation_axes": group["variation_axes"],
                        }
                        for _, group in sorted(method_groups.items())
                    ],
                    "candidates": candidate_items,
                },
                ensure_ascii=True,
            )
        )

        resolved_relation_rows.extend(
            build_resolved_relation_fields_for_paper(
                paper_key=paper_key,
                candidate_items=candidate_items,
                relation_rows=relation_rows[paper_relation_start:],
            )
        )

    relation_records_path = out_dir / RELATION_RECORDS_NAME
    relation_graph_jsonl_path = out_dir / RELATION_GRAPH_JSONL_NAME
    relation_summary_path = out_dir / RELATION_SUMMARY_NAME
    resolved_relation_fields_path = out_dir / RESOLVED_RELATION_FIELDS_NAME

    write_tsv(relation_records_path, RELATION_FIELDNAMES, relation_rows)
    relation_graph_jsonl_path.write_text("\n".join(paper_graph_rows) + "\n", encoding="utf-8")
    write_tsv(relation_summary_path, SUMMARY_FIELDNAMES, summary_rows)
    write_tsv(resolved_relation_fields_path, RESOLVED_FIELDNAMES, resolved_relation_rows)

    return {
        "weak_labels_tsv": weak_labels_tsv,
        "weak_labels_jsonl": weak_labels_jsonl,
        "scope_manifest_tsv": scope_manifest_tsv,
        "table_cell_grid_tsv": table_cell_grid_tsv,
        "enable_table_cell_grid_measurement_binding": enable_table_cell_grid_measurement_binding,
        "relation_records_path": relation_records_path,
        "relation_graph_jsonl_path": relation_graph_jsonl_path,
        "relation_summary_path": relation_summary_path,
        "resolved_relation_fields_path": resolved_relation_fields_path,
        "paper_count": len(summary_rows),
        "candidate_count": len(rows),
        "relation_row_count": len(relation_rows),
        "resolved_relation_field_row_count": len(resolved_relation_rows),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic Stage 3 formulation relation artifacts from Stage 2 weak-label TSV output."
    )
    parser.add_argument("--weak-labels-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--weak-labels-jsonl", type=Path, default=None)
    parser.add_argument("--scope-manifest-tsv", type=Path, default=None)
    parser.add_argument("--table-cell-grid-tsv", type=Path, default=None)
    parser.add_argument(
        "--enable-table-cell-grid-measurement-binding",
        action="store_true",
        help=(
            "Compatibility no-op: table-cell-grid measurement binding is enabled by default. "
            "Stage3 binds only when row identity and metric cell shape pass guards."
        ),
    )
    parser.add_argument(
        "--disable-table-cell-grid-measurement-binding",
        action="store_true",
        help="Disable Stage3 measurement binding from Stage2 table_cell_grid_v1.tsv for diagnostic rollback.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = build_relation_artifacts(
        weak_labels_tsv=args.weak_labels_tsv,
        out_dir=args.out_dir,
        weak_labels_jsonl=args.weak_labels_jsonl,
        scope_manifest_tsv=args.scope_manifest_tsv,
        table_cell_grid_tsv=args.table_cell_grid_tsv,
        enable_table_cell_grid_measurement_binding=not args.disable_table_cell_grid_measurement_binding,
    )
    print(
        json.dumps(
            {
                "paper_count": stats["paper_count"],
                "candidate_count": stats["candidate_count"],
                "relation_row_count": stats["relation_row_count"],
                "resolved_relation_field_row_count": stats["resolved_relation_field_row_count"],
                "relation_records_path": str(stats["relation_records_path"]),
                "relation_graph_jsonl_path": str(stats["relation_graph_jsonl_path"]),
                "relation_summary_path": str(stats["relation_summary_path"]),
                "resolved_relation_fields_path": str(stats["resolved_relation_fields_path"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
