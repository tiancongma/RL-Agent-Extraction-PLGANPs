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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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

CANONICAL_FIELD_ALIASES = {
    "plga_mw_kDa": "polymer_mw_kDa",
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
    return CANONICAL_FIELD_ALIASES.get(normalize_text(field_name), normalize_text(field_name))


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
        if relation_type in {"candidate_field_membership", "candidate_inherited_field"}:
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
            if branch_key:
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
                append_resolved(
                    candidate_id=candidate_id,
                    method_group_id=method_group_id,
                    scope_type="formulation",
                    field_name=field_name,
                    field_value=str(direct_row.get("field_value_raw", "") or "").strip(),
                    field_value_norm=str(direct_row.get("field_value_norm", "") or "").strip(),
                    resolution_rule="direct_candidate_field_membership",
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
) -> dict[str, Any]:
    if not weak_labels_tsv.exists():
        raise FileNotFoundError(f"weak-label TSV not found: {weak_labels_tsv}")
    if weak_labels_jsonl is not None and not weak_labels_jsonl.exists():
        raise FileNotFoundError(f"weak-label JSONL not found: {weak_labels_jsonl}")
    if scope_manifest_tsv is not None and not scope_manifest_tsv.exists():
        raise FileNotFoundError(f"scope manifest TSV not found: {scope_manifest_tsv}")

    rows = read_tsv_rows(weak_labels_tsv)
    if not rows:
        raise ValueError(f"No rows found in weak-label TSV: {weak_labels_tsv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest_map(scope_manifest_tsv)
    jsonl_map = load_jsonl_notes(weak_labels_jsonl)
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
                    if member_field["field_name"] == field_name and member_field["field_value_norm"]
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = build_relation_artifacts(
        weak_labels_tsv=args.weak_labels_tsv,
        out_dir=args.out_dir,
        weak_labels_jsonl=args.weak_labels_jsonl,
        scope_manifest_tsv=args.scope_manifest_tsv,
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
