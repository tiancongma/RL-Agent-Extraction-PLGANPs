#!/usr/bin/env python3
from __future__ import annotations

"""
Build a deterministic explicit-only Step 2 value backfill table from a frozen Step 1 final table.

Contract:
- downstream of `final_formulation_table_v1.tsv`
- preserves frozen final-row identity and membership
- fills values only when explicit support exists in frozen-final or relation-carried surfaces
- does not create, split, merge, or delete formulations
- does not donor-fill or use LLM/external APIs
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_TABLE_NAME = "step2_value_backfill_table_v1.tsv"
OUTPUT_EVIDENCE_NAME = "step2_value_backfill_evidence_v1.tsv"
OUTPUT_SUMMARY_NAME = "step2_value_backfill_summary_v1.md"
TABLE_LIKE_EVIDENCE_TYPES = {"table_row", "table_cell", "table_header"}


@dataclass(frozen=True)
class FieldSpec:
    output_field: str
    final_value_column: str = ""
    final_raw_text_column: str = ""
    final_evidence_region_column: str = ""
    final_missing_reason_column: str = ""
    relation_field_name: str = ""


FIELD_SPECS = [
    FieldSpec("polymer_mw_kDa", "polymer_mw_kDa_value", "polymer_mw_kDa_value_text", "polymer_mw_kDa_evidence_region_type", "polymer_mw_kDa_missing_reason", "polymer_mw_kDa"),
    FieldSpec("la_ga_ratio", "la_ga_ratio_value", "la_ga_ratio_value_text", "la_ga_ratio_evidence_region_type", "la_ga_ratio_missing_reason", "la_ga_ratio"),
    FieldSpec("surfactant_name", "surfactant_name_value", "surfactant_name_value_text", "surfactant_name_evidence_region_type", "surfactant_name_missing_reason", "surfactant_name"),
    FieldSpec("surfactant_concentration", "surfactant_concentration_text_value", "surfactant_concentration_text_value_text", "surfactant_concentration_text_evidence_region_type", "surfactant_concentration_text_missing_reason"),
    FieldSpec("organic_solvent", "organic_solvent_value", "organic_solvent_value_text", "organic_solvent_evidence_region_type", "organic_solvent_missing_reason", "organic_solvent"),
    FieldSpec("drug_name", "drug_name_value", "drug_name_value_text", "drug_name_evidence_region_type", "drug_name_missing_reason"),
    FieldSpec("drug_feed_amount", "drug_feed_amount_text_value", "drug_feed_amount_text_value_text", "drug_feed_amount_text_evidence_region_type", "drug_feed_amount_text_missing_reason"),
    FieldSpec("polymer_amount", "plga_mass_mg_value", "plga_mass_mg_value_text", "plga_mass_mg_evidence_region_type", "plga_mass_mg_missing_reason"),
    FieldSpec("phase_ratio"),
    FieldSpec("encapsulation_efficiency_percent", "encapsulation_efficiency_percent_value", "encapsulation_efficiency_percent_value_text", "encapsulation_efficiency_percent_evidence_region_type", "encapsulation_efficiency_percent_missing_reason"),
    FieldSpec("loading_capacity_percent", "loading_content_percent_value", "loading_content_percent_value_text", "loading_content_percent_evidence_region_type", "loading_content_percent_missing_reason"),
    FieldSpec("particle_size_nm", "size_nm_value", "size_nm_value_text", "size_nm_evidence_region_type", "size_nm_missing_reason"),
    FieldSpec("pdi", "pdi_value", "pdi_value_text", "pdi_evidence_region_type", "pdi_missing_reason"),
    FieldSpec("zeta_potential_mV", "zeta_mV_value", "zeta_mV_value_text", "zeta_mV_evidence_region_type", "zeta_mV_missing_reason"),
]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in fieldnames})


def parse_json_object_list(value: Any) -> list[dict[str, str]]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [{str(key): normalize_text(item_value) for key, item_value in item.items()} for item in parsed if isinstance(item, dict)]


def parse_json_string_list(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [normalize_text(item) for item in parsed if normalize_text(item)]


def optional_rows(path_text: str) -> list[dict[str, str]]:
    path = Path(path_text).resolve() if path_text else None
    if path is None or not path.exists():
        return []
    return read_tsv_rows(path)


def binding_rows(path_text: str) -> list[dict[str, str]]:
    path = Path(path_text).resolve() if path_text else None
    if path is None or not path.exists():
        return []
    return read_tsv_rows(path)


def optional_path_text(path_value: Path) -> str:
    text = normalize_text(path_value)
    if text in {"", "."}:
        return ""
    return str(path_value.resolve())


def parse_mass_to_mg(value: Any) -> float | None:
    text = normalize_text(value).lower().replace(",", "")
    if not text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(mg|g|ug|mcg|ng)?\b", text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = normalize_text(match.group(2)) or "mg"
    if unit == "g":
        return amount * 1000.0
    if unit in {"ug", "mcg"}:
        return amount / 1000.0
    if unit == "ng":
        return amount / 1_000_000.0
    return amount


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return normalize_text(value)


def first_evidence_ref(row: dict[str, str]) -> dict[str, str]:
    refs = parse_json_object_list(row.get("supporting_evidence_refs"))
    return refs[0] if refs else {}


def build_relation_indexes(
    resolved_relation_rows: list[dict[str, str]],
    relation_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[str, dict[str, str]]]:
    resolved_index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in resolved_relation_rows:
        key = (
            normalize_text(row.get("paper_key")),
            normalize_text(row.get("formulation_candidate_id")),
            normalize_text(row.get("field_name")),
        )
        if key not in resolved_index:
            resolved_index[key] = row
    relation_index = {normalize_text(row.get("relation_row_id")): row for row in relation_rows if normalize_text(row.get("relation_row_id"))}
    return resolved_index, relation_index


def relation_support(
    *,
    paper_key: str,
    candidate_id: str,
    field_name: str,
    relation_index: dict[tuple[str, str, str], dict[str, str]],
    relation_rows_by_id: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    resolved_row = relation_index.get((paper_key, candidate_id, field_name))
    if resolved_row is None:
        return None
    relation_row_ids = parse_json_string_list(resolved_row.get("source_relation_row_ids"))
    evidence_row = relation_rows_by_id.get(relation_row_ids[0], {}) if relation_row_ids else {}
    return {
        "value": normalize_text(resolved_row.get("field_value")),
        "raw_text": normalize_text(resolved_row.get("field_value")),
        "support_status": "relation_carried_explicit",
        "evidence_source_type": "relation_resolved_field",
        "evidence_text": normalize_text(evidence_row.get("evidence_snippet")) or normalize_text(resolved_row.get("field_value")),
        "evidence_locator": normalize_text(evidence_row.get("evidence_section")) or normalize_text(resolved_row.get("source_relation_row_ids")),
    }


def direct_support(row: dict[str, str], spec: FieldSpec) -> dict[str, str] | None:
    value = normalize_text(row.get(spec.final_value_column))
    raw_text = normalize_text(row.get(spec.final_raw_text_column)) or value
    field_evidence_type = normalize_text(row.get(spec.final_evidence_region_column))
    missing_reason = normalize_text(row.get(spec.final_missing_reason_column))
    ref = first_evidence_ref(row)
    evidence_text = normalize_text(ref.get("supporting_snippet")) or normalize_text(row.get("evidence_span_text")) or raw_text
    evidence_locator = normalize_text(ref.get("source_locator_text")) or normalize_text(row.get("evidence_section"))
    evidence_source_type = field_evidence_type or normalize_text(row.get("instance_evidence_region_type"))

    if not raw_text and not value:
        if missing_reason and "not_projectable" in missing_reason.lower():
            return {
                "value": "",
                "raw_text": "",
                "support_status": "unsupported_text",
                "evidence_source_type": evidence_source_type,
                "evidence_text": evidence_text,
                "evidence_locator": evidence_locator,
            }
        return None

    if field_evidence_type in TABLE_LIKE_EVIDENCE_TYPES and not normalize_text(row.get("table_row_id")):
        return {
            "value": "",
            "raw_text": raw_text,
            "support_status": "unresolved_table",
            "evidence_source_type": field_evidence_type,
            "evidence_text": evidence_text,
            "evidence_locator": evidence_locator,
        }

    return {
        "value": value or raw_text,
        "raw_text": raw_text,
        "support_status": "explicit_supported",
        "evidence_source_type": evidence_source_type or "text_span",
        "evidence_text": evidence_text,
        "evidence_locator": evidence_locator,
    }


def binding_support(
    *,
    row: dict[str, str],
    spec: FieldSpec,
    binding_index: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str] | None:
    binding_row = binding_index.get((normalize_text(row.get("final_formulation_id")), spec.output_field))
    if binding_row is None:
        return None
    if normalize_text(binding_row.get("binding_status")) != "resolved_row_local":
        return None
    value = normalize_text(binding_row.get("source_value_normalized")) or normalize_text(binding_row.get("source_value_text"))
    raw_text = normalize_text(binding_row.get("source_value_text")) or value
    if not value and not raw_text:
        return None
    return {
        "value": value,
        "raw_text": raw_text,
        "support_status": "explicit_supported",
        "evidence_source_type": "table_row_binding_v1",
        "evidence_text": normalize_text(binding_row.get("source_row_text")) or raw_text,
        "evidence_locator": normalize_text(binding_row.get("source_table_row_id")) or normalize_text(binding_row.get("source_table_id")),
    }


def parameter_binding_support(
    *,
    row: dict[str, str],
    spec: FieldSpec,
    parameter_binding_index: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str] | None:
    binding_row = parameter_binding_index.get((normalize_text(row.get("final_formulation_id")), spec.output_field))
    if binding_row is None:
        return None
    if normalize_text(binding_row.get("binding_status")) not in {
        "resolved_relation_context",
        "resolved_shared_context",
        "resolved_article_native_match",
    }:
        return None
    value = normalize_text(binding_row.get("source_value_normalized")) or normalize_text(binding_row.get("source_value_text"))
    raw_text = normalize_text(binding_row.get("source_value_text")) or value
    if not value and not raw_text:
        return None
    return {
        "value": value,
        "raw_text": raw_text,
        "support_status": "relation_carried_explicit",
        "evidence_source_type": normalize_text(binding_row.get("source_type")) or "parameter_binding_v1",
        "evidence_text": normalize_text(binding_row.get("source_value_text")) or raw_text,
        "evidence_locator": normalize_text(binding_row.get("source_locator")) or normalize_text(binding_row.get("binding_rule_used")),
    }


def blank_support() -> dict[str, str]:
    return {
        "value": "",
        "raw_text": "",
        "support_status": "blank_not_reported",
        "evidence_source_type": "",
        "evidence_text": "",
        "evidence_locator": "",
    }


def choose_field_support(
    *,
    row: dict[str, str],
    spec: FieldSpec,
    relation_index: dict[tuple[str, str, str], dict[str, str]],
    relation_rows_by_id: dict[str, dict[str, str]],
    binding_index: dict[tuple[str, str], dict[str, str]],
    parameter_binding_index: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str]:
    direct = direct_support(row, spec) if spec.final_value_column else None
    if direct is not None and direct["support_status"] == "explicit_supported":
        return direct
    if direct is not None and direct["support_status"] == "unresolved_table":
        bound = binding_support(row=row, spec=spec, binding_index=binding_index)
        if bound is not None:
            return bound
    parameter_bound = parameter_binding_support(row=row, spec=spec, parameter_binding_index=parameter_binding_index)
    if parameter_bound is not None:
        return parameter_bound
    relation = None
    if spec.relation_field_name:
        relation = relation_support(
            paper_key=normalize_text(row.get("key")),
            candidate_id=normalize_text(row.get("representative_source_formulation_id")),
            field_name=spec.relation_field_name,
            relation_index=relation_index,
            relation_rows_by_id=relation_rows_by_id,
        )
    if relation is not None:
        return relation
    if direct is not None:
        return direct
    return blank_support()


def derive_drug_polymer_ratio(field_state: dict[str, dict[str, str]]) -> dict[str, str]:
    drug = field_state["drug_feed_amount"]
    polymer = field_state["polymer_amount"]
    if drug["support_status"] not in {"explicit_supported", "relation_carried_explicit"}:
        return blank_support()
    if polymer["support_status"] not in {"explicit_supported", "relation_carried_explicit"}:
        return blank_support()
    drug_mg = parse_mass_to_mg(drug["raw_text"] or drug["value"])
    polymer_mg = parse_mass_to_mg(polymer["raw_text"] or polymer["value"])
    if drug_mg is None or polymer_mg is None or polymer_mg == 0:
        return {
            "value": "",
            "raw_text": f"{drug['raw_text']} / {polymer['raw_text']}".strip(" /"),
            "support_status": "parse_failed",
            "evidence_source_type": "derived_from_explicit_values",
            "evidence_text": "drug_polymer_ratio requested, but explicit inputs could not be parsed safely.",
            "evidence_locator": "drug_feed_amount|polymer_amount",
        }
    return {
        "value": format_value(drug_mg / polymer_mg),
        "raw_text": f"{format_value(drug_mg)} mg / {format_value(polymer_mg)} mg",
        "support_status": "explicit_supported",
        "evidence_source_type": "derived_from_explicit_values",
        "evidence_text": f"Derived from explicit drug_feed_amount={drug['raw_text'] or drug['value']} and polymer_amount={polymer['raw_text'] or polymer['value']}.",
        "evidence_locator": "drug_feed_amount|polymer_amount",
    }


def summarize_statuses(evidence_rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for row in evidence_rows:
        summary[row["field_name"]][row["support_status"]] += 1
    return summary


def build_summary_markdown(
    *,
    final_rows: list[dict[str, str]],
    output_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    per_field_summary: dict[str, Counter[str]],
) -> str:
    lines = [
        "# Deterministic Step 2 Value Backfill Summary v1",
        "",
        "## Contract",
        "- downstream of a frozen Step 1 final table",
        "- one output row per existing final_formulation_id",
        "- explicit-only value attachment",
        "- no split, merge, add, or delete",
        "",
        "## Counts",
        f"- source final rows: `{len(final_rows)}`",
        f"- output rows: `{len(output_rows)}`",
        f"- evidence decisions: `{len(evidence_rows)}`",
        "",
        "## Field Status Counts",
    ]
    for field_name in [spec.output_field for spec in FIELD_SPECS] + ["drug_polymer_ratio"]:
        counter = per_field_summary.get(field_name, Counter())
        if counter:
            parts = [f"{status}={count}" for status, count in sorted(counter.items())]
            lines.append(f"- `{field_name}`: `{', '.join(parts)}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-table-tsv", required=True, type=Path)
    parser.add_argument("--decision-trace-tsv", default="", type=Path)
    parser.add_argument("--relation-records-tsv", default="", type=Path)
    parser.add_argument("--resolved-relation-fields-tsv", default="", type=Path)
    parser.add_argument("--audit-ready-tsv", default="", type=Path)
    parser.add_argument("--field-gt-review-seed-rows-tsv", default="", type=Path)
    parser.add_argument("--baseline-assessment-tsv", default="", type=Path)
    parser.add_argument("--scope-manifest-tsv", default="", type=Path)
    parser.add_argument("--source-run-dir", default="", type=Path)
    parser.add_argument("--table-row-binding-tsv", default="", type=Path)
    parser.add_argument("--parameter-binding-tsv", default="", type=Path)
    parser.add_argument("--paper-key", action="append", default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_rows_all = read_tsv_rows(args.final_table_tsv.resolve())
    requested_papers = {normalize_text(value) for value in args.paper_key if normalize_text(value)}
    final_rows = [row for row in final_rows_all if not requested_papers or normalize_text(row.get("key")) in requested_papers]

    relation_rows = optional_rows(optional_path_text(args.relation_records_tsv))
    resolved_relation_rows = optional_rows(optional_path_text(args.resolved_relation_fields_tsv))
    baseline_rows = optional_rows(optional_path_text(args.baseline_assessment_tsv))
    binding_rows_data = binding_rows(optional_path_text(args.table_row_binding_tsv))
    parameter_binding_rows_data = binding_rows(optional_path_text(args.parameter_binding_tsv))
    baseline_by_paper = {normalize_text(row.get("paper_key")): row for row in baseline_rows if normalize_text(row.get("paper_key"))}
    binding_index = {
        (normalize_text(row.get("final_formulation_id")), normalize_text(row.get("field_name"))): row
        for row in binding_rows_data
        if normalize_text(row.get("final_formulation_id")) and normalize_text(row.get("field_name"))
    }
    parameter_binding_index = {
        (normalize_text(row.get("final_formulation_id")), normalize_text(row.get("field_name"))): row
        for row in parameter_binding_rows_data
        if normalize_text(row.get("final_formulation_id")) and normalize_text(row.get("field_name"))
    }

    relation_index, relation_rows_by_id = build_relation_indexes(resolved_relation_rows, relation_rows)
    output_rows: list[dict[str, str]] = []
    evidence_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for row in final_rows:
        final_id = normalize_text(row.get("final_formulation_id"))
        if final_id in seen_ids:
            duplicate_ids.append(final_id)
        seen_ids.add(final_id)

        paper_key = normalize_text(row.get("key"))
        baseline_row = baseline_by_paper.get(paper_key, {})
        per_field_state: dict[str, dict[str, str]] = {}
        for spec in FIELD_SPECS:
            per_field_state[spec.output_field] = choose_field_support(
                row=row,
                spec=spec,
                relation_index=relation_index,
                relation_rows_by_id=relation_rows_by_id,
                binding_index=binding_index,
                parameter_binding_index=parameter_binding_index,
            )
        per_field_state["drug_polymer_ratio"] = derive_drug_polymer_ratio(per_field_state)

        out_row: dict[str, str] = {
            "final_formulation_id": final_id,
            "paper_key": paper_key,
            "source_run_dir": normalize_text(args.source_run_dir),
            "source_final_table_path": normalize_text(args.final_table_tsv.resolve()),
            "baseline_identity_classification": normalize_text(baseline_row.get("classification")),
            "baseline_identity_rationale": normalize_text(baseline_row.get("rationale")),
        }
        for field_name, field_state in per_field_state.items():
            out_row[f"{field_name}_value"] = field_state["value"]
            out_row[f"{field_name}_raw_text"] = field_state["raw_text"]
            out_row[f"{field_name}_support_status"] = field_state["support_status"]
            out_row[f"{field_name}_evidence_source_type"] = field_state["evidence_source_type"]
            out_row[f"{field_name}_evidence_text"] = field_state["evidence_text"]
            out_row[f"{field_name}_evidence_locator"] = field_state["evidence_locator"]
            evidence_rows.append({
                "final_formulation_id": final_id,
                "paper_key": paper_key,
                "field_name": field_name,
                "value": field_state["value"],
                "raw_text": field_state["raw_text"],
                "support_status": field_state["support_status"],
                "evidence_source_type": field_state["evidence_source_type"],
                "evidence_text": field_state["evidence_text"],
                "evidence_locator": field_state["evidence_locator"],
                "source_representative_formulation_id": normalize_text(row.get("representative_source_formulation_id")),
            })
        output_rows.append(out_row)

    if duplicate_ids:
        raise ValueError(f"Duplicate final_formulation_id detected in source final table: {sorted(set(duplicate_ids))}")

    fieldnames = [
        "final_formulation_id",
        "paper_key",
        "source_run_dir",
        "source_final_table_path",
        "baseline_identity_classification",
        "baseline_identity_rationale",
    ]
    for field_name in [spec.output_field for spec in FIELD_SPECS] + ["drug_polymer_ratio"]:
        fieldnames.extend([
            f"{field_name}_value",
            f"{field_name}_raw_text",
            f"{field_name}_support_status",
            f"{field_name}_evidence_source_type",
            f"{field_name}_evidence_text",
            f"{field_name}_evidence_locator",
        ])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / OUTPUT_TABLE_NAME, fieldnames, output_rows)
    write_tsv(
        args.out_dir / OUTPUT_EVIDENCE_NAME,
        [
            "final_formulation_id",
            "paper_key",
            "field_name",
            "value",
            "raw_text",
            "support_status",
            "evidence_source_type",
            "evidence_text",
            "evidence_locator",
            "source_representative_formulation_id",
        ],
        evidence_rows,
    )
    per_field_summary = summarize_statuses(evidence_rows)
    (args.out_dir / OUTPUT_SUMMARY_NAME).write_text(
        build_summary_markdown(final_rows=final_rows, output_rows=output_rows, evidence_rows=evidence_rows, per_field_summary=per_field_summary),
        encoding="utf-8",
    )

    print(json.dumps({
        "source_final_rows": len(final_rows),
        "output_rows": len(output_rows),
        "duplicate_ids": len(duplicate_ids),
        "out_dir": str(args.out_dir.resolve()),
        "explicit_supported_by_field": {
            field_name: counter.get("explicit_supported", 0) + counter.get("relation_carried_explicit", 0)
            for field_name, counter in per_field_summary.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
