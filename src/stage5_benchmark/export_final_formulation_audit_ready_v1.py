#!/usr/bin/env python3
"""
Build an audit-ready TSV from an existing Stage 5 final formulation table.

Purpose:
- add an explicit human-review layer on top of existing final formulation rows
- reuse current provenance and evidence fields without changing extraction logic

Inputs:
- Stage 5 `final_formulation_table_v1.tsv`
- optional Stage 2 weak-label TSV for representative-source enrichment
- optional Stage 5 `final_output_decision_trace_v1.tsv` for closure-rule context

Outputs:
- `final_formulation_table_audit_ready_v1.tsv`

Stage role:
- postprocessing export after Stage 5 final-output generation
- not part of the scientific extraction path

What this script does not do:
- does not rerun Stage 2, Stage 3, or Stage 5
- does not modify benchmark counts
- does not assign scientific truth labels
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


OUTPUT_NAME = "final_formulation_table_audit_ready_v1.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-table-tsv",
        required=True,
        help="Path to Stage 5 final_formulation_table_v1.tsv.",
    )
    parser.add_argument(
        "--out-tsv",
        default="",
        help="Explicit output TSV path. Defaults next to the final table.",
    )
    parser.add_argument(
        "--source-weak-labels-tsv",
        default="",
        help="Optional Stage 2 weak-label TSV for representative-source enrichment.",
    )
    parser.add_argument(
        "--decision-trace-tsv",
        default="",
        help="Optional Stage 5 final_output_decision_trace_v1.tsv for decision-rule context.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def parse_json_list(value: object) -> list[str]:
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


def parse_json_object_list(value: object) -> list[dict[str, str]]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append({str(k): normalize_text(v) for k, v in item.items()})
    return out


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def weak_label_key(row: dict[str, str]) -> str:
    return f"{normalize_text(row.get('key'))}::{normalize_text(row.get('formulation_id'))}"


def decision_trace_key(row: dict[str, str]) -> str:
    return f"{normalize_text(row.get('zotero_key'))}::{normalize_text(row.get('source_formulation_id'))}"


def extract_table_id(evidence_refs: list[dict[str, str]], fallback_section: str) -> str:
    for ref in evidence_refs:
        region_type = normalize_text(ref.get("region_type")).lower()
        section = normalize_text(ref.get("section"))
        if region_type in {"table_row", "table_cell", "table_header"} and section:
            return section
        if section.lower().startswith("table "):
            return section
        if "__table_" in section.lower():
            return section
    if (
        fallback_section
        and (
            fallback_section.lower().startswith("table ")
            or "__table_" in fallback_section.lower()
        )
    ):
        return fallback_section
    return ""


def extract_row_anchor(
    representative_label: str,
    representative_id: str,
    evidence_refs: list[dict[str, str]],
) -> str:
    label = representative_label.strip()
    if re.match(r"^\d+\.?$", label):
        return label
    match = re.search(r"DOE_Row_(\d+)", representative_id)
    if match:
        return match.group(1)
    for ref in evidence_refs:
        span = normalize_text(ref.get("span_text"))
        match = re.match(r"^(\d+\.?)\s+\|", span)
        if match:
            return match.group(1)
    return label


def select_primary_evidence(
    final_row: dict[str, str],
    source_row: dict[str, str] | None,
) -> tuple[list[dict[str, str]], str, str]:
    final_refs = parse_json_object_list(final_row.get("supporting_evidence_refs", ""))
    if final_refs:
        primary_region = normalize_text(final_row.get("instance_evidence_region_type"))
        primary_section = normalize_text(final_refs[0].get("section"))
        return final_refs, primary_region, primary_section

    if source_row is not None:
        source_refs = parse_json_object_list(source_row.get("supporting_evidence_refs", ""))
        if source_refs:
            primary_region = normalize_text(source_row.get("instance_evidence_region_type"))
            primary_section = normalize_text(source_refs[0].get("section"))
            return source_refs, primary_region, primary_section

    return [], normalize_text(final_row.get("instance_evidence_region_type")), ""


def build_fields_summary(row: dict[str, str]) -> str:
    fields: list[str] = []
    pairs = [
        ("polymer", row.get("polymer_identity_final") or row.get("polymer_identity")),
        ("drug", row.get("drug_name_value")),
        ("feed", row.get("drug_feed_amount_text_value")),
        ("surfactant", row.get("surfactant_name_value")),
        ("surfactant_conc", row.get("surfactant_concentration_text_value")),
        ("polymer_mass", row.get("plga_mass_mg_value")),
        ("solvent", row.get("organic_solvent_value")),
        ("size_nm", row.get("size_nm_value")),
        ("pdi", row.get("pdi_value")),
    ]
    for name, value in pairs:
        text = normalize_text(value)
        if text:
            fields.append(f"{name}={text}")
    return "; ".join(fields)


def compute_confidence_and_priority(
    row: dict[str, str],
    source_candidates: list[str],
    table_id: str,
    row_anchor: str,
    evidence_refs: list[dict[str, str]],
    evidence_source_type: str,
) -> tuple[str, str, str]:
    candidate_source = normalize_text(row.get("candidate_source")).lower()
    instance_confidence = normalize_text(row.get("instance_confidence")).lower()
    source_candidate_count = int(normalize_text(row.get("source_candidate_count")) or "0")

    reasons: list[str] = []
    lower_sources = {item.lower() for item in source_candidates if item}

    if (
        candidate_source == "doe_numbered_table_row"
        and evidence_source_type == "table_row"
        and table_id
        and row_anchor
    ):
        return "tier1_structured_row", "low", "explicit_numbered_table_row"

    if evidence_source_type in {"table_row", "table_cell"} or "doe_numbered_table_row" in lower_sources:
        if not table_id:
            reasons.append("table_like_evidence_without_table_id")
        return "tier2_table_derived", "medium", "; ".join(reasons) or "table_derived_row"

    if candidate_source == "llm_extracted":
        if evidence_source_type in {"", "unknown"}:
            reasons.append("evidence_region_unknown")
        first_section = normalize_text(evidence_refs[0].get("section")) if evidence_refs else ""
        if first_section.lower() == "full_text_window":
            reasons.append("full_text_window_only")
        if source_candidate_count > 1:
            reasons.append("collapsed_multi_source_row")
        if re.search(r"\boptimized\b|\bblank\b|\bcontrol\b|^f\s*\d+\b", normalize_text(row.get("representative_source_raw_formulation_label")).lower()):
            reasons.append("special_case_label")
        priority = "high" if reasons else "medium"
        tier = "tier3_llm_with_evidence" if instance_confidence in {"high", "medium"} and evidence_refs else "tier4_review_required"
        return tier, priority, "; ".join(reasons) or "llm_extracted_row"

    if candidate_source in {"figure_variable_sweep", "synthetic_figure_sweep"}:
        return "tier4_review_required", "high", "figure_or_sweep_derived_row"

    if source_candidate_count > 1:
        return "tier3_llm_with_evidence", "medium", "collapsed_multi_source_row"

    return "tier4_review_required", "high", "weak_or_unclassified_provenance"


def build_evidence_locator(evidence_refs: list[dict[str, str]], evidence_source_type: str, table_id: str) -> str:
    if evidence_refs:
        first_ref = evidence_refs[0]
        section = normalize_text(first_ref.get("section"))
        span = normalize_text(first_ref.get("span_text"))
        span = re.sub(r"\s+", " ", span)
        if len(span) > 160:
            span = span[:157] + "..."
        return f"{evidence_source_type or first_ref.get('region_type')}::{section}::{span}"
    if table_id:
        return f"{evidence_source_type}::{table_id}"
    return evidence_source_type


def build_provenance_pointer(
    row: dict[str, str],
    decision_row: dict[str, str] | None,
    table_id: str,
) -> str:
    parts = [
        f"representative_source={normalize_text(row.get('representative_source_formulation_id'))}",
        f"final_rule={normalize_text(row.get('final_output_rule'))}",
    ]
    if table_id:
        parts.append(f"table_id={table_id}")
    if decision_row is not None:
        decision_rule = normalize_text(decision_row.get("decision_rule"))
        if decision_rule:
            parts.append(f"decision_rule={decision_rule}")
    return "; ".join(parts)


def build_export_rows(
    final_rows: list[dict[str, str]],
    weak_label_rows: dict[str, dict[str, str]],
    decision_rows: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    export_rows: list[dict[str, str]] = []
    for row in final_rows:
        source_key = f"{normalize_text(row.get('key'))}::{normalize_text(row.get('representative_source_formulation_id'))}"
        source_row = weak_label_rows.get(source_key)
        decision_row = decision_rows.get(source_key)
        evidence_refs, evidence_source_type, fallback_section = select_primary_evidence(row, source_row)
        table_id = extract_table_id(evidence_refs, fallback_section)
        row_anchor = extract_row_anchor(
            normalize_text(row.get("representative_source_raw_formulation_label")),
            normalize_text(row.get("representative_source_formulation_id")),
            evidence_refs,
        )
        source_candidates = parse_json_list(row.get("source_candidate_sources", ""))
        confidence_tier, review_priority, needs_review_reason = compute_confidence_and_priority(
            row=row,
            source_candidates=source_candidates,
            table_id=table_id,
            row_anchor=row_anchor,
            evidence_refs=evidence_refs,
            evidence_source_type=evidence_source_type.lower(),
        )

        export_rows.append(
            {
                "paper_id": normalize_text(row.get("key")),
                "doi": normalize_text(row.get("doi")),
                "formulation_id": normalize_text(row.get("final_formulation_id")),
                "representative_source_formulation_id": normalize_text(row.get("representative_source_formulation_id")),
                "table_id": table_id,
                "row_anchor": row_anchor,
                "candidate_source": normalize_text(row.get("candidate_source")),
                "evidence_source_type": evidence_source_type,
                "confidence_tier": confidence_tier,
                "review_priority": review_priority,
                "needs_review_reason": needs_review_reason,
                "key_formulation_fields_summary": build_fields_summary(row),
                "encapsulation_efficiency_percent": normalize_text(
                    row.get("encapsulation_efficiency_percent_value_text")
                    or row.get("encapsulation_efficiency_percent_value")
                ),
                "evidence_locator": build_evidence_locator(evidence_refs, evidence_source_type, table_id),
                "provenance_pointer": build_provenance_pointer(row, decision_row, table_id),
                "source_candidate_count": normalize_text(row.get("source_candidate_count")),
                "final_output_rule": normalize_text(row.get("final_output_rule")),
                "auditor_decision": "",
                "auditor_note": "",
            }
        )
    return export_rows


def main() -> None:
    args = parse_args()
    final_table_path = Path(args.final_table_tsv).resolve()
    out_path = (
        Path(args.out_tsv).resolve()
        if args.out_tsv
        else final_table_path.with_name(OUTPUT_NAME)
    )

    final_rows = read_tsv(final_table_path)

    weak_label_rows: dict[str, dict[str, str]] = {}
    if args.source_weak_labels_tsv:
        for row in read_tsv(Path(args.source_weak_labels_tsv).resolve()):
            weak_label_rows[weak_label_key(row)] = row

    decision_rows: dict[str, dict[str, str]] = {}
    if args.decision_trace_tsv:
        for row in read_tsv(Path(args.decision_trace_tsv).resolve()):
            decision_rows[decision_trace_key(row)] = row

    export_rows = build_export_rows(final_rows, weak_label_rows, decision_rows)

    fieldnames = [
        "paper_id",
        "doi",
        "formulation_id",
        "representative_source_formulation_id",
        "table_id",
        "row_anchor",
        "candidate_source",
        "evidence_source_type",
        "confidence_tier",
        "review_priority",
        "needs_review_reason",
        "key_formulation_fields_summary",
        "encapsulation_efficiency_percent",
        "evidence_locator",
        "provenance_pointer",
        "source_candidate_count",
        "final_output_rule",
        "auditor_decision",
        "auditor_note",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export_rows)

    print(out_path)


if __name__ == "__main__":
    main()
