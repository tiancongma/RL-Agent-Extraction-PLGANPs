#!/usr/bin/env python3
from __future__ import annotations

"""
Build a deterministic formulation-parameter binding surface for frozen Step 1 rows.

Purpose:
- resolve lawful formulation-level ownership for composition-side fields
- reuse deterministic Stage3 relation outputs and frozen Step 1 identity surfaces
- prepare additive ownership evidence for the existing Step 2 helper

This helper does not fill values into the final dataset by itself.
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


CANDIDATES_NAME = "formulation_parameter_binding_candidates_v1.tsv"
RESOLVED_NAME = "formulation_parameter_binding_resolved_v1.tsv"
SUMMARY_NAME = "formulation_parameter_binding_summary_v1.md"


@dataclass(frozen=True)
class FieldSpec:
    field_name: str
    relation_field_names: tuple[str, ...]
    priority_tier: str


FIELD_SPECS = [
    FieldSpec("polymer_mw_kDa", ("polymer_mw_kDa",), "tier1"),
    FieldSpec("la_ga_ratio", ("la_ga_ratio",), "tier1"),
    FieldSpec("polymer_amount", ("plga_mass_mg",), "tier1"),
    FieldSpec("drug_feed_amount", ("drug_feed_amount_text",), "tier1"),
    FieldSpec("surfactant_concentration", ("surfactant_concentration_text", "pva_conc_percent"), "tier2"),
    FieldSpec("phase_ratio", tuple(), "tier2"),
    FieldSpec("drug_polymer_ratio", tuple(), "tier2"),
    FieldSpec("surfactant_name", ("surfactant_name",), "tier3"),
    FieldSpec("organic_solvent", ("organic_solvent",), "tier3"),
]


SUPPORTED_STATUSES = {
    "resolved_shared_context",
    "resolved_relation_context",
    "resolved_article_native_match",
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_label(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


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


def relation_evidence_row(
    resolved_row: dict[str, str],
    relation_rows_by_id: dict[str, dict[str, str]],
) -> dict[str, str]:
    relation_row_ids = parse_json_string_list(resolved_row.get("source_relation_row_ids"))
    for row_id in relation_row_ids:
        relation_row = relation_rows_by_id.get(row_id)
        if relation_row is not None:
            return relation_row
    return {}


def final_row_candidate_ids(row: dict[str, str]) -> list[str]:
    values = parse_json_string_list(row.get("source_candidate_ids"))
    representative = normalize_text(row.get("representative_source_formulation_id"))
    if representative and representative not in values:
        values.append(representative)
    return values


def final_row_candidate_labels(row: dict[str, str]) -> list[str]:
    values = parse_json_string_list(row.get("source_candidate_labels"))
    for key in ["raw_formulation_label", "representative_source_raw_formulation_label"]:
        value = normalize_text(row.get(key))
        if value and value not in values:
            values.append(value)
    return values


def build_relation_indexes(
    resolved_relation_rows: list[dict[str, str]],
    relation_rows: list[dict[str, str]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, str]]],
    dict[str, dict[str, str]],
    dict[tuple[str, str], list[dict[str, str]]],
]:
    by_paper_field: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_method_group_field: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in resolved_relation_rows:
        paper_key = normalize_text(row.get("paper_key"))
        field_name = normalize_text(row.get("field_name"))
        method_group_id = normalize_text(row.get("method_group_id"))
        if paper_key and field_name:
            by_paper_field[(paper_key, field_name)].append(row)
        if method_group_id and field_name:
            by_method_group_field[(method_group_id, field_name)].append(row)
    relation_rows_by_id = {
        normalize_text(row.get("relation_row_id")): row
        for row in relation_rows
        if normalize_text(row.get("relation_row_id"))
    }
    return by_paper_field, relation_rows_by_id, by_method_group_field


def build_output_row(
    *,
    row: dict[str, str],
    field_name: str,
    source_type: str,
    source_section: str,
    source_locator: str,
    source_value_text: str,
    source_value_normalized: str,
    binding_rule_used: str,
    binding_confidence_class: str,
    binding_status: str,
) -> dict[str, str]:
    return {
        "final_formulation_id": normalize_text(row.get("final_formulation_id")),
        "paper_key": normalize_text(row.get("key")),
        "field_name": field_name,
        "source_type": source_type,
        "source_section": source_section,
        "source_locator": source_locator,
        "source_value_text": source_value_text,
        "source_value_normalized": source_value_normalized,
        "binding_rule_used": binding_rule_used,
        "binding_confidence_class": binding_confidence_class,
        "binding_status": binding_status,
    }


def unique_value_rows(rows: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = normalize_text(row.get("field_value_norm")) or normalize_text(row.get("field_value"))
        if key:
            grouped[key].append(row)
    if len(grouped) == 1:
        key = next(iter(grouped))
        return key, grouped[key]
    return "", []


def resolve_field_binding(
    *,
    final_row: dict[str, str],
    spec: FieldSpec,
    resolved_rows_by_paper_field: dict[tuple[str, str], list[dict[str, str]]],
    relation_rows_by_id: dict[str, dict[str, str]],
    resolved_rows_by_method_group_field: dict[tuple[str, str], list[dict[str, str]]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    candidate_rows: list[dict[str, str]] = []
    resolved_rows: list[dict[str, str]] = []
    paper_key = normalize_text(final_row.get("key"))
    candidate_ids = set(final_row_candidate_ids(final_row))
    candidate_labels = {normalize_label(value) for value in final_row_candidate_labels(final_row) if normalize_label(value)}
    method_group_ids = set(parse_json_string_list(final_row.get("relation_method_group_ids")))

    if not spec.relation_field_names:
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="unsupported_context",
                source_section="",
                source_locator="",
                source_value_text="",
                source_value_normalized="",
                binding_rule_used="no_relation_surface_available",
                binding_confidence_class="low",
                binding_status="unsupported_context",
            )
        )
        return candidate_rows, resolved_rows

    exact_pool: list[dict[str, str]] = []
    label_pool: list[dict[str, str]] = []
    method_pool: list[dict[str, str]] = []
    paper_pool: list[dict[str, str]] = []

    for relation_field_name in spec.relation_field_names:
        rows = resolved_rows_by_paper_field.get((paper_key, relation_field_name), [])
        paper_pool.extend(rows)
        for resolved_row in rows:
            formulation_candidate_id = normalize_text(resolved_row.get("formulation_candidate_id"))
            evidence_row = relation_evidence_row(resolved_row, relation_rows_by_id)
            candidate_label = normalize_label(evidence_row.get("candidate_label"))
            if formulation_candidate_id and formulation_candidate_id in candidate_ids:
                exact_pool.append(resolved_row)
                candidate_rows.append(
                    build_output_row(
                        row=final_row,
                        field_name=spec.field_name,
                        source_type="resolved_relation_field",
                        source_section=normalize_text(evidence_row.get("evidence_section")),
                        source_locator=normalize_text(resolved_row.get("source_relation_row_ids")),
                        source_value_text=normalize_text(resolved_row.get("field_value")),
                        source_value_normalized=normalize_text(resolved_row.get("field_value_norm")),
                        binding_rule_used="exact_source_candidate_relation_match",
                        binding_confidence_class=normalize_text(resolved_row.get("deterministic_confidence")) or "medium",
                        binding_status="candidate_exact_match",
                    )
                )
            elif candidate_label and candidate_label in candidate_labels:
                label_pool.append(resolved_row)
                candidate_rows.append(
                    build_output_row(
                        row=final_row,
                        field_name=spec.field_name,
                        source_type="resolved_relation_field",
                        source_section=normalize_text(evidence_row.get("evidence_section")),
                        source_locator=normalize_text(resolved_row.get("source_relation_row_ids")),
                        source_value_text=normalize_text(resolved_row.get("field_value")),
                        source_value_normalized=normalize_text(resolved_row.get("field_value_norm")),
                        binding_rule_used="article_native_label_relation_match",
                        binding_confidence_class=normalize_text(resolved_row.get("deterministic_confidence")) or "medium",
                        binding_status="candidate_article_native_match",
                    )
                )
        for method_group_id in method_group_ids:
            method_rows = resolved_rows_by_method_group_field.get((method_group_id, relation_field_name), [])
            method_pool.extend(method_rows)

    exact_value_key, exact_value_rows = unique_value_rows(exact_pool)
    if exact_value_rows:
        chosen = exact_value_rows[0]
        evidence_row = relation_evidence_row(chosen, relation_rows_by_id)
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="resolved_relation_field",
                source_section=normalize_text(evidence_row.get("evidence_section")),
                source_locator=normalize_text(chosen.get("source_relation_row_ids")),
                source_value_text=normalize_text(chosen.get("field_value")),
                source_value_normalized=exact_value_key,
                binding_rule_used="exact_source_candidate_relation_match",
                binding_confidence_class=normalize_text(chosen.get("deterministic_confidence")) or "medium",
                binding_status="resolved_relation_context",
            )
        )
        return candidate_rows, resolved_rows
    if exact_pool:
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="resolved_relation_field",
                source_section="",
                source_locator="",
                source_value_text="",
                source_value_normalized="",
                binding_rule_used="exact_source_candidate_relation_match",
                binding_confidence_class="low",
                binding_status="ambiguous_multiple_targets",
            )
        )
        return candidate_rows, resolved_rows

    label_value_key, label_value_rows = unique_value_rows(label_pool)
    if label_value_rows:
        chosen = label_value_rows[0]
        evidence_row = relation_evidence_row(chosen, relation_rows_by_id)
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="resolved_relation_field",
                source_section=normalize_text(evidence_row.get("evidence_section")),
                source_locator=normalize_text(chosen.get("source_relation_row_ids")),
                source_value_text=normalize_text(chosen.get("field_value")),
                source_value_normalized=label_value_key,
                binding_rule_used="article_native_label_relation_match",
                binding_confidence_class=normalize_text(chosen.get("deterministic_confidence")) or "medium",
                binding_status="resolved_article_native_match",
            )
        )
        return candidate_rows, resolved_rows
    if label_pool:
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="resolved_relation_field",
                source_section="",
                source_locator="",
                source_value_text="",
                source_value_normalized="",
                binding_rule_used="article_native_label_relation_match",
                binding_confidence_class="low",
                binding_status="ambiguous_multiple_targets",
            )
        )
        return candidate_rows, resolved_rows

    method_value_key, method_value_rows = unique_value_rows(method_pool)
    if method_value_rows:
        chosen = method_value_rows[0]
        evidence_row = relation_evidence_row(chosen, relation_rows_by_id)
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="resolved_relation_field",
                source_section=normalize_text(evidence_row.get("evidence_section")),
                source_locator=normalize_text(chosen.get("method_group_id")),
                source_value_text=normalize_text(chosen.get("field_value")),
                source_value_normalized=method_value_key,
                binding_rule_used="method_group_uniform_value",
                binding_confidence_class=normalize_text(chosen.get("deterministic_confidence")) or "medium",
                binding_status="resolved_shared_context",
            )
        )
        return candidate_rows, resolved_rows
    if method_pool:
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="resolved_relation_field",
                source_section="",
                source_locator="",
                source_value_text="",
                source_value_normalized="",
                binding_rule_used="method_group_uniform_value",
                binding_confidence_class="low",
                binding_status="ambiguous_multiple_targets",
            )
        )
        return candidate_rows, resolved_rows

    paper_value_key, paper_value_rows = unique_value_rows(paper_pool)
    if paper_value_rows:
        chosen = paper_value_rows[0]
        evidence_row = relation_evidence_row(chosen, relation_rows_by_id)
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="resolved_relation_field",
                source_section=normalize_text(evidence_row.get("evidence_section")),
                source_locator=paper_key,
                source_value_text=normalize_text(chosen.get("field_value")),
                source_value_normalized=paper_value_key,
                binding_rule_used="paper_global_uniform_value",
                binding_confidence_class=normalize_text(chosen.get("deterministic_confidence")) or "medium",
                binding_status="resolved_shared_context",
            )
        )
        return candidate_rows, resolved_rows
    if paper_pool:
        resolved_rows.append(
            build_output_row(
                row=final_row,
                field_name=spec.field_name,
                source_type="resolved_relation_field",
                source_section="",
                source_locator=paper_key,
                source_value_text="",
                source_value_normalized="",
                binding_rule_used="paper_global_uniform_value",
                binding_confidence_class="low",
                binding_status="ambiguous_multiple_targets",
            )
        )
        return candidate_rows, resolved_rows

    resolved_rows.append(
        build_output_row(
            row=final_row,
            field_name=spec.field_name,
            source_type="resolved_relation_field",
            source_section="",
            source_locator="",
            source_value_text="",
            source_value_normalized="",
            binding_rule_used="no_matching_relation_value",
            binding_confidence_class="low",
            binding_status="no_matching_target",
        )
    )
    return candidate_rows, resolved_rows


def build_summary_markdown(
    *,
    final_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    resolved_rows: list[dict[str, str]],
) -> str:
    status_counter = Counter(row["binding_status"] for row in resolved_rows)
    field_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for row in resolved_rows:
        field_counter[row["field_name"]][row["binding_status"]] += 1

    lines = [
        "# Formulation Parameter Binding Summary v1",
        "",
        "## Contract",
        "- frozen Step 1 formulation membership preserved",
        "- deterministic formulation-level ownership only",
        "- no direct value fill here; this surface is additive evidence for Step 2",
        "",
        "## Counts",
        f"- source final rows: `{len(final_rows)}`",
        f"- candidate rows: `{len(candidate_rows)}`",
        f"- resolved rows: `{len(resolved_rows)}`",
        "",
        "## Status Totals",
        f"- resolved_relation_context: `{status_counter.get('resolved_relation_context', 0)}`",
        f"- resolved_shared_context: `{status_counter.get('resolved_shared_context', 0)}`",
        f"- resolved_article_native_match: `{status_counter.get('resolved_article_native_match', 0)}`",
        f"- ambiguous_multiple_targets: `{status_counter.get('ambiguous_multiple_targets', 0)}`",
        f"- no_matching_target: `{status_counter.get('no_matching_target', 0)}`",
        f"- unsupported_context: `{status_counter.get('unsupported_context', 0)}`",
        "",
        "## Field Detail",
    ]
    for spec in FIELD_SPECS:
        counter = field_counter.get(spec.field_name, Counter())
        parts = [f"{status}={count}" for status, count in sorted(counter.items())]
        lines.append(f"- `{spec.field_name}` ({spec.priority_tier}): `{', '.join(parts) if parts else 'none'}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic formulation-parameter binding unit for frozen Step 1 rows.")
    parser.add_argument("--final-table-tsv", required=True, type=Path)
    parser.add_argument("--decision-trace-tsv", default="", type=Path)
    parser.add_argument("--relation-records-tsv", required=True, type=Path)
    parser.add_argument("--resolved-relation-fields-tsv", required=True, type=Path)
    parser.add_argument("--scope-manifest-tsv", default="", type=Path)
    parser.add_argument("--paper-key", action="append", default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_rows_all = read_tsv_rows(args.final_table_tsv.resolve())
    requested_papers = {normalize_text(value) for value in args.paper_key if normalize_text(value)}
    final_rows = [row for row in final_rows_all if not requested_papers or normalize_text(row.get("key")) in requested_papers]
    relation_rows = read_tsv_rows(args.relation_records_tsv.resolve())
    resolved_relation_rows = read_tsv_rows(args.resolved_relation_fields_tsv.resolve())
    resolved_rows_by_paper_field, relation_rows_by_id, resolved_rows_by_method_group_field = build_relation_indexes(
        resolved_relation_rows,
        relation_rows,
    )

    candidate_rows: list[dict[str, str]] = []
    resolved_rows: list[dict[str, str]] = []
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    for final_row in final_rows:
        final_id = normalize_text(final_row.get("final_formulation_id"))
        if final_id in seen_ids:
            duplicate_ids.append(final_id)
        seen_ids.add(final_id)
        for spec in FIELD_SPECS:
            field_candidates, field_resolved = resolve_field_binding(
                final_row=final_row,
                spec=spec,
                resolved_rows_by_paper_field=resolved_rows_by_paper_field,
                relation_rows_by_id=relation_rows_by_id,
                resolved_rows_by_method_group_field=resolved_rows_by_method_group_field,
            )
            candidate_rows.extend(field_candidates)
            resolved_rows.extend(field_resolved)

    if duplicate_ids:
        raise ValueError(f"Duplicate final_formulation_id detected in source final table: {sorted(set(duplicate_ids))}")

    fieldnames = [
        "final_formulation_id",
        "paper_key",
        "field_name",
        "source_type",
        "source_section",
        "source_locator",
        "source_value_text",
        "source_value_normalized",
        "binding_rule_used",
        "binding_confidence_class",
        "binding_status",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / CANDIDATES_NAME, fieldnames, candidate_rows)
    write_tsv(args.out_dir / RESOLVED_NAME, fieldnames, resolved_rows)
    (args.out_dir / SUMMARY_NAME).write_text(
        build_summary_markdown(final_rows=final_rows, candidate_rows=candidate_rows, resolved_rows=resolved_rows),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "source_final_rows": len(final_rows),
                "candidate_rows": len(candidate_rows),
                "resolved_rows": len(resolved_rows),
                "resolved_supported_rows": sum(1 for row in resolved_rows if row["binding_status"] in SUPPORTED_STATUSES),
                "status_counts": dict(Counter(row["binding_status"] for row in resolved_rows)),
                "out_dir": str(args.out_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
