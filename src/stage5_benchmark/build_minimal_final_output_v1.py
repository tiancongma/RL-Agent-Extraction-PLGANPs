#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DECISION_TRACE_NAME = "final_output_decision_trace_v1.tsv"
FINAL_TABLE_NAME = "final_formulation_table_v1.tsv"
SUMMARY_NAME = "final_output_summary_v1.md"
RELATION_RECORDS_NAME = "formulation_relation_records_v1.tsv"


@dataclass(frozen=True)
class RowDecision:
    decision: str
    target_final_formulation_id: str
    decision_rule: str
    decision_reason: str
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
    }


def build_key_fields_used(core_fields: dict[str, str]) -> str:
    return json.dumps(core_fields, ensure_ascii=True, sort_keys=True)


def has_context_tag(row: dict[str, str], target_tags: set[str]) -> bool:
    observed = {
        normalize_text(tag)
        for tag in parse_json_list(row.get("instance_context_tags", "[]"))
        + parse_json_list(row.get("change_context_tags", "[]"))
    }
    return not observed.isdisjoint(target_tags)


def should_filter_non_formulation(
    row: dict[str, str], core_fields: dict[str, str]
) -> tuple[bool, str, str]:
    if normalize_text(row.get("instance_kind")) == "candidate_non_formulation":
        return (
            True,
            "explicit_candidate_non_formulation",
            "Stage2 explicitly marked this row as candidate_non_formulation.",
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

    return False, "", ""


def collapse_exclusion_reason(
    row: dict[str, str], core_fields: dict[str, str]
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
    if has_context_tag(row, {"doe", "checkpoint_validation", "center_point", "post_processing"}):
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
    ]
    return "|".join(signature_parts)


def populated_core_field_count(core_fields: dict[str, str]) -> int:
    return sum(1 for value in core_fields.values() if value and value != "unknown")


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


def group_has_clear_redundancy_signal(group_rows: list[dict[str, str]]) -> bool:
    candidate_sources = {
        normalize_text(row.get("candidate_source", "")) for row in group_rows if row.get("candidate_source")
    }
    return "figure_variable_sweep" in candidate_sources and "llm_extracted" in candidate_sources


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
        return list(reader)


def load_relation_metadata(
    relation_records_tsv: Path | None,
) -> dict[str, dict[str, Any]]:
    if relation_records_tsv is None:
        return {}
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


def build_summary_markdown(
    input_path: Path,
    final_rows: list[dict[str, str]],
    decision_rows: list[dict[str, str]],
    summary_path: Path,
    relation_records_tsv: Path | None,
) -> None:
    decision_counts = defaultdict(int)
    for row in decision_rows:
        decision_counts[row["decision"]] += 1

    per_key_final = defaultdict(int)
    for row in final_rows:
        per_key_final[row["key"]] += 1

    content = [
        "# Final Output Summary v1",
        "",
        "## Scope",
        "",
        "This summary describes phase 1 of the minimal final-output layer. It is intentionally conservative and only applies explicit non-formulation filtering plus narrow clear-signature collapse.",
        "",
        "## Input",
        "",
        f"- candidate_input_tsv: `{input_path}`",
        (
            f"- relation_records_tsv: `{relation_records_tsv}`"
            if relation_records_tsv is not None
            else "- relation_records_tsv: `not provided`"
        ),
        "",
        "## What phase 1 currently handles",
        "",
        "- filters rows explicitly marked as non-formulation or characterization-only post-processing rows",
        "- computes a conservative core-parameter signature from current candidate-row fields",
        "- collapses rows only when signature completeness is high and exclusion tags are absent",
        "- preserves provenance by retaining representative-row metadata and a row-level decision trace",
        "",
        "## What phase 1 intentionally does not handle",
        "",
        "- broad scientific reconstruction or inheritance repair",
        "- generalized DOE collapse beyond conservative phase-1 exclusions",
        "- Stage 5B benchmark comparison against GT",
        "- modeling-target-specific filtering such as PLGA-only export subsets",
        "",
        "## Filtering rules applied",
        "",
        "- `explicit_candidate_non_formulation`",
        "- `characterization_only_post_processing`",
        "",
        "## Collapse rules applied",
        "",
        "- collapse only if polymer identity and loaded state are known",
        "- collapse only if the row is not tagged as `doe`, `checkpoint_validation`, `center_point`, or `post_processing`",
        "- collapse only if the conservative core signature has at least five populated components",
        "- collapse only if a clear mixed-source redundancy signal is present, currently `llm_extracted` plus `figure_variable_sweep` for the same signature",
        "- if uncertain, keep rows separate",
        "",
        "## Decision counts",
        "",
        f"- kept: `{decision_counts['kept']}`",
        f"- filtered_non_formulation: `{decision_counts['filtered_non_formulation']}`",
        f"- collapsed_into_existing: `{decision_counts['collapsed_into_existing']}`",
        f"- final_rows: `{len(final_rows)}`",
        "",
        "## Final rows by paper",
        "",
    ]
    for key in sorted(per_key_final):
        content.append(f"- `{key}`: `{per_key_final[key]}`")
    content.extend(
        [
            "",
            "## Open questions still visible after phase 1",
            "",
            "- exact core-signature fields for broader collapse remain unresolved",
            "- baseline versus optimized provenance handling is still conservative",
            "- parent/variant collapse policy is still intentionally narrow",
            "- relation artifacts are currently carried as provenance only and do not yet drive phase-1 collapse rules",
            "- DOE-aware closure still needs a later explicit contract",
            "- benchmark comparison still requires the separate Stage 5B comparison step",
        ]
    )
    summary_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def build_minimal_final_output(
    input_tsv: Path,
    out_dir: Path,
    relation_records_tsv: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_candidate_rows(input_tsv)
    if not rows:
        raise ValueError(f"No candidate rows found in {input_tsv}")
    relation_metadata = load_relation_metadata(relation_records_tsv)

    original_fieldnames = list(rows[0].keys())
    core_by_id: dict[str, dict[str, str]] = {}
    filtered_ids: set[str] = set()
    filter_rules: dict[str, tuple[str, str]] = {}
    eligible_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    collapse_signature_by_id: dict[str, str] = {}

    for row in rows:
        source_id = row_source_key(row)
        core_fields = build_core_fields(row)
        core_by_id[source_id] = core_fields
        should_filter, filter_rule, filter_reason = should_filter_non_formulation(row, core_fields)
        if should_filter:
            filtered_ids.add(source_id)
            filter_rules[source_id] = (filter_rule, filter_reason)
            continue

        exclusion = collapse_exclusion_reason(row, core_fields)
        if exclusion:
            continue
        signature = build_collapse_signature(row, core_fields)
        collapse_signature_by_id[source_id] = signature
        eligible_groups[signature].append(row)

    representative_by_signature: dict[str, dict[str, str]] = {}
    final_id_by_source_id: dict[str, str] = {}
    collapsed_ids: set[str] = set()

    for signature, group_rows in eligible_groups.items():
        if len(group_rows) < 2:
            continue
        if not group_has_clear_redundancy_signal(group_rows):
            continue
        representative = choose_representative(group_rows, core_by_id)
        representative_by_signature[signature] = representative
        final_formulation_id = make_final_formulation_id(representative, signature)
        for row in group_rows:
            source_key = row_source_key(row)
            final_id_by_source_id[source_key] = final_formulation_id
            if source_key != row_source_key(representative):
                collapsed_ids.add(source_key)

    representative_source_keys = {
        row_source_key(representative)
        for representative in representative_by_signature.values()
    }

    final_rows: list[dict[str, str]] = []
    decision_rows: list[dict[str, str]] = []

    source_rows_by_final_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        source_id = row["formulation_id"]
        source_key = row_source_key(row)
        core_fields = core_by_id[source_key]
        key_fields_used = build_key_fields_used(core_fields)

        if source_key in filtered_ids:
            rule, reason = filter_rules[source_key]
            decision = RowDecision(
                decision="filtered_non_formulation",
                target_final_formulation_id="",
                decision_rule=rule,
                decision_reason=reason,
                key_fields_used=key_fields_used,
                confidence_or_rule_scope="phase1_conservative_filter",
                notes="Row is excluded from final formulation closure.",
            )
        elif source_key in collapsed_ids:
            target_final_formulation_id = final_id_by_source_id[source_key]
            decision = RowDecision(
                decision="collapsed_into_existing",
                target_final_formulation_id=target_final_formulation_id,
                decision_rule="clear_core_signature_overlap",
                decision_reason="Row shares a conservative phase-1 core signature with a higher-priority representative row.",
                key_fields_used=key_fields_used,
                confidence_or_rule_scope="phase1_conservative_collapse",
                notes=f"collapse_signature={collapse_signature_by_id[source_key]}",
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
            decision = RowDecision(
                decision="kept",
                target_final_formulation_id=target_final_formulation_id,
                decision_rule=(
                    "kept_as_representative_after_collapse"
                    if source_key in representative_source_keys
                    else "kept_no_clear_phase1_overlap"
                ),
                decision_reason=(
                    "Representative row retained for a clear overlap group."
                    if source_key in representative_source_keys
                    else "No explicit non-formulation rule or clear conservative collapse rule applied."
                ),
                key_fields_used=key_fields_used,
                confidence_or_rule_scope="phase1_final_output",
                notes=(f"collapse_signature={collapse_signature}" if collapse_signature else "No collapse signature used."),
            )
            source_rows_by_final_id[target_final_formulation_id].append(row)

        decision_rows.append(
            {
                "zotero_key": row.get("key", ""),
                "source_formulation_id": source_id,
                "source_raw_formulation_label": row.get("raw_formulation_label", ""),
                "decision": decision.decision,
                "target_final_formulation_id": decision.target_final_formulation_id,
                "decision_rule": decision.decision_rule,
                "decision_reason": decision.decision_reason,
                "key_fields_used": decision.key_fields_used,
                "confidence_or_rule_scope": decision.confidence_or_rule_scope,
                "notes": decision.notes,
            }
        )

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
            "collapse_signature": collapse_signature_by_id.get(
                row_source_key(representative), ""
            ),
            "loaded_state_final": representative_core["loaded_state"],
            "polymer_identity_final": representative_core["polymer_identity"],
            "final_output_rule": (
                "representative_after_collapse"
                if len(source_group) > 1
                else "kept_without_collapse"
            ),
            "relation_graph_ids": json.dumps(relation_graph_ids, ensure_ascii=True),
            "relation_method_group_ids": json.dumps(relation_method_group_ids, ensure_ascii=True),
            "relation_parent_candidate_ids": json.dumps(
                relation_parent_candidate_ids, ensure_ascii=True
            ),
            "relation_record_count": str(relation_row_count),
        }
        for field in original_fieldnames:
            final_row[field] = representative.get(field, "")
        final_rows.append(final_row)

    decision_trace_path = out_dir / DECISION_TRACE_NAME
    final_table_path = out_dir / FINAL_TABLE_NAME
    summary_path = out_dir / SUMMARY_NAME

    write_tsv(
        decision_trace_path,
        [
            "zotero_key",
            "source_formulation_id",
            "source_raw_formulation_label",
            "decision",
            "target_final_formulation_id",
            "decision_rule",
            "decision_reason",
            "key_fields_used",
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
            "collapse_signature",
            "loaded_state_final",
            "polymer_identity_final",
            "final_output_rule",
            "relation_graph_ids",
            "relation_method_group_ids",
            "relation_parent_candidate_ids",
            "relation_record_count",
            *original_fieldnames,
        ],
        final_rows,
    )

    build_summary_markdown(
        input_tsv,
        final_rows,
        decision_rows,
        summary_path,
        relation_records_tsv,
    )

    return {
        "input_rows": len(rows),
        "final_rows": len(final_rows),
        "filtered_rows": sum(1 for row in decision_rows if row["decision"] == "filtered_non_formulation"),
        "collapsed_rows": sum(1 for row in decision_rows if row["decision"] == "collapsed_into_existing"),
        "kept_rows": sum(1 for row in decision_rows if row["decision"] == "kept"),
        "final_table_path": final_table_path,
        "decision_trace_path": decision_trace_path,
        "summary_path": summary_path,
        "relation_records_tsv": relation_records_tsv,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build phase-1 minimal final-output artifacts from Stage2 candidate-instance TSV output."
    )
    parser.add_argument("--input-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--relation-records-tsv", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = build_minimal_final_output(
        args.input_tsv,
        args.out_dir,
        relation_records_tsv=args.relation_records_tsv,
    )
    print(
        json.dumps(
            {
                "input_rows": stats["input_rows"],
                "final_rows": stats["final_rows"],
                "filtered_rows": stats["filtered_rows"],
                "collapsed_rows": stats["collapsed_rows"],
                "kept_rows": stats["kept_rows"],
                "relation_records_tsv": (
                    str(stats["relation_records_tsv"]) if stats["relation_records_tsv"] else ""
                ),
                "final_table_path": str(stats["final_table_path"]),
                "decision_trace_path": str(stats["decision_trace_path"]),
                "summary_path": str(stats["summary_path"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
