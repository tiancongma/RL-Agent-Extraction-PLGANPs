#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT

csv.field_size_limit(sys.maxsize)

FINAL_TABLE = "final_formulation_table_v1.tsv"
DECISION_TRACE = "final_output_decision_trace_v1.tsv"
SUMMARY = "final_output_summary_v1.md"

DONOR_COPY_SUFFIXES = (
    "_value",
    "_value_text",
    "_scope",
    "_membership_confidence",
    "_evidence_region_type",
    "_missing_reason",
)
DONOR_COPY_FIELDS = {
    "polymer_identity_final",
    "loaded_state_final",
    "preparation_method",
    "emulsion_structure",
    "identity_variables_json",
    "method_group_signature_hint",
    "table_cell_bindings_json",
    "table_formulation_scopes_json",
    "table_variable_roles_json",
    "selection_markers_json",
    "inheritance_markers_json",
    "context_inheritance_markers_json",
    "protocol_inheritance_markers_json",
    "relation_cues_json",
    "typed_inheritance_fields_json",
    "typed_doe_factors_json",
    "result_binding_candidates_json",
    "boundary_markers_json",
    "preparation_inheritance_json",
    "shared_parameters_json",
}
IDENTITY_LOCKED_FIELDS = {
    "final_formulation_id",
    "representative_source_formulation_id",
    "representative_source_raw_formulation_label",
    "source_candidate_count",
    "source_candidate_ids",
    "source_candidate_labels",
    "source_candidate_sources",
    "retention_reason",
    "review_needed",
    "family_id",
    "parent_core_row_id",
    "variant_role",
    "payload_state",
    "benchmark_default_include",
    "final_output_rule",
    "key",
    "model",
    "local_instance_id",
    "formulation_id",
    "raw_formulation_label",
    "candidate_source",
    "stage2_semantic_source_mode",
    "semantic_universe_authority",
    "row_materialization_mode",
    "semantic_scope_authority",
    "semantic_scope_ref",
    "table_id",
    "table_row_id",
    "supporting_evidence_refs",
    "evidence_section",
    "evidence_span_text",
    "instance_evidence_region_type",
}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar(row.get(field, "")) for field in fields})


def norm_label(value: str) -> str:
    value = str(value or "").lower().strip()
    value = re.sub(r"[\s_./:;|()[\]{}+-]+", "", value)
    return value


def label_values(row: dict[str, str]) -> set[str]:
    values = {
        row.get("formulation_id", ""),
        row.get("local_instance_id", ""),
        row.get("raw_formulation_label", ""),
        row.get("representative_source_formulation_id", ""),
        row.get("representative_source_raw_formulation_label", ""),
    }
    for field in ("source_candidate_ids", "source_candidate_labels", "universe_aliases_json"):
        raw = row.get(field, "")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    values.update(str(item) for item in parsed)
            except json.JSONDecodeError:
                values.add(raw)
    return {norm_label(value) for value in values if norm_label(value)}


def build_old_index(old_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in old_rows:
        key = row.get("key", "") or row.get("paper_key", "")
        for label in label_values(row):
            index[(key, label)].append(row)
    return index


def find_unique_donor(row: dict[str, str], old_index: dict[tuple[str, str], list[dict[str, str]]]) -> tuple[dict[str, str] | None, str]:
    key = row.get("key", "")
    labels = label_values(row)
    hits: dict[str, dict[str, str]] = {}
    for label in labels:
        for hit in old_index.get((key, label), []):
            hits[hit.get("final_formulation_id", "")] = hit
    if len(hits) == 1:
        return next(iter(hits.values())), "unique_paper_local_identity_label_match"
    if len(hits) > 1:
        return None, "ambiguous_identity_match"
    return None, "no_identity_match"


def copy_allowed_value_fields(target: dict[str, Any], donor: dict[str, str]) -> list[str]:
    copied: list[str] = []
    for field, value in donor.items():
        if field in IDENTITY_LOCKED_FIELDS:
            continue
        if field in DONOR_COPY_FIELDS or field.endswith(DONOR_COPY_SUFFIXES):
            if value:
                target[field] = value
                copied.append(field)
    if donor.get("doi"):
        target["doi"] = donor["doi"]
    return copied


def build_candidate_rows(promoted_rows: list[dict[str, str]], old_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    old_fields = list(old_rows[0].keys()) if old_rows else []
    old_index = build_old_index(old_rows)
    final_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for row in promoted_rows:
        out = {field: "" for field in old_fields}
        out.update(row)
        out["final_output_rule"] = "kept_from_274_frozen_universe"
        out["retention_reason"] = "Retained from 274 formulation-universe row authority; downstream value layers may not create or remove rows."
        donor, match_rule = find_unique_donor(row, old_index)
        copied: list[str] = []
        if donor is not None:
            copied = copy_allowed_value_fields(out, donor)
        out["value_reuse_source_final_formulation_id"] = donor.get("final_formulation_id", "") if donor else ""
        out["value_reuse_match_rule"] = match_rule
        out["value_reuse_copied_field_count"] = str(len(copied))
        out["value_reuse_copied_fields_json"] = json.dumps(copied, ensure_ascii=False)
        final_rows.append(out)
        reuse_rows.append(
            {
                "paper_key": row.get("key", ""),
                "final_formulation_id": row.get("final_formulation_id", ""),
                "universe_label": row.get("raw_formulation_label", ""),
                "match_rule": match_rule,
                "donor_final_formulation_id": donor.get("final_formulation_id", "") if donor else "",
                "copied_field_count": len(copied),
                "copied_fields_json": copied,
            }
        )
        if not donor:
            gap_rows.append(
                {
                    "paper_key": row.get("key", ""),
                    "final_formulation_id": row.get("final_formulation_id", ""),
                    "universe_label": row.get("raw_formulation_label", ""),
                    "gap_type": match_rule,
                    "recommended_action": "value_extraction_or_manual_mapping_without_row_creation",
                }
            )
    return final_rows, reuse_rows, gap_rows


def write_context(out_dir: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]], reuse_rows: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> None:
    match_counts = Counter(row.get("match_rule", "") for row in reuse_rows)
    lines = [
        "# RUN_CONTEXT",
        "",
        "## Run purpose",
        "Build a 2853-row final-table candidate from the promoted 274 formulation universe.",
        "",
        "## Run type",
        "Diagnostic Stage5-compatible final-table candidate. Not benchmark-valid final output.",
        "",
        "## Boundary",
        "The fixed row universe is the promoted 274 formulation universe. Old Stage5 rows are advisory value/evidence donors only under unique identity matches.",
        "",
        "## Inputs",
        f"- promoted_stage2_tsv: `{repo_path(args.promoted_stage2_tsv)}`",
        f"- old_final_table_tsv: `{repo_path(args.old_final_table_tsv)}`",
        "",
        "## Outputs",
        f"- `{FINAL_TABLE}`",
        "- `analysis/value_reuse_ledger_v1.tsv`",
        "- `analysis/value_gap_queue_v1.tsv`",
        "",
        "## Counts",
        f"- final_rows: {len(rows)}",
        f"- value_gap_rows: {len(gaps)}",
    ]
    for key, value in sorted(match_counts.items()):
        lines.append(f"- match.{key}: {value}")
    lines.append("")
    lines.append(f"generated_at: `{datetime.now(timezone.utc).isoformat()}`")
    (out_dir / "RUN_CONTEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = [
        "# Final Output Summary v1",
        "",
        "Diagnostic final-table candidate built from 274 formulation-universe authority.",
        "",
        f"- final_rows: `{len(rows)}`",
        f"- value_gap_rows: `{len(gaps)}`",
    ]
    for key, value in sorted(match_counts.items()):
        summary.append(f"- match.{key}: `{value}`")
    (out_dir / SUMMARY).write_text("\n".join(summary) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build final-table candidate from promoted formulation universe rows.")
    parser.add_argument("--promoted-stage2-tsv", required=True)
    parser.add_argument("--old-final-table-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = repo_path(args.out_dir)
    promoted_rows = read_tsv(repo_path(args.promoted_stage2_tsv))
    old_rows = read_tsv(repo_path(args.old_final_table_tsv))
    rows, reuse_rows, gaps = build_candidate_rows(promoted_rows, old_rows)
    old_fields = list(old_rows[0].keys()) if old_rows else []
    extra_fields = [field for field in rows[0].keys() if field not in old_fields] if rows else []
    fields = old_fields + extra_fields
    write_tsv(out_dir / FINAL_TABLE, rows, fields)
    trace_rows = [
        {
            "final_formulation_id": row["final_formulation_id"],
            "decision": "kept",
            "decision_rule": "kept_from_274_frozen_universe",
            "decision_reason": row["retention_reason"],
            "source_row_authority": "274_formulation_universe_frozen_v1",
            "value_reuse_match_rule": row.get("value_reuse_match_rule", ""),
            "value_reuse_source_final_formulation_id": row.get("value_reuse_source_final_formulation_id", ""),
        }
        for row in rows
    ]
    write_tsv(out_dir / DECISION_TRACE, trace_rows, list(trace_rows[0].keys()) if trace_rows else [])
    write_tsv(out_dir / "analysis" / "value_reuse_ledger_v1.tsv", reuse_rows, list(reuse_rows[0].keys()) if reuse_rows else [])
    write_tsv(out_dir / "analysis" / "value_gap_queue_v1.tsv", gaps, list(gaps[0].keys()) if gaps else ["paper_key", "final_formulation_id"])
    write_context(out_dir, args=args, rows=rows, reuse_rows=reuse_rows, gaps=gaps)
    print(f"final_rows={len(rows)}")
    print(f"value_gap_rows={len(gaps)}")
    print(f"out_tsv={out_dir / FINAL_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
