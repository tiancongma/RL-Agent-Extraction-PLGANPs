#!/usr/bin/env python3
from __future__ import annotations

"""
S5-4 value authority validator skeleton.

This validator is an internal Stage5 guardrail for candidate value rows produced by
S5 direct-value layers before any downstream merge.  It validates only candidate
row authority/evidence requirements and intentionally does not read benchmark GT
or system final-table values.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    from src.stage5_benchmark import build_s5_3_llm_direct_value_candidates_v1 as s5_3
except ImportError:  # pragma: no cover - supports direct script execution from repo root/path variants.
    s5_3 = None  # type: ignore[assignment]

ENTRYPOINT = "src/stage5_benchmark/validate_s5_value_candidates_v1.py"
BOUNDARY_CLASS = "Stage5 internal validation boundary"
BENCHMARK_VALID_STATUS = "no"

DECISIONS_TSV_NAME = "s5_4_value_authority_decisions_v1.tsv"
ACCEPTED_TSV_NAME = "s5_4_accepted_direct_values_v1.tsv"
REJECTED_TSV_NAME = "s5_4_rejected_value_candidates_v1.tsv"
REVIEW_TSV_NAME = "s5_4_value_review_queue_v1.tsv"
SUMMARY_JSON_NAME = "s5_4_value_authority_summary_v1.json"
RUN_CONTEXT_NAME = "RUN_CONTEXT.md"

BASE_CANDIDATE_COLUMNS = list(getattr(s5_3, "CANDIDATE_COLUMNS", [])) or [
    "paper_key",
    "formulation_id",
    "field_name",
    "value_text",
    "unit_text",
    "raw_cell_text",
    "direct_or_derived",
    "evidence_type",
    "evidence_scope",
    "source_file",
    "source_table_id",
    "source_row_id",
    "source_quote",
    "confidence",
    "needs_review",
]

AUTHORITY_COLUMNS = [
    "s5_4_candidate_id",
    "s5_4_source_layer",
    "s5_4_decision",
    "s5_4_reason",
    "s5_4_review_needed",
]

AMBIGUOUS_SCOPE_VALUES = {
    "ambiguous",
    "unknown",
    "unclear",
    "mixed",
    "multiple_possible",
    "needs_scope_review",
}

S5_3_EXCLUDED_MECHANICAL_FIELDS = {
    "drug_name",
    "polymer_name",
    "polymer_mass_mg",
    "drug_mass_mg",
    "O_volume_mL",
    "external_aqueous_phase_volume_mL",
    "surfactant_name",
    "particle_size_nm",
    "pdi",
    "zeta_mV",
    "ee_percent",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _clean(value).lower().replace("-", "_").replace(" ", "_")


def resolve_existing_path(value: Path, role: str) -> Path:
    path = value.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required explicit {role} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required explicit {role} is not a file: {path}")
    return path


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return [], []
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _clean(row.get(column)) for column in columns})


def _union_columns(*column_sets: list[str]) -> list[str]:
    columns: list[str] = []
    for column_set in column_sets:
        for column in column_set:
            if column and column not in columns:
                columns.append(column)
    return columns


def load_candidates(candidate_tsv: Path, rule_direct_values_tsv: Path | None) -> tuple[list[str], list[dict[str, str]]]:
    candidate_columns, candidate_rows = read_tsv(candidate_tsv)
    all_columns = _union_columns(BASE_CANDIDATE_COLUMNS, candidate_columns)
    rows: list[dict[str, str]] = []
    for index, row in enumerate(candidate_rows, start=1):
        enriched = dict(row)
        enriched["s5_4_candidate_id"] = f"s5_3:{index:06d}"
        enriched["s5_4_source_layer"] = "s5_3_llm_direct_value_candidates_v1"
        rows.append(enriched)

    if rule_direct_values_tsv is not None:
        rule_columns, rule_rows = read_tsv(rule_direct_values_tsv)
        all_columns = _union_columns(all_columns, rule_columns)
        for index, row in enumerate(rule_rows, start=1):
            enriched = dict(row)
            enriched["s5_4_candidate_id"] = f"s5_2:{index:06d}"
            enriched["s5_4_source_layer"] = "s5_2_rule_direct_values_v1"
            rows.append(enriched)

    return all_columns, rows


def evaluate_candidate(row: dict[str, str]) -> tuple[str, str, str]:
    field_name = _clean(row.get("field_name"))
    direct_or_derived = _norm(row.get("direct_or_derived"))
    source_quote = _clean(row.get("source_quote"))
    evidence_scope = _norm(row.get("evidence_scope"))

    if field_name in S5_3_EXCLUDED_MECHANICAL_FIELDS:
        return "rejected", "s5_3_excluded_mechanical_field_not_allowed", "no"
    if direct_or_derived == "derived":
        return "rejected", "derived_value_not_allowed_in_direct_layer", "no"
    if direct_or_derived != "direct":
        return "rejected", "direct_or_derived_must_be_direct", "no"
    if not source_quote:
        return "rejected", "missing_source_quote_for_direct_candidate", "no"
    if evidence_scope in AMBIGUOUS_SCOPE_VALUES:
        return "review_needed", "ambiguous_scope_requires_manual_review", "yes"
    return "accepted", "direct_candidate_has_required_source_quote_and_scope", "no"


def validate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    decisions: list[dict[str, str]] = []
    for row in rows:
        decision, reason, review_needed = evaluate_candidate(row)
        decided = dict(row)
        decided["s5_4_decision"] = decision
        decided["s5_4_reason"] = reason
        decided["s5_4_review_needed"] = review_needed
        if review_needed == "yes":
            decided["needs_review"] = "yes"
        decisions.append(decided)
    return decisions


def render_run_context(
    *,
    candidate_tsv: Path,
    rule_direct_values_tsv: Path | None,
    out_dir: Path,
    outputs: dict[str, Path],
    summary: dict[str, Any],
) -> str:
    rule_path = str(rule_direct_values_tsv) if rule_direct_values_tsv else ""
    return "\n".join(
        [
            "# RUN_CONTEXT",
            "",
            "## 1. Entrypoint",
            "",
            f"- entrypoint: `{ENTRYPOINT}`",
            f"- boundary_class: `{BOUNDARY_CLASS}`",
            "",
            "## 2. Benchmark-valid status",
            "",
            f"- benchmark_valid_status: `{BENCHMARK_VALID_STATUS}`",
            "- benchmark_valid: `no`",
            "- reason: `S5-4 is an internal validation boundary for candidate value authority, not a benchmark scoring step`",
            "",
            "## 3. Exact inputs",
            "",
            f"- candidate_tsv: `{candidate_tsv}`",
            f"- rule_direct_values_tsv: `{rule_path}`",
            "",
            "## 4. Output paths",
            "",
            f"- out_dir: `{out_dir}`",
            *[f"- {name}: `{path}`" for name, path in sorted(outputs.items())],
            "",
            "## 5. Validation boundary and authority policy",
            "",
            "- Stage5 internal validation boundary: validates candidate rows before merge.",
            "- Does not change formulation membership.",
            "- Does not consult GT values, benchmark answer keys, or system final-table values as authority.",
            "- Direct candidates require direct_or_derived=direct and a non-empty source_quote.",
            "- Derived rows from direct layers are rejected; ambiguous scope is routed to review-needed.",
            "- All inputs are resolved only from explicit CLI paths; no latest-directory, glob-first, or ACTIVE_RUN inference is used.",
            "",
            "## 6. Outcome summary",
            "",
            *[f"- {key}: `{value}`" for key, value in sorted(summary.items())],
            "",
        ]
    )


def build_summary(decisions: list[dict[str, str]]) -> dict[str, Any]:
    accepted = [row for row in decisions if row.get("s5_4_decision") == "accepted"]
    rejected = [row for row in decisions if row.get("s5_4_decision") == "rejected"]
    review = [row for row in decisions if row.get("s5_4_decision") == "review_needed"]
    return {
        "benchmark_valid": "no",
        "boundary_class": BOUNDARY_CLASS,
        "candidate_rows": len(decisions),
        "accepted_direct_rows": len(accepted),
        "rejected_rows": len(rejected),
        "review_needed_rows": len(review),
        "derived_rejections": sum(1 for row in rejected if row.get("s5_4_reason") == "derived_value_not_allowed_in_direct_layer"),
        "missing_quote_rejections": sum(
            1 for row in rejected if row.get("s5_4_reason") == "missing_source_quote_for_direct_candidate"
        ),
        "ambiguous_scope_review_rows": sum(
            1 for row in review if row.get("s5_4_reason") == "ambiguous_scope_requires_manual_review"
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate S5 direct value candidate authority before merge.")
    parser.add_argument("--candidate-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rule-direct-values-tsv", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_tsv = resolve_existing_path(args.candidate_tsv, "candidate-tsv")
    rule_direct_values_tsv = (
        resolve_existing_path(args.rule_direct_values_tsv, "rule-direct-values-tsv")
        if args.rule_direct_values_tsv is not None
        else None
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_columns, rows = load_candidates(candidate_tsv, rule_direct_values_tsv)
    decisions = validate_rows(rows)
    accepted = [row for row in decisions if row.get("s5_4_decision") == "accepted"]
    rejected = [row for row in decisions if row.get("s5_4_decision") == "rejected"]
    review = [row for row in decisions if row.get("s5_4_decision") == "review_needed"]

    output_columns = _union_columns(AUTHORITY_COLUMNS, input_columns)
    accepted_columns = _union_columns(AUTHORITY_COLUMNS, input_columns)
    rejected_columns = _union_columns(AUTHORITY_COLUMNS, input_columns)
    review_columns = _union_columns(AUTHORITY_COLUMNS, input_columns)

    decisions_path = out_dir / DECISIONS_TSV_NAME
    accepted_path = out_dir / ACCEPTED_TSV_NAME
    rejected_path = out_dir / REJECTED_TSV_NAME
    review_path = out_dir / REVIEW_TSV_NAME
    summary_path = out_dir / SUMMARY_JSON_NAME
    run_context_path = out_dir / RUN_CONTEXT_NAME

    write_tsv(decisions_path, output_columns, decisions)
    write_tsv(accepted_path, accepted_columns, accepted)
    write_tsv(rejected_path, rejected_columns, rejected)
    write_tsv(review_path, review_columns, review)

    summary = build_summary(decisions)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = {
        "authority_decisions_tsv": decisions_path,
        "accepted_direct_values_tsv": accepted_path,
        "rejected_value_candidates_tsv": rejected_path,
        "value_review_queue_tsv": review_path,
        "value_authority_summary_json": summary_path,
        "run_context_md": run_context_path,
    }
    run_context_path.write_text(
        render_run_context(
            candidate_tsv=candidate_tsv,
            rule_direct_values_tsv=rule_direct_values_tsv,
            out_dir=out_dir,
            outputs=outputs,
            summary=summary,
        ),
        encoding="utf-8",
    )

    return {"status": "ok", "out_dir": str(out_dir), "outputs": {k: str(v) for k, v in outputs.items()}, **summary}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
