#!/usr/bin/env python3
from __future__ import annotations

"""
Diagnostic S5-3 scope-candidate policy builder.

This helper designs and applies the non-GT S5-3 scope policy from the
2026-05-06 end-to-end boundary repair plan.  It does not call an LLM, does not
read GT, does not create value candidates, and does not alter Stage5 row
membership.  It only classifies explicitly supplied source-observability rows as
eligible or excluded S5-3 scope targets for an already fixed Stage5 row surface.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ENTRYPOINT = "src/stage5_benchmark/build_s5_3_scope_candidates_v1.py"
BOUNDARY_CLASS = "diagnostic/supporting_scope_policy"
BENCHMARK_VALID_STATUS = "no"
SCOPE_CANDIDATES_TSV_NAME = "s5_3_scope_candidates_v1.tsv"
SUMMARY_JSON_NAME = "s5_3_scope_candidates_summary_v1.json"
INPUT_MANIFEST_TSV_NAME = "s5_3_scope_input_manifest_v1.tsv"
RUN_CONTEXT_NAME = "RUN_CONTEXT.md"

PAPER_KEY_ALIASES = ("paper_key", "key", "article_key", "source_key")
FORMULATION_ID_ALIASES = (
    "formulation_id",
    "final_formulation_id",
    "article_formulation_id",
    "parent_core_row_id",
)
FIELD_NAME_ALIASES = ("field_name", "target_field", "target_field_name", "schema_field")

VALID_SCOPE_TRIGGERS = {
    "upstream_reported_s5_2_failed",
    "evidence_block_mentions_value_type",
    "s2_2_unmapped_row_local_cell",
    "stage3_shared_value_not_promoted",
    "typed_direct_candidate_unassigned",
}

DERIVED_MARKERS = {"derived", "derived_only", "calculated", "calculation", "formula"}
AMBIGUOUS_SCOPE_MARKERS = {"ambiguous", "ambiguous_source_scope", "unclear_scope"}
NO_SOURCE_MARKERS = {"not_reported", "no_source", "absent", "not_reported_no_source"}
UNRELATED_SCOPE_MARKERS = {"unrelated", "assay", "control", "comparator", "post_processing_only"}
UNRESOLVED_IDENTITY_MARKERS = {"unresolved", "blocked_alignment", "identity_unresolved", "row_identity_unresolved"}

OUTPUT_COLUMNS = [
    "paper_key",
    "formulation_id",
    "field_name",
    "scope_trigger",
    "source_locator",
    "upstream_boundary",
    "why_s5_2_failed",
    "eligible_for_s5_3",
    "exclusion_reason",
    "source_observability_status",
    "row_identity_status",
    "direct_or_derived",
    "source_context",
    "input_row_index",
]

INPUT_MANIFEST_COLUMNS = ["input_role", "path", "exists", "metadata_json"]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _clean(value).lower().replace(" ", "_").replace("-", "_")


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = _clean(row.get(name))
        if value:
            return value
    return ""


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _clean(row.get(column)) for column in columns})


def resolve_existing_path(value: Path, role: str) -> Path:
    path = value.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required explicit {role} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required explicit {role} is not a file: {path}")
    return path


def fixed_row_key(row: dict[str, str]) -> tuple[str, str]:
    return (_first_present(row, PAPER_KEY_ALIASES), _first_present(row, FORMULATION_ID_ALIASES))


def build_fixed_row_index(final_rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {key for key in (fixed_row_key(row) for row in final_rows) if key[0] and key[1]}


def classify_scope_row(row: dict[str, str], fixed_rows: set[tuple[str, str]], input_row_index: int) -> dict[str, str]:
    paper_key = _first_present(row, PAPER_KEY_ALIASES)
    formulation_id = _first_present(row, FORMULATION_ID_ALIASES)
    field_name = _first_present(row, FIELD_NAME_ALIASES)
    scope_trigger = _norm(row.get("scope_trigger"))
    source_locator = _clean(row.get("source_locator"))
    upstream_boundary = _clean(row.get("upstream_boundary"))
    why_s5_2_failed = _clean(row.get("why_s5_2_failed"))
    source_status = _norm(row.get("source_observability_status") or row.get("source_status"))
    row_identity_status = _norm(row.get("row_identity_status") or row.get("identity_status"))
    direct_or_derived = _norm(row.get("direct_or_derived") or row.get("value_kind"))
    source_context = _clean(row.get("source_context") or row.get("evidence_context"))

    exclusion_reason = ""
    if not paper_key or not formulation_id:
        exclusion_reason = "missing_fixed_row_identity"
    elif (paper_key, formulation_id) not in fixed_rows:
        exclusion_reason = "row_not_in_fixed_stage5_universe"
    elif not field_name:
        exclusion_reason = "missing_field_name"
    elif not scope_trigger or scope_trigger not in VALID_SCOPE_TRIGGERS:
        exclusion_reason = "blank_field_or_no_source_observability_signal"
    elif not source_locator:
        exclusion_reason = "blank_field_or_no_source_observability_signal"
    elif source_status in NO_SOURCE_MARKERS:
        exclusion_reason = "not_reported_no_source"
    elif source_status in AMBIGUOUS_SCOPE_MARKERS:
        exclusion_reason = "ambiguous_source_scope"
    elif source_status in UNRELATED_SCOPE_MARKERS or _norm(source_context) in UNRELATED_SCOPE_MARKERS:
        exclusion_reason = "value_only_in_unrelated_context"
    elif direct_or_derived in DERIVED_MARKERS:
        exclusion_reason = "derived_only_candidate_for_direct_field"
    elif row_identity_status in UNRESOLVED_IDENTITY_MARKERS:
        exclusion_reason = "row_identity_alignment_unresolved"
    elif not upstream_boundary:
        exclusion_reason = "missing_upstream_boundary"
    elif not why_s5_2_failed:
        exclusion_reason = "missing_s5_2_failure_reason"

    return {
        "paper_key": paper_key,
        "formulation_id": formulation_id,
        "field_name": field_name,
        "scope_trigger": scope_trigger,
        "source_locator": source_locator,
        "upstream_boundary": upstream_boundary,
        "why_s5_2_failed": why_s5_2_failed,
        "eligible_for_s5_3": "yes" if not exclusion_reason else "no",
        "exclusion_reason": exclusion_reason,
        "source_observability_status": source_status or "source_observable",
        "row_identity_status": row_identity_status or "resolved",
        "direct_or_derived": direct_or_derived or "direct",
        "source_context": source_context,
        "input_row_index": str(input_row_index),
    }


def build_scope_candidates(final_rows: list[dict[str, str]], observability_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    fixed_rows = build_fixed_row_index(final_rows)
    return [classify_scope_row(row, fixed_rows, index) for index, row in enumerate(observability_rows, start=1)]


def summarize(rows: list[dict[str, str]], *, final_table_tsv: Path, source_observability_tsv: Path) -> dict[str, Any]:
    exclusion_counts = Counter(row["exclusion_reason"] or "eligible" for row in rows)
    trigger_counts = Counter(row["scope_trigger"] or "missing" for row in rows)
    eligible_count = sum(1 for row in rows if row.get("eligible_for_s5_3") == "yes")
    return {
        "entrypoint": ENTRYPOINT,
        "boundary_class": BOUNDARY_CLASS,
        "diagnostic_only": True,
        "benchmark_valid": "no",
        "live_llm_calls": "no",
        "gt_consumed": "no",
        "final_table_tsv": str(final_table_tsv),
        "source_observability_tsv": str(source_observability_tsv),
        "scope_candidate_rows": len(rows),
        "eligible_for_s5_3_rows": eligible_count,
        "excluded_rows": len(rows) - eligible_count,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "scope_trigger_counts": dict(sorted(trigger_counts.items())),
    }


def render_run_context(
    *,
    final_table_tsv: Path,
    source_observability_tsv: Path,
    out_dir: Path,
    outputs: dict[str, Path],
    summary: dict[str, Any],
) -> str:
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
            "- diagnostic_only: `yes`",
            "- reason: `S5-3 non-GT source-observability scope policy only; not benchmark-valid and no value extraction performed`",
            "",
            "## 3. Exact inputs",
            "",
            f"- final_table_tsv: `{final_table_tsv}`",
            f"- source_observability_tsv: `{source_observability_tsv}`",
            "- gt_inputs_consumed: `no`",
            "- live_llm_calls: `no`",
            "",
            "## 4. Policy",
            "",
            "- Empty final-table schema slots alone are never eligible S5-3 targets.",
            "- Eligibility requires an admitted fixed Stage5 row plus a source-observability trigger, source locator, upstream boundary, and S5-2 failure reason.",
            "- Excluded classes include not-reported/no-source, ambiguous scope, derived-only direct-field candidates, unresolved row identity/alignment, and unrelated assay/control/comparator context.",
            "- This helper does not read GT and does not use latest-directory, glob-first, or recency inference.",
            "",
            "## 5. Output paths",
            "",
            f"- out_dir: `{out_dir}`",
            *[f"- {name}: `{path}`" for name, path in sorted(outputs.items())],
            "",
            "## 6. Outcome summary",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build diagnostic S5-3 scope candidates from explicit non-GT source-observability signals."
    )
    parser.add_argument("--final-table-tsv", type=Path, required=True)
    parser.add_argument("--source-observability-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    final_table_tsv = resolve_existing_path(args.final_table_tsv, "final-table-tsv")
    source_observability_tsv = resolve_existing_path(args.source_observability_tsv, "source-observability-tsv")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    final_rows = read_tsv(final_table_tsv)
    observability_rows = read_tsv(source_observability_tsv)
    candidate_rows = build_scope_candidates(final_rows, observability_rows)

    candidates_path = out_dir / SCOPE_CANDIDATES_TSV_NAME
    summary_path = out_dir / SUMMARY_JSON_NAME
    input_manifest_path = out_dir / INPUT_MANIFEST_TSV_NAME
    run_context_path = out_dir / RUN_CONTEXT_NAME

    summary = summarize(candidate_rows, final_table_tsv=final_table_tsv, source_observability_tsv=source_observability_tsv)
    outputs = {
        "scope_candidates_tsv": candidates_path,
        "summary_json": summary_path,
        "input_manifest_tsv": input_manifest_path,
        "run_context_md": run_context_path,
    }

    write_tsv(candidates_path, OUTPUT_COLUMNS, candidate_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(
        input_manifest_path,
        INPUT_MANIFEST_COLUMNS,
        [
            {"input_role": "final_table_tsv", "path": final_table_tsv, "exists": "yes", "metadata_json": "{}"},
            {
                "input_role": "source_observability_tsv",
                "path": source_observability_tsv,
                "exists": "yes",
                "metadata_json": json.dumps({"gt_consumed": "no", "source_resolution": "explicit_cli_only"}, sort_keys=True),
            },
        ],
    )
    run_context_path.write_text(
        render_run_context(
            final_table_tsv=final_table_tsv,
            source_observability_tsv=source_observability_tsv,
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
