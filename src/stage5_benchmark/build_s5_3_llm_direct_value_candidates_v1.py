#!/usr/bin/env python3
from __future__ import annotations

"""
S5-3 diagnostic runner skeleton for direct LLM value candidates.

This is a diagnostic/supporting runner only.  It resolves only explicit input
paths supplied on the CLI, writes reproducibility placeholders, records exact
source-path metadata, and stops before any live LLM provider call.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ENTRYPOINT = "src/stage5_benchmark/build_s5_3_llm_direct_value_candidates_v1.py"
BOUNDARY_CLASS = "diagnostic/supporting_boundary"
BENCHMARK_VALID_STATUS = "no"
NO_BACKEND_MESSAGE = (
    "No LLM backend is configured; wrote diagnostic placeholders only and made no live LLM calls."
)
BACKEND_UNSUPPORTED_MESSAGE = (
    "LLM backend configuration was detected, but this S5-3 skeleton does not execute live providers."
)

CANDIDATE_TSV_NAME = "s5_3_llm_direct_value_candidates_v1.tsv"
AUDIT_TSV_NAME = "s5_3_llm_direct_value_evidence_sidecar_v1.tsv"
PROMPT_PLACEHOLDER_TSV_NAME = "s5_3_llm_prompt_audit_v1.tsv"
INPUT_MANIFEST_TSV_NAME = "s5_3_llm_input_manifest_v1.tsv"
RAW_RESPONSES_DIR_NAME = "s5_3_llm_raw_responses"
RUN_CONTEXT_NAME = "RUN_CONTEXT.md"

CANDIDATE_COLUMNS = [
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
    "llm_model",
    "prompt_hash",
    "input_artifact_hash",
]

AUDIT_COLUMNS = [
    "audit_event",
    "status",
    "message",
    "final_table_tsv",
    "decision_trace_tsv",
    "scope_manifest_tsv",
    "source_inventory_tsv",
    "source_paths_json",
]

PROMPT_PLACEHOLDER_COLUMNS = [
    "prompt_id",
    "paper_key",
    "final_formulation_id",
    "source_text_path",
    "source_table_path",
    "source_pdf_path",
    "prompt_status",
    "prompt_text_placeholder",
]

INPUT_MANIFEST_COLUMNS = [
    "input_role",
    "path",
    "exists",
    "metadata_json",
]

PAPER_KEY_ALIASES = ("paper_key", "key", "article_key", "source_key")
FORMULATION_ID_ALIASES = (
    "final_formulation_id",
    "formulation_id",
    "article_formulation_id",
    "parent_core_row_id",
)
SOURCE_TEXT_ALIASES = ("source_text_path", "text_path", "full_text_path", "article_text_path")
SOURCE_TABLE_ALIASES = ("source_table_path", "table_path", "tables_path", "extracted_tables_path")
SOURCE_PDF_ALIASES = ("source_pdf_path", "pdf_path", "article_pdf_path")


def _clean(value: Any) -> str:
    return str(value or "").strip()


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


def build_source_index(source_inventory_tsv: Path | None) -> dict[str, dict[str, str]]:
    if source_inventory_tsv is None:
        return {}
    source_index: dict[str, dict[str, str]] = {}
    for row in read_tsv(source_inventory_tsv):
        paper_key = _first_present(row, PAPER_KEY_ALIASES)
        if not paper_key:
            continue
        source_index[paper_key] = {
            "source_text_path": _first_present(row, SOURCE_TEXT_ALIASES),
            "source_table_path": _first_present(row, SOURCE_TABLE_ALIASES),
            "source_pdf_path": _first_present(row, SOURCE_PDF_ALIASES),
        }
    return source_index


def build_prompt_placeholders(
    final_rows: list[dict[str, str]],
    source_index: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, final_row in enumerate(final_rows, start=1):
        paper_key = _first_present(final_row, PAPER_KEY_ALIASES)
        formulation_id = _first_present(final_row, FORMULATION_ID_ALIASES)
        source_paths = source_index.get(paper_key, {})
        rows.append(
            {
                "prompt_id": f"prompt_{index:06d}",
                "paper_key": paper_key,
                "final_formulation_id": formulation_id,
                "source_text_path": source_paths.get("source_text_path", ""),
                "source_table_path": source_paths.get("source_table_path", ""),
                "source_pdf_path": source_paths.get("source_pdf_path", ""),
                "prompt_status": "placeholder_not_sent_no_llm_backend",
                "prompt_text_placeholder": "S5-3 diagnostic placeholder only; no live LLM call was made.",
            }
        )
    return rows


def source_paths_payload(source_index: dict[str, dict[str, str]]) -> str:
    return json.dumps(source_index, sort_keys=True, separators=(",", ":"))


def render_run_context(
    *,
    final_table_tsv: Path,
    decision_trace_tsv: Path,
    scope_manifest_tsv: Path,
    source_inventory_tsv: Path | None,
    out_dir: Path,
    outputs: dict[str, Path],
    source_index: dict[str, dict[str, str]],
    final_row_count: int,
    prompt_row_count: int,
    message: str,
) -> str:
    source_inventory_rendered = str(source_inventory_tsv) if source_inventory_tsv else ""
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
            "- reason: `diagnostic skeleton writes placeholders only, is not benchmark-valid, and performs no live LLM provider calls`",
            "",
            "## 3. Exact inputs",
            "",
            f"- final_table_tsv: `{final_table_tsv}`",
            f"- decision_trace_tsv: `{decision_trace_tsv}`",
            f"- scope_manifest_tsv: `{scope_manifest_tsv}`",
            f"- source_inventory_tsv: `{source_inventory_rendered}`",
            "",
            "## 4. Source path metadata",
            "",
            "```json",
            json.dumps(source_index, indent=2, sort_keys=True),
            "```",
            "",
            "## 5. Output paths",
            "",
            f"- out_dir: `{out_dir}`",
            *[f"- {name}: `{path}`" for name, path in sorted(outputs.items())],
            "",
            "## 6. Resolution policy",
            "",
            "- All source artifacts are resolved exclusively from explicit CLI arguments.",
            "- This runner does not use latest-directory lookup, glob-first search, or ACTIVE_RUN inference.",
            "",
            "## 7. Outcome summary",
            "",
            f"- final_input_rows: `{final_row_count}`",
            f"- prompt_placeholder_rows: `{prompt_row_count}`",
            "- candidate_rows: `0`",
            f"- message: `{message}`",
            "",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write S5-3 diagnostic placeholders for direct LLM value candidates from explicit inputs only."
    )
    parser.add_argument("--final-table-tsv", type=Path, required=True)
    parser.add_argument("--decision-trace-tsv", type=Path, required=True)
    parser.add_argument("--scope-manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-inventory-tsv", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    final_table_tsv = resolve_existing_path(args.final_table_tsv, "final-table-tsv")
    decision_trace_tsv = resolve_existing_path(args.decision_trace_tsv, "decision-trace-tsv")
    scope_manifest_tsv = resolve_existing_path(args.scope_manifest_tsv, "scope-manifest-tsv")
    source_inventory_tsv = (
        resolve_existing_path(args.source_inventory_tsv, "source-inventory-tsv")
        if args.source_inventory_tsv is not None
        else None
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    final_rows = read_tsv(final_table_tsv)
    source_index = build_source_index(source_inventory_tsv)
    prompt_rows = build_prompt_placeholders(final_rows, source_index)

    candidate_path = out_dir / CANDIDATE_TSV_NAME
    audit_path = out_dir / AUDIT_TSV_NAME
    prompt_path = out_dir / PROMPT_PLACEHOLDER_TSV_NAME
    input_manifest_path = out_dir / INPUT_MANIFEST_TSV_NAME
    raw_responses_dir = out_dir / RAW_RESPONSES_DIR_NAME
    run_context_path = out_dir / RUN_CONTEXT_NAME
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    backend = _clean(os.environ.get("STAGE5_LLM_BACKEND"))
    message = BACKEND_UNSUPPORTED_MESSAGE if backend else NO_BACKEND_MESSAGE
    status = "backend_configured_but_live_calls_disabled" if backend else "no_llm_backend_configured"

    # No-backend diagnostic mode must not emit value candidates or audit findings.
    # Exact input/source metadata is recorded in RUN_CONTEXT.md and the input manifest.
    write_tsv(candidate_path, CANDIDATE_COLUMNS, [])
    write_tsv(audit_path, AUDIT_COLUMNS, [])
    write_tsv(prompt_path, PROMPT_PLACEHOLDER_COLUMNS, prompt_rows)
    write_tsv(
        input_manifest_path,
        INPUT_MANIFEST_COLUMNS,
        [
            {"input_role": "final_table_tsv", "path": final_table_tsv, "exists": "yes", "metadata_json": "{}"},
            {"input_role": "decision_trace_tsv", "path": decision_trace_tsv, "exists": "yes", "metadata_json": "{}"},
            {"input_role": "scope_manifest_tsv", "path": scope_manifest_tsv, "exists": "yes", "metadata_json": "{}"},
            {
                "input_role": "source_inventory_tsv",
                "path": source_inventory_tsv or "",
                "exists": "yes" if source_inventory_tsv else "not_provided",
                "metadata_json": source_paths_payload(source_index),
            },
        ],
    )

    outputs = {
        "candidate_tsv": candidate_path,
        "evidence_sidecar_tsv": audit_path,
        "prompt_audit_tsv": prompt_path,
        "input_manifest_tsv": input_manifest_path,
        "raw_responses_dir": raw_responses_dir,
        "run_context_md": run_context_path,
    }
    run_context_path.write_text(
        render_run_context(
            final_table_tsv=final_table_tsv,
            decision_trace_tsv=decision_trace_tsv,
            scope_manifest_tsv=scope_manifest_tsv,
            source_inventory_tsv=source_inventory_tsv,
            out_dir=out_dir,
            outputs=outputs,
            source_index=source_index,
            final_row_count=len(final_rows),
            prompt_row_count=len(prompt_rows),
            message=message,
        ),
        encoding="utf-8",
    )

    return {
        "status": status,
        "message": message,
        "out_dir": str(out_dir),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "final_input_rows": len(final_rows),
        "prompt_placeholder_rows": len(prompt_rows),
        "candidate_rows": 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(result["message"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
