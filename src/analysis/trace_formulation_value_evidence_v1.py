#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.active_data_source import resolve_run_context
from src.utils.paths import PROJECT_ROOT


SCRIPT_ROLE = "diagnostic_learning_only"
DEFAULT_PAPER_KEY = "QLYKLPKT"
DEFAULT_FIELDS = [
    "encapsulation_efficiency_percent",
    "loading_content_percent",
    "dl_percent",
    "drug_to_polymer_ratio_raw",
    "polymer_to_drug_ratio_raw",
    "drug_concentration_value",
    "surfactant_name",
    "surfactant_concentration_text",
    "organic_solvent",
    "drug_name",
]
SHARED_CONTEXT_FIELDS = [
    "emul_method",
    "la_ga_ratio",
    "la_ga_ratio_raw",
    "la_ga_ratio_normalized",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "surfactant_name",
    "surfactant_concentration_text",
    "organic_solvent",
    "drug_name",
    "drug_feed_amount_text",
]


def norm(value: Any) -> str:
    return str(value or "").strip()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path, limit: int | None = None) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def existing_path(path: Path | None) -> str:
    if path is None:
        return "not_resolved"
    return "exists" if path.exists() else "missing"


def pointer_files_from_context(run_context: dict[str, Any]) -> dict[str, str]:
    payload = run_context.get("pointer_payload") or {}
    files = payload.get("authoritative_terminal_files") or {}
    if not isinstance(files, dict):
        return {}
    return {str(key): str(value) for key, value in files.items()}


def resolve_pointer_path(pointer_files: dict[str, str], key: str) -> Path | None:
    value = norm(pointer_files.get(key))
    if not value:
        return None
    return repo_path(value)


def count_rows_for_key(path: Path | None, paper_key: str) -> tuple[int, list[dict[str, str]]]:
    if path is None or not path.exists():
        return 0, []
    rows = read_tsv(path)
    key_fields = ["key", "zotero_key", "paper_key", "document_key"]
    matched = [
        row
        for row in rows
        if any(norm(row.get(field)) == paper_key for field in key_fields)
    ]
    return len(matched), matched


def find_field_columns(rows: list[dict[str, str]], base_field: str) -> list[str]:
    if not rows:
        return []
    fieldnames = list(rows[0].keys())
    prefixes = [
        f"{base_field}_",
        f"{base_field}_value",
        f"{base_field}_text",
        f"{base_field}_scope",
        f"{base_field}_evidence",
        base_field,
    ]
    found: list[str] = []
    for name in fieldnames:
        if name == base_field or any(name.startswith(prefix) for prefix in prefixes):
            found.append(name)
    return found


def summarize_field_presence(
    *,
    rows: list[dict[str, str]],
    fields: list[str],
    max_examples: int,
) -> list[str]:
    lines: list[str] = []
    for field in fields:
        columns = find_field_columns(rows, field)
        if not columns:
            lines.append(f"- {field}: no matching columns")
            continue
        value_columns = [
            col
            for col in columns
            if col == field or col.endswith("_value") or col.endswith("_value_text") or col.endswith("_text")
        ]
        non_empty = 0
        examples: list[str] = []
        for row in rows:
            values = [norm(row.get(col)) for col in value_columns if norm(row.get(col))]
            if not values:
                continue
            non_empty += 1
            if len(examples) < max_examples:
                row_id = (
                    norm(row.get("final_formulation_id"))
                    or norm(row.get("formulation_id"))
                    or norm(row.get("local_instance_id"))
                    or f"row_{non_empty}"
                )
                examples.append(f"{row_id}: {' | '.join(values[:3])}")
        lines.append(
            f"- {field}: columns={len(columns)}, rows_with_values={non_empty}, examples={examples}"
        )
    return lines


def rows_by_formulation_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {norm(row.get("formulation_id")): row for row in rows if norm(row.get("formulation_id"))}


def summarize_filtered_context_rows(
    *,
    stage2_rows: list[dict[str, str]],
    decision_rows: list[dict[str, str]],
    fields: list[str],
    max_examples: int,
) -> list[str]:
    stage2_by_id = rows_by_formulation_id(stage2_rows)
    lines: list[str] = []
    emitted = 0
    for decision in decision_rows:
        if norm(decision.get("decision")) == "kept":
            continue
        source_id = norm(decision.get("source_formulation_id"))
        source_row = stage2_by_id.get(source_id, {})
        if not source_row:
            continue
        values: list[str] = []
        for field in fields:
            value = (
                norm(source_row.get(f"{field}_value"))
                or norm(source_row.get(f"{field}_value_text"))
                or norm(source_row.get(field))
            )
            if value:
                values.append(f"{field}={value}")
        if not values:
            continue
        emitted += 1
        if emitted > max_examples:
            break
        label = norm(source_row.get("raw_formulation_label"))
        reason = norm(decision.get("decision_reason")) or norm(decision.get("collapse_reason"))
        tags = norm(source_row.get("instance_context_tags")) or norm(source_row.get("change_context_tags"))
        lines.append(
            f"- {source_id}: decision={norm(decision.get('decision'))}, "
            f"label={label!r}, tags={tags!r}, reason={reason!r}, values={values}"
        )
    if not lines:
        lines.append("- no filtered context rows with target/shared values found")
    return lines


def inspect_json_payload(path: Path | None, paper_key: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"status": existing_path(path)}
    payload = read_json(path)
    text = json.dumps(payload, ensure_ascii=True)
    return {
        "status": "exists",
        "bytes": path.stat().st_size,
        "mentions_paper_key": paper_key in text,
        "top_level_type": type(payload).__name__,
        "top_level_keys": list(payload.keys())[:20] if isinstance(payload, dict) else [],
    }


def inspect_jsonl_for_key(path: Path | None, paper_key: str, max_scan: int | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"status": existing_path(path)}
    matches = 0
    scanned = 0
    key_fields = ["key", "zotero_key", "paper_key", "document_key"]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            scanned += 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and any(norm(row.get(field)) == paper_key for field in key_fields):
                matches += 1
            elif paper_key in text:
                matches += 1
            if max_scan is not None and scanned >= max_scan:
                break
    return {"status": "exists", "scanned": scanned, "matches": matches}


def likely_stage1_table_paths(scope_rows: list[dict[str, str]], paper_key: str) -> list[Path]:
    paths: list[Path] = []
    for row in scope_rows:
        if not any(norm(row.get(field)) == paper_key for field in ["key", "zotero_key", "paper_key"]):
            continue
        for field, value in row.items():
            field_l = field.lower()
            if "table" not in field_l and not field_l.endswith("_dir"):
                continue
            text = norm(value)
            if text:
                paths.append(repo_path(text))
    fallback = PROJECT_ROOT / "data" / "cleaned" / "goren_2025" / "tables" / paper_key
    paths.append(fallback.resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def print_section(title: str) -> None:
    print()
    print(f"## {title}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trace table/value evidence surfaces for one paper. "
            "This is a read-only diagnostic learning script."
        )
    )
    parser.add_argument("--paper-key", default=DEFAULT_PAPER_KEY)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--stage2-tsv", type=Path, default=None)
    parser.add_argument("--stage5-final-table-tsv", type=Path, default=None)
    parser.add_argument("--stage5-decision-trace-tsv", type=Path, default=None)
    parser.add_argument("--scope-manifest-tsv", type=Path, default=None)
    parser.add_argument("--stage3-relation-records-tsv", type=Path, default=None)
    parser.add_argument("--resolved-relation-fields-tsv", type=Path, default=None)
    parser.add_argument("--layer3-compare-cells-tsv", type=Path, default=None)
    parser.add_argument("--normalized-table-payload-json", type=Path, default=None)
    parser.add_argument("--evidence-block-json", type=Path, default=None)
    parser.add_argument("--semantic-objects-jsonl", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-jsonl-scan", type=int, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    paper_key = norm(args.paper_key)
    fields = args.field or DEFAULT_FIELDS

    run_context = resolve_run_context(explicit_run_dir=args.run_dir)
    pointer_files = pointer_files_from_context(run_context)

    stage2_tsv = args.stage2_tsv or resolve_pointer_path(pointer_files, "stage2_compatibility_tsv")
    stage5_tsv = args.stage5_final_table_tsv or (
        resolve_pointer_path(pointer_files, "stage5_final_table_tsv")
        or resolve_pointer_path(pointer_files, "stage5_final_table")
    )
    decision_trace_tsv = args.stage5_decision_trace_tsv or (
        resolve_pointer_path(pointer_files, "stage5_decision_trace_tsv")
        or resolve_pointer_path(pointer_files, "stage5_decision_trace")
    )
    scope_tsv = args.scope_manifest_tsv or resolve_pointer_path(pointer_files, "scope_manifest_tsv")
    relation_tsv = args.stage3_relation_records_tsv or resolve_pointer_path(pointer_files, "stage3_relation_records_tsv")
    resolved_fields_tsv = args.resolved_relation_fields_tsv or resolve_pointer_path(pointer_files, "resolved_relation_fields_tsv")
    layer3_cells_tsv = args.layer3_compare_cells_tsv or (
        resolve_pointer_path(pointer_files, "layer3_compare_cells_tsv")
        or resolve_pointer_path(pointer_files, "layer3_compare_cells")
    )
    semantic_jsonl = args.semantic_objects_jsonl or resolve_pointer_path(pointer_files, "stage2_semantic_objects_jsonl")

    payload_root = resolve_pointer_path(pointer_files, "stage2_s2_2_normalized_table_payloads_root")
    evidence_root = resolve_pointer_path(pointer_files, "stage2_s2_2_evidence_blocks_root")
    payload_json = args.normalized_table_payload_json or (
        payload_root / paper_key / "normalized_table_payloads_v1.json" if payload_root else None
    )
    evidence_json = args.evidence_block_json or (
        evidence_root / paper_key / "evidence_blocks_v1.json" if evidence_root else None
    )

    print("# Formulation Value Evidence Trace")
    print()
    print(f"- script_role: `{SCRIPT_ROLE}`")
    print(f"- paper_key: `{paper_key}`")
    print(f"- source_resolution: `{run_context['resolution_source']}`")
    print(f"- source_run_id: `{run_context['run_id']}`")
    print(f"- source_run_dir: `{run_context['run_dir']}`")
    print("- benchmark_valid: `no`; this script is read-only and diagnostic/learning-only.")

    print_section("Resolved Artifacts")
    artifacts = {
        "scope_manifest_tsv": scope_tsv,
        "stage2_compatibility_tsv": stage2_tsv,
        "stage2_semantic_objects_jsonl": semantic_jsonl,
        "stage2_normalized_table_payload_json": payload_json,
        "stage2_evidence_block_json": evidence_json,
        "stage3_relation_records_tsv": relation_tsv,
        "resolved_relation_fields_tsv": resolved_fields_tsv,
        "stage5_final_table_tsv": stage5_tsv,
        "stage5_decision_trace_tsv": decision_trace_tsv,
        "layer3_compare_cells_tsv": layer3_cells_tsv,
    }
    for name, path in artifacts.items():
        print(f"- {name}: {existing_path(path)} `{path or ''}`")

    scope_rows: list[dict[str, str]] = []
    if scope_tsv and scope_tsv.exists():
        scope_count, scope_rows_for_key = count_rows_for_key(scope_tsv, paper_key)
        scope_rows = read_tsv(scope_tsv)
        print_section("Scope Manifest")
        print(f"- rows_for_paper: `{scope_count}`")
        if scope_rows_for_key:
            row = scope_rows_for_key[0]
            print(f"- title: `{norm(row.get('title'))}`")
            print(f"- doi: `{norm(row.get('doi'))}`")
            print(f"- text_path: `{norm(row.get('text_path'))}`")

    print_section("Stage1 Table Candidates")
    stage1_paths = likely_stage1_table_paths(scope_rows, paper_key) if scope_rows else []
    if not stage1_paths:
        print("- no scope-derived table paths available")
    for path in stage1_paths:
        if path.is_dir():
            csv_count = len(list(path.glob("*.csv")))
            print(f"- exists dir `{path}` csv_files={csv_count}")
        else:
            print(f"- {existing_path(path)} `{path}`")

    print_section("Stage2 Table And Evidence Surfaces")
    print(f"- normalized_table_payload: `{inspect_json_payload(payload_json, paper_key)}`")
    print(f"- evidence_block: `{inspect_json_payload(evidence_json, paper_key)}`")
    print(f"- semantic_objects: `{inspect_jsonl_for_key(semantic_jsonl, paper_key, args.max_jsonl_scan)}`")

    print_section("Row Counts By Layer")
    stage2_count, stage2_rows = count_rows_for_key(stage2_tsv, paper_key)
    relation_count, relation_rows = count_rows_for_key(relation_tsv, paper_key)
    resolved_count, resolved_rows = count_rows_for_key(resolved_fields_tsv, paper_key)
    stage5_count, stage5_rows = count_rows_for_key(stage5_tsv, paper_key)
    decision_count, decision_rows = count_rows_for_key(decision_trace_tsv, paper_key)
    layer3_count, layer3_rows = count_rows_for_key(layer3_cells_tsv, paper_key)
    print(f"- Stage2 completed rows: `{stage2_count}`")
    print(f"- Stage3 relation records: `{relation_count}`")
    print(f"- Stage3 resolved relation rows: `{resolved_count}`")
    print(f"- Stage5 final rows: `{stage5_count}`")
    print(f"- Stage5 decision trace rows: `{decision_count}`")
    print(f"- Layer3 compare/audit cells: `{layer3_count}`")

    print_section("Target Field Presence In Stage2 Rows")
    for line in summarize_field_presence(rows=stage2_rows, fields=fields, max_examples=args.max_examples):
        print(line)

    print_section("Target Field Presence In Stage5 Rows")
    for line in summarize_field_presence(rows=stage5_rows, fields=fields, max_examples=args.max_examples):
        print(line)

    print_section("Filtered Context Rows With Shared Values")
    for line in summarize_filtered_context_rows(
        stage2_rows=stage2_rows,
        decision_rows=decision_rows,
        fields=SHARED_CONTEXT_FIELDS,
        max_examples=args.max_examples,
    ):
        print(line)

    print_section("Learning Notes")
    print("- If Stage1 table files exist but normalized payload is missing, inspect Stage2 S2-2 table authority construction.")
    print("- If normalized payload exists but evidence block/prompt omits it, inspect evidence selection and prompt packing.")
    print("- If LLM semantic objects lack table scope, inspect prompt visibility and semantic contract burden.")
    print("- If Stage2 rows have values but Stage5 rows do not, inspect Stage3 relation fields and Stage5 carry-through.")
    print("- If Stage5 has a value but no evidence path, inspect Layer3/audit/evidence-binding surfaces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
