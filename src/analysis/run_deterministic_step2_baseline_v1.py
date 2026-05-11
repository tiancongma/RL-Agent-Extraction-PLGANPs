#!/usr/bin/env python3
from __future__ import annotations

"""
Run the deterministic Step 2 baseline helper from an explicit Step 1 run directory.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import DATA_RESULTS_DIR
from src.utils.run_id import resolve_results_write_target


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def run_cmd(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Step 2 baseline from an explicit Step 1 run directory.")
    parser.add_argument("--step1-run-dir", required=True, type=Path)
    parser.add_argument("--paper-key", action="append", default=[])
    parser.add_argument("--table-row-binding-tsv", default="", type=Path)
    parser.add_argument("--parameter-binding-tsv", default="", type=Path)
    parser.add_argument("--run-dir", default="", help="Explicit results run directory.")
    parser.add_argument("--run-id", default="", help="Explicit child/run identifier.")
    parser.add_argument("--execution-cue", default="deterministic_step2_baseline")
    return parser.parse_args()


def build_run_context(
    *,
    run_id: str,
    step1_run_dir: Path,
    run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    command: list[str],
    papers_processed: list[str],
    output_row_count: int,
    row_count_match: bool,
    id_preservation_pass: bool,
    top_fields: list[str],
) -> str:
    lines = [
        "# RUN_CONTEXT",
        "",
        "## 1. Run ID",
        "",
        f"- `{run_id}`",
        "",
        "## 2. Run type",
        "",
        "- `component_regression_run`",
        "- `deterministic_step2_baseline_run`",
        f"- run_dir_kind: `{run_dir_kind}`",
        f"- run_selection_mode: `{run_selection_mode}`",
        f"- bucket_dir: `{bucket_dir}`",
        "",
        "## 3. Purpose",
        "",
        "- Attach explicit-only values onto frozen Step 1 formulation identities without changing membership.",
        "",
        "## 4. Starting inputs",
        "",
        f"- step1_run_dir: `{repo_rel(step1_run_dir)}`",
        f"- source_final_table: `{repo_rel(step1_run_dir / 'final_formulation_table_v1.tsv')}`",
        f"- source_relation_records: `{repo_rel(step1_run_dir / 'formulation_relation_v1' / 'formulation_relation_records_v1.tsv')}`",
        f"- source_resolved_relation_fields: `{repo_rel(step1_run_dir / 'formulation_relation_v1' / 'resolved_relation_fields_v1.tsv')}`",
        "",
        "## 5. Command",
        "",
        "```powershell",
        " ".join(command),
        "```",
        "",
        "## 6. Validation",
        "",
        f"- papers_processed: `{', '.join(papers_processed)}`",
        f"- output_row_count: `{output_row_count}`",
        f"- row_count_match_step1: `{'yes' if row_count_match else 'no'}`",
        f"- id_preservation_pass: `{'yes' if id_preservation_pass else 'no'}`",
        f"- top_fields_with_explicit_supported_fills: `{', '.join(top_fields) if top_fields else 'none'}`",
        "",
        "## 7. Final artifacts",
        "",
        f"- `{repo_rel(run_dir / 'step2_value_backfill_table_v1.tsv')}`",
        f"- `{repo_rel(run_dir / 'step2_value_backfill_evidence_v1.tsv')}`",
        f"- `{repo_rel(run_dir / 'step2_value_backfill_summary_v1.md')}`",
        f"- `{repo_rel(run_dir / 'RUN_CONTEXT.md')}`",
        f"- `{repo_rel(run_dir / 'command_execution_log_v1.json')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    step1_run_dir = args.step1_run_dir.resolve()
    final_table = step1_run_dir / "final_formulation_table_v1.tsv"
    decision_trace = step1_run_dir / "final_output_decision_trace_v1.tsv"
    relation_records = step1_run_dir / "formulation_relation_v1" / "formulation_relation_records_v1.tsv"
    resolved_fields = step1_run_dir / "formulation_relation_v1" / "resolved_relation_fields_v1.tsv"
    baseline_assessment = step1_run_dir / "analysis" / "baseline_ready_identity_assessment_v1.tsv"
    scope_manifest = step1_run_dir / "eligible_scope_manifest.tsv"

    for path in [final_table, relation_records, resolved_fields]:
        if not path.exists():
            raise FileNotFoundError(f"Required Step 1 artifact not found: {path}")

    target = resolve_results_write_target(
        results_root=DATA_RESULTS_DIR,
        default_child_cue=args.execution_cue,
        explicit_run_dir=args.run_dir,
        explicit_legacy_run_id=args.run_id,
    )
    run_id = target["run_basename"]
    run_dir = Path(target["run_dir"])
    run_dir_kind = target["path_kind"]
    run_selection_mode = target["selection_mode"]
    bucket_dir = Path(target["bucket_dir"])
    if run_dir_kind == "v2_child_execution":
        bucket_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)

    command = [
        sys.executable,
        str(REPO_ROOT / "src" / "stage5_benchmark" / "build_deterministic_step2_value_backfill_v1.py"),
        "--final-table-tsv",
        str(final_table),
        "--relation-records-tsv",
        str(relation_records),
        "--resolved-relation-fields-tsv",
        str(resolved_fields),
        "--decision-trace-tsv",
        str(decision_trace),
        "--source-run-dir",
        str(step1_run_dir),
        "--out-dir",
        str(run_dir),
    ]
    if scope_manifest.exists():
        command.extend(["--scope-manifest-tsv", str(scope_manifest)])
    if baseline_assessment.exists():
        command.extend(["--baseline-assessment-tsv", str(baseline_assessment)])
    if normalize_text(args.table_row_binding_tsv):
        command.extend(["--table-row-binding-tsv", str(args.table_row_binding_tsv.resolve())])
    if normalize_text(args.parameter_binding_tsv):
        command.extend(["--parameter-binding-tsv", str(args.parameter_binding_tsv.resolve())])
    for paper_key in args.paper_key:
        command.extend(["--paper-key", paper_key])

    completed = run_cmd(command, cwd=REPO_ROOT)
    command_log = [{"command": " ".join(command), "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(), "status": "ok"}]

    source_rows = read_tsv_rows(final_table)
    if args.paper_key:
        selected = {normalize_text(key) for key in args.paper_key if normalize_text(key)}
        source_rows = [row for row in source_rows if normalize_text(row.get("key")) in selected]
    output_rows = read_tsv_rows(run_dir / "step2_value_backfill_table_v1.tsv")
    evidence_rows = read_tsv_rows(run_dir / "step2_value_backfill_evidence_v1.tsv")

    source_ids = [normalize_text(row.get("final_formulation_id")) for row in source_rows]
    output_ids = [normalize_text(row.get("final_formulation_id")) for row in output_rows]
    row_count_match = len(source_rows) == len(output_rows)
    id_preservation_pass = source_ids == output_ids and len(set(output_ids)) == len(output_ids)
    papers_processed = sorted({normalize_text(row.get("paper_key")) for row in output_rows if normalize_text(row.get("paper_key"))})

    fills_by_field: Counter[str] = Counter()
    blanks_by_field: Counter[str] = Counter()
    for row in evidence_rows:
        field_name = normalize_text(row.get("field_name"))
        status = normalize_text(row.get("support_status"))
        if status in {"explicit_supported", "relation_carried_explicit"}:
            fills_by_field[field_name] += 1
        if status == "blank_not_reported":
            blanks_by_field[field_name] += 1
    top_fields = [name for name, _ in fills_by_field.most_common(5)]

    (run_dir / "RUN_CONTEXT.md").write_text(
        build_run_context(
            run_id=run_id,
            step1_run_dir=step1_run_dir,
            run_dir=run_dir,
            run_dir_kind=run_dir_kind,
            run_selection_mode=run_selection_mode,
            bucket_dir=bucket_dir,
            command=command,
            papers_processed=papers_processed,
            output_row_count=len(output_rows),
            row_count_match=row_count_match,
            id_preservation_pass=id_preservation_pass,
            top_fields=top_fields,
        ),
        encoding="utf-8",
    )
    (run_dir / "command_execution_log_v1.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "step2_run_path": str(run_dir),
                "source_step1_run_path": str(step1_run_dir),
                "papers_processed": len(papers_processed),
                "output_row_count": len(output_rows),
                "id_preservation_pass": id_preservation_pass,
                "top_fields_with_explicit_supported_fills": top_fields,
                "explicit_supported_by_field": dict(fills_by_field),
                "blank_by_field": dict(blanks_by_field),
            },
            indent=2,
        )
    )
    return 0 if row_count_match and id_preservation_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
