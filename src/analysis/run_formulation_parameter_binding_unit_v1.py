#!/usr/bin/env python3
from __future__ import annotations

"""
Run the deterministic formulation-parameter binding unit and a Step 2 before/after comparison.
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


RESOLVED_NAME = "formulation_parameter_binding_resolved_v1.tsv"
SUMMARY_NAME = "formulation_parameter_binding_summary_v1.md"
SUPPORTED_STATUSES = {"explicit_supported", "relation_carried_explicit"}
TARGET_FIELDS = [
    "polymer_mw_kDa",
    "la_ga_ratio",
    "polymer_amount",
    "drug_feed_amount",
    "surfactant_concentration",
    "phase_ratio",
    "drug_polymer_ratio",
]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_cmd(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic formulation-parameter binding and a Step 2 before/after comparison.")
    parser.add_argument("--step1-run-dir", required=True, type=Path)
    parser.add_argument("--paper-key", action="append", default=[])
    parser.add_argument("--run-dir", default="", help="Explicit results run directory.")
    parser.add_argument("--run-id", default="", help="Explicit child/run identifier.")
    parser.add_argument("--execution-cue", default="formulation_parameter_binding_unit")
    return parser.parse_args()


def supported_counts(evidence_rows: list[dict[str, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in evidence_rows:
        if normalize_text(row.get("support_status")) in SUPPORTED_STATUSES:
            counter[normalize_text(row.get("field_name"))] += 1
    return counter


def resolved_binding_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        if normalize_text(row.get("binding_status")) in {
            "resolved_relation_context",
            "resolved_shared_context",
            "resolved_article_native_match",
        }:
            counter[normalize_text(row.get("field_name"))] += 1
    return counter


def ambiguous_binding_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        if normalize_text(row.get("binding_status")) == "ambiguous_multiple_targets":
            counter[normalize_text(row.get("field_name"))] += 1
    return counter


def modeling_core_rows(table_rows: list[dict[str, str]]) -> int:
    required_fields = [
        "drug_name",
        "polymer_mw_kDa",
        "la_ga_ratio",
        "encapsulation_efficiency_percent",
    ]
    total = 0
    for row in table_rows:
        if all(normalize_text(row.get(f"{field}_support_status")) in SUPPORTED_STATUSES for field in required_fields):
            total += 1
    return total


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    step1_run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    commands: list[list[str]],
    paper_keys: list[str],
    resolved_total: int,
    before_counts: Counter[str],
    after_counts: Counter[str],
    modeling_core_before: int,
    modeling_core_after: int,
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
        "- `formulation_parameter_binding_validation_run`",
        f"- run_dir_kind: `{run_dir_kind}`",
        f"- run_selection_mode: `{run_selection_mode}`",
        f"- bucket_dir: `{bucket_dir}`",
        "",
        "## 3. Purpose",
        "",
        "- Resolve lawful formulation-level ownership for composition-side fields and measure how many Step 2 cells can upgrade without weakening the explicit-only contract.",
        "",
        "## 4. Starting inputs",
        "",
        f"- step1_run_dir: `{repo_rel(step1_run_dir)}`",
        f"- final_table_tsv: `{repo_rel(step1_run_dir / 'final_formulation_table_v1.tsv')}`",
        f"- decision_trace_tsv: `{repo_rel(step1_run_dir / 'final_output_decision_trace_v1.tsv')}`",
        f"- relation_records_tsv: `{repo_rel(step1_run_dir / 'formulation_relation_v1' / 'formulation_relation_records_v1.tsv')}`",
        f"- resolved_relation_fields_tsv: `{repo_rel(step1_run_dir / 'formulation_relation_v1' / 'resolved_relation_fields_v1.tsv')}`",
        "",
        "## 5. Papers tested",
        "",
        f"- `{', '.join(paper_keys) if paper_keys else 'all papers in the source Step 1 run'}`",
        "",
        "## 6. Commands",
        "",
        "```powershell",
        *[" ".join(command) for command in commands],
        "```",
        "",
        "## 7. Validation summary",
        "",
        f"- resolved formulation-parameter bindings total: `{resolved_total}`",
        f"- polymer_mw support before -> after: `{before_counts.get('polymer_mw_kDa', 0)} -> {after_counts.get('polymer_mw_kDa', 0)}`",
        f"- la_ga_ratio support before -> after: `{before_counts.get('la_ga_ratio', 0)} -> {after_counts.get('la_ga_ratio', 0)}`",
        f"- polymer_amount support before -> after: `{before_counts.get('polymer_amount', 0)} -> {after_counts.get('polymer_amount', 0)}`",
        f"- drug_feed_amount support before -> after: `{before_counts.get('drug_feed_amount', 0)} -> {after_counts.get('drug_feed_amount', 0)}`",
        f"- surfactant_concentration support before -> after: `{before_counts.get('surfactant_concentration', 0)} -> {after_counts.get('surfactant_concentration', 0)}`",
        f"- phase_ratio support before -> after: `{before_counts.get('phase_ratio', 0)} -> {after_counts.get('phase_ratio', 0)}`",
        f"- drug_polymer_ratio support before -> after: `{before_counts.get('drug_polymer_ratio', 0)} -> {after_counts.get('drug_polymer_ratio', 0)}`",
        f"- modeling-core rows before -> after: `{modeling_core_before} -> {modeling_core_after}`",
        "",
        "## 8. Final artifacts",
        "",
        f"- `{repo_rel(run_dir / RESOLVED_NAME)}`",
        f"- `{repo_rel(run_dir / SUMMARY_NAME)}`",
        f"- `{repo_rel(run_dir / 'step2_before_parameter_binding' / 'step2_value_backfill_table_v1.tsv')}`",
        f"- `{repo_rel(run_dir / 'step2_after_parameter_binding' / 'step2_value_backfill_table_v1.tsv')}`",
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
    scope_manifest = step1_run_dir / "eligible_scope_manifest.tsv"
    table_row_binding = step1_run_dir / "table_row_binding_resolved_v1.tsv"

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

    commands: list[list[str]] = []
    command_log: list[dict[str, str]] = []

    binding_command = [
        sys.executable,
        str(REPO_ROOT / "src" / "analysis" / "build_formulation_parameter_binding_unit_v1.py"),
        "--final-table-tsv",
        str(final_table),
        "--decision-trace-tsv",
        str(decision_trace),
        "--relation-records-tsv",
        str(relation_records),
        "--resolved-relation-fields-tsv",
        str(resolved_fields),
        "--out-dir",
        str(run_dir),
    ]
    if scope_manifest.exists():
        binding_command.extend(["--scope-manifest-tsv", str(scope_manifest)])
    for paper_key in args.paper_key:
        binding_command.extend(["--paper-key", paper_key])
    commands.append(binding_command)
    completed = run_cmd(binding_command, cwd=REPO_ROOT)
    command_log.append({"command": " ".join(binding_command), "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(), "status": "ok"})

    before_dir = run_dir / "step2_before_parameter_binding"
    after_dir = run_dir / "step2_after_parameter_binding"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    common_step2_args = [
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
    ]
    if scope_manifest.exists():
        common_step2_args.extend(["--scope-manifest-tsv", str(scope_manifest)])
    if table_row_binding.exists():
        common_step2_args.extend(["--table-row-binding-tsv", str(table_row_binding)])
    for paper_key in args.paper_key:
        common_step2_args.extend(["--paper-key", paper_key])

    before_command = common_step2_args + ["--out-dir", str(before_dir)]
    commands.append(before_command)
    completed = run_cmd(before_command, cwd=REPO_ROOT)
    command_log.append({"command": " ".join(before_command), "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(), "status": "ok"})

    after_command = common_step2_args + [
        "--out-dir",
        str(after_dir),
        "--parameter-binding-tsv",
        str(run_dir / RESOLVED_NAME),
    ]
    commands.append(after_command)
    completed = run_cmd(after_command, cwd=REPO_ROOT)
    command_log.append({"command": " ".join(after_command), "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(), "status": "ok"})

    binding_rows = read_tsv_rows(run_dir / RESOLVED_NAME)
    before_evidence = read_tsv_rows(before_dir / "step2_value_backfill_evidence_v1.tsv")
    after_evidence = read_tsv_rows(after_dir / "step2_value_backfill_evidence_v1.tsv")
    before_table = read_tsv_rows(before_dir / "step2_value_backfill_table_v1.tsv")
    after_table = read_tsv_rows(after_dir / "step2_value_backfill_table_v1.tsv")

    before_counts = supported_counts(before_evidence)
    after_counts = supported_counts(after_evidence)
    binding_counts = resolved_binding_counts(binding_rows)
    ambiguous_counts = ambiguous_binding_counts(binding_rows)
    paper_keys = sorted({normalize_text(row.get("paper_key")) for row in binding_rows if normalize_text(row.get("paper_key"))})
    resolved_total = sum(binding_counts.values())
    modeling_core_before = modeling_core_rows(before_table)
    modeling_core_after = modeling_core_rows(after_table)

    before_ids = [row["final_formulation_id"] for row in before_table]
    after_ids = [row["final_formulation_id"] for row in after_table]
    if before_ids != after_ids:
        raise ValueError("Step 2 final_formulation_id order changed after parameter binding.")

    (run_dir / "RUN_CONTEXT.md").write_text(
        build_run_context(
            run_id=run_id,
            run_dir=run_dir,
            step1_run_dir=step1_run_dir,
            run_dir_kind=run_dir_kind,
            run_selection_mode=run_selection_mode,
            bucket_dir=bucket_dir,
            commands=commands,
            paper_keys=paper_keys,
            resolved_total=resolved_total,
            before_counts=before_counts,
            after_counts=after_counts,
            modeling_core_before=modeling_core_before,
            modeling_core_after=modeling_core_after,
        ),
        encoding="utf-8",
    )
    (run_dir / "command_execution_log_v1.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_path": str(run_dir),
                "papers_tested": paper_keys,
                "resolved_parameter_bindings_total": resolved_total,
                "resolved_bindings_by_field": dict(binding_counts),
                "ambiguous_bindings_by_field": dict(ambiguous_counts),
                "before_supported_counts": {field: before_counts.get(field, 0) for field in TARGET_FIELDS},
                "after_supported_counts": {field: after_counts.get(field, 0) for field in TARGET_FIELDS},
                "modeling_core_rows_before_after": [modeling_core_before, modeling_core_after],
                "row_count_before_after": [len(before_table), len(after_table)],
                "id_preservation_pass": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
