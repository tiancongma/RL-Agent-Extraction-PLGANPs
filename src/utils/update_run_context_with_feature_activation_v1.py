#!/usr/bin/env python3
from __future__ import annotations

"""
Inject or refresh the Feature Unit Activation section inside a run's RUN_CONTEXT.md.

Workflow:
1. Build or refresh the run-local feature activation report.
2. Derive the run activation gate from the report rows.
3. Replace the existing Feature Unit Activation section if present, otherwise append it.
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from src.utils.build_feature_activation_report_v1 import (
        build_report_rows,
        compute_activation_gate,
        load_matrix,
        load_registry,
        write_report_tsv,
    )
    from src.utils.paths import PROJECT_DIR
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.build_feature_activation_report_v1 import (
        build_report_rows,
        compute_activation_gate,
        load_matrix,
        load_registry,
        write_report_tsv,
    )
    from src.utils.paths import PROJECT_DIR


SECTION_HEADING = "## Feature Unit Activation"
BOUNDARY_SECTION_HEADING = "## Boundary Governance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update RUN_CONTEXT.md with feature activation metadata.")
    parser.add_argument(
        "--registry",
        default=str(PROJECT_DIR / "feature_units" / "feature_unit_registry.json"),
        help="Path to the feature unit registry JSON.",
    )
    parser.add_argument(
        "--matrix",
        default=str(PROJECT_DIR / "feature_units" / "feature_intervention_matrix.tsv"),
        help="Path to the feature intervention matrix TSV.",
    )
    parser.add_argument("--run-dir", required=True, help="Target run directory.")
    parser.add_argument(
        "--report-tsv",
        default="",
        help="Optional explicit feature activation report path. Default: <run-dir>/analysis/feature_activation_report_v1.tsv",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR.parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def render_activation_section(
    *,
    report_path: Path,
    rows: list[dict[str, str]],
    gate: dict[str, object],
) -> str:
    required_units = list(gate["required_feature_units"])
    missing_units = list(gate["missing_required_feature_units"])
    lines = [
        SECTION_HEADING,
        "",
        f"- `feature_activation_report_path`: `{repo_rel(report_path)}`",
        f"- `required_feature_units`: `{json.dumps(required_units)}`",
        f"- `missing_required_feature_units`: `{json.dumps(missing_units)}`",
        f"- `run_activation_gate`: `{gate['run_activation_gate']}`",
        "",
        "| feature_id | expected_for_run | observed_activation | activation_status | activation_state | evidence_path |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        evidence_path = row["evidence_path"] or ""
        lines.append(
            f"| `{row['feature_id']}` | `{row['expected_for_run']}` | `{row['observed_activation']}` | `{row['activation_status']}` | `{row.get('activation_state', '')}` | `{evidence_path}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def replace_or_append_section(existing_text: str, heading: str, new_section: str) -> str:
    text = existing_text.rstrip() + "\n"
    heading_token = f"\n{heading}\n"
    if text.startswith(f"{heading}\n"):
        start = 0
    else:
        start = text.find(heading_token)
        if start != -1:
            start += 1
    if start == -1:
        return text.rstrip() + "\n\n" + new_section

    remainder = text[start + len(heading):]
    next_heading_index = remainder.find("\n## ")
    if next_heading_index == -1:
        return text[:start].rstrip() + "\n\n" + new_section
    end = start + len(heading) + next_heading_index + 1
    return text[:start].rstrip() + "\n\n" + new_section + "\n" + text[end:].lstrip("\n")


def extract_run_context_value(run_context_text: str, key: str) -> str:
    pattern = rf"{re.escape(key)}:\s*`([^`]*)`"
    match = re.search(pattern, run_context_text)
    return match.group(1).strip() if match else ""


def detect_boundary_contract(run_dir: Path, run_context_text: str) -> dict[str, str]:
    semantic_dir = run_dir / "semantic_stage2_objects"
    compat_dir = run_dir / "semantic_to_widerow_adapter"
    has_compare = (run_dir / "final_table_vs_gt_counts.tsv").exists()
    has_final_table = (run_dir / "final_formulation_table_v1.tsv").exists()
    has_stage3 = any(run_dir.rglob("formulation_relation_records_v1.tsv"))
    has_completed_stage2 = (compat_dir / "weak_labels__v7pilot_r3_fixparse.tsv").exists()
    has_raw_responses = any((semantic_dir / "raw_responses").glob("*__stage2_v2_raw_response.json")) if (semantic_dir / "raw_responses").exists() else False
    has_semantic_objects = (semantic_dir / "semantic_stage2_v2_objects.jsonl").exists()

    if has_compare:
        boundary_class = "benchmark_terminal_boundary"
        authoritative_for_downstream = "yes"
        lawful_resume_boundary = "no"
        resume_entrypoint = "not_applicable"
        schema_contract = "stage5_final_table_vs_gt_comparison_v1"
        evidence_path = repo_rel(run_dir / "final_table_vs_gt_counts.tsv")
    elif has_final_table:
        boundary_class = "mainline_resume_boundary"
        authoritative_for_downstream = "yes"
        lawful_resume_boundary = "yes"
        resume_entrypoint = "src/stage5_benchmark/compare_final_table_to_gt_v1.py"
        schema_contract = "stage5_final_formulation_table_v1"
        evidence_path = repo_rel(run_dir / "final_formulation_table_v1.tsv")
    elif has_stage3:
        boundary_class = "mainline_resume_boundary"
        authoritative_for_downstream = "yes"
        lawful_resume_boundary = "yes"
        resume_entrypoint = "src/stage5_benchmark/build_minimal_final_output_v1.py"
        schema_contract = "stage3_relation_artifacts_v1"
        evidence_path = repo_rel(next(run_dir.rglob("formulation_relation_records_v1.tsv")))
    elif has_completed_stage2:
        boundary_class = "mainline_resume_boundary"
        authoritative_for_downstream = "yes"
        lawful_resume_boundary = "yes"
        resume_entrypoint = "src/stage3_relation/build_formulation_relation_artifacts_v1.py"
        schema_contract = "completed_stage2_compatibility_projection_v1"
        evidence_path = repo_rel(compat_dir / "weak_labels__v7pilot_r3_fixparse.tsv")
    elif has_raw_responses:
        boundary_class = "diagnostic_boundary"
        authoritative_for_downstream = "no"
        lawful_resume_boundary = "no"
        resume_entrypoint = "not_applicable"
        schema_contract = "stage2_raw_response_archive_v1"
        evidence_path = repo_rel(semantic_dir / "raw_responses")
    elif has_semantic_objects:
        boundary_class = "internal_intermediate"
        authoritative_for_downstream = "no"
        lawful_resume_boundary = "no"
        resume_entrypoint = "not_applicable"
        schema_contract = "stage2_semantic_intermediate_v2"
        evidence_path = repo_rel(semantic_dir / "semantic_stage2_v2_objects.jsonl")
    else:
        boundary_class = "diagnostic_boundary"
        authoritative_for_downstream = "unknown"
        lawful_resume_boundary = "unknown"
        resume_entrypoint = "unknown"
        schema_contract = "unknown"
        evidence_path = ""

    source_run_dir = extract_run_context_value(run_context_text, "source_run_dir") or "not_applicable"
    source_run_id = extract_run_context_value(run_context_text, "source_run_id") or "not_applicable"
    upstream_authority_source = (
        extract_run_context_value(run_context_text, "source_run_dir")
        or extract_run_context_value(run_context_text, "manifest_tsv")
        or extract_run_context_value(run_context_text, "scope_manifest_tsv")
        or extract_run_context_value(run_context_text, "weak_labels_tsv")
        or extract_run_context_value(run_context_text, "candidate_input_tsv")
        or extract_run_context_value(run_context_text, "final_table_tsv")
        or "unknown"
    )
    replay_mode = extract_run_context_value(run_context_text, "source_mode") or extract_run_context_value(run_context_text, "replay_mode")
    if not replay_mode:
        if extract_run_context_value(run_context_text, "source_resolution"):
            replay_mode = "none"
        elif has_final_table or has_stage3:
            replay_mode = "deterministic_only"
        else:
            replay_mode = "none"

    source_file_keys = [
        "manifest_tsv",
        "scope_manifest_tsv",
        "weak_labels_tsv",
        "weak_labels_jsonl",
        "candidate_input_tsv",
        "relation_records_tsv",
        "resolved_relation_fields_tsv",
        "final_table_tsv",
        "gt_workbook_xlsx",
        "gt_xlsx",
        "legacy_raw_responses_dir",
    ]
    source_files = [value for key in source_file_keys if (value := extract_run_context_value(run_context_text, key))]
    benchmark_terminal = "yes" if boundary_class == "benchmark_terminal_boundary" else "no"

    return {
        "boundary_class": boundary_class,
        "authoritative_for_downstream": authoritative_for_downstream,
        "lawful_resume_boundary": lawful_resume_boundary,
        "resume_entrypoint": resume_entrypoint,
        "schema_contract": schema_contract,
        "upstream_authority_source": upstream_authority_source,
        "replay_mode": replay_mode,
        "source_run_dir": source_run_dir,
        "source_run_id": source_run_id,
        "source_files": json.dumps(source_files),
        "benchmark_terminal": benchmark_terminal,
        "boundary_evidence_path": evidence_path or "unknown",
    }


def render_boundary_section(boundary: dict[str, str]) -> str:
    lines = [
        BOUNDARY_SECTION_HEADING,
        "",
        f"- `boundary_class`: `{boundary['boundary_class']}`",
        f"- `authoritative_for_downstream`: `{boundary['authoritative_for_downstream']}`",
        f"- `lawful_resume_boundary`: `{boundary['lawful_resume_boundary']}`",
        f"- `resume_entrypoint`: `{boundary['resume_entrypoint']}`",
        f"- `schema_contract`: `{boundary['schema_contract']}`",
        f"- `upstream_authority_source`: `{boundary['upstream_authority_source']}`",
        f"- `replay_mode`: `{boundary['replay_mode']}`",
        f"- `source_run_dir`: `{boundary['source_run_dir']}`",
        f"- `source_run_id`: `{boundary['source_run_id']}`",
        f"- `source_files`: `{boundary['source_files']}`",
        f"- `benchmark_terminal`: `{boundary['benchmark_terminal']}`",
        f"- `boundary_evidence_path`: `{boundary['boundary_evidence_path']}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    run_context_path = run_dir / "RUN_CONTEXT.md"
    if not run_context_path.exists():
        raise FileNotFoundError(f"RUN_CONTEXT.md not found: {run_context_path}")

    registry = load_registry(Path(args.registry))
    matrix = load_matrix(Path(args.matrix))
    report_path = Path(args.report_tsv) if args.report_tsv else run_dir / "analysis" / "feature_activation_report_v1.tsv"
    existing_text = run_context_path.read_text(encoding="utf-8")
    boundary = detect_boundary_contract(run_dir, existing_text)
    boundary_section = render_boundary_section(boundary)
    updated_text = replace_or_append_section(existing_text, BOUNDARY_SECTION_HEADING, boundary_section)
    run_context_path.write_text(updated_text, encoding="utf-8")

    existing_text = run_context_path.read_text(encoding="utf-8")
    rows = build_report_rows(registry=registry, matrix=matrix, run_dir=run_dir)
    write_report_tsv(report_path, rows)
    gate = compute_activation_gate(rows)
    new_section = render_activation_section(report_path=report_path, rows=rows, gate=gate)
    updated_text = replace_or_append_section(existing_text, SECTION_HEADING, new_section)
    run_context_path.write_text(updated_text, encoding="utf-8")

    # Rebuild once more so features that depend on RUN_CONTEXT.md can observe the
    # generated section from the final saved file rather than the pre-update state.
    rows = build_report_rows(registry=registry, matrix=matrix, run_dir=run_dir)
    write_report_tsv(report_path, rows)
    gate = compute_activation_gate(rows)
    new_section = render_activation_section(report_path=report_path, rows=rows, gate=gate)
    final_text = run_context_path.read_text(encoding="utf-8")
    boundary = detect_boundary_contract(run_dir, final_text)
    boundary_section = render_boundary_section(boundary)
    updated_text = replace_or_append_section(final_text, BOUNDARY_SECTION_HEADING, boundary_section)
    updated_text = replace_or_append_section(updated_text, SECTION_HEADING, new_section)
    run_context_path.write_text(updated_text, encoding="utf-8")
    print(str(run_context_path))


if __name__ == "__main__":
    main()
