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
from pathlib import Path

from src.utils.build_feature_activation_report_v1 import (
    build_report_rows,
    compute_activation_gate,
    load_matrix,
    load_registry,
    write_report_tsv,
)
from src.utils.paths import PROJECT_DIR


SECTION_HEADING = "## Feature Unit Activation"


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
        "| feature_id | expected_for_run | observed_activation | activation_status | evidence_path |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        evidence_path = row["evidence_path"] or ""
        lines.append(
            f"| `{row['feature_id']}` | `{row['expected_for_run']}` | `{row['observed_activation']}` | `{row['activation_status']}` | `{evidence_path}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def replace_or_append_section(existing_text: str, new_section: str) -> str:
    text = existing_text.rstrip() + "\n"
    heading_token = f"\n{SECTION_HEADING}\n"
    if text.startswith(f"{SECTION_HEADING}\n"):
        start = 0
    else:
        start = text.find(heading_token)
        if start != -1:
            start += 1
    if start == -1:
        return text.rstrip() + "\n\n" + new_section

    remainder = text[start + len(SECTION_HEADING):]
    next_heading_index = remainder.find("\n## ")
    if next_heading_index == -1:
        return text[:start].rstrip() + "\n\n" + new_section
    end = start + len(SECTION_HEADING) + next_heading_index + 1
    return text[:start].rstrip() + "\n\n" + new_section + "\n" + text[end:].lstrip("\n")


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
    rows = build_report_rows(registry=registry, matrix=matrix, run_dir=run_dir)
    write_report_tsv(report_path, rows)
    gate = compute_activation_gate(rows)
    new_section = render_activation_section(report_path=report_path, rows=rows, gate=gate)
    updated_text = replace_or_append_section(existing_text, new_section)
    run_context_path.write_text(updated_text, encoding="utf-8")

    # Rebuild once more so features that depend on RUN_CONTEXT.md can observe the
    # generated section from the final saved file rather than the pre-update state.
    rows = build_report_rows(registry=registry, matrix=matrix, run_dir=run_dir)
    write_report_tsv(report_path, rows)
    gate = compute_activation_gate(rows)
    new_section = render_activation_section(report_path=report_path, rows=rows, gate=gate)
    updated_text = replace_or_append_section(run_context_path.read_text(encoding="utf-8"), new_section)
    run_context_path.write_text(updated_text, encoding="utf-8")
    print(str(run_context_path))


if __name__ == "__main__":
    main()
