#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


STATUS_TSV = "dev15_paper_status.tsv"
SUMMARY_MD = "dev15_go_no_go_summary.md"
REGRESSION_TSV = "regression_triggered_diagnostics.tsv"


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def by_key(rows: list[dict[str, str]], key_field: str = "paper_key") -> dict[str, dict[str, str]]:
    return {normalize_text(row.get(key_field)): row for row in rows}


def classify_status(delta_vs_gt: int, total_mismatch: int) -> tuple[str, str, str]:
    if delta_vs_gt == 0 and total_mismatch == 0:
        return "exact_match", "exact_match", "benchmark-facing behavior matches GT on this paper."
    if abs(delta_vs_gt) <= 1 and total_mismatch <= 1:
        return "minor_deviation", "minor_deviation", "Small GT-facing deviation only."
    if abs(delta_vs_gt) >= 2 or total_mismatch >= 2:
        return "major_regression", "major_regression", "Material benchmark-facing deviation."
    return "needs_targeted_review", "needs_targeted_review", "Ambiguous or mixed signal."


def final_decision(major_count: int, exact_count: int, total_count: int) -> str:
    if major_count == 0:
        return "GO"
    if major_count <= 2 and exact_count >= max(1, total_count - 4):
        return "GO_WITH_CAUTIONS"
    return "NO_GO"


def build_summary(
    *,
    run_dir: Path,
    paper_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    decision: str,
) -> str:
    exact_count = sum(1 for row in paper_rows if row["overall_status"] == "exact_match")
    minor_count = sum(1 for row in paper_rows if row["overall_status"] == "minor_deviation")
    major_count = sum(1 for row in paper_rows if row["overall_status"] == "major_regression")
    lines = [
        "# DEV15 Replacement Go / No-Go Summary",
        "",
        f"- run_directory: `{run_dir.name}`",
        f"- total_papers_processed: `{len(paper_rows)}`",
        f"- exact_match_count: `{exact_count}`",
        f"- minor_deviation_count: `{minor_count}`",
        f"- major_regression_count: `{major_count}`",
        f"- final_decision: `{decision}`",
        "",
        "## Direct answers",
        "",
        f"1. Did the true replacement path run end-to-end on all DEV15 papers? `{'yes' if len(paper_rows) == 15 else 'no'}`",
        f"2. How many papers are exact match at Layer1? `{sum(1 for row in paper_rows if row['layer1_status'] == 'match')}`",
        f"3. How many papers are acceptable minor deviations? `{minor_count}`",
        f"4. How many papers are major regressions? `{major_count}`",
        f"5. Is the replacement path currently usable enough to proceed? `{'yes' if decision != 'NO_GO' else 'no'}`",
        f"6. Final decision: `{decision}`",
        "",
        "## Paper status",
        "",
    ]
    for row in paper_rows:
        lines.append(
            f"- {row['zotero_key']}: Layer1 `{row['layer1_pred_count']}/{row['layer1_gt_count']}`; "
            f"Layer2 `{row['layer2_status']}`; overall `{row['overall_status']}`; note: {row['concise_note']}"
        )
    if regression_rows:
        lines.extend(["", "## Triggered regression diagnostics", ""])
        for row in regression_rows:
            lines.append(
                f"- {row['zotero_key']}: reference Layer1 `{row['reference_layer1_status']}` -> current `{row['current_layer1_status']}`; "
                f"owner `{row['practical_owner']}`; note: {row['diagnostic_note']}"
            )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a practical DEV15 replacement go/no-go report.")
    parser.add_argument("--layer1-tsv", required=True, type=Path)
    parser.add_argument("--layer2-tsv", required=True, type=Path)
    parser.add_argument("--failure-taxonomy-tsv", required=True, type=Path)
    parser.add_argument("--paper-selection-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-layer1-tsv", type=Path, default=None)
    parser.add_argument("--reference-layer2-tsv", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    layer1 = by_key(read_tsv(args.layer1_tsv))
    layer2 = by_key(read_tsv(args.layer2_tsv))
    failure = by_key(read_tsv(args.failure_taxonomy_tsv))
    selection_rows = read_tsv(args.paper_selection_tsv)
    selection = {normalize_text(row.get("paper_key") or row.get("key")): row for row in selection_rows}
    ref_layer1 = by_key(read_tsv(args.reference_layer1_tsv)) if args.reference_layer1_tsv else {}
    ref_layer2 = by_key(read_tsv(args.reference_layer2_tsv)) if args.reference_layer2_tsv else {}

    paper_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    for key in sorted(selection):
        l1 = layer1[key]
        l2 = layer2[key]
        failure_row = failure[key]
        delta_vs_gt = int(normalize_text(l1.get("delta_vs_gt")) or "0")
        total_mismatch = int(normalize_text(l2.get("total_mismatch")) or "0")
        layer1_status, layer2_status, note = classify_status(delta_vs_gt, total_mismatch)
        concise_note = normalize_text(failure_row.get("notes")) or note
        paper_row = {
            "zotero_key": key,
            "layer1_pred_count": normalize_text(l1.get("stage5_final_rows")),
            "layer1_gt_count": normalize_text(l1.get("gt_rows")),
            "layer1_status": normalize_text(l1.get("layer1_status")),
            "layer2_status": layer2_status,
            "overall_status": layer1_status,
            "targeted_regression_analysis_triggered": "yes" if layer1_status == "major_regression" else "no",
            "concise_note": concise_note,
        }
        paper_rows.append(paper_row)

        if layer1_status == "major_regression":
            ref_l1 = ref_layer1.get(key, {})
            ref_l2 = ref_layer2.get(key, {})
            practical_owner = normalize_text(failure_row.get("repair_target")) or normalize_text(failure_row.get("likely_owner")) or "other"
            if "adapter" in practical_owner.lower():
                practical_owner = "adapter information loss"
            elif "hidden" in practical_owner.lower():
                practical_owner = "hidden downstream dependency"
            elif "stage2" in practical_owner.lower():
                practical_owner = "emitter under-recovery"
            elif practical_owner == "other":
                if delta_vs_gt < 0:
                    practical_owner = "emitter under-recovery"
                elif delta_vs_gt > 0:
                    practical_owner = "emitter over-generation"
            diagnostic_note = concise_note
            regression_rows.append(
                {
                    "zotero_key": key,
                    "reference_layer1_status": normalize_text(ref_l1.get("comparison_status") or ref_l1.get("layer1_status") or "not_available"),
                    "current_layer1_status": normalize_text(l1.get("layer1_status")),
                    "reference_layer2_mismatch": normalize_text(ref_l2.get("total_mismatch") or "not_available"),
                    "current_layer2_mismatch": normalize_text(l2.get("total_mismatch")),
                    "practical_owner": practical_owner,
                    "diagnostic_note": diagnostic_note,
                }
            )

    decision = final_decision(
        major_count=sum(1 for row in paper_rows if row["overall_status"] == "major_regression"),
        exact_count=sum(1 for row in paper_rows if row["overall_status"] == "exact_match"),
        total_count=len(paper_rows),
    )

    write_tsv(
        args.out_dir / STATUS_TSV,
        paper_rows,
        [
            "zotero_key",
            "layer1_pred_count",
            "layer1_gt_count",
            "layer1_status",
            "layer2_status",
            "overall_status",
            "targeted_regression_analysis_triggered",
            "concise_note",
        ],
    )
    write_tsv(
        args.out_dir / REGRESSION_TSV,
        regression_rows,
        [
            "zotero_key",
            "reference_layer1_status",
            "current_layer1_status",
            "reference_layer2_mismatch",
            "current_layer2_mismatch",
            "practical_owner",
            "diagnostic_note",
        ],
    )
    (args.out_dir / SUMMARY_MD).write_text(
        build_summary(run_dir=args.out_dir, paper_rows=paper_rows, regression_rows=regression_rows, decision=decision),
        encoding="utf-8",
    )
    print(json.dumps({"papers": len(paper_rows), "major_regression_count": len(regression_rows), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
