#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DELTA_TSV_NAME = "emitter_vs_lift_delta.tsv"
DELTA_MD_NAME = "emitter_vs_lift_summary.md"


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_identity_labels(path: Path) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for document in read_jsonl(path):
        key = normalize_text(document.get("document_key"))
        labels[key] = {
            normalize_text(item.get("raw_formulation_label"))
            for item in document.get("formulation_identity_candidates", [])
            if normalize_text(item.get("raw_formulation_label"))
        }
    return labels


def rows_by_key(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {normalize_text(row.get("paper_key")): row for row in rows}


def relation_summary_by_key(path: Path) -> dict[str, dict[str, str]]:
    return rows_by_key(read_tsv(path))


def relation_diff(current_row: dict[str, str], legacy_row: dict[str, str]) -> int | None:
    current = normalize_text(current_row.get("relation_row_count") or current_row.get("stage3_relation_rows"))
    baseline = normalize_text(legacy_row.get("relation_row_count") or legacy_row.get("baseline_stage3_relation_rows"))
    if not current or not baseline:
        return None
    return abs(int(current) - int(baseline))


def delta_md(rows: list[dict[str, Any]], prior_run_id: str, current_run_id: str, primary_next_action: str) -> str:
    lines = [
        "# Emitter Vs Lift Summary",
        "",
        f"- prior_run_id: `{prior_run_id}`",
        f"- current_run_id: `{current_run_id}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['paper_key']}",
                "",
                f"- newly recovered semantic objects: {row['newly_recovered_labels'] or 'none'}",
                f"- Stage3 divergence change: `{row['stage3_divergence_change']}`",
                f"- Layer1 count change: `{row['layer1_change']}` ({row['prior_layer1_status']} -> {row['current_layer1_status']})",
                f"- Layer2 boundary behavior change: `{row['layer2_change']}`",
                f"- primary remaining issue: `{row['primary_remaining_issue']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            "",
            f"- primary_next_action: `{primary_next_action}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a paper-driven replacement validation run against the earlier lift-based run.")
    parser.add_argument("--prior-run-dir", required=True, type=Path)
    parser.add_argument("--current-run-dir", required=True, type=Path)
    parser.add_argument("--paper-keys", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prior_layer1 = rows_by_key(read_tsv(args.prior_run_dir / "layer1_count_comparison.tsv"))
    current_layer1 = rows_by_key(read_tsv(args.current_run_dir / "layer1_count_comparison.tsv"))
    prior_layer2 = rows_by_key(read_tsv(args.prior_run_dir / "layer2_identity_comparison.tsv"))
    current_layer2 = rows_by_key(read_tsv(args.current_run_dir / "layer2_identity_comparison.tsv"))
    prior_failure = rows_by_key(read_tsv(args.prior_run_dir / "replacement_failure_taxonomy.tsv"))
    current_failure = rows_by_key(read_tsv(args.current_run_dir / "replacement_failure_taxonomy.tsv"))
    prior_relation_summary = relation_summary_by_key(args.prior_run_dir / "formulation_relation_v1" / "formulation_relation_summary_v1.tsv")
    prior_legacy_relation_summary = relation_summary_by_key(args.prior_run_dir / "legacy_reference" / "formulation_relation_v1" / "formulation_relation_summary_v1.tsv")
    current_relation_summary = relation_summary_by_key(args.current_run_dir / "formulation_relation_v1" / "formulation_relation_summary_v1.tsv")
    current_legacy_relation_summary = relation_summary_by_key(args.current_run_dir / "legacy_reference" / "formulation_relation_v1" / "formulation_relation_summary_v1.tsv")
    prior_labels = load_identity_labels(args.prior_run_dir / "semantic_stage2_objects" / "semantic_stage2_objects_v1.jsonl")
    current_labels = load_identity_labels(args.current_run_dir / "semantic_stage2_objects" / "semantic_stage2_objects_v1.jsonl")

    rows: list[dict[str, Any]] = []
    for paper_key in args.paper_keys:
        prior_l1 = prior_layer1[paper_key]
        current_l1 = current_layer1[paper_key]
        prior_l2 = prior_layer2[paper_key]
        current_l2 = current_layer2[paper_key]
        prior_f = prior_failure[paper_key]
        current_f = current_failure[paper_key]
        prior_rel_diff = relation_diff(prior_relation_summary[paper_key], prior_legacy_relation_summary[paper_key])
        current_rel_diff = relation_diff(current_relation_summary[paper_key], current_legacy_relation_summary[paper_key])
        if prior_rel_diff is None or current_rel_diff is None:
            stage3_change = "unresolved"
        elif current_rel_diff < prior_rel_diff:
            stage3_change = "improved"
        elif current_rel_diff > prior_rel_diff:
            stage3_change = "worsened"
        else:
            stage3_change = "stayed_same"

        prior_abs = abs(int(prior_l1["delta_vs_gt"]))
        current_abs = abs(int(current_l1["delta_vs_gt"]))
        if current_abs < prior_abs:
            layer1_change = "improved"
        elif current_abs > prior_abs:
            layer1_change = "worsened"
        else:
            layer1_change = "stayed_same"

        prior_mismatch = int(prior_l2["total_mismatch"])
        current_mismatch = int(current_l2["total_mismatch"])
        if current_mismatch < prior_mismatch:
            layer2_change = "improved"
        elif current_mismatch > prior_mismatch:
            layer2_change = "worsened"
        else:
            layer2_change = "stayed_same"

        new_labels = sorted(current_labels.get(paper_key, set()) - prior_labels.get(paper_key, set()))
        rows.append(
            {
                "paper_key": paper_key,
                "prior_identity_count": prior_f.get("semantic_identity_count", ""),
                "current_identity_count": current_f.get("semantic_identity_count", ""),
                "newly_recovered_labels": "; ".join(new_labels),
                "prior_stage3_relation_rows": prior_relation_summary[paper_key].get("relation_row_count", ""),
                "current_stage3_relation_rows": current_relation_summary[paper_key].get("relation_row_count", ""),
                "prior_stage3_baseline_relation_rows": prior_legacy_relation_summary[paper_key].get("relation_row_count", ""),
                "current_stage3_baseline_relation_rows": current_legacy_relation_summary[paper_key].get("relation_row_count", ""),
                "stage3_divergence_change": stage3_change,
                "prior_layer1_status": prior_l1.get("layer1_status", ""),
                "current_layer1_status": current_l1.get("layer1_status", ""),
                "prior_delta_vs_gt": prior_l1.get("delta_vs_gt", ""),
                "current_delta_vs_gt": current_l1.get("delta_vs_gt", ""),
                "layer1_change": layer1_change,
                "prior_failure_family": prior_f.get("dominant_failure_family", ""),
                "current_failure_family": current_f.get("dominant_failure_family", ""),
                "layer2_change": layer2_change,
                "primary_remaining_issue": current_f.get("repair_target", ""),
            }
        )

    primary_next_action = "improve_emitter"
    current_statuses = {row["layer1_change"] for row in rows}
    if all(row["stage3_divergence_change"] == "worsened" for row in rows):
        primary_next_action = "expose_hidden_stage3_dependency"
    elif all(row["primary_remaining_issue"] == "hidden Stage3 dependency" for row in rows):
        primary_next_action = "expose_hidden_stage3_dependency"
    elif "improved" in current_statuses and any(row["current_delta_vs_gt"] != "0" for row in rows):
        primary_next_action = "improve_emitter"
    elif all(row["current_delta_vs_gt"] == "0" for row in rows):
        primary_next_action = "hold_architecture_steady_and_expand"

    write_tsv(
        args.out_dir / DELTA_TSV_NAME,
        rows,
        [
            "paper_key",
            "prior_identity_count",
            "current_identity_count",
            "newly_recovered_labels",
            "prior_stage3_relation_rows",
            "current_stage3_relation_rows",
            "prior_stage3_baseline_relation_rows",
            "current_stage3_baseline_relation_rows",
            "stage3_divergence_change",
            "prior_layer1_status",
            "current_layer1_status",
            "prior_delta_vs_gt",
            "current_delta_vs_gt",
            "layer1_change",
            "prior_failure_family",
            "current_failure_family",
            "layer2_change",
            "primary_remaining_issue",
        ],
    )
    (args.out_dir / DELTA_MD_NAME).write_text(
        delta_md(rows, args.prior_run_dir.name, args.current_run_dir.name, primary_next_action),
        encoding="utf-8",
    )
    print(json.dumps({"papers": len(rows), "out_dir": str(args.out_dir), "primary_next_action": primary_next_action}, indent=2))


if __name__ == "__main__":
    main()
