#!/usr/bin/env python3
from __future__ import annotations

"""
Validate identity freeze against a frozen upstream scaffold surface.

Purpose:
- enforce the identity-freeze gate at the Stage5
  post-materialization boundary
- detect row count drift, identity reassignment, unresolved scaffold rows, and
  ambiguous bindings
- emit diagnostics without mutating benchmark-valid outputs
- fail fast on violations unless explicitly run in report-only mode
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_TSV_NAME = "identity_freeze_report_v1.tsv"
SUMMARY_TSV_NAME = "identity_freeze_summary_v1.tsv"
SUMMARY_MD_NAME = "identity_freeze_summary_v1.md"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(row.get(key, "")) for key in fieldnames})


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(normalize_text(row.get(column)).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def build_identity_freeze_report(
    *,
    scaffold_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
    paper_keys: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    report_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    final_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in final_rows:
        paper_key = normalize_text(row.get("key") or row.get("paper_key"))
        if paper_key:
            final_by_paper[paper_key].append(row)

    scaffold_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scaffold_rows:
        paper_key = normalize_text(row.get("paper_key"))
        if paper_key:
            scaffold_by_paper[paper_key].append(row)

    for paper_key in paper_keys:
        scoped_scaffold = scaffold_by_paper.get(paper_key, [])
        scoped_final = final_by_paper.get(paper_key, [])
        selected_rep_ids = {
            normalize_text(row.get("selected_new_representative_source_formulation_ids"))
            for row in scoped_scaffold
            if normalize_text(row.get("selected_new_representative_source_formulation_ids"))
        }
        final_rep_ids = {
            normalize_text(row.get("representative_source_formulation_id"))
            for row in scoped_final
            if normalize_text(row.get("representative_source_formulation_id"))
        }
        unresolved = [row for row in scoped_scaffold if normalize_text(row.get("binding_status")) == "unresolved"]
        ambiguous = [row for row in scoped_scaffold if normalize_text(row.get("binding_status")) == "ambiguous"]
        missing_from_final = sorted(selected_rep_ids - final_rep_ids)
        unbound_final = sorted(final_rep_ids - selected_rep_ids)

        upstream_identity_count = len(scoped_scaffold)
        final_table_count = len(scoped_final)
        selected_binding_count = len(selected_rep_ids)
        row_count_drift = upstream_identity_count != final_table_count
        membership_drift = bool(missing_from_final or unbound_final)
        violation = row_count_drift or membership_drift or bool(unresolved or ambiguous)

        summary_rows.append(
            {
                "paper_key": paper_key,
                "upstream_identity_count": str(upstream_identity_count),
                "final_table_count": str(final_table_count),
                "selected_binding_count": str(selected_binding_count),
                "row_count_drift_detected": "yes" if row_count_drift else "no",
                "identity_reassignment_detected": "yes" if membership_drift else "no",
                "unresolved_scaffold_rows": str(len(unresolved)),
                "ambiguous_scaffold_rows": str(len(ambiguous)),
                "violation": "yes" if violation else "no",
                "status": "pass" if not violation else "fail",
            }
        )

        for row in scoped_scaffold:
            selected_rep_id = normalize_text(row.get("selected_new_representative_source_formulation_ids"))
            selected_final_ids = normalize_text(row.get("selected_new_final_formulation_ids"))
            binding_status = normalize_text(row.get("binding_status"))
            report_rows.append(
                {
                    "paper_key": paper_key,
                    "scaffold_key": normalize_text(row.get("scaffold_key")),
                    "gt_formulation_id": normalize_text(row.get("gt_formulation_id")),
                    "scaffold_native_label": normalize_text(row.get("scaffold_native_label")),
                    "selected_binding_rule": normalize_text(row.get("selected_new_binding_rule")),
                    "selected_representative_source_formulation_id": selected_rep_id,
                    "selected_final_formulation_id": selected_final_ids,
                    "binding_status": binding_status,
                    "missing_from_final_table": "yes" if selected_rep_id and selected_rep_id not in final_rep_ids else "no",
                    "ambiguous_binding": "yes" if binding_status == "ambiguous" else "no",
                    "violation": "yes" if binding_status in {"unresolved", "ambiguous"} else "no",
                }
            )

        for rep_id in unbound_final:
            report_rows.append(
                {
                    "paper_key": paper_key,
                    "scaffold_key": "",
                    "gt_formulation_id": "",
                    "scaffold_native_label": "",
                    "selected_binding_rule": "",
                    "selected_representative_source_formulation_id": rep_id,
                    "selected_final_formulation_id": "",
                    "binding_status": "unbound_final_row",
                    "missing_from_final_table": "no",
                    "ambiguous_binding": "no",
                    "violation": "yes",
                }
            )

    return report_rows, summary_rows


def build_summary_markdown(
    *,
    scaffold_rows_tsv: Path,
    final_table_tsv: Path,
    paper_keys: list[str],
    summary_rows: list[dict[str, str]],
) -> str:
    lines = [
        "# Identity Freeze Summary v1",
        "",
        "IDENTITY_FREEZE_RULE_V1 check at the Stage5 post-materialization boundary.",
        "",
        "## Inputs",
        f"- upstream identity scaffold: `{scaffold_rows_tsv}`",
        f"- Stage5 final table: `{final_table_tsv}`",
        f"- paper scope: `{', '.join(paper_keys)}`",
        "",
        "## Rule",
        "- formulation count must remain invariant after identity freeze",
        "- formulation membership must remain invariant after identity freeze",
        "- downstream stages may attach, resolve, and derive fields only",
        "- downstream stages must not implicitly split or merge formulations",
        "",
        "## Results",
        markdown_table(
            summary_rows,
            [
                "paper_key",
                "upstream_identity_count",
                "final_table_count",
                "selected_binding_count",
                "row_count_drift_detected",
                "identity_reassignment_detected",
                "unresolved_scaffold_rows",
                "ambiguous_scaffold_rows",
                "violation",
                "status",
            ],
        ),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce the identity freeze rule.")
    parser.add_argument("--identity-scaffold-rows-tsv", type=Path, required=True)
    parser.add_argument("--final-table-tsv", type=Path, required=True)
    parser.add_argument("--paper-key", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write diagnostics without failing non-zero on violations. Default behavior is hard-gate enforcement.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scaffold_rows_tsv = args.identity_scaffold_rows_tsv.resolve()
    final_table_tsv = args.final_table_tsv.resolve()
    out_dir = args.out_dir.resolve()
    paper_keys = [normalize_text(value) for value in args.paper_key if normalize_text(value)]

    scaffold_rows = [
        row for row in read_tsv_rows(scaffold_rows_tsv) if normalize_text(row.get("paper_key")) in set(paper_keys)
    ]
    final_rows = [
        row for row in read_tsv_rows(final_table_tsv) if normalize_text(row.get("key") or row.get("paper_key")) in set(paper_keys)
    ]

    report_rows, summary_rows = build_identity_freeze_report(
        scaffold_rows=scaffold_rows,
        final_rows=final_rows,
        paper_keys=paper_keys,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        out_dir / REPORT_TSV_NAME,
        [
            "paper_key",
            "scaffold_key",
            "gt_formulation_id",
            "scaffold_native_label",
            "selected_binding_rule",
            "selected_representative_source_formulation_id",
            "selected_final_formulation_id",
            "binding_status",
            "missing_from_final_table",
            "ambiguous_binding",
            "violation",
        ],
        report_rows,
    )
    write_tsv(
        out_dir / SUMMARY_TSV_NAME,
        [
            "paper_key",
            "upstream_identity_count",
            "final_table_count",
            "selected_binding_count",
            "row_count_drift_detected",
            "identity_reassignment_detected",
            "unresolved_scaffold_rows",
            "ambiguous_scaffold_rows",
            "violation",
            "status",
        ],
        summary_rows,
    )
    (out_dir / SUMMARY_MD_NAME).write_text(
        build_summary_markdown(
            scaffold_rows_tsv=scaffold_rows_tsv,
            final_table_tsv=final_table_tsv,
            paper_keys=paper_keys,
            summary_rows=summary_rows,
        ),
        encoding="utf-8",
    )

    failure_conditions: list[str] = [
        "row count drift",
        "identity reassignment",
        "unresolved scaffold binding",
        "ambiguous scaffold binding",
    ]
    violations = [row for row in summary_rows if normalize_text(row.get("violation")) == "yes"]

    print(
        json.dumps(
            {
                "identity_scaffold_rows_tsv": str(scaffold_rows_tsv),
                "final_table_tsv": str(final_table_tsv),
                "paper_keys": paper_keys,
                "out_dir": str(out_dir),
                "mode": "report_only" if args.report_only else "hard_gate",
                "failure_conditions": failure_conditions,
            },
            ensure_ascii=True,
        )
    )
    print("failure_conditions=row count drift | identity reassignment | unresolved scaffold binding | ambiguous scaffold binding")
    for row in summary_rows:
        print(
            "paper={paper_key} upstream_identity_count={upstream_identity_count} "
            "final_table_count={final_table_count} drift_detected={row_count_drift_detected} "
            "identity_reassignment={identity_reassignment_detected} violation={violation}".format(**row)
        )
    if violations and not args.report_only:
        print("IDENTITY_FREEZE_GATE=FAIL")
        for row in violations:
            reasons: list[str] = []
            if normalize_text(row.get("row_count_drift_detected")) == "yes":
                reasons.append("row count drift")
            if normalize_text(row.get("identity_reassignment_detected")) == "yes":
                reasons.append("identity reassignment")
            if normalize_text(row.get("unresolved_scaffold_rows")) not in {"", "0"}:
                reasons.append(f"unresolved scaffold rows={row.get('unresolved_scaffold_rows')}")
            if normalize_text(row.get("ambiguous_scaffold_rows")) not in {"", "0"}:
                reasons.append(f"ambiguous scaffold rows={row.get('ambiguous_scaffold_rows')}")
            print(
                "gate_failure paper={paper_key} reasons={reasons}".format(
                    paper_key=row.get("paper_key", ""),
                    reasons="; ".join(reasons) or "unknown",
                )
            )
        return 2
    if violations:
        print("IDENTITY_FREEZE_GATE=VIOLATIONS_RECORDED_REPORT_ONLY")
    else:
        print("IDENTITY_FREEZE_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
