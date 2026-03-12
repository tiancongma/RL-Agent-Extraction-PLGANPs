#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


COUNTS_NAME = "final_table_vs_gt_counts.tsv"
SUMMARY_NAME = "final_table_vs_gt_summary.md"
EE_SUBSET_NAME = "final_table_vs_gt_ee_subset.tsv"


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def read_scope_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_final_table_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_gt_rows_from_workbook(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook["review_formulations"]
    raw_rows = worksheet.iter_rows(values_only=True)
    header = [str(value) if value is not None else "" for value in next(raw_rows)]
    rows: list[dict[str, str]] = []
    for values in raw_rows:
        rows.append(
            {
                header[idx]: "" if idx >= len(values) or values[idx] is None else str(values[idx])
                for idx in range(len(header))
            }
        )
    return rows, header


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_markdown(
    scope_name: str,
    manifest_path: Path,
    final_table_path: Path,
    gt_xlsx_path: Path,
    counts_rows: list[dict[str, str]],
    ee_supported: bool,
    summary_path: Path,
) -> None:
    totals = {
        "final_table_rows": sum(int(row["final_table_count"]) for row in counts_rows),
        "gt_rows": sum(int(row["gt_count"]) for row in counts_rows),
        "matched_papers": sum(1 for row in counts_rows if row["comparison_status"] == "match"),
        "mismatched_papers": sum(1 for row in counts_rows if row["comparison_status"] != "match"),
    }
    lines = [
        "# Final Table vs GT Summary",
        "",
        "## Declared scope",
        "",
        f"- scope_name: `{scope_name}`",
        f"- scope_manifest_tsv: `{manifest_path}`",
        f"- final_formulation_table_tsv: `{final_table_path}`",
        f"- gt_workbook: `{gt_xlsx_path}`",
        "",
        "## Benchmark-validity statement",
        "",
        "- This comparison is benchmark-valid for the declared scope because it evaluates only the complete-pipeline final formulation table produced by the full pipeline runner.",
        "- No intermediate Stage2 or other partial-layer artifacts are used as the official evaluation object.",
        "",
        "## Supported benchmark views",
        "",
        "- per-DOI final-formulation count comparison: supported",
        f"- EE subset comparison: {'supported' if ee_supported else 'not supported by the current authoritative GT artifact'}",
        "",
        "## Aggregate outcome",
        "",
        f"- total_final_table_rows: `{totals['final_table_rows']}`",
        f"- total_gt_rows: `{totals['gt_rows']}`",
        f"- matched_papers: `{totals['matched_papers']}`",
        f"- mismatched_papers: `{totals['mismatched_papers']}`",
        "",
        "## Per-paper counts",
        "",
    ]
    for row in counts_rows:
        lines.append(
            "- `{paper_key}`: final=`{final_table_count}` gt=`{gt_count}` diff=`{count_diff}` status=`{comparison_status}`".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The current benchmark comparison is limited to final-formulation counts for this declared scope.",
            "- The authoritative fixed DEV15 skeleton workbook does not expose structured EE ground-truth fields, so no benchmark-valid EE subset comparison is emitted in this first full-pipeline run.",
            "- Any mismatch investigation must start from these final-table results and only then trace backward into Stage 5A decision-trace artifacts or Stage 2 candidate rows.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_final_table_to_gt(
    final_table_tsv: Path,
    gt_xlsx: Path,
    scope_manifest_tsv: Path,
    out_dir: Path,
    scope_name: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_scope_manifest(scope_manifest_tsv)
    scope_by_key = {row["key"]: row for row in manifest_rows}
    scope_keys = set(scope_by_key)

    final_rows = [
        row for row in read_final_table_rows(final_table_tsv) if row.get("key", "") in scope_keys
    ]
    gt_rows, gt_header = read_gt_rows_from_workbook(gt_xlsx)
    gt_rows = [
        row
        for row in gt_rows
        if row.get("paper_key", "") in scope_keys
        and normalize_text(row.get("formulation_exists_gt")) == "yes"
    ]

    final_counts = Counter(row["key"] for row in final_rows)
    gt_counts = Counter(row["paper_key"] for row in gt_rows)

    ee_supported = any("encapsulation" in normalize_text(column) for column in gt_header)

    counts_rows: list[dict[str, str]] = []
    for key in sorted(scope_keys):
        manifest_row = scope_by_key[key]
        final_count = int(final_counts.get(key, 0))
        gt_count = int(gt_counts.get(key, 0))
        diff = final_count - gt_count
        if diff == 0:
            status = "match"
        elif diff > 0:
            status = "over"
        else:
            status = "under"
        counts_rows.append(
            {
                "paper_key": key,
                "doi": manifest_row.get("doi", ""),
                "paper_title": manifest_row.get("title", ""),
                "final_table_count": str(final_count),
                "gt_count": str(gt_count),
                "count_diff": str(diff),
                "comparison_status": status,
                "final_table_artifact": str(final_table_tsv),
                "gt_artifact": str(gt_xlsx),
                "notes": (
                    "count_match"
                    if status == "match"
                    else "final_table_vs_fixed_skeleton_count_mismatch"
                ),
            }
        )

    counts_path = out_dir / COUNTS_NAME
    summary_path = out_dir / SUMMARY_NAME
    write_tsv(
        counts_path,
        [
            "paper_key",
            "doi",
            "paper_title",
            "final_table_count",
            "gt_count",
            "count_diff",
            "comparison_status",
            "final_table_artifact",
            "gt_artifact",
            "notes",
        ],
        counts_rows,
    )
    build_summary_markdown(
        scope_name=scope_name,
        manifest_path=scope_manifest_tsv,
        final_table_path=final_table_tsv,
        gt_xlsx_path=gt_xlsx,
        counts_rows=counts_rows,
        ee_supported=ee_supported,
        summary_path=summary_path,
    )

    result = {
        "scope_name": scope_name,
        "final_table_path": str(final_table_tsv),
        "gt_xlsx_path": str(gt_xlsx),
        "counts_path": str(counts_path),
        "summary_path": str(summary_path),
        "ee_subset_path": "",
        "ee_subset_supported": ee_supported,
        "papers_in_scope": len(scope_keys),
        "papers_matching": sum(1 for row in counts_rows if row["comparison_status"] == "match"),
        "papers_mismatching": sum(1 for row in counts_rows if row["comparison_status"] != "match"),
        "total_final_table_rows": sum(int(row["final_table_count"]) for row in counts_rows),
        "total_gt_rows": sum(int(row["gt_count"]) for row in counts_rows),
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a final formulation table against the authoritative fixed skeleton GT workbook."
    )
    parser.add_argument("--final-table-tsv", required=True, type=Path)
    parser.add_argument("--gt-xlsx", required=True, type=Path)
    parser.add_argument("--scope-manifest-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--scope-name", default="controlled_scope")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = compare_final_table_to_gt(
        final_table_tsv=args.final_table_tsv,
        gt_xlsx=args.gt_xlsx,
        scope_manifest_tsv=args.scope_manifest_tsv,
        out_dir=args.out_dir,
        scope_name=args.scope_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
