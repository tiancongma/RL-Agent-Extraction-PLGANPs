#!/usr/bin/env python3
"""
Validate DEV15 formulation skeleton review workbook.

Example:
python src/archive_methods/dev15_skeleton_bootstrap/validate_dev15_formulation_skeleton_review_v1.py --xlsx data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1.xlsx
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

try:
    from src.archive_methods.dev15_skeleton_bootstrap.formulation_skeleton_common import (
        BOUNDARY_CONFIDENCE_OPTIONS,
        FORMULATION_EXISTS_OPTIONS,
        REVIEW_STATUS_OPTIONS,
        SOURCE_TYPE_OPTIONS,
        norm_text,
        read_review_sheet_rows,
        write_tsv,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from formulation_skeleton_common import (
        BOUNDARY_CONFIDENCE_OPTIONS,
        FORMULATION_EXISTS_OPTIONS,
        REVIEW_STATUS_OPTIONS,
        SOURCE_TYPE_OPTIONS,
        norm_text,
        read_review_sheet_rows,
        write_tsv,
    )


def _issue(
    issues: List[Dict[str, str]],
    issue_type: str,
    paper_key: str,
    excel_row: str,
    formulation_id: str,
    details: str,
    severity: str = "warning",
) -> None:
    issues.append(
        {
            "severity": severity,
            "issue_type": issue_type,
            "paper_key": paper_key,
            "excel_row": excel_row,
            "formulation_id": formulation_id,
            "details": details,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, required=True, help="Reviewed workbook path.")
    ap.add_argument("--sheet", type=str, default="review_formulations", help="Review worksheet name.")
    ap.add_argument("--out-report-json", type=Path, default=None, help="Validation summary JSON path.")
    ap.add_argument("--out-issues-tsv", type=Path, default=None, help="Detailed issues TSV path.")
    args = ap.parse_args()

    if not args.xlsx.exists():
        raise FileNotFoundError(f"Workbook not found: {args.xlsx}")

    default_dir = args.xlsx.parent
    out_report = args.out_report_json or (default_dir / "dev15_formulation_skeleton_validation_report_v1.json")
    out_issues = args.out_issues_tsv or (default_dir / "dev15_formulation_skeleton_validation_issues_v1.tsv")

    rows = read_review_sheet_rows(args.xlsx, args.sheet)
    issues: List[Dict[str, str]] = []

    allowed_sets = {
        "source_type": set(SOURCE_TYPE_OPTIONS),
        "formulation_exists_gt": set(FORMULATION_EXISTS_OPTIONS),
        "formulation_boundary_confidence": set(BOUNDARY_CONFIDENCE_OPTIONS),
        "review_status": set(REVIEW_STATUS_OPTIONS),
    }

    paper_confirmed: Dict[str, int] = {}
    id_counts: Dict[tuple[str, str], int] = {}
    uncertain_rows = 0
    needs_second_pass_rows = 0

    for row in rows:
        paper_key = norm_text(row.get("paper_key"))
        formulation_id = norm_text(row.get("formulation_id"))
        excel_row = norm_text(row.get("_excel_row"))

        # Missing/invalid dropdown checks
        for col, allowed in allowed_sets.items():
            value = norm_text(row.get(col)).lower()
            if not value:
                _issue(
                    issues,
                    "missing_required_dropdown",
                    paper_key,
                    excel_row,
                    formulation_id,
                    f"Missing value in {col}",
                    "error",
                )
            elif value not in allowed:
                _issue(
                    issues,
                    "invalid_dropdown_value",
                    paper_key,
                    excel_row,
                    formulation_id,
                    f"Invalid {col}={value}",
                    "error",
                )

        exists_gt = norm_text(row.get("formulation_exists_gt")).lower()
        review_status = norm_text(row.get("review_status")).lower()
        conf = norm_text(row.get("formulation_boundary_confidence")).lower()

        if exists_gt == "yes":
            paper_confirmed[paper_key] = paper_confirmed.get(paper_key, 0) + 1
        if exists_gt == "uncertain" or conf == "low":
            uncertain_rows += 1
            _issue(
                issues,
                "uncertain_row",
                paper_key,
                excel_row,
                formulation_id,
                f"formulation_exists_gt={exists_gt}, formulation_boundary_confidence={conf}",
            )
        if review_status == "needs_second_pass":
            needs_second_pass_rows += 1
            _issue(
                issues,
                "needs_second_pass",
                paper_key,
                excel_row,
                formulation_id,
                "Row flagged for second pass",
            )

        if paper_key and formulation_id:
            key = (paper_key, formulation_id)
            id_counts[key] = id_counts.get(key, 0) + 1
            if not re.fullmatch(re.escape(paper_key) + r"_F\d{2,}", formulation_id):
                _issue(
                    issues,
                    "invalid_formulation_id_format",
                    paper_key,
                    excel_row,
                    formulation_id,
                    "Expected format: {paper_key}_F## (2+ zero-padded digits).",
                    "error",
                )

    # Duplicate IDs within paper
    for row in rows:
        paper_key = norm_text(row.get("paper_key"))
        formulation_id = norm_text(row.get("formulation_id"))
        excel_row = norm_text(row.get("_excel_row"))
        if not paper_key or not formulation_id:
            continue
        if id_counts.get((paper_key, formulation_id), 0) > 1:
            _issue(
                issues,
                "duplicate_formulation_id_within_paper",
                paper_key,
                excel_row,
                formulation_id,
                "Duplicate formulation_id found within same paper",
                "error",
            )

    # Papers with zero confirmed formulations
    all_papers = sorted({norm_text(r.get("paper_key")) for r in rows if norm_text(r.get("paper_key"))})
    zero_confirmed = [p for p in all_papers if paper_confirmed.get(p, 0) == 0]
    for paper_key in zero_confirmed:
        _issue(
            issues,
            "paper_zero_confirmed_formulations",
            paper_key,
            "",
            "",
            "No row with formulation_exists_gt=yes",
            "warning",
        )

    write_tsv(
        out_issues,
        fieldnames=["severity", "issue_type", "paper_key", "excel_row", "formulation_id", "details"],
        rows=issues,
    )

    summary = {
        "workbook": str(args.xlsx),
        "sheet": args.sheet,
        "row_count": len(rows),
        "issues_total": len(issues),
        "issues_error": sum(1 for i in issues if i["severity"] == "error"),
        "issues_warning": sum(1 for i in issues if i["severity"] == "warning"),
        "uncertain_rows": uncertain_rows,
        "needs_second_pass_rows": needs_second_pass_rows,
        "papers_total": len(all_papers),
        "papers_with_zero_confirmed_formulations": zero_confirmed,
        "issues_tsv": str(out_issues),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] report_json\t{out_report}")
    print(f"[OK] issues_tsv\t{out_issues}")
    print(f"[OK] rows\t{len(rows)}")
    print(f"[OK] issues\t{len(issues)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


