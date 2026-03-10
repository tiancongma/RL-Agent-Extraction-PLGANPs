#!/usr/bin/env python3
"""
Export reviewed DEV15 formulation skeleton workbook into clean TSV.

Example:
python src/stage3_gt/export_dev15_formulation_skeleton_gt_v1.py --xlsx data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

try:
    from src.stage3_gt.formulation_skeleton_common import norm_text, read_review_sheet_rows, write_tsv
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage3_gt.formulation_skeleton_common import norm_text, read_review_sheet_rows, write_tsv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, required=True, help="Reviewed workbook path.")
    ap.add_argument("--sheet", type=str, default="review_formulations", help="Review worksheet name.")
    ap.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Output TSV path. Defaults to workbook directory.",
    )
    ap.add_argument(
        "--include-needs-second-pass",
        action="store_true",
        help="Include rows with review_status=needs_second_pass in export.",
    )
    args = ap.parse_args()

    if not args.xlsx.exists():
        raise FileNotFoundError(f"Workbook not found: {args.xlsx}")
    out_tsv = args.out_tsv or (args.xlsx.parent / "dev15_formulation_skeleton_gt_v1.tsv")

    rows = read_review_sheet_rows(args.xlsx, args.sheet)
    exported: List[Dict[str, str]] = []
    allowed_status = {"reviewed"}
    if args.include_needs_second_pass:
        allowed_status.add("needs_second_pass")

    for row in rows:
        exists_gt = norm_text(row.get("formulation_exists_gt")).lower()
        review_status = norm_text(row.get("review_status")).lower()
        if exists_gt != "yes":
            continue
        if review_status not in allowed_status:
            continue

        exported.append(
            {
                "paper_key": norm_text(row.get("paper_key")),
                "doi": norm_text(row.get("doi")),
                "formulation_id": norm_text(row.get("formulation_id")),
                "formulation_label_raw": norm_text(row.get("formulation_label_raw")),
                "source_type": norm_text(row.get("source_type")),
                "source_locator": norm_text(row.get("source_locator")),
                "formulation_boundary_confidence": norm_text(row.get("formulation_boundary_confidence")),
                "notes": norm_text(row.get("notes")),
            }
        )

    # Remove duplicate rows by stable key while preserving order
    dedup: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in exported:
        key = (row["paper_key"], row["formulation_id"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    dedup.sort(key=lambda r: (r["paper_key"], r["formulation_id"]))
    write_tsv(
        out_tsv,
        fieldnames=[
            "paper_key",
            "doi",
            "formulation_id",
            "formulation_label_raw",
            "source_type",
            "source_locator",
            "formulation_boundary_confidence",
            "notes",
        ],
        rows=dedup,
    )

    print(f"[OK] out_tsv\t{out_tsv}")
    print(f"[OK] exported_rows\t{len(dedup)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
