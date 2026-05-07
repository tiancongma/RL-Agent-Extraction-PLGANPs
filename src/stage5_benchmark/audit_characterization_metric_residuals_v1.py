#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

MEASUREMENT_FIELDS = {
    "particle_size_nm",
    "pdi",
    "zeta_mV",
    "ee_percent",
    "lc_percent",
    "dl_percent",
}

OUTPUT_NAME = "characterization_metric_residual_audit_v1.tsv"
SUMMARY_NAME = "characterization_metric_residual_audit_summary_v1.json"


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def classify_measurement_residual(row: dict[str, str]) -> str:
    """Classify the first visible failure boundary for a metric compare cell.

    This is diagnostic-only.  It does not read GT snippets or source documents;
    it only classifies the compare surface so implementation can focus on the
    lawful upstream boundary before attempting value materialization.
    """

    status = normalize_text(row.get("compare_status"))
    alignment_rule = normalize_text(row.get("alignment_rule")).lower()
    source_type = normalize_text(row.get("system_value_source_type")).lower()
    evidence_detail = normalize_text(row.get("evidence_status_detail")).lower()

    if status == "blocked_alignment" or "blocked" in alignment_rule:
        return "alignment_blocked_before_metric_projection"
    if status == "extra_in_system":
        return "extra_metric_surface_requires_review"
    if status == "present_but_mismatch":
        return "present_but_mismatch_endpoint_or_value_policy"
    if status == "missing_in_system":
        if "missing_system_field_surface" in {source_type, evidence_detail}:
            return "missing_system_field_surface"
        if "table" in source_type or "header" in source_type:
            return "measurement_table_header_binding_gap"
        return "measurement_projection_gap"
    if status == "present_and_match":
        return "ok_present_and_match"
    if status == "not_reported_in_gt":
        return "not_reported_in_gt"
    return "other_compare_status"


def audit_rows(cells: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    audited: list[dict[str, str]] = []
    for row in cells:
        field = normalize_text(row.get("field_name"))
        if field not in MEASUREMENT_FIELDS:
            continue
        audited.append(
            {
                "paper_key": normalize_text(row.get("paper_key")),
                "field_name": field,
                "compare_status": normalize_text(row.get("compare_status")),
                "gt_formulation_id": normalize_text(row.get("gt_formulation_id")),
                "matched_system_formulation_id": normalize_text(row.get("matched_system_formulation_id")),
                "system_value_raw": normalize_text(row.get("system_value_raw")),
                "gt_value_raw": normalize_text(row.get("gt_value_raw")),
                "system_value_source_type": normalize_text(row.get("system_value_source_type")),
                "evidence_status_detail": normalize_text(row.get("evidence_status_detail")),
                "alignment_rule": normalize_text(row.get("alignment_rule")),
                "first_failure_boundary": classify_measurement_residual(row),
                "notes": "diagnostic_only_compare_surface_classification",
            }
        )
    return audited


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "paper_key",
        "field_name",
        "compare_status",
        "gt_formulation_id",
        "matched_system_formulation_id",
        "system_value_raw",
        "gt_value_raw",
        "system_value_source_type",
        "evidence_status_detail",
        "alignment_rule",
        "first_failure_boundary",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-cells-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--benchmark-valid", default="no", choices=["yes", "no"])
    args = parser.parse_args()

    cells_path = Path(args.compare_cells_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = audit_rows(csv.DictReader(cells_path.open(), delimiter="\t"))
    write_tsv(out_dir / OUTPUT_NAME, rows)
    boundary_counts = Counter(r["first_failure_boundary"] for r in rows)
    field_status_counts = Counter((r["field_name"], r["compare_status"]) for r in rows)
    summary = {
        "benchmark_valid": args.benchmark_valid,
        "input_compare_cells_tsv": str(cells_path),
        "row_count": len(rows),
        "boundary_counts": dict(boundary_counts),
        "field_status_counts": {f"{field}\t{status}": count for (field, status), count in field_status_counts.items()},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# RUN_CONTEXT\n\n"
        f"benchmark_valid: {args.benchmark_valid}\n"
        "purpose: Diagnostic-only characterization measurement metric residual boundary audit.\n"
        f"input_compare_cells_tsv: {cells_path}\n"
        f"outputs: {OUTPUT_NAME}, {SUMMARY_NAME}\n"
        "notes: Compare-surface classification only; does not materialize values or read raw/GT snippets.\n"
    )
    print(json.dumps({"out_dir": str(out_dir), "row_count": len(rows), "boundary_counts": dict(boundary_counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
