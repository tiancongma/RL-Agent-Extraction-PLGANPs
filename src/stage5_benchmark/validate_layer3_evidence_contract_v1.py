#!/usr/bin/env python3
from __future__ import annotations

"""
Validate the Layer 3 evidence handoff contract against golden workbook cases.

This is a lightweight regression safeguard for reviewer-facing Layer 3 outputs.
It does not modify Stage 2, Stage 3, or Stage 5 semantics.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import DOCS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DOCS_DIR


DEFAULT_GOLDEN_CASES = DOCS_DIR / "methods" / "layer3_evidence_handoff_golden_cases_v1.tsv"


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def get_state(value: str) -> str:
    return "nonempty" if normalize_text(value) else "empty"


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        normalize_text(row.get("paper_key")),
        normalize_text(row.get("formulation_id")),
        normalize_text(row.get("field_name")),
    )


def match_expected_state(actual_value: str, expected_state: str) -> bool:
    expected = normalize_text(expected_state).lower() or "any"
    if expected == "any":
        return True
    return get_state(actual_value) == expected


def assert_regex_absent(value: str, pattern: str) -> bool:
    compact_pattern = normalize_text(pattern)
    if not compact_pattern:
        return True
    return re.search(compact_pattern, normalize_text(value)) is None


def validate_case(case: dict[str, str], seed_row: dict[str, str]) -> list[str]:
    errors: list[str] = []

    extracted_value = normalize_text(seed_row.get("extracted_value"))
    evidence_text = normalize_text(seed_row.get("evidence_text"))
    evidence_anchor_text = normalize_text(seed_row.get("evidence_anchor_text"))
    evidence_status_detail = normalize_text(seed_row.get("evidence_status_detail"))
    relation_resolution_rule = normalize_text(seed_row.get("relation_resolution_rule"))
    review_warning = normalize_text(seed_row.get("review_warning"))
    evidence_source_type = normalize_text(seed_row.get("evidence_source_type"))

    expected_value_state = case.get("expected_extracted_value_state", "any")
    if not match_expected_state(extracted_value, expected_value_state):
        errors.append(
            f"expected extracted_value state {expected_value_state}, got {get_state(extracted_value)}"
        )

    expected_value_exact = normalize_text(case.get("expected_extracted_value_exact"))
    if expected_value_exact and extracted_value != expected_value_exact:
        errors.append(
            f"expected extracted_value={expected_value_exact!r}, got {extracted_value!r}"
        )

    expected_anchor_state = case.get("expected_evidence_anchor_state", "any")
    if not match_expected_state(evidence_anchor_text, expected_anchor_state):
        errors.append(
            f"expected evidence_anchor_text state {expected_anchor_state}, got {get_state(evidence_anchor_text)}"
        )

    expected_status = normalize_text(case.get("expected_evidence_status_detail"))
    if expected_status and evidence_status_detail != expected_status:
        errors.append(
            f"expected evidence_status_detail={expected_status!r}, got {evidence_status_detail!r}"
        )

    expected_relation_state = case.get("expected_relation_resolution_rule_state", "any")
    if not match_expected_state(relation_resolution_rule, expected_relation_state):
        errors.append(
            "expected relation_resolution_rule state "
            f"{expected_relation_state}, got {get_state(relation_resolution_rule)}"
        )

    expected_review_warning = normalize_text(case.get("expected_review_warning"))
    if expected_review_warning and review_warning != expected_review_warning:
        errors.append(
            f"expected review_warning={expected_review_warning!r}, got {review_warning!r}"
        )

    require_not_supported = normalize_text(case.get("require_not_supported")).lower()
    if require_not_supported in {"1", "true", "yes"} and evidence_source_type == "text":
        errors.append("expected case to remain unsupported, but evidence_source_type=text")

    forbidden_anchor_regex = case.get("forbidden_anchor_regex", "")
    if not assert_regex_absent(evidence_anchor_text, forbidden_anchor_regex):
        errors.append("evidence_anchor_text matched forbidden regex")

    forbidden_evidence_regex = case.get("forbidden_evidence_regex", "")
    if not assert_regex_absent(evidence_text, forbidden_evidence_regex):
        errors.append("evidence_text matched forbidden regex")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Layer 3 evidence handoff golden cases.")
    parser.add_argument("--seed-tsv", type=Path, required=True, help="Path to field_gt_review_seed_rows_v*.tsv")
    parser.add_argument(
        "--golden-cases-tsv",
        type=Path,
        default=DEFAULT_GOLDEN_CASES,
        help="Path to golden case TSV. Defaults to docs/methods/layer3_evidence_handoff_golden_cases_v1.tsv",
    )
    args = parser.parse_args()

    seed_rows = read_tsv_rows(args.seed_tsv.resolve())
    golden_cases = read_tsv_rows(args.golden_cases_tsv.resolve())
    row_index = {row_key(row): row for row in seed_rows}

    failures: list[str] = []
    passes = 0

    for case in golden_cases:
        case_name = normalize_text(case.get("case_id")) or "unnamed_case"
        lookup_key = (
            normalize_text(case.get("paper_key")),
            normalize_text(case.get("formulation_id")),
            normalize_text(case.get("field_name")),
        )
        seed_row = row_index.get(lookup_key)
        if seed_row is None:
            failures.append(f"[FAIL] {case_name}: missing seed row for key={lookup_key}")
            continue
        errors = validate_case(case, seed_row)
        if errors:
            failures.append(f"[FAIL] {case_name}: " + "; ".join(errors))
        else:
            passes += 1
            print(f"[PASS] {case_name}")

    if failures:
        for message in failures:
            print(message)
        print(f"Layer 3 evidence contract validation failed: {len(failures)} case(s) failed, {passes} passed.")
        return 1

    print(f"Layer 3 evidence contract validation passed: {passes} case(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
