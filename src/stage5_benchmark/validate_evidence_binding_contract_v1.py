#!/usr/bin/env python3
from __future__ import annotations

"""
Validate Evidence Binding contract hardening while preserving Layer3 evidence handoff golden cases.

This Phase-8 validator intentionally reads the existing Layer3 evidence handoff
golden cases first, then optionally validates Evidence Binding Pack status/path
cases. It is diagnostic-only and does not modify pipeline artifacts.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from src.stage5_benchmark import validate_layer3_evidence_contract_v1 as layer3_validator
    from src.utils.paths import DOCS_DIR
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage5_benchmark import validate_layer3_evidence_contract_v1 as layer3_validator
    from src.utils.paths import DOCS_DIR

DEFAULT_LAYER3_GOLDEN_CASES = DOCS_DIR / "methods" / "layer3_evidence_handoff_golden_cases_v1.tsv"
DEFAULT_BINDING_GOLDEN_CASES = DOCS_DIR / "methods" / "evidence_binding_golden_cases_v1.tsv"


def _norm(value: Any) -> str:
    return str(value or "").strip()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_layer3_cases(seed_tsv: Path, golden_cases_tsv: Path) -> tuple[int, list[str]]:
    seed_rows = layer3_validator.read_tsv_rows(seed_tsv.resolve())
    golden_cases = layer3_validator.read_tsv_rows(golden_cases_tsv.resolve())
    row_index = {layer3_validator.row_key(row): row for row in seed_rows}
    passes = 0
    failures: list[str] = []
    for case in golden_cases:
        case_name = layer3_validator.normalize_text(case.get("case_id")) or "unnamed_layer3_case"
        lookup_key = (
            layer3_validator.normalize_text(case.get("paper_key")),
            layer3_validator.normalize_text(case.get("formulation_id")),
            layer3_validator.normalize_text(case.get("field_name")),
        )
        seed_row = row_index.get(lookup_key)
        if seed_row is None:
            failures.append(f"[FAIL] {case_name}: missing seed row for key={lookup_key}")
            continue
        errors = layer3_validator.validate_case(case, seed_row)
        if errors:
            failures.append(f"[FAIL] {case_name}: " + "; ".join(errors))
        else:
            passes += 1
    return passes, failures


def binding_pack_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (_norm(row.get("paper_key")), _norm(row.get("final_formulation_id")), _norm(row.get("field_name")))


def validate_binding_cases(packs_jsonl: Path | None, golden_cases_tsv: Path | None) -> tuple[int, list[str]]:
    if packs_jsonl is None or golden_cases_tsv is None or not golden_cases_tsv.exists():
        return 0, []
    pack_index = {binding_pack_key(row): row for row in read_jsonl(packs_jsonl)}
    cases = read_tsv_rows(golden_cases_tsv)
    passes = 0
    failures: list[str] = []
    for case in cases:
        case_name = _norm(case.get("case_id")) or "unnamed_binding_case"
        lookup_key = (_norm(case.get("paper_key")), _norm(case.get("final_formulation_id")), _norm(case.get("field_name")))
        pack = pack_index.get(lookup_key)
        if pack is None:
            failures.append(f"[FAIL] {case_name}: missing binding pack for key={lookup_key}")
            continue
        errors: list[str] = []
        expected_status = _norm(case.get("expected_binding_status"))
        if expected_status and _norm(pack.get("binding_status")) != expected_status:
            errors.append(f"expected binding_status={expected_status!r}, got {_norm(pack.get('binding_status'))!r}")
        expected_path = _norm(case.get("expected_assignment_path"))
        if expected_path and _norm(pack.get("assignment_path")) != expected_path:
            errors.append(f"expected assignment_path={expected_path!r}, got {_norm(pack.get('assignment_path'))!r}")
        forbidden_status = _norm(case.get("forbidden_binding_status"))
        if forbidden_status and _norm(pack.get("binding_status")) == forbidden_status:
            errors.append(f"forbidden binding_status {forbidden_status!r} was present")
        if errors:
            failures.append(f"[FAIL] {case_name}: " + "; ".join(errors))
        else:
            passes += 1
    return passes, failures


def run_validation(
    *,
    seed_tsv: Path,
    layer3_golden_cases_tsv: Path = DEFAULT_LAYER3_GOLDEN_CASES,
    evidence_binding_packs_jsonl: Path | None = None,
    evidence_binding_golden_cases_tsv: Path | None = DEFAULT_BINDING_GOLDEN_CASES,
) -> dict[str, int | list[str]]:
    layer3_passed, layer3_failures = validate_layer3_cases(seed_tsv, layer3_golden_cases_tsv)
    binding_passed, binding_failures = validate_binding_cases(
        evidence_binding_packs_jsonl,
        evidence_binding_golden_cases_tsv,
    )
    failures = layer3_failures + binding_failures
    return {
        "layer3_passed": layer3_passed,
        "binding_passed": binding_passed,
        "failed": len(failures),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Layer3 + Evidence Binding contract golden cases.")
    parser.add_argument("--seed-tsv", type=Path, required=True)
    parser.add_argument("--layer3-golden-cases-tsv", type=Path, default=DEFAULT_LAYER3_GOLDEN_CASES)
    parser.add_argument("--evidence-binding-packs-jsonl", type=Path, default=None)
    parser.add_argument("--evidence-binding-golden-cases-tsv", type=Path, default=DEFAULT_BINDING_GOLDEN_CASES)
    args = parser.parse_args(argv)
    result = run_validation(
        seed_tsv=args.seed_tsv,
        layer3_golden_cases_tsv=args.layer3_golden_cases_tsv,
        evidence_binding_packs_jsonl=args.evidence_binding_packs_jsonl,
        evidence_binding_golden_cases_tsv=args.evidence_binding_golden_cases_tsv,
    )
    for failure in result["failures"]:
        print(failure)
    print(
        f"Evidence Binding contract validation: layer3_passed={result['layer3_passed']} "
        f"binding_passed={result['binding_passed']} failed={result['failed']}"
    )
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
