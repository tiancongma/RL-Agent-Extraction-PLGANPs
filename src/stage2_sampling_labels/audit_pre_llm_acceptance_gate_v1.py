#!/usr/bin/env python3
"""Diagnostic-only pre-LLM acceptance gate for clean text and evidence selector.

This script materializes the repository's pre-live LLM acceptance standard.  It
consumes a per-paper TSV assembled from dryrun/audit artifacts and emits a
pass/hold report plus pass/hold manifests.  It does not call an LLM, does not
materialize Stage5 values, and does not compare benchmark performance.

The gate is intentionally about boundary diagnosis: user/source anchors may be
used by upstream visibility audits to reveal gaps, but this gate never forces
system outputs to equal those anchors and never treats anchors as runtime value
authority.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

PASS_STATUS = "pass_for_live_llm"
HOLD_STATUS = "hold_for_selector_or_cleantext_review"

READY_STRUCTURE_STATUSES = {
    "loaded",
    "consumed",
    "explicitly_not_applicable",
    "not_applicable",
    "not_available_with_reason",
}
READY_TABLE_CELL_STATUSES = {
    "consumed",
    "loaded",
    "explicitly_not_applicable",
    "not_applicable",
    "not_available_with_reason",
    "no_selected_tables",
}
READY_TABLE_AUTHORITY_STATUSES = {
    "backed_by_full_authority",
    "no_selected_tables",
    "not_applicable",
    "explicitly_not_applicable",
}

OUTPUT_FIELDNAMES = [
    "paper_key",
    "pre_llm_acceptance_status",
    "first_failure_boundary",
    "gate_reasons",
    "clean_text_status",
    "structure_sidecar_status",
    "table_cell_sidecar_status",
    "candidate_has_ee_or_loading_signal",
    "selected_has_ee_or_loading_signal",
    "candidate_has_preparation_core",
    "selected_has_preparation_core",
    "prompt_size_policy_status",
    "s2_3_ready_overall",
    "selector_authority_filter_violations",
    "selected_table_authority_status",
    "tail_noise_with_weak_ee_signal",
    "diagnostic_only",
    "benchmark_valid",
]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_yes(value: Any) -> bool:
    return _norm(value) in {"yes", "true", "1", "y"}


def _as_int(value: Any) -> int:
    try:
        return int(float(str(value or "0").strip()))
    except (TypeError, ValueError):
        return 0


def _first_failure(reasons: list[tuple[str, str]]) -> str:
    return reasons[0][0] if reasons else "none"


def evaluate_pre_llm_acceptance_row(row: dict[str, Any]) -> dict[str, str]:
    """Evaluate one paper row against the diagnostic pre-live acceptance gate."""
    reasons: list[tuple[str, str]] = []

    clean_status = _norm(row.get("clean_text_status")) or _norm(row.get("source_quality_status"))
    structure_status = _norm(row.get("structure_sidecar_status")) or _norm(row.get("stage1_structure_sidecar_status"))
    table_cell_status = _norm(row.get("table_cell_sidecar_status")) or _norm(row.get("stage1_cell_sidecar_status"))
    table_authority_status = _norm(row.get("selected_table_authority_status")) or _norm(row.get("table_authority_status"))

    if clean_status not in {"pass", "visible", "partial", "ok", "usable_for_dryrun"}:
        reasons.append(("stage1_clean_text_visibility", "clean_text_not_accepted"))

    if structure_status not in READY_STRUCTURE_STATUSES:
        reasons.append(("stage1_structure_sidecar_missing", "structure_sidecar_missing_or_silent"))

    if table_cell_status not in READY_TABLE_CELL_STATUSES:
        reasons.append(("stage1_table_cell_sidecar_missing", "table_cell_sidecar_missing_or_silent"))

    candidate_has_ee = _is_yes(row.get("candidate_has_ee_or_loading_signal"))
    selected_has_ee = _is_yes(row.get("selected_has_ee_or_loading_signal"))
    if candidate_has_ee and not selected_has_ee:
        reasons.append(("stage2_selector_missing_ee_or_loading_signal", "selected_evidence_missing_ee_or_loading_signal"))

    candidate_has_prep = _is_yes(row.get("candidate_has_preparation_core"))
    selected_has_prep = _is_yes(row.get("selected_has_preparation_core"))
    if candidate_has_prep and not selected_has_prep:
        reasons.append(("stage2_selector_missing_preparation_core", "missing_preparation_core"))

    if _norm(row.get("prompt_size_policy_status")) in {"oversized", "fail", "failed"}:
        reasons.append(("stage2_prompt_size_policy", "oversized_prompt"))

    s23_ready = _norm(row.get("s2_3_ready_overall"))
    if s23_ready and s23_ready not in {"pass", "ready", "yes", "ok"}:
        reasons.append(("stage2_prompt_readiness", "s2_3_not_ready"))

    if _as_int(row.get("selector_authority_filter_violations")) > 0:
        reasons.append(("stage2_selector_boundary", "selector_authority_filter_violation"))

    if table_authority_status and table_authority_status not in READY_TABLE_AUTHORITY_STATUSES:
        reasons.append(("stage2_table_authority_backing", "selected_table_without_full_authority_backing"))

    if _is_yes(row.get("tail_noise_with_weak_ee_signal")):
        reasons.append(("stage2_selector_noise_takeover", "tail_noise_with_weak_ee_signal"))

    # Deduplicate reason codes while preserving first-failure order.
    seen_reason_codes: set[str] = set()
    reason_codes: list[str] = []
    for _boundary, code in reasons:
        if code not in seen_reason_codes:
            seen_reason_codes.add(code)
            reason_codes.append(code)

    status = PASS_STATUS if not reasons else HOLD_STATUS
    output = {field: str(row.get(field, "")) for field in OUTPUT_FIELDNAMES}
    output.update(
        {
            "paper_key": str(row.get("paper_key") or row.get("key") or ""),
            "pre_llm_acceptance_status": status,
            "first_failure_boundary": _first_failure(reasons),
            "gate_reasons": ";".join(reason_codes),
            "clean_text_status": str(row.get("clean_text_status") or row.get("source_quality_status") or ""),
            "structure_sidecar_status": str(row.get("structure_sidecar_status") or row.get("stage1_structure_sidecar_status") or ""),
            "table_cell_sidecar_status": str(row.get("table_cell_sidecar_status") or row.get("stage1_cell_sidecar_status") or ""),
            "selected_table_authority_status": str(row.get("selected_table_authority_status") or row.get("table_authority_status") or ""),
            "diagnostic_only": "yes",
            "benchmark_valid": "no",
        }
    )
    return output


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _by_key(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("paper_key") or ""): row for row in rows if row.get("paper_key")}


def _table_authority_by_key(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = str(row.get("paper_key") or "")
        if key:
            grouped.setdefault(key, []).append(row)
    out: dict[str, str] = {}
    for key, paper_rows in grouped.items():
        if not paper_rows:
            continue
        has_failure = any(
            _norm(row.get("missing_rows_detected")) == "yes"
            or _norm(row.get("column_collapse_detected")) == "yes"
            for row in paper_rows
        )
        out[key] = "authority_warning" if has_failure else "backed_by_full_authority"
    return out


def build_acceptance_input_rows_from_audits(
    *,
    noise_gate_tsv: Path,
    preparation_audit_tsv: Path,
    s2_3_boundary_tsv: Path,
    table_authority_tsv: Path | None = None,
    structure_sidecar_status: str = "missing",
    table_cell_sidecar_status: str = "missing",
) -> list[dict[str, str]]:
    """Assemble gate input rows from existing no-live dryrun diagnostic artifacts."""
    noise_by_key = _by_key(_read_tsv(noise_gate_tsv))
    prep_by_key = _by_key(_read_tsv(preparation_audit_tsv))
    s23_by_key = _by_key(_read_tsv(s2_3_boundary_tsv))
    table_status = _table_authority_by_key(_read_tsv(table_authority_tsv)) if table_authority_tsv and table_authority_tsv.exists() else {}
    keys = sorted(set(noise_by_key) | set(prep_by_key) | set(s23_by_key))
    rows: list[dict[str, str]] = []
    for key in keys:
        noise = noise_by_key.get(key, {})
        prep = prep_by_key.get(key, {})
        s23 = s23_by_key.get(key, {})
        selected_ee_count = _as_int(noise.get("ee_or_loading_selected_block_count"))
        gate_reasons = _norm(noise.get("gate_reasons"))
        candidate_has_ee = "yes" if selected_ee_count > 0 or "selected_evidence_missing_ee_or_loading_signal" in gate_reasons else "no"
        selected_table_count = _as_int(noise.get("table_blocks_selected_count"))
        table_authority = "no_selected_tables"
        if selected_table_count > 0:
            table_authority = table_status.get(key, "missing_full_table_authority")
        rows.append(
            {
                "paper_key": key,
                "clean_text_status": prep.get("source_quality_status") or "unknown",
                "structure_sidecar_status": structure_sidecar_status,
                "table_cell_sidecar_status": table_cell_sidecar_status,
                "candidate_has_ee_or_loading_signal": candidate_has_ee,
                "selected_has_ee_or_loading_signal": "yes" if selected_ee_count > 0 else "no",
                "candidate_has_preparation_core": prep.get("candidate_has_preparation_core") or "no",
                "selected_has_preparation_core": prep.get("selected_has_preparation_core") or "no",
                "prompt_size_policy_status": s23.get("prompt_size_policy_status") or noise.get("prompt_size_policy_status") or "unknown",
                "s2_3_ready_overall": s23.get("s2_3_ready_overall") or "unknown",
                "selector_authority_filter_violations": "0",
                "selected_table_authority_status": table_authority,
                "tail_noise_with_weak_ee_signal": "yes" if "tail_noise_with_weak_ee_signal" in gate_reasons else "no",
            }
        )
    return rows


def _write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_pre_llm_acceptance_gate(*, input_tsv: Path, out_dir: Path) -> None:
    """Write diagnostic pass/hold reports for a per-paper acceptance input TSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_pre_llm_acceptance_row(row) for row in _read_tsv(input_tsv)]
    _write_pre_llm_outputs(rows=rows, out_dir=out_dir, source_description=str(input_tsv))


def write_pre_llm_acceptance_gate_from_audits(
    *,
    noise_gate_tsv: Path,
    preparation_audit_tsv: Path,
    s2_3_boundary_tsv: Path,
    out_dir: Path,
    table_authority_tsv: Path | None = None,
    structure_sidecar_status: str = "missing",
    table_cell_sidecar_status: str = "missing",
) -> None:
    """Build gate inputs from existing dryrun audits and write acceptance outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows = build_acceptance_input_rows_from_audits(
        noise_gate_tsv=noise_gate_tsv,
        preparation_audit_tsv=preparation_audit_tsv,
        s2_3_boundary_tsv=s2_3_boundary_tsv,
        table_authority_tsv=table_authority_tsv,
        structure_sidecar_status=structure_sidecar_status,
        table_cell_sidecar_status=table_cell_sidecar_status,
    )
    _write_tsv(out_dir / "pre_llm_acceptance_input_v1.tsv", input_rows, OUTPUT_FIELDNAMES[0:1] + OUTPUT_FIELDNAMES[4:16])
    rows = [evaluate_pre_llm_acceptance_row(row) for row in input_rows]
    _write_pre_llm_outputs(rows=rows, out_dir=out_dir, source_description="assembled_from_dryrun_audits")


def _write_pre_llm_outputs(*, rows: list[dict[str, str]], out_dir: Path, source_description: str) -> None:
    """Write common output files for evaluated acceptance rows."""
    _write_tsv(out_dir / "pre_llm_acceptance_gate_v1.tsv", rows, OUTPUT_FIELDNAMES)
    pass_rows = [row for row in rows if row["pre_llm_acceptance_status"] == PASS_STATUS]
    hold_rows = [row for row in rows if row["pre_llm_acceptance_status"] == HOLD_STATUS]
    _write_tsv(out_dir / "pre_llm_acceptance_pass_manifest_v1.tsv", pass_rows, OUTPUT_FIELDNAMES)
    _write_tsv(out_dir / "pre_llm_acceptance_hold_manifest_v1.tsv", hold_rows, OUTPUT_FIELDNAMES)

    first_failure_counts: dict[str, int] = {}
    for row in rows:
        first_failure_counts[row["first_failure_boundary"]] = first_failure_counts.get(row["first_failure_boundary"], 0) + 1
    with (out_dir / "pre_llm_acceptance_gate_summary_v1.tsv").open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"paper_count\t{len(rows)}\n")
        handle.write(f"pass_for_live_llm\t{len(pass_rows)}\n")
        handle.write(f"hold_for_selector_or_cleantext_review\t{len(hold_rows)}\n")
        for boundary, count in sorted(first_failure_counts.items()):
            handle.write(f"first_failure.{boundary}\t{count}\n")

    generated_at = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "generated_by": "audit_pre_llm_acceptance_gate_v1.py",
        "generated_at": generated_at,
        "diagnostic_only": True,
        "benchmark_valid": "no",
        "input_tsv": source_description,
        "semantics": "pre-live clean text / structure / table authority / evidence selector acceptance gate; no LLM calls; source anchors are diagnostic gap anchors only, not runtime authority",
    }
    (out_dir / "pre_llm_acceptance_gate_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# pre_llm_acceptance_gate_diagnostic\n\n"
        "- diagnostic_only: yes\n"
        "- benchmark_valid: no\n"
        f"- generated_at: {generated_at}\n"
        f"- input_source: {source_description}\n"
        "- boundary: pre-live LLM acceptance over clean text, unified Stage1 sidecars, table authority, and selector evidence only.\n"
        "- source_anchor_policy: uploaded/source excerpts are audit anchors for finding system gaps, not runtime extraction authority and not equality targets.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, help="Per-paper acceptance inputs assembled from dryrun/audit artifacts.")
    parser.add_argument("--noise-gate-tsv", type=Path, help="pre_llm_ee_noise_gate_v1.tsv from a no-live dryrun.")
    parser.add_argument("--preparation-audit-tsv", type=Path, help="preparation_evidence_sufficiency_audit_v1.tsv from a no-live dryrun.")
    parser.add_argument("--s2-3-boundary-tsv", type=Path, help="s2_3_boundary_validation.tsv from a no-live dryrun.")
    parser.add_argument("--table-authority-tsv", type=Path, help="Optional table_authority_validation_v1.tsv.")
    parser.add_argument("--structure-sidecar-status", default="missing")
    parser.add_argument("--table-cell-sidecar-status", default="missing")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input_tsv:
        write_pre_llm_acceptance_gate(input_tsv=args.input_tsv, out_dir=args.out_dir)
    else:
        required = [args.noise_gate_tsv, args.preparation_audit_tsv, args.s2_3_boundary_tsv]
        if any(path is None for path in required):
            parser.error("provide --input-tsv or all of --noise-gate-tsv, --preparation-audit-tsv, and --s2-3-boundary-tsv")
        write_pre_llm_acceptance_gate_from_audits(
            noise_gate_tsv=args.noise_gate_tsv,
            preparation_audit_tsv=args.preparation_audit_tsv,
            s2_3_boundary_tsv=args.s2_3_boundary_tsv,
            table_authority_tsv=args.table_authority_tsv,
            structure_sidecar_status=args.structure_sidecar_status,
            table_cell_sidecar_status=args.table_cell_sidecar_status,
            out_dir=args.out_dir,
        )
    print(f"wrote pre-LLM acceptance gate to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
