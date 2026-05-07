#!/usr/bin/env python3
"""Unified Stage1/Stage2 visibility boundary classifier diagnostic.

Combines clean-text, table-authority, selector recall, and prompt semantic
adequacy audits into a first-failure report. This is diagnostic-only and does
not evaluate benchmark performance or materialize Stage5 values.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def classify_visibility_boundary(
    *,
    cleantext_visibility: str,
    table_authority_visibility: str,
    selector_selected: bool,
    prompt_adequate: bool,
    selector_registry_retained: bool = True,
) -> dict[str, str]:
    clean = (cleantext_visibility or "absent").lower()
    table = (table_authority_visibility or "absent").lower()
    if clean in {"absent", "no_checkable_fragments"}:
        first = "stage1_clean_text_visibility"
    elif table in {"absent", "no_checkable_fragments"}:
        first = "stage1_table_authority_visibility"
    elif not selector_registry_retained:
        first = "stage2_selector_boundary"
    elif not prompt_adequate:
        first = "stage2_prompt_summary_semantic_adequacy"
    else:
        first = "none"
    selector_status = "pass" if selector_registry_retained else "fail"
    return {
        "visibility_boundary_status": "pass" if first == "none" else "fail",
        "first_failure_layer": first,
        "selector_ranker_not_filter_status": selector_status,
        "materialization_allowed": "no",
        "diagnostic_only": "yes",
        "benchmark_valid": "no",
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _by_key(rows: list[dict[str, str]], key_field: str = "paper_key") -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = row.get(key_field, "")
        if key:
            out.setdefault(key, []).append(row)
    return out


def write_boundary_classifier_report(
    *,
    cleantext_tsv: Path,
    table_tsv: Path,
    selector_tsv: Path,
    prompt_tsv: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_by_key = {row["paper_key"]: row for row in _read_tsv(cleantext_tsv) if row.get("paper_key")}
    table_by_key = {row["paper_key"]: row for row in _read_tsv(table_tsv) if row.get("paper_key")}
    selector_by_key = _by_key(_read_tsv(selector_tsv))
    prompt_by_key = _by_key(_read_tsv(prompt_tsv))
    keys = sorted(set(clean_by_key) | set(table_by_key) | set(selector_by_key) | set(prompt_by_key))
    rows: list[dict[str, str]] = []
    for key in keys:
        selector_rows = selector_by_key.get(key, [])
        selected = any(r.get("selected_for_prompt") == "yes" for r in selector_rows)
        selector_violation = any(r.get("selector_is_authority_filter_violation") == "yes" for r in selector_rows)
        retained = not selector_violation
        adequate = any(r.get("semantic_adequacy") == "adequate" for r in prompt_by_key.get(key, []))
        classified = classify_visibility_boundary(
            cleantext_visibility=clean_by_key.get(key, {}).get("anchor_visibility", "absent"),
            table_authority_visibility=table_by_key.get(key, {}).get("anchor_visibility", "absent"),
            selector_selected=selected,
            selector_registry_retained=retained,
            prompt_adequate=adequate,
        )
        rows.append({
            "paper_key": key,
            "cleantext_visibility": clean_by_key.get(key, {}).get("anchor_visibility", "absent"),
            "table_authority_visibility": table_by_key.get(key, {}).get("anchor_visibility", "absent"),
            "selector_selected_any": "yes" if selected else "no",
            "selector_registry_retained_all": "yes" if retained else "no",
            "prompt_adequate_any": "yes" if adequate else "no",
            **classified,
        })
    fieldnames = [
        "paper_key", "cleantext_visibility", "table_authority_visibility", "selector_selected_any",
        "selector_registry_retained_all", "prompt_adequate_any", "visibility_boundary_status",
        "first_failure_layer", "selector_ranker_not_filter_status", "materialization_allowed",
        "diagnostic_only", "benchmark_valid",
    ]
    with (out_dir / "stage1_stage2_visibility_boundary_classifier_v1.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["first_failure_layer"]] = counts.get(row["first_failure_layer"], 0) + 1
    with (out_dir / "stage1_stage2_visibility_boundary_classifier_summary_v1.tsv").open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"paper_count\t{len(rows)}\n")
        for reason, count in sorted(counts.items()):
            handle.write(f"first_failure.{reason}\t{count}\n")
    generated_at = datetime.now().isoformat(timespec="seconds")
    (out_dir / "stage1_stage2_visibility_boundary_classifier_metadata.json").write_text(json.dumps({
        "generated_by": "audit_stage1_stage2_visibility_boundary_classifier_v1.py",
        "generated_at": generated_at,
        "diagnostic_only": True,
        "benchmark_valid": "no",
        "inputs": [str(cleantext_tsv), str(table_tsv), str(selector_tsv), str(prompt_tsv)],
    }, indent=2) + "\n", encoding="utf-8")
    (out_dir / "RUN_CONTEXT.md").write_text(
        "# stage1_stage2_visibility_boundary_classifier_diagnostic\n\n"
        "- diagnostic_only: yes\n- benchmark_valid: no\n"
        f"- generated_at: {generated_at}\n"
        "- boundary: first-failure classifier across visibility diagnostics only; no Stage5 raw-source mining/bypass.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleantext-tsv", type=Path, required=True)
    parser.add_argument("--table-tsv", type=Path, required=True)
    parser.add_argument("--selector-tsv", type=Path, required=True)
    parser.add_argument("--prompt-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    write_boundary_classifier_report(cleantext_tsv=args.cleantext_tsv, table_tsv=args.table_tsv, selector_tsv=args.selector_tsv, prompt_tsv=args.prompt_tsv, out_dir=args.out_dir)
    print(f"wrote boundary classifier report to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
