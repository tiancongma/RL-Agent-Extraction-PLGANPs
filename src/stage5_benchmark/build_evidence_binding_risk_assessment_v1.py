#!/usr/bin/env python3
from __future__ import annotations

"""
Build Evidence Binding risk assessment from frozen Evidence Binding Packs.

This Phase-6 script consumes only pack JSONL plus optional already-materialized
review sidecars in future extensions. It does not re-resolve source evidence,
create rows/values, mutate packs, or render workbooks.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RISK_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

FIELD_RISK_COLUMNS = [
    "paper_key",
    "final_formulation_id",
    "field_name",
    "risk_level",
    "risk_type",
    "source_of_flag",
    "reason",
    "evidence_status",
    "assignment_path",
    "frozen_value_present",
]

ROLLUP_COLUMNS = [
    "group_key",
    "risk_level",
    "risk_type_summary",
    "field_count",
    "critical_count",
    "high_count",
    "medium_count",
    "low_count",
]

STATUS_RISK_MAP: dict[str, tuple[str, str, str]] = {
    "direct_supported": ("low", "supported", "Direct support recorded in frozen binding pack."),
    "relation_supported": ("low", "supported", "Stage3 relation path is visible in frozen binding pack."),
    "derived_supported": ("low", "derived_value", "Derived value has recorded provenance in frozen binding pack."),
    "role_tolerant_supported": ("low", "supported", "Role-tolerant support is recorded in frozen binding pack."),
    "blank_value": ("low", "blank_value", "Frozen final field is blank; no evidence support required by risk script."),
    "identity_only_match": ("low", "identity_only_match", "Identity fields are row identity support, not field-value evidence."),
    "raw_value_supported_normalization_pending": ("medium", "unit_or_normalization_only", "Raw value support exists but normalization remains pending."),
    "normalization_pending": ("medium", "unit_or_normalization_only", "Normalization remains pending."),
    "coded_value_supported_decode_pending": ("medium", "coded_value_decode_pending", "DOE/factor code support exists but decoded physical value remains pending."),
    "value_only_match": ("high", "unsupported_value", "Value-only match is not support evidence."),
    "missing_evidence_anchor": ("high", "missing_evidence_anchor", "Frozen non-blank value lacks a legal evidence anchor in the pack."),
    "relation_path_missing": ("high", "relation_path_missing", "Relation-resolved value lacks a visible relation path."),
    "ambiguous_assignment": ("high", "ambiguity", "Assignment path is ambiguous."),
    "unresolved_table": ("high", "unsupported_value", "Table support is unresolved."),
    "unsupported_text": ("high", "unsupported_value", "Text support is unsupported."),
    "derived_without_direct_text": ("medium", "derived_value", "Derived value lacks direct text evidence but may have derivation provenance."),
    "source_surface_missing": ("critical", "source_surface_missing", "Required source surface is missing."),
    "authority_alias_conflict": ("critical", "authority_alias_conflict", "Authority alias conflict invalidates evidence interpretation until resolved."),
    "conflict": ("critical", "conflict", "Conflicting evidence or assignment detected."),
}


def _norm(value: Any) -> str:
    return str(value or "").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def assess_pack_risk(pack: dict[str, Any]) -> dict[str, str]:
    status = _norm(pack.get("binding_status")) or "source_surface_missing"
    level, risk_type, reason = STATUS_RISK_MAP.get(
        status,
        ("high", "unknown_binding_status", f"Unknown binding status {status!r} requires review."),
    )
    frozen_value = _norm(pack.get("frozen_value"))
    return {
        "paper_key": _norm(pack.get("paper_key")),
        "final_formulation_id": _norm(pack.get("final_formulation_id")),
        "field_name": _norm(pack.get("field_name")),
        "risk_level": level,
        "risk_type": risk_type,
        "source_of_flag": "evidence_binding_pack",
        "reason": reason,
        "evidence_status": status,
        "assignment_path": _norm(pack.get("assignment_path")),
        "frozen_value_present": "yes" if frozen_value else "no",
    }


def highest_risk_level(levels: list[str]) -> str:
    if not levels:
        return "low"
    return max(levels, key=lambda item: RISK_LEVEL_ORDER.get(item, 99))


def rollup_risk(rows: list[dict[str, str]], *, group_field: str) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_norm(row.get(group_field))].append(row)
    out: list[dict[str, str]] = []
    for group_key, group_rows in sorted(grouped.items()):
        levels = [row["risk_level"] for row in group_rows]
        level = highest_risk_level(levels)
        risk_counts = Counter(row["risk_type"] for row in group_rows if row.get("risk_type"))
        level_counts = Counter(levels)
        out.append(
            {
                "group_key": group_key,
                "risk_level": level,
                "risk_type_summary": ";".join(f"{key}:{value}" for key, value in sorted(risk_counts.items())),
                "field_count": str(len(group_rows)),
                "critical_count": str(level_counts.get("critical", 0)),
                "high_count": str(level_counts.get("high", 0)),
                "medium_count": str(level_counts.get("medium", 0)),
                "low_count": str(level_counts.get("low", 0)),
            }
        )
    return out


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_run_context(path: Path, *, pack_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RUN_CONTEXT",
        "",
        "## Run purpose",
        "Build risk assessment from frozen Evidence Binding Packs.",
        "",
        "## Run type",
        "Diagnostic-only risk sidecar. Not benchmark-valid final output.",
        "",
        "## Boundary",
        "This script consumes frozen packs only; it does not re-resolve evidence, create rows, create values, mutate packs, or render workbooks.",
        "",
        "## Starting inputs",
        f"- evidence_binding_pack_path: `{pack_path}`",
        "",
        "## Outputs",
        "- `evidence_binding_field_risk_v1.tsv`",
        "- `evidence_binding_formulation_risk_v1.tsv`",
        "- `evidence_binding_paper_risk_v1.tsv`",
        "- `evidence_binding_risk_assessment_metadata_v1.json`",
        "",
        "## Counts",
        f"- field_row_count: {summary['field_row_count']}",
        f"- formulation_row_count: {summary['formulation_row_count']}",
        f"- paper_row_count: {summary['paper_row_count']}",
        "",
        "## Risk distribution",
    ]
    for key, value in sorted(summary["risk_level_distribution"].items()):
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_risk_assessment(*, pack_path: Path, out_dir: Path) -> dict[str, Any]:
    pack_path = pack_path.resolve()
    out_dir = out_dir.resolve()
    packs = read_jsonl(pack_path)
    field_rows = [assess_pack_risk(pack) for pack in packs]
    formulation_rows = rollup_risk(field_rows, group_field="final_formulation_id")
    paper_rows = rollup_risk(field_rows, group_field="paper_key")
    write_tsv(out_dir / "evidence_binding_field_risk_v1.tsv", field_rows, FIELD_RISK_COLUMNS)
    write_tsv(out_dir / "evidence_binding_formulation_risk_v1.tsv", formulation_rows, ROLLUP_COLUMNS)
    write_tsv(out_dir / "evidence_binding_paper_risk_v1.tsv", paper_rows, ROLLUP_COLUMNS)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/stage5_benchmark/build_evidence_binding_risk_assessment_v1.py",
        "benchmark_valid": "no",
        "evidence_binding_pack_path": str(pack_path),
        "field_row_count": len(field_rows),
        "formulation_row_count": len(formulation_rows),
        "paper_row_count": len(paper_rows),
        "risk_level_distribution": dict(Counter(row["risk_level"] for row in field_rows)),
        "risk_type_distribution": dict(Counter(row["risk_type"] for row in field_rows)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence_binding_risk_assessment_metadata_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_run_context(out_dir / "RUN_CONTEXT.md", pack_path=pack_path, summary=summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Evidence Binding risk assessment from frozen packs.")
    parser.add_argument("--evidence-binding-packs-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_risk_assessment(pack_path=args.evidence_binding_packs_jsonl, out_dir=args.out_dir)
    print(f"field_row_count={summary['field_row_count']}")
    print(f"formulation_row_count={summary['formulation_row_count']}")
    print(f"paper_row_count={summary['paper_row_count']}")
    for key, value in sorted(summary["risk_level_distribution"].items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
