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
    "binding_strength",
    "source_surface_type",
    "source_locator_text",
    "evidence_contains_exact_value",
    "review_display_text",
]

ROW_REVIEW_COLUMNS = [
    "paper_key",
    "final_formulation_id",
    "row_risk_level",
    "review_priority",
    "review_scope",
    "row_risk_reasons",
    "core_field_count",
    "core_fields_present",
    "core_fields_blank",
    "core_fields_high_risk",
    "core_fields_medium_risk",
    "binding_strength_summary",
    "assignment_path_summary",
    "suggested_review_focus",
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
    "missing_exact_value_evidence": ("high", "missing_exact_value_evidence", "Frozen non-blank value has no value-specific evidence text or cell containing the exact value."),
    "relation_path_missing": ("high", "relation_path_missing", "Relation-resolved value lacks a visible relation path."),
    "ambiguous_assignment": ("high", "ambiguity", "Assignment path is ambiguous."),
    "unresolved_table": ("high", "unsupported_value", "Table support is unresolved."),
    "unsupported_text": ("high", "unsupported_value", "Text support is unsupported."),
    "derived_without_direct_text": ("medium", "derived_value", "Derived value lacks direct text evidence but may have derivation provenance."),
    "source_surface_missing": ("critical", "source_surface_missing", "Required source surface is missing."),
    "authority_alias_conflict": ("critical", "authority_alias_conflict", "Authority alias conflict invalidates evidence interpretation until resolved."),
    "conflict": ("critical", "conflict", "Conflicting evidence or assignment detected."),
}

EE_MODELING_CORE_FIELDS = {
    "polymer_identity_final",
    "drug_name_value",
    "la_ga_ratio_value",
    "polymer_mw_kDa_value",
    "plga_mass_mg_value",
    "drug_feed_amount_text_value",
    "surfactant_name_value",
    "surfactant_concentration_text_value",
    "surfactant_concentration_value_value",
    "surfactant_concentration_unit_value",
    "organic_solvent_value",
    "organic_phase_volume_mL_value",
    "external_aqueous_phase_volume_mL_value",
    "size_nm_value",
    "pdi_value",
    "zeta_mV_value",
    "encapsulation_efficiency_percent_value",
    "loading_content_percent_value",
    "dl_percent_value",
}

FIELD_PROFILE_MAP = {
    "ee_modeling_core": EE_MODELING_CORE_FIELDS,
    "all": set(),
}


def _norm(value: Any) -> str:
    return str(value or "").strip()


def field_allowed(field_name: str, field_profile: str) -> bool:
    profile = FIELD_PROFILE_MAP.get(field_profile)
    if profile is None:
        raise ValueError(f"unknown field profile {field_profile!r}")
    if not profile:
        return True
    return field_name in profile


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
    assignment_path = _norm(pack.get("assignment_path"))
    binding_strength = _norm(pack.get("binding_strength"))
    source_locator = _norm(pack.get("source_locator_text"))
    if binding_strength in {"broad_anchor", "value_only", "weak_or_missing"}:
        level = "high"
        risk_type = "weak_binding_chain"
        reason = "Binding chain is weak, value-only, broad-anchor, or missing a legal field-level evidence anchor."
    elif binding_strength == "shared_context" or assignment_path in {"shared_method_context", "parent_inheritance"}:
        level = "medium" if source_locator else "high"
        risk_type = "shared_or_inherited_context"
        reason = (
            "Shared or inherited context is recorded; review only when the locator/scope is weak or this row is otherwise high impact."
            if source_locator
            else "Shared or inherited context lacks a concrete source locator and requires review."
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
        "assignment_path": assignment_path,
        "frozen_value_present": "yes" if frozen_value else "no",
        "binding_strength": binding_strength,
        "source_surface_type": _norm(pack.get("source_surface_type")),
        "source_locator_text": source_locator,
        "evidence_contains_exact_value": _norm(pack.get("evidence_contains_exact_value")),
        "review_display_text": _norm(pack.get("review_display_text")),
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


def summarize_counter(rows: list[dict[str, str]], field: str) -> str:
    counts = Counter(_norm(row.get(field)) or "<blank>" for row in rows)
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def build_row_review_queue_rows(field_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in field_rows:
        grouped[(_norm(row.get("paper_key")), _norm(row.get("final_formulation_id")))].append(row)

    priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    out: list[dict[str, str]] = []
    for (paper_key, final_id), rows in sorted(grouped.items()):
        levels = [row["risk_level"] for row in rows]
        row_level = highest_risk_level(levels)
        present = [row["field_name"] for row in rows if row["frozen_value_present"] == "yes"]
        blank = [row["field_name"] for row in rows if row["frozen_value_present"] != "yes"]
        high_fields = [row["field_name"] for row in rows if row["risk_level"] in {"critical", "high"}]
        medium_fields = [row["field_name"] for row in rows if row["risk_level"] == "medium"]
        risk_reasons = []
        for row in rows:
            if row["risk_level"] in {"critical", "high", "medium"}:
                risk_reasons.append(f"{row['field_name']}={row['risk_type']}")
        focus_fields = high_fields or medium_fields
        if not focus_fields and blank:
            focus_fields = blank[:5]
        out.append(
            {
                "paper_key": paper_key,
                "final_formulation_id": final_id,
                "row_risk_level": row_level,
                "review_priority": str(priority_order.get(row_level, 0)),
                "review_scope": "row",
                "row_risk_reasons": ";".join(risk_reasons[:12]),
                "core_field_count": str(len(rows)),
                "core_fields_present": ";".join(present),
                "core_fields_blank": ";".join(blank),
                "core_fields_high_risk": ";".join(high_fields),
                "core_fields_medium_risk": ";".join(medium_fields),
                "binding_strength_summary": summarize_counter(rows, "binding_strength"),
                "assignment_path_summary": summarize_counter(rows, "assignment_path"),
                "suggested_review_focus": ";".join(focus_fields[:8]),
            }
        )
    return sorted(
        out,
        key=lambda row: (-int(row["review_priority"]), row["paper_key"], row["final_formulation_id"]),
    )


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
        "- `evidence_binding_row_review_queue_v1.tsv`",
        "- `evidence_binding_formulation_risk_v1.tsv`",
        "- `evidence_binding_paper_risk_v1.tsv`",
        "- `evidence_binding_risk_assessment_metadata_v1.json`",
        "",
        "## Counts",
        f"- field_profile: {summary['field_profile']}",
        f"- input_pack_count: {summary['input_pack_count']}",
        f"- field_row_count: {summary['field_row_count']}",
        f"- row_review_queue_count: {summary['row_review_queue_count']}",
        f"- formulation_row_count: {summary['formulation_row_count']}",
        f"- paper_row_count: {summary['paper_row_count']}",
        "",
        "## Risk distribution",
    ]
    for key, value in sorted(summary["risk_level_distribution"].items()):
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_risk_assessment(*, pack_path: Path, out_dir: Path, field_profile: str = "ee_modeling_core") -> dict[str, Any]:
    pack_path = pack_path.resolve()
    out_dir = out_dir.resolve()
    packs = read_jsonl(pack_path)
    filtered_packs = [pack for pack in packs if field_allowed(_norm(pack.get("field_name")), field_profile)]
    field_rows = [assess_pack_risk(pack) for pack in filtered_packs]
    row_review_rows = build_row_review_queue_rows(field_rows)
    formulation_rows = rollup_risk(field_rows, group_field="final_formulation_id")
    paper_rows = rollup_risk(field_rows, group_field="paper_key")
    write_tsv(out_dir / "evidence_binding_field_risk_v1.tsv", field_rows, FIELD_RISK_COLUMNS)
    write_tsv(out_dir / "evidence_binding_row_review_queue_v1.tsv", row_review_rows, ROW_REVIEW_COLUMNS)
    write_tsv(out_dir / "evidence_binding_formulation_risk_v1.tsv", formulation_rows, ROLLUP_COLUMNS)
    write_tsv(out_dir / "evidence_binding_paper_risk_v1.tsv", paper_rows, ROLLUP_COLUMNS)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/stage5_benchmark/build_evidence_binding_risk_assessment_v1.py",
        "benchmark_valid": "no",
        "evidence_binding_pack_path": str(pack_path),
        "field_profile": field_profile,
        "input_pack_count": len(packs),
        "field_row_count": len(field_rows),
        "row_review_queue_count": len(row_review_rows),
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
    parser.add_argument(
        "--field-profile",
        choices=sorted(FIELD_PROFILE_MAP),
        default="ee_modeling_core",
        help="Field subset used for risk assessment. Default is the no-GT EE modeling core row-audit profile.",
    )
    args = parser.parse_args(argv)
    summary = run_risk_assessment(
        pack_path=args.evidence_binding_packs_jsonl,
        out_dir=args.out_dir,
        field_profile=args.field_profile,
    )
    print(f"field_row_count={summary['field_row_count']}")
    print(f"row_review_queue_count={summary['row_review_queue_count']}")
    print(f"formulation_row_count={summary['formulation_row_count']}")
    print(f"paper_row_count={summary['paper_row_count']}")
    for key, value in sorted(summary["risk_level_distribution"].items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
