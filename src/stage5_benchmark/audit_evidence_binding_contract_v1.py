#!/usr/bin/env python3
from __future__ import annotations

"""
Run-scoped Evidence Binding contract audit.

This Phase-1 diagnostic script consumes the Phase-0 authority manifest and
proves which current artifacts/code surfaces expose direct evidence, broad
anchors, Stage3 relation provenance, legacy fallback evidence snippets, or no
consumed evidence. It does not build binding packs, assign risk, create rows, or
create values.
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT


AUDIT_FIELDS = [
    "surface_name",
    "component",
    "surface_path",
    "surface_status",
    "evidence_logic_class",
    "proof_kind",
    "proof_detail",
    "row_count",
    "column_count",
    "notes",
]

CLASS_RULES = [
    (
        "direct_evidence",
        {"evidence_text"},
        "Direct evidence text is an explicit support surface when kept separate from anchors.",
    ),
    (
        "broad_anchor",
        {"evidence_anchor_text"},
        "Anchor text is a locator/review aid and must not be promoted to direct support.",
    ),
    (
        "stage3_relation_provenance",
        {"relation_resolution_rule"},
        "Relation rule columns expose Stage3 assignment provenance.",
    ),
    (
        "stage3_relation_provenance",
        {"relation_resolution_source_ids"},
        "Relation source ids expose Stage3 relation path hints.",
    ),
    (
        "stage3_relation_provenance",
        {"resolution_rule"},
        "Resolved relation field rule exposes Stage3 assignment provenance.",
    ),
    (
        "stage3_relation_provenance",
        {"source_relation_row_ids"},
        "Resolved relation field source ids expose relation path hints.",
    ),
    (
        "broad_anchor",
        {"supporting_evidence_refs"},
        "Stage2 supporting evidence refs are anchors/locators, not direct value support.",
    ),
    (
        "legacy_fallback",
        {"evidence_snippet"},
        "Legacy risk/cross-audit snippet collapses evidence into display text.",
    ),
    (
        "legacy_fallback",
        {"evidence_status"},
        "Legacy risk/cross-audit evidence status is not a full binding chain.",
    ),
    (
        "legacy_fallback",
        {"evidence_status_detail"},
        "Compare/workbook status detail is a flattened legacy evidence classification, not a binding chain.",
    ),
    (
        "legacy_fallback",
        {"source_paths"},
        "Legacy source path list is useful but does not itself establish field support.",
    ),
]

STATIC_CODE_SURFACES = [
    (
        "layer3_workbook_builder_v1_source",
        "Layer3 workbook builder source",
        PROJECT_ROOT / "src/stage5_benchmark/build_field_gt_review_workbook_v1.py",
        True,
    ),
    (
        "layer3_evidence_contract_validator_source",
        "Layer3 evidence handoff validator source",
        PROJECT_ROOT / "src/stage5_benchmark/validate_layer3_evidence_contract_v1.py",
        True,
    ),
    (
        "layer3_evidence_handoff_golden_cases",
        "Layer3 golden cases",
        PROJECT_ROOT / "docs/methods/layer3_evidence_handoff_golden_cases_v1.tsv",
        True,
    ),
]

MANIFEST_SURFACE_NAMES = {
    "layer3_risk_review_queue": ("Layer3 risk queue / compare queue", False),
    "layer3_compare_cells": ("Layer3 compare cells", False),
    "stage3_relation_records": ("Stage3 relation records", True),
    "stage3_resolved_relation_fields": ("Stage3 resolved relation fields", True),
    "frozen_final_table": ("Frozen final table", True),
    "final_output_decision_trace": ("Final output decision trace", True),
    "stage2_compatibility_tsv": ("Stage2 compatibility TSV", True),
    "stage2_projection_trace": ("Stage2 projection trace", False),
}


def _read_tsv_header_and_count(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        count = sum(1 for _ in reader)
    return fields, count


def _row(
    *,
    surface_name: str,
    component: str,
    path: Path,
    status: str,
    logic_class: str,
    proof_kind: str,
    proof_detail: str,
    row_count: int = 0,
    column_count: int = 0,
    notes: str = "",
) -> dict[str, str]:
    surface_path = "" if proof_kind == "path_supplied" and proof_detail == "empty" else str(path)
    return {
        "surface_name": surface_name,
        "component": component,
        "surface_path": surface_path,
        "surface_status": status,
        "evidence_logic_class": logic_class,
        "proof_kind": proof_kind,
        "proof_detail": proof_detail,
        "row_count": str(row_count),
        "column_count": str(column_count),
        "notes": notes,
    }


def audit_tsv_surface(
    *,
    surface_name: str,
    path: Path,
    expected_component: str,
    required: bool = True,
) -> list[dict[str, str]]:
    path = Path(path)
    if not str(path):
        return [
            _row(
                surface_name=surface_name,
                component=expected_component,
                path=path,
                status="missing_required" if required else "missing_optional",
                logic_class="source_surface_missing" if required else "not_consumed",
                proof_kind="path_supplied",
                proof_detail="empty",
                notes="Required source path is empty" if required else "Optional source path is empty for this run",
            )
        ]
    if not path.exists():
        return [
            _row(
                surface_name=surface_name,
                component=expected_component,
                path=path,
                status="missing_required" if required else "missing_optional",
                logic_class="source_surface_missing" if required else "not_consumed",
                proof_kind="path_exists",
                proof_detail="missing",
                notes="Required source is missing" if required else "Optional source is absent for this run",
            )
        ]
    if path.suffix.lower() != ".tsv":
        return [
            _row(
                surface_name=surface_name,
                component=expected_component,
                path=path,
                status="present",
                logic_class="not_consumed",
                proof_kind="file_type",
                proof_detail=f"non_tsv:{path.suffix}",
                notes="Present non-TSV surface; not column-classified by Phase1 audit.",
            )
        ]
    fields, row_count = _read_tsv_header_and_count(path)
    field_set = set(fields)
    rows: list[dict[str, str]] = []
    for logic_class, required_columns, detail in CLASS_RULES:
        hits = sorted(field_set & required_columns)
        if hits:
            rows.append(
                _row(
                    surface_name=surface_name,
                    component=expected_component,
                    path=path,
                    status="present",
                    logic_class=logic_class,
                    proof_kind="columns",
                    proof_detail=",".join(hits),
                    row_count=row_count,
                    column_count=len(fields),
                    notes=detail,
                )
            )
    if not rows:
        rows.append(
            _row(
                surface_name=surface_name,
                component=expected_component,
                path=path,
                status="present",
                logic_class="not_consumed",
                proof_kind="columns",
                proof_detail=",".join(fields[:20]),
                row_count=row_count,
                column_count=len(fields),
                notes="No recognized evidence-handoff columns in this surface.",
            )
        )
    return rows


def audit_text_surface(
    *,
    surface_name: str,
    path: Path,
    expected_component: str,
    required: bool = True,
) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return [
            _row(
                surface_name=surface_name,
                component=expected_component,
                path=path,
                status="missing_required" if required else "missing_optional",
                logic_class="source_surface_missing" if required else "not_consumed",
                proof_kind="path_exists",
                proof_detail="missing",
            )
        ]
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    probes = [
        ("direct_evidence", "evidence_text"),
        ("broad_anchor", "evidence_anchor_text"),
        ("stage3_relation_provenance", "relation_resolution_rule"),
        ("legacy_fallback", "evidence_snippet"),
        ("legacy_fallback", "source_paths"),
    ]
    for logic_class, token in probes:
        if token in text:
            rows.append(
                _row(
                    surface_name=surface_name,
                    component=expected_component,
                    path=path,
                    status="present",
                    logic_class=logic_class,
                    proof_kind="source_token",
                    proof_detail=token,
                    row_count=text.count("\n") + 1,
                    notes="Static source/document token proves this component references the evidence surface.",
                )
            )
    if not rows:
        rows.append(
            _row(
                surface_name=surface_name,
                component=expected_component,
                path=path,
                status="present",
                logic_class="not_consumed",
                proof_kind="source_token",
                proof_detail="no_recognized_evidence_tokens",
                row_count=text.count("\n") + 1,
            )
        )
    return rows


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "artifacts" not in payload or not isinstance(payload["artifacts"], list):
        raise ValueError(f"Authority manifest missing artifacts list: {path}")
    return payload


def rows_from_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_name = {str(row.get("semantic_name", "")): row for row in manifest.get("artifacts", [])}
    for semantic_name, (component, required) in MANIFEST_SURFACE_NAMES.items():
        artifact = by_name.get(semantic_name) or {}
        selected_path = str(artifact.get("selected_path") or "")
        if not selected_path:
            rows.append(
                _row(
                    surface_name=semantic_name,
                    component=component,
                    path=Path(""),
                    status="missing_required" if required else "missing_optional",
                    logic_class="source_surface_missing" if required else "not_consumed",
                    proof_kind="path_supplied",
                    proof_detail="empty",
                    notes="Required authority path is empty" if required else "Optional authority path is empty for this run",
                )
            )
            continue
        rows.extend(
            audit_tsv_surface(
                surface_name=semantic_name,
                path=Path(selected_path),
                expected_component=component,
                required=required,
            )
        )
    return rows


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row["evidence_logic_class"] for row in rows))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, *, manifest: dict[str, Any], rows: list[dict[str, str]]) -> None:
    summary = summarize_rows(rows)
    lines = [
        "# Evidence Binding Contract Audit v1",
        "",
        "Diagnostic-only, not benchmark-valid final output.",
        "",
        "## Authority source",
        f"- active_run_id: `{manifest.get('active_run_id', '')}`",
        f"- active_run_dir: `{manifest.get('active_run_dir', '')}`",
        f"- pointer_path: `{manifest.get('pointer_path', '')}`",
        "",
        "## Evidence logic distribution",
    ]
    for key in sorted(summary):
        lines.append(f"- {key}: {summary[key]}")
    lines.extend(["", "## Surface findings"])
    for row in rows:
        lines.append(
            f"- {row['surface_name']} / {row['evidence_logic_class']}: {row['proof_kind']}={row['proof_detail']} ({row['surface_status']})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_context(path: Path, *, manifest_path: Path, manifest: dict[str, Any], rows: list[dict[str, str]]) -> None:
    summary = summarize_rows(rows)
    lines = [
        "# RUN_CONTEXT",
        "",
        "## Run purpose",
        "Audit current Layer3/Stage5 evidence-handoff surfaces before building Evidence Binding Packs.",
        "",
        "## Run type",
        "Diagnostic-only contract audit. Not benchmark-valid final output.",
        "",
        "## Starting inputs",
        f"- authority_manifest_path: `{manifest_path}`",
        f"- active_run_id: `{manifest.get('active_run_id', '')}`",
        f"- active_run_dir: `{manifest.get('active_run_dir', '')}`",
        "",
        "## Outputs",
        "- `analysis/evidence_binding_contract_audit_v1.tsv`",
        "- `analysis/evidence_binding_contract_audit_v1.md`",
        "- `analysis/evidence_binding_contract_audit_summary_v1.json`",
        "",
        "## Evidence logic distribution",
    ]
    for key in sorted(summary):
        lines.append(f"- {key}: {summary[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contract_audit(*, authority_manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    authority_manifest_path = authority_manifest_path.resolve()
    out_dir = out_dir.resolve()
    manifest = load_manifest(authority_manifest_path)
    rows = rows_from_manifest(manifest)
    for surface_name, component, path, required in STATIC_CODE_SURFACES:
        rows.extend(
            audit_text_surface(
                surface_name=surface_name,
                path=path,
                expected_component=component,
                required=required,
            )
        )
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(analysis_dir / "evidence_binding_contract_audit_v1.tsv", rows)
    write_markdown(analysis_dir / "evidence_binding_contract_audit_v1.md", manifest=manifest, rows=rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/stage5_benchmark/audit_evidence_binding_contract_v1.py",
        "benchmark_valid": "no",
        "authority_manifest_path": str(authority_manifest_path),
        "active_run_id": manifest.get("active_run_id", ""),
        "active_run_dir": manifest.get("active_run_dir", ""),
        "evidence_logic_distribution": summarize_rows(rows),
        "row_count": len(rows),
    }
    (analysis_dir / "evidence_binding_contract_audit_summary_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_run_context(out_dir / "RUN_CONTEXT.md", manifest_path=authority_manifest_path, manifest=manifest, rows=rows)
    return {"manifest": manifest, "rows": rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit current evidence handoff contract surfaces.")
    parser.add_argument("--authority-manifest-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_contract_audit(authority_manifest_path=args.authority_manifest_json, out_dir=args.out_dir)
    print(f"active_run_dir={result['manifest'].get('active_run_dir', '')}")
    print(f"audit_rows={len(result['rows'])}")
    for key, value in sorted(result["summary"]["evidence_logic_distribution"].items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
