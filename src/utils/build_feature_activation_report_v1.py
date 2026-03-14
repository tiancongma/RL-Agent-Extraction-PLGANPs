#!/usr/bin/env python3
from __future__ import annotations

"""
Build a run-scoped feature activation report from deterministic artifact signals.

Purpose:
- Distinguish repository feature existence from run-local feature activation.
- Make lineage reuse failures visible when a child validation run proves a fix,
  but a benchmark-valid parent run still reuses older artifacts.

This utility is intentionally conservative:
- mark a feature as active only when run artifacts provide direct evidence
- prefer "unclear" over invented activation
- ignore descendant lineage child runs when inspecting a parent run
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.utils.paths import PROJECT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a run-scoped feature activation report.")
    parser.add_argument(
        "--registry",
        default=str(PROJECT_DIR / "feature_units" / "feature_unit_registry.json"),
        help="Path to the feature unit registry JSON.",
    )
    parser.add_argument(
        "--matrix",
        default=str(PROJECT_DIR / "feature_units" / "feature_intervention_matrix.tsv"),
        help="Path to the feature intervention matrix TSV.",
    )
    parser.add_argument("--run-dir", required=True, help="Target run directory.")
    parser.add_argument(
        "--out-tsv",
        default="",
        help="Optional explicit output TSV path. Default: <run-dir>/analysis/feature_activation_report_v1.tsv",
    )
    return parser.parse_args()


def load_registry(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrix(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return {
            row["feature_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def is_lineage_child_path(run_dir: Path, path: Path) -> bool:
    rel_parts = path.relative_to(run_dir).parts
    return len(rel_parts) >= 2 and rel_parts[0] == "lineage" and rel_parts[1] == "children"


def find_run_files(run_dir: Path, name: str) -> list[Path]:
    matches = []
    for path in run_dir.rglob(name):
        if is_lineage_child_path(run_dir, path):
            continue
        matches.append(path)
    return sorted(matches, key=lambda p: (len(p.relative_to(run_dir).parts), str(p).lower()))


def first_run_file(run_dir: Path, name: str) -> Path | None:
    matches = find_run_files(run_dir, name)
    return matches[0] if matches else None


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR.parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def looks_like_numeric_label(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    digits = "".join(ch for ch in text if ch.isdigit())
    return bool(digits) and text.replace(".", "").replace(" ", "").isdigit()


def detect_surfaces(run_dir: Path) -> dict[str, Any]:
    weak_labels_path = first_run_file(run_dir, "weak_labels__v7pilot_r3_fixparse.tsv")
    final_table_path = first_run_file(run_dir, "final_formulation_table_v1.tsv")
    decision_trace_path = first_run_file(run_dir, "final_output_decision_trace_v1.tsv")
    compare_counts_path = first_run_file(run_dir, "final_table_vs_gt_counts_by_doi.tsv")
    run_context_path = run_dir / "RUN_CONTEXT.md"
    surfaces = {
        "weak_labels_path": weak_labels_path,
        "final_table_path": final_table_path,
        "decision_trace_path": decision_trace_path,
        "compare_counts_path": compare_counts_path,
        "run_context_path": run_context_path if run_context_path.exists() else None,
        "stage2_active": weak_labels_path is not None,
        "stage2_child_validation": "validation" in run_dir.name.lower(),
        "stage3_resolution": first_run_file(run_dir, "formulation_relation_summary_v1.tsv") is not None,
        "stage4_eval": first_run_file(run_dir, "per_doi_formulation_instance_summary.tsv") is not None,
        "stage5_final": final_table_path is not None and decision_trace_path is not None,
        "benchmark_compare": compare_counts_path is not None,
        "run_context": run_context_path.exists(),
        "regression_guard": first_run_file(run_dir, "numbered_doe_regression_guard_v1.tsv") is not None,
    }
    if weak_labels_path is not None:
        weak_rows = read_tsv(weak_labels_path)
        surfaces["contains_ufxx"] = any(row.get("key") == "UFXX9WXE" for row in weak_rows)
    else:
        surfaces["contains_ufxx"] = False
    return surfaces


def expected_for_run(feature_id: str, matrix_row: dict[str, str], surfaces: dict[str, Any]) -> str:
    if feature_id in {
        "numbered_doe_row_enumeration_priority",
        "numbered_doe_regression_guard",
        "table_first_evidence_binding",
    } and not surfaces.get("contains_ufxx"):
        return "no"

    surface_columns = [
        "stage2_active",
        "stage2_child_validation",
        "stage3_resolution",
        "stage4_eval",
        "stage5_final",
        "benchmark_compare",
        "run_context",
        "regression_guard",
    ]
    for column in surface_columns:
        if matrix_row.get(column) == "required" and surfaces.get(column):
            return "yes"
    return "no"


def compute_activation_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    required_rows = [row for row in rows if row["expected_for_run"] == "yes"]
    missing_required = [row["feature_id"] for row in required_rows if row["activation_status"] == "missing"]
    unclear_required = [row["feature_id"] for row in required_rows if row["activation_status"] == "unclear"]
    if missing_required:
        gate = "fail"
    elif unclear_required:
        gate = "warn"
    else:
        gate = "pass"
    return {
        "required_feature_units": [row["feature_id"] for row in required_rows],
        "missing_required_feature_units": missing_required,
        "unclear_required_feature_units": unclear_required,
        "run_activation_gate": gate,
    }


def observe_numbered_doe_regression_guard(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    guard_path = first_run_file(run_dir, "numbered_doe_regression_guard_v1.tsv")
    if guard_path is None:
        return {
            "observed_activation": "missing",
            "activation_status": "missing",
            "evidence_path": "",
            "evidence_detail": "No numbered_doe_regression_guard_v1.tsv found in the run-local artifacts.",
            "notes": "The guard may exist in code but is not evidenced in this run.",
        }
    rows = read_tsv(guard_path)
    fail_count = sum(1 for row in rows if row.get("guard_status") == "fail")
    warn_count = sum(1 for row in rows if row.get("guard_status") == "warn")
    return {
        "observed_activation": "active",
        "activation_status": "active",
        "evidence_path": to_repo_rel(guard_path),
        "evidence_detail": f"guard_rows={len(rows)} fail={fail_count} warn={warn_count}",
        "notes": "Active because the run emitted a guard artifact.",
    }


def observe_variant_aware_gt_authority_switch(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    counts_files = find_run_files(run_dir, "final_table_vs_gt_counts_by_doi.tsv")
    for path in counts_files:
        rows = read_tsv(path)
        if any("dev15_formulation_skeleton_review_v2_variantaware.xlsx" in row.get("gt_authority_file", "") for row in rows):
            return {
                "observed_activation": "active",
                "activation_status": "active",
                "evidence_path": to_repo_rel(path),
                "evidence_detail": "Compare artifact records the v2 variant-aware GT workbook path.",
                "notes": "The feature is active only because the run-local compare artifact proves the v2 authority was used.",
            }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": "",
        "evidence_detail": "No run-local compare artifact proved use of dev15_formulation_skeleton_review_v2_variantaware.xlsx.",
        "notes": "Code existence alone is not enough for activation.",
    }


def observe_numbered_doe_row_enumeration_priority(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    weak_labels_path = surfaces.get("weak_labels_path")
    if weak_labels_path is None:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No weak-label TSV found.",
            "notes": "Cannot inspect Stage2 activation without run-local weak labels.",
        }
    rows = [row for row in read_tsv(weak_labels_path) if row.get("key") == "UFXX9WXE"]
    if not rows:
        return {
            "observed_activation": "not_expected",
            "activation_status": "not_expected",
            "evidence_path": to_repo_rel(weak_labels_path),
            "evidence_detail": "UFXX9WXE is not present in this run's Stage2 weak labels.",
            "notes": "Current activation logic for this feature is grounded on UFXX9WXE-class detectable papers.",
        }
    doe_count = sum(1 for row in rows if row.get("candidate_source") == "doe_numbered_table_row")
    llm_numeric_count = sum(
        1 for row in rows
        if row.get("candidate_source") == "llm_extracted" and looks_like_numeric_label(row.get("raw_formulation_label", ""))
    )
    if doe_count > 0:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(weak_labels_path),
            "evidence_detail": f"UFXX9WXE doe_numbered_table_row={doe_count} llm_numeric_rows={llm_numeric_count}",
            "notes": "Active because structured DOE rows reached the run-local Stage2 candidate surface.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": to_repo_rel(weak_labels_path),
        "evidence_detail": f"UFXX9WXE rows={len(rows)} doe_numbered_table_row=0 llm_numeric_rows={llm_numeric_count}",
        "notes": "UFXX9WXE is in scope, but the run-local Stage2 artifact does not carry deterministic numbered DOE rows.",
    }


def observe_table_first_evidence_binding(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    weak_labels_path = surfaces.get("weak_labels_path")
    if weak_labels_path is None:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No weak-label TSV found.",
            "notes": "Cannot inspect table-first evidence anchors without run-local Stage2 rows.",
        }
    rows = [row for row in read_tsv(weak_labels_path) if row.get("key") == "UFXX9WXE"]
    table_rows = [
        row for row in rows
        if row.get("candidate_source") == "doe_numbered_table_row"
        and row.get("instance_evidence_region_type") == "table_row"
        and "numbered_doe_table" in row.get("evidence_section", "")
    ]
    if table_rows:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(weak_labels_path),
            "evidence_detail": f"UFXX9WXE table_row_anchors={len(table_rows)}",
            "notes": "Active because run-local Stage2 rows retain explicit table_row anchors.",
        }
    if rows:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": to_repo_rel(weak_labels_path),
            "evidence_detail": "UFXX9WXE rows are present, but no explicit numbered table_row anchors were found.",
            "notes": "Current artifacts do not prove table-first binding for this run.",
        }
    return {
        "observed_activation": "not_expected",
        "activation_status": "not_expected",
        "evidence_path": to_repo_rel(weak_labels_path),
        "evidence_detail": "UFXX9WXE is not present in this run's Stage2 weak labels.",
        "notes": "Current detection for this feature is grounded on UFXX9WXE-class structured DOE papers.",
    }


def observe_family_variant_retention_governance(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    trace_path = surfaces.get("decision_trace_path")
    if trace_path is None:
        return {
            "observed_activation": "unclear",
            "activation_status": "unclear",
            "evidence_path": "",
            "evidence_detail": "No final_output_decision_trace_v1.tsv found.",
            "notes": "Cannot inspect Stage5 governance activation without a run-local decision trace.",
        }
    rows = read_tsv(trace_path)
    variant_rows = [row for row in rows if row.get("confidence_or_rule_scope") == "phase1_variant_governance"]
    if variant_rows:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(trace_path),
            "evidence_detail": f"phase1_variant_governance_rows={len(variant_rows)}",
            "notes": "Active because the run-local Stage5 trace records variant-governance intervention.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": to_repo_rel(trace_path),
        "evidence_detail": "No phase1_variant_governance rows were found in the run-local decision trace.",
        "notes": "Child replays do not count as activation for the parent run.",
    }


def observe_feature_unit_governance_layer(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    report_path = run_dir / "analysis" / "feature_activation_report_v1.tsv"
    registry_path = PROJECT_DIR / "feature_units" / "feature_unit_registry.json"
    matrix_path = PROJECT_DIR / "feature_units" / "feature_intervention_matrix.tsv"
    if report_path.exists() and registry_path.exists() and matrix_path.exists():
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(report_path),
            "evidence_detail": (
                "Run-local activation report exists and the project-level registry plus matrix are available."
            ),
            "notes": "Active because this run has already materialized feature governance into a report.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": "",
        "evidence_detail": "Run-local activation report or project-level registry/matrix is missing.",
        "notes": "The governance layer only counts as active when the run materializes it into local evidence.",
    }


def observe_run_context_feature_activation_integration(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    run_context_path = surfaces.get("run_context_path")
    if run_context_path is None:
        return {
            "observed_activation": "missing",
            "activation_status": "missing",
            "evidence_path": "",
            "evidence_detail": "RUN_CONTEXT.md was not found for this run.",
            "notes": "Cannot prove run-context activation integration without RUN_CONTEXT.md.",
        }
    text = run_context_path.read_text(encoding="utf-8")
    if "## Feature Unit Activation" in text and "run_activation_gate" in text and "feature_activation_report_path" in text:
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(run_context_path),
            "evidence_detail": "RUN_CONTEXT.md contains the generated Feature Unit Activation section.",
            "notes": "Active because the run metadata includes feature activation status and gate details.",
        }
    return {
        "observed_activation": "missing",
        "activation_status": "missing",
        "evidence_path": to_repo_rel(run_context_path),
        "evidence_detail": "RUN_CONTEXT.md exists but does not contain the generated Feature Unit Activation section.",
        "notes": "The updater must be run for this feature to count as active.",
    }


def observe_doi_level_gt_vs_pred_count_audit(run_dir: Path, surfaces: dict[str, Any]) -> dict[str, str]:
    compare_counts_path = surfaces.get("compare_counts_path")
    if compare_counts_path is None:
        return {
            "observed_activation": "missing",
            "activation_status": "missing",
            "evidence_path": "",
            "evidence_detail": "No final_table_vs_gt_counts_by_doi.tsv found.",
            "notes": "The DOI-level count audit feature requires a run-local compare artifact.",
        }
    rows = read_tsv(compare_counts_path)
    required_columns = {"doi", "gt_count", "pred_count", "delta_count", "count_status"}
    observed_columns = set(rows[0].keys()) if rows else set()
    if rows and required_columns.issubset(observed_columns):
        return {
            "observed_activation": "active",
            "activation_status": "active",
            "evidence_path": to_repo_rel(compare_counts_path),
            "evidence_detail": f"rows={len(rows)} required_columns_present=yes",
            "notes": "Active because the run emitted the DOI-level GT versus prediction count audit table.",
        }
    return {
        "observed_activation": "unclear",
        "activation_status": "unclear",
        "evidence_path": to_repo_rel(compare_counts_path),
        "evidence_detail": "Counts-by-doi artifact exists but did not expose the full required schema.",
        "notes": "Keep this conservative until the compare artifact clearly exposes the audit columns.",
    }


OBSERVERS = {
    "numbered_doe_regression_guard": observe_numbered_doe_regression_guard,
    "variant_aware_gt_authority_switch": observe_variant_aware_gt_authority_switch,
    "numbered_doe_row_enumeration_priority": observe_numbered_doe_row_enumeration_priority,
    "table_first_evidence_binding": observe_table_first_evidence_binding,
    "family_variant_retention_governance": observe_family_variant_retention_governance,
    "feature_unit_governance_layer": observe_feature_unit_governance_layer,
    "run_context_feature_activation_integration": observe_run_context_feature_activation_integration,
    "doi_level_gt_vs_pred_count_audit": observe_doi_level_gt_vs_pred_count_audit,
}


def build_report_rows(
    *,
    registry: list[dict[str, Any]],
    matrix: dict[str, dict[str, str]],
    run_dir: Path,
) -> list[dict[str, str]]:
    surfaces = detect_surfaces(run_dir)
    rows: list[dict[str, str]] = []
    for feature in registry:
        feature_id = feature["feature_id"]
        matrix_row = matrix[feature_id]
        expected = expected_for_run(feature_id, matrix_row, surfaces)
        observer = OBSERVERS.get(feature_id)
        if observer is None:
            observed = {
                "observed_activation": "unclear",
                "activation_status": "unclear",
                "evidence_path": "",
                "evidence_detail": "No observer implemented for this feature.",
                "notes": "Add a deterministic artifact-based observer before relying on this feature in reports.",
            }
        else:
            observed = observer(run_dir, surfaces)
        if expected == "no" and observed["activation_status"] not in {"active"}:
            observed["activation_status"] = "not_expected"
            observed["observed_activation"] = "not_expected"
        rows.append(
            {
                "feature_id": feature_id,
                "expected_for_run": expected,
                "observed_activation": observed["observed_activation"],
                "activation_status": observed["activation_status"],
                "evidence_path": observed["evidence_path"],
                "evidence_detail": observed["evidence_detail"],
                "notes": observed["notes"],
            }
        )
    return rows


def write_report_tsv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature_id",
        "expected_for_run",
        "observed_activation",
        "activation_status",
        "evidence_path",
        "evidence_detail",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    registry = load_registry(Path(args.registry))
    matrix = load_matrix(Path(args.matrix))
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_path = Path(args.out_tsv) if args.out_tsv else run_dir / "analysis" / "feature_activation_report_v1.tsv"
    rows = build_report_rows(registry=registry, matrix=matrix, run_dir=run_dir)
    write_report_tsv(output_path, rows)
    print(str(output_path))


if __name__ == "__main__":
    main()
