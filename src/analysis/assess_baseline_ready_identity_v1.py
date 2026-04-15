#!/usr/bin/env python3
from __future__ import annotations

"""
Assess whether a deterministic Step 1 run is ready for baseline Step 2 use.

This analysis layer does not modify the strict identity-freeze gate. It reads
the frozen Step 1 artifacts and classifies each paper as:

- strict_pass
- baseline_ready
- fatal_blocked
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ASSESSMENT_TSV_NAME = "baseline_ready_identity_assessment_v1.tsv"
SUMMARY_MD_NAME = "baseline_ready_identity_summary_v1.md"

# Explicit, governed heuristics for the baseline-ready analysis layer.
DUPLICATE_ID_TOLERANCE = 0
MAX_ACCEPTABLE_ROW_DRIFT_RATIO = 0.30
FATAL_ROW_EXPLOSION_RATIO = 1.75
MAX_ACCEPTABLE_REASSIGNMENT_RATIO = 0.20
MAX_ACCEPTABLE_MISSING_CORE_IDENTITY_RATIO = 0.10
FATAL_MISSING_CORE_IDENTITY_RATIO = 0.25
MAX_ACCEPTABLE_UNRESOLVED_ONLY_RATIO = 0.40


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in fieldnames})


def repo_rel(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root)).replace("\\", "/")


def parse_int(value: Any) -> int:
    text = normalize_text(value)
    return int(text) if text else 0


def format_ratio(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.4f}"


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def reassignment_severity_label(value: float) -> str:
    if value <= 0:
        return "none"
    if value <= MAX_ACCEPTABLE_REASSIGNMENT_RATIO:
        return "local"
    return "widespread"


def discover_run_inputs(run_dir: Path) -> dict[str, Path]:
    analysis_dir = run_dir / "analysis"
    freeze_dir = run_dir / "audit" / "identity_freeze_guardrail_v1"
    scaffold_dir = run_dir / "audit" / "layer2_identity_scaffold_binding_v1"
    paths = {
        "final_table": run_dir / "final_formulation_table_v1.tsv",
        "freeze_report": freeze_dir / "identity_freeze_report_v1.tsv",
        "freeze_summary": freeze_dir / "identity_freeze_summary_v1.tsv",
        "scaffold_rows": scaffold_dir / "layer2_identity_scaffold_rows_v1.tsv",
        "assessment_tsv": analysis_dir / ASSESSMENT_TSV_NAME,
        "summary_md": analysis_dir / SUMMARY_MD_NAME,
    }
    required = ("final_table", "freeze_report", "freeze_summary")
    missing = [name for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"Run dir is missing required assessment inputs: {missing}")
    return paths


def build_paper_indexes(final_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in final_rows:
        paper_key = normalize_text(row.get("key") or row.get("paper_key"))
        if paper_key:
            by_paper[paper_key].append(row)
    return by_paper


def detect_id_stability(rows: list[dict[str, str]]) -> tuple[str, int, int]:
    seen: set[str] = set()
    duplicate_count = 0
    missing_core_count = 0
    for row in rows:
        final_id = normalize_text(row.get("final_formulation_id"))
        rep_id = normalize_text(row.get("representative_source_formulation_id"))
        paper_key = normalize_text(row.get("key") or row.get("paper_key"))
        if not final_id or not rep_id or not paper_key:
            missing_core_count += 1
        if final_id:
            if final_id in seen:
                duplicate_count += 1
            seen.add(final_id)
    if duplicate_count > DUPLICATE_ID_TOLERANCE:
        return "fail", duplicate_count, missing_core_count
    return "pass", duplicate_count, missing_core_count


def failure_types_for_summary(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    if normalize_text(row.get("row_count_drift_detected")) == "yes":
        failures.append("row_count_drift")
    if normalize_text(row.get("identity_reassignment_detected")) == "yes":
        failures.append("identity_reassignment")
    if parse_int(row.get("unresolved_scaffold_rows")) > 0:
        failures.append("unresolved_scaffold_rows")
    if parse_int(row.get("ambiguous_scaffold_rows")) > 0:
        failures.append("ambiguous_scaffold_rows")
    return failures


def report_metrics(report_rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    for row in report_rows:
        binding_status = normalize_text(row.get("binding_status"))
        if binding_status:
            counts[f"binding_status::{binding_status}"] += 1
        if normalize_text(row.get("missing_from_final_table")) == "yes":
            counts["missing_from_final_table"] += 1
        if normalize_text(row.get("ambiguous_binding")) == "yes":
            counts["ambiguous_binding"] += 1
        if normalize_text(row.get("violation")) == "yes":
            counts["violation"] += 1
    return dict(counts)


def classify_paper(
    *,
    paper_key: str,
    final_rows: list[dict[str, str]],
    summary_row: dict[str, str],
    report_rows: list[dict[str, str]],
) -> dict[str, str]:
    predicted_count = len(final_rows)
    reference_count = parse_int(summary_row.get("upstream_identity_count"))
    freeze_status = normalize_text(summary_row.get("status")) or "fail"
    failure_types = failure_types_for_summary(summary_row)
    id_stability, duplicate_id_count, missing_core_count = detect_id_stability(final_rows)

    metrics = report_metrics(report_rows)
    unresolved_count = parse_int(summary_row.get("unresolved_scaffold_rows"))
    ambiguous_count = parse_int(summary_row.get("ambiguous_scaffold_rows"))
    unbound_final_count = metrics.get("binding_status::unbound_final_row", 0)
    reassignment_events = metrics.get("missing_from_final_table", 0) + unbound_final_count + ambiguous_count

    drift_ratio = (
        abs(predicted_count - reference_count) / reference_count if reference_count > 0 else None
    )
    predicted_to_reference_ratio = (
        predicted_count / reference_count if reference_count > 0 else float(predicted_count > 0)
    )
    reassignment_ratio = ratio(reassignment_events, max(predicted_count, reference_count))
    unresolved_ratio = ratio(unresolved_count, reference_count)
    missing_core_ratio = ratio(missing_core_count, predicted_count)
    reassignment_severity = reassignment_severity_label(reassignment_ratio)

    classification = "baseline_ready"
    rationale_parts: list[str] = []

    if freeze_status == "pass" and id_stability == "pass":
        classification = "strict_pass"
        rationale_parts.append("passes strict identity freeze")
    else:
        if duplicate_id_count > DUPLICATE_ID_TOLERANCE:
            classification = "fatal_blocked"
            rationale_parts.append("duplicate final_formulation_id detected")
        if missing_core_ratio > FATAL_MISSING_CORE_IDENTITY_RATIO:
            classification = "fatal_blocked"
            rationale_parts.append("bulk missing core identity fields")
        if predicted_to_reference_ratio > FATAL_ROW_EXPLOSION_RATIO:
            classification = "fatal_blocked"
            rationale_parts.append("catastrophic row explosion")
        if reassignment_ratio > MAX_ACCEPTABLE_REASSIGNMENT_RATIO:
            classification = "fatal_blocked"
            rationale_parts.append("widespread identity reassignment pattern")
        if classification != "fatal_blocked":
            if id_stability == "fail":
                classification = "fatal_blocked"
                rationale_parts.append("id stability failed")
            elif drift_ratio is not None and drift_ratio > MAX_ACCEPTABLE_ROW_DRIFT_RATIO and predicted_count > reference_count:
                classification = "fatal_blocked"
                rationale_parts.append("overproduction drift exceeds baseline-ready threshold")
            elif missing_core_ratio > MAX_ACCEPTABLE_MISSING_CORE_IDENTITY_RATIO:
                classification = "fatal_blocked"
                rationale_parts.append("too many rows missing core identity fields")
            elif reassignment_ratio <= MAX_ACCEPTABLE_REASSIGNMENT_RATIO:
                if drift_ratio is not None and drift_ratio > MAX_ACCEPTABLE_ROW_DRIFT_RATIO:
                    rationale_parts.append("moderate under/over-count drift accepted for baseline use")
                if unresolved_ratio > 0:
                    if unresolved_ratio <= MAX_ACCEPTABLE_UNRESOLVED_ONLY_RATIO:
                        rationale_parts.append("limited unresolved scaffold rows accepted")
                    else:
                        classification = "fatal_blocked"
                        rationale_parts.append("too many unresolved scaffold rows")
                if classification != "fatal_blocked" and not rationale_parts:
                    rationale_parts.append("strict freeze failed, but in-run identities remain stable and drift is bounded")

    if classification == "baseline_ready" and not rationale_parts:
        rationale_parts.append("strict freeze failed, but baseline safety heuristics passed")

    return {
        "paper_key": paper_key,
        "predicted_count": str(predicted_count),
        "reference_count": str(reference_count) if reference_count > 0 else "",
        "drift_ratio": format_ratio(drift_ratio),
        "freeze_status": freeze_status,
        "failure_types": "|".join(failure_types),
        "id_stability": id_stability,
        "reassignment_severity": reassignment_severity,
        "classification": classification,
        "rationale": "; ".join(rationale_parts),
    }


def build_summary_markdown(
    *,
    repo_root: Path,
    run_dir: Path,
    assessment_rows: list[dict[str, str]],
    assessment_tsv: Path,
    strict_pass: list[str],
    baseline_ready: list[str],
    fatal_blocked: list[str],
) -> str:
    eligible = strict_pass + baseline_ready
    return (
        "\n".join(
            [
                "# Baseline-Ready Identity Summary v1",
                "",
                "## Inputs",
                f"- run_dir: `{repo_rel(run_dir, repo_root)}`",
                f"- assessment_tsv: `{repo_rel(assessment_tsv, repo_root)}`",
                "",
                "## Thresholds",
                f"- duplicate_id_tolerance: `{DUPLICATE_ID_TOLERANCE}`",
                f"- max_acceptable_row_drift_ratio: `{MAX_ACCEPTABLE_ROW_DRIFT_RATIO}`",
                f"- fatal_row_explosion_ratio: `{FATAL_ROW_EXPLOSION_RATIO}`",
                f"- max_acceptable_reassignment_ratio: `{MAX_ACCEPTABLE_REASSIGNMENT_RATIO}`",
                f"- max_acceptable_missing_core_identity_ratio: `{MAX_ACCEPTABLE_MISSING_CORE_IDENTITY_RATIO}`",
                f"- fatal_missing_core_identity_ratio: `{FATAL_MISSING_CORE_IDENTITY_RATIO}`",
                f"- max_acceptable_unresolved_only_ratio: `{MAX_ACCEPTABLE_UNRESOLVED_ONLY_RATIO}`",
                "",
                "## Counts",
                f"- total papers: `{len(assessment_rows)}`",
                f"- strict_pass: `{len(strict_pass)}`",
                f"- baseline_ready: `{len(baseline_ready)}`",
                f"- fatal_blocked: `{len(fatal_blocked)}`",
                f"- Step 2 eligible total: `{len(eligible)}`",
                "",
                "## Step 2 Eligible Set",
                f"- strict_pass papers: `{', '.join(strict_pass) if strict_pass else 'none'}`",
                f"- baseline_ready papers: `{', '.join(baseline_ready) if baseline_ready else 'none'}`",
                f"- fatal_blocked papers: `{', '.join(fatal_blocked) if fatal_blocked else 'none'}`",
            ]
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess whether a deterministic Step 1 run is baseline-ready for Step 2.")
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    paths = discover_run_inputs(run_dir)

    final_rows = read_tsv_rows(paths["final_table"])
    freeze_report_rows = read_tsv_rows(paths["freeze_report"])
    freeze_summary_rows = read_tsv_rows(paths["freeze_summary"])

    final_by_paper = build_paper_indexes(final_rows)
    report_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in freeze_report_rows:
        paper_key = normalize_text(row.get("paper_key"))
        if paper_key:
            report_by_paper[paper_key].append(row)

    assessment_rows = [
        classify_paper(
            paper_key=normalize_text(summary_row.get("paper_key")),
            final_rows=final_by_paper.get(normalize_text(summary_row.get("paper_key")), []),
            summary_row=summary_row,
            report_rows=report_by_paper.get(normalize_text(summary_row.get("paper_key")), []),
        )
        for summary_row in freeze_summary_rows
    ]

    write_tsv(
        paths["assessment_tsv"],
        [
            "paper_key",
            "predicted_count",
            "reference_count",
            "drift_ratio",
            "freeze_status",
            "failure_types",
            "id_stability",
            "reassignment_severity",
            "classification",
            "rationale",
        ],
        assessment_rows,
    )

    strict_pass = [row["paper_key"] for row in assessment_rows if row["classification"] == "strict_pass"]
    baseline_ready = [row["paper_key"] for row in assessment_rows if row["classification"] == "baseline_ready"]
    fatal_blocked = [row["paper_key"] for row in assessment_rows if row["classification"] == "fatal_blocked"]

    summary_text = build_summary_markdown(
        repo_root=repo_root,
        run_dir=run_dir,
        assessment_rows=assessment_rows,
        assessment_tsv=paths["assessment_tsv"],
        strict_pass=strict_pass,
        baseline_ready=baseline_ready,
        fatal_blocked=fatal_blocked,
    )
    paths["summary_md"].write_text(summary_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "strict_pass": strict_pass,
                "baseline_ready": baseline_ready,
                "fatal_blocked": fatal_blocked,
                "step2_eligible_total": len(strict_pass) + len(baseline_ready),
                "assessment_tsv": str(paths["assessment_tsv"]),
                "summary_md": str(paths["summary_md"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
