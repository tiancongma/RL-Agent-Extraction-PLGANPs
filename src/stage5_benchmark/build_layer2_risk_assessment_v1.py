#!/usr/bin/env python3
from __future__ import annotations

"""
Build deterministic paper-level Layer 2 risk labels from an existing
Layer 2 identity-comparison artifact.

Purpose:
- stratify Layer 2 residual mismatch risk for downstream Layer 3 field audit
- preserve benchmark-valid final outputs unchanged
- keep risk labeling as output-layer metadata only

This helper does not modify Stage 2, Stage 3, or Stage 5 semantics.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context


RISK_TSV_NAME = "paper_risk_assessment.tsv"
RISK_SUMMARY_NAME = "paper_risk_assessment_summary.md"
DEFAULT_LAYER2_NAME = "layer2_identity_comparison.tsv"


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_int(row: dict[str, str], key: str) -> int:
    value = str(row.get(key, "")).strip()
    if not value:
        return 0
    return int(value)


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def resolve_layer2_compare_path(
    layer2_compare_tsv: Path | None,
    run_context: dict[str, Any] | None,
) -> Path:
    if run_context is None and layer2_compare_tsv is None:
        raise ValueError("Provide either --layer2-compare-tsv or an explicit/active source run.")
    return resolve_artifact_path(
        explicit_path=layer2_compare_tsv,
        run_context=run_context or {},
        pointer_key="layer2_identity_comparison_tsv",
        canonical_relative=f"analysis/{DEFAULT_LAYER2_NAME}",
    )


def resolve_out_dir(out_dir: Path | None, run_context: dict[str, Any] | None) -> Path:
    if out_dir is not None:
        return out_dir.resolve()
    if run_context is not None:
        return (Path(run_context["run_dir"]) / "analysis").resolve()
    raise ValueError("Provide either --out-dir or an explicit/active source run.")


def classify_risk_level(extra_count: int, missing_count: int) -> str:
    if extra_count <= 1 and missing_count == 0:
        return "LOW"
    if extra_count > 3 or missing_count >= 2:
        return "HIGH"
    return "MEDIUM"


def classify_risk_source(
    likely_owner: str,
    dominant_failure_family: str,
    extra_count: int,
    missing_count: int,
) -> str:
    owner = normalize_text(likely_owner)
    family = normalize_text(dominant_failure_family)

    if "stage2_identity_generation" in owner:
        if missing_count > extra_count:
            return "stage2_under_generation"
        return "stage2_over_generation"
    if "stage5_identity_closure" in owner:
        if extra_count > 0 and missing_count == 0:
            return "stage5_over_retention"
        return "mixed"
    if "mixed" in owner:
        return "mixed"

    if "over-retention" in family and missing_count == 0:
        return "stage5_over_retention"
    if "over-generation" in family:
        return "stage2_over_generation"
    if "under-generation" in family or (family.startswith("missing ") and extra_count == 0 and missing_count > 0):
        return "stage2_under_generation"
    if extra_count > 0 and missing_count > 0:
        return "mixed"
    return "unknown"


def classify_layer3_flag(risk_level: str) -> str:
    if risk_level == "LOW":
        return "INCLUDE"
    if risk_level == "MEDIUM":
        return "REVIEW"
    return "HOLD"


def build_rationale(
    risk_level: str,
    extra_count: int,
    missing_count: int,
    dominant_failure_family: str,
    risk_source: str,
) -> str:
    family = str(dominant_failure_family or "").strip() or "residual mismatch"
    return (
        f"{risk_level}: extra={extra_count}, missing={missing_count}; "
        f"source={risk_source}; family={family}"
    )


def build_risk_rows(layer2_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in layer2_rows:
        gt_count = parse_int(row, "gt_count")
        matched_count = parse_int(row, "matched_count")
        extra_count = parse_int(row, "extra_count")
        missing_count = parse_int(row, "missing_count")
        total_mismatch = extra_count + missing_count
        mismatch_ratio = (total_mismatch / gt_count) if gt_count else 0.0
        dominant_failure_family = row.get("dominant_failure_family", "")
        likely_owner = row.get("likely_owner", "")
        risk_level = classify_risk_level(extra_count, missing_count)
        risk_source = classify_risk_source(
            likely_owner,
            dominant_failure_family,
            extra_count,
            missing_count,
        )
        layer3_flag = classify_layer3_flag(risk_level)
        out.append(
            {
                "paper_key": row.get("paper_key", ""),
                "doi": row.get("doi", ""),
                "matched_count": matched_count,
                "extra_count": extra_count,
                "missing_count": missing_count,
                "total_mismatch": total_mismatch,
                "mismatch_ratio_within_paper": f"{mismatch_ratio:.4f}",
                "paper_risk_level": risk_level,
                "risk_source": risk_source,
                "layer3_inclusion_flag": layer3_flag,
                "rationale_short": build_rationale(
                    risk_level,
                    extra_count,
                    missing_count,
                    dominant_failure_family,
                    risk_source,
                ),
            }
        )
    return out


def build_summary_markdown(
    layer2_compare_tsv: Path,
    out_path: Path,
    risk_rows: list[dict[str, Any]],
) -> None:
    total = len(risk_rows)
    low = sum(1 for row in risk_rows if row["paper_risk_level"] == "LOW")
    medium = sum(1 for row in risk_rows if row["paper_risk_level"] == "MEDIUM")
    high = sum(1 for row in risk_rows if row["paper_risk_level"] == "HIGH")
    hold = sum(1 for row in risk_rows if row["layer3_inclusion_flag"] == "HOLD")
    review = sum(1 for row in risk_rows if row["layer3_inclusion_flag"] == "REVIEW")
    include = sum(1 for row in risk_rows if row["layer3_inclusion_flag"] == "INCLUDE")
    lines = [
        "# Paper Risk Assessment Summary",
        "",
        "## Purpose",
        "",
        "- This artifact stratifies Layer 2 residual mismatch risk for downstream Layer 3 field-level GT audit.",
        "- It is output-layer metadata only and does not modify benchmark-valid final outputs.",
        "",
        "## Inputs",
        "",
        f"- layer2_identity_comparison_tsv: `{layer2_compare_tsv}`",
        "",
        "## Deterministic contract",
        "",
        "- `LOW`: `extra_count <= 1` and `missing_count == 0`",
        "- `HIGH`: `extra_count > 3` or `missing_count >= 2`",
        "- `MEDIUM`: every non-LOW and non-HIGH paper",
        "- `INCLUDE` for `LOW`, `REVIEW` for `MEDIUM`, `HOLD` for `HIGH`",
        "",
        "## Aggregate counts",
        "",
        f"- paper_count: `{total}`",
        f"- low_risk: `{low}`",
        f"- medium_risk: `{medium}`",
        f"- high_risk: `{high}`",
        f"- include: `{include}`",
        f"- review: `{review}`",
        f"- hold: `{hold}`",
        "",
        "## Per-paper labels",
        "",
    ]
    for row in risk_rows:
        lines.append(
            "- `{paper_key}`: risk=`{paper_risk_level}` flag=`{layer3_inclusion_flag}` mismatch=`{total_mismatch}` rationale=`{rationale_short}`".format(
                **row
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic paper-level risk labels from an existing Layer 2 comparison TSV."
    )
    parser.add_argument("--layer2-compare-tsv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_context = None
    if args.run_dir is not None or str(args.run_id or "").strip():
        run_context = resolve_run_context(
            explicit_run_dir=args.run_dir,
            explicit_run_id=str(args.run_id or "").strip(),
        )
    elif args.layer2_compare_tsv is None:
        run_context = resolve_run_context(explicit_run_dir=None, explicit_run_id="")
    layer2_path = resolve_layer2_compare_path(args.layer2_compare_tsv, run_context)
    out_dir = resolve_out_dir(args.out_dir, run_context)
    print(
        json.dumps(
            {
                "resolved_source_run_dir": str(run_context["run_dir"]) if run_context else "",
                "resolved_source_run_id": str(run_context["run_id"]) if run_context else "",
                "source_resolution": str(run_context["resolution_source"]) if run_context else "explicit_artifact_only",
                "active_run_pointer_path": str(run_context.get("pointer_path") or "") if run_context else "",
                "resolved_input_files": {
                    "layer2_compare_tsv": str(layer2_path),
                },
                "resolved_out_dir": str(out_dir),
            },
            indent=2,
        )
    )

    layer2_rows = read_tsv_rows(layer2_path)
    risk_rows = build_risk_rows(layer2_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    risk_tsv_path = out_dir / RISK_TSV_NAME
    risk_summary_path = out_dir / RISK_SUMMARY_NAME
    fieldnames = [
        "paper_key",
        "doi",
        "matched_count",
        "extra_count",
        "missing_count",
        "total_mismatch",
        "mismatch_ratio_within_paper",
        "paper_risk_level",
        "risk_source",
        "layer3_inclusion_flag",
        "rationale_short",
    ]
    write_tsv(risk_tsv_path, fieldnames, risk_rows)
    build_summary_markdown(layer2_path, risk_summary_path, risk_rows)

    print(
        {
            "layer2_compare_tsv": str(layer2_path),
            "paper_risk_assessment_tsv": str(risk_tsv_path),
            "paper_risk_assessment_summary_md": str(risk_summary_path),
            "paper_count": len(risk_rows),
        }
    )


if __name__ == "__main__":
    main()
