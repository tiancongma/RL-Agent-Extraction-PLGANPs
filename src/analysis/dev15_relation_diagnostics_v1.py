#!/usr/bin/env python3
from __future__ import annotations

"""
Build a deterministic, run-scoped relation diagnostic pack for selected papers.

Purpose:
- Diagnose paper-level count mismatches using Stage 2 weak labels, Stage 3
  relation records, Stage 5 final rows, and the checked GT comparison table.
- Produce a small, human-readable audit pack focused on failure localization.

Inputs:
- `--weak-labels-tsv`: Stage 2 weak-label TSV.
- `--relation-records-tsv`: Stage 3 `formulation_relation_records_v1.tsv`.
- `--final-table-tsv`: Stage 5 `final_formulation_table_v1.tsv`.
- `--gt-comparison-tsv`: Stage 5 GT comparison TSV.
- `--paper-keys`: one or more paper keys to analyze.
- `--run-id`: existing run id whose `analysis/` subdirectory will receive
  outputs.

Outputs written only to:
- `data/results/<run_id>/analysis/paper_diagnostic_summary.tsv`
- `data/results/<run_id>/analysis/candidate_level_triage.tsv`
- `data/results/<run_id>/analysis/paper_audit_pack.md`

Stage role:
- Deterministic diagnostic analysis tool outside the core Stage 0 to Stage 5
  pipeline.

This script does not:
- call any LLM or external API,
- run extraction,
- modify core pipeline outputs,
- create uncontrolled audit directories outside the declared run-scoped
  `analysis/` location.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.paths import DATA_RESULTS_DIR


SUMMARY_NAME = "paper_diagnostic_summary.tsv"
TRIAGE_NAME = "candidate_level_triage.tsv"
PACK_NAME = "paper_audit_pack.md"

SUMMARY_FIELDS = [
    "paper_key",
    "doi",
    "stage2_candidate_count",
    "stage3_method_group_count",
    "stage3_shared_parameter_count",
    "stage3_variation_axis_count",
    "stage5_final_row_count",
    "gt_row_count",
    "count_delta",
    "suspected_error_type",
]

TRIAGE_FIELDS = [
    "paper_key",
    "doi",
    "formulation_candidate_id",
    "method_group_id",
    "parent_entity_id",
    "relation_types_present",
    "field_count",
    "shared_parameter_count",
    "variation_axis_indicator",
    "relation_record_count",
    "evidence_source_types",
    "triage_label",
    "triage_reason",
]

ALLOWED_ERROR_TYPES = {
    "over segmentation",
    "variation axis expansion",
    "shared parameter propagation failure",
    "collapse over-merging",
    "table interpretation issue",
    "relation structure missing",
    "other",
}

ALLOWED_TRIAGE_LABELS = {
    "high_confidence",
    "low_confidence",
    "uncertain",
}

RUN_ID_REGEX = r"^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def parse_jsonish_list(value: str) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [normalize_text(item) for item in parsed if normalize_text(item)]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(";") if part.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_run_id(run_id: str) -> str:
    rid = normalize_text(run_id)
    if not re.fullmatch(RUN_ID_REGEX, rid):
        raise ValueError(f"run_id does not match required pattern: {rid}")
    return rid


def validate_existing_run_dir(run_id: str) -> Path:
    run_dir = DATA_RESULTS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory does not exist. This tool writes only under an existing run: {run_dir}"
        )
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_dir}")
    return run_dir


def ensure_analysis_targets(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    analysis_dir = run_dir / "analysis"
    summary_path = analysis_dir / SUMMARY_NAME
    triage_path = analysis_dir / TRIAGE_NAME
    pack_path = analysis_dir / PACK_NAME
    for path in [summary_path, triage_path, pack_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing analysis artifact: {path}")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir, summary_path, triage_path, pack_path


def candidate_id(row: dict[str, str]) -> str:
    return normalize_text(row.get("local_instance_id") or row.get("formulation_id"))


def paper_doi(*rows: dict[str, str]) -> str:
    for row in rows:
        doi = normalize_text(row.get("doi"))
        if doi:
            return doi
    return ""


def paper_title_from_relation_rows(relation_rows: list[dict[str, str]]) -> str:
    for row in relation_rows:
        title = normalize_text(row.get("paper_title"))
        if title:
            return title
    return ""


def gt_count_from_row(row: dict[str, str]) -> int:
    for key in ["gt_count", "gt_formulation_count"]:
        value = normalize_text(row.get(key))
        if value:
            return int(value)
    raise ValueError(f"GT comparison row missing GT count column: {row}")


def final_count_from_row(row: dict[str, str]) -> int:
    for key in ["final_table_count", "predicted_formulation_count"]:
        value = normalize_text(row.get(key))
        if value:
            return int(value)
    raise ValueError(f"GT comparison row missing final count column: {row}")


def join_sorted(values: list[str] | set[str]) -> str:
    items = sorted({normalize_text(v) for v in values if normalize_text(v)})
    return "; ".join(items)


def classify_paper_error(
    *,
    stage2_candidate_count: int,
    stage3_method_group_count: int,
    stage3_shared_parameter_count: int,
    stage3_variation_axis_count: int,
    stage5_final_row_count: int,
    gt_row_count: int,
    parent_link_count: int,
    context_tags: set[str],
) -> str:
    delta = stage5_final_row_count - gt_row_count
    if delta > 0:
        if (
            stage3_method_group_count == 1
            and stage3_variation_axis_count >= 1
            and parent_link_count >= max(1, stage5_final_row_count - 1)
        ):
            return "variation axis expansion"
        if stage3_shared_parameter_count == 0 and stage3_method_group_count > 1:
            return "shared parameter propagation failure"
        return "over segmentation"

    if delta < 0:
        if stage2_candidate_count < gt_row_count and {"doe", "sweep", "optimized"} & context_tags:
            return "table interpretation issue"
        if stage5_final_row_count < stage2_candidate_count and stage2_candidate_count >= gt_row_count:
            return "collapse over-merging"
        if stage3_shared_parameter_count == 0 and stage3_method_group_count > 1:
            return "shared parameter propagation failure"
        if stage3_method_group_count <= 1:
            return "relation structure missing"
        return "other"

    return "other"


def candidate_triage(
    *,
    paper_error_type: str,
    weak_row: dict[str, str],
    method_group_id: str,
    parent_entity_id: str,
    relation_types: set[str],
    variation_fields: set[str],
    relation_record_count: int,
) -> tuple[str, str, int]:
    tags = set(parse_jsonish_list(weak_row.get("instance_context_tags", "")))
    label = "uncertain"
    reason = "Mixed relation signals; manual inspection required."
    score = 50

    if paper_error_type == "variation axis expansion":
        if parent_entity_id or variation_fields:
            label = "high_confidence"
            axis_text = join_sorted(variation_fields) or "variant axis"
            reason = (
                "Parent-linked variant within one method group; likely contributes to "
                f"variant expansion on {axis_text}."
            )
            score = 95
        else:
            label = "low_confidence"
            reason = "Root candidate in an over-expanded family; review but prioritize child variants first."
            score = 20
    elif paper_error_type == "table interpretation issue":
        if "candidate_non_formulation" == normalize_text(weak_row.get("instance_kind")):
            label = "low_confidence"
            reason = "Characterization-only row; less likely to explain the severe undercount."
            score = 10
        elif {"doe", "sweep", "optimized"} & tags:
            label = "high_confidence"
            reason = (
                "One of very few extracted DOE-style anchors in a severely under-enumerated paper; "
                "verify neighboring design-matrix or table rows for missing formulations."
            )
            score = 90
        else:
            label = "uncertain"
            reason = "Paper is under-enumerated; inspect whether this row anchors a missed table block."
            score = 60
    elif paper_error_type == "collapse over-merging":
        label = "high_confidence" if relation_record_count else "uncertain"
        reason = "Candidate should be checked against final-table collapse behavior."
        score = 85 if relation_record_count else 45
    elif paper_error_type == "shared parameter propagation failure":
        label = "high_confidence" if not method_group_id else "uncertain"
        reason = "Shared-field propagation looks unstable; verify whether group-level parameters were lost."
        score = 88 if not method_group_id else 55
    elif paper_error_type == "relation structure missing":
        label = "high_confidence"
        reason = "Paper-level relation structure is too thin for the GT multiplicity; inspect missing grouped variants."
        score = 85
    else:
        if "candidate_parent_link" in relation_types:
            label = "uncertain"
            reason = "Parent-linked candidate in a mismatch paper."
            score = 65
        else:
            label = "low_confidence"
            reason = "No strong deterministic triage signal."
            score = 25

    if label not in ALLOWED_TRIAGE_LABELS:
        raise ValueError(f"Unexpected triage label: {label}")
    return label, reason, score


def build_markdown_pack(
    *,
    paper_order: list[str],
    summary_rows: list[dict[str, Any]],
    triage_by_paper: dict[str, list[dict[str, Any]]],
    method_group_notes: dict[str, list[str]],
    paper_titles: dict[str, str],
) -> str:
    lines: list[str] = [
        "# Paper Audit Pack",
        "",
        "Diagnostic-only, deterministic audit pack built from Stage 2, Stage 3, Stage 5, and GT comparison artifacts.",
        "",
    ]
    summary_map = {row["paper_key"]: row for row in summary_rows}

    for paper_key in paper_order:
        summary = summary_map[paper_key]
        triage_rows = triage_by_paper.get(paper_key, [])
        suspicious_rows = [row for row in triage_rows if row["_triage_score"] >= 80][:5]
        lines.extend(
            [
                f"## {paper_key}",
                "",
                f"- DOI: `{summary['doi']}`",
                f"- Title: {paper_titles.get(paper_key) or 'not available'}",
                f"- GT count: `{summary['gt_row_count']}`",
                f"- Predicted count: `{summary['stage5_final_row_count']}`",
                f"- Delta: `{summary['count_delta']}`",
                f"- Suspected error type: `{summary['suspected_error_type']}`",
                "",
                "### Method groups summary",
                "",
            ]
        )
        group_lines = method_group_notes.get(paper_key, [])
        if group_lines:
            lines.extend([f"- {item}" for item in group_lines])
        else:
            lines.append("- No method-group summary available.")
        lines.extend(["", "### Candidate list summary", ""])
        lines.extend(
            [
                "| candidate_id | triage_label | variation_axis_indicator | relation_record_count | triage_reason |",
                "|---|---|---|---:|---|",
            ]
        )
        for row in triage_rows:
            lines.append(
                f"| `{row['formulation_candidate_id']}` | `{row['triage_label']}` | "
                f"`{row['variation_axis_indicator'] or 'none'}` | `{row['relation_record_count']}` | {row['triage_reason']} |"
            )
        lines.extend(["", "### Most suspicious candidates", ""])
        if suspicious_rows:
            for row in suspicious_rows:
                lines.append(
                    f"- `{row['formulation_candidate_id']}`: {row['triage_reason']}"
                )
        else:
            lines.append("- No high-confidence suspicious candidates were identified.")
        lines.extend(["", "### Recommended manual verification targets", ""])
        if summary["suspected_error_type"] == "variation axis expansion":
            lines.extend(
                [
                    "- Check whether drug-loading state (`blank`, `FITC`, `KGN`) should define separate benchmark formulations or only assay/control variants.",
                    "- Check whether polymer-family variants under one nanoprecipitation method group should collapse to a smaller benchmark-facing set.",
                    "- Prioritize the parent-linked variant chain before reviewing the root formulation.",
                ]
            )
        elif summary["suspected_error_type"] == "table interpretation issue":
            lines.extend(
                [
                    "- Inspect the DOE design matrix and optimization tables to verify how many formulation rows should have been enumerated.",
                    "- Check whether numbered formulations beyond the extracted anchors were missed during Stage 2 table reading.",
                    "- Compare the optimized row against the numbered DOE rows to determine whether it is additional or derived from an existing design point.",
                ]
            )
        elif summary["suspected_error_type"] == "collapse over-merging":
            lines.extend(
                [
                    "- Compare candidate rows to final retained rows and verify whether multiple distinct candidates were collapsed together.",
                    "- Inspect core-signature overlap and exclusion tags in Stage 5.",
                ]
            )
        else:
            lines.append("- Start from the highest-scoring candidates and verify relation-group membership against the source paper.")
        lines.extend(["", "---", ""])

    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a run-scoped diagnostic relation audit pack for selected DEV15 papers."
    )
    parser.add_argument("--weak-labels-tsv", required=True, type=Path)
    parser.add_argument("--relation-records-tsv", required=True, type=Path)
    parser.add_argument("--final-table-tsv", required=True, type=Path)
    parser.add_argument("--gt-comparison-tsv", required=True, type=Path)
    parser.add_argument("--paper-keys", required=True, nargs="+")
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_id = validate_run_id(args.run_id)
    run_dir = validate_existing_run_dir(run_id)
    _, summary_path, triage_path, pack_path = ensure_analysis_targets(run_dir)

    weak_rows = read_tsv(args.weak_labels_tsv)
    relation_rows = read_tsv(args.relation_records_tsv)
    final_rows = read_tsv(args.final_table_tsv)
    gt_rows = read_tsv(args.gt_comparison_tsv)

    paper_keys = [normalize_text(key) for key in args.paper_keys]
    weak_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    relation_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    final_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    gt_by_key: dict[str, dict[str, str]] = {}

    for row in weak_rows:
        weak_by_key[normalize_text(row.get("key"))].append(row)
    for row in relation_rows:
        relation_by_key[normalize_text(row.get("paper_key"))].append(row)
    for row in final_rows:
        final_by_key[normalize_text(row.get("key"))].append(row)
    for row in gt_rows:
        paper_key = normalize_text(row.get("paper_key"))
        if paper_key:
            gt_by_key[paper_key] = row

    summary_rows: list[dict[str, Any]] = []
    triage_rows: list[dict[str, Any]] = []
    triage_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_group_notes: dict[str, list[str]] = defaultdict(list)
    paper_titles: dict[str, str] = {}

    for paper_key in paper_keys:
        if paper_key not in weak_by_key:
            raise KeyError(f"Paper key not found in Stage 2 weak labels: {paper_key}")
        if paper_key not in gt_by_key:
            raise KeyError(f"Paper key not found in GT comparison TSV: {paper_key}")

        paper_weak_rows = weak_by_key[paper_key]
        paper_relation_rows = relation_by_key.get(paper_key, [])
        paper_final_rows = final_by_key.get(paper_key, [])
        paper_gt_row = gt_by_key[paper_key]

        doi = paper_doi(
            paper_gt_row,
            paper_weak_rows[0] if paper_weak_rows else {},
            paper_final_rows[0] if paper_final_rows else {},
        )
        paper_titles[paper_key] = paper_title_from_relation_rows(paper_relation_rows) or normalize_text(
            paper_gt_row.get("paper_title")
        )

        method_groups: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: {
                "candidate_ids": set(),
                "shared_fields": set(),
                "variation_axis_ids": set(),
                "variation_fields": set(),
            }
        )
        candidate_relations: dict[str, list[dict[str, str]]] = defaultdict(list)
        context_tags: set[str] = set()
        parent_link_count = 0

        for row in paper_weak_rows:
            context_tags.update(parse_jsonish_list(row.get("instance_context_tags", "")))

        for row in paper_relation_rows:
            mg_id = normalize_text(row.get("method_group_id"))
            candidate = normalize_text(row.get("formulation_candidate_id"))
            relation_type = normalize_text(row.get("relation_type"))
            field_name = normalize_text(row.get("field_name"))
            variation_axis_id = normalize_text(row.get("variation_axis_id"))
            if candidate:
                candidate_relations[candidate].append(row)
            if mg_id:
                if candidate:
                    method_groups[mg_id]["candidate_ids"].add(candidate)
                if relation_type == "method_group_shared_field" and field_name:
                    method_groups[mg_id]["shared_fields"].add(field_name)
                if relation_type == "method_group_variation_axis":
                    if variation_axis_id:
                        method_groups[mg_id]["variation_axis_ids"].add(variation_axis_id)
                    if field_name:
                        method_groups[mg_id]["variation_fields"].add(field_name)
            if relation_type == "candidate_parent_link":
                parent_link_count += 1

        stage2_candidate_count = len(paper_weak_rows)
        stage3_method_group_count = len(method_groups)
        stage3_shared_parameter_count = sum(len(stats["shared_fields"]) for stats in method_groups.values())
        stage3_variation_axis_count = sum(len(stats["variation_axis_ids"]) for stats in method_groups.values())
        stage5_final_row_count = len(paper_final_rows)
        gt_row_count = gt_count_from_row(paper_gt_row)
        count_delta = stage5_final_row_count - gt_row_count
        error_type = classify_paper_error(
            stage2_candidate_count=stage2_candidate_count,
            stage3_method_group_count=stage3_method_group_count,
            stage3_shared_parameter_count=stage3_shared_parameter_count,
            stage3_variation_axis_count=stage3_variation_axis_count,
            stage5_final_row_count=stage5_final_row_count,
            gt_row_count=gt_row_count,
            parent_link_count=parent_link_count,
            context_tags=context_tags,
        )
        if error_type not in ALLOWED_ERROR_TYPES:
            raise ValueError(f"Unexpected suspected_error_type: {error_type}")

        summary_row = {
            "paper_key": paper_key,
            "doi": doi,
            "stage2_candidate_count": stage2_candidate_count,
            "stage3_method_group_count": stage3_method_group_count,
            "stage3_shared_parameter_count": stage3_shared_parameter_count,
            "stage3_variation_axis_count": stage3_variation_axis_count,
            "stage5_final_row_count": stage5_final_row_count,
            "gt_row_count": gt_row_count,
            "count_delta": count_delta,
            "suspected_error_type": error_type,
        }
        summary_rows.append(summary_row)

        for mg_id in sorted(method_groups):
            mg = method_groups[mg_id]
            method_group_notes[paper_key].append(
                f"`{mg_id}`: candidates={len(mg['candidate_ids'])}, "
                f"shared_fields={join_sorted(mg['shared_fields']) or 'none'}, "
                f"variation_fields={join_sorted(mg['variation_fields']) or 'none'}"
            )

        for weak_row in paper_weak_rows:
            cand_id = candidate_id(weak_row)
            rel_rows = candidate_relations.get(cand_id, [])
            relation_types = {normalize_text(row.get("relation_type")) for row in rel_rows if normalize_text(row.get("relation_type"))}
            method_group_id = next(
                (normalize_text(row.get("method_group_id")) for row in rel_rows if normalize_text(row.get("method_group_id"))),
                "",
            )
            parent_entity_id = normalize_text(weak_row.get("parent_instance_id")) or next(
                (
                    normalize_text(row.get("parent_entity_id"))
                    for row in rel_rows
                    if normalize_text(row.get("relation_type")) == "candidate_parent_link"
                    and normalize_text(row.get("parent_entity_id"))
                ),
                "",
            )
            field_count = sum(
                1
                for row in rel_rows
                if normalize_text(row.get("relation_type")) == "candidate_field_membership"
            )
            shared_parameter_count = len(method_groups.get(method_group_id, {}).get("shared_fields", set()))
            variation_fields = {
                normalize_text(row.get("field_name"))
                for row in rel_rows
                if normalize_text(row.get("relation_type")) == "candidate_variation_axis_membership"
                and normalize_text(row.get("field_name"))
            }
            evidence_source_types = {
                normalize_text(row.get("evidence_source_type"))
                for row in rel_rows
                if normalize_text(row.get("evidence_source_type"))
            }
            triage_label, triage_reason, triage_score = candidate_triage(
                paper_error_type=error_type,
                weak_row=weak_row,
                method_group_id=method_group_id,
                parent_entity_id=parent_entity_id,
                relation_types=relation_types,
                variation_fields=variation_fields,
                relation_record_count=len(rel_rows),
            )
            triage_row = {
                "paper_key": paper_key,
                "doi": doi,
                "formulation_candidate_id": cand_id,
                "method_group_id": method_group_id,
                "parent_entity_id": parent_entity_id,
                "relation_types_present": join_sorted(relation_types),
                "field_count": field_count,
                "shared_parameter_count": shared_parameter_count,
                "variation_axis_indicator": join_sorted(variation_fields),
                "relation_record_count": len(rel_rows),
                "evidence_source_types": join_sorted(evidence_source_types),
                "triage_label": triage_label,
                "triage_reason": triage_reason,
                "_triage_score": triage_score,
            }
            triage_rows.append(triage_row)
            triage_by_paper[paper_key].append(triage_row)

        triage_by_paper[paper_key].sort(
            key=lambda row: (-int(row["_triage_score"]), row["formulation_candidate_id"])
        )

    write_tsv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_tsv(
        triage_path,
        TRIAGE_FIELDS,
        [{key: row[key] for key in TRIAGE_FIELDS} for row in triage_rows],
    )
    pack_text = build_markdown_pack(
        paper_order=paper_keys,
        summary_rows=summary_rows,
        triage_by_paper=triage_by_paper,
        method_group_notes=method_group_notes,
        paper_titles=paper_titles,
    )
    pack_path.write_text(pack_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "analysis_dir": str(summary_path.parent),
                "paper_count": len(paper_keys),
                "summary_path": str(summary_path),
                "triage_path": str(triage_path),
                "pack_path": str(pack_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
