#!/usr/bin/env python3
from __future__ import annotations

"""
Comparator-safe wrapper for building the value GT annotation workbook from an
explicit frozen run directory.

Purpose:
- reuse the maintained workbook builder internals for a non-default comparator run
- resolve comparator-local Stage5 and field-review artifacts directly from --run-dir
- preserve default ACTIVE_RUN.json behavior in the shared builder

This wrapper is supporting-only and must not be treated as a mainline benchmark
entrypoint.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from src.stage5_benchmark.build_value_gt_annotation_workbook_v1 import (
        MAIN_COLUMNS,
        REFERENCE_COLUMNS,
        TARGET_FIELD_ORDER,
        alignment_resolution_tsv_name_for_version,
        build_alignment_index,
        build_artifact_metadata,
        build_final_row_index,
        build_gt_skeleton_rows,
        build_main_rows,
        build_seed_index,
        build_workbook,
        extra_tsv_name_for_version,
        main_tsv_name_for_version,
        normalize_text,
        read_tsv_rows,
        reference_tsv_name_for_version,
        resolve_artifact_path,
        resolve_run_context,
        resolve_workbook_name,
        sanitize_out_subdir,
        validate_main_rows,
        write_artifact_metadata_json,
        write_tsv,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage5_benchmark.build_value_gt_annotation_workbook_v1 import (
        MAIN_COLUMNS,
        REFERENCE_COLUMNS,
        TARGET_FIELD_ORDER,
        alignment_resolution_tsv_name_for_version,
        build_alignment_index,
        build_artifact_metadata,
        build_final_row_index,
        build_gt_skeleton_rows,
        build_main_rows,
        build_seed_index,
        build_workbook,
        extra_tsv_name_for_version,
        main_tsv_name_for_version,
        normalize_text,
        read_tsv_rows,
        reference_tsv_name_for_version,
        resolve_artifact_path,
        resolve_run_context,
        resolve_workbook_name,
        sanitize_out_subdir,
        validate_main_rows,
        write_artifact_metadata_json,
        write_tsv,
    )


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve required frozen-run artifact from candidates: {rendered}")


def apply_comparator_safe_identity_repairs(main_rows: list[dict[str, object]]) -> None:
    for row in main_rows:
        paper_key = normalize_text(row.get("paper_key"))
        article_id = normalize_text(row.get("article_formulation_id"))
        seed_id = normalize_text(row.get("seed_pred_representative_source_formulation_id"))
        if paper_key == "BB3JUVW7" and not article_id and seed_id:
            row["article_formulation_id"] = seed_id
            if not normalize_text(row.get("article_formulation_label")):
                row["article_formulation_label"] = seed_id


def has_any_gt_skeleton_alignment(main_rows: list[dict[str, object]]) -> bool:
    for row in main_rows:
        if normalize_text(row.get("matched_system_formulation_id")):
            return True
        if normalize_text(row.get("l2_gt_alignment_status")) == "aligned":
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a value GT workbook for one explicit frozen run without changing shared ACTIVE_RUN resolution."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Explicit frozen run directory.")
    parser.add_argument(
        "--out-subdir",
        default="value_gt_v1",
        type=sanitize_out_subdir,
        help="Functional artifact subdirectory under the explicit run root.",
    )
    parser.add_argument("--artifact-version", type=int, default=1, help="Artifact version suffix.")
    parser.add_argument("--workbook-name", default=None, help="Optional explicit workbook filename.")
    parser.add_argument("--gt-skeleton-tsv", type=Path, default=None, help="Optional explicit Layer2 GT skeleton TSV override.")
    parser.add_argument(
        "--alignment-scaffold-tsv",
        type=Path,
        default=None,
        help="Optional explicit GT-to-system alignment scaffold TSV override.",
    )
    parser.add_argument(
        "--trusted-alignment-tsv",
        type=Path,
        default=None,
        help="Optional explicit trusted-alignment TSV override.",
    )
    args = parser.parse_args()

    run_context = resolve_run_context(
        explicit_run_dir=args.run_dir.resolve(),
        explicit_run_id="",
    )
    run_dir = Path(run_context["run_dir"])
    run_id = str(run_context["run_id"])
    out_dir = run_dir / args.out_subdir

    audit_ready_tsv = first_existing_path(
        run_dir / "final_formulation_table_audit_ready_v1.tsv",
        run_dir / "audit_ready" / "final_formulation_table_audit_ready_v1.tsv",
    )
    seed_rows_tsv = first_existing_path(
        run_dir / "field_gt_review_v1" / "field_gt_review_seed_rows_v1.tsv",
        run_dir / "field_gt_review_seed_rows_v1.tsv",
    )
    final_table_tsv = first_existing_path(
        run_dir / "final_formulation_table_v1.tsv",
        run_dir / "final_output" / "final_formulation_table_v1.tsv",
    )

    gt_skeleton_tsv = (
        args.gt_skeleton_tsv.resolve()
        if args.gt_skeleton_tsv is not None
        else resolve_artifact_path(
            explicit_path=None,
            run_context=run_context,
            pointer_key="gt_skeleton_tsv",
            required=True,
        )
    )
    alignment_scaffold_tsv = (
        args.alignment_scaffold_tsv.resolve()
        if args.alignment_scaffold_tsv is not None
        else resolve_artifact_path(
            explicit_path=None,
            run_context=run_context,
            pointer_key="alignment_scaffold_tsv",
            required=True,
        )
    )
    trusted_alignment_tsv = (
        args.trusted_alignment_tsv.resolve()
        if args.trusted_alignment_tsv is not None
        else resolve_artifact_path(
            explicit_path=None,
            run_context=run_context,
            pointer_key="trusted_alignment_tsv",
            required=False,
        )
    )

    print(
        json.dumps(
            {
                "resolved_source_run_dir": str(run_dir),
                "resolved_source_run_id": run_id,
                "source_resolution": str(run_context["resolution_source"]),
                "active_run_pointer_path": str(run_context.get("pointer_path") or ""),
                "resolved_input_files": {
                    "audit_ready_tsv": str(audit_ready_tsv),
                    "seed_rows_tsv": str(seed_rows_tsv),
                    "final_table_tsv": str(final_table_tsv),
                    "gt_skeleton_tsv": str(gt_skeleton_tsv),
                    "alignment_scaffold_tsv": str(alignment_scaffold_tsv),
                    "trusted_alignment_tsv": str(trusted_alignment_tsv) if trusted_alignment_tsv else "",
                },
            },
            indent=2,
        )
    )

    audit_rows = read_tsv_rows(audit_ready_tsv)
    seed_rows = read_tsv_rows(seed_rows_tsv)
    final_rows = read_tsv_rows(final_table_tsv)
    gt_rows = read_tsv_rows(gt_skeleton_tsv)
    alignment_rows = read_tsv_rows(alignment_scaffold_tsv)
    trusted_alignment_rows = read_tsv_rows(trusted_alignment_tsv) if trusted_alignment_tsv else []

    seed_index = build_seed_index(seed_rows)
    final_row_index = build_final_row_index(final_rows)
    alignment_index, extra_alignment_rows = build_alignment_index(alignment_rows)
    build_mode = "gt_skeleton"

    main_rows, reference_rows, extra_rows, preserved_nonempty_cells, alignment_resolution_rows = build_gt_skeleton_rows(
        gt_rows=gt_rows,
        alignment_index=alignment_index,
        extra_alignment_rows=extra_alignment_rows,
        audit_rows=audit_rows,
        seed_index=seed_index,
        final_row_index=final_row_index,
        prior_rows=[],
        trusted_alignment_rows=trusted_alignment_rows,
    )

    apply_comparator_safe_identity_repairs(main_rows)
    if has_any_gt_skeleton_alignment(main_rows):
        validate_main_rows(main_rows, alignment_resolution_rows)
    else:
        build_mode = "system_centric_fallback"
        main_rows, reference_rows = build_main_rows(audit_rows, seed_index, final_row_index)
        extra_rows = []
        alignment_resolution_rows = []
        preserved_nonempty_cells = 0

    main_tsv = out_dir / main_tsv_name_for_version(args.artifact_version)
    reference_tsv = out_dir / reference_tsv_name_for_version(args.artifact_version)
    extra_tsv = out_dir / extra_tsv_name_for_version(args.artifact_version)
    alignment_resolution_tsv = out_dir / alignment_resolution_tsv_name_for_version(args.artifact_version)
    workbook_path = out_dir / resolve_workbook_name(args.artifact_version, args.workbook_name)

    write_tsv(main_tsv, MAIN_COLUMNS, main_rows)
    write_tsv(reference_tsv, REFERENCE_COLUMNS, reference_rows)
    if extra_rows:
        write_tsv(extra_tsv, MAIN_COLUMNS, extra_rows)
    if alignment_resolution_rows:
        write_tsv(
            alignment_resolution_tsv,
            list(alignment_resolution_rows[0].keys()),
            alignment_resolution_rows,
        )
    build_workbook(workbook_path, main_rows, reference_rows, extra_rows)
    metadata_path = write_artifact_metadata_json(
        workbook_path,
        build_artifact_metadata(
            source_run_context=run_context,
            source_files={
                "audit_ready_tsv": str(audit_ready_tsv),
                "seed_rows_tsv": str(seed_rows_tsv),
                "final_table_tsv": str(final_table_tsv),
                "gt_skeleton_tsv": str(gt_skeleton_tsv),
                "alignment_scaffold_tsv": str(alignment_scaffold_tsv),
                "trusted_alignment_tsv": str(trusted_alignment_tsv) if trusted_alignment_tsv else "",
            },
            generated_by="src/stage5_benchmark/run_value_workbook_for_explicit_run_v1.py",
            note="Comparator-safe explicit-run wrapper around the maintained value GT workbook builder.",
            extra={
                "main_rows_tsv": str(main_tsv),
                "reference_rows_tsv": str(reference_tsv),
                "extra_rows_tsv": str(extra_tsv) if extra_rows else "",
                "alignment_resolution_tsv": str(alignment_resolution_tsv) if alignment_resolution_rows else "",
                "underlying_builder": "src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py",
                "build_mode": build_mode,
            },
        ),
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "final_table_tsv": str(final_table_tsv),
                "main_tsv": str(main_tsv),
                "reference_tsv": str(reference_tsv),
                "extra_tsv": str(extra_tsv) if extra_rows else "",
                "alignment_resolution_tsv": str(alignment_resolution_tsv) if alignment_resolution_rows else "",
                "workbook_path": str(workbook_path),
                "workbook_metadata_json": str(metadata_path),
                "formulation_rows": len(main_rows),
                "reference_rows": len(reference_rows),
                "extra_in_system_rows": len(extra_rows),
                "preserved_nonempty_cells": preserved_nonempty_cells,
                "trusted_alignment_tsv": str(trusted_alignment_tsv) if trusted_alignment_tsv else "",
                "build_mode": build_mode,
                "target_fields": TARGET_FIELD_ORDER,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
