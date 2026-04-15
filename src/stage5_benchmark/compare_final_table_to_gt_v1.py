#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from src.utils.active_data_source import (
        build_artifact_metadata,
        resolve_artifact_path,
        resolve_run_context,
        write_artifact_metadata_json,
    )
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.active_data_source import (
        build_artifact_metadata,
        resolve_artifact_path,
        resolve_run_context,
        write_artifact_metadata_json,
    )
    from src.utils.paths import PROJECT_ROOT


COUNTS_NAME = "final_table_vs_gt_counts.tsv"
SUMMARY_NAME = "final_table_vs_gt_summary.md"
EE_SUBSET_NAME = "final_table_vs_gt_ee_subset.tsv"
AUDIT_COUNTS_NAME = "final_table_vs_gt_counts_by_doi.tsv"
AUDIT_SUMMARY_NAME = "final_table_vs_gt_count_audit.md"
IDENTITY_FREEZE_SUMMARY_NAME = "identity_freeze_summary_v1.tsv"


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def read_scope_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_final_table_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_gt_counts_from_tsv(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    counts: dict[str, int] = {}
    for row in rows:
        paper_key = str(row.get("paper_key", "")).strip()
        if not paper_key:
            continue
        counts[paper_key] = int(str(row.get("gt_count", "0")).strip() or "0")
    return counts


def read_identity_freeze_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_identity_freeze(summary_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], bool]:
    by_paper: dict[str, dict[str, str]] = {}
    has_violations = False
    for row in summary_rows:
        paper_key = str(row.get("paper_key", "")).strip()
        if not paper_key:
            continue
        by_paper[paper_key] = row
        if normalize_text(row.get("violation")) == "yes":
            has_violations = True
    return by_paper, has_violations


def build_summary_markdown(
    scope_name: str,
    manifest_path: Path,
    final_table_path: Path,
    gt_counts_tsv_path: Path,
    counts_rows: list[dict[str, str]],
    ee_supported: bool,
    identity_freeze_mode: str,
    identity_freeze_summary_tsv: Path,
    benchmark_valid: bool,
    identity_freeze_failed: bool,
    summary_path: Path,
) -> None:
    totals = {
        "final_table_rows": sum(int(row["final_table_count"]) for row in counts_rows),
        "gt_rows": sum(int(row["gt_count"]) for row in counts_rows),
        "matched_papers": sum(1 for row in counts_rows if row["comparison_status"] == "match"),
        "mismatched_papers": sum(1 for row in counts_rows if row["comparison_status"] != "match"),
    }
    lines = [
        "# Final Table vs GT Summary",
        "",
        "## Declared scope",
        "",
        f"- scope_name: `{scope_name}`",
        f"- scope_manifest_tsv: `{manifest_path}`",
        f"- final_formulation_table_tsv: `{final_table_path}`",
        f"- gt_counts_tsv: `{gt_counts_tsv_path}`",
        f"- identity_freeze_summary_tsv: `{identity_freeze_summary_tsv}`",
        "",
        "## Compare Contract",
        "",
        f"- identity_freeze_mode: `{identity_freeze_mode}`",
        f"- benchmark_valid: `{'yes' if benchmark_valid else 'no'}`",
        f"- identity_freeze_failed: `{'yes' if identity_freeze_failed else 'no'}`",
        "",
        "## Benchmark-validity statement",
        "",
    ]
    if benchmark_valid:
        lines.extend(
            [
                "- This comparison is benchmark-valid for the declared scope because it evaluates only the complete-pipeline final formulation table after identity freeze passed.",
                "- No intermediate Stage2 or other partial-layer artifacts are used as the official evaluation object.",
            ]
        )
    else:
        lines.extend(
            [
                "- This comparison is diagnostic-only because identity freeze failed and compare was explicitly continued in `debug_identity` mode.",
                "- No intermediate Stage2 or other partial-layer artifacts are used as the evaluation object, but this output is not legal benchmark evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Supported benchmark views",
            "",
            "- per-DOI final-formulation count comparison: supported",
            f"- EE subset comparison: {'supported' if ee_supported else 'not supported by the current authoritative GT artifact'}",
            "",
            "## Aggregate outcome",
            "",
            f"- total_final_table_rows: `{totals['final_table_rows']}`",
            f"- total_gt_rows: `{totals['gt_rows']}`",
            f"- matched_papers: `{totals['matched_papers']}`",
            f"- mismatched_papers: `{totals['mismatched_papers']}`",
            "",
            "## Per-paper counts",
            "",
        ]
    )
    for row in counts_rows:
        lines.append(
            "- `{paper_key}`: final=`{final_table_count}` gt=`{gt_count}` diff=`{count_diff}` status=`{comparison_status}` freeze_failed=`{identity_freeze_failed}`".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The current comparison is limited to final-formulation counts for this declared scope.",
            "- The authoritative fixed DEV15 skeleton workbook does not expose structured EE ground-truth fields, so no benchmark-valid EE subset comparison is emitted in this first full-pipeline run.",
            "- Any mismatch investigation must start from these final-table results and only then trace backward into Stage 5A decision-trace artifacts or Stage 2 candidate rows.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_count_audit_markdown(
    counts_rows: list[dict[str, str]],
    gt_counts_tsv_path: Path,
    final_table_path: Path,
    audit_summary_path: Path,
) -> None:
    exact = sum(1 for row in counts_rows if row["count_status"] == "match")
    plus = sum(1 for row in counts_rows if row["count_status"] == "pred_gt_plus")
    minus = sum(1 for row in counts_rows if row["count_status"] == "pred_gt_minus")
    mismatches = sorted(
        [row for row in counts_rows if row["count_status"] != "match"],
        key=lambda row: abs(int(row["delta_count"])),
        reverse=True,
    )
    lines = [
        "# DEV15 GT vs Pred Count Audit",
        "",
        f"- gt_authority_file: `{gt_counts_tsv_path}`",
        f"- pred_source_file: `{final_table_path}`",
        f"- total_doi_count: `{len(counts_rows)}`",
        f"- exact_matches: `{exact}`",
        f"- positive_deltas: `{plus}`",
        f"- negative_deltas: `{minus}`",
        "",
        "## Top mismatches",
        "",
    ]
    if not mismatches:
        lines.append("- No DOI-level count mismatches.")
    else:
        for row in mismatches:
            lines.append(
                "- `{doi}` / `{paper_key}`: gt=`{gt_count}` pred=`{pred_count}` delta=`{delta_count}` status=`{count_status}`".format(
                    **row
                )
            )
    audit_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_run_context(
    *,
    run_dir: Path,
    source_run_context: dict[str, Any],
    final_table_tsv: Path,
    gt_counts_tsv: Path,
    scope_manifest_tsv: Path,
    identity_freeze_summary_tsv: Path,
    out_dir: Path,
    scope_name: str,
    identity_freeze_mode: str,
    benchmark_valid: bool,
    identity_freeze_failed: bool,
    result: dict[str, Any],
) -> str:
    generated_at = datetime.now().isoformat(timespec="seconds")
    return "\n".join(
        [
            "# RUN_CONTEXT",
            "",
            "## 1. Run Type",
            "",
            f"- `{'full_pipeline_benchmark_run' if benchmark_valid else 'intermediate_diagnostic_run'}`",
            "",
            "## 2. Purpose",
            "",
            "- Compare the completed Stage 5 final table against the authoritative frozen Layer1 GT counts TSV for the declared scope.",
            "- Respect the explicit dual-mode identity-freeze contract before writing compare outputs.",
            "",
            "## 3. Source Authority Resolution",
            "",
            f"- source_resolution: `{source_run_context['resolution_source']}`",
            f"- source_run_id: `{source_run_context['run_id']}`",
            f"- source_run_dir: `{source_run_context['run_dir']}`",
            f"- active_run_pointer_path: `{source_run_context.get('pointer_path') or ''}`",
            f"- scope_manifest_tsv: `{scope_manifest_tsv}`",
            f"- final_table_tsv: `{final_table_tsv}`",
            f"- layer1_gt_counts_tsv: `{gt_counts_tsv}`",
            f"- identity_freeze_summary_tsv: `{identity_freeze_summary_tsv}`",
            "",
            "## 4. Exact Script Execution Order",
            "",
            f"1. Run `src/stage5_benchmark/compare_final_table_to_gt_v1.py --identity-freeze-mode {identity_freeze_mode}` on the completed Stage 5 final table.",
            "2. Refresh `RUN_CONTEXT.md` via `src/utils/update_run_context_with_feature_activation_v1.py` so feature activation lineage is recorded in the compare run.",
            "",
            "## 5. Outputs",
            "",
            f"- `{out_dir / COUNTS_NAME}`",
            f"- `{out_dir / SUMMARY_NAME}`",
            f"- `{out_dir / AUDIT_COUNTS_NAME}`",
            f"- `{out_dir / AUDIT_SUMMARY_NAME}`",
            f"- `{out_dir / 'RUN_CONTEXT.md'}`",
            "",
            "## 6. Benchmark Status",
            "",
            f"- compare_mode: `{identity_freeze_mode}`",
            f"- benchmark_valid: `{'yes' if benchmark_valid else 'no'}`",
            f"- identity_freeze_failed: `{'yes' if identity_freeze_failed else 'no'}`",
            (
                "- `benchmark-valid`"
                if benchmark_valid
                else "- `diagnostic-only, not benchmark-valid final output`"
            ),
            (
                "- Reason: identity freeze passed, so benchmark compare remained lawful."
                if benchmark_valid
                else "- Reason: identity freeze failed and compare was explicitly continued in `debug_identity` mode."
            ),
            "",
            "## 7. Reproduction Metadata",
            "",
            f"- generated_at: `{generated_at}`",
            f"- scope_name: `{scope_name}`",
            f"- final_table_rows: `{result['total_final_table_rows']}`",
            f"- gt_rows: `{result['total_gt_rows']}`",
            f"- matched_papers: `{result['papers_matching']}`",
            f"- mismatched_papers: `{result['papers_mismatching']}`",
        ]
    ) + "\n"


def compare_final_table_to_gt(
    final_table_tsv: Path,
    gt_counts_tsv: Path,
    scope_manifest_tsv: Path,
    identity_freeze_summary_tsv: Path,
    identity_freeze_mode: str,
    identity_freeze_by_paper: dict[str, dict[str, str]],
    benchmark_valid: bool,
    out_dir: Path,
    scope_name: str,
    source_run_context: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_scope_manifest(scope_manifest_tsv)
    scope_by_key = {row["key"]: row for row in manifest_rows}
    scope_keys = set(scope_by_key)

    final_rows = [
        row for row in read_final_table_rows(final_table_tsv) if row.get("key", "") in scope_keys
    ]
    final_counts = Counter(row["key"] for row in final_rows)
    gt_counts_all = read_gt_counts_from_tsv(gt_counts_tsv)
    gt_counts = Counter({key: gt_counts_all.get(key, 0) for key in scope_keys})
    ee_supported = False
    identity_freeze_failed = any(
        normalize_text(row.get("violation")) == "yes" for row in identity_freeze_by_paper.values()
    )

    counts_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for key in sorted(scope_keys):
        manifest_row = scope_by_key[key]
        freeze_row = identity_freeze_by_paper.get(key, {})
        freeze_failed = normalize_text(freeze_row.get("violation")) == "yes"
        final_count = int(final_counts.get(key, 0))
        gt_count = int(gt_counts.get(key, 0))
        diff = final_count - gt_count
        if diff == 0:
            status = "match"
        elif diff > 0:
            status = "over"
        else:
            status = "under"
        count_status = "match" if diff == 0 else ("pred_gt_plus" if diff > 0 else "pred_gt_minus")
        counts_rows.append(
            {
                "paper_key": key,
                "doi": manifest_row.get("doi", ""),
                "paper_title": manifest_row.get("title", ""),
                "final_table_count": str(final_count),
                "gt_count": str(gt_count),
                "count_diff": str(diff),
                "comparison_status": status,
                "identity_freeze_failed": "yes" if freeze_failed else "no",
                "compare_mode": identity_freeze_mode,
                "benchmark_valid": "yes" if benchmark_valid else "no",
                "final_table_artifact": str(final_table_tsv),
                "gt_artifact": str(gt_counts_tsv),
                "notes": (
                    "count_match"
                    if status == "match"
                    else "final_table_vs_fixed_skeleton_count_mismatch"
                ),
            }
        )
        audit_rows.append(
            {
                "doi": manifest_row.get("doi", ""),
                "paper_key": key,
                "gt_count": str(gt_count),
                "pred_count": str(final_count),
                "delta_count": str(diff),
                "count_status": count_status,
                "identity_freeze_failed": "yes" if freeze_failed else "no",
                "compare_mode": identity_freeze_mode,
                "benchmark_valid": "yes" if benchmark_valid else "no",
                "gt_authority_file": str(gt_counts_tsv),
                "pred_source_file": str(final_table_tsv),
                "matched_rows": "",
                "missing_rows": "",
                "spurious_rows": "",
            }
        )

    counts_path = out_dir / COUNTS_NAME
    summary_path = out_dir / SUMMARY_NAME
    audit_counts_path = out_dir / AUDIT_COUNTS_NAME
    audit_summary_path = out_dir / AUDIT_SUMMARY_NAME
    write_tsv(
        counts_path,
        [
            "paper_key",
            "doi",
            "paper_title",
            "final_table_count",
            "gt_count",
            "count_diff",
            "comparison_status",
            "identity_freeze_failed",
            "compare_mode",
            "benchmark_valid",
            "final_table_artifact",
            "gt_artifact",
            "notes",
        ],
        counts_rows,
    )
    write_tsv(
        audit_counts_path,
        [
            "doi",
            "paper_key",
            "gt_count",
            "pred_count",
            "delta_count",
            "count_status",
            "identity_freeze_failed",
            "compare_mode",
            "benchmark_valid",
            "gt_authority_file",
            "pred_source_file",
            "matched_rows",
            "missing_rows",
            "spurious_rows",
        ],
        audit_rows,
    )
    build_summary_markdown(
        scope_name=scope_name,
        manifest_path=scope_manifest_tsv,
        final_table_path=final_table_tsv,
        gt_counts_tsv_path=gt_counts_tsv,
        counts_rows=counts_rows,
        ee_supported=ee_supported,
        identity_freeze_mode=identity_freeze_mode,
        identity_freeze_summary_tsv=identity_freeze_summary_tsv,
        benchmark_valid=benchmark_valid,
        identity_freeze_failed=identity_freeze_failed,
        summary_path=summary_path,
    )
    build_count_audit_markdown(
        counts_rows=audit_rows,
        gt_counts_tsv_path=gt_counts_tsv,
        final_table_path=final_table_tsv,
        audit_summary_path=audit_summary_path,
    )

    result = {
        "scope_name": scope_name,
        "final_table_path": str(final_table_tsv),
        "gt_counts_tsv_path": str(gt_counts_tsv),
        "identity_freeze_summary_tsv": str(identity_freeze_summary_tsv),
        "counts_path": str(counts_path),
        "summary_path": str(summary_path),
        "audit_counts_path": str(audit_counts_path),
        "audit_summary_path": str(audit_summary_path),
        "ee_subset_path": "",
        "ee_subset_supported": ee_supported,
        "papers_in_scope": len(scope_keys),
        "papers_matching": sum(1 for row in counts_rows if row["comparison_status"] == "match"),
        "papers_mismatching": sum(1 for row in counts_rows if row["comparison_status"] != "match"),
        "total_final_table_rows": sum(int(row["final_table_count"]) for row in counts_rows),
        "total_gt_rows": sum(int(row["gt_count"]) for row in counts_rows),
        "identity_freeze_mode": identity_freeze_mode,
        "benchmark_valid": benchmark_valid,
        "identity_freeze_failed": identity_freeze_failed,
    }
    (out_dir / "RUN_CONTEXT.md").write_text(
        build_run_context(
            run_dir=out_dir,
            source_run_context=source_run_context,
            final_table_tsv=final_table_tsv,
            gt_counts_tsv=gt_counts_tsv,
            scope_manifest_tsv=scope_manifest_tsv,
            identity_freeze_summary_tsv=identity_freeze_summary_tsv,
            out_dir=out_dir,
            scope_name=scope_name,
            identity_freeze_mode=identity_freeze_mode,
            benchmark_valid=benchmark_valid,
            identity_freeze_failed=identity_freeze_failed,
            result=result,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "utils" / "update_run_context_with_feature_activation_v1.py"),
            "--run-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        check=True,
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a final formulation table against the authoritative frozen Layer1 GT counts TSV."
    )
    parser.add_argument("--final-table-tsv", type=Path, default=None)
    parser.add_argument("--gt-counts-tsv", type=Path, default=None)
    parser.add_argument(
        "--gt-xlsx",
        type=Path,
        default=None,
        help="Deprecated legacy compare input; blocked when GT authority lock is enabled.",
    )
    parser.add_argument("--scope-manifest-tsv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--scope-name", default="controlled_scope")
    parser.add_argument(
        "--identity-freeze-mode",
        choices=["benchmark", "debug_identity"],
        default="benchmark",
        help="Explicit compare contract mode. `benchmark` blocks compare on failed identity freeze; `debug_identity` writes diagnostic-only compare outputs.",
    )
    parser.add_argument(
        "--identity-freeze-summary-tsv",
        type=Path,
        default=None,
        help="Optional explicit identity-freeze summary TSV. Defaults to <run_dir>/audit/identity_freeze_guardrail_v1/identity_freeze_summary_v1.tsv.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_context = resolve_run_context(
        explicit_run_dir=args.run_dir,
        explicit_run_id=str(args.run_id or "").strip(),
    )
    final_table_tsv = resolve_artifact_path(
        explicit_path=args.final_table_tsv,
        run_context=run_context,
        pointer_key="stage5_final_table_tsv",
        canonical_relative="final_formulation_table_v1.tsv",
    )
    gt_counts_tsv = resolve_artifact_path(
        explicit_path=args.gt_counts_tsv,
        run_context=run_context,
        pointer_key="layer1_gt_path",
    )
    if args.gt_xlsx is not None:
        raise ValueError(
            "--gt-xlsx is no longer the authoritative Layer1 compare input. "
            "Use --gt-counts-tsv or rely on ACTIVE_RUN.json layer1_gt_path."
        )
    scope_manifest_tsv = resolve_artifact_path(
        explicit_path=args.scope_manifest_tsv,
        run_context=run_context,
        pointer_key="scope_manifest_tsv",
        preferred_run_local_names=["dev15_scope.tsv", "scope.tsv", "scope_manifest.tsv"],
    )
    if args.identity_freeze_summary_tsv is not None:
        identity_freeze_summary_tsv = args.identity_freeze_summary_tsv.resolve()
    else:
        identity_freeze_summary_tsv = (
            Path(run_context["run_dir"]) / "audit" / "identity_freeze_guardrail_v1" / IDENTITY_FREEZE_SUMMARY_NAME
        ).resolve()
    if not identity_freeze_summary_tsv.exists():
        raise FileNotFoundError(
            "Identity-freeze summary TSV is required for compare. "
            f"Expected: {identity_freeze_summary_tsv}"
        )
    identity_freeze_rows = read_identity_freeze_summary(identity_freeze_summary_tsv)
    identity_freeze_by_paper, identity_freeze_failed = analyze_identity_freeze(identity_freeze_rows)
    benchmark_valid = args.identity_freeze_mode == "benchmark" and not identity_freeze_failed
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (Path(run_context["run_dir"]) / "gt_authority_v2_variantaware").resolve()
    )
    print(
        json.dumps(
            {
                "resolved_source_run_dir": str(run_context["run_dir"]),
                "resolved_source_run_id": str(run_context["run_id"]),
                "source_resolution": str(run_context["resolution_source"]),
                "active_run_pointer_path": str(run_context.get("pointer_path") or ""),
                "resolved_input_files": {
                    "final_table_tsv": str(final_table_tsv),
                    "gt_counts_tsv": str(gt_counts_tsv),
                    "scope_manifest_tsv": str(scope_manifest_tsv),
                    "identity_freeze_summary_tsv": str(identity_freeze_summary_tsv),
                },
                "resolved_out_dir": str(out_dir),
                "identity_freeze_mode": args.identity_freeze_mode,
                "identity_freeze_failed": identity_freeze_failed,
                "benchmark_valid": benchmark_valid,
            },
            indent=2,
        )
    )
    if args.identity_freeze_mode == "benchmark" and identity_freeze_failed:
        raise SystemExit(
            "Benchmark compare blocked: identity freeze failed. "
            "Use --identity-freeze-mode debug_identity for diagnostic-only compare outputs."
        )
    result = compare_final_table_to_gt(
        final_table_tsv=final_table_tsv,
        gt_counts_tsv=gt_counts_tsv,
        scope_manifest_tsv=scope_manifest_tsv,
        identity_freeze_summary_tsv=identity_freeze_summary_tsv,
        identity_freeze_mode=args.identity_freeze_mode,
        identity_freeze_by_paper=identity_freeze_by_paper,
        benchmark_valid=benchmark_valid,
        out_dir=out_dir,
        scope_name=args.scope_name,
        source_run_context=run_context,
    )
    metadata_path = write_artifact_metadata_json(
        Path(result["counts_path"]),
        build_artifact_metadata(
            source_run_context=run_context,
            source_files={
                "final_table_tsv": str(final_table_tsv),
                "gt_counts_tsv": str(gt_counts_tsv),
                "scope_manifest_tsv": str(scope_manifest_tsv),
                "identity_freeze_summary_tsv": str(identity_freeze_summary_tsv),
            },
            generated_by="src/stage5_benchmark/compare_final_table_to_gt_v1.py",
            note="Stage5 final-table vs GT comparison authority metadata.",
            extra={
                "summary_path": str(result["summary_path"]),
                "audit_counts_path": str(result["audit_counts_path"]),
                "audit_summary_path": str(result["audit_summary_path"]),
                "scope_name": args.scope_name,
                "identity_freeze_mode": args.identity_freeze_mode,
                "benchmark_valid": benchmark_valid,
                "identity_freeze_failed": identity_freeze_failed,
            },
        ),
    )
    result["counts_metadata_json"] = str(metadata_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
