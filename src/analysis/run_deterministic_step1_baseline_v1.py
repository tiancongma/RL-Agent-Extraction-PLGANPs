#!/usr/bin/env python3
from __future__ import annotations

"""
Run the deterministic Step 1 baseline chain on a manifest-driven scope.

This runner is orchestration-only. It reuses the existing deterministic
comparator/fallback chain, records an explicit run contract, and enforces
identity freeze as a hard gate.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stage2_sampling_labels.emit_semantic_objects_from_cleaned_papers_v1 import (
    build_document,
    supported_paper_keys,
)
from src.utils.paths import DATA_RESULTS_DIR, DEV15_LAYER2_IDENTITY_TSV, PROJECT_ROOT
from src.utils.run_id import resolve_results_write_target


DEFAULT_BASELINE_FINAL_TSV = (
    DATA_RESULTS_DIR
    / "run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1"
    / "lineage"
    / "children"
    / "44_stage5_descendant_fix_v1fixed_recompare"
    / "run_20260321_1454_5fa3ed0_dev15_stage5_descendant_fix_v1fixed_recompare_v1"
    / "final_formulation_table_v1.tsv"
)

SEMANTIC_DIR_NAME = "semantic_stage2_objects"
COMPAT_DIR_NAME = "semantic_to_widerow_adapter"
RELATION_DIR_NAME = "formulation_relation_v1"
SCAFFOLD_DIR_NAME = "audit/layer2_identity_scaffold_binding_v1"
FREEZE_DIR_NAME = "audit/identity_freeze_guardrail_v1"


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


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def run_cmd(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def selected_manifest_rows(
    manifest_rows: list[dict[str, str]],
    requested_keys: list[str],
) -> list[dict[str, str]]:
    if not requested_keys:
        return list(manifest_rows)
    wanted = {normalize_text(key) for key in requested_keys if normalize_text(key)}
    scoped_rows = [row for row in manifest_rows if normalize_text(row.get("key")) in wanted]
    found = {normalize_text(row.get("key")) for row in scoped_rows}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Requested paper keys not found in manifest: {missing}")
    return scoped_rows


def eligibility_row(record: dict[str, str], supported_keys: set[str]) -> dict[str, str]:
    key = normalize_text(record.get("key"))
    text_path_text = normalize_text(record.get("text_path"))
    if not text_path_text:
        return {
            "key": key,
            "doi": normalize_text(record.get("doi")),
            "text_path": "",
            "eligibility_status": "skipped",
            "reason_code": "no_text_path_in_manifest",
            "eligibility_reason": "Manifest row does not expose a cleaned full-text path.",
        }

    text_path = Path(text_path_text)
    if not text_path.is_absolute():
        text_path = PROJECT_ROOT / text_path
    if not text_path.exists():
        return {
            "key": key,
            "doi": normalize_text(record.get("doi")),
            "text_path": text_path_text,
            "eligibility_status": "skipped",
            "reason_code": "text_path_missing_on_disk",
            "eligibility_reason": "Manifest declares cleaned text, but the file is not present on disk.",
        }

    if key not in supported_keys:
        return {
            "key": key,
            "doi": normalize_text(record.get("doi")),
            "text_path": text_path_text,
            "eligibility_status": "skipped",
            "reason_code": "unsupported_by_deterministic_emitter",
            "eligibility_reason": "Cleaned text is present, but this paper has no governed deterministic builder in the emitter.",
        }

    try:
        build_document(record)
    except Exception as exc:
        return {
            "key": key,
            "doi": normalize_text(record.get("doi")),
            "text_path": text_path_text,
            "eligibility_status": "skipped",
            "reason_code": "builder_preflight_failed",
            "eligibility_reason": f"Deterministic builder preflight failed: {exc}",
        }

    return {
        "key": key,
        "doi": normalize_text(record.get("doi")),
        "text_path": text_path_text,
        "eligibility_status": "eligible",
        "reason_code": "supported_and_preflight_passed",
        "eligibility_reason": "Supported deterministic builder plus required cleaned assets resolved successfully.",
    }


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    manifest_tsv: Path,
    scoped_manifest_tsv: Path,
    eligible_manifest_tsv: Path,
    inventory_tsv: Path,
    baseline_final_tsv: Path,
    commands: list[list[str]],
    processed_keys: list[str],
    skipped_rows: list[dict[str, str]],
    final_row_count: int,
    freeze_status: str,
    run_status: str,
    failure_note: str,
) -> str:
    command_block = "\n".join(" ".join(command) for command in commands)
    skipped_lines = (
        [f"- `{row['key']}`: `{row['reason_code']}` - {row['eligibility_reason']}" for row in skipped_rows]
        if skipped_rows
        else ["- none"]
    )
    processed_lines = [f"- `{key}`" for key in processed_keys] if processed_keys else ["- none"]
    return (
        "\n".join(
            [
                "# RUN_CONTEXT",
                "",
                "## 1. Run ID",
                "",
                f"- `{run_id}`",
                "",
                "## 2. Run type",
                "",
                "- `component_regression_run`",
                "- `deterministic_step1_baseline_run`",
                f"- run_dir_kind: `{run_dir_kind}`",
                f"- run_selection_mode: `{run_selection_mode}`",
                f"- bucket_dir: `{bucket_dir}`",
                "",
                "## 3. Purpose",
                "",
                "- Execute deterministic Step 1 only: manifest-driven formulation-boundary reconstruction, Stage5 final-table materialization, and mandatory identity-freeze validation.",
                "- This run is comparator/rules-only and makes no LLM or external API calls.",
                "",
                "## 4. Starting inputs",
                "",
                f"- manifest_tsv: `{repo_rel(manifest_tsv)}`",
                f"- scoped_manifest_tsv: `{repo_rel(scoped_manifest_tsv)}`",
                f"- eligible_manifest_tsv: `{repo_rel(eligible_manifest_tsv)}`",
                f"- eligibility_inventory_tsv: `{repo_rel(inventory_tsv)}`",
                f"- gt_identity_tsv: `{repo_rel(DEV15_LAYER2_IDENTITY_TSV)}`",
                f"- scaffold_baseline_final_tsv: `{repo_rel(baseline_final_tsv)}`",
                "",
                "## 5. Exact script execution order",
                "",
                "1. Resolve the manifest-driven scope and preflight deterministic eligibility.",
                "2. Run `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py` on the eligible subset only.",
                "3. Run `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`.",
                "4. Run `src/stage3_relation/build_formulation_relation_artifacts_v1.py`.",
                "5. Run `src/stage5_benchmark/build_minimal_final_output_v1.py`.",
                "6. Run `src/stage5_benchmark/build_layer2_identity_scaffold_binding_v1.py`.",
                "7. Run `src/stage5_benchmark/enforce_identity_freeze_v1.py` as a hard gate.",
                "",
                "## 6. Commands used",
                "",
                "```powershell",
                command_block,
                "```",
                "",
                "## 7. Processed papers",
                "",
                *processed_lines,
                "",
                "## 8. Skipped papers",
                "",
                *skipped_lines,
                "",
                "## 9. Output status",
                "",
                f"- `{run_status}`",
                "- `diagnostic-only, not benchmark-valid final output`",
                "- Reason: this Step 1 run ends at frozen Stage5 output plus identity-freeze validation and does not execute benchmark GT comparison.",
                f"- identity_freeze_result: `{freeze_status}`",
                f"- final_formulation_row_count: `{final_row_count}`",
                f"- failure_note: `{failure_note or 'none'}`",
                "",
                "## 10. Final artifacts",
                "",
                f"- `{repo_rel(run_dir / 'final_formulation_table_v1.tsv')}`",
                f"- `{repo_rel(run_dir / 'downstream_variant_records_v1.tsv')}`",
                f"- `{repo_rel(run_dir / 'final_output_decision_trace_v1.tsv')}`",
                f"- `{repo_rel(run_dir / SCAFFOLD_DIR_NAME / 'layer2_identity_scaffold_rows_v1.tsv')}`",
                f"- `{repo_rel(run_dir / FREEZE_DIR_NAME / 'identity_freeze_summary_v1.tsv')}`",
                f"- `{repo_rel(run_dir / 'RUN_CONTEXT.md')}`",
            ]
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic Step 1 baseline on a manifest-driven scope.")
    parser.add_argument("--manifest-tsv", required=True, type=Path)
    parser.add_argument("--paper-key", action="append", default=[], help="Optional repeatable key filter.")
    parser.add_argument("--run-dir", default="", help="Explicit results run directory.")
    parser.add_argument("--run-id", default="", help="Explicit legacy-compatible run_id or child name.")
    parser.add_argument(
        "--execution-cue",
        default="deterministic_step1_baseline",
        help="Cue used when auto-allocating a v2 child execution directory.",
    )
    parser.add_argument(
        "--identity-baseline-final-tsv",
        type=Path,
        default=DEFAULT_BASELINE_FINAL_TSV,
        help="Frozen baseline final table used by the scaffold-binding helper.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_tsv = args.manifest_tsv.resolve()
    baseline_final_tsv = args.identity_baseline_final_tsv.resolve()
    if not manifest_tsv.exists():
        raise FileNotFoundError(f"Manifest TSV not found: {manifest_tsv}")
    if baseline_final_tsv != DEFAULT_BASELINE_FINAL_TSV.resolve():
        raise ValueError(
            "Identity scaffold baseline override is not allowed for this governed runner. "
            f"Expected {DEFAULT_BASELINE_FINAL_TSV.resolve()}, got {baseline_final_tsv}."
        )
    if not baseline_final_tsv.exists():
        raise FileNotFoundError(f"Identity scaffold baseline final TSV not found: {baseline_final_tsv}")

    target = resolve_results_write_target(
        results_root=DATA_RESULTS_DIR,
        default_child_cue=args.execution_cue,
        explicit_run_dir=args.run_dir,
        explicit_legacy_run_id=args.run_id,
    )
    run_id = target["run_basename"]
    run_dir = Path(target["run_dir"])
    run_dir_kind = target["path_kind"]
    run_selection_mode = target["selection_mode"]
    bucket_dir = Path(target["bucket_dir"])
    if run_dir_kind == "v2_child_execution":
        bucket_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest_rows = read_tsv_rows(manifest_tsv)
    if not manifest_rows:
        raise RuntimeError(f"Manifest TSV contains no rows: {manifest_tsv}")
    scoped_rows = selected_manifest_rows(manifest_rows, args.paper_key)
    supported_keys = set(supported_paper_keys())
    inventory_rows = [eligibility_row(row, supported_keys) for row in scoped_rows]
    eligible_rows = [
        row for row, inventory in zip(scoped_rows, inventory_rows, strict=True) if inventory["eligibility_status"] == "eligible"
    ]
    skipped_rows = [row for row in inventory_rows if row["eligibility_status"] != "eligible"]
    processed_keys = [normalize_text(row.get("key")) for row in eligible_rows]

    if not processed_keys:
        raise RuntimeError("No eligible papers remain after deterministic Step 1 scope resolution.")

    scoped_manifest_tsv = run_dir / "scoped_manifest.tsv"
    eligible_manifest_tsv = run_dir / "eligible_scope_manifest.tsv"
    inventory_tsv = run_dir / "eligible_corpus_inventory.tsv"
    requested_keys_tsv = run_dir / "requested_scope_keys.tsv"

    manifest_fieldnames = list(manifest_rows[0].keys())
    write_tsv(scoped_manifest_tsv, manifest_fieldnames, scoped_rows)
    write_tsv(eligible_manifest_tsv, manifest_fieldnames, eligible_rows)
    write_tsv(requested_keys_tsv, ["paper_key"], [{"paper_key": key} for key in processed_keys])
    write_tsv(
        inventory_tsv,
        ["key", "doi", "text_path", "eligibility_status", "reason_code", "eligibility_reason"],
        inventory_rows,
    )

    semantic_dir = run_dir / SEMANTIC_DIR_NAME
    compat_dir = run_dir / COMPAT_DIR_NAME
    relation_dir = run_dir / RELATION_DIR_NAME
    scaffold_dir = run_dir / SCAFFOLD_DIR_NAME
    freeze_dir = run_dir / FREEZE_DIR_NAME

    commands = [
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "emit_semantic_objects_from_cleaned_papers_v1.py"),
            "--manifest-tsv",
            str(eligible_manifest_tsv),
            "--out-dir",
            str(semantic_dir),
            "--paper-keys",
            *processed_keys,
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "build_stage2_compatibility_projection_v1.py"),
            "--input-jsonl",
            str(semantic_dir / "semantic_stage2_objects_v1.jsonl"),
            "--output-dir",
            str(compat_dir),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "stage3_relation" / "build_formulation_relation_artifacts_v1.py"),
            "--weak-labels-tsv",
            str(compat_dir / "weak_labels__v7pilot_r3_fixparse.tsv"),
            "--weak-labels-jsonl",
            str(compat_dir / "weak_labels__v7pilot_r3_fixparse.jsonl"),
            "--scope-manifest-tsv",
            str(eligible_manifest_tsv),
            "--out-dir",
            str(relation_dir),
        ],
        [
            sys.executable,
            "-m",
            "src.stage5_benchmark.build_minimal_final_output_v1",
            "--input-tsv",
            str(compat_dir / "weak_labels__v7pilot_r3_fixparse.tsv"),
            "--relation-records-tsv",
            str(relation_dir / "formulation_relation_records_v1.tsv"),
            "--resolved-relation-fields-tsv",
            str(relation_dir / "resolved_relation_fields_v1.tsv"),
            "--out-dir",
            str(run_dir),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "stage5_benchmark" / "build_layer2_identity_scaffold_binding_v1.py"),
            "--baseline-final-tsv",
            str(baseline_final_tsv),
            "--new-final-tsv",
            str(run_dir / "final_formulation_table_v1.tsv"),
            *[item for key in processed_keys for item in ("--paper-key", key)],
            "--out-dir",
            str(scaffold_dir),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "stage5_benchmark" / "enforce_identity_freeze_v1.py"),
            "--identity-scaffold-rows-tsv",
            str(scaffold_dir / "layer2_identity_scaffold_rows_v1.tsv"),
            "--final-table-tsv",
            str(run_dir / "final_formulation_table_v1.tsv"),
            *[item for key in processed_keys for item in ("--paper-key", key)],
            "--out-dir",
            str(freeze_dir),
        ],
    ]

    command_results: list[dict[str, str]] = []
    failure_note = ""
    run_status = "completed"
    for command in commands:
        try:
            completed = run_cmd(command, cwd=PROJECT_ROOT)
            command_results.append(
                {
                    "command": " ".join(command),
                    "status": "ok",
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
            )
        except subprocess.CalledProcessError as exc:
            run_status = "failed"
            failure_note = f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
            command_results.append(
                {
                    "command": " ".join(command),
                    "status": "failed",
                    "stdout": (exc.stdout or "").strip(),
                    "stderr": (exc.stderr or "").strip(),
                }
            )
            break

    final_rows = read_tsv_rows(run_dir / "final_formulation_table_v1.tsv") if (run_dir / "final_formulation_table_v1.tsv").exists() else []
    freeze_rows = read_tsv_rows(freeze_dir / "identity_freeze_summary_v1.tsv") if (freeze_dir / "identity_freeze_summary_v1.tsv").exists() else []
    freeze_status = "pass" if freeze_rows and all(row.get("status") == "pass" for row in freeze_rows) else "fail"

    run_context = build_run_context(
        run_id=run_id,
        run_dir=run_dir,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        manifest_tsv=manifest_tsv,
        scoped_manifest_tsv=scoped_manifest_tsv,
        eligible_manifest_tsv=eligible_manifest_tsv,
        inventory_tsv=inventory_tsv,
        baseline_final_tsv=baseline_final_tsv,
        commands=commands,
        processed_keys=processed_keys,
        skipped_rows=skipped_rows,
        final_row_count=len(final_rows),
        freeze_status=freeze_status,
        run_status=run_status,
        failure_note=failure_note,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")
    (run_dir / "command_execution_log_v1.json").write_text(json.dumps(command_results, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "manifest_tsv": str(manifest_tsv),
                "eligible_manifest_tsv": str(eligible_manifest_tsv),
                "papers_processed": len(processed_keys),
                "papers_skipped": len(skipped_rows),
                "final_formulation_rows": len(final_rows),
                "identity_freeze": freeze_status,
                "run_status": run_status,
                "failure_note": failure_note,
            },
            indent=2,
        )
    )
    return 0 if run_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
