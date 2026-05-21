#!/usr/bin/env python3
from __future__ import annotations

"""
Daily baseline audit governance utility.

This utility is intentionally additive and read-oriented:
- it snapshots explicit existing run artifacts
- it validates explicit declared maintained chains
- it compares baseline vs rerun artifact surfaces

It does not execute pipeline stages and it does not define a hidden
orchestration layer.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from src.utils.active_data_source import resolve_run_context
    from src.utils.build_feature_activation_report_v1 import (
        build_report_rows,
        compute_activation_gate,
        load_matrix,
        load_registry,
    )
    from src.utils.paths import (
        DATA_CLEANED_CONTENT_DIR,
        DATA_CLEANED_INDEX_DIR,
        DOCS_DIR,
        PROJECT_DIR,
        PROJECT_ROOT,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.active_data_source import resolve_run_context
    from src.utils.build_feature_activation_report_v1 import (
        build_report_rows,
        compute_activation_gate,
        load_matrix,
        load_registry,
    )
    from src.utils.paths import (
        DATA_CLEANED_CONTENT_DIR,
        DATA_CLEANED_INDEX_DIR,
        DOCS_DIR,
        PROJECT_DIR,
        PROJECT_ROOT,
    )


AUDIT_ROOT_SUBPATH = Path("audit") / "daily_baseline_v1"
MAINTAINED_SURFACE_TSV = DOCS_DIR / "maintained_script_surface.tsv"
FEATURE_REGISTRY_JSON = PROJECT_DIR / "feature_units" / "feature_unit_registry.json"
FEATURE_MATRIX_TSV = PROJECT_DIR / "feature_units" / "feature_intervention_matrix.tsv"
SCOPE_CONTRACT_TSV = DOCS_DIR / "feature_governance" / "daily_audit_scope_contract_v1.tsv"

STATUS_MAINTAINED_ENTRYPOINT = "maintained_entrypoint"
STATUS_SUPPORTING_NONDEFAULT = "supporting_nondefault"
FINAL_JUDGMENT_COMPARABLE = "comparable"
FINAL_JUDGMENT_NOT_COMPARABLE = "not comparable"
COMPARISON_STATUS_COMPARABLE_CLEAN = "comparable_clean"
COMPARISON_STATUS_COMPARABLE_WITH_SCOPE_VIOLATION = "comparable_with_scope_violation"
COMPARISON_STATUS_NOT_COMPARABLE = "not_comparable"
SCOPE_ASSESSMENT_UNCHANGED = "unchanged"
SCOPE_ASSESSMENT_ALLOWED_CHANGED = "allowed_changed"
SCOPE_ASSESSMENT_WARNING_CHANGED = "warning_changed"
SCOPE_ASSESSMENT_OUT_OF_SCOPE = "out_of_scope_drift"
SCOPE_ASSESSMENT_NOT_DECLARED = "scope_not_declared"

STAGE2_FAMILY_NAMES = [
    "stage2_candidate_blocks",
    "stage2_evidence_blocks",
    "stage2_prompt_artifacts",
    "stage2_raw_responses",
    "stage2_semantic_objects",
    "stage2_compat_projection",
]
STAGE3_FAMILY_NAMES = ["stage3_relation_artifacts"]
STAGE5_FAMILY_NAMES = ["stage5_final_outputs", "stage5_compare_outputs"]

COMMON_RUN_CONTEXT_KEYS = [
    "manifest_tsv",
    "scope_manifest_tsv",
    "weak_labels_tsv",
    "weak_labels_jsonl",
    "candidate_input_tsv",
    "relation_records_tsv",
    "resolved_relation_fields_tsv",
    "final_table_tsv",
    "gt_counts_tsv",
    "layer1_gt_counts_tsv",
    "legacy_raw_responses_dir",
    "source_run_dir",
]
COMMAND_PATH_FLAGS = [
    "--manifest-tsv",
    "--scope-manifest-tsv",
    "--gt-xlsx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Governance-support daily baseline audit utility."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser(
        "start_baseline_audit",
        help="Freeze a baseline snapshot from explicit authority and an explicit maintained chain.",
    )
    start_parser.add_argument("--baseline-id", required=True)
    source_group = start_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-run-dir", type=Path, default=None)
    source_group.add_argument("--use-active-run", action="store_true")
    start_parser.add_argument("--scope-name", required=True)
    chain_group = start_parser.add_mutually_exclusive_group(required=True)
    chain_group.add_argument(
        "--declared-chain-json",
        default="",
        help="JSON string or path to a JSON file containing an explicit ordered list of maintained script paths.",
    )
    chain_group.add_argument(
        "--declared-script",
        action="append",
        dest="declared_scripts",
        default=[],
        help="Repeatable maintained script path. Order matters.",
    )

    scope_parser = subparsers.add_parser(
        "declare_modification_scope",
        help="Declare the intended engineering modification scope for an existing baseline audit.",
    )
    scope_parser.add_argument("--baseline-id", required=True)
    scope_parser.add_argument("--scope-id", required=True)
    scope_parser.add_argument("--audit-dir", type=Path, required=True)
    scope_parser.add_argument("--notes", default="")

    delta_parser = subparsers.add_parser(
        "build_layered_delta_report",
        help="Compare a baseline audit snapshot against an explicit rerun artifact surface.",
    )
    delta_parser.add_argument("--baseline-id", required=True)
    delta_parser.add_argument("--baseline-audit-dir", type=Path, required=True)
    delta_parser.add_argument("--rerun-run-dir", type=Path, required=True)

    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def parse_bool(value: Any) -> bool:
    return normalize_text(value).lower() in {"true", "yes", "1"}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_rows_for_path(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return None


def fingerprint_path(path: Path) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {
        "path": repo_rel(path),
        "exists": exists,
        "path_type": "missing",
        "size_bytes": None,
        "sha256": "",
        "row_count": None,
    }
    if not exists:
        return payload
    if path.is_dir():
        hasher = hashlib.sha256()
        file_count = 0
        for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item).lower()):
            child_rel = str(child.relative_to(path)).replace("\\", "/")
            hasher.update(child_rel.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(sha256_file(child).encode("utf-8"))
            hasher.update(b"\n")
            file_count += 1
        payload["path_type"] = "directory"
        payload["sha256"] = hasher.hexdigest()
        payload["size_bytes"] = file_count
        return payload
    payload["path_type"] = "file"
    payload["size_bytes"] = path.stat().st_size
    payload["sha256"] = sha256_file(path)
    payload["row_count"] = count_rows_for_path(path)
    return payload


def is_lineage_child_path(run_dir: Path, path: Path) -> bool:
    rel_parts = path.relative_to(run_dir).parts
    return len(rel_parts) >= 2 and rel_parts[0] == "lineage" and rel_parts[1] == "children"


def find_run_paths(run_dir: Path, name: str) -> list[Path]:
    matches: list[Path] = []
    for path in run_dir.rglob(name):
        if is_lineage_child_path(run_dir, path):
            continue
        matches.append(path)
    return sorted(matches, key=lambda item: (len(item.relative_to(run_dir).parts), str(item).lower()))


def first_run_path(run_dir: Path, name: str) -> Path | None:
    matches = find_run_paths(run_dir, name)
    return matches[0] if matches else None


def first_run_file(run_dir: Path, name: str) -> Path | None:
    candidate = first_run_path(run_dir, name)
    if candidate is None or not candidate.is_file():
        return None
    return candidate


def parse_run_context_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        if ": `" not in stripped:
            continue
        key_part, value_part = stripped[2:].split(": `", 1)
        key = key_part.strip("` ").strip()
        value = value_part.rsplit("`", 1)[0].strip()
        if key:
            values[key] = value
    return values


def parse_script_order_from_run_context(text: str) -> list[str]:
    scripts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped[:2].isdigit() and not stripped[:1].isdigit():
            continue
        for segment in stripped.split("`"):
            if segment.startswith("src/") and segment.endswith(".py"):
                scripts.append(segment)
    return scripts


def classify_benchmark_status(run_dir: Path, run_context_text: str) -> dict[str, str]:
    lowered = run_context_text.lower()
    if (
        "gt_subset_role: `mainline_integration_diagnostic`" in lowered
        or "diagnostic_status: `full_pipeline_gt_diagnostic`" in lowered
    ):
        return {
            "run_type_label": "gt_diagnostic",
            "benchmark_valid": "no",
            "status_source": "run_context",
        }
    if "diagnostic-only, not benchmark-valid final output" in lowered:
        return {
            "run_type_label": "diagnostic_only",
            "benchmark_valid": "no",
            "status_source": "run_context",
        }
    if "benchmark-valid final output" in lowered:
        return {
            "run_type_label": "benchmark_valid",
            "benchmark_valid": "yes",
            "status_source": "run_context",
        }
    compare_path = first_run_file(run_dir, "final_table_vs_gt_counts.tsv")
    if compare_path is not None:
        return {
            "run_type_label": "compare_present_status_unknown",
            "benchmark_valid": "unknown",
            "status_source": "compare_artifact_only",
        }
    return {
        "run_type_label": "status_unknown",
        "benchmark_valid": "unknown",
        "status_source": "missing_explicit_status",
    }


def load_maintained_surface() -> dict[str, dict[str, str]]:
    rows = read_tsv(MAINTAINED_SURFACE_TSV)
    return {row["script_path"]: row for row in rows if normalize_text(row.get("script_path"))}


def parse_declared_chain(args: argparse.Namespace) -> list[str]:
    if args.declared_scripts:
        chain = [normalize_text(item) for item in args.declared_scripts if normalize_text(item)]
        if not chain:
            raise ValueError("At least one --declared-script must be provided.")
        return chain
    raw = normalize_text(args.declared_chain_json)
    if not raw:
        raise ValueError("Missing declared chain input.")
    candidate = Path(raw)
    payload = read_json(candidate) if candidate.exists() else json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Declared chain JSON must be a non-empty JSON list of script paths.")
    chain = [normalize_text(item) for item in payload if normalize_text(item)]
    if not chain:
        raise ValueError("Declared chain JSON resolved to an empty script list.")
    return chain


def validate_declared_chain(declared_chain: list[str]) -> list[dict[str, str]]:
    maintained = load_maintained_surface()
    validated_rows: list[dict[str, str]] = []
    for order, script_path in enumerate(declared_chain, start=1):
        surface_row = maintained.get(script_path)
        if surface_row is None:
            raise ValueError(
                f"Declared chain script is not present in docs/maintained_script_surface.tsv: {script_path}"
            )
        status = normalize_text(surface_row.get("status"))
        if status not in {STATUS_MAINTAINED_ENTRYPOINT, STATUS_SUPPORTING_NONDEFAULT}:
            raise ValueError(
                f"Declared chain script is not an allowed maintained surface: {script_path} "
                f"(status={surface_row.get('status', '')})"
            )
        validated_rows.append(
            {
                "order_index": str(order),
                "script_path": script_path,
                "stage_or_workflow": surface_row.get("stage_or_workflow", ""),
                "role": surface_row.get("role", ""),
                "status": surface_row.get("status", ""),
                "must_use_active_data_source_contract": surface_row.get(
                    "must_use_active_data_source_contract", ""
                ),
            }
        )
    return validated_rows


def resolve_source_context_for_baseline(
    *,
    source_run_dir: Path | None,
    use_active_run: bool,
) -> dict[str, Any]:
    if use_active_run:
        return resolve_run_context(explicit_run_dir=None, explicit_run_id="")
    if source_run_dir is None:
        raise ValueError("Either --source-run-dir or --use-active-run is required.")
    return resolve_run_context(explicit_run_dir=source_run_dir.resolve(), explicit_run_id="")


def resolve_repo_or_run_path(run_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    repo_candidate = (PROJECT_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (run_dir / candidate).resolve()


def extract_command_flag_paths(run_context_text: str, run_dir: Path) -> dict[str, Path]:
    extracted: dict[str, Path] = {}
    for flag in COMMAND_PATH_FLAGS:
        pattern = re.compile(rf"{re.escape(flag)}\s+([^\s`]+)")
        match = pattern.search(run_context_text)
        if match is None:
            continue
        extracted[flag] = resolve_repo_or_run_path(run_dir, match.group(1).strip())
    return extracted


def read_two_column_tsv(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not path.exists() or not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            rows.append((normalize_text(row[0]), normalize_text(row[1])))
    return rows


def common_existing_parent(paths: list[Path]) -> Path | None:
    existing_parents = [path.parent.resolve() for path in paths if path.exists()]
    if not existing_parents:
        return None
    common_path = Path(os.path.commonpath([str(path) for path in existing_parents]))
    return common_path.resolve() if common_path.exists() else None


def resolve_cleaned_text_root(key2txt_path: Path) -> Path | None:
    pairs = read_two_column_tsv(key2txt_path)
    candidate_paths = [resolve_repo_or_run_path(PROJECT_ROOT, raw_path) for _, raw_path in pairs if raw_path]
    common_root = common_existing_parent(candidate_paths)
    if common_root is not None:
        return common_root
    fallback = DATA_CLEANED_CONTENT_DIR
    return fallback.resolve() if fallback.exists() else None


def resolve_table_assets_root(manifest_current_path: Path) -> Path | None:
    if not manifest_current_path.exists() or not manifest_current_path.is_file():
        return None
    with manifest_current_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        table_dirs = [
            resolve_repo_or_run_path(PROJECT_ROOT, row["table_dir"])
            for row in reader
            if normalize_text(row.get("table_dir"))
        ]
    return common_existing_parent(table_dirs)


def fingerprint_input_component(name: str, path: Path) -> dict[str, Any]:
    payload = fingerprint_path(path)
    return {
        "name": name,
        "path": payload["path"],
        "type": "dir" if payload["path_type"] == "directory" else "file" if payload["path_type"] == "file" else "missing",
        "fingerprint": payload["sha256"],
        "exists": bool(payload["exists"]),
    }


def build_input_surface_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["exists"]]
    missing_names = [row["name"] for row in rows if not row["exists"]]
    warnings: list[str] = []
    if not valid_rows:
        warnings.append(
            "No valid upstream input components were resolved. This baseline is invalid for comparability."
        )
    if missing_names:
        warnings.append(
            "Some declared input components were missing: " + ", ".join(missing_names)
        )
    return {
        "component_count": len(rows),
        "valid_component_count": len(valid_rows),
        "missing_component_names": missing_names,
        "baseline_valid_for_comparability": "yes" if valid_rows else "no",
        "warnings": warnings,
    }


def collect_input_surface(
    run_dir: Path,
    run_context_values: dict[str, str],
    run_context_text: str,
) -> dict[str, Any]:
    command_flag_paths = extract_command_flag_paths(run_context_text, run_dir)
    manifest_current_path = (DATA_CLEANED_INDEX_DIR / "manifest_current.tsv").resolve()
    key2txt_path = (DATA_CLEANED_INDEX_DIR / "key2txt.tsv").resolve()
    scope_manifest_path: Path | None = None

    raw_scope_manifest = normalize_text(run_context_values.get("scope_manifest_tsv"))
    if raw_scope_manifest:
        scope_manifest_path = resolve_repo_or_run_path(run_dir, raw_scope_manifest)
    elif "--scope-manifest-tsv" in command_flag_paths:
        scope_manifest_path = command_flag_paths["--scope-manifest-tsv"]

    candidate_blocks_dir = first_run_path(run_dir, "candidate_blocks")
    evidence_blocks_dir = first_run_path(run_dir, "evidence_blocks")
    cleaned_text_root = resolve_cleaned_text_root(key2txt_path)
    table_assets_root = resolve_table_assets_root(manifest_current_path)

    component_specs: list[tuple[str, Path | None]] = [
        ("manifest_current_tsv", manifest_current_path),
        ("key2txt_tsv", key2txt_path),
        ("cleaned_text_root", cleaned_text_root),
        ("scope_manifest_tsv", scope_manifest_path),
        ("stage2_evidence_blocks", evidence_blocks_dir),
        ("stage2_candidate_blocks", candidate_blocks_dir),
        ("table_assets_root", table_assets_root),
    ]
    rows: list[dict[str, Any]] = []
    for name, path in component_specs:
        if path is None:
            rows.append(
                {
                    "name": name,
                    "path": "",
                    "type": "missing",
                    "fingerprint": "",
                    "exists": False,
                }
            )
            continue
        rows.append(fingerprint_input_component(name, path.resolve()))
    summary = build_input_surface_summary(rows)
    return {
        "input_surface": rows,
        "summary": summary,
    }


def missing_fingerprint() -> dict[str, Any]:
    return {
        "path": "",
        "exists": False,
        "path_type": "missing",
        "size_bytes": None,
        "sha256": "",
        "row_count": None,
    }


def build_stage_output_inventory(run_dir: Path) -> dict[str, Any]:
    families = {
        "stage2_candidate_blocks": {
            "candidate_blocks_dir": first_run_path(run_dir, "candidate_blocks"),
            "normalized_table_payloads_dir": first_run_path(run_dir, "normalized_table_payloads"),
            "candidate_segmentation_debug_tsv": first_run_file(run_dir, "candidate_segmentation_debug_v1.tsv"),
            "table_authority_validation_tsv": first_run_file(run_dir, "table_authority_validation_v1.tsv"),
        },
        "stage2_evidence_blocks": {
            "evidence_blocks_dir": first_run_path(run_dir, "evidence_blocks"),
            "table_selection_debug_json": first_run_file(run_dir, "table_selection_debug_v1.json"),
        },
        "stage2_prompt_artifacts": {
            "prompt_preview_tsv": first_run_file(run_dir, "stage2_prompt_preview_v1.tsv"),
            "s2_4a_prompt_template_txt": first_run_file(run_dir, "s2_4a_prompt_template_v1.txt"),
            "s2_4a_prompts_jsonl": first_run_file(run_dir, "s2_4a_prompts_v1.jsonl"),
            "s2_4a_prompt_audit_tsv": first_run_file(run_dir, "s2_4a_prompt_audit_v1.tsv"),
        },
        "stage2_raw_responses": {
            "raw_responses_dir": first_run_path(run_dir, "raw_responses"),
            "request_metadata_dir": first_run_path(run_dir, "request_metadata"),
            "s2_4b_request_summary_tsv": first_run_file(run_dir, "s2_4b_request_summary_v1.tsv"),
        },
        "stage2_semantic_objects": {
            "semantic_objects_jsonl": first_run_file(run_dir, "semantic_stage2_v2_objects.jsonl"),
            "semantic_summary_tsv": first_run_file(run_dir, "semantic_stage2_v2_summary.tsv"),
            "semantic_contract_report_json": first_run_file(
                run_dir, "stage2_semantic_authority_contract_report_v1.json"
            ),
        },
        "stage2_compat_projection": {
            "completed_stage2_tsv": first_run_file(run_dir, "weak_labels__v7pilot_r3_fixparse.tsv"),
            "completed_stage2_jsonl": first_run_file(run_dir, "weak_labels__v7pilot_r3_fixparse.jsonl"),
            "compatibility_summary_json": first_run_file(run_dir, "compatibility_projection_summary_v1.json"),
            "compatibility_trace_tsv": first_run_file(run_dir, "compatibility_projection_trace_v1.tsv"),
        },
        "stage3_relation_artifacts": {
            "relation_records_tsv": first_run_file(run_dir, "formulation_relation_records_v1.tsv"),
            "relation_summary_tsv": first_run_file(run_dir, "formulation_relation_summary_v1.tsv"),
            "resolved_relation_fields_tsv": first_run_file(run_dir, "resolved_relation_fields_v1.tsv"),
        },
        # Stage5 remains artifact-first here. The audit layer consumes explicit
        # Stage5 artifacts and any wrapper/run-context surfaces already present,
        # rather than turning build_minimal_final_output_v1.py into a run-manager.
        "stage5_final_outputs": {
            "final_table_tsv": first_run_file(run_dir, "final_formulation_table_v1.tsv"),
            "decision_trace_tsv": first_run_file(run_dir, "final_output_decision_trace_v1.tsv"),
            "downstream_variant_records_tsv": first_run_file(run_dir, "downstream_variant_records_v1.tsv"),
            "final_output_summary_md": first_run_file(run_dir, "final_output_summary_v1.md"),
        },
        "stage5_compare_outputs": {
            "final_table_vs_gt_counts_tsv": first_run_file(run_dir, "final_table_vs_gt_counts.tsv"),
            "final_table_vs_gt_summary_md": first_run_file(run_dir, "final_table_vs_gt_summary.md"),
        },
    }
    inventory: dict[str, Any] = {"run_dir": repo_rel(run_dir), "families": {}}
    for family_name, artifact_map in families.items():
        family_rows: dict[str, Any] = {}
        for artifact_key, artifact_path in artifact_map.items():
            family_rows[artifact_key] = (
                fingerprint_path(artifact_path) if artifact_path is not None else missing_fingerprint()
            )
        inventory["families"][family_name] = family_rows
    return inventory


def build_feature_activation_snapshot(run_dir: Path, target_path: Path) -> dict[str, Any]:
    registry = load_registry(FEATURE_REGISTRY_JSON)
    matrix = load_matrix(FEATURE_MATRIX_TSV)
    rows = build_report_rows(registry=registry, matrix=matrix, run_dir=run_dir)
    write_tsv(
        target_path,
        [
            "feature_id",
            "expected_for_run",
            "observed_activation",
            "activation_status",
            "activation_state",
            "evidence_path",
            "evidence_detail",
            "notes",
        ],
        rows,
    )
    gate = compute_activation_gate(rows)
    return {
        "row_count": len(rows),
        "required_feature_units": gate["required_feature_units"],
        "missing_required_feature_units": gate["missing_required_feature_units"],
        "unclear_required_feature_units": gate["unclear_required_feature_units"],
        "run_activation_gate": gate["run_activation_gate"],
    }


def audit_dir_for(source_run_dir: Path, baseline_id: str) -> Path:
    return source_run_dir / AUDIT_ROOT_SUBPATH / baseline_id


def require_baseline_manifest(audit_dir: Path, baseline_id: str) -> dict[str, Any]:
    manifest_path = audit_dir / "audit_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Baseline audit manifest not found for baseline_id={baseline_id}: {manifest_path}"
        )
    manifest = read_json(manifest_path)
    if normalize_text(manifest.get("baseline_id")) != baseline_id:
        raise ValueError(
            f"Baseline audit manifest baseline_id mismatch. expected={baseline_id} "
            f"found={manifest.get('baseline_id', '')}"
        )
    return manifest


def explode_json_list(raw_value: str) -> list[str]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scope contract value must be valid JSON list: {raw_value}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Scope contract value must decode to a list: {raw_value}")
    return [normalize_text(item) for item in parsed if normalize_text(item)]


def rows_by_feature_id(path: Path) -> dict[str, dict[str, str]]:
    return {row["feature_id"]: row for row in read_tsv(path) if normalize_text(row.get("feature_id"))}


def load_declared_scope(audit_dir: Path) -> dict[str, Any] | None:
    scope_path = audit_dir / "modification_scope.json"
    if not scope_path.exists():
        return None
    return read_json(scope_path)


def family_has_change(delta_rows: list[dict[str, Any]]) -> bool:
    for row in delta_rows:
        if row.get("notes") not in {
            "artifact fingerprints match",
            "artifact contents match but path differs",
            "artifact missing in both baseline and rerun",
        }:
            return True
    return False


def assess_family_scope(
    *,
    family_name: str,
    family_changed: bool,
    scope_payload: dict[str, Any] | None,
) -> tuple[str, str]:
    if not family_changed:
        return SCOPE_ASSESSMENT_UNCHANGED, "family remained unchanged"
    if scope_payload is None:
        return (
            SCOPE_ASSESSMENT_NOT_DECLARED,
            "no declared modification scope was available for this baseline audit",
        )

    warning_families = set(scope_payload.get("warn_on_changed_artifact_families", []))
    allowed_families = set(scope_payload.get("allowed_changed_artifact_families", []))
    expected_unchanged = set(scope_payload.get("expected_unchanged_artifact_families", []))

    if family_name in warning_families:
        return (
            SCOPE_ASSESSMENT_WARNING_CHANGED,
            "family changed within a warning-tier surface and should trigger audit review",
        )
    if family_name in allowed_families:
        return SCOPE_ASSESSMENT_ALLOWED_CHANGED, "family changed within the declared scope"
    if family_name in expected_unchanged:
        return (
            SCOPE_ASSESSMENT_OUT_OF_SCOPE,
            "family changed even though the declared scope expected this surface to remain unchanged",
        )
    return (
        SCOPE_ASSESSMENT_OUT_OF_SCOPE,
        "family changed outside the declared scope contract",
    )


def derive_artifact_delta_note(
    *,
    baseline_exists: bool,
    rerun_exists: bool,
    path_match: bool,
    sha_match: bool,
    row_count_match: bool,
) -> str:
    if not baseline_exists and not rerun_exists:
        return "artifact missing in both baseline and rerun"
    if baseline_exists and not rerun_exists:
        return "artifact present in baseline but missing in rerun"
    if not baseline_exists and rerun_exists:
        return "artifact missing in baseline but present in rerun"
    if sha_match and row_count_match and not path_match:
        return "artifact contents match but path differs"
    if path_match and sha_match and row_count_match:
        return "artifact fingerprints match"
    mismatches: list[str] = []
    if not path_match:
        mismatches.append("path differs")
    if not sha_match:
        mismatches.append("sha256 differs")
    if not row_count_match:
        mismatches.append("row count differs")
    return "; ".join(mismatches)


def compare_file_fingerprint_rows(
    *,
    baseline_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
    key_fields: list[str],
    label_field: str,
) -> list[dict[str, Any]]:
    baseline_map = {
        tuple(normalize_text(row.get(field)) for field in key_fields): row
        for row in baseline_rows
    }
    rerun_map = {
        tuple(normalize_text(row.get(field)) for field in key_fields): row
        for row in rerun_rows
    }
    all_keys = sorted(set(baseline_map) | set(rerun_map))
    delta_rows: list[dict[str, Any]] = []
    for key in all_keys:
        baseline_row = baseline_map.get(key, {})
        rerun_row = rerun_map.get(key, {})
        baseline_exists = normalize_text(baseline_row.get("exists", "false")) == "true"
        rerun_exists = normalize_text(rerun_row.get("exists", "false")) == "true"
        path_match = normalize_text(baseline_row.get("path")) == normalize_text(rerun_row.get("path"))
        sha_match = normalize_text(baseline_row.get("sha256")) == normalize_text(rerun_row.get("sha256"))
        row_count_match = str(baseline_row.get("row_count")) == str(rerun_row.get("row_count"))
        delta_rows.append(
            {
                label_field: key[0] if len(key) == 1 else "::".join(key),
                "baseline_path": baseline_row.get("path", ""),
                "rerun_path": rerun_row.get("path", ""),
                "baseline_exists": "yes" if baseline_exists else "no",
                "rerun_exists": "yes" if rerun_exists else "no",
                "path_match": "yes" if path_match else "no",
                "sha256_match": "yes" if sha_match else "no",
                "row_count_match": "yes" if row_count_match else "no",
                "baseline_row_count": baseline_row.get("row_count", ""),
                "rerun_row_count": rerun_row.get("row_count", ""),
                "notes": derive_artifact_delta_note(
                    baseline_exists=baseline_exists,
                    rerun_exists=rerun_exists,
                    path_match=path_match,
                    sha_match=sha_match,
                    row_count_match=row_count_match,
                ),
            }
        )
    return delta_rows


def compare_input_surface_rows(
    *,
    baseline_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_map = {normalize_text(row.get("name")): row for row in baseline_rows}
    rerun_map = {normalize_text(row.get("name")): row for row in rerun_rows}
    all_names = sorted(set(baseline_map) | set(rerun_map))
    delta_rows: list[dict[str, Any]] = []
    for name in all_names:
        baseline_row = baseline_map.get(name, {})
        rerun_row = rerun_map.get(name, {})
        baseline_exists = parse_bool(baseline_row.get("exists", False))
        rerun_exists = parse_bool(rerun_row.get("exists", False))
        path_match = normalize_text(baseline_row.get("path")) == normalize_text(rerun_row.get("path"))
        type_match = normalize_text(baseline_row.get("type")) == normalize_text(rerun_row.get("type"))
        fingerprint_match = normalize_text(baseline_row.get("fingerprint")) == normalize_text(
            rerun_row.get("fingerprint")
        )
        notes = "input component fingerprints match"
        if not baseline_exists and not rerun_exists:
            notes = "input component missing in both baseline and rerun"
        elif baseline_exists and not rerun_exists:
            notes = "input component present in baseline but missing in rerun"
        elif not baseline_exists and rerun_exists:
            notes = "input component missing in baseline but present in rerun"
        elif not path_match:
            notes = "input component path differs"
        elif not type_match:
            notes = "input component type differs"
        elif not fingerprint_match:
            notes = "input component fingerprint differs"
        delta_rows.append(
            {
                "input_name": name,
                "baseline_path": baseline_row.get("path", ""),
                "rerun_path": rerun_row.get("path", ""),
                "baseline_type": baseline_row.get("type", ""),
                "rerun_type": rerun_row.get("type", ""),
                "baseline_exists": "yes" if baseline_exists else "no",
                "rerun_exists": "yes" if rerun_exists else "no",
                "path_match": "yes" if path_match else "no",
                "type_match": "yes" if type_match else "no",
                "fingerprint_match": "yes" if fingerprint_match else "no",
                "notes": notes,
            }
        )
    return delta_rows


def write_stage_delta(
    *,
    baseline_inventory: dict[str, Any],
    rerun_inventory: dict[str, Any],
    family_names: list[str],
    output_path: Path,
    scope_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    baseline_rows = []
    rerun_rows = []
    delta_rows: list[dict[str, Any]] = []
    for family_name in family_names:
        for artifact_key, payload in baseline_inventory["families"][family_name].items():
            baseline_rows.append(
                {
                    "artifact_family": family_name,
                    "artifact_key": artifact_key,
                    "path": payload.get("path", ""),
                    "exists": str(payload.get("exists", False)).lower(),
                    "sha256": payload.get("sha256", ""),
                    "row_count": payload.get("row_count", ""),
                }
            )
        for artifact_key, payload in rerun_inventory["families"][family_name].items():
            rerun_rows.append(
                {
                    "artifact_family": family_name,
                    "artifact_key": artifact_key,
                    "path": payload.get("path", ""),
                    "exists": str(payload.get("exists", False)).lower(),
                    "sha256": payload.get("sha256", ""),
                    "row_count": payload.get("row_count", ""),
                }
            )
        family_baseline_rows = [row for row in baseline_rows if row["artifact_family"] == family_name]
        family_rerun_rows = [row for row in rerun_rows if row["artifact_family"] == family_name]
        family_delta_rows = compare_file_fingerprint_rows(
            baseline_rows=family_baseline_rows,
            rerun_rows=family_rerun_rows,
            key_fields=["artifact_key"],
            label_field="artifact_key",
        )
        family_scope_assessment, family_scope_notes = assess_family_scope(
            family_name=family_name,
            family_changed=family_has_change(family_delta_rows),
            scope_payload=scope_payload,
        )
        for row in family_delta_rows:
            row["artifact_family"] = family_name
            row["scope_assessment"] = family_scope_assessment
            row["scope_notes"] = family_scope_notes
        delta_rows.extend(family_delta_rows)
    write_tsv(
        output_path,
        [
            "artifact_family",
            "artifact_key",
            "baseline_path",
            "rerun_path",
            "baseline_exists",
            "rerun_exists",
            "path_match",
            "sha256_match",
            "row_count_match",
            "baseline_row_count",
            "rerun_row_count",
            "notes",
            "scope_assessment",
            "scope_notes",
        ],
        delta_rows,
    )
    return delta_rows


def print_command_result(
    *,
    payload: dict[str, Any],
    audit_dir: Path,
    written_files: list[Path],
) -> None:
    printable_payload = dict(payload)
    printable_payload["resolved_audit_dir"] = repo_rel(audit_dir)
    printable_payload["files_written"] = [repo_rel(path) for path in written_files]
    print(json.dumps(printable_payload, indent=2))


def command_start_baseline_audit(args: argparse.Namespace) -> None:
    declared_chain = parse_declared_chain(args)
    validated_chain = validate_declared_chain(declared_chain)
    source_context = resolve_source_context_for_baseline(
        source_run_dir=args.source_run_dir,
        use_active_run=args.use_active_run,
    )
    source_run_dir = Path(source_context["run_dir"]).resolve()
    run_context_path = source_run_dir / "RUN_CONTEXT.md"
    if not run_context_path.exists():
        raise FileNotFoundError(f"RUN_CONTEXT.md not found in source run directory: {run_context_path}")
    audit_dir = audit_dir_for(source_run_dir, args.baseline_id)
    if audit_dir.exists():
        raise FileExistsError(f"Baseline audit directory already exists: {audit_dir}")
    audit_dir.mkdir(parents=True, exist_ok=False)

    run_context_text = read_text(run_context_path)
    run_context_values = parse_run_context_values(run_context_text)
    actual_script_order = parse_script_order_from_run_context(run_context_text)
    benchmark_status = classify_benchmark_status(source_run_dir, run_context_text)

    input_surface_payload = collect_input_surface(source_run_dir, run_context_values, run_context_text)
    write_json(
        audit_dir / "baseline_input_fingerprint.json",
        {
            "baseline_id": args.baseline_id,
            "source_run_id": source_context["run_id"],
            "source_run_dir": repo_rel(source_run_dir),
            "input_surface": input_surface_payload["input_surface"],
            "summary": input_surface_payload["summary"],
        },
    )

    script_snapshot_payload = {
        "baseline_id": args.baseline_id,
        "declared_chain": validated_chain,
        "source_run_context_script_order": actual_script_order,
    }
    write_json(audit_dir / "script_chain_snapshot.json", script_snapshot_payload)

    feature_snapshot_path = audit_dir / "feature_activation_snapshot.tsv"
    feature_summary = build_feature_activation_snapshot(source_run_dir, feature_snapshot_path)

    stage_inventory = build_stage_output_inventory(source_run_dir)
    write_json(audit_dir / "stage_output_inventory.json", stage_inventory)

    manifest_payload = {
        "baseline_id": args.baseline_id,
        "scope_name": args.scope_name,
        "source_run_id": source_context["run_id"],
        "source_run_dir": repo_rel(source_run_dir),
        "source_resolution": source_context["resolution_source"],
        "active_run_pointer_path": normalize_text(source_context.get("pointer_path")),
        "run_context_path": repo_rel(run_context_path),
        "benchmark_status": benchmark_status,
        "input_surface_summary": input_surface_payload["summary"],
        "declared_chain_paths": declared_chain,
        "source_run_context_script_order": actual_script_order,
        "feature_activation_summary": feature_summary,
        "audit_artifact_paths": {
            "audit_manifest_json": repo_rel(audit_dir / "audit_manifest.json"),
            "baseline_input_fingerprint_json": repo_rel(audit_dir / "baseline_input_fingerprint.json"),
            "script_chain_snapshot_json": repo_rel(audit_dir / "script_chain_snapshot.json"),
            "feature_activation_snapshot_tsv": repo_rel(feature_snapshot_path),
            "stage_output_inventory_json": repo_rel(audit_dir / "stage_output_inventory.json"),
        },
    }
    write_json(audit_dir / "audit_manifest.json", manifest_payload)
    print_command_result(
        payload=manifest_payload,
        audit_dir=audit_dir,
        written_files=[
            audit_dir / "audit_manifest.json",
            audit_dir / "baseline_input_fingerprint.json",
            audit_dir / "script_chain_snapshot.json",
            feature_snapshot_path,
            audit_dir / "stage_output_inventory.json",
        ],
    )


def command_declare_modification_scope(args: argparse.Namespace) -> None:
    audit_dir = args.audit_dir.resolve()
    manifest = require_baseline_manifest(audit_dir, args.baseline_id)
    scope_rows = read_tsv(SCOPE_CONTRACT_TSV)
    scope_row = next((row for row in scope_rows if normalize_text(row.get("scope_id")) == args.scope_id), None)
    if scope_row is None:
        raise ValueError(f"Unknown scope_id in daily audit scope contract: {args.scope_id}")

    allowed_scripts = explode_json_list(scope_row["allowed_scripts"])
    forbidden_scripts = explode_json_list(scope_row["forbidden_scripts"])
    expected_unchanged = explode_json_list(scope_row["expected_unchanged_artifact_families"])
    allowed_changed = explode_json_list(scope_row["allowed_changed_artifact_families"])
    warn_on_changed = explode_json_list(scope_row.get("warn_on_changed_artifact_families", "[]"))

    modification_payload = {
        "baseline_id": args.baseline_id,
        "scope_id": args.scope_id,
        "scope_name": manifest["scope_name"],
        "notes": normalize_text(args.notes),
        "allowed_scripts": allowed_scripts,
        "forbidden_scripts": forbidden_scripts,
        "expected_unchanged_artifact_families": expected_unchanged,
        "allowed_changed_artifact_families": allowed_changed,
        "warn_on_changed_artifact_families": warn_on_changed,
        "contract_notes": scope_row.get("notes", ""),
    }
    write_json(audit_dir / "modification_scope.json", modification_payload)

    expectation_rows: list[dict[str, str]] = []
    for category, values in [
        ("allowed_scripts", allowed_scripts),
        ("forbidden_scripts", forbidden_scripts),
        ("expected_unchanged_artifact_families", expected_unchanged),
        ("allowed_changed_artifact_families", allowed_changed),
        ("warn_on_changed_artifact_families", warn_on_changed),
    ]:
        for value in values:
            expectation_rows.append(
                {
                    "baseline_id": args.baseline_id,
                    "scope_id": args.scope_id,
                    "expectation_type": category,
                    "value": value,
                    "notes": scope_row.get("notes", ""),
                }
            )
    write_tsv(
        audit_dir / "scope_expectation_matrix.tsv",
        ["baseline_id", "scope_id", "expectation_type", "value", "notes"],
        expectation_rows,
    )
    print_command_result(
        payload=modification_payload,
        audit_dir=audit_dir,
        written_files=[
            audit_dir / "modification_scope.json",
            audit_dir / "scope_expectation_matrix.tsv",
        ],
    )


def command_build_layered_delta_report(args: argparse.Namespace) -> None:
    baseline_audit_dir = args.baseline_audit_dir.resolve()
    manifest = require_baseline_manifest(baseline_audit_dir, args.baseline_id)
    rerun_context = resolve_run_context(explicit_run_dir=args.rerun_run_dir.resolve(), explicit_run_id="")
    rerun_run_dir = Path(rerun_context["run_dir"]).resolve()
    rerun_run_context_path = rerun_run_dir / "RUN_CONTEXT.md"
    if not rerun_run_context_path.exists():
        raise FileNotFoundError(f"RUN_CONTEXT.md not found in rerun run directory: {rerun_run_context_path}")

    baseline_input_payload = read_json(baseline_audit_dir / "baseline_input_fingerprint.json")
    baseline_script_payload = read_json(baseline_audit_dir / "script_chain_snapshot.json")
    baseline_stage_inventory = read_json(baseline_audit_dir / "stage_output_inventory.json")
    scope_payload = load_declared_scope(baseline_audit_dir)

    rerun_run_context_text = read_text(rerun_run_context_path)
    rerun_run_context_values = parse_run_context_values(rerun_run_context_text)
    rerun_benchmark_status = classify_benchmark_status(rerun_run_dir, rerun_run_context_text)
    rerun_input_payload = collect_input_surface(rerun_run_dir, rerun_run_context_values, rerun_run_context_text)
    rerun_script_order = parse_script_order_from_run_context(rerun_run_context_text)
    rerun_feature_snapshot_path = baseline_audit_dir / "_tmp_rerun_feature_activation_snapshot.tsv"
    feature_summary = build_feature_activation_snapshot(rerun_run_dir, rerun_feature_snapshot_path)
    rerun_feature_rows = read_tsv(rerun_feature_snapshot_path)
    rerun_feature_snapshot_path.unlink(missing_ok=True)
    rerun_stage_inventory = build_stage_output_inventory(rerun_run_dir)

    input_delta_rows = compare_input_surface_rows(
        baseline_rows=baseline_input_payload["input_surface"],
        rerun_rows=rerun_input_payload["input_surface"],
    )
    write_tsv(
        baseline_audit_dir / "input_surface_delta.tsv",
        [
            "input_name",
            "baseline_path",
            "rerun_path",
            "baseline_type",
            "rerun_type",
            "baseline_exists",
            "rerun_exists",
            "path_match",
            "type_match",
            "fingerprint_match",
            "notes",
        ],
        input_delta_rows,
    )

    script_delta_rows: list[dict[str, str]] = []
    declared_chain = baseline_script_payload["declared_chain"]
    allowed_scope_scripts = set(scope_payload.get("allowed_scripts", [])) if scope_payload else set()
    forbidden_scope_scripts = set(scope_payload.get("forbidden_scripts", [])) if scope_payload else set()
    max_len = max(len(declared_chain), len(rerun_script_order))
    for index in range(max_len):
        baseline_script = declared_chain[index]["script_path"] if index < len(declared_chain) else ""
        rerun_script = rerun_script_order[index] if index < len(rerun_script_order) else ""
        if baseline_script == rerun_script and baseline_script:
            scope_script_assessment = SCOPE_ASSESSMENT_UNCHANGED
            scope_script_notes = "script lineage remained unchanged"
        elif not rerun_script:
            scope_script_assessment = SCOPE_ASSESSMENT_OUT_OF_SCOPE
            scope_script_notes = "baseline declared script is missing from rerun script order"
        elif rerun_script in forbidden_scope_scripts:
            scope_script_assessment = SCOPE_ASSESSMENT_OUT_OF_SCOPE
            scope_script_notes = "rerun introduced a script that the declared scope marks as forbidden"
        elif scope_payload is None:
            scope_script_assessment = SCOPE_ASSESSMENT_NOT_DECLARED
            scope_script_notes = "no declared modification scope was available for script drift review"
        elif allowed_scope_scripts and rerun_script not in allowed_scope_scripts:
            scope_script_assessment = SCOPE_ASSESSMENT_OUT_OF_SCOPE
            scope_script_notes = "rerun introduced a changed script outside the declared allowed script surface"
        else:
            scope_script_assessment = SCOPE_ASSESSMENT_ALLOWED_CHANGED
            scope_script_notes = "changed script remains within the declared allowed script surface"
        script_delta_rows.append(
            {
                "order_index": str(index + 1),
                "baseline_script_path": baseline_script,
                "rerun_script_path": rerun_script,
                "match": "yes" if baseline_script == rerun_script and baseline_script else "no",
                "scope_assessment": scope_script_assessment,
                "scope_notes": scope_script_notes,
                "notes": (
                    "scripts match"
                    if baseline_script == rerun_script and baseline_script
                    else "script order differs or is missing"
                ),
            }
        )
    write_tsv(
        baseline_audit_dir / "script_chain_delta.tsv",
        [
            "order_index",
            "baseline_script_path",
            "rerun_script_path",
            "match",
            "notes",
            "scope_assessment",
            "scope_notes",
        ],
        script_delta_rows,
    )

    baseline_feature_map = rows_by_feature_id(baseline_audit_dir / "feature_activation_snapshot.tsv")
    rerun_feature_map = {row["feature_id"]: row for row in rerun_feature_rows}
    feature_ids = sorted(set(baseline_feature_map) | set(rerun_feature_map))
    feature_delta_rows: list[dict[str, str]] = []
    feature_scope_assessment, feature_scope_notes = assess_family_scope(
        family_name="feature_activation",
        family_changed=not all(
            baseline_feature_map.get(feature_id, {}).get("activation_status", "")
            == rerun_feature_map.get(feature_id, {}).get("activation_status", "")
            and baseline_feature_map.get(feature_id, {}).get("observed_activation", "")
            == rerun_feature_map.get(feature_id, {}).get("observed_activation", "")
            for feature_id in feature_ids
        ),
        scope_payload=scope_payload,
    )
    for feature_id in feature_ids:
        baseline_row = baseline_feature_map.get(feature_id, {})
        rerun_row = rerun_feature_map.get(feature_id, {})
        feature_delta_rows.append(
            {
                "feature_id": feature_id,
                "baseline_expected_for_run": baseline_row.get("expected_for_run", ""),
                "rerun_expected_for_run": rerun_row.get("expected_for_run", ""),
                "baseline_activation_status": baseline_row.get("activation_status", ""),
                "rerun_activation_status": rerun_row.get("activation_status", ""),
                "baseline_observed_activation": baseline_row.get("observed_activation", ""),
                "rerun_observed_activation": rerun_row.get("observed_activation", ""),
                "status_match": (
                    "yes"
                    if baseline_row.get("activation_status", "") == rerun_row.get("activation_status", "")
                    and baseline_row.get("observed_activation", "") == rerun_row.get("observed_activation", "")
                    else "no"
                ),
                "scope_assessment": feature_scope_assessment,
                "scope_notes": feature_scope_notes,
                "notes": (
                    "feature activation matches"
                    if baseline_row.get("activation_status", "") == rerun_row.get("activation_status", "")
                    and baseline_row.get("observed_activation", "") == rerun_row.get("observed_activation", "")
                    else "feature activation differs"
                ),
            }
        )
    write_tsv(
        baseline_audit_dir / "feature_activation_delta.tsv",
        [
            "feature_id",
            "baseline_expected_for_run",
            "rerun_expected_for_run",
            "baseline_activation_status",
            "rerun_activation_status",
            "baseline_observed_activation",
            "rerun_observed_activation",
            "status_match",
            "notes",
            "scope_assessment",
            "scope_notes",
        ],
        feature_delta_rows,
    )

    stage2_rows = write_stage_delta(
        baseline_inventory=baseline_stage_inventory,
        rerun_inventory=rerun_stage_inventory,
        family_names=STAGE2_FAMILY_NAMES,
        output_path=baseline_audit_dir / "stage2_delta.tsv",
        scope_payload=scope_payload,
    )
    stage3_rows = write_stage_delta(
        baseline_inventory=baseline_stage_inventory,
        rerun_inventory=rerun_stage_inventory,
        family_names=STAGE3_FAMILY_NAMES,
        output_path=baseline_audit_dir / "stage3_delta.tsv",
        scope_payload=scope_payload,
    )
    stage5_rows = write_stage_delta(
        baseline_inventory=baseline_stage_inventory,
        rerun_inventory=rerun_stage_inventory,
        family_names=STAGE5_FAMILY_NAMES,
        output_path=baseline_audit_dir / "stage5_delta.tsv",
        scope_payload=scope_payload,
    )

    stage_delta_rows = stage2_rows + stage3_rows + stage5_rows
    family_scope_rows: list[dict[str, str]] = []
    family_names_in_report = STAGE2_FAMILY_NAMES + STAGE3_FAMILY_NAMES + STAGE5_FAMILY_NAMES + ["feature_activation"]
    for family_name in family_names_in_report:
        if family_name == "feature_activation":
            family_scope_rows.append(
                {
                    "artifact_family": family_name,
                    "scope_assessment": feature_scope_assessment,
                    "scope_notes": feature_scope_notes,
                }
            )
            continue
        family_rows = [row for row in stage_delta_rows if row["artifact_family"] == family_name]
        if not family_rows:
            continue
        family_scope_rows.append(
            {
                "artifact_family": family_name,
                "scope_assessment": family_rows[0]["scope_assessment"],
                "scope_notes": family_rows[0]["scope_notes"],
            }
        )

    out_of_scope_families = [
        row["artifact_family"]
        for row in family_scope_rows
        if row["scope_assessment"] == SCOPE_ASSESSMENT_OUT_OF_SCOPE
    ]
    warning_families = [
        row["artifact_family"]
        for row in family_scope_rows
        if row["scope_assessment"] == SCOPE_ASSESSMENT_WARNING_CHANGED
    ]
    out_of_scope_script_rows = [
        row
        for row in script_delta_rows
        if row["scope_assessment"] == SCOPE_ASSESSMENT_OUT_OF_SCOPE
    ]

    baseline_input_valid = (
        normalize_text(baseline_input_payload.get("summary", {}).get("baseline_valid_for_comparability")) == "yes"
    )
    rerun_input_valid = (
        normalize_text(rerun_input_payload.get("summary", {}).get("baseline_valid_for_comparability")) == "yes"
    )
    input_surface_comparable = baseline_input_valid and rerun_input_valid and all(
        row["notes"] == "input component fingerprints match"
        or row["notes"] == "input component missing in both baseline and rerun"
        for row in input_delta_rows
    )
    script_chain_comparable = all(
        row["match"] == "yes" for row in script_delta_rows if row["baseline_script_path"] or row["rerun_script_path"]
    )
    feature_activation_comparable = all(row["status_match"] == "yes" for row in feature_delta_rows)
    diagnostic_state_comparable = (
        manifest["benchmark_status"]["benchmark_valid"] == rerun_benchmark_status["benchmark_valid"]
    )

    if not (
        input_surface_comparable
        and script_chain_comparable
        and feature_activation_comparable
        and diagnostic_state_comparable
    ):
        comparison_status = COMPARISON_STATUS_NOT_COMPARABLE
    elif out_of_scope_families:
        comparison_status = COMPARISON_STATUS_COMPARABLE_WITH_SCOPE_VIOLATION
    else:
        comparison_status = COMPARISON_STATUS_COMPARABLE_CLEAN

    final_judgment = (
        FINAL_JUDGMENT_COMPARABLE
        if comparison_status == COMPARISON_STATUS_COMPARABLE_CLEAN
        else FINAL_JUDGMENT_NOT_COMPARABLE
    )
    comparison_status_explanation_lines = [
        f"- input_and_chain_match: `{'yes' if input_surface_comparable and script_chain_comparable else 'no'}`",
        f"- baseline_input_surface_valid: `{'yes' if baseline_input_valid else 'no'}`",
        f"- rerun_input_surface_valid: `{'yes' if rerun_input_valid else 'no'}`",
        f"- scope_violation_present: `{'yes' if out_of_scope_families else 'no'}`",
        (
            "- safe_for_engineering_comparison: `yes`"
            if comparison_status == COMPARISON_STATUS_COMPARABLE_CLEAN
            else "- safe_for_engineering_comparison: `no`"
        ),
    ]

    report_lines = [
        "# Comparability Report",
        "",
        f"- baseline_id: `{args.baseline_id}`",
        f"- baseline_audit_dir: `{repo_rel(baseline_audit_dir)}`",
        f"- baseline_source_run_dir: `{manifest['source_run_dir']}`",
        f"- rerun_run_dir: `{repo_rel(rerun_run_dir)}`",
        f"- comparison_status: `{comparison_status}`",
        f"- final_comparability_judgment: `{final_judgment}`",
        "",
        "## Comparison Status",
        "",
        f"- comparison_status: `{comparison_status}`",
        *comparison_status_explanation_lines,
        "",
        "## Governing checks",
        "",
        f"- input_surface_comparable: `{'yes' if input_surface_comparable else 'no'}`",
        f"- script_chain_comparable: `{'yes' if script_chain_comparable else 'no'}`",
        f"- feature_activation_comparable: `{'yes' if feature_activation_comparable else 'no'}`",
        f"- diagnostic_state_comparable: `{'yes' if diagnostic_state_comparable else 'no'}`",
        "",
        "## GT diagnostic status",
        "",
        f"- baseline_legacy_benchmark_valid_flag: `{manifest['benchmark_status']['benchmark_valid']}`",
        f"- rerun_legacy_benchmark_valid_flag: `{rerun_benchmark_status['benchmark_valid']}`",
        f"- baseline_status_source: `{manifest['benchmark_status']['status_source']}`",
        f"- rerun_status_source: `{rerun_benchmark_status['status_source']}`",
        "",
        "## Scope Discipline",
        "",
        f"- declared_scope_id: `{scope_payload['scope_id'] if scope_payload else 'not_declared'}`",
        f"- out_of_scope_artifact_families: `{', '.join(out_of_scope_families) if out_of_scope_families else 'none'}`",
        f"- warning_artifact_families: `{', '.join(warning_families) if warning_families else 'none'}`",
        f"- out_of_scope_script_drift_count: `{len(out_of_scope_script_rows)}`",
        "",
        "## Layer summaries",
        "",
        f"- stage2_rows_compared: `{len(stage2_rows)}`",
        f"- stage3_rows_compared: `{len(stage3_rows)}`",
        f"- stage5_rows_compared: `{len(stage5_rows)}`",
        f"- rerun_feature_activation_gate: `{feature_summary['run_activation_gate']}`",
        "",
        "## Notes",
        "",
        "- v1.1 performs count-level and fingerprint-level comparison only.",
        "- Missing artifacts are reported explicitly; no semantic diffing is attempted.",
        "- Scope-freeze enforcement is family-based, so broad wrapper use is not treated as proof that a touched boundary stayed in scope.",
        "- Feature activation remains visible as its own family and warning-tier drift is never treated as automatically harmless.",
        "- Stage5 comparison is artifact-first and does not require the Stage5 builder to become a wrapper or orchestrator surface.",
    ]
    (baseline_audit_dir / "comparability_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print_command_result(
        payload={
            "baseline_id": args.baseline_id,
            "rerun_run_dir": repo_rel(rerun_run_dir),
            "comparison_status": comparison_status,
            "final_comparability_judgment": final_judgment,
            "input_surface_comparable": input_surface_comparable,
            "script_chain_comparable": script_chain_comparable,
            "feature_activation_comparable": feature_activation_comparable,
            "diagnostic_state_comparable": diagnostic_state_comparable,
            "declared_scope_id": scope_payload["scope_id"] if scope_payload else "",
            "out_of_scope_artifact_families": out_of_scope_families,
            "warning_artifact_families": warning_families,
            "out_of_scope_script_drift_count": len(out_of_scope_script_rows),
        },
        audit_dir=baseline_audit_dir,
        written_files=[
            baseline_audit_dir / "input_surface_delta.tsv",
            baseline_audit_dir / "script_chain_delta.tsv",
            baseline_audit_dir / "feature_activation_delta.tsv",
            baseline_audit_dir / "stage2_delta.tsv",
            baseline_audit_dir / "stage3_delta.tsv",
            baseline_audit_dir / "stage5_delta.tsv",
            baseline_audit_dir / "comparability_report.md",
        ],
    )


def main() -> None:
    args = parse_args()
    if args.command == "start_baseline_audit":
        command_start_baseline_audit(args)
        return
    if args.command == "declare_modification_scope":
        command_declare_modification_scope(args)
        return
    if args.command == "build_layered_delta_report":
        command_build_layered_delta_report(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
