#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "baselines" / "BASELINE_REGISTRY.tsv"

REGISTRY_REQUIRED_FIELDS = [
    "baseline_id",
    "baseline_date",
    "baseline_type",
    "authority_root",
    "primary_lineage_root",
    "stage_coverage",
    "lawful_resume_boundary",
    "benchmark_validity",
    "active_status",
    "manifest_path",
    "notes",
]

MANIFEST_REQUIRED_FIELDS = [
    "baseline_id",
    "baseline_type",
    "date",
    "purpose",
    "authority_root",
    "source_artifacts",
    "artifact_chain",
    "lineage_chain",
    "stage_coverage",
    "boundary_classification",
    "benchmark_validity",
    "limitations",
    "linked_audits",
]

BASELINE_TYPE_VOCAB = {
    "operational_replay_baseline",
    "full_pipeline_baseline",
    "benchmark_baseline",
    "deterministic_two_step_baseline",
}

BENCHMARK_VALIDITY_VOCAB = {
    "benchmark_valid",
    "not_benchmark_valid",
    "diagnostic_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read, resolve, and validate the governed baseline registry."
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to the baseline registry TSV.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered baselines.")
    list_parser.add_argument("--json", action="store_true", help="Emit JSON instead of TSV-style text.")

    show_parser = subparsers.add_parser("show", help="Resolve and show one baseline object.")
    show_parser.add_argument(
        "--query",
        required=True,
        help="Exact baseline_id, exact baseline_date (YYYY-MM-DD), or compact date (YYYYMMDD).",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate the registry and manifests.")
    validate_parser.add_argument(
        "--query",
        default="",
        help="Optional exact baseline_id or date-like query to validate only one baseline.",
    )

    return parser.parse_args()


def die(message: str) -> None:
    raise SystemExit(message)


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_date_key(value: str) -> str:
    return normalize_text(value).replace("-", "")


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def read_registry(registry_path: Path) -> list[dict[str, str]]:
    if not registry_path.exists():
        die(f"Baseline registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        die(f"Baseline registry is empty: {registry_path}")
    return rows


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        die(f"Baseline manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_registry_row(row: dict[str, str], registry_path: Path) -> None:
    missing = [field for field in REGISTRY_REQUIRED_FIELDS if not normalize_text(row.get(field))]
    if missing:
        die(
            f"Registry row in {registry_path} for baseline_id={row.get('baseline_id', '')!r} "
            f"is missing required fields: {missing}"
        )
    if row["baseline_type"] not in BASELINE_TYPE_VOCAB:
        die(
            f"Registry row {row['baseline_id']} has invalid baseline_type={row['baseline_type']!r}. "
            f"Allowed: {sorted(BASELINE_TYPE_VOCAB)}"
        )
    if row["benchmark_validity"] not in BENCHMARK_VALIDITY_VOCAB:
        die(
            f"Registry row {row['baseline_id']} has invalid benchmark_validity={row['benchmark_validity']!r}. "
            f"Allowed: {sorted(BENCHMARK_VALIDITY_VOCAB)}"
        )


def validate_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    missing = [field for field in MANIFEST_REQUIRED_FIELDS if field not in manifest]
    if missing:
        die(f"Baseline manifest {manifest_path} is missing required top-level fields: {missing}")
    for field in ("baseline_id", "baseline_type", "date", "purpose", "authority_root", "benchmark_validity"):
        if not normalize_text(manifest.get(field)):
            die(f"Baseline manifest {manifest_path} has blank required field: {field}")
    if manifest["baseline_type"] not in BASELINE_TYPE_VOCAB:
        die(
            f"Baseline manifest {manifest_path} has invalid baseline_type={manifest['baseline_type']!r}. "
            f"Allowed: {sorted(BASELINE_TYPE_VOCAB)}"
        )
    if manifest["benchmark_validity"] not in BENCHMARK_VALIDITY_VOCAB:
        die(
            f"Baseline manifest {manifest_path} has invalid benchmark_validity={manifest['benchmark_validity']!r}. "
            f"Allowed: {sorted(BENCHMARK_VALIDITY_VOCAB)}"
        )
    if not isinstance(manifest["source_artifacts"], dict) or not manifest["source_artifacts"]:
        die(f"Baseline manifest {manifest_path} must contain a non-empty source_artifacts object.")
    for field in ("artifact_chain", "lineage_chain", "stage_coverage", "limitations", "linked_audits"):
        if not isinstance(manifest[field], list) or not manifest[field]:
            die(f"Baseline manifest {manifest_path} must contain a non-empty list in {field}.")
    if not isinstance(manifest["boundary_classification"], dict) or not manifest["boundary_classification"]:
        die(f"Baseline manifest {manifest_path} must contain a non-empty boundary_classification object.")


def validate_cross_reference(row: dict[str, str], manifest: dict[str, Any], manifest_path: Path) -> None:
    if row["baseline_id"] != manifest["baseline_id"]:
        die(
            f"Registry baseline_id={row['baseline_id']!r} does not match manifest baseline_id="
            f"{manifest['baseline_id']!r} in {manifest_path}"
        )
    if row["baseline_type"] != manifest["baseline_type"]:
        die(
            f"Registry baseline_type={row['baseline_type']!r} does not match manifest baseline_type="
            f"{manifest['baseline_type']!r} in {manifest_path}"
        )
    if row["baseline_date"] != manifest["date"]:
        die(
            f"Registry baseline_date={row['baseline_date']!r} does not match manifest date="
            f"{manifest['date']!r} in {manifest_path}"
        )
    if row["authority_root"] != manifest["authority_root"]:
        die(
            f"Registry authority_root={row['authority_root']!r} does not match manifest authority_root="
            f"{manifest['authority_root']!r} in {manifest_path}"
        )
    if row["benchmark_validity"] != manifest["benchmark_validity"]:
        die(
            f"Registry benchmark_validity={row['benchmark_validity']!r} does not match manifest benchmark_validity="
            f"{manifest['benchmark_validity']!r} in {manifest_path}"
        )


def build_baseline_object(row: dict[str, str], manifest: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    return {
        "baseline_id": row["baseline_id"],
        "baseline_date": row["baseline_date"],
        "baseline_type": row["baseline_type"],
        "authority_root": row["authority_root"],
        "primary_lineage_root": row["primary_lineage_root"],
        "stage_coverage": manifest["stage_coverage"],
        "lawful_resume_boundary": row["lawful_resume_boundary"],
        "benchmark_validity": row["benchmark_validity"],
        "active_status": row["active_status"],
        "manifest_path": row["manifest_path"],
        "notes": row["notes"],
        "artifact_chain": manifest["artifact_chain"],
        "lineage_chain": manifest["lineage_chain"],
        "boundary_classification": manifest["boundary_classification"],
        "linked_audits": manifest["linked_audits"],
        "registry_path": repo_rel(registry_path),
    }


def resolve_matches(rows: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    query_text = normalize_text(query)
    if not query_text:
        die("A non-empty baseline query is required.")
    query_date_key = normalize_date_key(query_text)
    matches: list[dict[str, str]] = []
    for row in rows:
        if row["baseline_id"] == query_text:
            matches.append(row)
            continue
        if row["manifest_path"] == query_text:
            matches.append(row)
            continue
        if normalize_date_key(row["baseline_date"]) == query_date_key:
            matches.append(row)
            continue
    return matches


def resolve_baseline_row(rows: list[dict[str, str]], query: str) -> dict[str, str]:
    matches = resolve_matches(rows, query)
    if not matches:
        die(f"No baseline matched query={query!r}.")
    if len(matches) > 1:
        ids = [row["baseline_id"] for row in matches]
        die(f"Baseline query={query!r} is ambiguous. Matching baseline_ids: {ids}")
    return matches[0]


def command_list(rows: list[dict[str, str]], as_json: bool) -> None:
    validated_rows = []
    seen_ids: set[str] = set()
    for row in rows:
        validate_registry_row(row, DEFAULT_REGISTRY_PATH)
        baseline_id = row["baseline_id"]
        if baseline_id in seen_ids:
            die(f"Duplicate baseline_id found in registry: {baseline_id}")
        seen_ids.add(baseline_id)
        validated_rows.append(row)
    if as_json:
        print(json.dumps(validated_rows, indent=2, ensure_ascii=True))
        return
    print("\t".join(REGISTRY_REQUIRED_FIELDS))
    for row in validated_rows:
        print("\t".join(row[field] for field in REGISTRY_REQUIRED_FIELDS))


def command_show(rows: list[dict[str, str]], registry_path: Path, query: str) -> None:
    matches = resolve_matches(rows, query)
    if not matches:
        die(f"No baseline matched query={query!r}.")
    objects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in matches:
        validate_registry_row(row, registry_path)
        baseline_id = row["baseline_id"]
        if baseline_id in seen_ids:
            die(f"Duplicate baseline_id found in registry: {baseline_id}")
        seen_ids.add(baseline_id)
        manifest_path = PROJECT_ROOT / Path(row["manifest_path"])
        manifest = read_manifest(manifest_path)
        validate_manifest(manifest, manifest_path)
        validate_cross_reference(row, manifest, manifest_path)
        objects.append(build_baseline_object(row, manifest, registry_path))
    if len(objects) == 1:
        print(json.dumps(objects[0], indent=2, ensure_ascii=True))
        return
    print(json.dumps(objects, indent=2, ensure_ascii=True))


def command_validate(rows: list[dict[str, str]], registry_path: Path, query: str) -> None:
    seen_ids: set[str] = set()
    target_rows = rows
    if normalize_text(query):
        target_rows = [resolve_baseline_row(rows, query)]
    for row in target_rows:
        validate_registry_row(row, registry_path)
        baseline_id = row["baseline_id"]
        if baseline_id in seen_ids:
            die(f"Duplicate baseline_id found in registry: {baseline_id}")
        seen_ids.add(baseline_id)
        manifest_path = PROJECT_ROOT / Path(row["manifest_path"])
        manifest = read_manifest(manifest_path)
        validate_manifest(manifest, manifest_path)
        validate_cross_reference(row, manifest, manifest_path)
    print(
        json.dumps(
            {
                "ok": True,
                "validated_registry": repo_rel(registry_path),
                "validated_baseline_count": len(target_rows),
                "validated_baseline_ids": [row["baseline_id"] for row in target_rows],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def main() -> None:
    args = parse_args()
    registry_path = args.registry_path.resolve()
    rows = read_registry(registry_path)
    if args.command == "list":
        command_list(rows, as_json=args.json)
        return
    if args.command == "show":
        command_show(rows, registry_path=registry_path, query=args.query)
        return
    if args.command == "validate":
        command_validate(rows, registry_path=registry_path, query=args.query)
        return
    die(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
