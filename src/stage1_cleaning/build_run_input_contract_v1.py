#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a diagnostic-only run input contract from the canonical Stage1 manifest.

This helper implements the S1-5 boundary described by the end-to-end boundary
repair plan. It records a selected run scope as a contract and scope TSV while
keeping the canonical manifest as the single manifest authority. It does not
create a competing manifest and it does not infer scope from latest/glob/mtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402

SCHEMA_ID = "run_input_contract_v1"
SCOPE_SCHEMA_ID = "run_input_scope_v1"

SCOPE_FIELDS = [
    "paper_key",
    "selection_reason",
    "canonical_manifest_path",
    "canonical_manifest_sha256",
    "text_path",
    "text_source_type",
    "text_available",
    "table_asset_root",
    "table_asset_refs",
    "table_available",
    "structure_path",
    "structure_available",
    "stage1_table_cell_sidecar_path",
    "stage1_table_cell_sidecar_available",
    "dataset_id",
    "split_tag",
    "benchmark_tag",
    "source_collection",
    "source_record_path",
]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_key(value: Any) -> str:
    return normalize_text(value)


def normalize_filter(value: Any) -> str:
    return normalize_text(value).lower()


def resolve_project_path(path_value: Path | str) -> Path:
    path_obj = path_value if isinstance(path_value, Path) else Path(path_value)
    return path_obj if path_obj.is_absolute() else (paths.PROJECT_ROOT / path_obj).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _row_paper_key(row: dict[str, str]) -> str:
    return normalize_key(row.get("paper_key") or row.get("key") or row.get("zotero_key"))


def _selected_by_filters(
    row: dict[str, str],
    *,
    paper_keys: set[str],
    dataset_ids: set[str],
    split_tags: set[str],
    benchmark_tags: set[str],
) -> bool:
    if paper_keys and normalize_filter(_row_paper_key(row)) not in paper_keys:
        return False
    if dataset_ids and normalize_filter(row.get("dataset_id")) not in dataset_ids:
        return False
    if split_tags and normalize_filter(row.get("split_tag")) not in split_tags:
        return False
    if benchmark_tags and normalize_filter(row.get("benchmark_tag")) not in benchmark_tags:
        return False
    return True


def _selection_reason(row: dict[str, str], selection_rule_id: str) -> str:
    parts = [selection_rule_id or "explicit_cli_scope"]
    for field in ("dataset_id", "split_tag", "benchmark_tag"):
        value = normalize_text(row.get(field))
        if value:
            parts.append(f"{field}={value}")
    return ";".join(parts)


def _first_present(row: dict[str, str], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = normalize_text(row.get(field))
        if value:
            return value
    return ""


def build_run_input_contract(
    *,
    canonical_manifest: Path,
    out_dir: Path,
    run_id: str,
    selection_rule_id: str,
    selection_note: str = "",
    paper_keys: list[str] | None = None,
    dataset_ids: list[str] | None = None,
    split_tags: list[str] | None = None,
    benchmark_tags: list[str] | None = None,
) -> dict[str, Any]:
    canonical_manifest = resolve_project_path(canonical_manifest)
    out_dir = resolve_project_path(out_dir)
    if not canonical_manifest.exists():
        raise FileNotFoundError(f"canonical manifest not found: {canonical_manifest}")
    if not selection_rule_id:
        raise ValueError("selection_rule_id is required; scope must be explicit")

    _, rows = read_tsv(canonical_manifest)
    if not rows:
        raise RuntimeError(f"canonical manifest has no rows: {canonical_manifest}")

    filter_sets = {
        "paper_key": {normalize_filter(value) for value in (paper_keys or []) if normalize_text(value)},
        "dataset_id": {normalize_filter(value) for value in (dataset_ids or []) if normalize_text(value)},
        "split_tag": {normalize_filter(value) for value in (split_tags or []) if normalize_text(value)},
        "benchmark_tag": {normalize_filter(value) for value in (benchmark_tags or []) if normalize_text(value)},
    }
    if not any(filter_sets.values()):
        raise ValueError("at least one explicit selector is required")

    manifest_hash = sha256_file(canonical_manifest)
    selected_rows: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for row in rows:
        if not _selected_by_filters(
            row,
            paper_keys=filter_sets["paper_key"],
            dataset_ids=filter_sets["dataset_id"],
            split_tags=filter_sets["split_tag"],
            benchmark_tags=filter_sets["benchmark_tag"],
        ):
            continue
        paper_key = _row_paper_key(row)
        if not paper_key:
            raise RuntimeError("selected manifest row lacks paper_key/key/zotero_key")
        if paper_key in seen_keys:
            raise RuntimeError(f"duplicate selected paper_key in canonical manifest: {paper_key}")
        seen_keys.add(paper_key)
        selected_rows.append(row)

    if not selected_rows:
        raise RuntimeError("selection resolved zero rows from canonical manifest")

    scope_rows: list[dict[str, str]] = []
    missing_text: list[str] = []
    for row in selected_rows:
        paper_key = _row_paper_key(row)
        text_path = _first_present(row, ("text_path", "clean_text_path"))
        table_root = _first_present(row, ("table_dir", "table_asset_root", "tables_root"))
        table_refs = _first_present(row, ("table_asset_refs", "table_manifest_path", "tables_manifest_path", "table_files"))
        structure_path = _first_present(row, ("structure_path", "stage1_structure_path", "structure_sidecar_path"))
        table_cell_sidecar_path = _first_present(
            row,
            (
                "stage1_table_cell_sidecar_path",
                "table_cell_sidecar_path",
                "tables_cell_sidecar_path",
                "stage1_cells_path",
            ),
        )
        text_available = normalize_text(row.get("text_available")) or ("yes" if text_path else "no")
        table_available = normalize_text(row.get("table_available")) or ("yes" if table_root or table_refs else "no")
        structure_available = normalize_text(row.get("structure_available")) or ("yes" if structure_path else "no")
        table_cell_sidecar_available = normalize_text(row.get("stage1_table_cell_sidecar_available")) or ("yes" if table_cell_sidecar_path else "no")
        if not text_path:
            missing_text.append(paper_key)
        scope_rows.append(
            {
                "paper_key": paper_key,
                "selection_reason": _selection_reason(row, selection_rule_id),
                "canonical_manifest_path": str(canonical_manifest),
                "canonical_manifest_sha256": manifest_hash,
                "text_path": text_path,
                "text_source_type": normalize_text(row.get("text_source_type")),
                "text_available": text_available,
                "table_asset_root": table_root,
                "table_asset_refs": table_refs,
                "table_available": table_available,
                "structure_path": structure_path,
                "structure_available": structure_available,
                "stage1_table_cell_sidecar_path": table_cell_sidecar_path,
                "stage1_table_cell_sidecar_available": table_cell_sidecar_available,
                "dataset_id": normalize_text(row.get("dataset_id")),
                "split_tag": normalize_text(row.get("split_tag")),
                "benchmark_tag": normalize_text(row.get("benchmark_tag")),
                "source_collection": _first_present(row, ("source_collection", "collection", "library_id", "dataset_source")),
                "source_record_path": _first_present(row, ("source_record_path", "raw_source_path", "raw_jsonl_path")),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    scope_path = out_dir / "run_input_scope_v1.tsv"
    contract_path = out_dir / "run_input_contract_v1.json"
    write_tsv(scope_path, SCOPE_FIELDS, scope_rows)

    contract: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "diagnostic_only": True,
        "benchmark_valid": False,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_manifest": {
            "path": str(canonical_manifest),
            "sha256": manifest_hash,
            "authority_role": "single_canonical_manifest",
        },
        "selection": {
            "selection_rule_id": selection_rule_id,
            "selection_note": selection_note,
            "filters": {key: sorted(value) for key, value in filter_sets.items()},
            "selected_row_count": len(scope_rows),
            "selected_paper_keys": [row["paper_key"] for row in scope_rows],
            "selected_source_paths": [
                {
                    "paper_key": row["paper_key"],
                    "text_path": row["text_path"],
                    "table_asset_root": row["table_asset_root"],
                    "table_asset_refs": row["table_asset_refs"],
                    "structure_path": row["structure_path"],
                    "stage1_table_cell_sidecar_path": row["stage1_table_cell_sidecar_path"],
                }
                for row in scope_rows
            ],
        },
        "artifacts": {
            "run_input_scope_tsv": str(scope_path),
            "run_input_contract_json": str(contract_path),
        },
        "scope_schema": {
            "schema": SCOPE_SCHEMA_ID,
            "fields": SCOPE_FIELDS,
        },
        "asset_diagnostics": {
            "missing_text_paper_keys": missing_text,
            "table_available_yes_count": sum(1 for row in scope_rows if normalize_filter(row.get("table_available")) == "yes"),
            "structure_available_yes_count": sum(1 for row in scope_rows if normalize_filter(row.get("structure_available")) == "yes"),
            "stage1_table_cell_sidecar_available_yes_count": sum(
                1 for row in scope_rows if normalize_filter(row.get("stage1_table_cell_sidecar_available")) == "yes"
            ),
        },
        "governance": {
            "subset_manifest_created": False,
            "notes": "Scope is recorded as run parameters and scope TSV, not as a competing manifest authority.",
        },
    }
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return contract


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        default=paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv",
        help="Single canonical manifest authority. Default: data/cleaned/index/manifest_current.tsv",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Run directory where contract artifacts are written.")
    parser.add_argument("--run-id", required=True, help="Explicit diagnostic run id recorded in the contract.")
    parser.add_argument("--selection-rule-id", required=True, help="Explicit scope selection rule id.")
    parser.add_argument("--selection-note", default="", help="Optional human-readable scope note.")
    parser.add_argument("--paper-key", action="append", dest="paper_keys", default=[], help="Repeatable paper-key selector.")
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids", default=[], help="Repeatable dataset_id selector.")
    parser.add_argument("--split-tag", action="append", dest="split_tags", default=[], help="Repeatable split_tag selector.")
    parser.add_argument("--benchmark-tag", action="append", dest="benchmark_tags", default=[], help="Repeatable benchmark_tag selector.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    contract = build_run_input_contract(
        canonical_manifest=args.canonical_manifest,
        out_dir=args.out_dir,
        run_id=args.run_id,
        selection_rule_id=args.selection_rule_id,
        selection_note=args.selection_note,
        paper_keys=args.paper_keys,
        dataset_ids=args.dataset_ids,
        split_tags=args.split_tags,
        benchmark_tags=args.benchmark_tags,
    )
    print("diagnostic_only=true")
    print("benchmark_valid=false")
    print(f"canonical_manifest={contract['canonical_manifest']['path']}")
    print(f"canonical_manifest_sha256={contract['canonical_manifest']['sha256']}")
    print(f"run_input_contract={contract['artifacts']['run_input_contract_json']}")
    print(f"run_input_scope={contract['artifacts']['run_input_scope_tsv']}")
    print(f"selected_rows={contract['selection']['selected_row_count']}")
    print("selected_paper_keys=" + ",".join(contract["selection"]["selected_paper_keys"]))
    for source in contract["selection"]["selected_source_paths"]:
        print(
            "selected_source="
            f"{source['paper_key']}|text={source['text_path']}|"
            f"table_root={source['table_asset_root']}|table_refs={source['table_asset_refs']}"
        )
    print("scope_fields=" + ",".join(contract["scope_schema"]["fields"]))


if __name__ == "__main__":
    main()
