#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derive a targeted manifest from the canonical master manifest with explicit,
reproducible selection provenance.

Purpose
- treat data/cleaned/index/manifest_current.tsv as the single master authority
- write scope-specific manifests only as derived products
- record the exact selection rule used to produce the targeted manifest

This helper is additive compatibility infrastructure. It does not change Stage2
semantics and it does not infer scope by recency, convenience, or hidden state.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_key(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_bool(value: Any) -> str:
    text = normalize_text(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return "yes"
    if text in {"0", "false", "no", "n"}:
        return "no"
    return ""


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-manifest",
        type=Path,
        default=paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv",
        help="Canonical full manifest. Default: data/cleaned/index/manifest_current.tsv",
    )
    parser.add_argument("--out-tsv", type=Path, required=True, help="Target path for the derived manifest TSV.")
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        default=[],
        help="Repeatable exact-match dataset_id filter.",
    )
    parser.add_argument(
        "--split-tag",
        action="append",
        dest="split_tags",
        default=[],
        help="Repeatable exact-match split_tag filter.",
    )
    parser.add_argument(
        "--benchmark-tag",
        action="append",
        dest="benchmark_tags",
        default=[],
        help="Repeatable exact-match benchmark_tag filter.",
    )
    parser.add_argument(
        "--paper-key",
        action="append",
        dest="paper_keys",
        default=[],
        help="Repeatable exact-match paper-key filter.",
    )
    parser.add_argument(
        "--require-text-available",
        action="store_true",
        help="Retain only rows where text_available resolves to yes.",
    )
    parser.add_argument(
        "--require-table-available",
        action="store_true",
        help="Retain only rows where table_available resolves to yes.",
    )
    parser.add_argument(
        "--selection-rule-id",
        default="",
        help="Explicit operator-defined rule id recorded in metadata.",
    )
    parser.add_argument(
        "--selection-note",
        default="",
        help="Optional free-text note recorded in metadata.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    master_manifest = args.master_manifest
    if not master_manifest.is_absolute():
        master_manifest = (paths.PROJECT_ROOT / master_manifest).resolve()
    if not master_manifest.exists():
        raise FileNotFoundError(f"Canonical master manifest not found: {master_manifest}")

    out_tsv = args.out_tsv
    if not out_tsv.is_absolute():
        out_tsv = (paths.PROJECT_ROOT / out_tsv).resolve()

    fieldnames, rows = read_tsv(master_manifest)
    if not rows:
        raise RuntimeError(f"No rows found in canonical master manifest: {master_manifest}")

    dataset_ids = {normalize_key(value) for value in args.dataset_ids if normalize_text(value)}
    split_tags = {normalize_key(value) for value in args.split_tags if normalize_text(value)}
    benchmark_tags = {normalize_key(value) for value in args.benchmark_tags if normalize_text(value)}
    paper_keys = {normalize_key(value) for value in args.paper_keys if normalize_text(value)}

    selected_rows: list[dict[str, str]] = []
    for row in rows:
        paper_key = normalize_key(row.get("paper_key") or row.get("key"))
        dataset_id = normalize_key(row.get("dataset_id"))
        split_tag = normalize_key(row.get("split_tag"))
        benchmark_tag = normalize_key(row.get("benchmark_tag"))
        text_available = normalize_bool(row.get("text_available"))
        table_available = normalize_bool(row.get("table_available"))

        if dataset_ids and dataset_id not in dataset_ids:
            continue
        if split_tags and split_tag not in split_tags:
            continue
        if benchmark_tags and benchmark_tag not in benchmark_tags:
            continue
        if paper_keys and paper_key not in paper_keys:
            continue
        if args.require_text_available and text_available != "yes":
            continue
        if args.require_table_available and table_available != "yes":
            continue
        selected_rows.append(row)

    if not selected_rows:
        raise RuntimeError("Selection rule resolved zero rows from the canonical master manifest.")

    write_tsv(out_tsv, fieldnames, selected_rows)

    metadata_path = out_tsv.with_name(f"{out_tsv.stem}__selection_metadata_v1.json")
    metadata = {
        "schema": "target_manifest_selection_metadata_v1",
        "master_manifest_path": str(master_manifest),
        "derived_manifest_path": str(out_tsv),
        "selection_rule_id": normalize_text(args.selection_rule_id) or "explicit_cli_predicate",
        "selection_filters": {
            "dataset_id": sorted(dataset_ids),
            "split_tag": sorted(split_tags),
            "benchmark_tag": sorted(benchmark_tags),
            "paper_key": sorted(paper_keys),
            "require_text_available": bool(args.require_text_available),
            "require_table_available": bool(args.require_table_available),
        },
        "selection_note": normalize_text(args.selection_note),
        "selected_row_count": len(selected_rows),
        "selected_paper_keys": [
            normalize_text(row.get("paper_key") or row.get("key"))
            for row in selected_rows
            if normalize_text(row.get("paper_key") or row.get("key"))
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"master_manifest={master_manifest}")
    print(f"derived_manifest={out_tsv}")
    print(f"selection_metadata={metadata_path}")
    print(f"selected_rows={len(selected_rows)}")


if __name__ == "__main__":
    main()
