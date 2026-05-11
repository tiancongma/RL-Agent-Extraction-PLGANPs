#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zotero_raw_to_manifest.py

Convert one or more declared Zotero raw JSONL sources into the authoritative
TSV manifest used by cleaning/extraction.

Inputs (default via paths.py)
- data/raw/zotero/zotero_selected_items.jsonl
- optional additional declared raw JSONL sources passed by repeatable --input

Outputs (default via paths.py)
- data/cleaned/index/manifest_current.tsv

Manifest columns (minimum)
- key, title, pdf, html
Recommended
- doi, year, notes, zotero_key

Additive provenance columns
- source_collection
- source_manifest_lineage
- source_selection_rule
- raw_source_jsonl

Selection policy
- Prefer HTML if present, else PDF.
- Keep items even if missing fulltext, but mark in notes (NO_LOCAL_FULLTEXT).
  This allows you to track coverage and re-run after downloading more PDFs.
- In multi-source mode, merge rows by Zotero key and preserve source lineage as
  pipe-delimited unions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402


def normalize_doi(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^doi\s*:\s*", "", text)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\.org/", "", text)
    return text.strip()


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        raise FileNotFoundError(f"input jsonl not found: {p}")
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_union_value(value: str) -> str:
    return str(value or "").strip()


def join_unique(values: List[str]) -> str:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        text = normalize_union_value(value)
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return "|".join(ordered)


def merge_notes(existing: str, incoming: str) -> str:
    tokens: List[str] = []
    for block in (existing, incoming):
        for token in str(block or "").split(";"):
            token = token.strip()
            if token:
                tokens.append(token)
    return ";".join(dict.fromkeys(tokens))


def first_nonempty(*values: str) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def extract_selector_from_message(message: str) -> str:
    match = re.search(r"(?:^|;)selector=([^;]+)", str(message or ""))
    if not match:
        return ""
    return match.group(1).strip()


def extract_collection_name(selector: str) -> str:
    match = re.match(r"collection:([^(;]+)", str(selector or "").strip())
    if not match:
        return ""
    return match.group(1).strip()


def to_repo_rel_or_abs(path_value: Path) -> str:
    try:
        return str(path_value.resolve().relative_to(paths.PROJECT_ROOT.resolve()))
    except Exception:
        return str(path_value.resolve())


def validate_parallel_args(values: List[str], arg_name: str, expected_len: int) -> List[str]:
    if not values:
        return [""] * expected_len
    if len(values) != expected_len:
        raise ValueError(
            f"{arg_name} must be supplied either zero times or exactly once per --input "
            f"(expected {expected_len}, got {len(values)})"
        )
    return [normalize_text(value) for value in values]


def merge_record(existing: Dict[str, str], incoming: Dict[str, str]) -> Dict[str, str]:
    for field in ("title", "doi", "normalized_doi", "year", "zotero_key", "paper_key"):
        existing_value = normalize_text(existing.get(field))
        incoming_value = normalize_text(incoming.get(field))
        if existing_value and incoming_value and existing_value != incoming_value:
            raise ValueError(
                f"Conflicting '{field}' for key {existing.get('key')}: "
                f"{existing_value!r} vs {incoming_value!r}"
            )

    for field in ("pdf", "html", "text_path", "text_source_type", "text_available", "table_dir", "table_available"):
        existing[field] = first_nonempty(existing.get(field, ""), incoming.get(field, ""))

    incoming_dataset_id = normalize_text(incoming.get("dataset_id"))
    existing_dataset_id = normalize_text(existing.get("dataset_id"))
    if existing_dataset_id and incoming_dataset_id and existing_dataset_id != incoming_dataset_id:
        raise ValueError(
            f"Conflicting 'dataset_id' for key {existing.get('key')}: "
            f"{existing_dataset_id!r} vs {incoming_dataset_id!r}"
        )
    existing["dataset_id"] = first_nonempty(existing_dataset_id, incoming_dataset_id)

    existing["notes"] = merge_notes(existing.get("notes", ""), incoming.get("notes", ""))
    for field in ("source_collection", "source_manifest_lineage", "source_selection_rule", "raw_source_jsonl"):
        existing[field] = join_unique([existing.get(field, ""), incoming.get(field, "")])
    return existing


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Zotero raw JSONL to manifest_current.tsv")
    ap.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Repeatable raw JSONL input. Default: data/raw/zotero/zotero_selected_items.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=(paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv"),
        help="Output manifest TSV (default via paths.py).",
    )
    ap.add_argument(
        "--source-collection",
        action="append",
        default=[],
        help="Repeatable source collection label aligned to --input order (for example: wos_all, goren_2025).",
    )
    ap.add_argument(
        "--source-manifest-lineage",
        action="append",
        default=[],
        help="Repeatable source manifest lineage label aligned to --input order.",
    )
    ap.add_argument(
        "--source-selection-rule",
        action="append",
        default=[],
        help="Repeatable source selection rule aligned to --input order.",
    )
    ap.add_argument(
        "--input-dataset-id",
        action="append",
        default=[],
        help="Repeatable dataset_id assignment aligned to --input order. Leave blank for global-source rows.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output manifest if exists.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {args.out}")

    input_paths = args.input or [paths.DATA_RAW_DIR / "zotero" / "zotero_selected_items.jsonl"]
    source_collections = validate_parallel_args(args.source_collection, "--source-collection", len(input_paths))
    source_manifest_lineages = validate_parallel_args(
        args.source_manifest_lineage, "--source-manifest-lineage", len(input_paths)
    )
    source_selection_rules = validate_parallel_args(
        args.source_selection_rule, "--source-selection-rule", len(input_paths)
    )
    input_dataset_ids = validate_parallel_args(args.input_dataset_id, "--input-dataset-id", len(input_paths))

    records_by_key: Dict[str, Dict[str, str]] = {}
    with_pdf = 0
    with_html = 0
    for idx, input_path in enumerate(input_paths):
        resolved_input = input_path if input_path.is_absolute() else (paths.PROJECT_ROOT / input_path).resolve()
        rows = load_jsonl(resolved_input)
        for r in rows:
            zk = normalize_text(r.get("zotero_key"))
            if not zk:
                continue
            title = normalize_text(r.get("title"))
            doi = normalize_text(r.get("doi"))
            year = normalize_text(r.get("year"))

            paths_block = r.get("paths", {}) or {}
            pdf = normalize_text(paths_block.get("pdf"))
            html = normalize_text(paths_block.get("html"))

            if html:
                with_html += 1
            if pdf:
                with_pdf += 1

            status = normalize_text(r.get("status"))
            msg = normalize_text(r.get("message"))
            inferred_selector = extract_selector_from_message(msg)
            source_selection_rule = source_selection_rules[idx] or inferred_selector
            source_collection = source_collections[idx] or extract_collection_name(source_selection_rule)

            notes = []
            if status:
                notes.append(status)
            if msg:
                notes.append(msg)
            if not pdf and not html:
                notes.append("NO_LOCAL_FULLTEXT")

            incoming_record = {
                "key": zk,
                "paper_key": zk,
                "zotero_key": zk,
                "title": title,
                "doi": doi,
                "normalized_doi": normalize_doi(doi),
                "year": year,
                "pdf": pdf,
                "html": html,
                "text_path": "",
                "text_source_type": "",
                "text_available": "",
                "table_dir": "",
                "table_available": "",
                "dataset_id": input_dataset_ids[idx],
                "split_tag": "",
                "benchmark_tag": "",
                "source_collection": source_collection,
                "source_manifest_lineage": source_manifest_lineages[idx],
                "source_selection_rule": source_selection_rule,
                "raw_source_jsonl": to_repo_rel_or_abs(resolved_input),
                "notes": ";".join(notes),
            }
            if zk in records_by_key:
                records_by_key[zk] = merge_record(records_by_key[zk], incoming_record)
            else:
                records_by_key[zk] = incoming_record

    df = pd.DataFrame.from_records([records_by_key[key] for key in sorted(records_by_key)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)

    if args.verbose:
        print(f"[OK] wrote manifest: {args.out}")
        print(
            f"[INFO] rows={len(df)} | with_pdf={with_pdf} | with_html={with_html} | "
            f"declared_inputs={len(input_paths)}"
        )

    print(str(args.out))


if __name__ == "__main__":
    main()
