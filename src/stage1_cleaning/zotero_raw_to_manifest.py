#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zotero_raw_to_manifest.py

Convert Zotero raw JSONL (built by zotero_api_sync_selected.py) into the
authoritative TSV manifest used by cleaning/extraction.

Inputs (default via paths.py)
- data/raw/zotero/zotero_selected_items.jsonl

Outputs (default via paths.py)
- data/cleaned/index/manifest_current.tsv

Manifest columns (minimum)
- key, title, pdf, html
Recommended
- doi, year, notes, zotero_key

Selection policy
- Prefer HTML if present, else PDF.
- Keep items even if missing fulltext, but mark in notes (NO_LOCAL_FULLTEXT).
  This allows you to track coverage and re-run after downloading more PDFs.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Zotero raw JSONL to manifest_current.tsv")
    ap.add_argument(
        "--input",
        type=Path,
        default=(paths.DATA_RAW_DIR / "zotero" / "zotero_selected_items.jsonl"),
        help="Input JSONL (default via paths.py).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=(paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv"),
        help="Output manifest TSV (default via paths.py).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output manifest if exists.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {args.out}")

    rows = load_jsonl(args.input)

    records: List[Dict[str, Any]] = []
    with_pdf = 0
    with_html = 0
    for r in rows:
        zk = str(r.get("zotero_key", "")).strip()
        title = str(r.get("title", "")).strip()
        doi = str(r.get("doi", "")).strip()
        year = str(r.get("year", "")).strip()

        paths_block = r.get("paths", {}) or {}
        pdf = paths_block.get("pdf") or ""
        html = paths_block.get("html") or ""

        # choose prefer html, else pdf
        if html:
            with_html += 1
        if pdf:
            with_pdf += 1

        status = str(r.get("status", "")).strip()
        msg = str(r.get("message", "")).strip()

        notes = []
        if status:
            notes.append(status)
        if msg:
            notes.append(msg)
        if not pdf and not html:
            notes.append("NO_LOCAL_FULLTEXT")

        records.append(
            {
                "key": zk,
                "zotero_key": zk,
                "title": title,
                "doi": doi,
                "year": year,
                "pdf": pdf,
                "html": html,
                "notes": ";".join(notes),
            }
        )

    df = pd.DataFrame.from_records(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)

    if args.verbose:
        print(f"[OK] wrote manifest: {args.out}")
        print(f"[INFO] rows={len(df)} | with_pdf={with_pdf} | with_html={with_html}")

    print(str(args.out))


if __name__ == "__main__":
    main()
