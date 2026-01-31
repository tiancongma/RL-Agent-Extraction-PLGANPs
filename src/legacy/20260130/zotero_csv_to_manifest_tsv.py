#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zotero_csv_to_manifest_tsv.py

Purpose
- Convert a Zotero-exported CSV into the project's authoritative TSV manifest:
  data/cleaned/index/manifest_current.tsv

Design constraints
- No hard-coded repository paths. Default inputs/outputs must come from src/utils/paths.py.
- This script does NOT perform cleaning. It only builds the manifest.

Expected downstream
- clean_manifest_to_text.py consumes manifest_current.tsv to generate cleaned text and key2txt.tsv.

Manifest schema (TSV, tab-separated)
Required columns (used by pdf2clean manifest mode):
- key: unique identifier (prefer Zotero key)
- title: article title
- pdf: path to a local PDF (may be empty)
- html: path to a local HTML snapshot (may be empty)
Optional columns:
- url, doi, year, notes

Heuristics
- The script tries to locate a key column among common Zotero/WoS exports.
- Attachment parsing is heuristic: it searches for .html/.htm and .pdf paths in typical attachment fields.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _bootstrap_import_paths() -> None:
    """
    Ensure we can import src.utils.paths without requiring the user to set PYTHONPATH.
    We locate repo root by searching upward for 'src/utils/paths.py' and prepend it to sys.path.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _normalize_attachment_field(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    # Zotero CSV sometimes separates attachments with semicolons/newlines.
    return s.replace("\\r", "\n").replace("\\n", "\n")


_HTML_RE = re.compile(r"(?P<p>(?:[A-Za-z]:\\|/)[^\n;]*\.(?:html?|xhtml))(?:\b|$)", re.IGNORECASE)
_PDF_RE = re.compile(r"(?P<p>(?:[A-Za-z]:\\|/)[^\n;]*\.pdf)(?:\b|$)", re.IGNORECASE)


def _extract_pdf_html_from_attachments(attachments_text: str) -> Tuple[str, str]:
    """
    Return (pdf_path, html_path) as strings. Prefer first match of each type.
    """
    pdf = ""
    html = ""
    if not attachments_text:
        return pdf, html

    # Search all matches, keep the first plausible for each
    for m in _HTML_RE.finditer(attachments_text):
        html = m.group("p").strip()
        break
    for m in _PDF_RE.finditer(attachments_text):
        pdf = m.group("p").strip()
        break

    return pdf, html


def build_manifest(
    input_csv: Path,
    output_tsv: Path,
    overwrite: bool,
    verbose: bool,
) -> Path:
    if not input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")

    if output_tsv.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output_tsv}")

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    # Identify columns (heuristic)
    key_col = _pick_col(df, ["Key", "Item Key", "itemKey", "Zotero Key", "zotero_key", "Citation Key", "ID"])
    title_col = _pick_col(df, ["Title", "title"])
    doi_col = _pick_col(df, ["DOI", "doi"])
    url_col = _pick_col(df, ["Url", "URL", "url", "Link", "link"])
    year_col = _pick_col(df, ["Year", "year", "Publication Year", "Date", "date"])

    # Attachments columns vary widely in Zotero exports
    attach_col = _pick_col(df, [
        "File Attachments",
        "file attachments",
        "Attachments",
        "attachments",
        "File Attachment",
        "file attachment",
    ])

    out = pd.DataFrame()
    if key_col:
        out["key"] = df[key_col].astype(str).str.strip()
    else:
        # Fallback: stable synthetic key from row index
        out["key"] = df.index.astype(str)

    out["title"] = df[title_col].astype(str).str.strip() if title_col else ""
    out["doi"] = df[doi_col].astype(str).str.strip() if doi_col else ""
    out["url"] = df[url_col].astype(str).str.strip() if url_col else ""

    if year_col:
        # Keep as-is; do not parse into datetime (avoid locale/format issues)
        out["year"] = df[year_col].astype(str).str.strip()
    else:
        out["year"] = ""

    pdf_list: list[str] = []
    html_list: list[str] = []
    notes_list: list[str] = []

    for i, row in df.iterrows():
        attachments_text = _normalize_attachment_field(row.get(attach_col, "")) if attach_col else ""
        pdf, html = _extract_pdf_html_from_attachments(attachments_text)

        # If neither found, leave empty and record a note
        note = ""
        if not pdf and not html:
            note = "NO_LOCAL_ATTACHMENT"
        pdf_list.append(pdf)
        html_list.append(html)
        notes_list.append(note)

    out["pdf"] = pdf_list
    out["html"] = html_list
    out["notes"] = notes_list

    # Ensure output directory exists
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    # Write TSV
    out.to_csv(output_tsv, sep="\t", index=False)

    if verbose:
        n = len(out)
        n_pdf = sum(1 for p in pdf_list if p)
        n_html = sum(1 for h in html_list if h)
        print(f"[OK] Wrote manifest: {output_tsv}")
        print(f"[INFO] rows={n} | with_pdf={n_pdf} | with_html={n_html}")

    return output_tsv


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert Zotero CSV export to manifest_current.tsv (TSV).")

    ap.add_argument(
        "--input",
        type=Path,
        default=(paths.DATA_RAW_DIR / "zotero_export.csv"),
        help="Input Zotero CSV export. Default: data/raw/zotero_export.csv (via paths.py).",
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=(paths.DATA_CLEANED_INDEX_DIR / "manifest_current.tsv"),
        help="Output TSV manifest path. Default: data/cleaned/index/manifest_current.tsv (via paths.py).",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output TSV if it exists.",
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print summary information.",
    )

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    out_path = build_manifest(
        input_csv=args.input,
        output_tsv=args.out,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )

    # Print the canonical path on success for easy copy/paste into next step
    print(str(out_path))


if __name__ == "__main__":
    main()
