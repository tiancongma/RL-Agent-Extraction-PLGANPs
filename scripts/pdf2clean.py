#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf2clean.py — Unified cleaner for PDF/HTML with quality metadata,
plus a legacy single-PDF API shim: process_pdf(...).

What it does (manifest mode)
----------------------------
1) Read a manifest (CSV/TSV) listing items (keys, titles, file paths or URLs).
2) Extract clean text from either HTML or PDF (or both, order controlled by --prefer).
3) Write cleaned .txt files under out_dir/text/.
4) Produce out_dir/key2txt.tsv with metadata columns:
   key, title, source_type(PDF/HTML), txt_path, text_length,
   table_detected(0/1), parse_quality(low/medium/high), notes, page_count, url.

Legacy compatibility
--------------------
Exports process_pdf(pdf_path, outdir, keep_sections, tables, debug_trace=False, debug_skim=False, preview_lines=6)
so older scripts that import pdf2clean.process_pdf keep working. This shim uses the same
PDF text extraction, writes <outdir>/text/<stem>.pdf.txt, and returns (ok, meta).

Quick start (no arguments; uses fixed defaults)
-----------------------------------------------
python scripts/pdf2clean.py
  -> uses manifest .\\data\\cleaned\\manifest.tsv
  -> writes to     .\\data\\cleaned

Or explicit:
python scripts/pdf2clean.py --manifest .\\data\\cleaned\\manifest.tsv --out-dir .\\data\\cleaned --prefer html --overwrite --verbose

Dependencies
------------
pip install pandas beautifulsoup4 lxml pymupdf
"""

from __future__ import annotations
import os
import re
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd

# Optional PDF import
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from bs4 import BeautifulSoup

# ----------------------------
# Column picking
# ----------------------------
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        lc = c.lower()
        if lc in cols_lower:
            return cols_lower[lc]
    return None

# ----------------------------
# Heuristics & helpers
# ----------------------------
TABLE_HEADER_HINTS = [
    r"\b(mean|sd|se|n)\b",
    r"\b(size|diameter|pdi|zeta|ee|dl|loading|encapsulation)\b",
    r"\b(concentration|dose|w/?o|w1|w2|%|ratio)\b",
]

def detect_table_like_text(txt: str) -> bool:
    if not txt:
        return False
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return False
    tabular_like = 0
    num_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*[±x×]\s*\d+(?:\.\d+)?)?")
    for ln in lines[:2000]:
        tokens = re.split(r"[\t|,:; ]{2,}", ln.strip())
        numeric_cells = sum(1 for t in tokens if num_re.search(t))
        if len(tokens) >= 3 and numeric_cells >= 2:
            tabular_like += 1
    header_hit = any(re.search(p, txt.lower()) for p in TABLE_HEADER_HINTS)
    return tabular_like >= 5 or header_hit

def estimate_parse_quality(txt: str, source_type: str) -> str:
    if not txt:
        return "low"
    lines = [ln for ln in txt.splitlines()]
    if not lines:
        return "low"
    avg_line = sum(len(ln) for ln in lines) / max(1, len(lines))
    hyphens = txt.count("-\n") + txt.count("­\n")
    ligatures = txt.count("ﬁ") + txt.count("ﬂ")
    short_lines = sum(1 for ln in lines if 0 < len(ln) < 20)
    ratio_short = short_lines / max(1, len(lines))
    score = 0.0
    score += 1.0 if source_type.upper() == "HTML" else 0.2
    if avg_line > 40: score += 0.6
    if ratio_short < 0.3: score += 0.4
    if hyphens < 5: score += 0.3
    if ligatures < 3: score += 0.2
    if score >= 1.8: return "high"
    if score >= 1.0: return "medium"
    return "low"

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ----------------------------
# HTML extraction
# ----------------------------
def extract_text_from_html(html_path: Path) -> Tuple[str, bool]:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    parts = []
    for elem in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","figcaption","caption"]):
        txt = elem.get_text(separator=" ", strip=True)
        if txt:
            parts.append(txt)
    body_text = "\n".join(parts)
    tables = soup.find_all("table")
    table_detected = len(tables) > 0
    table_blocks = []
    for ti, table in enumerate(tables, start=1):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(separator=" ", strip=True) for c in tr.find_all(["th","td"])]
            if cells:
                rows.append("\t".join(cells))
        if rows:
            block = f"=== TABLE {ti} (TSV) ===\n" + "\n".join(rows)
            table_blocks.append(block)
    combined = body_text
    if table_blocks:
        combined += "\n\n" + "\n\n".join(table_blocks)
    combined = normalize_whitespace(combined)
    if not table_detected and detect_table_like_text(combined):
        table_detected = True
    return combined, table_detected

# ----------------------------
# PDF extraction (PyMuPDF)
# ----------------------------
def extract_text_from_pdf(pdf_path: Path, max_pages: int = 0) -> Tuple[str, int, bool]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Please `pip install pymupdf`.")
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        pages = range(page_count) if max_pages <= 0 else range(min(max_pages, page_count))
        chunks = []
        for i in pages:
            page = doc[i]
            txt = page.get_text("text")
            if txt:
                chunks.append(txt)
        text = "\n".join(chunks)
        text = normalize_whitespace(text)
        table_detected = detect_table_like_text(text)
    return text, page_count, table_detected

# ----------------------------
# IO helpers
# ----------------------------
def ensure_out_dirs(base: Path) -> Tuple[Path, Path]:
    text_dir = base / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    return base, text_dir

def load_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_MINIMAL, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df

# ----------------------------
# Row processing (manifest mode)
# ----------------------------
def process_row(row: pd.Series,
                out_text_dir: Path,
                prefer: str,
                single_output: bool,
                max_pages: int,
                overwrite: bool,
                verbose: bool) -> List[Dict[str, str]]:
    meta_records: List[Dict[str, str]] = []

    key = (row.get("key") or row.get("id") or "").strip()
    title = (row.get("title") or row.get("name") or "").strip()
    url = (row.get("url") or row.get("link") or "").strip()

    if not key:
        for k in ["key", "id", "uid", "doi", "wosid"]:
            if k in row and str(row[k]).strip():
                key = str(row[k]).strip()
                break

    pdf_col = None
    html_col = None
    for c in ["pdf", "pdf_path", "pdffile", "file_pdf"]:
        if c in row and str(row[c]).strip():
            pdf_col = str(row[c]).strip()
            break
    for c in ["html", "html_path", "htmlfile", "file_html"]:
        if c in row and str(row[c]).strip():
            html_col = str(row[c]).strip()
            break

    pdf_path = Path(pdf_col) if pdf_col else None
    html_path = Path(html_col) if html_col else None

    if not key:
        return [{
            "key": "",
            "title": title,
            "source_type": "",
            "txt_path": "",
            "text_length": "0",
            "table_detected": "0",
            "parse_quality": "low",
            "notes": "SKIP: missing key",
            "page_count": "",
            "url": url
        }]

    have_pdf = bool(pdf_path and pdf_path.exists() and pdf_path.is_file())
    have_html = bool(html_path and html_path.exists() and html_path.is_file())

    if not have_pdf and not have_html:
        if verbose:
            print(f"[SKIP] No HTML/PDF found for {key}  title={title[:80]}")
        return [{
            "key": key,
            "title": title,
            "source_type": "",
            "txt_path": "",
            "text_length": "0",
            "table_detected": "0",
            "parse_quality": "low",
            "notes": "SKIP: no input file",
            "page_count": "",
            "url": url
        }]

    def write_record(text: str, source_type: str, page_count: Optional[int], table_flag: bool) -> Dict[str, str]:
        safe_src = source_type.lower()
        out_name = f"{key}.{safe_src}.txt"
        out_path = out_text_dir / out_name

        if out_path.exists() and not overwrite:
            try:
                prev = out_path.read_text(encoding="utf-8", errors="ignore")
                text_len = len(prev)
            except Exception:
                prev = ""
                text_len = 0
            return {
                "key": key,
                "title": title,
                "source_type": source_type,
                "txt_path": str(out_path.relative_to(out_text_dir.parent)),
                "text_length": str(text_len),
                "table_detected": "1" if table_flag else "0",
                "parse_quality": estimate_parse_quality(prev if text_len else "", source_type) if text_len else "low",
                "notes": "OK (skipped write; exists)",
                "page_count": str(page_count) if page_count is not None else "",
                "url": url
            }

        out_path.write_text(text, encoding="utf-8")
        text_len = len(text)
        quality = estimate_parse_quality(text, source_type)
        return {
            "key": key,
            "title": title,
            "source_type": source_type,
            "txt_path": str(out_path.relative_to(out_text_dir.parent)),
            "text_length": str(text_len),
            "table_detected": "1" if table_flag else "0",
            "parse_quality": quality,
            "notes": "OK",
            "page_count": str(page_count) if page_count is not None else "",
            "url": url
        }

    order: List[Tuple[str, Path]] = []
    if have_pdf and have_html:
        if prefer.lower() == "html":
            order = [("HTML", html_path), ("PDF", pdf_path)]
        else:
            order = [("PDF", pdf_path), ("HTML", html_path)]
    elif have_pdf:
        order = [("PDF", pdf_path)]
    else:
        order = [("HTML", html_path)]

    for _, (stype, path_obj) in enumerate(order):
        try:
            if stype == "HTML":
                text, table_flag = extract_text_from_html(path_obj)
                rec = write_record(text, "HTML", None, table_flag)
            else:
                text, pgc, table_flag = extract_text_from_pdf(path_obj, max_pages=max_pages)
                if not text.strip():
                    raise RuntimeError("no extractable text (scanned/encrypted?)")
                rec = write_record(text, "PDF", pgc, table_flag)
            meta_records.append(rec)
        except Exception as e:
            meta_records.append({
                "key": key,
                "title": title,
                "source_type": stype,
                "txt_path": "",
                "text_length": "0",
                "table_detected": "0",
                "parse_quality": "low",
                "notes": f"ERROR: {type(e).__name__}: {e}",
                "page_count": "",
                "url": url
            })
        if single_output:
            break

    return meta_records

# ----------------------------
# Legacy single-PDF API shim
# ----------------------------
def process_pdf(pdf_path: Path,
                outdir: Path,
                keep_sections: List[str] = None,
                tables: str = "none",
                debug_trace: bool = False,
                debug_skim: bool = False,
                preview_lines: int = 6) -> Tuple[bool, dict]:
    """
    Legacy-compatible single-PDF cleaner.

    Writes: <outdir>/text/<stem>.pdf.txt
    Returns: (ok: bool, meta: dict) where meta has fields:
        file, out_txt, text_length, page_count, table_detected, parse_quality, notes
    Notes:
        - We do NOT attempt section splitting here (keep_sections ignored),
          because this unified cleaner focuses on robust plain-text extraction.
        - 'tables' is accepted for API compatibility but not used here.
    """
    meta = {
        "file": str(pdf_path),
        "out_txt": "",
        "text_length": 0,
        "page_count": 0,
        "table_detected": 0,
        "parse_quality": "low",
        "notes": "",
    }
    try:
        outdir = Path(outdir)
        _, text_dir = ensure_out_dirs(outdir)

        if not pdf_path.exists():
            meta["notes"] = "ERROR: file not found"
            return False, meta

        text, page_count, table_flag = extract_text_from_pdf(pdf_path, max_pages=0)
        if not text.strip():
            meta["notes"] = "ERROR: no extractable text (scanned/encrypted?)"
            return False, meta

        out_txt = text_dir / f"{pdf_path.stem}.pdf.txt"
        out_txt.write_text(text, encoding="utf-8")

        meta.update({
            "out_txt": str(out_txt),
            "text_length": len(text),
            "page_count": page_count,
            "table_detected": 1 if table_flag else 0,
            "parse_quality": estimate_parse_quality(text, "PDF"),
            "notes": "OK",
        })
        return True, meta

    except Exception as e:
        meta["notes"] = f"ERROR: {type(e).__name__}: {e}"
        return False, meta

# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Clean text from PDF/HTML and produce key2txt.tsv with quality metadata."
    )
    # New (manifest) interface
    ap.add_argument("--manifest", required=False, help="Path to CSV/TSV manifest.")
    ap.add_argument("--out-dir", required=False, help="Output base directory (e.g., ./data/cleaned).")
    ap.add_argument("--prefer", choices=["pdf", "html"], default="html",
                    help="If both sources exist, which to prefer as the primary (default: html).")
    ap.add_argument("--single-output", action="store_true",
                    help="Only export the preferred source when both exist.")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="PDF: limit number of pages to extract (0=all).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing text files.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")

    # ---- Legacy compatibility (aliases / no-ops) ----
    # Positional single-PDF path: `pdf2clean.py <file.pdf>`
    ap.add_argument("pdf", nargs="?", default=None,
                    help="LEGACY: Single PDF file to clean (if provided, manifest is ignored).")
    # `--outdir` alias of `--out-dir`
    ap.add_argument("--outdir", required=False, help="LEGACY alias of --out-dir")
    # Old flags accepted but ignored in manifest mode; passed through in legacy single-file mode
    ap.add_argument("--tables", choices=["camelot", "tabula", "none"], default="none",
                    help="LEGACY: table extraction hint (ignored in manifest mode).")
    ap.add_argument("--debug-trace", action="store_true",
                    help="LEGACY: dump intermediates (ignored in manifest mode).")

    return ap


def main():
    # Fixed defaults if no args passed (so double-click / empty run works)
    if len(sys.argv) == 1:
        sys.argv += [
            "--manifest", r".\data\cleaned\manifest.tsv",
            "--out-dir",  r".\data\cleaned",
            "--prefer",   "html",
            "--overwrite",
            "--verbose",
        ]

    ap = build_arg_parser()

    # Accept unknown legacy extras instead of crashing
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unrecognized arguments: {' '.join(unknown)}")

    # ---- Resolve out_dir from either --out-dir or legacy --outdir
    out_dir_arg = args.out_dir or args.outdir or r".\data\cleaned"

    # ---- Legacy single-file mode (take priority if a positional PDF is provided)
    if args.pdf:
        pdf_path = Path(args.pdf)
        out_base = Path(out_dir_arg).expanduser().resolve()
        out_base.mkdir(parents=True, exist_ok=True)

        # Ensure text/ subdir exists
        _, text_dir = ensure_out_dirs(out_base)

        ok, meta = process_pdf(
            pdf_path=pdf_path,
            outdir=out_base,
            keep_sections=None,         # not used by unified cleaner
            tables=args.tables,         # accepted for compatibility
            debug_trace=args.debug_trace,
            debug_skim=False,
            preview_lines=6,
        )
        status = "[OK]" if ok else "[ERR]"
        print(f"{status} Legacy single-file: {pdf_path.name} -> {meta.get('out_txt','(no out)')} | notes={meta.get('notes')}")
        return

    # ---- Manifest mode (default)
    if not args.manifest:
        # If no positional PDF and no manifest, fall back to fixed default
        args.manifest = r".\data\cleaned\manifest.tsv"

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_base = Path(out_dir_arg).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    _, out_text_dir = ensure_out_dirs(out_base)

    df = load_manifest(manifest_path)
    if args.verbose:
        print(f"[INFO] Loaded manifest: {manifest_path}  rows={len(df)}")

    key_col   = pick_col(df, ["key", "id", "uid", "wosid", "doi"])
    title_col = pick_col(df, ["title", "name"])
    pdf_col   = pick_col(df, ["pdf", "pdf_path", "pdffile", "file_pdf"])
    html_col  = pick_col(df, ["html", "html_path", "htmlfile", "file_html"])
    url_col   = pick_col(df, ["url", "link"])

    work = pd.DataFrame()
    work["key"]   = df[key_col] if key_col else df.index.astype(str)
    work["title"] = df[title_col] if title_col else ""
    work["pdf"]   = df[pdf_col] if pdf_col else ""
    work["html"]  = df[html_col] if html_col else ""
    work["url"]   = df[url_col] if url_col else ""

    all_meta: List[Dict[str, str]] = []
    for _, row in work.iterrows():
        metas = process_row(row, out_text_dir,
                            prefer=args.prefer,
                            single_output=args.single_output,
                            max_pages=args.max_pages,
                            overwrite=args.overwrite,
                            verbose=args.verbose)
        all_meta.extend(metas)

    out_tsv = out_base / "key2txt.tsv"
    meta_df = pd.DataFrame(all_meta, columns=[
        "key","title","source_type","txt_path","text_length",
        "table_detected","parse_quality","notes","page_count","url"
    ])
    meta_df.to_csv(out_tsv, sep="\t", index=False)

    ok = (meta_df["notes"].str.startswith("OK")).sum()
    skipped = (meta_df["notes"].str.startswith("SKIP")).sum()
    errors = (meta_df["notes"].str.startswith("ERROR")).sum()
    produced = (meta_df["txt_path"].str.len() > 0).sum()

    print(f"[OK] key2txt -> {out_tsv}")
    print(f"[INFO] Produced text files: {produced}")
    print(f"[INFO] OK records: {ok} | SKIP: {skipped} | ERROR: {errors}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user.")
