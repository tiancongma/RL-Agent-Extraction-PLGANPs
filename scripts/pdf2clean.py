#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf2clean.py

Extract and clean scientific PDF text for LLM consumption, focusing on core sections:
Abstract, Materials/Methods, Results, Discussion, Conclusion.
Supports table extraction (Camelot or Tabula), plus two debug modes:

- --debug-trace : dump intermediate artifacts to out/debug/
- --debug-skim  : analyze only (prints previews), do NOT write cleaned outputs

Enhancement (robust skipping)
-----------------------------
- In single-file and directory modes, this script now **SKIPs** files that:
    * do not exist,
    * have size < --min-size (default: 1024 bytes),
    * have unsupported extension (non-PDF; HTML can be optionally allowed but still skipped from PDF cleaning).
- SKIPs are logged as:
    [SKIP] No HTML/PDF found for <name> (reason: <text>)
  and the program **does not** exit with error.

You can also import this file as a module and call:
  process_pdf(pdf_path: Path, outdir: Path, keep_sections: List[str], tables: str,
              debug_trace=False, debug_skim=False, preview_lines=6) -> (ok: bool, meta: dict)
"""

import argparse
import json
import os
import re
import regex as re2
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from unidecode import unidecode

# ---------- optional imports (lazy for table extraction) ----------
def _try_import_camelot():
    try:
        import camelot  # type: ignore
        return camelot
    except Exception:
        return None

def _try_import_tabula():
    try:
        import tabula  # type: ignore
        return tabula
    except Exception:
        return None

# ---------- PDF text extraction (PyMuPDF) ----------
def extract_pages_text(pdf_path: Path) -> List[str]:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        blocks = page.get_text("blocks")
        # sort by y, then x to reduce column mixing
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        text = "\n".join(b[4] for b in blocks if b[4].strip())
        pages.append(text)
    doc.close()
    return pages

# ---------- cleaning helpers ----------
HEADER_FOOTER_MAXLEN = 120

def remove_headers_footers(pages: List[str]) -> List[str]:
    from collections import Counter
    candidates = Counter()
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip() and len(l.strip()) <= HEADER_FOOTER_MAXLEN]
        head = lines[:3]
        tail = lines[-3:]
        for l in head + tail:
            candidates[l] += 1
    threshold = max(2, len(pages) // 4)  # appears on >= 25% of pages
    to_remove = set(k for k, v in candidates.items() if v >= threshold)

    cleaned = []
    for p in pages:
        kept = []
        for l in p.splitlines():
            s = l.strip()
            if s in to_remove:
                continue
            kept.append(l)
        cleaned.append("\n".join(kept))
    return cleaned

def normalize_text(txt: str) -> str:
    t = unidecode(txt)              # fi/ff ligatures -> ascii
    t = t.replace("\r", "")
    t = re2.sub(r"(\w)-\n(\w)", r"\1\2", t)  # poly-\nmer -> polymer
    t = re2.sub(r"[ \t]+\n", "\n", t)
    t = re2.sub(r"\n{3,}", "\n\n", t)
    t = re2.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

# ---------- section detection ----------
SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "introduction": r"\bintroduction\b",
    "materials": r"\bmaterials?\b",
    "methods": r"\bmethods?\b|\bmethodology\b|\bexperimental\b|\bmaterials and methods\b",
    "results": r"\bresults?\b",
    "discussion": r"\bdiscussion\b|\bresults and discussion\b",
    "conclusion": r"\bconclusion(s)?\b|\bconcluding remarks\b",
    "acknowledgments": r"\backnowledg(e)?ments?\b",
    "supplementary": r"\bsupplement(ary)? (information|data|material)\b",
    "references": r"\breferences\b|\bbibliography\b|^refs?\.$",
}

def split_into_sections(full_text: str) -> Dict[str, str]:
    pats = [(re2.compile(p, re2.I), k) for k, p in SECTION_PATTERNS.items()]
    lines = full_text.splitlines()
    indices = []
    for ln, line in enumerate(lines):
        s = line.strip()
        if 0 < len(s) <= 80:
            for pat, key in pats:
                if pat.search(s):
                    indices.append((ln, key, s))
                    break
    if not indices:
        return {"body": full_text}

    result = {}
    indices.sort(key=lambda x: x[0])
    for i, (ln, key, _hdr) in enumerate(indices):
        start = ln + 1
        end = indices[i+1][0] if i+1 < len(indices) else len(lines)
        chunk = "\n".join(lines[start:end]).strip()
        result.setdefault(key, "")
        result[key] += ("\n\n" + chunk if result[key] else chunk)

    # merge materials + methods when present
    if "materials" in result or "methods" in result:
        mm = []
        if "materials" in result: mm.append(result.pop("materials"))
        if "methods" in result:   mm.append(result.pop("methods"))
        result["materials_methods"] = "\n\n".join(mm).strip()
    return result

# ---------- content pruning ----------
FIGCAP_RE = re2.compile(r"^(figure|fig\.)\s*\d+[:.].*$", re2.I | re2.M)
TABLECAP_RE = re2.compile(r"^(table)\s*\d+[:.].*$", re2.I | re2.M)
CITATION_INLINE_RE = re2.compile(r"\((?:[A-Z][A-Za-z\-]+ et al\.,?\s*\d{4}|[A-Z][A-Za-z\-]+,\s*\d{4}|[\d,\s;]+)\)")
BRACKET_CITE_RE = re2.compile(r"\[\d{1,3}(?:[,\-\s]\d{1,3})*\]")

def prune_noise(txt: str) -> str:
    t = txt
    t = FIGCAP_RE.sub("", t)
    t = TABLECAP_RE.sub("", t)
    t = CITATION_INLINE_RE.sub("", t)
    t = BRACKET_CITE_RE.sub("", t)
    t = re2.sub(r"\n{3,}", "\n\n", t).strip()
    return t

DEFAULT_KEEP = {"abstract", "materials_methods", "results", "discussion", "conclusion"}

def select_core_sections(sections: Dict[str, str], keep: Optional[set]) -> Tuple[str, Dict[str, str]]:
    if keep is None:
        keep = DEFAULT_KEEP
    kept = {k: v for k, v in sections.items() if k in keep and v.strip()}
    ordered = []
    for k in ["abstract", "materials_methods", "results", "discussion", "conclusion"]:
        if k in kept:
            header = k.replace("_", " ").title()
            ordered.append(f"## {header}\n{kept[k].strip()}")
    return "\n\n".join(ordered).strip(), kept

# ---------- tables ----------
def extract_tables(pdf_path: Path, method: str, outdir: Path) -> int:
    out_tables = outdir / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    count = 0

    if method == "camelot":
        camelot = _try_import_camelot()
        if camelot is None:
            print("[WARN] Camelot not available. Skipping tables.", file=sys.stderr)
            return 0
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
            for i, t in enumerate(tables):
                fp = out_tables / f"{pdf_path.stem}_table_{count+i+1}.csv"
                t.to_csv(str(fp), index=False)
            count += len(tables)
        except Exception:
            pass
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
            for i, t in enumerate(tables):
                fp = out_tables / f"{pdf_path.stem}_table_{count+i+1}.csv"
                t.to_csv(str(fp), index=False)
            count += len(tables)
        except Exception:
            pass
        return count

    if method == "tabula":
        tabula = _try_import_tabula()
        if tabula is None:
            print("[WARN] Tabula not available. Skipping tables.", file=sys.stderr)
            return 0
        try:
            dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True, lattice=True, stream=True)
            for i, df in enumerate(dfs):
                if df is None or df.empty:
                    continue
                fp = outdir / "tables" / f"{pdf_path.stem}_table_{count+i+1}.csv"
                df.to_csv(fp, index=False)
            count += len(dfs)
        except Exception:
            pass
        return count

    return 0

# ---------- debug helpers ----------
def safe_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def est_tokens(chars: int) -> int:
    # rough estimate: 1 token ≈ 4 chars
    return max(1, chars // 4)

# ---------- file gating (NEW) ----------
SUPPORTED_PDF = {".pdf"}
SUPPORTED_HTML = {".htm", ".html"}  # 我们不做 HTML 清洗，仅用于“受支持但跳过”的判断

def _file_reason_to_skip(p: Path, min_size: int, allow_html: bool) -> Optional[str]:
    """Return reason string if the file should be SKIPPED; otherwise None."""
    if not p.exists():
        return "not downloaded"
    try:
        size = p.stat().st_size
    except Exception:
        return "stat failed"
    if size < min_size:
        return f"too small ({size}B)"
    ext = p.suffix.lower()
    if ext in SUPPORTED_PDF:
        return None
    if allow_html and ext in SUPPORTED_HTML:
        # 允许识别 HTML 以避免误报失败，但我们仍然跳过 PDF 清洗
        return "html attachment (skipped for PDF cleaning)"
    return f"unsupported type ({ext})"

# ---------- main worker ----------
def process_pdf(pdf_path: Path, outdir: Path, keep_sections: List[str], tables: str,
                debug_trace: bool=False, debug_skim: bool=False, preview_lines: int=6) -> Tuple[bool, dict]:
    t0 = time.time()
    meta = {"file": str(pdf_path), "sections": {}, "kept": [], "tables": 0, "time_sec": 0.0}
    try:
        # 1) raw extraction
        pages = extract_pages_text(pdf_path)
        if debug_trace:
            safe_write(outdir / f"debug/{pdf_path.stem}.raw_pages.txt", "\n\n---PAGE---\n\n".join(pages))

        # 2) remove headers/footers
        pages2 = remove_headers_footers(pages)
        if debug_trace:
            safe_write(outdir / f"debug/{pdf_path.stem}.noheader.txt", "\n\n---PAGE---\n\n".join(pages2))

        # 3) normalize
        full = "\n\n".join(pages2)
        full = normalize_text(full)
        if debug_trace:
            safe_write(outdir / f"debug/{pdf_path.stem}.normalized.txt", full)

        # 4) cut after references/bibliography if present
        full = re2.split(r"\n(?i:references|bibliography)\n", full, maxsplit=1)[0]

        # 5) split sections
        sections = split_into_sections(full)
        if debug_trace:
            safe_write(outdir / f"debug/{pdf_path.stem}.sections.preprune.json",
                       json.dumps(sections, ensure_ascii=False, indent=2))

        # 6) prune noise within sections
        for k in list(sections.keys()):
            sections[k] = prune_noise(sections[k])

        # 7) keep-only selection
        keep = set([s.strip().lower() for s in keep_sections]) if keep_sections else None
        cleaned_text, kept_sections = select_core_sections(sections, keep)

        # fill meta
        for k, v in sections.items():
            meta["sections"][k] = {"chars": len(v), "tokens_est": est_tokens(len(v))}
        meta["kept"] = list(kept_sections.keys())

        # debug skim
        if debug_skim:
            print(f"\n[SKIM] {pdf_path.name}")
            for k in meta["kept"]:
                text = kept_sections[k]
                lines = text.splitlines()
                head = "\n".join(lines[:preview_lines])
                print(f"  - {k}: {len(text)} chars (~{est_tokens(len(text))} tok)\n    preview:\n{head}\n")
            meta["time_sec"] = round(time.time() - t0, 3)
            return True, meta

        # 8) outputs
        out_txt = outdir / f"{pdf_path.stem}.cleaned.txt"
        out_json = outdir / f"{pdf_path.stem}.sections.json"
        safe_write(out_txt, cleaned_text)
        safe_write(out_json, json.dumps(kept_sections, ensure_ascii=False, indent=2))
        if debug_trace:
            safe_write(outdir / f"debug/{pdf_path.stem}.sections.keep.txt", cleaned_text)

        # 9) tables
        table_count = 0
        if tables and tables.lower() in {"camelot", "tabula"}:
            table_count = extract_tables(pdf_path, tables.lower(), outdir)
        meta["tables"] = table_count

        print(f"[OK] {pdf_path.name} -> {out_txt.name} | sections={meta['kept']} | tables={table_count}")
        meta["time_sec"] = round(time.time() - t0, 3)
        return True, meta

    except Exception as e:
        print(f"[ERR] {pdf_path.name}: {e}", file=sys.stderr)
        meta["time_sec"] = round(time.time() - t0, 3)
        return False, meta

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Extract & clean scientific PDFs for LLMs.")
    # original args
    ap.add_argument("pdf", nargs="?", default=None, help="Input PDF (optional if using --indir).")
    ap.add_argument("--indir", type=str, default=None, help="Directory containing PDFs (recursive).")
    ap.add_argument("--pattern", type=str, default="**/*.pdf", help="Glob pattern under --indir.")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N PDFs (small-batch).")
    ap.add_argument("--outdir", type=str, default="out", help="Output directory.")
    ap.add_argument("--keep-sections", type=str,
                    default="abstract,materials_methods,results,discussion,conclusion",
                    help="Comma-separated section keys to keep "
                         "(abstract, materials, methods, materials_methods, results, discussion, conclusion)")
    ap.add_argument("--tables", type=str, default="none", choices=["camelot", "tabula", "none"],
                    help="Extract tables using Camelot or Tabula.")
    ap.add_argument("--debug-trace", action="store_true", help="Dump intermediates to out/debug/.")
    ap.add_argument("--debug-skim", action="store_true", help="Analyze only; do not write cleaned outputs.")
    ap.add_argument("--preview-lines", type=int, default=6, help="Preview lines per section in --debug-skim.")
    # new gating args
    ap.add_argument("--min-size", type=int, default=1024,
                    help="Minimum file size (bytes) to consider as 'has content'. Default: 1024.")
    ap.add_argument("--allow-html", action="store_true",
                    help="Treat .htm/.html as recognized but still SKIP (prevents 'unsupported' noise).")
    return ap.parse_args()

def _list_usable_pdfs_from_dir(root: Path, pattern: str, min_size: int, allow_html: bool) -> List[Path]:
    files = sorted(root.glob(pattern))
    usable: List[Path] = []
    for p in files:
        reason = _file_reason_to_skip(p, min_size, allow_html)
        if reason is None:
            usable.append(p)
        else:
            print(f"[SKIP] No HTML/PDF found for {p.name} (reason: {reason})")
    return usable

def _run_on_single_path(p: Path, min_size: int, allow_html: bool,
                        outdir: Path, keep: List[str], tables: str,
                        debug_trace: bool, debug_skim: bool, preview_lines: int) -> Tuple[int, int]:
    """Return (ok_cnt, fail_cnt); handles SKIP gracefully."""
    reason = _file_reason_to_skip(p, min_size, allow_html)
    if reason is not None:
        print(f"[SKIP] No HTML/PDF found for {p.name} (reason: {reason})")
        return (0, 0)
    ok, _meta = process_pdf(pdf_path=p,
                            outdir=outdir,
                            keep_sections=keep,
                            tables=tables,
                            debug_trace=debug_trace,
                            debug_skim=debug_skim,
                            preview_lines=preview_lines)
    return (1, 0) if ok else (0, 1)

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    keep = [s.strip().lower() for s in args.keep_sections.split(",") if s.strip()]

    total = okc = failc = 0

    # single file
    if args.pdf:
        p = Path(args.pdf)
        _ok, _fail = _run_on_single_path(p, args.min_size, args.allow_html,
                                         outdir, keep, args.tables,
                                         args.debug_trace, args.debug_skim, args.preview_lines)
        total = 1 if (_ok or _fail) else 1  # even SKIP counts as 1 attempted input
        okc += _ok
        failc += _fail

    # directory batch
    if args.indir:
        root = Path(args.indir)
        if not root.exists():
            print(f"[ERR] Indir not found: {root}", file=sys.stderr)
            sys.exit(1)
        cand = _list_usable_pdfs_from_dir(root, args.pattern, args.min_size, args.allow_html)
        if args.limit and args.limit > 0:
            cand = cand[:args.limit]
        total += len(cand)
        for p in cand:
            _ok, _fail = _run_on_single_path(p, args.min_size, args.allow_html,
                                             outdir, keep, args.tables,
                                             args.debug_trace, args.debug_skim, args.preview_lines)
            okc += _ok
            failc += _fail

    if not args.pdf and not args.indir:
        print("[ERR] No input PDFs. Provide a file or --indir.", file=sys.stderr)
        sys.exit(1)

    # summary
    print(f"\n=== SUMMARY ===")
    print(f"files: {total}, ok: {okc}, failed: {failc}")

if __name__ == "__main__":
    main()
