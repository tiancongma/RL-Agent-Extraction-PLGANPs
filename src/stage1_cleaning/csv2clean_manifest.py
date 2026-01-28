#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
csv2clean_manifest.py  (HTML-first, PDF-fallback, resumable; paths always preserved)

What’s new vs your previous version:
- Resolve all attachment candidates up-front (HTML & PDF), cache best guesses.
- Write paths.html / paths.pdf in ALL branches (OK_*, SKIP_EXISTING, SKIP_RESUME, SKIP).
- For SKIP_RESUME/SKIP_EXISTING, try to infer paths.text / sections_json from known stem.
- Add source_type ("html"|"pdf"|None) and keep source_mode for trace.
"""

import argparse, json, os, sys, shutil, subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import regex as re2
from unidecode import unidecode
try:
    import trafilatura
except Exception:
    trafilatura = None
from bs4 import BeautifulSoup  # type: ignore

# ---------- CSV helpers ----------
CAND_ID_COLS    = ["Key", "itemKey", "zotero_key"]
CAND_DOI_COLS   = ["DOI", "doi"]
CAND_TITLE_COLS = ["Title", "title"]
CAND_YEAR_COLS  = ["Year", "Date", "date", "year"]
CAND_PDF_COLS   = ["File Attachments", "Attachments", "Attachment", "File", "file", "files"]

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def split_attachment_paths(val: str) -> List[str]:
    if not val:
        return []
    parts = re2.split(r";\s*", str(val))
    out = []
    for p in parts:
        out.extend([q for q in p.split("|") if q.strip()])
    return [p.strip() for p in out if p.strip()]

# ---------- text normalization / sectioning ----------
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
FIGCAP_RE = re2.compile(r"^(figure|fig\.)\s*\d+[:.].*$", re2.I | re2.M)
TABLECAP_RE = re2.compile(r"^(table)\s*\d+[:.].*$", re2.I | re2.M)
CITATION_INLINE_RE = re2.compile(r"\((?:[A-Z][A-Za-z\-]+ et al\.,?\s*\d{4}|[A-Z][A-Za-z\-]+,\s*\d{4}|[\d,\s;]+)\)")
BRACKET_CITE_RE = re2.compile(r"\[\d{1,3}(?:[,\-\s]\d{1,3})*\]")

def normalize_text(txt: str) -> str:
    t = unidecode(txt).replace("\r", "")
    t = re2.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re2.sub(r"[ \t]+\n", "\n", t)
    t = re2.sub(r"\n{3,}", "\n\n", t)
    t = re2.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def prune_noise(txt: str) -> str:
    t = txt
    t = FIGCAP_RE.sub("", t)
    t = TABLECAP_RE.sub("", t)
    t = CITATION_INLINE_RE.sub("", t)
    t = BRACKET_CITE_RE.sub("", t)
    t = re2.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def split_into_sections(full_text: str) -> Dict[str, str]:
    pats = [(re2.compile(p, re2.I), k) for k, p in SECTION_PATTERNS.items()]
    lines = full_text.splitlines()
    indices = []
    for ln, line in enumerate(lines):
        s = line.strip()
        if 0 < len(s) <= 80:
            for pat, key in pats:
                if pat.search(s):
                    indices.append((ln, key, s)); break
    if not indices:
        return {"body": full_text}
    result = {}
    indices.sort(key=lambda x: x[0])
    for i, (ln, key, _hdr) in enumerate(indices):
        start = ln + 1
        end = indices[i+1][0] if i+1 < len(indices) else len(lines)
        chunk = "\n".join(lines[start:end]).strip()
        result.setdefault(key, ""); result[key] += ("\n\n" + chunk if result[key] else chunk)
    if "materials" in result or "methods" in result:
        mm = []
        if "materials" in result: mm.append(result.pop("materials"))
        if "methods" in result:   mm.append(result.pop("methods"))
        result["materials_methods"] = "\n\n".join(mm).strip()
    return result

def select_core_sections(sections: Dict[str, str], keep=None) -> Tuple[str, Dict[str, str]]:
    keep_set = set(keep or ["abstract","materials_methods","results","discussion","conclusion"])
    kept = {k:v for k,v in sections.items() if k in keep_set and v.strip()}
    ordered=[]
    for k in ["abstract","materials_methods","results","discussion","conclusion"]:
        if k in kept:
            ordered.append(f"## {k.replace('_',' ').title()}\n{kept[k].strip()}")
    return "\n\n".join(ordered).strip(), kept

# ---------- attachment resolution ----------
def resolve_html(entry: str, pdf_root: Optional[Path], item_key: Optional[str]) -> Optional[Path]:
    if not entry: return None
    raw = entry.strip().strip('"').strip()
    cand = Path(raw)
    if cand.is_absolute() and cand.exists() and cand.suffix.lower() in (".html",".htm"):
        return cand
    if raw.lower().startswith("storage:") and pdf_root:
        parts = raw.split(":")
        if len(parts) >= 3:
            key = parts[1]; filename = parts[-1]
            p = Path(pdf_root) / key / filename
            if p.exists() and p.suffix.lower() in (".html",".htm"): return p
        if len(parts) >= 2:
            key = parts[1]
            for fn in ("index.html","index.htm"):
                p = Path(pdf_root)/key/fn
                if p.exists(): return p
            hits = list((Path(pdf_root)/key).glob("*.htm*"))
            if hits: return sorted(hits)[0]
    if pdf_root:
        p = Path(pdf_root) / raw
        if p.exists() and p.suffix.lower() in (".html",".htm"): return p
        parts = Path(raw).parts
        if len(parts) >= 3 and parts[0].lower()=="storage":
            p2 = Path(pdf_root) / Path(*parts[1:])
            if p2.exists() and p2.suffix.lower() in (".html",".htm"): return p2
            if p2.exists() and p2.is_dir():
                for fn in ("index.html","index.htm"):
                    if (p2/fn).exists(): return p2/fn
                hits = list(p2.glob("*.htm*"))
                if hits: return sorted(hits)[0]
        if len(parts)==1:
            p3 = Path(pdf_root) / parts[0]
            if p3.exists() and p3.is_dir():
                for fn in ("index.html","index.htm"):
                    if (p3/fn).exists(): return p3/fn
                hits = list(p3.glob("*.htm*"))
                if hits: return sorted(hits)[0]
    if pdf_root and item_key:
        d = Path(pdf_root)/item_key
        if d.exists():
            for fn in ("index.html","index.htm"):
                if (d/fn).exists(): return d/fn
            hits = list(d.glob("*.htm*"))
            if hits: return sorted(hits)[0]
    return None

def resolve_pdf(entry: str, pdf_root: Optional[Path]) -> Optional[Path]:
    if not entry: return None
    raw = entry.strip().strip('"').strip()
    cand = Path(raw)
    if cand.exists() and cand.suffix.lower()==".pdf":
        return cand
    if raw.lower().startswith("storage:") and pdf_root:
        parts = raw.split(":")
        if len(parts) >= 3:
            key = parts[1]; filename = parts[-1]
            p = pdf_root / key / filename
            if p.exists() and p.suffix.lower()==".pdf": return p
        if len(parts) >= 2:
            key = parts[1]
            hits = list((pdf_root / key).glob("*.pdf"))
            if hits: return sorted(hits)[0]
    if pdf_root:
        p = pdf_root / raw
        if p.exists() and p.suffix.lower()==".pdf":
            return p
        parts = Path(raw).parts
        if len(parts) >= 3 and parts[0].lower()=="storage":
            p2 = pdf_root / Path(*parts[1:])
            if p2.exists() and p2.suffix.lower()==".pdf": return p2
            if p2.exists() and p2.is_dir():
                hits = list(p2.glob("*.pdf"))
                if hits: return sorted(hits)[0]
        if len(parts)==1:
            p3 = pdf_root / parts[0]
            if p3.exists() and p3.is_dir():
                hits = list(p3.glob("*.pdf"))
                if hits: return sorted(hits)[0]
    return None

# ---------- pdf2clean import-or-CLI ----------
_pdf2clean_mod = None
_resolved_pdf2clean: Optional[Path] = None

def try_import_pdf2clean(module_path: Path):
    import importlib.util
    candidates = [module_path, Path(__file__).parent/"pdf2clean.py", Path.cwd()/ "scripts"/"pdf2clean.py", Path.cwd()/ "pdf2clean.py"]
    env_p = os.getenv("PDF2CLEAN_PATH")
    if env_p: candidates.append(Path(env_p))
    for p in candidates:
        try:
            if p and p.exists():
                spec = importlib.util.spec_from_file_location("pdf2clean", str(p))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod, p
        except Exception as e:
            print(f"[WARN] Failed to import from {p}: {e}", file=sys.stderr)
    return None, None

def run_pdf2clean_cli(python_exe: str, pdf2clean_path: Path, pdf_path: Path,
                      outdir: Path, tables: str, debug_trace: bool,
                      debug_skim: bool, preview_lines: int) -> bool:
    cmd = [python_exe, str(pdf2clean_path), str(pdf_path),
           "--outdir", str(outdir), "--tables", tables]
    if debug_trace: cmd.append("--debug-trace")
    if debug_skim:
        cmd.append("--debug-skim")
        cmd.extend(["--preview-lines", str(preview_lines)])
    try:
        res = subprocess.run(cmd, check=False)
        return res.returncode == 0
    except Exception as e:
        print(f"[ERR] Subprocess failed: {e}", file=sys.stderr)
        return False

def load_ok_keys(manifest_path: Path) -> set:
    ok = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                st = str(rec.get("status", ""))
                if st.startswith("OK_") or st == "OK":
                    key = rec.get("zotero_key") or rec.get("key")
                    if key:
                        ok.add(str(key))
    return ok

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="HTML-first cleaning from Zotero CSV; fallback to PDF; build JSONL manifest (paths preserved).")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outroot", default="data")
    ap.add_argument("--tables", default="camelot", choices=["camelot","tabula","none"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--pdf2clean-path", default="scripts/pdf2clean.py")
    ap.add_argument("--pdf-root", default=None, help="Zotero storage root, e.g. C:\\Users\\YOU\\Zotero\\storage")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--doi-col", default=None)
    ap.add_argument("--title-col", default=None)
    ap.add_argument("--year-col", default=None)
    ap.add_argument("--pdf-col", default=None)
    ap.add_argument("--debug-trace", action="store_true")
    ap.add_argument("--debug-skim", action="store_true")
    ap.add_argument("--preview-lines", type=int, default=6)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}", file=sys.stderr); sys.exit(1)

    global _pdf2clean_mod, _resolved_pdf2clean
    _pdf2clean_mod, _resolved_pdf2clean = try_import_pdf2clean(Path(args.pdf2clean_path))
    use_cli_if_needed = _pdf2clean_mod is None
    if use_cli_if_needed:
        candidates = [Path(args.pdf2clean_path), Path(__file__).parent/"pdf2clean.py",
                      Path.cwd()/"scripts"/"pdf2clean.py", Path.cwd()/"pdf2clean.py"]
        env_p = os.getenv("PDF2CLEAN_PATH")
        if env_p: candidates.append(Path(env_p))
        hits = [p for p in candidates if p and p.exists()]
        if not hits:
            tried = "\n  - ".join(str(p) for p in candidates)
            print("[ERR] Could not locate pdf2clean.py for CLI. Tried:\n  - " + tried, file=sys.stderr)
            sys.exit(1)
        _resolved_pdf2clean = hits[0]
        print(f"[INFO] Using pdf2clean CLI: {_resolved_pdf2clean}")
    else:
        print(f"[INFO] Imported pdf2clean from: {_resolved_pdf2clean}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    id_col    = args.id_col    or pick_col(df, CAND_ID_COLS)
    doi_col   = args.doi_col   or pick_col(df, CAND_DOI_COLS)
    title_col = args.title_col or pick_col(df, CAND_TITLE_COLS)
    year_col  = args.year_col  or pick_col(df, CAND_YEAR_COLS)
    att_col   = args.pdf_col   or pick_col(df, CAND_PDF_COLS)
    if not id_col:  print(f"[ERR] Missing Zotero key col, tried {CAND_ID_COLS}", file=sys.stderr); sys.exit(1)
    if not att_col: print(f"[ERR] Missing attachment col, tried {CAND_PDF_COLS}", file=sys.stderr); sys.exit(1)

    work = df.copy()
    if args.limit: work = work.head(args.limit)

    outroot      = Path(args.outroot)
    cleaned_root = outroot / "cleaned"
    text_dir     = cleaned_root / "text"
    sections_dir = cleaned_root / "sections"
    tables_dir   = cleaned_root / "tables"
    manifest_dir = cleaned_root / "manifests"
    for d in (cleaned_root, text_dir, sections_dir, tables_dir, manifest_dir):
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "zotero_llm_relevant.jsonl"
    pdf_root = Path(args.pdf_root) if args.pdf_root else None

    ok_keys = load_ok_keys(manifest_path)
    fout_mode = "a" if manifest_path.exists() else "w"

    processed = skipped = failed = 0

    with manifest_path.open(fout_mode, encoding="utf-8") as fout:
        for _, row in work.iterrows():
            item_key = str(row.get(id_col)).strip()
            title    = str(row.get(title_col, "")).strip()
            year     = str(row.get(year_col, "")).strip()
            doi      = (str(row.get(doi_col, "")).strip() or None)

            # collect all attachment candidates (once)
            att_entries = split_attachment_paths(row.get(att_col, ""))
            html_path: Optional[Path] = None
            pdf_path: Optional[Path]  = None
            for e in att_entries:
                hp = resolve_html(e, pdf_root, item_key)
                if hp and hp.exists(): html_path = hp; break
            if not html_path:
                for e in att_entries:
                    pp = resolve_pdf(e, pdf_root)
                    if pp and pp.exists(): pdf_path = pp; break

            # resume branch: we still record paths if we decide to skip
            if args.skip_existing and item_key in ok_keys:
                # try to infer text/sections by stem (prefer html stem)
                stem = (html_path or pdf_path).stem if (html_path or pdf_path) else None
                text_path = str(text_dir / f"{stem}.cleaned.txt") if stem and (text_dir / f"{stem}.cleaned.txt").exists() else None
                sect_path = str(sections_dir / f"{stem}.sections.json") if stem and (sections_dir / f"{stem}.sections.json").exists() else None
                rec = {
                    "zotero_key": item_key, "title": title, "year": year, "doi": doi,
                    "paths": {
                        "pdf": str(pdf_path) if pdf_path else None,
                        "html": str(html_path) if html_path else None,
                        "text": text_path, "sections_json": sect_path,
                        "tables_csv": []
                    },
                    "sections": {}, "source_mode": "resume", "source_type": "html" if html_path else ("pdf" if pdf_path else None),
                    "status": "SKIP_RESUME", "message": "already OK in manifest"
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[SKIP resume] {item_key}")
                skipped += 1
                continue

            # try HTML first
            if html_path and html_path.exists():
                stem = html_path.stem
                final_txt  = text_dir / f"{stem}.cleaned.txt"
                final_json = sections_dir / f"{stem}.sections.json"

                if args.skip_existing and final_txt.exists() and final_json.exists():
                    rec = {
                        "zotero_key": item_key, "title": title, "year": year, "doi": doi,
                        "paths": {"pdf": str(pdf_path) if pdf_path else None, "html": str(html_path),
                                  "text": str(final_txt), "sections_json": str(final_json), "tables_csv": []},
                        "sections": {}, "source_mode": "cached", "source_type": "html",
                        "status": "SKIP_EXISTING", "message": "outputs already exist"
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[SKIP existing] {item_key} -> {final_txt.name}")
                    skipped += 1
                    continue

                # extract HTML
                raw = html_path.read_text(encoding="utf-8", errors="ignore")
                text = None
                if trafilatura is not None:
                    try: text = trafilatura.extract(raw, include_comments=False, include_tables=False)
                    except Exception: text = None
                if not text:
                    soup = BeautifulSoup(raw, "lxml")
                    for tag in soup(["script","style","nav","footer","header","noscript"]): tag.decompose()
                    text = soup.get_text("\n")

                text = normalize_text(text)
                text = re2.split(r"\n(?i:references|bibliography)\n", text, maxsplit=1)[0]
                sections = split_into_sections(text)
                for k in list(sections.keys()): sections[k] = prune_noise(sections[k])
                cleaned_text, kept = select_core_sections(sections, None)
                final_txt.parent.mkdir(parents=True, exist_ok=True)
                final_json.parent.mkdir(parents=True, exist_ok=True)
                final_txt.write_text(cleaned_text, encoding="utf-8")
                final_json.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

                rec = {
                    "zotero_key": item_key, "title": title, "year": year, "doi": doi,
                    "paths": {"pdf": str(pdf_path) if pdf_path else None, "html": str(html_path),
                              "text": str(final_txt), "sections_json": str(final_json), "tables_csv": []},
                    "sections": kept, "source_mode": "html", "source_type": "html",
                    "status": "OK_HTML", "message": "cleaned successfully"
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[OK][HTML] {html_path.name} -> cleaned + sections")
                processed += 1
                continue

            # fallback to PDF
            if pdf_path and pdf_path.exists():
                stem = pdf_path.stem
                final_txt  = text_dir / f"{stem}.cleaned.txt"
                final_json = sections_dir / f"{stem}.sections.json"

                if args.skip_existing and final_txt.exists() and final_json.exists():
                    rec = {
                        "zotero_key": item_key, "title": title, "year": year, "doi": doi,
                        "paths": {"pdf": str(pdf_path), "html": str(html_path) if html_path else None,
                                  "text": str(final_txt), "sections_json": str(final_json),
                                  "tables_csv": sorted([str(p) for p in (tables_dir.glob(f"{stem}_table_*.csv"))])},
                        "sections": {}, "source_mode": "cached", "source_type": "pdf",
                        "status": "SKIP_EXISTING", "message": "outputs already exist"
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[SKIP existing] {item_key} -> {final_txt.name}")
                    skipped += 1
                    continue

                # run pdf2clean
                tmp_txt  = cleaned_root / f"{stem}.cleaned.txt"
                tmp_json = cleaned_root / f"{stem}.sections.json"
                ok = False
                if _pdf2clean_mod is not None:
                    try:
                        ok, _meta = _pdf2clean_mod.process_pdf(
                            pdf_path=pdf_path, outdir=cleaned_root,
                            keep_sections=["abstract","materials_methods","results","discussion","conclusion"],
                            tables=args.tables, debug_trace=args.debug_trace,
                            debug_skim=args.debug_skim, preview_lines=args.preview_lines
                        )
                    except Exception as e:
                        print(f"[WARN] pdf2clean import call failed for {pdf_path.name}: {e}", file=sys.stderr)
                        ok = False
                if _pdf2clean_mod is None or not ok:
                    ok = run_pdf2clean_cli(sys.executable, _resolved_pdf2clean, pdf_path,
                                           cleaned_root, args.tables, args.debug_trace,
                                           args.debug_skim, args.preview_lines)
                if not ok:
                    rec = {
                        "zotero_key": item_key, "title": title, "year": year, "doi": doi,
                        "paths": {"pdf": str(pdf_path), "html": str(html_path) if html_path else None,
                                  "text": None, "sections_json": None, "tables_csv": []},
                        "sections": {}, "source_mode": "pdf", "source_type": "pdf",
                        "status": "FAIL", "message": "pdf2clean failed"
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[ERR] PDF cleaning failed for {item_key}  file={pdf_path.name}", file=sys.stderr)
                    failed += 1
                    continue

                if tmp_txt.exists():
                    final_txt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(tmp_txt), str(final_txt))
                if tmp_json.exists():
                    final_json.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(tmp_json), str(final_json))
                table_paths = sorted([str(p) for p in tables_dir.glob(f"{stem}_table_*.csv")])
                try:
                    sections_obj = json.loads(final_json.read_text(encoding="utf-8")) if final_json.exists() else {}
                except Exception:
                    sections_obj = {}

                rec = {
                    "zotero_key": item_key, "title": title, "year": year, "doi": doi,
                    "paths": {"pdf": str(pdf_path), "html": str(html_path) if html_path else None,
                              "text": str(final_txt), "sections_json": str(final_json), "tables_csv": table_paths},
                    "sections": sections_obj, "source_mode": "pdf", "source_type": "pdf",
                    "status": "OK_PDF", "message": "cleaned successfully"
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[OK][PDF] {pdf_path.name} -> cleaned + sections")
                processed += 1
                continue

            # nothing found: still record attachments as None to stay explicit
            rec = {
                "zotero_key": item_key, "title": title, "year": year, "doi": doi,
                "paths": {"pdf": str(pdf_path) if pdf_path else None, "html": str(html_path) if html_path else None,
                          "text": None, "sections_json": None, "tables_csv": []},
                "sections": {}, "source_mode": None, "source_type": None,
                "status": "SKIP", "message": "no local html/pdf found"
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[SKIP] No HTML/PDF found for {item_key}  title={title[:80]}")
            skipped += 1

    print(f"\nSUMMARY: processed={processed}, skipped={skipped}, failed={failed}, manifest={manifest_path}")

if __name__ == "__main__":
    main()
