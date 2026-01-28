#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sample_from_manifest_html_first.py (robust text-path recovery)

What’s new vs your previous version:
- If rec.paths.text is missing but paths.html/pdf exists, try to recover text path
  at data/cleaned/text/{stem}.cleaned.txt (and only keep when it exists).
- Only admit records with a real text file into the pool, to avoid SKIP storms later.
"""

import argparse, json, random, csv
from pathlib import Path

HTML_EXTS = (".html", ".htm", ".xhtml")

def infer_source_type(rec, text_path: str) -> str:
    st = (rec.get("source_type") or "").strip().lower()
    if st in ("html","pdf"): return st
    paths = rec.get("paths", {}) or {}
    if any(k in paths and paths[k] for k in ("html","html_text")): return "html"
    if any(k in paths and paths[k] for k in ("pdf","pdf_text")):   return "pdf"
    tp = (text_path or "").lower()
    if tp.endswith(HTML_EXTS): return "html"
    return "pdf"

def try_recover_text_path(rec: dict, cleaned_root: Path) -> str:
    """Attempt to recover text path from html/pdf stem if paths.text is missing."""
    paths = rec.get("paths", {}) or {}
    cand = None
    for k in ("html","pdf","html_text","pdf_text"):
        p = paths.get(k)
        if p and Path(p).exists():
            cand = Path(p); break
    if not cand: return ""
    stem = cand.stem
    text_dir = cleaned_root / "text"
    guess = text_dir / f"{stem}.cleaned.txt"
    return str(guess) if guess.exists() else ""

def load_pool(manifest_path: Path):
    cleaned_root = manifest_path.parents[1] if manifest_path.name.endswith(".jsonl") else Path("data")/"cleaned"
    pool_html, pool_pdf = [], []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            status = str(rec.get("status", ""))
            if not (status.startswith("OK_") or status == "OK" or status.startswith("SKIP_EXISTING")):
                # 允许 SKIP_EXISTING（缓存命中）的记录进入池
                continue

            key = rec.get("zotero_key") or rec.get("key")
            text_path = (rec.get("paths", {}) or {}).get("text") or ""
            if not text_path:
                # 尝试补推 text_path
                text_path = try_recover_text_path(rec, cleaned_root)
                if text_path:
                    rec.setdefault("paths", {})["text"] = text_path  # 回填到记录，写回样本文件

            if not key or not text_path:
                continue
            if not Path(text_path).exists():
                continue

            st = infer_source_type(rec, text_path)
            item = {
                "key": key,
                "title": rec.get("title", ""),
                "year": rec.get("year", ""),
                "doi": rec.get("doi", ""),
                "text_path": text_path,
                "source_type": st,
                "parse_quality": rec.get("parse_quality", ""),
            }
            if st == "html":
                pool_html.append(item)
            else:
                pool_pdf.append(item)
    return pool_html, pool_pdf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/cleaned/manifests/zotero_llm_relevant.jsonl",
                    help="Path to the main JSONL manifest.")
    ap.add_argument("--outdir", default="data/cleaned/samples",
                    help="Directory to store outputs.")
    ap.add_argument("--n", type=int, default=10,
                    help="Number of items to sample (default: 10).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    ap.add_argument("--ground-truth", default=None,
                    help="Optional JSONL with records like {'key':..., 'fields':{...}} to subset.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pool_html, pool_pdf = load_pool(manifest_path)
    if not (pool_html or pool_pdf):
        raise SystemExit("No eligible entries with existing text files found in manifest.")

    random.seed(args.seed)
    random.shuffle(pool_html); random.shuffle(pool_pdf)

    need = args.n; picked = []
    take_html = min(need, len(pool_html))
    if take_html > 0:
        picked.extend(pool_html[:take_html]); need -= take_html
    if need > 0 and len(pool_pdf) > 0:
        take_pdf = min(need, len(pool_pdf))
        picked.extend(pool_pdf[:take_pdf]); need -= take_pdf
    if need > 0:
        print(f"[WARN] Not enough items to reach N={args.n}. Available={args.n - need} total.")

    base = f"sample{args.n}_htmlfirst"
    sample_jsonl = outdir / f"{base}.jsonl"
    key2txt_path = outdir / "key2txt.tsv"
    sample_tsv   = outdir / f"{base}.tsv"

    # mini manifest JSONL
    with sample_jsonl.open("w", encoding="utf-8") as w:
        for r in picked:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    # key -> txt
    with key2txt_path.open("w", encoding="utf-8", newline="") as w:
        writer = csv.writer(w, delimiter="\t", lineterminator="\n")
        for r in picked:
            writer.writerow([r["key"], r["text_path"]])

    # readable TSV
    with sample_tsv.open("w", encoding="utf-8", newline="") as w:
        writer = csv.writer(w, delimiter="\t", lineterminator="\n")
        writer.writerow(["key", "source_type", "year", "doi", "title", "text_path", "parse_quality"])
        for r in picked:
            writer.writerow([r["key"], r["source_type"], r["year"], r["doi"], r["title"], r["text_path"], r["parse_quality"]])

    h = sum(1 for r in picked if r["source_type"] == "html")
    p = sum(1 for r in picked if r["source_type"] == "pdf")
    print(f"[OK] sample -> {sample_jsonl}")
    print(f"[OK] key2txt -> {key2txt_path}")
    print(f"[OK] sample_tsv -> {sample_tsv}")
    print(f"[INFO] Sample size: {len(picked)}  (html={h}, pdf={p})")

if __name__ == "__main__":
    main()
