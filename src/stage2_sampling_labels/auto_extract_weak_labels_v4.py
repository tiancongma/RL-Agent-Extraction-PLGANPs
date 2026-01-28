#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_extract_weak_labels_v4.py  —  FULL REPLACEMENT

Purpose
-------
Extract weak labels of PLGA formulations from cleaned text with a
**deterministic characterization fallback** (size/PDI/zeta) when the LLM
returns zero rows. Designed for small-batch, debuggable runs.

Minimal Repro (small batch)
---------------------------
# 1) Ensure cleaned text exists (HTML preferred)
#    python .\scripts\pdf2clean.py --manifest .\data\cleaned\manifest.tsv \
#       --out-dir .\data\cleaned --prefer html --overwrite --verbose
#
# 2) Run weak labels (point to your actual sample + key2txt)
#    python .\scripts\auto_extract_weak_labels_v4.py \
#       --sample-jsonl data\cleaned\samples\sample10.jsonl \
#       --key2txt data\cleaned\samples\key2txt.tsv \
#       --outdir data\cleaned\weak_labels_v4 \
#       --model gemini-2.5-flash-lite \
#       --max-chars 60000 --max-items 10 --sleep 0.3 --verbose

Key Design Changes vs v3
------------------------
1) If LLM returns 0 formulations, run regex fallback to pull size from full text.
2) Characterization rows are **decoupled** from full formulation recognition:
   we will emit a row even if emulsion-type etc. are missing (set to null).
3) Sliding-window chunking with overlap for long texts to reduce truncation misses.
4) Robust key2txt loader (accepts header-less 2-col tsv or with headers).
5) Per-paper diagnostics to explain why a paper yielded 0 or 1+ rows.

Output
------
- Master TSV: {outdir}/weak_labels.tsv
- Optional per-item TSVs when --per-item-dir is set.
- Log lines like: [i/N] KEY -> k formulation(s) ✓/✗

Exit Codes
----------
- SystemExit(…): path errors or argument issues (clear messages).

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import csv
from statistics import mean

# Third-party
from dotenv import load_dotenv

# Optional: only import if available; we handle absence gracefully.
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

###############################################################
# Regex fallback for particle size (and mild table hints)
###############################################################
SIZE_UNIT = r"(?:nm|μm|um|micromet(?:er|re)s?)"
NUM = r"(?:\d{2,5}(?:\.\d+)?)"
PLUSMINUS = r"(?:±|\+/-|\u00B1)"

SIZE_PATTERNS = [
    # mean ± sd  e.g., 152 ± 18 nm
    rf"(?P<label>hydrodynamic diameter|z[-\s]?avg|z[-\s]?average|size|particle size|diameter)[^.\n]{{0,60}}?(?P<val>{NUM})\s*{PLUSMINUS}\s*(?P<sd>{NUM})\s*(?P<Unit>{SIZE_UNIT})",
    # mean (sd)  e.g., 152 (18) nm
    rf"(?P<label>hydrodynamic diameter|z[-\s]?avg|z[-\s]?average|size|particle size|diameter)[^.\n]{{0,60}}?(?P<val>{NUM})\s*\(\s*(?P<sd>{NUM})\s*\)\s*(?P<Unit>{SIZE_UNIT})",
    # range 100–150 nm / 100-150 nm
    rf"(?P<label>hydrodynamic diameter|z[-\s]?avg|z[-\s]?average|size|particle size|diameter)[^.\n]{{0,60}}?(?P<v1>{NUM})\s*[–-]\s*(?P<v2>{NUM})\s*(?P<Unit>{SIZE_UNIT})",
    # simple: size 150 nm
    rf"(?P<label>hydrodynamic diameter|z[-\s]?avg|z[-\s]?average|size|particle size|diameter)\s*(?:=|:)?\s*(?P<val>{NUM})\s*(?P<Unit>{SIZE_UNIT})",
]

TABLE_HEADER_HINTS = [
    r"(?:size|diameter|z[-\s]?avg|hydrodynamic\s+diameter)\s*\(\s*nm\s*\)",
]

def _to_nm(value: float, unit: str) -> float:
    unit = (unit or "nm").lower()
    if unit in ("μm", "um", "micrometer", "micrometers", "micrometre", "micrometres"):
        return value * 1000.0
    return value


def deterministic_size_fallback(txt: str) -> Optional[dict]:
    text = txt or ""
    # First: scan for table header hints and nearby nm numbers
    for th in TABLE_HEADER_HINTS:
        if re.search(th, text, re.IGNORECASE):
            cand = re.findall(rf"\b({NUM})\s*nm\b", text, re.IGNORECASE)
            if cand:
                vals = [float(x) for x in cand[:10]]
                if vals:
                    return {
                        "size_nm": round(mean(vals), 2),
                        "size_sd_nm": None,
                        "size_note": "from table header hint",
                    }
    # Then: paragraph patterns
    for pat in SIZE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        gd = m.groupdict()
        if gd.get("v1") and gd.get("v2"):
            v = (_to_nm(float(gd["v1"]), gd.get("Unit", "nm")) + _to_nm(float(gd["v2"]), gd.get("Unit", "nm"))) / 2.0
            return {
                "size_nm": round(v, 2),
                "size_sd_nm": None,
                "size_note": f"range avg of {gd['v1']}-{gd['v2']} {gd.get('Unit','nm')}",
            }
        if gd.get("val"):
            v = _to_nm(float(gd["val"]), gd.get("Unit", "nm"))
            sd = gd.get("sd")
            return {
                "size_nm": round(v, 2),
                "size_sd_nm": (round(_to_nm(float(sd), gd.get("Unit", "nm")), 2) if sd else None),
                "size_note": gd.get("label", "size"),
            }
    return None

###############################################################
# Data structures & utilities
###############################################################
@dataclasses.dataclass
class Paper:
    key: str
    title: str = ""
    year: str = ""
    doi: str = ""


def die(msg: str):
    raise SystemExit(msg)


def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Bad JSONL line in {path.name}: {e}")
    return out


def infer_papers(sample_jsonl: Path) -> List[Paper]:
    items = load_jsonl(sample_jsonl)
    papers: List[Paper] = []
    for it in items:
        key = it.get("key") or it.get("Key") or it.get("id")
        if not key:
            # try nested structures from earlier versions
            key = it.get("paper_key") or it.get("zotero_key")
        if not key:
            print(f"[WARN] skip line with no key: {it}")
            continue
        papers.append(Paper(key=key, title=it.get("title", ""), year=it.get("year", ""), doi=it.get("doi", "")))
    return papers


def load_key2txt(key2txt: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    with key2txt.open("r", encoding="utf-8") as f:
        sniffer = csv.Sniffer()
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.excel_tab
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        die(f"Empty key2txt: {key2txt}")

    # heuristics: header vs no header
    header_like = rows[0]
    def looks_like_path(s: str) -> bool:
        return ("/" in s or "\\" in s) and (s.endswith(".txt") or s.endswith(".cleaned.txt"))

    start_idx = 0
    if len(header_like) >= 2 and not looks_like_path(header_like[1]):
        # seems a header; start from row 1
        start_idx = 1

    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        k = row[0].strip()
        p = Path(row[1].strip())
        mapping[k] = p
    if not mapping:
        die(f"key2txt.tsv must contain key + path columns. Got header: {rows[0]}")
    return mapping


def read_text_for_key(k: str, key2txt: Dict[str, Path]) -> Tuple[str, Optional[Path]]:
    p = key2txt.get(k)
    if not p:
        return "", None
    if not p.exists():
        return "", p
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = p.read_text(encoding="latin-1", errors="ignore")
    # Try to append figure/table captions from sibling sections.json if present
    try:
        sec = p.with_suffix("").with_suffix(".sections.json")
        if sec.exists():
            js = json.loads(sec.read_text(encoding="utf-8", errors="ignore"))
            caps = []
            for s in js if isinstance(js, list) else []:
                if isinstance(s, dict) and s.get("type") in ("figure_caption", "table", "table_caption"):
                    caps.append(s.get("text", ""))
            if caps:
                txt = txt + "\n\n" + "\n".join(caps)
    except Exception:
        pass
    return txt, p


###############################################################
# LLM interface (Gemini) with simple retries
###############################################################
LLM_PROMPT_TEMPLATE = (
    "You are an expert reading a methods+results section about PLGA nanoparticle "
    "formulations (emulsion or nanoprecipitation, etc.).\n"
    "Extract a JSON array named formulations. Each item has: id, fields, notes.\n"
    "The 'fields' object may contain missing values as null.\n"
    "Do NOT drop characterization-only rows: if you find size/pdi/zeta even when the "
    "emulsion details are missing, still output a row with missing fields as null.\n\n"
    "fields schema (best effort, null if unknown):\n"
    "- emul_type: string (e.g., 'W/O/W', 'O/W', 'nanoprecipitation')\n"
    "- emul_method: string (e.g., 'double emulsion solvent evaporation')\n"
    "- pva_conc_percent: number|null (w/v %)\n"
    "- organic_solvent: string|null (e.g., 'dichloromethane', 'ethyl acetate')\n"
    "- plga_mw_kDa: number|null\n"
    "- la_ga_ratio: string|null (e.g., '50:50')\n"
    "- size_nm: number|null (Z-average or hydrodynamic diameter in nm)\n"
    "- size_sd_nm: number|null\n"
    "- pdi: number|null\n"
    "- zeta_mV: number|null\n"
    "- drug_name: string|null\n"
    "- drug_feed_mg: number|null\n"
    "- dl_percent: number|null (drug loading %)\n"
    "- ee_percent: number|null (encapsulation efficiency %)\n\n"
    "Return ONLY JSON (no prose)."
)


def ensure_genai(model: str):
    if not HAS_GENAI:
        die("google-generativeai is not installed in this environment.")
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        die("GEMINI_API_KEY / GOOGLE_API_KEY is missing in environment.")
    genai.configure(api_key=key)
    try:
        _ = genai.GenerativeModel(model)
    except Exception as e:
        die(f"Failed to initialize Gemini model '{model}': {e}")


def call_gemini(prompt: str, model: str, retries: int = 2, sleep_sec: float = 0.5) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            mdl = genai.GenerativeModel(model)
            resp = mdl.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text
            # Sometimes response has candidates
            try:
                cand = resp.candidates[0].content.parts[0].text
                if cand:
                    return cand
            except Exception:
                pass
            last_err = RuntimeError("Empty response text")
        except Exception as e:
            last_err = e
        time.sleep(sleep_sec)
    raise last_err or RuntimeError("Gemini call failed")


###############################################################
# Chunking / windowing
###############################################################

def chunk_text(txt: str, max_chars: int, window: int = 12000, overlap: int = 1000) -> List[str]:
    if not txt:
        return []
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars]
    if len(txt) <= window:
        return [txt]
    chunks: List[str] = []
    i = 0
    while i < len(txt):
        chunk = txt[i:i+window]
        chunks.append(chunk)
        i += max(1, (window - overlap))
        if len(chunks) >= 10:  # safety bound
            break
    return chunks


###############################################################
# Postprocessing helpers
###############################################################

def safe_json_parse(s: str) -> List[dict]:
    s = s.strip()
    # If model wrapped JSON in code fences or prose, try to extract first [ ... ]
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "formulations" in data:
            f = data["formulations"]
            return f if isinstance(f, list) else []
    except Exception:
        pass
    return []


def merge_llm_rows(rows: List[dict]) -> List[dict]:
    """Light sanity pass ensuring required structure and numeric coercions."""
    out = []
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict):
            continue
        fields = r.get("fields") or {}
        if not isinstance(fields, dict):
            fields = {}
        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None
        # normalize expected numeric fields
        for numk in ("pva_conc_percent", "plga_mw_kDa", "size_nm", "size_sd_nm", "pdi", "zeta_mV", "drug_feed_mg", "dl_percent", "ee_percent"):
            if numk in fields and fields[numk] is not None:
                fields[numk] = to_float(fields[numk])
        out.append({
            "id": r.get("id", i),
            "fields": fields,
            "notes": r.get("notes")
        })
    return out


def write_tsv(path: Path, rows: List[dict], header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            fields = r.get("fields", {})
            w.writerow([
                fields.get("emul_type"),
                fields.get("emul_method"),
                fields.get("pva_conc_percent"),
                fields.get("organic_solvent"),
                fields.get("plga_mw_kDa"),
                fields.get("la_ga_ratio"),
                fields.get("size_nm"),
                fields.get("size_sd_nm"),
                fields.get("pdi"),
                fields.get("zeta_mV"),
                fields.get("drug_name"),
                fields.get("drug_feed_mg"),
                fields.get("dl_percent"),
                fields.get("ee_percent"),
                r.get("notes"),
            ])


MASTER_HEADER = [
    "emul_type","emul_method","pva_conc_percent","organic_solvent","plga_mw_kDa","la_ga_ratio",
    "size_nm","size_sd_nm","pdi","zeta_mV","drug_name","drug_feed_mg","dl_percent","ee_percent","notes"
]


###############################################################
# Main
###############################################################

def main():
    ap = argparse.ArgumentParser(description="Weak label extractor with deterministic size fallback (v4)")
    ap.add_argument("--sample-jsonl", dest="sample_jsonl", default="data/cleaned/samples/sample10.jsonl")
    ap.add_argument("--key2txt", dest="key2txt", default="data/cleaned/samples/key2txt.tsv")
    ap.add_argument("--outdir", dest="outdir", default="data/cleaned/weak_labels_v4")
    ap.add_argument("--per-item-dir", dest="per_item_dir", action="store_true")
    ap.add_argument("--model", dest="model", default="gemini-2.5-flash-lite")
    ap.add_argument("--max-chars", dest="max_chars", type=int, default=60000)
    ap.add_argument("--sleep", dest="sleep", type=float, default=0.3)
    ap.add_argument("--max-items", dest="max_items", type=int, default=10)
    ap.add_argument("--verbose", dest="verbose", action="store_true")
    ap.add_argument("--overwrite", dest="overwrite", action="store_true")

    args = ap.parse_args()

    sample = Path(args.sample_jsonl)
    if not sample.exists():
        die(f"Sample JSONL not found: {sample}")
    k2t = Path(args.key2txt)
    if not k2t.exists():
        die(f"key2txt not found: {k2t}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    master_tsv = outdir / "weak_labels.tsv"
    if master_tsv.exists() and not args.overwrite:
        print(f"[INFO] Overwriting disabled; removing existing {master_tsv} to append new run")
        master_tsv.unlink(missing_ok=True)

    papers = infer_papers(sample)
    keymap = load_key2txt(k2t)

    if HAS_GENAI:
        ensure_genai(args.model)

    total = min(len(papers), args.max_items) if args.max_items > 0 else len(papers)
    total_rows = 0
    parsed = 0
    failed = 0

    # Prepare master rows list and write at the end
    master_rows: List[dict] = []

    for idx, paper in enumerate(papers[:total], 1):
        txt, tp = read_text_for_key(paper.key, keymap)
        if not txt:
            print(f"[{idx}/{total}] {paper.key} -> 0 formulation(s) ✗   (no text)")
            failed += 1
            continue

        chunks = chunk_text(txt, max_chars=args.max_chars, window=12000, overlap=1000)
        paper_rows: List[dict] = []
        llm_raw = None
        # Call LLM on concatenated limited context (first 1-2 windows) to limit token
        joined_for_prompt = "\n\n".join(chunks[:2]) if chunks else txt[:args.max_chars]
        prompt = LLM_PROMPT_TEMPLATE + "\n\nTEXT:\n" + joined_for_prompt

        if HAS_GENAI:
            try:
                if args.verbose:
                    chlen = len(joined_for_prompt)
                    print(f"[CALL] {paper.key} model={args.model} chars={chlen}")
                llm_raw = call_gemini(prompt, args.model, retries=2, sleep_sec=args.sleep)
                llm_rows = safe_json_parse(llm_raw)
                paper_rows = merge_llm_rows(llm_rows)
            except Exception as e:
                if args.verbose:
                    print(f"[WARN] LLM failed for {paper.key}: {e}")
        else:
            if args.verbose:
                print(f"[WARN] google-generativeai not available; skipping LLM for {paper.key}")

        if not paper_rows:
            # Deterministic fallback over full (possibly longer) text
            fb = deterministic_size_fallback(txt)
            if fb:
                paper_rows.append({
                    "id": 1,
                    "fields": {
                        "emul_type": None,
                        "emul_method": None,
                        "pva_conc_percent": None,
                        "organic_solvent": None,
                        "plga_mw_kDa": None,
                        "la_ga_ratio": None,
                        "size_nm": fb["size_nm"],
                        "size_sd_nm": fb.get("size_sd_nm"),
                        "pdi": None,
                        "zeta_mV": None,
                        "drug_name": None,
                        "drug_feed_mg": None,
                        "dl_percent": None,
                        "ee_percent": None,
                    },
                    "notes": fb.get("size_note"),
                })
                diag = "llm_empty_but_regex_hit"
            else:
                diag = "no_size_match_regex"
            if args.verbose:
                print(f"[MISS] {paper.key} {diag}")

        if paper_rows:
            parsed += 1
            total_rows += len(paper_rows)
            print(f"[{idx}/{total}] {paper.key} -> {len(paper_rows)} formulation(s) ✓   (total_rows={total_rows})")
            # per-item write
            if args.per_item_dir:
                perdir = outdir / paper.key
                perdir.mkdir(parents=True, exist_ok=True)
                write_tsv(perdir / f"{paper.key}.tsv", paper_rows, MASTER_HEADER)
            # collect for master
            master_rows.extend(paper_rows)
        else:
            failed += 1
            print(f"[{idx}/{total}] {paper.key} -> 0 formulation(s) ✗")

        time.sleep(args.sleep)

    # write master at end
    if master_rows:
        write_tsv(master_tsv, master_rows, MASTER_HEADER)

    print(f"\n[SUMMARY v4] papers_in={total}, papers_parsed={parsed}, tsv_rows={total_rows}, failed={failed}")


if __name__ == "__main__":
    main()
