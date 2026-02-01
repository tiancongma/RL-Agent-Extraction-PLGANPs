#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_extract_weak_labels.py  (v5 evidence-aware, incremental write + fail-safe)

Goal
- Run multi-model weak-label extraction on a sample of papers.
- Prefer sectioned text (materials_methods/results/abstract) when available.
- Emit a flattened TSV compatible with multi_model_merge_qc.py, plus a minimal
  "Auditable Evidence Contract" at the row level:
    evidence_section, evidence_span_text, evidence_span_start, evidence_span_end,
    evidence_method, evidence_quality

Notes
- This is intentionally conservative: evidence is row-level, not per-field.
- Offsets (start/end) are within the chosen section text (not global fulltext).

Paths
- This script does NOT hard-code repo paths. Provide CLI args, or (optionally)
  have your repo's src/utils/paths.py expose a helper we can call.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Optional Gemini dependency
HAS_GENAI = False
try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False


# -----------------------------
# Optional paths.py integration
# -----------------------------

def try_load_paths_defaults() -> Dict[str, str]:
    """
    Best-effort attempt to obtain default paths from src/utils/paths.py.

    We do NOT assume an interface. We try a few common patterns:
      - get_paths().<attr> or dict
      - Paths().<attr> or dict
      - DEFAULTS dict

    If nothing works, return {} and rely on CLI args.
    """
    try:
        import importlib
        mod = importlib.import_module("src.utils.paths")
    except Exception:
        return {}

    # 1) DEFAULTS dict
    if hasattr(mod, "DEFAULTS") and isinstance(getattr(mod, "DEFAULTS"), dict):
        d = dict(getattr(mod, "DEFAULTS"))
        return {k: str(v) for k, v in d.items()}

    # 2) get_paths()
    if hasattr(mod, "get_paths"):
        try:
            obj = mod.get_paths()
            return _paths_obj_to_dict(obj)
        except Exception:
            pass

    # 3) Paths class
    if hasattr(mod, "Paths"):
        try:
            obj = mod.Paths()
            return _paths_obj_to_dict(obj)
        except Exception:
            pass

    return {}


def _paths_obj_to_dict(obj: Any) -> Dict[str, str]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: str(v) for k, v in obj.items()}
    out: Dict[str, str] = {}
    for name in (
        "sample_jsonl",
        "key2txt_tsv",
        "sections_dir",
        "out_jsonl",
        "out_tsv",
    ):
        if hasattr(obj, name):
            out[name] = str(getattr(obj, name))
    return out


# -----------------------------
# Data loading
# -----------------------------

@dataclass
class Paper:
    key: str


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def read_jsonl_keys(sample_jsonl: Path) -> List[Paper]:
    papers: List[Paper] = []
    with sample_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("zotero_key") or obj.get("key") or obj.get("Key") or obj.get("id")
            if key:
                papers.append(Paper(key=str(key)))
    if not papers:
        die(f"No keys found in sample JSONL: {sample_jsonl}")
    return papers


def load_key2txt(key2txt_tsv: Path) -> Dict[str, str]:
    """
    Accepts either:
      - headerless: key<TAB>path
      - headered: key, text_path (or similar)
    """
    df = pd.read_csv(key2txt_tsv, sep="\t", dtype=str, header=None)
    if df.shape[1] >= 2:
        # Could be headerless; detect by whether first row looks like a path
        first0 = str(df.iloc[0, 0])
        first1 = str(df.iloc[0, 1])
        if first0.lower() in ("key", "zotero_key") or "path" in first1.lower():
            # Actually headered; reload with header
            df2 = pd.read_csv(key2txt_tsv, sep="\t", dtype=str)
            # try common column names
            kcol = None
            pcol = None
            for cand in ("key", "zotero_key", "Key"):
                if cand in df2.columns:
                    kcol = cand
                    break
            for cand in ("text_path", "path", "txt_path", "clean_text_path"):
                if cand in df2.columns:
                    pcol = cand
                    break
            if not kcol or not pcol:
                die(f"key2txt missing expected columns: {key2txt_tsv}")
            return {str(k): str(p) for k, p in zip(df2[kcol], df2[pcol]) if pd.notna(k) and pd.notna(p)}
        else:
            # headerless
            return {str(k): str(p) for k, p in zip(df.iloc[:, 0], df.iloc[:, 1]) if pd.notna(k) and pd.notna(p)}
    die(f"Invalid key2txt TSV (need >=2 columns): {key2txt_tsv}")
    return {}


def read_text_for_key(key: str, key2txt: Dict[str, str]) -> Tuple[str, Optional[Path]]:
    p = key2txt.get(key)
    if not p:
        return "", None
    path = Path(p)
    if not path.exists():
        return "", path
    try:
        return path.read_text(encoding="utf-8", errors="ignore"), path
    except Exception:
        return "", path


# -----------------------------
# Sections handling (evidence source)
# -----------------------------

def load_sections(sections_path: Path) -> List[Dict[str, str]]:
    """
    Supports two schemas:
    1) dict mapping section_name -> text
       (like your examples: abstract/materials_methods/conclusion)
    2) list of objects with keys like name/type/section_name and text/content
    Returns list of {section_name, text}.
    """
    try:
        data = json.loads(sections_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []

    out: List[Dict[str, str]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str) and v.strip():
                out.append({"section_name": str(k), "text": v})
    elif isinstance(data, list):
        for it in data:
            if not isinstance(it, dict):
                continue
            name = it.get("section_name") or it.get("name") or it.get("type") or it.get("section") or ""
            txt = it.get("text") or it.get("content") or ""
            if isinstance(txt, str) and txt.strip():
                out.append({"section_name": str(name), "text": txt})
    return out


def build_sectioned_prompt_text(sections: List[Dict[str, str]], max_chars: int) -> str:
    """
    Join a prioritized subset of sections with explicit headers.
    """
    if not sections:
        return ""

    # Prioritize methods/results, then abstract, then others
    pri = {
        "materials_methods": 0,
        "materials and methods": 0,
        "methods": 0,
        "methodology": 0,
        "results": 1,
        "results_discussion": 1,
        "discussion": 2,
        "conclusion": 3,
        "abstract": 4,
    }

    def rank(name: str) -> Tuple[int, int]:
        n = (name or "").strip().lower()
        return (pri.get(n, 10), len(n))

    sections_sorted = sorted(sections, key=lambda s: rank(s.get("section_name", "")))
    parts: List[str] = []
    total = 0
    for s in sections_sorted:
        name = s.get("section_name", "").strip() or "section"
        txt = s.get("text", "")
        if not txt:
            continue
        block = f"\n\n### {name}\n{txt}"
        if max_chars and total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
        if max_chars and total >= max_chars:
            break
    return "".join(parts).strip()


# -----------------------------
# Evidence extraction (minimal contract)
# -----------------------------

_EVIDENCE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("size", re.compile(r"(size|diameter|particle\s+size|mean\s+diameter).{0,80}?(\d{2,4}(\.\d+)?)\s*(nm)", re.IGNORECASE)),
    ("pdi", re.compile(r"\b(PDI|polydispersity\s+index)\b.{0,60}?(\d\.\d+|\d{1,2}\.\d+)", re.IGNORECASE)),
    ("zeta", re.compile(r"(zeta\s+potential|ζ).{0,60}?(-?\d{1,3}(\.\d+)?)\s*(mV)", re.IGNORECASE)),
    ("pva", re.compile(r"\bPVA\b.{0,60}?(\d+(\.\d+)?)\s*%(\s*w/v|\s*\(w/v\)|\s*w\/v)?", re.IGNORECASE)),
    ("solvent", re.compile(r"\b(dichloromethane|methylene chloride|DCM|ethyl acetate|EA|chloroform|acetone)\b", re.IGNORECASE)),
    ("emulsion", re.compile(r"\b(W1\/O\/W2|W\/O\/W|O\/W|W\/O|double\s+emulsion|single\s+emulsion|solvent\s+evaporation)\b", re.IGNORECASE)),
]


def choose_canonical_evidence(sections: List[Dict[str, str]], prefer: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Pick one canonical evidence span from sections, using simple pattern hits.

    Returns dict with:
      evidence_section, evidence_span_text, evidence_span_start, evidence_span_end,
      evidence_method, evidence_quality
    """
    if not sections:
        return {
            "evidence_section": "",
            "evidence_span_text": "",
            "evidence_span_start": "",
            "evidence_span_end": "",
            "evidence_method": "",
            "evidence_quality": "D",
        }

    prefer = prefer or ["materials_methods", "methods", "results", "abstract"]
    prefer_rank = {name.lower(): i for i, name in enumerate(prefer)}

    best = None  # (score, pref_rank, section_name, start, end, span, method, quality)
    for sec in sections:
        sec_name = (sec.get("section_name") or "").strip()
        sec_key = sec_name.lower()
        txt = sec.get("text") or ""
        if not txt:
            continue

        pr = prefer_rank.get(sec_key, 99)

        # collect candidate hits
        hits: List[Tuple[int, int, str]] = []
        for tag, pat in _EVIDENCE_PATTERNS:
            m = pat.search(txt)
            if m:
                hits.append((m.start(), m.end(), tag))

        if not hits:
            continue

        # define a window around the earliest hit, but expand to include nearby hits
        hits.sort(key=lambda x: x[0])
        s0 = hits[0][0]
        e0 = hits[0][1]
        # merge hits within 300 chars
        s = s0
        e = e0
        for hs, he, _ in hits[1:]:
            if hs - e <= 300:
                e = max(e, he)
            else:
                break

        # expand window for readability
        win_left = max(0, s - 180)
        win_right = min(len(txt), e + 220)
        span = txt[win_left:win_right].strip()

        # scoring: #distinct tags in window + numeric-with-unit presence
        tags = set([t for _, _, t in hits])
        score = len(tags)

        has_unit = bool(re.search(r"\b(nm|mV|%(\s*w/v|\s*w\/v)?)\b", span, re.IGNORECASE))
        has_number = bool(re.search(r"\d", span))
        has_keyword = bool(re.search(r"\b(size|PDI|zeta|PVA|emulsion|solvent)\b", span, re.IGNORECASE))

        if has_unit and has_keyword:
            quality = "A"
        elif has_number:
            quality = "B"
        elif has_keyword:
            quality = "C"
        else:
            quality = "D"

        cand = (score, -pr, sec_name, win_left, win_right, span, "pattern_window", quality)
        if best is None or cand > best:
            best = cand

    if best is None:
        return {
            "evidence_section": "",
            "evidence_span_text": "",
            "evidence_span_start": "",
            "evidence_span_end": "",
            "evidence_method": "",
            "evidence_quality": "D",
        }

    _, _, sec_name, st, en, span, method, quality = best
    return {
        "evidence_section": sec_name,
        "evidence_span_text": span,
        "evidence_span_start": int(st),
        "evidence_span_end": int(en),
        "evidence_method": method,
        "evidence_quality": quality,
    }


# -----------------------------
# Deterministic fallback (minimal)
# -----------------------------

def deterministic_fallback_from_text(txt: str) -> Dict[str, Any]:
    """
    Very small fallback: try to find one size/PDI/zeta.
    Returns a partial fields dict.
    """
    out: Dict[str, Any] = {}
    m_size = re.search(r"(\d{2,4}(\.\d+)?)\s*nm", txt, re.IGNORECASE)
    if m_size:
        out["size_nm"] = float(m_size.group(1))
    m_pdi = re.search(r"\bPDI\b.{0,40}?(\d\.\d+|\d{1,2}\.\d+)", txt, re.IGNORECASE)
    if m_pdi:
        out["pdi"] = float(m_pdi.group(1))
    m_z = re.search(r"(zeta\s+potential|ζ).{0,40}?(-?\d{1,3}(\.\d+)?)\s*mV", txt, re.IGNORECASE)
    if m_z:
        out["zeta_mV"] = float(m_z.group(2))
    return out


# -----------------------------
# LLM extraction (Gemini)
# -----------------------------

LLM_PROMPT_TEMPLATE = (
    "You are an expert extracting PLGA nanoparticle formulation data.\n"
    "Return ONLY valid JSON with keys: formulations (array), paper_notes (string|null).\n"
    "Each formulation item has: id (string or int), fields (object), notes (string|null).\n"
    "Do not hallucinate. Use null for unknown.\n"
    "Do NOT drop characterization-only rows: if you find size/PDI/zeta but lack method details,\n"
    "still output a formulation row with missing fields as null.\n\n"
    "fields schema (best effort, null if unknown):\n"
    "- emul_type: string|null (e.g., 'W/O/W', 'O/W', 'nanoprecipitation')\n"
    "- emul_method: string|null\n"
    "- la_ga_ratio: string|null (e.g., '50:50')\n"
    "- plga_mw_kDa: number|null\n"
    "- plga_mass_mg: number|null\n"
    "- pva_conc_percent: number|null (w/v %)\n"
    "- organic_solvent: string|null\n"
    "- drug_name: string|null\n"
    "- drug_feed_amount_text: string|null (keep raw text, include units)\n"
    "- size_nm: number|null\n"
    "- pdi: number|null\n"
    "- zeta_mV: number|null\n"
    "- encapsulation_efficiency_percent: number|null\n"
    "- loading_content_percent: number|null\n\n"
)


def ensure_genai(model: str) -> None:
    if not HAS_GENAI:
        die("google-generativeai is not installed in this environment.")
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        die("GEMINI_API_KEY / GOOGLE_API_KEY is missing in environment.")
    genai.configure(api_key=key)
    _ = genai.GenerativeModel(model)


def call_gemini(model: str, prompt: str, retries: int, sleep_sec: float) -> str:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            mdl = genai.GenerativeModel(model)
            resp = mdl.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text
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


def safe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # try to salvage first {...}
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            return json.loads(m.group(0))
    return {"formulations": [], "paper_notes": None}


def merge_llm_formulations(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    formulations = data.get("formulations") or []
    paper_notes = data.get("paper_notes")
    out: List[Dict[str, Any]] = []
    if isinstance(formulations, list):
        for fobj in formulations:
            if not isinstance(fobj, dict):
                continue
            fid = fobj.get("id")
            fields = fobj.get("fields") or {}
            if not isinstance(fields, dict):
                fields = {}
            notes = fobj.get("notes")
            out.append({"formulation_id": fid, "fields": fields, "notes": notes, "paper_notes": paper_notes})
    return out, paper_notes


# -----------------------------
# Main
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    defaults = try_load_paths_defaults()

    p = argparse.ArgumentParser(description="Multi-model weak label extractor (evidence-aware)")
    p.add_argument("--sample-jsonl", default=defaults.get("sample_jsonl", None), required=False)
    p.add_argument("--key2txt", default=defaults.get("key2txt_tsv", None), required=False)
    p.add_argument("--sections-dir", default=defaults.get("sections_dir", None), required=False,
                   help="Directory containing <key>.sections.json files (optional).")
    p.add_argument("--out-jsonl", default=defaults.get("out_jsonl", None), required=False,
                   help="Write per-(key,model) raw JSONL output (optional).")
    p.add_argument("--out-tsv", default=defaults.get("out_tsv", None), required=False,
                   help="Write flattened TSV output (required unless you only want JSONL).")

    p.add_argument("--models", default="gemini-2.5-flash,gemma-3-12b-it",
                   help="Comma-separated model names.")
    p.add_argument("--max-chars", type=int, default=30000)
    p.add_argument("--max-items", type=int, default=0, help="0 means all in sample.")
    p.add_argument("--sleep", type=float, default=1)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if not args.sample_jsonl or not args.key2txt:
        die("Must provide --sample-jsonl and --key2txt (or configure src/utils/paths.py defaults).")

    sample_jsonl = Path(args.sample_jsonl)
    key2txt_tsv = Path(args.key2txt)
    if not sample_jsonl.exists():
        die(f"Sample JSONL not found: {sample_jsonl}")
    if not key2txt_tsv.exists():
        die(f"key2txt TSV not found: {key2txt_tsv}")

    sections_dir = Path(args.sections_dir) if args.sections_dir else None
    if sections_dir and not sections_dir.exists():
        die(f"sections-dir does not exist: {sections_dir}")

    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else None
    out_tsv = Path(args.out_tsv) if args.out_tsv else None
    if not out_tsv and not out_jsonl:
        die("Provide --out-tsv and/or --out-jsonl.")

    model_names = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if len(model_names) < 1:
        die("No models specified.")
    if HAS_GENAI:
        # init first model to validate API key
        ensure_genai(model_names[0])
    else:
        die("google-generativeai is required for this script (install it in your env).")

    papers = read_jsonl_keys(sample_jsonl)
    key2txt = load_key2txt(key2txt_tsv)

    n_total = len(papers) if args.max_items <= 0 else min(len(papers), args.max_items)

        # -----------------------------------------------------------------
    # Outputs: write incrementally so partial success is preserved
    # -----------------------------------------------------------------
    out_tsv_f = None
    tsv_writer = None

    # fixed column order for TSV (keep stable across runs)
    TSV_FIELDS = [
        "key","model","formulation_id",
        "emul_type","emul_method","la_ga_ratio","plga_mw_kDa","plga_mass_mg",
        "pva_conc_percent","organic_solvent",
        "drug_name","drug_feed_amount_text",
        "size_nm","pdi","zeta_mV",
        "encapsulation_efficiency_percent","loading_content_percent",
        "notes",
        "evidence_section","evidence_span_text","evidence_span_start","evidence_span_end",
        "evidence_method","evidence_quality",
    ]

    raw_out_f = None
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        raw_out_f = out_jsonl.open("w", encoding="utf-8")

    if out_tsv:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        # Always overwrite: this script is intended to be re-runnable
        out_tsv_f = out_tsv.open("w", encoding="utf-8", newline="")
        import csv
        tsv_writer = csv.DictWriter(out_tsv_f, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore")
        tsv_writer.writeheader()
        out_tsv_f.flush()

    try:
        for i, paper in enumerate(papers[:n_total], 1):
            key = paper.key
            fulltxt, _ = read_text_for_key(key, key2txt)
            if not fulltxt:
                print(f"[{i}/{n_total}] {key} -> SKIP (no text)")
                continue

            sections: List[Dict[str, str]] = []
            if sections_dir:
                sp = sections_dir / f"{key}.sections.json"
                if sp.exists():
                    sections = load_sections(sp)

            prompt_text = build_sectioned_prompt_text(sections, max_chars=args.max_chars) if sections else fulltxt[:args.max_chars]
            evidence = choose_canonical_evidence(sections if sections else [{"section_name": "fulltext", "text": fulltxt[:args.max_chars]}])

            for model in model_names:
                prompt = LLM_PROMPT_TEMPLATE + "\nTEXT:\n" + prompt_text
                if args.verbose:
                    print(f"[CALL] {key} model={model} chars={len(prompt_text)}")

                # --- LLM call (fail-safe) ---
                try:
                    raw = call_gemini(model, prompt, retries=args.retries, sleep_sec=args.sleep)
                except Exception as e:
                    print(f"[WARN] {key} model={model} -> CALL FAILED ({type(e).__name__}: {e})")
                    continue

                # --- JSON parse (fail-safe) ---
                try:
                    data = safe_json_load(raw)
                except Exception as e:
                    print(f"[WARN] {key} model={model} -> JSON PARSE FAILED ({type(e).__name__}: {e})")
                    # best-effort dump raw for debugging (optional)
                    try:
                        from src.utils import paths as _paths
                        dbg_dir = _paths.DATA_CLEANED_DEBUG_DIR / "llm_raw"
                        dbg_dir.mkdir(parents=True, exist_ok=True)
                        (dbg_dir / f"{key}__{model}.txt").write_text(raw or "", encoding="utf-8", errors="replace")
                    except Exception:
                        pass
                    continue

                formulations, paper_notes = merge_llm_formulations(data)

                # deterministic fallback if empty
                if not formulations:
                    fb_fields = deterministic_fallback_from_text(prompt_text)
                    if fb_fields:
                        formulations = [{
                            "formulation_id": 1,
                            "fields": fb_fields,
                            "notes": "llm_empty_fallback_regex",
                            "paper_notes": paper_notes,
                        }]

                # write raw jsonl (one record per key/model)
                if raw_out_f is not None:
                    raw_rec = {
                        "key": key,
                        "model": model,
                        "formulations": [
                            {"id": f.get("formulation_id"), "fields": f.get("fields", {}), "notes": f.get("notes")}
                            for f in formulations
                        ],
                        "paper_notes": paper_notes,
                    }
                    raw_out_f.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
                    raw_out_f.flush()

                # flatten and write immediately
                for f in formulations:
                    fid = f.get("formulation_id")
                    fields = f.get("fields") or {}
                    if not isinstance(fields, dict):
                        fields = {}

                    row = {
                        "key": key,
                        "model": model,
                        "formulation_id": fid,
                        "emul_type": fields.get("emul_type"),
                        "emul_method": fields.get("emul_method"),
                        "la_ga_ratio": fields.get("la_ga_ratio"),
                        "plga_mw_kDa": fields.get("plga_mw_kDa"),
                        "plga_mass_mg": fields.get("plga_mass_mg"),
                        "pva_conc_percent": fields.get("pva_conc_percent"),
                        "organic_solvent": fields.get("organic_solvent"),
                        "drug_name": fields.get("drug_name"),
                        "drug_feed_amount_text": fields.get("drug_feed_amount_text"),
                        "size_nm": fields.get("size_nm"),
                        "pdi": fields.get("pdi"),
                        "zeta_mV": fields.get("zeta_mV"),
                        "encapsulation_efficiency_percent": fields.get("encapsulation_efficiency_percent"),
                        "loading_content_percent": fields.get("loading_content_percent"),
                        "notes": f.get("notes"),
                        # evidence fields
                        **evidence,
                    }

                    if tsv_writer is not None:
                        tsv_writer.writerow(row)

                if out_tsv_f is not None:
                    out_tsv_f.flush()

                print(f"[{i}/{n_total}] {key} model={model} -> {len(formulations)} row(s)")

                time.sleep(args.sleep)

    finally:
        if raw_out_f is not None:
            raw_out_f.close()
            print(f"[OK] raw JSONL -> {out_jsonl}")
        if out_tsv_f is not None:
            out_tsv_f.close()
            print(f"[OK] flattened TSV -> {out_tsv}")

    print("[DONE]")


if __name__ == "__main__":
    main(sys.argv[1:])
