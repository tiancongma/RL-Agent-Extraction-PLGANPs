#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_extract_weak_labels_v5_split_single.py

What this adds vs your v5:
1) Safe single-model runs WITHOUT editing code:
   - Use --models <one_model> (or --model <one_model>).

2) Still supports multi-model extraction, but can optionally split outputs per model:
   - --split-by-model writes separate TSV/JSONL files per model while keeping the exact
     same row schema (including the "model" column) so downstream merge/QC logic stays unchanged.

3) Hard quota handling (stop immediately, no retries):
   - If the API error looks like a hard quota / rate-limit (RPD, per-minute token cap, 429, RESOURCE_EXHAUSTED),
     the script aborts immediately and preserves partial outputs.
   - This aligns with your "no meaningless retries" rule.

Notes:
- Evidence fields remain row-level (same as v5): evidence_section, evidence_span_text, evidence_span_start/end,
  evidence_method, evidence_quality.
- Offsets (start/end) are within the chosen section text.
"""

from __future__ import annotations

import argparse
import csv
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
        # Could be headerless; detect by whether first row looks like a header
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

        hits: List[Tuple[int, int, str]] = []
        for tag, pat in _EVIDENCE_PATTERNS:
            m = pat.search(txt)
            if m:
                hits.append((m.start(), m.end(), tag))

        if not hits:
            continue

        hits.sort(key=lambda x: x[0])
        s0 = hits[0][0]
        e0 = hits[0][1]

        s = s0
        e = e0
        for hs, he, _ in hits[1:]:
            if hs - e <= 300:
                e = max(e, he)
            else:
                break

        win_left = max(0, s - 180)
        win_right = min(len(txt), e + 220)
        span = txt[win_left:win_right].strip()

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


class HardQuotaError(RuntimeError):
    """Raised when we detect a hard quota/rate-limit condition and must stop immediately."""


def _looks_like_hard_quota_error(e: Exception) -> bool:
    """
    Heuristic detection for hard quota / rate limit / resource exhausted.
    We intentionally keep this broad. False positives are acceptable because your policy prefers
    "stop early" over "wasteful retries".
    """
    msg = f"{type(e).__name__}: {e}".lower()
    needles = [
        "429",
        "resource_exhausted",
        "rate limit",
        "ratelimit",
        "quota",
        "exceeded",
        "too many requests",
        "rpd",
        "requests per day",
        "tpm",
        "tokens per minute",
        "per-minute",
        "per minute",
    ]
    return any(n in msg for n in needles)


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
    """
    Policy:
    - If error looks like hard quota, raise HardQuotaError immediately (no retry).
    - Otherwise retry up to `retries` times (default is 0 here, see CLI).
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
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
            if _looks_like_hard_quota_error(e):
                raise HardQuotaError(str(e))
            last_err = e

        if attempt < retries:
            time.sleep(sleep_sec)

    raise last_err or RuntimeError("Gemini call failed")


def safe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
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
# Outputs (TSV/JSONL) helpers
# -----------------------------

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


@dataclass
class OutputHandles:
    tsv_path: Optional[Path]
    jsonl_path: Optional[Path]
    tsv_file: Optional[Any]
    tsv_writer: Optional[csv.DictWriter]
    jsonl_file: Optional[Any]


def _open_outputs(out_tsv: Optional[Path], out_jsonl: Optional[Path]) -> OutputHandles:
    tsv_f = None
    tsv_w = None
    jsonl_f = None

    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_f = out_jsonl.open("w", encoding="utf-8")

    if out_tsv:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        tsv_f = out_tsv.open("w", encoding="utf-8", newline="")
        tsv_w = csv.DictWriter(tsv_f, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore")
        tsv_w.writeheader()
        tsv_f.flush()

    return OutputHandles(out_tsv, out_jsonl, tsv_f, tsv_w, jsonl_f)


def _close_outputs(h: OutputHandles) -> None:
    if h.jsonl_file is not None:
        h.jsonl_file.close()
        print(f"[OK] raw JSONL -> {h.jsonl_path}")
    if h.tsv_file is not None:
        h.tsv_file.close()
        print(f"[OK] flattened TSV -> {h.tsv_path}")


def _suffix_path(p: Path, suffix: str) -> Path:
    # foo.tsv -> foo__suffix.tsv ; foo -> foo__suffix
    if p.suffix:
        return p.with_name(p.stem + suffix + p.suffix)
    return p.with_name(p.name + suffix)


# -----------------------------
# Main
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    defaults = try_load_paths_defaults()

    p = argparse.ArgumentParser(description="Weak label extractor (v5) with single-model & split-by-model support")

    p.add_argument("--sample-jsonl", default=defaults.get("sample_jsonl", None), required=False)
    p.add_argument("--key2txt", default=defaults.get("key2txt_tsv", None), required=False)
    p.add_argument("--sections-dir", default=defaults.get("sections_dir", None), required=False,
                   help="Directory containing <key>.sections.json files (optional).")

    # Output base paths (single files unless --split-by-model)
    p.add_argument("--out-jsonl", default=defaults.get("out_jsonl", None), required=False,
                   help="Write per-(key,model) raw JSONL output (optional).")
    p.add_argument("--out-tsv", default=defaults.get("out_tsv", None), required=False,
                   help="Write flattened TSV output (required unless you only want JSONL).")

    # Model selection
    p.add_argument("--models", default=None,
                   help="Comma-separated model names. Example: gemini-2.5-flash,gemma-3-12b-it")
    p.add_argument("--model", default=None,
                   help="Single model name. Overrides --models if provided.")
    p.add_argument("--split-by-model", action="store_true",
                   help="If set, write separate out files per model: <out>__<model>.tsv/jsonl")

    # Budget / rate behavior
    p.add_argument("--max-chars", type=int, default=30000)
    p.add_argument("--max-items", type=int, default=0, help="0 means all in sample.")
    p.add_argument("--sleep", type=float, default=1.0, help="Base sleep between successful calls.")
    p.add_argument("--retries", type=int, default=0,
                   help="Retry count for non-quota transient errors. Hard quota errors never retry.")

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

    out_jsonl_base = Path(args.out_jsonl) if args.out_jsonl else None
    out_tsv_base = Path(args.out_tsv) if args.out_tsv else None
    if not out_tsv_base and not out_jsonl_base:
        die("Provide --out-tsv and/or --out-jsonl.")

    # Resolve model list
    model_names: List[str] = []
    if args.model and str(args.model).strip():
        model_names = [str(args.model).strip()]
    elif args.models and str(args.models).strip():
        model_names = [m.strip() for m in str(args.models).split(",") if m.strip()]
    else:
        # Keep your original default if user did not specify
        model_names = ["gemini-2.5-flash", "gemma-3-12b-it"]

    if len(model_names) < 1:
        die("No models specified.")

    if not HAS_GENAI:
        die("google-generativeai is required for this script (install it in your env).")

    # init first model to validate API key
    ensure_genai(model_names[0])

    papers = read_jsonl_keys(sample_jsonl)
    key2txt = load_key2txt(key2txt_tsv)

    n_total = len(papers) if args.max_items <= 0 else min(len(papers), args.max_items)

    # Prepare outputs
    handles_by_model: Dict[str, OutputHandles] = {}
    if args.split_by_model:
        for m in model_names:
            # model string as safe filename chunk
            safe_m = re.sub(r"[^A-Za-z0-9_.-]+", "_", m)
            tsv_p = _suffix_path(out_tsv_base, f"__{safe_m}") if out_tsv_base else None
            jsonl_p = _suffix_path(out_jsonl_base, f"__{safe_m}") if out_jsonl_base else None
            handles_by_model[m] = _open_outputs(tsv_p, jsonl_p)
    else:
        handles_by_model["__ALL__"] = _open_outputs(out_tsv_base, out_jsonl_base)

    def get_handles(model: str) -> OutputHandles:
        if args.split_by_model:
            return handles_by_model[model]
        return handles_by_model["__ALL__"]

    # Run
    hard_stop_reason: Optional[str] = None
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

                try:
                    raw = call_gemini(model, prompt, retries=args.retries, sleep_sec=args.sleep)
                except HardQuotaError as e:
                    hard_stop_reason = f"{key} model={model} -> HARD QUOTA STOP: {e}"
                    raise
                except Exception as e:
                    print(f"[WARN] {key} model={model} -> CALL FAILED ({type(e).__name__}: {e})")
                    continue

                # JSON parse (fail-safe)
                try:
                    data = safe_json_load(raw)
                except Exception as e:
                    print(f"[WARN] {key} model={model} -> JSON PARSE FAILED ({type(e).__name__}: {e})")
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

                h = get_handles(model)

                # write raw jsonl (one record per key/model)
                if h.jsonl_file is not None:
                    raw_rec = {
                        "key": key,
                        "model": model,
                        "formulations": [
                            {"id": f.get("formulation_id"), "fields": f.get("fields", {}), "notes": f.get("notes")}
                            for f in formulations
                        ],
                        "paper_notes": paper_notes,
                    }
                    h.jsonl_file.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
                    h.jsonl_file.flush()

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
                        **evidence,
                    }

                    if h.tsv_writer is not None:
                        h.tsv_writer.writerow(row)

                if h.tsv_file is not None:
                    h.tsv_file.flush()

                print(f"[{i}/{n_total}] {key} model={model} -> {len(formulations)} row(s)")

                time.sleep(args.sleep)

    except HardQuotaError:
        # Do not retry; preserve partial outputs and stop immediately.
        if hard_stop_reason:
            print(f"[STOP] {hard_stop_reason}", file=sys.stderr)
    finally:
        for h in handles_by_model.values():
            _close_outputs(h)

    if hard_stop_reason:
        die("Stopped due to hard quota condition (see [STOP] line).", code=2)

    print("[DONE]")


if __name__ == "__main__":
    main(sys.argv[1:])
