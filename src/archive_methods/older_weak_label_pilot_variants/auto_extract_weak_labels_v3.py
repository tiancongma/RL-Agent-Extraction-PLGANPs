#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_extract_weak_labels_v3.py

Multi-formulation extractor for PLGA emulsion/microsphere/nanoparticle papers,
with two-layer outputs: RAW (as reported) and DERIVED (computed deterministically).

What this script does:
- Recognizes emulsion type (emul_type: 'W/O/W', 'O/W', 'W/O', 'O/O', 'O/O/W', ...).
- Extracts RAW volumes & ratios (w1/o/w2) as text; optionally normalized to mL if directly parseable.
- Does NOT let the LLM compute numbers; instead, postprocess() deterministically computes *_derived volumes.
- Writes per-paper JSONL and a TSV with one row per formulation.

Environment:
- Requires GEMINI_API_KEY or GOOGLE_API_KEY.
- Automatically loads .env from project root (two levels up from scripts/).
"""

# --- Load .env from project root automatically ---
from pathlib import Path
from dotenv import load_dotenv
import os

_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV)
    print(f"[INFO] Loaded environment from {_ROOT_ENV}")
else:
    print("[WARN] .env not found; relying on system environment variables.")

import argparse
import json
import re
import time
from typing import Any, Dict, List, Optional

# ---------------- Gemini Setup ----------------
def get_gemini():
    try:
        import google.generativeai as genai
    except Exception as e:
        raise SystemExit("Please install google-generativeai: pip install google-generativeai\n" + str(e))
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY / GOOGLE_API_KEY in environment.")
    genai.configure(api_key=api_key)
    return genai

# ---------------- Schema ----------------
# Tier-1 (core)
T1_FIELDS = [
    "emul_type",                  # NEW
    "plga_mw_kDa",
    "la_ga_ratio",
    "emul_method",
    "emul_time_s",
    "emul_intensity",
    "pva_conc_text",
    "pva_conc_percent",
    "organic_solvent",
    "size_nm",
    "pdi",
    "zeta_mV",
]

# Tier-2 (recommended)
T2_FIELDS = [
    # RAW volume strings (as reported)
    "w1_vol", "o_vol", "w2_vol",
    # Normalized numeric volumes (when directly parseable from text)
    "w1_vol_mL", "o_vol_mL", "w2_vol_mL",
    # Ratios (RAW and normalized)
    "w1_o_ratio_text", "w1_o_ratio_norm",
    "o_w2_ratio_text", "o_w2_ratio_norm",
    "w1_o_w2_ratio_text", "w1_o_w2_ratio_norm",
    # Total phase volume (RAW & numeric)
    "total_phase_vol_text", "total_phase_vol_mL",
    # DERIVED volumes + provenance + notes
    "w1_vol_mL_derived", "o_vol_mL_derived", "w2_vol_mL_derived",
    "w1_vol_mL_source", "o_vol_mL_source", "w2_vol_mL_source",
    "derived_notes",
    # Applicability mask for volumes (as 'true'/'false' strings)
    "w1_vol_applicable", "o_vol_applicable", "w2_vol_applicable",
    # Mass/efficiency/composition
    "plga_mass_g", "drug_feed_amount_g",
    "drug_polymer_ratio",
    "encapsulation_efficiency_percent",
    "drug_loading_percent",
]

# Tier-3 (optional/context)
T3_FIELDS = [
    "aux_materials",
    "organic_solvent_vol_mL",
    "release_profile_type",
    "drug_name",
]

ALL_FIELDS = T1_FIELDS + T2_FIELDS + T3_FIELDS

SYSTEM_INSTRUCTIONS = (
    "You are an expert extractor for PLGA emulsion/microsphere/nanoparticle synthesis papers.\n"
    "Your task: extract ALL distinct formulations/conditions reported in the text.\n"
    "Return a SINGLE JSON object with EXACTLY this top-level shape:\n"
    "{\n"
    '  "formulations": [\n'
    "    { <fields for formulation 1> },\n"
    "    { <fields for formulation 2> },\n"
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "For EACH formulation, include the following fields (use these exact keys):\n"
    + ", ".join(ALL_FIELDS) + "\n\n"
    "Rules:\n"
    "- Identify emulsion type as 'W/O/W', 'O/W', 'W/O', 'O/O', 'O/O/W' if given; else leave empty. "
    "If a 'double emulsion' with inner and external aqueous phase is described, use 'W/O/W'.\n"
    "- Do NOT perform arithmetic or infer missing values. Extract volumes as reported (keep unit strings) "
    "and, when possible, also provide normalized numeric values in mL (w1_vol_mL/o_vol_mL/w2_vol_mL). "
    "Extract any given ratios explicitly (e.g., '1:5', 'A:B:C') into *_ratio_text and normalized *_ratio_norm (like '1:5').\n"
    "- For numeric fields that are not present, return null. For text fields, return an empty string.\n"
    "- pva_conc_text captures the raw report (e.g., '1% w/v'); pva_conc_percent is numeric or null.\n"
    "- emul_intensity may be 'rpm', '%amp', 'W', etc.; keep as text.\n"
    "- If the paper has only one formulation, still return a list with a single object.\n"
    "- Return ONLY the JSON object (no markdown, no code fences)."
)

# ---------------- Helpers ----------------
JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)

def extract_first_json(s: str):
    if not s:
        return None
    m = JSON_BLOCK_RE.search(s)
    if not m:
        return None
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        cleaned = re.sub(r",\s*}", "}", block)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

def to_float_or_none(x):
    if x in (None, "", [], {}):
        return None
    try:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
        return float(nums[0]) if nums else None
    except Exception:
        return None

def parse_volume_to_mL(raw: str) -> Optional[float]:
    """
    Try to parse a volume string to mL. Accepts µL/uL, mL, L. Returns float mL or None.
    """
    if not raw:
        return None
    s = str(raw).strip()
    s = s.replace("μL", "uL").replace("µL", "uL").lower()
    m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(ul|ml|l)\b", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "ul":
        return val / 1000.0
    if unit == "ml":
        return val
    if unit == "l":
        return val * 1000.0
    return None

def norm_ratio(text: str) -> str:
    """
    Normalize ratio text to 'A:B' or 'A:B:C' numeric form. Returns '' if cannot parse.
    """
    if not text:
        return ""
    t = str(text).replace(" ", "").replace("v/v","").replace("V/V","")
    m3 = re.match(r"^\s*(\d*\.?\d+):(\d*\.?\d+):(\d*\.?\d+)\s*$", t)
    m2 = re.match(r"^\s*(\d*\.?\d+):(\d*\.?\d+)\s*$", t)
    if m3:
        a,b,c = m3.groups()
        return f"{float(a)}:{float(b)}:{float(c)}"
    if m2:
        a,b = m2.groups()
        return f"{float(a)}:{float(b)}"
    return ""

def normalize_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Fill defaults
    for f in ALL_FIELDS:
        if f in {
            "plga_mw_kDa","emul_time_s","pva_conc_percent","size_nm","pdi","zeta_mV",
            "w1_vol_mL","o_vol_mL","w2_vol_mL","total_phase_vol_mL",
            "encapsulation_efficiency_percent","drug_loading_percent",
            "organic_solvent_vol_mL","plga_mass_g","drug_feed_amount_g",
            "w1_vol_mL_derived","o_vol_mL_derived","w2_vol_mL_derived",
        }:
            obj.setdefault(f, None)
        else:
            obj.setdefault(f, "")

    # Coerce numeric where applicable
    for k in ["plga_mw_kDa","emul_time_s","pva_conc_percent","size_nm","pdi","zeta_mV",
              "w1_vol_mL","o_vol_mL","w2_vol_mL","total_phase_vol_mL",
              "encapsulation_efficiency_percent","drug_loading_percent",
              "organic_solvent_vol_mL","plga_mass_g","drug_feed_amount_g"]:
        obj[k] = to_float_or_none(obj.get(k))

    # Try parsing mL from raw strings if numeric not provided
    for raw_key, ml_key in [("w1_vol","w1_vol_mL"),("o_vol","o_vol_mL"),("w2_vol","w2_vol_mL")]:
        if obj.get(ml_key) is None and obj.get(raw_key):
            parsed = parse_volume_to_mL(obj.get(raw_key))
            if parsed is not None:
                obj[ml_key] = parsed

    # Normalize ratios
    for src, dst in [("w1_o_ratio_text","w1_o_ratio_norm"),
                     ("o_w2_ratio_text","o_w2_ratio_norm"),
                     ("w1_o_w2_ratio_text","w1_o_w2_ratio_norm")]:
        if not obj.get(dst) and obj.get(src):
            obj[dst] = norm_ratio(obj[src])

    # Basic string cleanup for text fields
    for k in ["emul_type","la_ga_ratio","emul_method","emul_intensity","pva_conc_text",
              "organic_solvent","aux_materials","release_profile_type","drug_name",
              "w1_vol","o_vol","w2_vol",
              "w1_o_ratio_text","o_w2_ratio_text","w1_o_w2_ratio_text",
              "total_phase_vol_text","drug_polymer_ratio","derived_notes",
              "w1_vol_mL_source","o_vol_mL_source","w2_vol_mL_source",
              "w1_vol_applicable","o_vol_applicable","w2_vol_applicable"]:
        v = obj.get(k)
        obj[k] = "" if v is None else str(v).strip()

    return obj

# --------- Buckets (Tier-1) ---------
def bucket_plga_mw(mw: Optional[float]) -> str:
    if mw is None: return ""
    if mw < 20: return "low(<20)"
    if mw <= 50: return "mid(20-50)"
    return "high(>50)"

def bucket_ratio_la_ga(r: str) -> str:
    if not r: return ""
    canonical = r.replace(" ", "").replace("%","")
    for cand in ["75:25","65:35","50:50","25:75","85:15"]:
        if cand in canonical:
            return cand
    return "other"

def bucket_pva(p: Optional[float]) -> str:
    if p is None: return ""
    if p < 0.5: return "<0.5%"
    if p <= 1.0: return "0.5–1%"
    return ">1%"

def bucket_emul_method(m: str) -> str:
    if not m: return ""
    low = m.lower()
    if "sonic" in low: return "sonication"
    if "homogen" in low: return "homogenization"
    return "other"

def bucket_time_s(t: Optional[float]) -> str:
    if t is None: return ""
    if t < 30: return "<30"
    if t <= 60: return "30–60"
    return ">60"

def bucket_size(s: Optional[float]) -> str:
    if s is None: return ""
    if s < 150: return "<150"
    if s <= 250: return "150–250"
    return ">250"

def bucket_pdi(p: Optional[float]) -> str:
    if p is None: return ""
    if p < 0.1: return "<0.1"
    if p <= 0.2: return "0.1–0.2"
    return ">0.2"

def bucket_zeta(z: Optional[float]) -> str:
    if z is None: return ""
    if z <= -30: return "≤-30"
    if z <= -10: return "(-30,-10]"
    return ">-10"

def make_bins(fields: Dict[str, Any]) -> Dict[str, str]:
    return {
        "plga_mw_kDa_bin": bucket_plga_mw(fields.get("plga_mw_kDa")),
        "la_ga_ratio_bin": bucket_ratio_la_ga(fields.get("la_ga_ratio","")),
        "pva_conc_bin": bucket_pva(fields.get("pva_conc_percent")),
        "emul_method_bin": bucket_emul_method(fields.get("emul_method","")),
        "emul_time_s_bin": bucket_time_s(fields.get("emul_time_s")),
        "size_nm_bin": bucket_size(fields.get("size_nm")),
        "pdi_bin": bucket_pdi(fields.get("pdi")),
        "zeta_mV_bin": bucket_zeta(fields.get("zeta_mV")),
    }

# --------- Postprocess (DERIVED volumes) ---------
def parse_ratio_numbers(ratio_text: str) -> Optional[List[float]]:
    """
    Robustly parse ratios like '1:5' or '1:5:10'. Accepts any input type; coerces to string.
    Returns list of floats or None.
    """
    if not ratio_text:
        return None
    s = str(ratio_text)
    nums = re.findall(r"\d*\.?\d+", s)
    if not nums:
        return None
    try:
        return [float(x) for x in nums]
    except Exception:
        return None

def postprocess_compute_volumes(f: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically compute *_derived (mL) using ratios + known volumes, without overriding reported values.
    Set *_source to 'reported' or 'computed_from_<rule>'.
    """
    # Mark sources for reported numeric volumes
    for key in ["w1_vol_mL","o_vol_mL","w2_vol_mL"]:
        if f.get(key) is not None and f.get(key) != "":
            f[key.replace("_mL","_mL_source")] = "reported"

    def set_derived(name: str, value: Optional[float], rule: str):
        if value is None:
            return
        dkey = name + "_derived"
        skey = name + "_source"
        if f.get(name) is None:
            f[dkey] = value
            f[skey] = f"computed_from_{rule}"

    # Two-phase ratio W1:O
    r_w1o = f.get("w1_o_ratio_norm") or ""
    parts = parse_ratio_numbers(r_w1o)
    if parts and len(parts) == 2:
        a, b = parts
        if a > 0 and b > 0:
            if f.get("w1_vol_mL") is not None and f.get("o_vol_mL") is None:
                set_derived("o_vol_mL", f["w1_vol_mL"] * (b / a), "w1_o_ratio+w1_vol_mL")
            if f.get("o_vol_mL") is not None and f.get("w1_vol_mL") is None:
                set_derived("w1_vol_mL", f["o_vol_mL"] * (a / b), "w1_o_ratio+o_vol_mL")

    # Two-phase ratio O:W2
    r_ow2 = f.get("o_w2_ratio_norm") or ""
    parts = parse_ratio_numbers(r_ow2)
    if parts and len(parts) == 2:
        a, b = parts
        if a > 0 and b > 0:
            if f.get("o_vol_mL") is not None and f.get("w2_vol_mL") is None:
                set_derived("w2_vol_mL", f["o_vol_mL"] * (b / a), "o_w2_ratio+o_vol_mL")
            if f.get("w2_vol_mL") is not None and f.get("o_vol_mL") is None:
                set_derived("o_vol_mL", f["w2_vol_mL"] * (a / b), "o_w2_ratio+w2_vol_mL")

    # Three-phase ratio W1:O:W2
    r_w1ow2 = f.get("w1_o_w2_ratio_norm") or ""
    parts = parse_ratio_numbers(r_w1ow2)
    if parts and len(parts) == 3:
        A, B, C = parts
        total = f.get("total_phase_vol_mL")
        denom = A + B + C if (A + B + C) > 0 else None
        if denom:
            if total is not None:
                base = total / denom
                set_derived("w1_vol_mL", base * A, "w1_o_w2_ratio+total_phase")
                set_derived("o_vol_mL",  base * B, "w1_o_w2_ratio+total_phase")
                set_derived("w2_vol_mL", base * C, "w1_o_w2_ratio+total_phase")
            else:
                # Exactly one known → scale all from that one
                known = [(k, f.get(k)) for k in ["w1_vol_mL","o_vol_mL","w2_vol_mL"] if f.get(k) is not None]
                if len(known) == 1:
                    kname, kval = known[0]
                    if kname == "w1_vol_mL" and A > 0:
                        base = kval / A
                    elif kname == "o_vol_mL" and B > 0:
                        base = kval / B
                    elif kname == "w2_vol_mL" and C > 0:
                        base = kval / C
                    else:
                        base = None
                    if base is not None:
                        set_derived("w1_vol_mL", base * A, "w1_o_w2_ratio+single_phase")
                        set_derived("o_vol_mL",  base * B, "w1_o_w2_ratio+single_phase")
                        set_derived("w2_vol_mL", base * C, "w1_o_w2_ratio+single_phase")

    return f

# --------- Applicability mask ---------
def infer_volume_applicability(fields: Dict[str, Any]) -> None:
    """
    Set volume applicability booleans ('true'/'false' strings) based on emulsion type.
    - W/O/W: w1, o, w2 applicable
    - O/W:   w1 not applicable; o, w2 applicable
    - W/O:   w1, o applicable; w2 not
    - O/O:   only o applicable
    Others left blank for annotator judgment.
    """
    et = (fields.get("emul_type") or "").upper()
    def set_app(k, val):
        fields[k] = "true" if val else "false"
    if not et:
        return
    if et == "W/O/W":
        set_app("w1_vol_applicable", True)
        set_app("o_vol_applicable", True)
        set_app("w2_vol_applicable", True)
    elif et == "O/W":
        set_app("w1_vol_applicable", False)
        set_app("o_vol_applicable", True)
        set_app("w2_vol_applicable", True)
    elif et == "W/O":
        set_app("w1_vol_applicable", True)
        set_app("o_vol_applicable", True)
        set_app("w2_vol_applicable", False)
    elif et == "O/O":
        set_app("w1_vol_applicable", False)
        set_app("o_vol_applicable", True)
        set_app("w2_vol_applicable", False)
    else:
        # leave blank
        pass

# --------- Safe string for TSV ---------
def _s(v: Any) -> str:
    """Safe string for TSV: None->'', list/dict->json, else str(v)."""
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-jsonl", default="data/cleaned/samples/sample30.jsonl")
    ap.add_argument("--outdir", default="data/cleaned/samples")
    ap.add_argument("--per-item-dir", default=None)
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--max_chars", type=int, default=18000)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    sample = Path(args.sample_jsonl)
    if not sample.exists():
        raise SystemExit(f"Sample manifest not found: {sample}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    per_item_dir = Path(args.per_item_dir) if args.per_item_dir else (outdir / "weak_labels_v3")
    per_item_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"; logs_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = outdir / "weak_labels_v3.jsonl"
    out_tsv   = outdir / "weak_labels_v3.tsv"
    fail_log  = logs_dir / "weak_labels_v3_failures.json"

    # Load sample items
    items: List[Dict[str, Any]] = []
    with sample.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("text_path"):
                items.append(rec)
    if not items:
        raise SystemExit("No valid items with text_path.")

    genai = get_gemini()
    model = genai.GenerativeModel(args.model)

    failures: List[Dict[str, Any]] = []
    total_rows = 0

    # TSV header (one row per formulation; includes RAW and DERIVED summaries)
    tsv_cols = [
        "key","title","formulation_id",
        "emul_type",
        # Tier-1 raw core
        "plga_mw_kDa","la_ga_ratio","emul_method","emul_time_s","emul_intensity",
        "pva_conc_text","pva_conc_percent","organic_solvent",
        "size_nm","pdi","zeta_mV",
        # Tier-1 bins
        "plga_mw_kDa_bin","la_ga_ratio_bin","pva_conc_bin","emul_method_bin",
        "emul_time_s_bin","size_nm_bin","pdi_bin","zeta_mV_bin",
        # Volumes/ratios (RAW)
        "w1_vol","o_vol","w2_vol",
        "w1_vol_mL","o_vol_mL","w2_vol_mL",
        "w1_o_ratio_text","w1_o_ratio_norm",
        "o_w2_ratio_text","o_w2_ratio_norm",
        "w1_o_w2_ratio_text","w1_o_w2_ratio_norm",
        "total_phase_vol_text","total_phase_vol_mL",
        # Applicability
        "w1_vol_applicable","o_vol_applicable","w2_vol_applicable",
        # DERIVED (computed)
        "w1_vol_mL_derived","o_vol_mL_derived","w2_vol_mL_derived",
        "w1_vol_mL_source","o_vol_mL_source","w2_vol_mL_source",
        "derived_notes",
        # Other T2/T3
        "plga_mass_g","drug_feed_amount_g","drug_polymer_ratio",
        "encapsulation_efficiency_percent","drug_loading_percent",
        "aux_materials","organic_solvent_vol_mL","release_profile_type","drug_name",
    ]

    with out_jsonl.open("w", encoding="utf-8") as wj, out_tsv.open("w", encoding="utf-8") as wt:
        wt.write("\t".join(tsv_cols) + "\n")

        for i, rec in enumerate(items, 1):
            key = rec.get("key") or rec.get("zotero_key") or f"item_{i}"
            title = rec.get("title","")
            year = str(rec.get("year",""))
            doi  = rec.get("doi","") or ""
            txtp = Path(rec["text_path"])

            if not txtp.exists():
                failures.append({"key": key, "reason": "text_path_not_found", "text_path": str(txtp)})
                continue

            try:
                text = txtp.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                failures.append({"key": key, "reason": "read_error", "error": str(e)})
                continue

            prompt = f"TITLE: {title}\nYEAR: {year}\nDOI: {doi}\n\n{text[:args.max_chars]}"

            tries, resp_text = 0, None
            while tries < 3 and resp_text is None:
                tries += 1
                try:
                    resp = model.generate_content([
                        {"role":"user","parts":[SYSTEM_INSTRUCTIONS]},
                        {"role":"user","parts":[prompt]},
                    ])
                    resp_text = (getattr(resp, "text", None) or "").strip()
                except Exception as e:
                    if tries >= 3:
                        failures.append({"key": key, "reason": "model_error", "error": str(e)})
                        break
                    time.sleep(1.5 * tries)
            if not resp_text:
                continue

            payload = extract_first_json(resp_text)
            formulations_raw: List[Dict[str, Any]] = []
            if isinstance(payload, dict) and isinstance(payload.get("formulations"), list):
                formulations_raw = [x for x in payload["formulations"] if isinstance(x, dict)]
            elif isinstance(payload, list):
                formulations_raw = [x for x in payload if isinstance(x, dict)]

            if not formulations_raw:
                failures.append({"key": key, "reason": "no_formulations_parsed", "response_excerpt": resp_text[:800]})
                continue

            # Build per-paper object (with normalized fields, bins, and derived)
            formulations_out: List[Dict[str, Any]] = []
            for idx, fobj in enumerate(formulations_raw, start=1):
                fields = normalize_fields(dict(fobj))
                infer_volume_applicability(fields)
                fields = postprocess_compute_volumes(fields)
                bins = make_bins(fields)
                formulations_out.append({"id": idx, "fields": fields, "bins": bins})

                # TSV row
                fld = fields; bn = bins
                row = [
                    _s(key), _s(title), _s(idx),
                    _s(fld.get("emul_type")),
                    _s(fld.get("plga_mw_kDa")), _s(fld.get("la_ga_ratio")), _s(fld.get("emul_method")), _s(fld.get("emul_time_s")), _s(fld.get("emul_intensity")),
                    _s(fld.get("pva_conc_text")), _s(fld.get("pva_conc_percent")), _s(fld.get("organic_solvent")),
                    _s(fld.get("size_nm")), _s(fld.get("pdi")), _s(fld.get("zeta_mV")),
                    _s(bn.get("plga_mw_kDa_bin")), _s(bn.get("la_ga_ratio_bin")), _s(bn.get("pva_conc_bin")), _s(bn.get("emul_method_bin")),
                    _s(bn.get("emul_time_s_bin")), _s(bn.get("size_nm_bin")), _s(bn.get("pdi_bin")), _s(bn.get("zeta_mV_bin")),
                    _s(fld.get("w1_vol")), _s(fld.get("o_vol")), _s(fld.get("w2_vol")),
                    _s(fld.get("w1_vol_mL")), _s(fld.get("o_vol_mL")), _s(fld.get("w2_vol_mL")),
                    _s(fld.get("w1_o_ratio_text")), _s(fld.get("w1_o_ratio_norm")),
                    _s(fld.get("o_w2_ratio_text")), _s(fld.get("o_w2_ratio_norm")),
                    _s(fld.get("w1_o_w2_ratio_text")), _s(fld.get("w1_o_w2_ratio_norm")),
                    _s(fld.get("total_phase_vol_text")), _s(fld.get("total_phase_vol_mL")),
                    _s(fld.get("w1_vol_applicable")), _s(fld.get("o_vol_applicable")), _s(fld.get("w2_vol_applicable")),
                    _s(fld.get("w1_vol_mL_derived")), _s(fld.get("o_vol_mL_derived")), _s(fld.get("w2_vol_mL_derived")),
                    _s(fld.get("w1_vol_mL_source")), _s(fld.get("o_vol_mL_source")), _s(fld.get("w2_vol_mL_source")),
                    _s(fld.get("derived_notes")),
                    _s(fld.get("plga_mass_g")), _s(fld.get("drug_feed_amount_g")), _s(fld.get("drug_polymer_ratio")),
                    _s(fld.get("encapsulation_efficiency_percent")), _s(fld.get("drug_loading_percent")),
                    _s(fld.get("aux_materials")), _s(fld.get("organic_solvent_vol_mL")), _s(fld.get("release_profile_type")), _s(fld.get("drug_name")),
                ]
                wt.write("\t".join(row) + "\n")
                total_rows += 1

            per_paper = {
                "key": key,
                "title": title,
                "year": year,
                "doi": doi,
                "formulations": formulations_out,
                "meta": {"model": args.model, "chars_used": min(len(text), args.max_chars)},
            }
            (per_item_dir / f"{key}.json").write_text(json.dumps(per_paper, ensure_ascii=False, indent=2), encoding="utf-8")
            with out_jsonl.open("a", encoding="utf-8") as wja:
                wja.write(json.dumps(per_paper, ensure_ascii=False) + "\n")

            print(f"[{i}/{len(items)}] {key} -> {len(formulations_out)} formulation(s)")
            time.sleep(args.sleep)

    # Failures log
    if failures:
        fail_log.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUMMARY] papers={len(items)}, tsv_rows={total_rows}, failed={len(failures)})")

if __name__ == "__main__":
    main()
