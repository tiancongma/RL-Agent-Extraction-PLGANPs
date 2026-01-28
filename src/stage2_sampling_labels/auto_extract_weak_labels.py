#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_extract_weak_labels_v2.py

Like v1, but aligned to the tiered schema & bucketization plan.
- Adds Tier-1/2/3 fields
- Captures raw strings (e.g., pva_conc_text) as well as numeric (pva_conc_percent)
- Produces a "bins" dict alongside "fields"
- TSV now includes key Tier-1 bins to reduce sparsity in quick views

Usage:
  python auto_extract_weak_labels_v2.py \
    --sample-jsonl data/cleaned/samples/sample30.jsonl \
    --outdir data/cleaned/samples \
    --model gemini-2.5-flash-lite
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# ---- Schema ----
# Tier-1
T1_FIELDS = [
    "plga_mw_kDa",
    "la_ga_ratio",
    "emul_method",
    "emul_time_s",
    "emul_intensity",
    "pva_conc_text",          # raw string like "1%" or "2 w/v%"
    "pva_conc_percent",       # numeric parsed percent if present
    "organic_solvent",
    # Outputs
    "size_nm",
    "pdi",
    "zeta_mV",
]

# Tier-2
T2_FIELDS = [
    "w1_vol_mL", "o_vol_mL", "w2_vol_mL",
    "plga_mass_g", "drug_feed_amount_g",
    "drug_polymer_ratio",
    "encapsulation_efficiency_percent",
    "drug_loading_percent",
]

# Tier-3
T3_FIELDS = [
    "aux_materials",              # string or list in text; we keep string
    "organic_solvent_vol_mL",
    "release_profile_type",
    "drug_name",
]

ALL_FIELDS = T1_FIELDS + T2_FIELDS + T3_FIELDS

SYSTEM_INSTRUCTIONS = (
    "You are an expert extractor for PLGA emulsion/microsphere/nanoparticle synthesis papers.\n"
    "Return a SINGLE JSON object with EXACT keys given below. Rules:\n"
    "- Prefer explicit values from Methods/Results. Do NOT invent values.\n"
    "- If a numeric field is missing, return null. For text fields, return an empty string.\n"
    "- For la_ga_ratio: use 'A:B' when reported.\n"
    "- pva_conc_text should capture the raw report (e.g., '1% w/v'); pva_conc_percent should be the numeric percent (e.g., 1.0) or null.\n"
    "- emul_intensity may be 'rpm', '%amp', 'W', or descriptive string; keep as text.\n"
    "- Return ONLY the JSON object, no markdown/code fences.\n"
)

def make_user_prompt(title: str, year: str, doi: str, text: str) -> str:
    schema_hint = "Fields: " + ", ".join(ALL_FIELDS) + "\n\n"
    header = f"TITLE: {title}\nYEAR: {year}\nDOI: {doi}\n\n"
    return header + schema_hint + text

JSON_OBJECT_REGEX = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
    m = JSON_OBJECT_REGEX.search(s)
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

def to_float_or_none(x) -> Optional[float]:
    if x in (None, "", [], {}):
        return None
    try:
        # pull first number
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
        return float(nums[0]) if nums else None
    except Exception:
        return None

def normalize_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure presence
    for f in ALL_FIELDS:
        obj.setdefault(f, None if f.endswith("_percent") or f in
            ["plga_mw_kDa", "emul_time_s", "size_nm", "pdi", "zeta_mV", "w1_vol_mL", "o_vol_mL", "w2_vol_mL",
             "plga_mass_g", "drug_feed_amount_g", "encapsulation_efficiency_percent", "organic_solvent_vol_mL", "drug_loading_percent"]
        else "")

    # Numeric coercion
    numeric_fields = [
        "plga_mw_kDa", "emul_time_s",
        "pva_conc_percent",
        "size_nm", "pdi", "zeta_mV",
        "w1_vol_mL", "o_vol_mL", "w2_vol_mL",
        "plga_mass_g", "drug_feed_amount_g",
        "encapsulation_efficiency_percent", "drug_loading_percent",
        "organic_solvent_vol_mL",
    ]
    for k in numeric_fields:
        obj[k] = to_float_or_none(obj.get(k))

    # Text normalize
    textish = [
        "la_ga_ratio", "emul_method", "emul_intensity", "pva_conc_text",
        "organic_solvent", "aux_materials", "release_profile_type", "drug_name",
        "drug_polymer_ratio"
    ]
    for k in textish:
        v = obj.get(k)
        obj[k] = "" if v is None else str(v).strip()

    # If only pva_conc_text like "2%" provided and pva_conc_percent missing, parse
    if obj.get("pva_conc_percent") is None and obj.get("pva_conc_text"):
        pct = to_float_or_none(obj["pva_conc_text"])
        # Accept only plausible 0-20%
        if pct is not None and 0 <= pct <= 20:
            obj["pva_conc_percent"] = pct

    return obj

# ---- Buckets ----
def bucket_plga_mw(mw: Optional[float]) -> str:
    if mw is None: return ""
    if mw < 20: return "low(<20)"
    if mw <= 50: return "mid(20-50)"
    return "high(>50)"

def bucket_ratio(r: str) -> str:
    if not r: return ""
    canonical = r.replace(" ", "").replace("%","")
    # map common ones
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
        "la_ga_ratio_bin": bucket_ratio(fields.get("la_ga_ratio","")),
        "pva_conc_bin": bucket_pva(fields.get("pva_conc_percent")),
        "emul_method_bin": bucket_emul_method(fields.get("emul_method","")),
        "emul_time_s_bin": bucket_time_s(fields.get("emul_time_s")),
        "size_nm_bin": bucket_size(fields.get("size_nm")),
        "pdi_bin": bucket_pdi(fields.get("pdi")),
        "zeta_mV_bin": bucket_zeta(fields.get("zeta_mV")),
    }

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
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    per_item_dir = Path(args.per_item_dir) if args.per_item_dir else (outdir / "weak_labels_v2")
    per_item_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"; logs_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = outdir / "weak_labels_v2.jsonl"
    out_tsv   = outdir / "weak_labels_v2.tsv"
    fail_log  = logs_dir / "weak_labels_v2_failures.json"

    # Load items
    items = []
    if not sample.exists():
        raise SystemExit(f"Sample manifest not found: {sample}")
    with sample.open("r", encoding="utf-8") as f:
        for line in f:
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

    failures = []
    written = 0

    # TSV header emphasizes Tier-1 raw + bins
    tsv_cols = [
        "key","title",
        "plga_mw_kDa","plga_mw_kDa_bin",
        "la_ga_ratio","la_ga_ratio_bin",
        "emul_method","emul_method_bin",
        "emul_time_s","emul_time_s_bin",
        "emul_intensity",
        "pva_conc_text","pva_conc_percent","pva_conc_bin",
        "organic_solvent",
        "size_nm","size_nm_bin",
        "pdi","pdi_bin",
        "zeta_mV","zeta_mV_bin",
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

            prompt = make_user_prompt(title, year, doi, text[:args.max_chars])

            tries, resp_text = 0, None
            while tries < 3 and resp_text is None:
                tries += 1
                try:
                    resp = model.generate_content([
                        {"role":"user","parts":[SYSTEM_INSTRUCTIONS]},
                        {"role":"user","parts":[prompt]},
                    ])
                    resp_text = (resp.text or "").strip()
                except Exception as e:
                    if tries >= 3:
                        failures.append({"key": key, "reason": "model_error", "error": str(e)})
                        break
                    time.sleep(1.5 * tries)
            if not resp_text:
                continue

            obj = extract_json_obj(resp_text) or {}
            # Ensure all fields present & normalized
            for fld in ALL_FIELDS:
                obj.setdefault(fld, None if fld in
                    ["plga_mw_kDa","emul_time_s","pva_conc_percent","size_nm","pdi","zeta_mV",
                     "w1_vol_mL","o_vol_mL","w2_vol_mL","plga_mass_g","drug_feed_amount_g",
                     "encapsulation_efficiency_percent","drug_loading_percent","organic_solvent_vol_mL"]
                    else "")
            fields = normalize_fields(obj)
            bins = make_bins(fields)

            per_item = {
                "key": key,
                "title": title,
                "year": year,
                "doi": doi,
                "fields": fields,     # raw (numeric+text)
                "bins": bins,         # categorical bins for Tier-1 targets
                "meta": {"model": args.model, "chars_used": min(len(text), args.max_chars)}
            }

            # Write artifacts
            (per_item_dir / f"{key}.json").write_text(json.dumps(per_item, ensure_ascii=False, indent=2), encoding="utf-8")
            wj.write(json.dumps(per_item, ensure_ascii=False) + "\n")

            # TSV row
            row = [
                key, title,
                str(fields["plga_mw_kDa"] or ""),
                bins["plga_mw_kDa_bin"],
                fields["la_ga_ratio"],
                bins["la_ga_ratio_bin"],
                fields["emul_method"],
                bins["emul_method_bin"],
                str(fields["emul_time_s"] or ""),
                bins["emul_time_s_bin"],
                fields["emul_intensity"],
                fields["pva_conc_text"],
                str(fields["pva_conc_percent"] or ""),
                bins["pva_conc_bin"],
                fields["organic_solvent"],
                str(fields["size_nm"] or ""),
                bins["size_nm_bin"],
                str(fields["pdi"] or ""),
                bins["pdi_bin"],
                str(fields["zeta_mV"] or ""),
                bins["zeta_mV_bin"],
            ]
            wt.write("\t".join(row) + "\n")

            written += 1
            print(f"[{i}/{len(items)}] {key} -> OK")
            time.sleep(args.sleep)

    if failures:
        (logs_dir / "weak_labels_v2_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUMMARY] written={written}, failed={len(failures)}")

if __name__ == "__main__":
    main()
