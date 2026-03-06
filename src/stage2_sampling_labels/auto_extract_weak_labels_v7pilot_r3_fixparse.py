#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pilot-only weak_labels_v7 extractor.

Purpose:
- Run a small pilot with stronger LLM-side semantic structure.
- Keep compatibility with current core fields while adding per-field semantic metadata.
- Do not change stable mainline extraction scripts.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise

HAS_GENAI = False
try:
    import google.generativeai as genai  # type: ignore

    HAS_GENAI = True
except Exception:
    HAS_GENAI = False


CORE_FIELDS = [
    "emul_type",
    "emul_method",
    "la_ga_ratio",
    "plga_mw_kDa",
    "plga_mass_mg",
    "surfactant_name",
    "surfactant_concentration_text",
    "pva_conc_percent",
    "organic_solvent",
    "drug_name",
    "drug_feed_amount_text",
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
]

# Observed fallback order when the model returns unnamed field objects as a list.
# This keeps pilot flattening deterministic without changing schema/prompt contracts.
UNNAMED_LIST_FALLBACK_ORDER = [
    "emul_method",
    "la_ga_ratio",
    "plga_mw_kDa",
    "plga_mass_mg",
    "surfactant_name",
    "surfactant_concentration_text",
    "organic_solvent",
    "drug_name",
    "drug_feed_amount_text",
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
]

VALUE_ALIASES_COMMON = [
    "value_raw",
    "raw_value",
    "raw",
    "value_num",
    "numeric_value",
    "value_number",
    "value_str",
    "normalized_value",
]

VALUE_ALIASES_BY_FIELD = {
    "la_ga_ratio": ["ratio", "la_ga", "la_ga_value", "ratio_value"],
    "plga_mw_kDa": ["mw_kda", "mw", "molecular_weight_kda", "plga_mw", "mw_value"],
}


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def normalize_doi(v: Any) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def choose_instance_evidence(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "evidence_region_type": "unknown",
            "evidence_section": "",
            "evidence_span_text": "",
            "evidence_span_start": "",
            "evidence_span_end": "",
        }
    m = re.search(
        r"(size|pdi|zeta|encapsulation\s+efficiency|drug/polymer|la/?ga).{0,260}",
        text,
        flags=re.IGNORECASE | re.S,
    )
    if not m:
        return {
            "evidence_region_type": "unknown",
            "evidence_section": "",
            "evidence_span_text": "",
            "evidence_span_start": "",
            "evidence_span_end": "",
        }
    s = max(0, m.start() - 60)
    e = min(len(text), m.end() + 120)
    span = text[s:e].replace("\n", " ").strip()
    return {
        "evidence_region_type": "unknown",
        "evidence_section": "full_text_window",
        "evidence_span_text": span,
        "evidence_span_start": int(s),
        "evidence_span_end": int(e),
    }


FEW_SHOT = r"""
Few-shot guidance (compact examples):
1) Shared method/header condition:
- Methods or table header states: "PLGA 50 kDa, LA/GA 50:50, solvent acetone for all formulations F1-F4".
- Use:
  plga_mw_kDa.scope = global_shared
  la_ga_ratio.scope = global_shared
  organic_solvent.scope = global_shared

2) Row-specific result:
- Table row F3 reports size and EE values.
- Use:
  size_nm.scope = instance_specific
  encapsulation_efficiency_percent.scope = instance_specific

3) Ambiguous evidence:
- Text does not clearly bind surfactant concentration to one formulation or all formulations.
- Use:
  surfactant_concentration_text.scope = unknown
  membership_confidence = low

4) Polymer product code vs molecular weight:
- Input text: "PLGA (Resomer RG 502) was used as the polymer."
- Interpret as polymer grade/product code, not direct MW value.
- Use:
  plga_mw_kDa.value = null
  plga_mw_kDa.value_text = "Resomer RG 502 (PLGA grade)"
  plga_mw_kDa.scope = global_shared (if stated once for all formulations)
  plga_mw_kDa.membership_confidence = medium

5) Baseline + additive variant:
- Input text: "Nanoparticles were prepared using 0.5% PVA as stabilizer. Variants were prepared using the same protocol with additional surfactants."
- Use:
  surfactant_name value "PVA" with scope = global_shared
  additional surfactant labels with scope = instance_specific
"""


LLM_PROMPT_TEMPLATE = (
    "You are extracting PLGA nanoparticle formulation data in weak_labels_v7 pilot format.\n"
    "Return ONLY valid JSON with keys: schema_version, paper_notes, formulations.\n"
    "Use schema_version='weak_labels_v7'.\n"
    "Each formulation object must include: formulation_id, formulation_role, instance_confidence, fields.\n"
    "Allowed formulation_role: baseline, control, optimized, variant, comparative, characterization_only, unknown.\n"
    "Allowed instance_confidence: high, medium, low.\n"
    "For each core field below, output a field object (or omit if fully unknown):\n"
    + ", ".join(CORE_FIELDS)
    + "\n"
    "Field object schema:\n"
    "- value: scalar or null\n"
    "- value_text: string\n"
    "- scope: global_shared | instance_specific | unknown\n"
    "- membership_confidence: high | medium | low\n"
    "- evidence_region_type: table_cell | table_row | table_header | table_block | methods_sentence | results_sentence | unknown\n"
    "- missing_reason: optional string for unknown/ambiguous cases\n\n"
    "Rules:\n"
    "- Do not hallucinate values; prefer unknown style values when uncertain.\n"
    "- Preserve units in value_text.\n"
    "- If table header defines shared conditions, mark scope=global_shared.\n"
    "- If value is clearly tied to one formulation row, mark scope=instance_specific.\n"
    "- Do not overuse unknown when the paper clearly states one common condition for all listed formulations.\n"
    "- Use global_shared only when the same condition truly applies to all extracted formulations in the paper or in one coherent table block.\n"
    "- Product codes like RG 502 / RG 503 / RG 504 are polymer grade names, not molecular-weight values.\n"
    "- If only a polymer product code is given, keep it in value_text and set plga_mw_kDa.value=null.\n"
    "- If a baseline stabilizer or solvent is declared once and reused across formulations, mark baseline condition as global_shared.\n"
    "- For ambiguous assignment, set scope=unknown and low membership_confidence.\n\n"
    + FEW_SHOT
    + "\nTEXT:\n"
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
    return {"schema_version": "weak_labels_v7", "paper_notes": None, "formulations": []}


def normalize_field_obj(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        val = v.get("value")
        if val is None or (isinstance(val, str) and not val.strip()):
            for k in VALUE_ALIASES_COMMON:
                cand = v.get(k)
                if cand is not None and (not isinstance(cand, str) or cand.strip()):
                    val = cand
                    break
        return {
            "value": val,
            "value_text": str(v.get("value_text", "") or ""),
            "scope": str(v.get("scope", "unknown") or "unknown"),
            "membership_confidence": str(v.get("membership_confidence", "low") or "low"),
            "evidence_region_type": str(v.get("evidence_region_type", "unknown") or "unknown"),
            "missing_reason": str(v.get("missing_reason", "") or ""),
            "value_source": str(v.get("value_source", "") or ""),
        }
    if v is None:
        return {
            "value": None,
            "value_text": "",
            "scope": "unknown",
            "membership_confidence": "low",
            "evidence_region_type": "unknown",
            "missing_reason": "not_reported",
        }
    return {
        "value": v,
        "value_text": str(v),
        "scope": "unknown",
        "membership_confidence": "medium",
        "evidence_region_type": "unknown",
        "missing_reason": "",
        "value_source": "",
    }


def sanitize_role(v: Any) -> str:
    allowed = {
        "baseline",
        "control",
        "optimized",
        "variant",
        "comparative",
        "characterization_only",
        "unknown",
    }
    s = str(v or "unknown").strip().lower()
    return s if s in allowed else "unknown"


def sanitize_conf(v: Any) -> str:
    allowed = {"high", "medium", "low"}
    s = str(v or "low").strip().lower()
    return s if s in allowed else "low"


def sanitize_scope(v: Any) -> str:
    allowed = {"global_shared", "instance_specific", "unknown"}
    s = str(v or "unknown").strip().lower()
    return s if s in allowed else "unknown"


def sanitize_region(v: Any) -> str:
    allowed = {
        "table_cell",
        "table_row",
        "table_header",
        "table_block",
        "methods_sentence",
        "results_sentence",
        "unknown",
    }
    s = str(v or "unknown").strip().lower()
    return s if s in allowed else "unknown"


def _norm_text(v: Any) -> str:
    s = "" if v is None else str(v)
    return re.sub(r"\s+", " ", s).strip()


def _infer_field_name(item: Dict[str, Any]) -> Optional[str]:
    blob = f"{_norm_text(item.get('value_text'))} {_norm_text(item.get('value'))}".lower()
    if not blob:
        return None
    if re.search(r"\b(single|double)\s*emulsion\b", blob):
        return "emul_type"
    if re.search(r"\bnanoprecipitation\b|\bemulsification\b|\bsolvent evaporation\b|\bsolvent diffusion\b", blob):
        return "emul_method"
    if re.search(r"\b\d+\s*:\s*\d+\b", blob):
        return "la_ga_ratio"
    if re.search(r"\bresomer\b|\brg\s*50[0-9]{1,2}\b|\bmw\b|\bkda\b|\bpolymer grade\b", blob):
        return "plga_mw_kDa"
    if re.search(r"\bplga\b.*\bmg\b|\bpolymer\b.*\bmg\b", blob):
        return "plga_mass_mg"
    if re.search(r"\bpva\b|\bpolyvinyl alcohol\b|\btween\b|\bpoloxamer\b|\bpluronic\b|\blabrafil\b|\bsurfactant\b", blob):
        if re.search(r"\b\d+(\.\d+)?\s*(%|w/?v|mg/ml|mg)\b", blob):
            return "surfactant_concentration_text"
        return "surfactant_name"
    if re.search(r"\bacetone\b|\bdcm\b|dichloromethane|ethyl acetate|chloroform|methanol|acetonitrile", blob):
        return "organic_solvent"
    if re.search(r"\bdrug\b|\brhodamine\b|\bdox(orubicin)?\b|\bcurcumin\b|\bpaclitaxel\b|\b5-fu\b", blob):
        if re.search(r"\b\d+(\.\d+)?\s*(mg|ug|µg)\b", blob):
            return "drug_feed_amount_text"
        return "drug_name"
    if re.search(r"\bpdi\b", blob):
        return "pdi"
    if re.search(r"\bzeta\b|\bmv\b", blob):
        return "zeta_mV"
    if re.search(r"encapsulation|entrapment|\bee\b", blob):
        return "encapsulation_efficiency_percent"
    if re.search(r"\bloading\b|\blc\b|mg/100\s*mg", blob):
        return "loading_content_percent"
    if re.search(r"\bnm\b", blob):
        return "size_nm"
    return None


def coerce_fields_map(raw_fields: Any) -> Dict[str, Any]:
    # Bugfix: r3 flattening dropped all list-style fields by forcing non-dict
    # payloads to {}. This mapper preserves value/scope/membership/evidence data.
    if isinstance(raw_fields, dict):
        return raw_fields
    if not isinstance(raw_fields, list):
        return {}

    out: Dict[str, Any] = {}
    unnamed: List[Dict[str, Any]] = []
    for it in raw_fields:
        if not isinstance(it, dict):
            continue
        explicit = it.get("field_name") or it.get("name") or it.get("field") or it.get("key")
        if explicit and str(explicit) in CORE_FIELDS:
            out[str(explicit)] = it
            continue
        guessed = _infer_field_name(it)
        if guessed and guessed not in out:
            out[guessed] = it
            continue
        unnamed.append(it)

    if unnamed:
        remaining = [f for f in UNNAMED_LIST_FALLBACK_ORDER if f not in out]
        for idx, it in enumerate(unnamed):
            if idx >= len(remaining):
                break
            out[remaining[idx]] = it
    return out


def _resolve_field_name(item: Dict[str, Any], fallback_name: str = "") -> str:
    nm = item.get("field_name") or item.get("name") or item.get("field") or item.get("key") or fallback_name
    nm = str(nm or "").strip()
    if nm in CORE_FIELDS:
        return nm
    guessed = _infer_field_name(item)
    if guessed:
        return guessed
    return nm


def _recover_value_from_aliases(field_name: str, item: Dict[str, Any]) -> Any:
    keys = ["value"] + VALUE_ALIASES_COMMON + VALUE_ALIASES_BY_FIELD.get(field_name, [])
    for k in keys:
        if k not in item:
            continue
        v = item.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _recover_value_from_value_text(field_name: str, item: Dict[str, Any]) -> Any:
    txt = _norm_text(item.get("value_text", ""))
    if not txt:
        return None
    if field_name == "la_ga_ratio":
        m = re.search(r"\b(\d{1,3}\s*:\s*\d{1,3})\b", txt)
        return m.group(1).replace(" ", "") if m else None
    if field_name == "plga_mw_kDa":
        # Accept explicit kDa notation.
        m_kda = re.search(r"\b(\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?)?\s*kda)\b", txt, flags=re.I)
        if m_kda:
            return _norm_text(m_kda.group(1))
        # Accept explicit molecular-weight ranges written as 50 000-75 000.
        m_range = re.search(r"\b(\d{2,3}\s*000\s*(?:[-–]\s*\d{2,3}\s*000))\b", txt)
        if m_range:
            return _norm_text(m_range.group(1))
    return None


def canonicalize_formulations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    forms = data.get("formulations", [])
    if not isinstance(forms, list):
        return []
    out_forms: List[Dict[str, Any]] = []
    for fm in forms:
        if not isinstance(fm, dict):
            continue
        new_fm = dict(fm)
        raw_fields = fm.get("fields", {})
        canon_map = coerce_fields_map(raw_fields)
        canon_fields: Dict[str, Dict[str, Any]] = {}
        for k, raw_obj in canon_map.items():
            if not isinstance(raw_obj, dict):
                raw_obj = {"value": raw_obj}
            field_name = _resolve_field_name(raw_obj, str(k))
            if field_name not in CORE_FIELDS:
                continue
            obj = dict(raw_obj)
            recovered = _recover_value_from_aliases(field_name, obj)
            if recovered is None:
                # Parser recovery from explicit field value_text only.
                recovered = _recover_value_from_value_text(field_name, obj)
            cur = obj.get("value")
            if (cur is None or (isinstance(cur, str) and not cur.strip())) and recovered is not None:
                obj["value"] = recovered
                obj["value_source"] = "parser_recovered"
            elif cur is not None and (not isinstance(cur, str) or cur.strip()):
                obj["value_source"] = str(obj.get("value_source", "") or "llm_direct")
            obj["field_name"] = field_name
            canon_fields[field_name] = obj
        new_fm["fields"] = canon_fields
        out_forms.append(new_fm)
    return out_forms


def _safe_filename_part(v: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(v).strip())


@dataclass
class PilotPaper:
    key: str
    doi: str
    title: str
    text_path: Path


def load_manifest(manifest_tsv: Path, max_items: int) -> List[PilotPaper]:
    if not manifest_tsv.exists():
        die(f"Manifest not found: {manifest_tsv}")
    df = pd.read_csv(manifest_tsv, sep="\t", dtype=str).fillna("")
    required = {"key", "doi", "title", "text_path"}
    missing = required - set(df.columns)
    if missing:
        die(f"Manifest missing required columns: {sorted(missing)}")
    rows: List[PilotPaper] = []
    for _, r in df.iterrows():
        p = Path(str(r["text_path"]).replace("\\", "/"))
        rows.append(
            PilotPaper(
                key=str(r["key"]).strip(),
                doi=normalize_doi(r["doi"]),
                title=str(r["title"]).strip(),
                text_path=p,
            )
        )
    if max_items > 0:
        rows = rows[:max_items]
    return rows


def flatten_row(
    key: str,
    doi: str,
    model: str,
    form: Dict[str, Any],
    instance_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "key": key,
        "doi": doi,
        "model": model,
        "formulation_id": form.get("formulation_id"),
        "formulation_role": sanitize_role(form.get("formulation_role")),
        "instance_confidence": sanitize_conf(form.get("instance_confidence")),
        "instance_evidence_region_type": sanitize_region(instance_evidence.get("evidence_region_type")),
        "evidence_section": instance_evidence.get("evidence_section", ""),
        "evidence_span_text": instance_evidence.get("evidence_span_text", ""),
        "evidence_span_start": instance_evidence.get("evidence_span_start", ""),
        "evidence_span_end": instance_evidence.get("evidence_span_end", ""),
    }
    fields = coerce_fields_map(form.get("fields", {}))
    for f in CORE_FIELDS:
        obj = normalize_field_obj(fields.get(f))
        out[f"{f}_value"] = obj.get("value")
        out[f"{f}_value_text"] = obj.get("value_text", "")
        out[f"{f}_scope"] = sanitize_scope(obj.get("scope"))
        out[f"{f}_membership_confidence"] = sanitize_conf(obj.get("membership_confidence"))
        out[f"{f}_evidence_region_type"] = sanitize_region(obj.get("evidence_region_type"))
        out[f"{f}_missing_reason"] = obj.get("missing_reason", "")
    return out


def build_output_columns() -> List[str]:
    cols = [
        "key",
        "doi",
        "model",
        "formulation_id",
        "formulation_role",
        "instance_confidence",
        "instance_evidence_region_type",
        "evidence_section",
        "evidence_span_text",
        "evidence_span_start",
        "evidence_span_end",
    ]
    for f in CORE_FIELDS:
        cols.extend(
            [
                f"{f}_value",
                f"{f}_value_text",
                f"{f}_scope",
                f"{f}_membership_confidence",
                f"{f}_evidence_region_type",
                f"{f}_missing_reason",
            ]
        )
    return cols


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pilot weak_labels_v7 extractor r3_fixparse (parser + flatten bugfixes).")
    p.add_argument(
        "--manifest-tsv",
        default="data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv",
    )
    p.add_argument("--model", default=PRIMARY_DEFAULT)
    p.add_argument("--max-chars", type=int, default=50000)
    p.add_argument("--max-items", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.4)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--out-dir", default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    validate_models_or_raise([args.model], context="v7pilot_r3_fixparse preflight")
    ensure_genai(args.model)

    papers = load_manifest(Path(args.manifest_tsv), args.max_items)
    if len(papers) == 0:
        die("No pilot papers loaded from manifest.")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    default_out = Path(f"data/results/run_{ts}_v7pilot3r3fixparse_dev/weak_labels_v7pilot_r3_fixparse")
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "weak_labels__v7pilot_r3_fixparse.jsonl"
    out_tsv = out_dir / "weak_labels__v7pilot_r3_fixparse.tsv"
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    columns = build_output_columns()
    tsv_f = out_tsv.open("w", encoding="utf-8", newline="")
    tsv_w = csv.DictWriter(tsv_f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
    tsv_w.writeheader()

    n_forms = 0
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for i, paper in enumerate(papers, start=1):
            if not paper.text_path.exists():
                print(f"[WARN] missing text path: {paper.text_path}")
                continue
            txt = paper.text_path.read_text(encoding="utf-8", errors="ignore")
            if args.max_chars > 0 and len(txt) > args.max_chars:
                txt = txt[: args.max_chars]
            prompt = LLM_PROMPT_TEMPLATE + txt

            raw = call_gemini(args.model, prompt, args.retries, args.sleep)
            raw_fp = raw_dir / f"{i:02d}_{_safe_filename_part(paper.key)}_{_safe_filename_part(paper.doi)}.txt"
            raw_fp.write_text(raw, encoding="utf-8")
            data = safe_json_load(raw)
            forms = canonicalize_formulations(data)

            line = {
                "key": paper.key,
                "doi": paper.doi,
                "title": paper.title,
                "model": args.model,
                "schema_version": data.get("schema_version", "weak_labels_v7"),
                "paper_notes": data.get("paper_notes"),
                "formulations": forms,
            }
            jf.write(json.dumps(line, ensure_ascii=False) + "\n")

            instance_fallback = choose_instance_evidence(txt)
            for idx, f in enumerate(forms, start=1):
                if not isinstance(f, dict):
                    continue
                formulation_id = f.get("formulation_id", f.get("id", idx))
                # Keep raw structure; flatten_row handles dict and list styles.
                fields = f.get("fields", {})
                row_form = {
                    "formulation_id": formulation_id,
                    "formulation_role": f.get("formulation_role", "unknown"),
                    "instance_confidence": f.get("instance_confidence", "low"),
                    "fields": fields,
                }
                inst_ev = f.get("instance_evidence", {})
                if not isinstance(inst_ev, dict):
                    inst_ev = {}
                if not inst_ev:
                    inst_ev = instance_fallback
                row = flatten_row(paper.key, paper.doi, args.model, row_form, inst_ev)
                tsv_w.writerow(row)
                n_forms += 1

            if args.verbose:
                print(f"[{i}/{len(papers)}] key={paper.key} doi={paper.doi} formulations={len(forms)}")

    tsv_f.close()
    summary = {
        "manifest_tsv": str(Path(args.manifest_tsv).resolve()),
        "model": args.model,
        "n_papers": len(papers),
        "n_formulations": n_forms,
        "out_dir": str(out_dir.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "out_tsv": str(out_tsv.resolve()),
        "raw_responses_dir": str(raw_dir.resolve()),
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
