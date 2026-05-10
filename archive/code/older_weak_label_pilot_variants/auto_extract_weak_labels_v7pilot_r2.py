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

from src.utils.model_policy import validate_models_or_raise
PRIMARY_DEFAULT = ""

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
        return {
            "value": v.get("value"),
            "value_text": str(v.get("value_text", "") or ""),
            "scope": str(v.get("scope", "unknown") or "unknown"),
            "membership_confidence": str(v.get("membership_confidence", "low") or "low"),
            "evidence_region_type": str(v.get("evidence_region_type", "unknown") or "unknown"),
            "missing_reason": str(v.get("missing_reason", "") or ""),
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
    fields = form.get("fields", {}) if isinstance(form.get("fields", {}), dict) else {}
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
    p = argparse.ArgumentParser(description="Pilot weak_labels_v7 extractor r2 (shared-condition prompt refinement).")
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
    validate_models_or_raise([args.model], context="v7pilot_r2 preflight")
    ensure_genai(args.model)

    papers = load_manifest(Path(args.manifest_tsv), args.max_items)
    if len(papers) == 0:
        die("No pilot papers loaded from manifest.")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    default_out = Path(f"data/results/run_{ts}_v7pilot3r2_dev/weak_labels_v7pilot_r2")
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "weak_labels__v7pilot_r2.jsonl"
    out_tsv = out_dir / "weak_labels__v7pilot_r2.tsv"

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
            data = safe_json_load(raw)

            line = {
                "key": paper.key,
                "doi": paper.doi,
                "title": paper.title,
                "model": args.model,
                "schema_version": data.get("schema_version", "weak_labels_v7"),
                "paper_notes": data.get("paper_notes"),
                "formulations": data.get("formulations", []),
            }
            jf.write(json.dumps(line, ensure_ascii=False) + "\n")

            forms = data.get("formulations", [])
            if not isinstance(forms, list):
                forms = []
            instance_fallback = choose_instance_evidence(txt)
            for idx, f in enumerate(forms, start=1):
                if not isinstance(f, dict):
                    continue
                formulation_id = f.get("formulation_id", f.get("id", idx))
                fields = f.get("fields", {}) if isinstance(f.get("fields", {}), dict) else {}
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
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
