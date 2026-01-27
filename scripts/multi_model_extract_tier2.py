#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_model_extract_tier2.py

Tier-2: improved multi-formulation + multi-model extractor
for small-batch PLGA papers (10-paper HTML sample).

Improvements over tier1:
- Uses a stricter prompt that only lists a SMALL set of REPRESENTATIVE
  formulations per paper (optimized, key controls, main comparison groups).
- Explicitly instructs the model NOT to treat Box–Behnken extremes,
  factorial design summary points, or dose groups as separate formulations.
- Uses stronger model pair by default:
    gemini-2.5-flash (main reference)
    gemini-2.0-flash-lite (cheap second opinion)
- High max_chars (default 100000) to avoid truncating important text.

Usage (from project root):

    .\.venv\Scripts\python.exe .\scripts\multi_model_extract_tier2.py ^
        --key2txt data\cleaned\samples\key2txt_html10.tsv ^
        --out-jsonl data\cleaned\samples\formulations_tier2_multi_model.jsonl ^
        --out-tsv  data\cleaned\samples\formulations_tier2_multi_model.tsv ^
        --models gemini-2.5-flash,gemini-2.0-flash-lite ^
        --max-chars 100000 ^
        --max-formulations 8
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from google import generativeai as genai

# ---- Project root & .env ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"

# Tier-1 style core fields ()
TIER1_FIELDS = [
    "emul_type",             # e.g., W/O/W, O/W
    "emul_method",           # e.g., double emulsion solvent evaporation
    "plga_mw_kDa",           # numeric
    "la_ga_ratio",           # "50:50", "75:25"
    "plga_mass_mg",          # numeric if possible
    "drug_name",
    "drug_feed_amount_text", # as reported
    "pva_conc_percent",      # numeric % if possible
    "organic_solvent",
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
    "notes",                 # short description of how this formulation is used
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tier-2 multi-formulation + multi-model extraction on small-batch key2txt."
    )
    p.add_argument(
        "--key2txt",
        type=Path,
        required=True,
        help="TSV with columns: key, text_path",
    )
    p.add_argument(
        "--out-jsonl",
        type=Path,
        required=True,
        help="Output JSONL, one record per (key, model).",
    )
    p.add_argument(
        "--out-tsv",
        type=Path,
        required=False,
        help="Optional TSV, flattened per formulation.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="gemini-2.5-flash,gemini-2.0-flash-lite",
        help="Comma-separated Gemini model names. Default: gemini-2.5-flash,gemini-2.0-flash-lite",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=100000,
        help="Max characters of text to send to LLM (truncate if longer).",
    )
    p.add_argument(
        "--max-formulations",
        type=int,
        default=8,
        help="Hint to LLM: maximum number of distinct formulations to list.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default 0.1 for extraction).",
    )
    return p


def build_prompt(text: str, max_formulations: int) -> str:
    """
    Improved prompt: focus on representative formulations, avoid Box–Behnken extremes,
    factorial summary points, and pure dose groups.
    """
    schema_desc = "\n".join(f'- "{f}"' for f in TIER1_FIELDS)
    return f"""
You are extracting PLGA emulsion / microparticle / nanoparticle FORMULATIONS
from the following scientific paper text.

Your goal is to identify a SMALL SET (at most ~{max_formulations}) of
REPRESENTATIVE formulations that correspond to DISTINCT nanoparticle /
microsphere COMPOSITIONS or PREPARATION CONDITIONS.

IMPORTANT – what COUNTS as a separate formulation:
- Different PLGA type or molecular weight (e.g., 17 kDa vs 50 kDa vs 96 kDa).
- Different LA:GA ratio (e.g., 50:50 vs 75:25).
- Different main organic solvent (e.g., DCM vs ethyl acetate vs methanol).
- Different stabilizer / surfactant systems (e.g., PVA vs polysorbate-80 vs none),
  or clearly different stabilizer concentration (e.g., 0.5% vs 2.0%).
- Clearly different drug:polymer feed ratios (e.g., 1:4 vs 1:2 vs 2:1).
- Important control formulations (e.g., blank PLGA particles without drug,
  FITC-labelled particles used as imaging controls, polymeric core without lipid).
- Key formulations that were fully characterized (size, EE and/or release)
  or used for key in vitro or in vivo experiments.

What does NOT count as a separate formulation (DO NOT list these as individual items):
- Statistical summaries from design-of-experiments:
  e.g., "minimum particle size among 29 batches", "maximum EE from Box–Behnken design".
- Generic descriptions of factorial or Box–Behnken designs without a concrete recipe
  for a specific batch (only mention these in "paper_notes" if important).
- In vivo dose groups that reuse the SAME nanoparticle formulation
  (e.g., HDNP vs LDNP groups differing only in injected dose).
- Repeated mentions of the same formulation used in different experiments
  (e.g., same NPs used for release, imaging, and in vivo).
- Pure control pastes/suspensions that do NOT involve PLGA particles
  (e.g., a simple Ca(OH)2 paste without PLGA).

If the paper reports many similar design points or batches, choose ONLY the most
representative formulations that:
- Were fully characterized (size, EE, release and/or zeta potential), and/or
- Were used for the main in vitro or in vivo studies.

TEXT:
---------------- TEXT START ----------------
{text}
---------------- TEXT END ------------------

For EACH formulation you DO decide to list, extract the following fields
inside "fields":

{schema_desc}

Output JSON structure (NO extra text):

{{
  "formulations": [
    {{
      "id": 1,
      "fields": {{
        "emul_type": "...",
        "emul_method": "...",
        "plga_mw_kDa": ...,
        "la_ga_ratio": "...",
        "plga_mass_mg": ...,
        "drug_name": "...",
        "drug_feed_amount_text": "...",
        "pva_conc_percent": ...,
        "organic_solvent": "...",
        "size_nm": ...,
        "pdi": ...,
        "zeta_mV": ...,
        "encapsulation_efficiency_percent": ...,
        "loading_content_percent": ...,
        "notes": "short note about how this formulation is used (e.g., optimized, blank control, imaging control, main in vivo group)."
      }}
    }},
    {{
      "id": 2,
      "fields": {{ ... }}
    }}
  ],
  "paper_notes": "short explanation of how many formulations were actually prepared, "
                 "and which ones you chose to list or omit (e.g., '29 Box–Behnken batches; "
                 "only optimized and blank formulations listed here')."
}}

Rules:
- List at MOST {max_formulations} formulations for this paper.
  If there are more, choose only the most representative ones as described above.
- If a field is not clearly reported for a formulation, set it to null.
- Numeric fields should be numbers, NOT strings (e.g., 150.0, 0.12, -25.3).
- la_ga_ratio should preserve the reported ratio as a string (e.g., "50:50").
- "drug_feed_amount_text" is a free-text copy of how the input dose / feed
  was reported for that formulation.
- Keep "notes" and "paper_notes" concise but informative.
- DO NOT invent values; prefer null + explanation in notes if unclear.
""".strip()


def call_gemini(
    model_name: str,
    text: str,
    max_formulations: int,
    temperature: float,
) -> Dict[str, Any]:
    """
    Call Gemini with the improved Tier-2 prompt and parse JSON.
    """
    model = genai.GenerativeModel(model_name)
    prompt = build_prompt(text, max_formulations)
    resp = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    raw = resp.text or ""
    try:
        data = json.loads(raw)
    except Exception:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        data = json.loads(cleaned)
    return data


def main() -> None:
    args = build_arg_parser().parse_args()

    # ---- Load .env & configure Gemini ----
    if DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH)
        print(f"[INFO] Loaded .env from {DOTENV_PATH}")
    else:
        print(f"[WARN] .env not found at {DOTENV_PATH}, relying on OS env vars.")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "[ERROR] GEMINI_API_KEY not found. Put it in .env at project root, e.g.\n"
            "    GEMINI_API_KEY=your_key_here\n"
        )

    try:
        genai.configure(api_key=api_key)
        print(f"[INFO] Gemini configured. API key suffix: ...{api_key[-6:]}")
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to configure Gemini: {e}")

    # ---- Load key2txt ----
    key2txt = pd.read_csv(args.key2txt, sep="\t")
    if not {"key", "text_path"}.issubset(key2txt.columns):
        raise SystemExit("[ERROR] key2txt must have columns: key, text_path")

    model_names: List[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise SystemExit("[ERROR] At least one model must be specified via --models")

    out_jsonl = args.out_jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    flat_rows: List[Dict[str, Any]] = []
    n_calls = 0

    with out_jsonl.open("w", encoding="utf-8") as f_out:
        for _, row in key2txt.iterrows():
            key = str(row["key"])
            text_path_str = str(row["text_path"])
            text_path = Path(text_path_str)
            if not text_path.is_absolute():
                text_path = (PROJECT_ROOT / text_path).resolve()

            if not text_path.exists():
                print(f"[WARN] Missing text file for key={key}: {text_path}")
                continue

            txt = text_path.read_text(encoding="utf-8", errors="ignore")
            orig_len = len(txt)
            if orig_len > args.max_chars:
                print(
                    f"[WARN] key={key}: text length={orig_len} > max_chars={args.max_chars}, truncating."
                )
                txt = txt[: args.max_chars]

            print(f"[INFO] key={key}: text chars={len(txt)}")

            for model_name in model_names:
                print(f"  [CALL] key={key}, model={model_name}")
                try:
                    data = call_gemini(
                        model_name=model_name,
                        text=txt,
                        max_formulations=args.max_formulations,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    print(f"  [ERROR] model={model_name} failed: {e}")
                    data = {
                        "formulations": [],
                        "paper_notes": f"LLM error: {e}",
                        "_error": str(e),
                    }

                rec = {
                    "key": key,
                    "model": model_name,
                    "formulations": data.get("formulations", []),
                    "paper_notes": data.get("paper_notes"),
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_calls += 1

                # flatten for TSV
                formulations = data.get("formulations") or []
                if isinstance(formulations, list):
                    for fobj in formulations:
                        fid = fobj.get("id")
                        fields = fobj.get("fields", {}) or {}
                        flat_row = {
                            "key": key,
                            "model": model_name,
                            "formulation_id": fid,
                        }
                        for field in TIER1_FIELDS:
                            flat_row[field] = fields.get(field)
                        flat_rows.append(flat_row)

    print(f"[INFO] Done. Total LLM calls: {n_calls}")
    print(f"[OK] JSONL written to: {out_jsonl}")

    if args.out_tsv and flat_rows:
        out_tsv = args.out_tsv
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        df_flat = pd.DataFrame(flat_rows)
        df_flat.to_csv(out_tsv, sep="\t", index=False)
        print(f"[OK] Flattened TSV written to: {out_tsv}")


if __name__ == "__main__":
    main()
