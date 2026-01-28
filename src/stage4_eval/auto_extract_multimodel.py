#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_extract_multimodel.py

Two-model weak label extractor + cross-check for PLGA formulation papers.

What this script does
---------------------
1. Reads a key2txt.tsv file, where each row maps:
      key    text_path
   to a cleaned text file (from pdf2clean).

2. For each paper (row):
   - Calls PRIMARY LLM to extract formulations as structured JSON.
   - Calls SECONDARY LLM to do the *same* extraction independently.
   - Compares the two JSON outputs:
       * If normalized JSON is identical  -> agree = 1 (auto-accepted)
       * Otherwise                        -> agree = 0 (needs manual review)

3. Saves:
   - TSV summary for manual labeling:
       key, agree, reason, primary_json, secondary_json
   - Optional JSONL files storing primary / secondary outputs per paper.

Minimal example (debug on a small batch)
----------------------------------------
python scripts/auto_extract_multimodel.py \
  --key2txt data/cleaned/samples/key2txt.tsv \
  --out-tsv data/cleaned/samples/multimodel_agreement.tsv \
  --out-jsonl-primary data/cleaned/samples/primary_extract.jsonl \
  --out-jsonl-secondary data/cleaned/samples/secondary_extract.jsonl \
  --max-papers 5 \
  --primary-model gemini-1.5-flash \
  --secondary-model gemini-1.5-flash-8b

Requirements
------------
- google-generativeai
- python-dotenv
- pandas
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import google.generativeai as genai


# -----------------------
# Utility: logging helpers
# -----------------------

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr)


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


# -----------------------
# LLM setup & call
# -----------------------

def configure_gemini_from_env() -> None:
    """
    Configure google.generativeai using API key from environment.

    We try the following env vars in order:
    - GEMINI_API_KEY
    - GOOGLE_API_KEY
    """
    load_dotenv()
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        log_error(
            "No Gemini API key found. Please set GEMINI_API_KEY or GOOGLE_API_KEY "
            "in your environment or .env file."
        )
        sys.exit(1)
    genai.configure(api_key=api_key)
    log_info("Configured google.generativeai with API key from environment.")


def build_extraction_prompt(paper_key: str, text: str) -> str:
    """
    Build a strict-JSON extraction prompt for PLGA formulations.
    You can extend the schema later; for now keep it minimal but useful.
    """
    schema_example = {
        "paper_key": paper_key,
        "formulations": [
            {
                "id": 1,
                "size_nm": 150.0,
                "pdi": 0.12,
                "zeta_mV": -20.5,
                "plga_mw_kDa": 30.0,
                "la_ga_ratio": "50:50",
                "pva_percent": 1.0,
                "organic_solvent": "dichloromethane",
                "emul_type": "W1/O/W2"
            }
        ]
    }

    prompt = f"""
You are an information extraction assistant for PLGA nanoparticle formulations.

Task:
Extract ALL distinct PLGA emulsion/microsphere/nanoparticle formulations from the text
below and return STRICT JSON, following the schema.

JSON schema (example):
{json.dumps(schema_example, indent=2)}

Field rules:
- "paper_key": must be the string "{paper_key}".
- "formulations": a list with one entry per distinct experimental formulation.
- "id": integer starting from 1 for this paper.
- For numeric fields ("size_nm", "pdi", "zeta_mV", "plga_mw_kDa", "pva_percent"):
    - Use a NUMBER (not string) when available.
    - If unknown or not reported, use null.
- For string fields ("la_ga_ratio", "organic_solvent", "emul_type"):
    - Use a non-empty string if you can reasonably infer it.
    - Otherwise use the empty string "".
- If the text clearly has NO PLGA formulations, set "formulations": [].

STRICT output requirements:
- Output MUST be PURE JSON (no markdown, no comments, no trailing commas).
- Do NOT wrap the JSON in code fences.
- Do NOT add explanations before or after the JSON.

Now extract from this text:

---------------- TEXT START ({paper_key}) ----------------
{text}
---------------- TEXT END ({paper_key}) ----------------
"""
    return prompt.strip()


def extract_json_from_llm_response(resp_text: str) -> Dict[str, Any]:
    """
    Try to robustly parse JSON from an LLM response.

    - Strips whitespace and optional ```json / ``` fences.
    - Looks for the first '{' and the last '}' as a crude boundary.
    """
    raw = resp_text.strip()

    # Remove markdown fences if present
    if raw.startswith("```"):
        # remove first fence line
        lines = raw.splitlines()
        # drop first line (``` or ```json)
        lines = lines[1:]
        # drop last line if it's a closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # Fallback: crop from first '{' to last '}'
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw = raw[start:end]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}\nRaw text:\n{raw[:1000]}")


def call_gemini_model(
    model_name: str,
    prompt: str,
    temperature: float = 0.1,
    max_output_tokens: int = 4096,
    top_p: float = 0.8,
    top_k: int = 40,
) -> Dict[str, Any]:
    """
    Call a Gemini model with the given prompt and parse JSON output.

    Raises ValueError if parsing fails.
    """
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k,
            },
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}") from e

    if not response or not getattr(response, "text", None):
        raise ValueError("Empty response from Gemini model.")

    return extract_json_from_llm_response(response.text)


# -----------------------
# Agreement logic
# -----------------------

def normalize_json_for_comparison(obj: Any) -> Any:
    """
    Normalize JSON-like data for equality comparison:

    - Sorts dict keys.
    - Recursively normalizes lists and dicts.
    """
    if isinstance(obj, dict):
        return {k: normalize_json_for_comparison(obj[k]) for k in sorted(obj.keys())}
    elif isinstance(obj, list):
        return [normalize_json_for_comparison(x) for x in obj]
    else:
        return obj


def compare_extractions(
    j1: Dict[str, Any],
    j2: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Compare two extraction JSON objects.

    For now we use a conservative rule:
    - If normalized JSON objects are exactly equal -> agree=True
    - Otherwise -> agree=False and a simple reason is returned.

    You can later extend this to fuzzy comparison (e.g. allow slight numeric differences).
    """
    n1 = normalize_json_for_comparison(j1)
    n2 = normalize_json_for_comparison(j2)

    if n1 == n2:
        return True, "Exact JSON match (normalized)."

    # Try to produce a brief reason
    f1 = j1.get("formulations", [])
    f2 = j2.get("formulations", [])
    if len(f1) != len(f2):
        return False, f"Different number of formulations: primary={len(f1)} vs secondary={len(f2)}."

    return False, "JSON differs in field values (same number of formulations)."


# -----------------------
# Main pipeline
# -----------------------

def load_key2txt(key2txt_path: Path) -> pd.DataFrame:
    """
    Load key2txt.tsv, expecting at least columns:
        - key
        - text_path   (or 'path', 'txt_path' etc. - we try to infer)
    """
    df = pd.read_csv(key2txt_path, sep="\t", dtype=str)

    # Try to find the text path column
    candidate_cols = ["text_path", "path", "txt_path", "cleaned_path"]
    text_col = None
    for c in candidate_cols:
        if c in df.columns:
            text_col = c
            break

    # Fallback: if exactly two columns and the second is path-like, use it
    if text_col is None and df.shape[1] == 2:
        text_col = df.columns[1]
        log_warn(
            f"key2txt.tsv has 2 columns but no explicit text_path header. "
            f"Using second column '{text_col}' as text path."
        )

    if "key" not in df.columns or text_col is None:
        raise ValueError(
            f"key2txt.tsv must contain 'key' and a text path column. "
            f"Got columns: {list(df.columns)}"
        )

    df = df[["key", text_col]].rename(columns={text_col: "text_path"})
    return df


def read_text_file(path: Path) -> str:
    """
    Read a text file (UTF-8 with fallback).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def run_multimodel_extraction(
    key2txt_path: Path,
    out_tsv: Path,
    primary_model: str,
    secondary_model: str,
    out_jsonl_primary: Optional[Path] = None,
    out_jsonl_secondary: Optional[Path] = None,
    max_papers: Optional[int] = None,
) -> None:
    """
    Core loop: for each paper in key2txt.tsv, run two models and compare outputs.
    """
    df = load_key2txt(key2txt_path)
    log_info(f"Loaded key2txt: {key2txt_path} rows={len(df)}")

    if max_papers is not None and max_papers > 0:
        df = df.head(max_papers)
        log_info(f"Subsampling to first {len(df)} papers due to --max-papers={max_papers}")

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl_primary:
        out_jsonl_primary.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl_secondary:
        out_jsonl_secondary.parent.mkdir(parents=True, exist_ok=True)

    # Open JSONL outputs if requested
    f_primary = open(out_jsonl_primary, "w", encoding="utf-8") if out_jsonl_primary else None
    f_secondary = open(out_jsonl_secondary, "w", encoding="utf-8") if out_jsonl_secondary else None

    rows_for_tsv: List[Dict[str, Any]] = []

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Papers"):
            key = str(row["key"])
            text_path = Path(row["text_path"])

            try:
                text = read_text_file(text_path)
            except Exception as e:
                log_warn(f"[{key}] Failed to read text file {text_path}: {e}")
                rows_for_tsv.append(
                    {
                        "key": key,
                        "agree": 0,
                        "reason": f"Failed to read text: {e}",
                        "primary_json": "",
                        "secondary_json": "",
                    }
                )
                continue

            prompt = build_extraction_prompt(key, text)

            # Call primary model
            try:
                primary_json = call_gemini_model(primary_model, prompt)
            except Exception as e:
                log_warn(f"[{key}] Primary model error: {e}")
                rows_for_tsv.append(
                    {
                        "key": key,
                        "agree": 0,
                        "reason": f"Primary model error: {e}",
                        "primary_json": "",
                        "secondary_json": "",
                    }
                )
                continue

            # Call secondary model
            try:
                secondary_json = call_gemini_model(secondary_model, prompt)
            except Exception as e:
                log_warn(f"[{key}] Secondary model error: {e}")
                rows_for_tsv.append(
                    {
                        "key": key,
                        "agree": 0,
                        "reason": f"Secondary model error: {e}",
                        "primary_json": json.dumps(primary_json, ensure_ascii=False),
                        "secondary_json": "",
                    }
                )
                continue

            # Write JSONL outputs
            if f_primary is not None:
                f_primary.write(json.dumps({"key": key, "extraction": primary_json}, ensure_ascii=False) + "\n")
            if f_secondary is not None:
                f_secondary.write(json.dumps({"key": key, "extraction": secondary_json}, ensure_ascii=False) + "\n")

            # Compare
            agree, reason = compare_extractions(primary_json, secondary_json)
            rows_for_tsv.append(
                {
                    "key": key,
                    "agree": 1 if agree else 0,
                    "reason": reason,
                    "primary_json": json.dumps(primary_json, ensure_ascii=False),
                    "secondary_json": json.dumps(secondary_json, ensure_ascii=False),
                }
            )

    finally:
        if f_primary is not None:
            f_primary.close()
        if f_secondary is not None:
            f_secondary.close()

    # Save TSV summary
    out_df = pd.DataFrame(rows_for_tsv)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    log_info(f"Saved agreement summary to: {out_tsv}")
    if out_jsonl_primary:
        log_info(f"Saved primary extractions to: {out_jsonl_primary}")
    if out_jsonl_secondary:
        log_info(f"Saved secondary extractions to: {out_jsonl_secondary}")

    # Quick stats
    n_total = len(out_df)
    n_agree = int(out_df["agree"].sum())
    n_disagree = n_total - n_agree
    log_info(f"Total papers processed: {n_total}")
    log_info(f"Agree (auto-accepted): {n_agree}")
    log_info(f"Disagree (need manual review): {n_disagree}")


# -----------------------
# CLI
# -----------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-model extraction + cross-check for PLGA formulations (Gemini-based)."
    )
    p.add_argument(
        "--key2txt",
        required=True,
        type=Path,
        help="Path to key2txt.tsv mapping paper keys to cleaned text paths.",
    )
    p.add_argument(
        "--out-tsv",
        required=True,
        type=Path,
        help="Output TSV summarizing agreement and including both JSON outputs.",
    )
    p.add_argument(
        "--primary-model",
        default="gemini-1.5-flash",
        help="Gemini model name for primary extraction (default: gemini-1.5-flash).",
    )
    p.add_argument(
        "--secondary-model",
        default="gemini-1.5-flash-8b",
        help="Gemini model name for secondary extraction (default: gemini-1.5-flash-8b).",
    )
    p.add_argument(
        "--out-jsonl-primary",
        type=Path,
        default=None,
        help="Optional: path to write primary extraction JSONL.",
    )
    p.add_argument(
        "--out-jsonl-secondary",
        type=Path,
        default=None,
        help="Optional: path to write secondary extraction JSONL.",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional: limit number of papers for debugging (e.g., 10).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    configure_gemini_from_env()

    run_multimodel_extraction(
        key2txt_path=args.key2txt,
        out_tsv=args.out_tsv,
        primary_model=args.primary_model,
        secondary_model=args.secondary_model,
        out_jsonl_primary=args.out_jsonl_primary,
        out_jsonl_secondary=args.out_jsonl_secondary,
        max_papers=args.max_papers,
    )


if __name__ == "__main__":
    main()
