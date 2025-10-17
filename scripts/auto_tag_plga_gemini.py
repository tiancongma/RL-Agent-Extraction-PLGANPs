"""
auto_tag_plga_gemini.py

Purpose:
  - Auto-screen literature (Title + Abstract) and label each record as "Relevant" or "Irrelevant"
    for PLGA emulsion-based nanoparticle formulations.

Key features:
  - Uses Google Gemini API (google-generativeai).
  - Loads API key from .env (kept out of Git).
  - Robust column detection (Title/Abstract).
  - Batching, simple rate limiting, minimal retries.
  - Writes tagged CSV + prints tag distribution.

Usage:
  python auto_tag_plga_gemini.py --in ../Data/wos_all.csv --out ../Data/wos_all_tagged_gemini.csv --model gemini-2.0-flash

Requirements:
  pip install pandas python-dotenv google-generativeai>=0.7.0
"""

import os
import time
import argparse
import pandas as pd
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai

DEFAULT_MODEL = "gemini-2.0-flash"
ENV_VAR_NAME = "GEMINI_API_KEY"

PROMPT_TEMPLATE = """You are screening literature for a dataset on PLGA emulsion-based nanoparticle formulations.

Task:
Given a short text consisting of a paper title and abstract, decide whether the paper is relevant to:
  - PLGA (poly(lactic-co-glycolic acid)) nanoparticles/microspheres/nanocarriers AND
  - Emulsion-related preparation (e.g., double emulsion, W1/O/W2, solvent evaporation) OR nanoprecipitation
  - AND/OR reports particle characterization (e.g., particle size, zeta potential, encapsulation efficiency, PDI).

Output strictly one token:
  Relevant
or
  Irrelevant

Text:
---
{TEXT}
---
Answer:"""

def load_api_client(model_name: str):
    """
    Initialize Gemini with API key from .env and return a GenerativeModel instance.
    """
    load_dotenv()  # loads GEMINI_API_KEY from .env
    api_key = os.getenv(ENV_VAR_NAME)
    if not api_key:
        raise SystemExit(
            f"{ENV_VAR_NAME} is not set. Create a .env with {ENV_VAR_NAME}=... and ensure it is not committed to Git."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def detect_columns(df: pd.DataFrame) -> tuple[str, Optional[str]]:
    """
    Detect title and abstract column names from a Zotero/Web of Science CSV export.
    Returns (title_col, abstract_col). abstract_col may be None if not found.
    """
    cand_title = ["Title", "title"]
    cand_abs = ["Abstract", "abstract", "Abstract Note", "abstractNote", "摘要"]

    title_col = None
    abs_col = None
    lower_map = {c.lower(): c for c in df.columns}

    for c in cand_title:
        if c in df.columns or c.lower() in lower_map:
            title_col = c if c in df.columns else lower_map[c.lower()]
            break
    for c in cand_abs:
        if c in df.columns or c.lower() in lower_map:
            abs_col = c if c in df.columns else lower_map[c.lower()]
            break

    if not title_col:
        raise ValueError(f"Could not find a Title column. Columns: {list(df.columns)}")
    return title_col, abs_col

def build_text(title: str, abstract: Optional[str], max_len: int = 2000) -> str:
    """
    Concatenate title and abstract, truncate to max_len characters to control latency/cost.
    """
    title = title or ""
    abstract = abstract or ""
    text = f"Title: {title}\nAbstract: {abstract}"
    return text[:max_len] if len(text) > max_len else text

def call_gemini(model, text: str, retries: int = 2, delay: float = 1.5) -> str:
    """
    Call Gemini with minimal retries. Return "Relevant" or "Irrelevant".
    Fallback is "Irrelevant" if the output deviates.
    """
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(PROMPT_TEMPLATE.format(TEXT=text))
            ans = (resp.text or "").strip()
            # Normalize to the expected labels
            first = ans.split()[0].capitalize() if ans else "Irrelevant"
            if first not in {"Relevant", "Irrelevant"}:
                first = "Irrelevant"
            return first
        except Exception as e:
            if attempt < retries:
                time.sleep(delay * (attempt + 1))
            else:
                return "Error"

def process_csv(
    in_path: str,
    out_path: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 100,
    sleep_between_batches: float = 1.0
) -> None:
    """
    Pipeline:
      - Read CSV
      - Detect columns
      - Build inputs
      - Classify in batches
      - Save tagged CSV
    """
    model = load_api_client(model_name)
    df = pd.read_csv(in_path, encoding="utf-8-sig")

    title_col, abs_col = detect_columns(df)

    # Build input texts
    inputs = []
    for _, row in df.iterrows():
        title = str(row.get(title_col, "") or "")
        abstract = str(row.get(abs_col, "") or "")
        inputs.append(build_text(title, abstract))
    df["AI_Input"] = inputs

    # Classify in batches
    tags = []
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df["AI_Input"].iloc[start:end].tolist()

        for text in batch:
            tag = call_gemini(model, text)
            tags.append(tag)

        if end < n:
            time.sleep(sleep_between_batches)

        print(f"Processed {end}/{n} rows...")

    df["AI_Tag"] = tags

    # Quick stats
    stats = df["AI_Tag"].value_counts(dropna=False).to_dict()
    print("Tag distribution:", stats)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Output saved to: {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Auto-tag PLGA emulsion literature with Gemini.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input CSV path (exported from Zotero/WoS).")
    parser.add_argument("--out", dest="out_path", required=True, help="Output CSV path with AI_Tag column.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--batch", type=int, default=100, help="Batch size for simple rate limiting.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between batches.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_csv(
        in_path=args.in_path,
        out_path=args.out_path,
        model_name=args.model,
        batch_size=args.batch,
        sleep_between_batches=args.sleep
    )
