"""
classify_gemini_grouped.py — readable debug + strict signals + tunable threshold

Screens Title+Abstract with Gemini in SMALL groups.
Forces JSON-only outputs, applies local thresholding to decide Download/Skip.

Quick start:
  python scripts/classify_gemini_grouped.py --group 6 --log DEBUG

Typical tuning:
  # stricter: require 2 numeric evidences
  python scripts/classify_gemini_grouped.py --min_numeric_hits 2

  # looser: allow nanoprecipitation (treat as not-excluded)
  python scripts/classify_gemini_grouped.py --allow_nanoprecipitation

Env:
  pip install google-generativeai python-dotenv pandas
  .env must contain GEMINI_API_KEY=<your_key>  (or set GOOGLE_API_KEY)
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import argparse
import logging
from pathlib import Path
from time import monotonic
from typing import Optional, List, Dict, Tuple

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------- Defaults ---------------------------- #

DEFAULT_MODEL = "gemini-2.5-flash-lite"

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IN_PATH = BASE_DIR / "Data" / "wos_prefiltered.csv"
DEFAULT_OUT_PATH = BASE_DIR / "Data" / "wos_llm_tagged.csv"

PROMPT_HDR = """You are a strict metadata screener for PLGA emulsion-based nanoparticle papers.

Return ONLY a JSON array (no prose, no markdown, no code fences).
For each paper, extract binary/float signals from Title+Abstract and propose a decision.
We will compute the final decision locally, but your proposal helps calibration.

Rules (strict):
- must have PLGA mention.
- must have EMULSION-type preparation (e.g., "double emulsion", "W1/O/W2", "W/O/W", "solvent evaporation").
- we prefer explicit numeric evidence in the abstract.
- EXCLUDE (skip) if review/protocol-only/surface-only; EXCLUDE if it is clearly non-emulsion
  (e.g., nanoprecipitation/microfluidic-only without emulsion).

Return a JSON array of objects with this exact schema:
[
  {
    "row": <original_row_index>,
    "signals": {
      "has_plga": true|false,
      "has_emulsion": true|false,
      "has_phase_terms": true|false,         // W1/O/W2, inner/outer water/oil phases
      "has_surfactant_info": true|false,     // PVA/Tween/etc (optionally with %)
      "has_numeric_size": true|false,        // nm mentioned
      "has_numeric_pdi": true|false,
      "has_numeric_zeta": true|false,        // mV mentioned
      "has_numeric_ee": true|false,          // encapsulation efficiency % mentioned
      "is_review": true|false,
      "is_protocol_only": true|false,
      "surface_only": true|false,
      "non_emulsion": true|false             // nanoprecipitation/microfluidic-only without emulsion
    },
    "llm_proposal": "Download" | "Skip"
  }
]

Guidance:
- Be conservative: if in doubt, set the signal to false and default llm_proposal to "Skip".
- Only trust what’s in Title/Abstract; do not infer full-text details.
"""

ITEM_FMT = "\n### Paper row={row}\nTitle: {title}\nAbstract: {abst}\n"
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", re.DOTALL)

# ---------------------------- Logging ----------------------------- #

def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("gemini_batch")


# --------------------------- Utilities ---------------------------- #

def detect_cols(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """Detect title/abstract columns with case-insensitive fallbacks."""
    title_cands = ["Title", "title"]
    abst_cands  = ["Abstract", "abstract", "Abstract Note", "abstractNote", "摘要"]

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        lowmap = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in lowmap:
                return lowmap[c.lower()]
        return None

    title = pick(title_cands)
    abst  = pick(abst_cands)
    if not title:
        raise ValueError(f"Title column not found. Columns: {list(df.columns)}")
    return title, abst


def load_model(model_name: str):
    """Configure Gemini and return a model with JSON-only output."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=2048,             # room for arrays
            response_mime_type="application/json",
        ),
    )


def approx_tokens(s: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    return max(1, math.ceil(len(s) / 4))


def strip_code_fence(txt: str) -> str:
    s = txt.strip()
    m = _CODE_FENCE_RE.match(s)
    if m:
        return m.group(1).strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    return s


def extract_json_array(txt: str) -> Optional[str]:
    """Best-effort extract of the first complete JSON array from noisy text."""
    s = txt.strip()
    start = s.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


def parse_json_array(txt: str) -> List[Dict]:
    """Parse JSON array with fallbacks (noisy fences, prefixes, suffixes)."""
    s = strip_code_fence(txt)
    try:
        return json.loads(s)
    except Exception:
        arr = extract_json_array(s)
        if arr is not None:
            return json.loads(arr)
        raise


def chunk_ranges(n: int, k: int):
    """Yield [start, end) ranges of size k."""
    i = 0
    while i < n:
        j = min(i + k, n)
        yield i, j
        i = j


# --------------------- Local decision (tunable) -------------------- #

def decide_download(
    signals: Dict,
    *,
    min_numeric_hits: int = 1,
    min_bonus_for_no_numeric: int = 2,
    allow_nanoprecipitation: bool = False,
) -> str:
    """
    Return 'Relevant' (Download) or 'Irrelevant' (Skip).
    - Must-have: has_plga AND has_emulsion.
    - Hard excludes: review/protocol-only/surface-only OR (non_emulsion unless override).
    - Numeric signals: size/pdi/zeta/ee. Need at least `min_numeric_hits`.
    - If numeric==0, allow passing only if bonus signals >= min_bonus_for_no_numeric
      (bonus = surfactant_info + phase_terms).
    """
    # Hard excludes
    if signals.get("is_review") or signals.get("is_protocol_only") or signals.get("surface_only"):
        return "Irrelevant"

    if signals.get("non_emulsion"):
        if not allow_nanoprecipitation:
            return "Irrelevant"

    # Must-haves
    if not (signals.get("has_plga") and signals.get("has_emulsion")):
        return "Irrelevant"

    # Numeric evidence
    numeric_hits = sum(bool(signals.get(k)) for k in [
        "has_numeric_size", "has_numeric_pdi", "has_numeric_zeta", "has_numeric_ee"
    ])

    # Bonus cues
    bonus = sum(bool(signals.get(k)) for k in ["has_surfactant_info", "has_phase_terms"])

    if numeric_hits >= min_numeric_hits:
        return "Relevant"
    if numeric_hits == 0 and bonus >= min_bonus_for_no_numeric:
        return "Relevant"

    return "Irrelevant"


# ----------------------------- Main ------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", help="Input CSV (prefiltered).")
    ap.add_argument("--out", dest="out_path", help="Output CSV path.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    ap.add_argument("--group", type=int, default=6, help="Papers per request.")
    ap.add_argument("--max_retries", type=int, default=3, help="Retries per request.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep after each request (seconds).")
    ap.add_argument("--rpm_limit", type=int, default=15, help="Requests/min hard cap.")
    ap.add_argument("--tpm_soft_cap", type=int, default=180_000, help="Soft tokens/min cap.")
    ap.add_argument("--abst_chars", type=int, default=3000, help="Abstract char limit (truncate).")
    ap.add_argument("--log", default="INFO", help="Log level: DEBUG|INFO|WARNING|ERROR.")

    # New tuning knobs
    ap.add_argument("--min_numeric_hits", type=int, default=2,
                    help="Require at least this many numeric signals (size/pdi/zeta/ee).")
    ap.add_argument("--min_bonus_for_no_numeric", type=int, default=50,
                    help="If numeric hits=0, require this many bonus cues (surfactant+phase) to pass.")
    ap.add_argument("--allow_nanoprecipitation", action="store_true",
                    help="Treat non_emulsion (e.g., nanoprecipitation-only) as allowed (not excluded).")

    args = ap.parse_args()
    log = setup_logging(args.log)

    in_path = Path(args.in_path) if args.in_path else DEFAULT_IN_PATH
    out_path = Path(args.out_path) if args.out_path else DEFAULT_OUT_PATH

    log.info(f"Input : {in_path}")
    log.info(f"Output: {out_path}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    n = len(df)
    if n == 0:
        log.warning("Input CSV has 0 rows. Writing passthrough and exiting.")
        out_df = df.copy()
        out_df["AI_Tag"] = []
        out_df["AI_Source"] = args.model
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    title_col, abst_col = detect_cols(df)
    model = load_model(args.model)

    labels: List[str] = [""] * n

    # Pacing state
    min_interval = 60.0 / max(1, args.rpm_limit)
    last_req_ts = 0.0

    win_start = monotonic()
    win_tokens = 0

    processed = 0

    for s, e in chunk_ranges(n, args.group):
        subset = df.iloc[s:e]

        # Build prompt
        prompt = PROMPT_HDR
        for row_idx, row in subset.iterrows():
            title = str(row.get(title_col, "") or "")
            abst  = str(row.get(abst_col, "") or "")
            if args.abst_chars > 0 and len(abst) > args.abst_chars:
                abst = abst[:args.abst_chars] + " ..."
            prompt += ITEM_FMT.format(row=row_idx, title=title, abst=abst)

        # RPM pacing
        now = monotonic()
        since_last = now - last_req_ts
        if since_last < min_interval:
            time.sleep(min_interval - since_last)

        # TPM pacing (rolling 60s window)
        est_tokens = approx_tokens(prompt) + 128  # small allowance for output
        now = monotonic()
        if now - win_start >= 60.0:
            win_start = now
            win_tokens = 0
        if win_tokens + est_tokens > args.tpm_soft_cap:
            wait = 60.0 - (now - win_start)
            if wait > 0:
                log.debug(f"TPM soft cap hit; sleeping {wait:.1f}s")
                time.sleep(wait)
            win_start = monotonic()
            win_tokens = 0

        # Request with retries
        resp_text = None
        last_err = ""
        backoff = 1.5
        for attempt in range(args.max_retries + 1):
            try:
                resp = model.generate_content(prompt)
                text = (getattr(resp, "text", "") or "").strip()
                if not text and getattr(resp, "candidates", None):
                    c0 = resp.candidates[0]
                    if getattr(c0, "content", None) and c0.content.parts:
                        text = c0.content.parts[0].text.strip()
                if not text:
                    raise RuntimeError("Empty response text")

                # Validate parseability once here
                _ = parse_json_array(text)
                resp_text = text
                break
            except Exception as exc:
                last_err = str(exc)
                lmsg = last_err.lower()
                transient = any(x in lmsg for x in [
                    "429", "resourceexhausted", "rate", "quota", "temporar",
                    "deadline", "timeout", "timed out", "unavailable", "503", "retry"
                ])
                if attempt < args.max_retries and transient:
                    log.warning(f"Transient error (attempt {attempt+1}/{args.max_retries}): {last_err}. "
                                f"Backoff {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if attempt < args.max_retries:
                    wait = 1.2 * (attempt + 1)
                    log.warning(f"Error (attempt {attempt+1}/{args.max_retries}): {last_err}. Sleep {wait:.1f}s")
                    time.sleep(wait)
                    continue
                # Give up this chunk; mark its rows as Error
                for row_idx in subset.index:
                    labels[row_idx] = "Error"
                resp_text = None

        last_req_ts = monotonic()
        win_tokens += est_tokens

        # Parse + apply labels with local thresholding
        if resp_text:
            try:
                arr = parse_json_array(resp_text)
                for obj in arr:
                    r = obj.get("row")
                    signals = obj.get("signals") or {}
                    lab = decide_download(
                        signals,
                        min_numeric_hits=args.min_numeric_hits,
                        min_bonus_for_no_numeric=args.min_bonus_for_no_numeric,
                        allow_nanoprecipitation=args.allow_nanoprecipitation,
                    )
                    if isinstance(r, int) and 0 <= r < n:
                        labels[r] = lab

                # Fill any missing rows in this chunk
                for row_idx in subset.index:
                    if not labels[row_idx]:
                        labels[row_idx] = "Irrelevant"
            except Exception as pe:
                for row_idx in subset.index:
                    labels[row_idx] = "Error"
                head = (resp_text or "")[:300].replace("\n", " ")
                logging.error(f"Parse error rows {s}-{e}: {pe} | RAW_HEAD: {head}")
        else:
            logging.error(f"Request error rows {s}-{e}: {last_err}")

        processed += 1
        if e < n and args.sleep > 0:
            time.sleep(args.sleep)

        logging.info(f"LLM requests: {processed} | rows covered: {e}/{n}")

    # Write output
    out_df = df.copy()
    out_df["AI_Tag"] = labels
    out_df["AI_Source"] = args.model
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    stats = out_df["AI_Tag"].value_counts(dropna=False).to_dict()
    logging.info(f"Done → {out_path} | tag distribution: {stats}")


if __name__ == "__main__":
    main()
