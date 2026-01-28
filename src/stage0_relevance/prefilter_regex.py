"""
prefilter_regex.py

Purpose:
  Local, zero-cost keyword prefilter on Title+Abstract to reduce the candidate set
  BEFORE sending to Gemini. Keeps papers that contain PLGA mentions AND
  (emulsion/nanoprecipitation OR characterization terms).

Inputs (defaults):
  <project_root>/Data/wos_all.csv

Outputs (defaults):
  <project_root>/Data/wos_prefiltered.csv
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# Resolve paths relative to THIS script file (robust no matter where you run it)
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent           # .../scripts
PROJECT_ROOT = SCRIPT_DIR.parent                       # .../RL-Agent-Extraction-PLGANPs
DEFAULT_IN  = PROJECT_ROOT / "Data" / "wos_all.csv"
DEFAULT_OUT = PROJECT_ROOT / "Data" / "wos_prefiltered.csv"

# ---------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------
TITLE_COL_CAND = ["Title", "title"]
ABST_COL_CAND  = ["Abstract", "abstract", "Abstract Note", "abstractNote", "摘要"]

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Pick the first column name from candidates that exists in df (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

# ---------------------------------------------------------------------
# Core prefilter
# ---------------------------------------------------------------------
def run_prefilter(in_path: Path, out_path: Path) -> tuple[int, int, Path]:
    """Read input CSV, prefilter rows by Title+Abstract, write kept rows to out_path."""
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {in_path}\n"
            f"Tips:\n"
            f"  - Ensure the file exists and name matches exactly (not 'wos_all (1).csv').\n"
            f"  - Or pass an absolute path via --in."
        )

    df = pd.read_csv(in_path, encoding="utf-8-sig")

    title_col = pick_col(df, TITLE_COL_CAND)
    abst_col  = pick_col(df, ABST_COL_CAND)
    if not title_col:
        raise ValueError(f"Cannot find a Title column. Available columns: {list(df.columns)}")

    # Build text (lower-cased)
    title_series = df[title_col].fillna("").astype(str)
    if abst_col:
        abst_series = df[abst_col].fillna("").astype(str)
    else:
        abst_series = pd.Series([""] * len(df))
    text = (title_series + " " + abst_series).str.lower()

    # MUST: PLGA mention
    must_have = [
        r"\bplga\b",
        r"poly\(lactic-co-glycolic acid\)",
        r"poly \(lactic-co-glycolic acid\)"
    ]
    # AND one of: method OR properties (light filter to cut edge cases)
    method = [
        r"double emulsion", r"\bw1/o/w2\b", r"\bw/o/w\b",
        r"solvent evaporation", r"nanoprecipitation", r"\bprecipitation\b"
    ]
    props = [
        r"particle size", r"\bzeta\b", r"zeta potential",
        r"encapsulation efficiency", r"\bpdi\b"
    ]

    mask_plga = text.str.contains("|".join(must_have), regex=True)
    mask_meth_props = text.str.contains("|".join(method + props), regex=True)

    keep_mask = mask_plga & mask_meth_props
    kept = df[keep_mask].copy()

    # Ensure output folder exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False, encoding="utf-8-sig")

    return len(kept), len(df), out_path

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def resolve_path(arg_path: str | None, default_path: Path) -> Path:
    """If arg_path is None: return default; if relative: resolve against PROJECT_ROOT."""
    if arg_path is None:
        return default_path
    p = Path(arg_path)
    if p.is_absolute():
        return p
    # Treat as project-root relative if not absolute
    return PROJECT_ROOT / p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefilter PLGA papers by Title+Abstract keywords.")
    parser.add_argument("--in",  dest="in_path",  default=None,
                        help=f"Input CSV path (default: {DEFAULT_IN})")
    parser.add_argument("--out", dest="out_path", default=None,
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    in_path  = resolve_path(args.in_path,  DEFAULT_IN)
    out_path = resolve_path(args.out_path, DEFAULT_OUT)

    kept_n, total_n, out_p = run_prefilter(in_path, out_path)
    print(f"[Prefilter] Kept {kept_n} / {total_n} rows → {out_p}")
