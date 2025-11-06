#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
gt_tool_v3.py  — robust template maker for weak-labels (JSONL or TSV)

Minimal example (auto-detect input type):
    python .\scripts\gt_tool_v3.py make-template ^
      --weak data\cleaned\samples\weak_labels_v3.jsonl ^
      --out  data\cleaned\samples\manual_labels_v3.tsv ^
      --tiers all ^
      --verbose

If your weak labels are TSV:
    python .\scripts\gt_tool_v3.py make-template ^
      --weak data\cleaned\samples\weak_labels_v3.tsv ^
      --input-type tsv ^
      --out  data\cleaned\samples\manual_labels_v3.tsv ^
      --verbose
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

PRED_COLS_CANDIDATES = [
    "plga_mw", "la_ga_ratio", "w1_w2", "organic_solvent",
    "emulsification_energy", "pva_conc", "polymer_mass_mg", "drug_feed_mg",
    "particle_size_nm", "pdi", "zeta_mV", "drug_loading_pct",
    "encapsulation_eff_pct", "release_temp_condition", "notes", "raw_json",
]
REQUIRED_META = ["key"]

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                # blank line
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                print(f"[WARN] skip bad JSONL line {i}: {e}")
    return rows

def sniff_is_tsv(path: Path, max_lines: int = 3) -> bool:
    # Heuristic: a .jsonl file that actually contains tabs/headers
    cnt = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            cnt += 1
            if "\t" in line:
                return True
            if cnt >= max_lines:
                break
    return False

def load_weak(path: Path, input_type: str) -> pd.DataFrame:
    if input_type == "auto":
        if path.suffix.lower() == ".tsv":
            input_type = "tsv"
        elif path.suffix.lower() in (".jsonl", ".json"):
            # sniff if it's actually a TSV mislabeled as jsonl
            input_type = "tsv" if sniff_is_tsv(path) else "jsonl"
        else:
            # default try jsonl first
            input_type = "jsonl"

    if input_type == "jsonl":
        rows = read_jsonl(path)
        if not rows:
            raise SystemExit(f"[ERROR] No valid JSON rows loaded from {path}")
        df = pd.DataFrame(rows)
    elif input_type == "tsv":
        df = pd.read_csv(path, sep="\t", dtype=str)
    else:
        raise SystemExit(f"[ERROR] Unknown input_type: {input_type}")

    # basic normalize
    for col in ["key", "tier"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "key" not in df.columns:
        raise SystemExit("[ERROR] weak labels must contain 'key' column")

    # ensure tier exists (default '1')
    if "tier" not in df.columns:
        df["tier"] = "1"

    return df

def select_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    meta_cols = [c for c in REQUIRED_META if c in df.columns] + ["tier"]
    pred_cols = [c for c in PRED_COLS_CANDIDATES if c in df.columns]
    # always put raw_json at end if present
    if "raw_json" in pred_cols:
        pred_cols = [c for c in pred_cols if c != "raw_json"] + ["raw_json"]
    return meta_cols, pred_cols

def make_template(df: pd.DataFrame, tiers: List[str]) -> pd.DataFrame:
    if tiers and (tiers != ["all"]):
        df = df[df["tier"].isin(tiers)].copy()
        kept = len(df)
        total = len(df.index)
        print(f"[INFO] Tier filter {tiers}: kept {kept}/{total}")
    else:
        print(f"[INFO] Tier filter [all]: kept {len(df)}/{len(df)}")

    meta_cols, pred_cols = select_cols(df)
    # Build output columns: meta + pred_* + gt_* (parallel)
    out_cols = meta_cols + [f"pred_{c}" for c in pred_cols if c != "raw_json"] + [f"gt_{c}" for c in pred_cols if c != "raw_json"] + (["pred_raw_json"] if "raw_json" in pred_cols else [])

    rows = []
    for _, r in df.iterrows():
        row = {}
        for m in meta_cols:
            row[m] = r.get(m, "")
        for c in pred_cols:
            if c == "raw_json":
                row["pred_raw_json"] = r.get("raw_json", "")
            else:
                row[f"pred_{c}"] = r.get(c, "")
                row[f"gt_{c}"] = ""  # empty for manual fill
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=out_cols)
    return out_df

def main():
    ap = argparse.ArgumentParser(description="Ground-truth tool (template maker).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make-template", help="Create manual label template from weak labels")
    mk.add_argument("--weak", required=True, help="Path to weak labels (JSONL or TSV)")
    mk.add_argument("--input-type", choices=["auto", "jsonl", "tsv"], default="auto", help="Force input type (default auto-detect)")
    mk.add_argument("--tiers", default="1,2", help="Comma-separated tiers to keep, or 'all'")
    mk.add_argument("--out", required=True, help="Output TSV path for manual labels")
    mk.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.cmd == "make-template":
        weak_path = Path(args.weak)
        out_path  = Path(args.out)
        tiers = [t.strip() for t in args.tiers.split(",")] if args.tiers.lower() != "all" else ["all"]

        if args.verbose:
            print(f"[INFO] Loading weak labels: {weak_path} (input_type={args.input_type})")

        df = load_weak(weak_path, input_type=args.input_type)

        if args.verbose:
            print(f"[INFO] Loaded {len(df)} weak-label rows from {weak_path}")

        tmpl = make_template(df, tiers)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmpl.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

        meta_cols, pred_cols = select_cols(df)
        print(f"[OK] Template written: {out_path}")
        print(f"[INFO] Rows={len(tmpl)}, pred_cols={len([c for c in pred_cols if c!='raw_json'])}, meta_cols={len(meta_cols)}")

if __name__ == "__main__":
    main()
