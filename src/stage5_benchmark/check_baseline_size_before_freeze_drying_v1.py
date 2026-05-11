#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regression check for size_nm__baseline_before_freeze_drying derivation on WIVUCMYG."
    )
    p.add_argument("--derived-tsv", required=True, help="Path to derived_values.tsv")
    p.add_argument("--key", default="WIVUCMYG")
    p.add_argument("--expected", type=float, default=324.30)
    p.add_argument("--tol", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.derived_tsv)
    if not path.exists():
        raise FileNotFoundError(f"Missing derived TSV: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    sub = df[
        (df["key"].astype(str).str.strip() == str(args.key))
        & (df["field_name"].astype(str).str.strip() == "size_nm__baseline_before_freeze_drying")
        & (df["rule_id"].astype(str).str.strip() == "baseline_size_before_freeze_drying_v1")
    ].copy()
    if sub.empty:
        raise RuntimeError("No baseline before-freeze-drying rows found for target key.")
    sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub[sub["value_num"].notna()].copy()
    if sub.empty:
        raise RuntimeError("Rows found, but numeric value parsing failed.")
    hit = sub[(sub["value_num"] - float(args.expected)).abs() <= float(args.tol)]
    print(sub[["key", "formulation_id", "field_name", "value", "rule_id", "value_source"]].to_string(index=False))
    if hit.empty:
        raise RuntimeError(
            f"Expected value near {args.expected} +/- {args.tol} not found for key={args.key}."
        )
    print(
        f"PASS key={args.key} field=size_nm__baseline_before_freeze_drying "
        f"expected~={args.expected} tol={args.tol}"
    )


if __name__ == "__main__":
    main()
