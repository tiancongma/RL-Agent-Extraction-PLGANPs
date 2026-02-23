#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report schema_v2 vs schema_v3 core row counts per DOI."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-tsv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    base = Path(f"data/results/{run_id}/benchmark_goren_2025")
    p_v2 = base / "schema_v2/formulation_core.tsv"
    p_v3 = base / "schema_v3/formulation_core.tsv"
    out = Path(args.out_tsv) if args.out_tsv else base / "schema_v3/schema_v2_vs_v3_core_counts.tsv"

    missing = [str(p) for p in [p_v2, p_v3] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    v2 = pd.read_csv(p_v2, sep="\t", dtype=str).fillna("")
    v3 = pd.read_csv(p_v3, sep="\t", dtype=str).fillna("")
    v2["reference_normalized_doi"] = v2["reference_normalized_doi"].str.strip().str.lower()
    v3["reference_normalized_doi"] = v3["reference_normalized_doi"].str.strip().str.lower()

    c2 = v2.groupby("reference_normalized_doi", dropna=False).size().reset_index(name="v2_core_rows")
    c3 = v3.groupby("reference_normalized_doi", dropna=False).size().reset_index(name="v3_core_rows")
    diff = c2.merge(c3, on="reference_normalized_doi", how="outer").fillna(0)
    diff["v2_core_rows"] = diff["v2_core_rows"].astype(int)
    diff["v3_core_rows"] = diff["v3_core_rows"].astype(int)
    diff["delta_v3_minus_v2"] = diff["v3_core_rows"] - diff["v2_core_rows"]
    diff = diff.sort_values(["delta_v3_minus_v2", "reference_normalized_doi"], ascending=[False, True]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    diff.to_csv(out, sep="\t", index=False)
    print(f"output_diff={out}")
    print(diff.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
