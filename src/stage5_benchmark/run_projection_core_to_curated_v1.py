#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project schema_v1 formulation_core rows into curated column schema."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--core-tsv",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/schema_v1/formulation_core.tsv",
    )
    parser.add_argument(
        "--curated-template",
        default="data/benchmark/goren_2025/NP_dataset_formulations.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/core_eval_v1",
    )
    return parser.parse_args()


def norm_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
    except ValueError:
        return s
    if f.is_integer():
        return str(int(f))
    return f"{f:.6g}"


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/{run_id}/benchmark_goren_2025/core_eval_v1")
    core_path = Path(args.core_tsv) if args.core_tsv else Path(
        f"data/results/{run_id}/benchmark_goren_2025/schema_v1/formulation_core.tsv"
    )
    curated_template = Path(args.curated_template)

    missing = [str(p) for p in [core_path, curated_template] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    core = pd.read_csv(core_path, sep="\t", dtype=str).fillna("")
    curated_cols = pd.read_csv(curated_template, nrows=0).columns.tolist()

    out_dir.mkdir(parents=True, exist_ok=True)
    projected_rows: list[dict[str, str]] = []
    trace_rows: list[dict[str, str]] = []

    for _, r in core.iterrows():
        row = {c: "" for c in curated_cols}
        mapped_fields: list[str] = []
        missing_fields: list[str] = []

        # Core mapping to curated columns.
        mapping = {
            "reference": str(r.get("reference_normalized_doi", "")),
            "small_molecule_name": str(r.get("drug_name", "")),
            "solvent": str(r.get("organic_solvent", "")),
            "LA/GA": norm_num(r.get("la_ga_ratio", "")),
            "drug/polymer": norm_num(r.get("drug_to_polymer_mass_ratio", "")),
        }

        # polymer_MW can be single or range.
        mw_lo = norm_num(r.get("polymer_mw_lower_kDa", ""))
        mw_hi = norm_num(r.get("polymer_mw_upper_kDa", ""))
        if mw_lo and mw_hi:
            mapping["polymer_MW"] = mw_lo if mw_lo == mw_hi else f"{mw_lo}-{mw_hi}"
        elif mw_lo or mw_hi:
            mapping["polymer_MW"] = mw_lo or mw_hi
        else:
            mapping["polymer_MW"] = ""

        for col in curated_cols:
            if col in mapping:
                row[col] = mapping[col]
                if str(mapping[col]).strip():
                    mapped_fields.append(col)
                else:
                    missing_fields.append(col)
            else:
                missing_fields.append(col)

        projected_rows.append(row)
        trace_rows.append(
            {
                "formulation_core_id": str(r.get("formulation_core_id", "")),
                "reference_normalized_doi": str(r.get("reference_normalized_doi", "")),
                "core_signature": str(r.get("core_signature", "")),
                "mapped_fields": ",".join(mapped_fields),
                "missing_fields": ",".join(missing_fields),
            }
        )

    projected_df = pd.DataFrame(projected_rows, columns=curated_cols)
    trace_df = pd.DataFrame(trace_rows)

    projected_path = out_dir / "core_projected_to_curated.tsv"
    trace_path = out_dir / "core_projection_trace.tsv"
    projected_df.to_csv(projected_path, sep="\t", index=False)
    trace_df.to_csv(trace_path, sep="\t", index=False)

    header_match = projected_df.columns.tolist() == curated_cols
    print(f"core_rows={len(projected_df)}")
    print(f"header_match_curated={header_match}")
    print(f"output_projected={projected_path}")
    print(f"output_trace={trace_path}")


if __name__ == "__main__":
    main()
