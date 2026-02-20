#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def norm_doi(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    if not s:
        return ""
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def parse_float_list(cell: object) -> List[float]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    out: List[float] = []
    for part in s.split(";"):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="DOI-level EE set matching (evaluation-only).")
    parser.add_argument(
        "--extracted-formulation-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/extracted_formulation_level_v1.tsv",
    )
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    extracted_p = (cwd / args.extracted_formulation_tsv).resolve()
    curated_p = (cwd / args.curated_tsv).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(extracted_p, sep="\t", dtype=str)
    cu = pd.read_csv(curated_p, sep="\t", dtype=str)

    # Curated DOI -> EE set
    if "doi_norm" in cu.columns:
        cu["doi_norm"] = cu["doi_norm"].map(norm_doi)
    elif "reference" in cu.columns:
        cu["doi_norm"] = cu["reference"].map(norm_doi)
    else:
        raise RuntimeError("Curated TSV missing DOI column.")

    ee_col_curated = "EE" if "EE" in cu.columns else "encapsulation_efficiency_percent"
    if ee_col_curated not in cu.columns:
        raise RuntimeError("Curated TSV missing EE column.")
    cu["ee_num"] = pd.to_numeric(cu[ee_col_curated], errors="coerce")

    curated_map: Dict[str, List[float]] = {}
    for doi, g in cu.groupby("doi_norm", dropna=False):
        vals = sorted({float(v) for v in g["ee_num"].dropna().tolist()})
        curated_map[str(doi)] = vals

    # Extracted DOI -> EE set from grouped table
    if "doi_norm" not in ex.columns:
        raise RuntimeError("Extracted formulation TSV missing doi_norm.")
    ex["doi_norm"] = ex["doi_norm"].map(norm_doi)

    extracted_map: Dict[str, List[float]] = {}
    if "unique_ee_values" in ex.columns:
        for doi, g in ex.groupby("doi_norm", dropna=False):
            vals: List[float] = []
            for cell in g["unique_ee_values"].tolist():
                vals.extend(parse_float_list(cell))
            extracted_map[str(doi)] = sorted(set(vals))
    else:
        col = "group_mean_EE" if "group_mean_EE" in ex.columns else None
        if col is None:
            raise RuntimeError("Extracted formulation TSV missing EE value columns.")
        for doi, g in ex.groupby("doi_norm", dropna=False):
            vals = pd.to_numeric(g[col], errors="coerce").dropna().astype(float).tolist()
            extracted_map[str(doi)] = sorted(set(vals))

    all_doi = sorted(set(curated_map.keys()) | set(extracted_map.keys()))
    rows: List[Dict[str, object]] = []
    improved_n = 0
    for doi in all_doi:
        c_vals = curated_map.get(doi, [])
        e_vals = extracted_map.get(doi, [])

        min_per_curated: List[float] = []
        if c_vals and e_vals:
            for cv in c_vals:
                min_per_curated.append(min(abs(cv - ev) for ev in e_vals))
            min_overall = min(min_per_curated) if min_per_curated else float("nan")
            any_le5 = any(d <= 5 for d in min_per_curated)
            any_le10 = any(d <= 10 for d in min_per_curated)
            # baseline: old DOI-mean absolute difference threshold
            mean_abs_diff = abs(sum(c_vals) / len(c_vals) - sum(e_vals) / len(e_vals))
            baseline_le5 = mean_abs_diff <= 5
            improved = any_le5 and (not baseline_le5)
        else:
            min_overall = float("nan")
            any_le5 = False
            any_le10 = False
            mean_abs_diff = float("nan")
            baseline_le5 = False
            improved = False

        if improved:
            improved_n += 1

        rows.append(
            {
                "doi_norm": doi,
                "n_curated_ee": len(c_vals),
                "n_extracted_ee": len(e_vals),
                "min_abs_diff_overall": min_overall,
                "any_match_le_5": any_le5,
                "any_match_le_10": any_le10,
                "mean_abs_diff_baseline": mean_abs_diff,
                "baseline_mean_le_5": baseline_le5,
                "improved_under_set_level": improved,
            }
        )

    doi_df = pd.DataFrame(rows)
    doi_out = out_dir / "doi_level_set_match.tsv"
    doi_df.to_csv(doi_out, sep="\t", index=False)

    n_total = int(len(doi_df))
    n_le5 = int(doi_df["any_match_le_5"].sum())
    n_le10 = int(doi_df["any_match_le_10"].sum())
    summary = pd.DataFrame(
        [
            {"metric": "n_doi_total", "value": n_total},
            {"metric": "n_doi_with_any_match_le_5", "value": n_le5},
            {"metric": "n_doi_with_any_match_le_10", "value": n_le10},
            {"metric": "proportion_le_5", "value": (n_le5 / n_total) if n_total else float("nan")},
            {"metric": "proportion_le_10", "value": (n_le10 / n_total) if n_total else float("nan")},
        ]
    )
    summary_out = out_dir / "set_match_summary.tsv"
    summary.to_csv(summary_out, sep="\t", index=False)

    print(f"DOIs improved under set-level definition: {improved_n}")


if __name__ == "__main__":
    main()
