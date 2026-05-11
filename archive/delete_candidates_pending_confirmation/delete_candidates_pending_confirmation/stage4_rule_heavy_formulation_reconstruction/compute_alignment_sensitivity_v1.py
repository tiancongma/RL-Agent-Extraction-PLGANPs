#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def clean_val(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def run_level(cu: pd.DataFrame, ex: pd.DataFrame, fields: Sequence[str]) -> Tuple[float, float]:
    matched_extracted_keys: set[Tuple[str, str]] = set()
    matched_curated = 0
    total_curated = len(cu)

    for doi, cu_d in cu.groupby("doi_norm", dropna=False):
        doi = str(doi)
        ex_d = ex[ex["doi_norm"] == doi].copy()
        for c_row in cu_d.itertuples(index=False):
            c = c_row._asdict()
            c_ee = c.get("ee_num")
            is_matched = False
            if pd.notna(c_ee) and len(ex_d) > 0:
                cand = ex_d.copy()
                for f in fields:
                    cv = c.get(f"_k_{f}", "")
                    if cv:
                        cand = cand[cand[f"_k_{f}"] == cv]
                if len(cand) > 0:
                    cand = cand[cand["ee_num"].notna()].copy()
                    if len(cand) > 0:
                        cand["abs_diff"] = (cand["ee_num"] - float(c_ee)).abs()
                        cand = cand[cand["abs_diff"] <= 5.0]
                        if len(cand) > 0:
                            best = cand.sort_values("abs_diff", ascending=True).iloc[0]
                            sig = str(best.get("formulation_signature", ""))
                            matched_extracted_keys.add((doi, sig))
                            is_matched = True
            if is_matched:
                matched_curated += 1

    total_extracted = len(ex)
    matched_extracted = len(matched_extracted_keys)
    recall = (matched_curated / total_curated) if total_curated else float("nan")
    precision = (matched_extracted / total_extracted) if total_extracted else float("nan")
    return recall, precision


def main() -> None:
    parser = argparse.ArgumentParser(description="Formulation alignment sensitivity analysis.")
    parser.add_argument(
        "--extracted-formulation-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/extracted_formulation_level_v1.tsv",
    )
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--out-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/alignment_sensitivity.tsv",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    extracted_p = (cwd / args.extracted_formulation_tsv).resolve()
    curated_p = (cwd / args.curated_tsv).resolve()
    out_p = (cwd / args.out_tsv).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(extracted_p, sep="\t", dtype=str)
    cu = pd.read_csv(curated_p, sep="\t", dtype=str)

    cu["doi_norm"] = cu["doi_norm"].map(norm_doi) if "doi_norm" in cu.columns else cu["reference"].map(norm_doi)
    ex["doi_norm"] = ex["doi_norm"].map(norm_doi)

    cu["ee_num"] = pd.to_numeric(cu["EE"], errors="coerce")
    ex["ee_num"] = pd.to_numeric(ex["group_mean_EE"], errors="coerce")

    # common field mapping
    ex_field_map: Dict[str, str] = {
        "small_molecule_name": "sig__drug_name",
        "polymer_MW": "sig__polymer_MW",
        "LA/GA": "sig__LA/GA",
        "surfactant_name": "sig__surfactant_name",
        "solvent": "sig__solvent",
        "drug/polymer": "sig__drug/polymer",
        "surfactant_concentration": "sig__surfactant_concentration",
        "aqueous/organic": "sig__aqueous/organic",
        "pH": "sig__pH",
    }
    all_fields = [
        "small_molecule_name",
        "polymer_MW",
        "LA/GA",
        "surfactant_name",
        "solvent",
        "drug/polymer",
        "surfactant_concentration",
        "aqueous/organic",
        "pH",
    ]
    for f in all_fields:
        cu[f"_k_{f}"] = cu[f].map(clean_val) if f in cu.columns else ""
        ex_col = ex_field_map[f]
        ex[f"_k_{f}"] = ex[ex_col].map(clean_val) if ex_col in ex.columns else ""

    levels = [
        ("Level_1_strict", all_fields),
        ("Level_2_core_fields", ["small_molecule_name", "polymer_MW", "LA/GA", "drug/polymer"]),
        ("Level_3_minimal", ["small_molecule_name"]),
    ]

    rows = []
    recalls = {}
    precisions = {}
    for name, fields in levels:
        r, p = run_level(cu, ex, fields)
        rows.append({"level": name, "formulation_recall": r, "formulation_precision": p})
        recalls[name] = r
        precisions[name] = p

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_p, sep="\t", index=False)

    print(f"recall_Level_1_strict={recalls['Level_1_strict']}")
    print(f"recall_Level_2_core_fields={recalls['Level_2_core_fields']}")
    print(f"recall_Level_3_minimal={recalls['Level_3_minimal']}")
    print(f"precision_Level_1_strict={precisions['Level_1_strict']}")
    print(f"precision_Level_2_core_fields={precisions['Level_2_core_fields']}")
    print(f"precision_Level_3_minimal={precisions['Level_3_minimal']}")
    print(f"delta_recall_L2_minus_L1={recalls['Level_2_core_fields'] - recalls['Level_1_strict']}")
    print(f"delta_recall_L3_minus_L2={recalls['Level_3_minimal'] - recalls['Level_2_core_fields']}")
    print(f"delta_recall_L3_minus_L1={recalls['Level_3_minimal'] - recalls['Level_1_strict']}")


if __name__ == "__main__":
    main()
