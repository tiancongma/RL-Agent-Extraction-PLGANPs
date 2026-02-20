#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Formulation-level alignment eval (curated vs extracted).")
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

    # Signature fields
    curated_fields = [
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

    if "doi_norm" not in cu.columns:
        if "reference" in cu.columns:
            cu["doi_norm"] = cu["reference"].map(norm_doi)
        else:
            raise RuntimeError("Curated input missing doi_norm/reference.")
    else:
        cu["doi_norm"] = cu["doi_norm"].map(norm_doi)
    if "doi_norm" not in ex.columns:
        raise RuntimeError("Extracted formulation input missing doi_norm.")
    ex["doi_norm"] = ex["doi_norm"].map(norm_doi)

    cu["ee_num"] = pd.to_numeric(cu["EE"], errors="coerce")
    ex["ee_num"] = pd.to_numeric(ex["group_mean_EE"], errors="coerce")

    # Build normalized comparison columns
    for cf in curated_fields:
        cu[f"_k_{cf}"] = cu[cf].map(clean_val) if cf in cu.columns else ""
    for cf, ef in ex_field_map.items():
        ex[f"_k_{cf}"] = ex[ef].map(clean_val) if ef in ex.columns else ""

    align_rows: List[Dict[str, object]] = []
    matched_extracted_keys: set[Tuple[str, str]] = set()
    per_doi_counts: Dict[str, Dict[str, int]] = {}

    for doi, cu_d in cu.groupby("doi_norm", dropna=False):
        doi = str(doi)
        ex_d = ex[ex["doi_norm"] == doi].copy()
        total = 0
        matched = 0
        for c_row in cu_d.itertuples(index=False):
            c = c_row._asdict()
            c_ee = c.get("ee_num")
            total += 1

            best_sig = ""
            best_diff = float("nan")
            is_matched = False

            if pd.notna(c_ee) and len(ex_d) > 0:
                cand = ex_d.copy()
                for field in curated_fields:
                    cv = c.get(f"_k_{field}", "")
                    if cv:
                        cand = cand[cand[f"_k_{field}"] == cv]
                if len(cand) > 0:
                    cand = cand[cand["ee_num"].notna()].copy()
                    if len(cand) > 0:
                        cand["abs_diff"] = (cand["ee_num"] - float(c_ee)).abs()
                        cand = cand[cand["abs_diff"] <= 5.0]
                        if len(cand) > 0:
                            best = cand.sort_values("abs_diff", ascending=True).iloc[0]
                            best_sig = str(best.get("formulation_signature", ""))
                            best_diff = float(best["abs_diff"])
                            is_matched = True
                            matched_extracted_keys.add((doi, best_sig))

            if is_matched:
                matched += 1
            align_rows.append(
                {
                    "doi_norm": doi,
                    "curated_signature": " | ".join(f"{f}={c.get(f'_k_{f}', '')}" for f in curated_fields),
                    "matched": bool(is_matched),
                    "matched_extracted_signature": best_sig,
                    "ee_abs_diff": best_diff,
                }
            )

        per_doi_counts[doi] = {"matched": matched, "total": total}

    alignment = pd.DataFrame(align_rows)
    alignment_out = out_dir / "formulation_alignment.tsv"
    alignment.to_csv(alignment_out, sep="\t", index=False)

    matched_curated = int(alignment["matched"].astype(bool).sum())
    total_curated = int(len(alignment))
    total_extracted = int(len(ex))
    matched_extracted = int(len(matched_extracted_keys))

    formulation_recall = (matched_curated / total_curated) if total_curated else float("nan")
    formulation_precision = (matched_extracted / total_extracted) if total_extracted else float("nan")

    per_doi_rows = []
    for doi in sorted(per_doi_counts.keys()):
        m = per_doi_counts[doi]["matched"]
        t = per_doi_counts[doi]["total"]
        per_doi_rows.append(
            {
                "doi_norm": doi,
                "matched_curated_formulations": m,
                "total_curated_formulations": t,
                "per_DOI_recall": (m / t) if t else float("nan"),
            }
        )
    per_doi_df = pd.DataFrame(per_doi_rows)
    per_doi_out = out_dir / "per_doi_recall.tsv"
    per_doi_df.to_csv(per_doi_out, sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "matched_curated_formulations", "value": matched_curated},
            {"metric": "total_curated_formulations", "value": total_curated},
            {"metric": "matched_extracted_formulations", "value": matched_extracted},
            {"metric": "total_extracted_formulations", "value": total_extracted},
            {"metric": "formulation_recall", "value": formulation_recall},
            {"metric": "formulation_precision", "value": formulation_precision},
        ]
    )
    summary_out = out_dir / "formulation_alignment_summary.tsv"
    summary.to_csv(summary_out, sep="\t", index=False)

    print(f"formulation_recall={formulation_recall}")
    print(f"formulation_precision={formulation_precision}")


if __name__ == "__main__":
    main()
