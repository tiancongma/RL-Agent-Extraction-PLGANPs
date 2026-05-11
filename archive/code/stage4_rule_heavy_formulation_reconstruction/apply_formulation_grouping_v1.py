#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def clean_str(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Formulation-level grouping within DOI (evaluation-only).")
    parser.add_argument(
        "--weak-tsv",
        default="data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/weak_labels__gemini.tsv",
    )
    parser.add_argument(
        "--sample-jsonl",
        default="data/cleaned/samples/sample_goren18.jsonl",
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
    weak_p = (cwd / args.weak_tsv).resolve()
    sample_p = (cwd / args.sample_jsonl).resolve()
    curated_p = (cwd / args.curated_tsv).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = pd.read_csv(weak_p, sep="\t", dtype=str)
    curated = pd.read_csv(curated_p, sep="\t", dtype=str)

    # key -> doi map
    key2doi: Dict[str, str] = {}
    with sample_p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = str(obj.get("key", "")).strip()
            d = norm_doi(obj.get("doi", ""))
            if k:
                key2doi[k] = d

    # DOI mapping for extracted
    if "doi_norm" in extracted.columns:
        extracted["doi_norm"] = extracted["doi_norm"].map(norm_doi)
    else:
        key_col = "key" if "key" in extracted.columns else "zotero_key" if "zotero_key" in extracted.columns else None
        if key_col is None:
            raise RuntimeError("No key column found in extracted TSV for DOI mapping.")
        extracted["doi_norm"] = extracted[key_col].map(lambda x: key2doi.get(str(x), ""))

    # EE column
    ee_col = "encapsulation_efficiency_percent" if "encapsulation_efficiency_percent" in extracted.columns else "EE"
    if ee_col not in extracted.columns:
        raise RuntimeError("No extracted EE column found.")
    extracted["ee_num"] = pd.to_numeric(extracted[ee_col], errors="coerce")

    # signature fields requested by user (map from available extracted schema where needed)
    sig_map = {
        "drug_name": "drug_name",
        "polymer_MW": "plga_mw_kDa" if "plga_mw_kDa" in extracted.columns else "polymer_MW",
        "LA/GA": "la_ga_ratio" if "la_ga_ratio" in extracted.columns else "LA/GA",
        "solvent": "organic_solvent" if "organic_solvent" in extracted.columns else "solvent",
        "surfactant_name": "surfactant_name",  # may be missing in extracted
        "drug/polymer": "drug/polymer",        # may be missing in extracted
        "surfactant_concentration": "surfactant_concentration",  # may be missing in extracted
        "aqueous/organic": "aqueous/organic",  # may be missing in extracted
        "pH": "pH",                            # may be missing in extracted
    }

    for logical, source in sig_map.items():
        if source in extracted.columns:
            extracted[f"sig__{logical}"] = extracted[source].map(clean_str)
        else:
            extracted[f"sig__{logical}"] = ""

    sig_cols = [f"sig__{k}" for k in sig_map.keys()]
    extracted["formulation_signature"] = extracted[sig_cols].apply(
        lambda r: " | ".join([f"{k}={r[f'sig__{k}']}" for k in sig_map.keys()]),
        axis=1,
    )

    # Only rows with DOI + numeric EE contribute to grouped EE stats
    ee_rows = extracted[extracted["doi_norm"].astype(str).str.len() > 0].copy()
    ee_rows = ee_rows[ee_rows["ee_num"].notna()].copy()
    n_rows_before = int(len(ee_rows))

    grouped = (
        ee_rows.groupby(["doi_norm", "formulation_signature"], dropna=False)
        .agg(
            group_size=("ee_num", "size"),
            n_unique_ee=("ee_num", lambda s: int(pd.Series(s).dropna().nunique())),
            unique_ee_values=("ee_num", lambda s: ";".join(str(v) for v in sorted(pd.Series(s).dropna().unique()))),
            group_mean_EE=("ee_num", "mean"),
        )
        .reset_index()
    )

    # Keep signature components in output for readability
    signature_parts = ee_rows[["doi_norm", "formulation_signature"] + sig_cols].drop_duplicates(
        subset=["doi_norm", "formulation_signature"]
    )
    grouped = grouped.merge(signature_parts, on=["doi_norm", "formulation_signature"], how="left")

    extracted_formulation_out = out_dir / "extracted_formulation_level_v1.tsv"
    grouped.to_csv(extracted_formulation_out, sep="\t", index=False)

    # Curated DOI EE mean
    if "doi_norm" in curated.columns:
        curated["doi_norm"] = curated["doi_norm"].map(norm_doi)
    elif "reference" in curated.columns:
        curated["doi_norm"] = curated["reference"].map(norm_doi)
    else:
        raise RuntimeError("No DOI column in curated TSV.")
    curated_ee_col = "EE" if "EE" in curated.columns else "encapsulation_efficiency_percent"
    if curated_ee_col not in curated.columns:
        raise RuntimeError("No curated EE column found.")
    curated["curated_ee_num"] = pd.to_numeric(curated[curated_ee_col], errors="coerce")
    curated_doi = (
        curated.groupby("doi_norm", dropna=False)["curated_ee_num"]
        .agg(curated_ee_mean="mean", curated_ee_n="count", n_rows_curated="size")
        .reset_index()
    )

    # Extracted DOI stats before/after grouping
    ext_before = (
        ee_rows.groupby("doi_norm", dropna=False)["ee_num"]
        .agg(extracted_ee_mean="mean", n_rows_before="size")
        .reset_index()
    )
    ext_after = (
        grouped.groupby("doi_norm", dropna=False)
        .agg(extracted_ee_mean_after=("group_mean_EE", "mean"), n_formulations_after=("formulation_signature", "size"))
        .reset_index()
    )

    scaffold = curated_doi.merge(ext_before, on="doi_norm", how="outer").merge(ext_after, on="doi_norm", how="outer")
    scaffold["ee_mean_abs_diff_before"] = (scaffold["curated_ee_mean"] - scaffold["extracted_ee_mean"]).abs()
    scaffold["ee_mean_abs_diff_after"] = (scaffold["curated_ee_mean"] - scaffold["extracted_ee_mean_after"]).abs()
    scaffold["improved"] = scaffold["ee_mean_abs_diff_after"] < scaffold["ee_mean_abs_diff_before"]

    scaffold_out = out_dir / "doi_level_ee_scaffold__formulation_grouped.tsv"
    scaffold.to_csv(scaffold_out, sep="\t", index=False)

    impact = scaffold[
        [
            "doi_norm",
            "n_rows_before",
            "n_formulations_after",
            "ee_mean_abs_diff_before",
            "ee_mean_abs_diff_after",
        ]
    ].copy()
    impact_out = out_dir / "formulation_group_impact_summary.tsv"
    impact.to_csv(impact_out, sep="\t", index=False)

    n_improved = int(scaffold["improved"].fillna(False).sum())
    print(f"DOIs improved in mean_abs_diff: {n_improved}")


if __name__ == "__main__":
    main()
