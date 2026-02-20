#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

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


def first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    cset = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in cset:
            return cset[c.lower()]
    return None


def clean_key_val(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply DOI-level extracted EE dedup prototype (evaluation only).")
    parser.add_argument(
        "--weak-tsv",
        default="data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/weak_labels__gemini.tsv",
    )
    parser.add_argument(
        "--sample-jsonl",
        default="data/cleaned/samples/sample_goren18.jsonl",
    )
    parser.add_argument(
        "--curated-overlap-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--scaffold-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/doi_level_ee_scaffold.tsv",
    )
    parser.add_argument(
        "--metrics-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/metrics_tables_v1",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/dedup_ee_v1",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    weak_p = (cwd / args.weak_tsv).resolve()
    sample_p = (cwd / args.sample_jsonl).resolve()
    curated_p = (cwd / args.curated_overlap_tsv).resolve()
    scaffold_p = (cwd / args.scaffold_tsv).resolve()
    metrics_dir = (cwd / args.metrics_dir).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = pd.read_csv(weak_p, sep="\t", dtype=str)
    curated = pd.read_csv(curated_p, sep="\t", dtype=str)
    scaffold = pd.read_csv(scaffold_p, sep="\t", dtype=str)
    top_audit_p = metrics_dir / "top_doi_for_audit.tsv"
    top_audit = pd.read_csv(top_audit_p, sep="\t", dtype=str) if top_audit_p.exists() else pd.DataFrame()

    # key -> doi map from sample jsonl
    key2doi: Dict[str, str] = {}
    with sample_p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj.get("key", "")).strip()
            doi = norm_doi(obj.get("doi", ""))
            if key:
                key2doi[key] = doi

    # map doi_norm in extracted
    if "doi_norm" in extracted.columns:
        extracted["doi_norm"] = extracted["doi_norm"].map(norm_doi)
    else:
        key_col = first_existing(list(extracted.columns), ["key", "zotero_key", "paper_id", "id"])
        if key_col is None:
            raise RuntimeError("Could not detect key column in extracted weak labels TSV.")
        extracted["doi_norm"] = extracted[key_col].map(lambda k: key2doi.get(str(k), ""))

    ee_col = first_existing(list(extracted.columns), ["encapsulation_efficiency_percent", "EE"])
    if ee_col is None:
        raise RuntimeError("Could not detect extracted EE column.")
    extracted["ee_value"] = pd.to_numeric(extracted[ee_col], errors="coerce")

    # curated DOI-level EE mean
    if "doi_norm" not in curated.columns:
        ref_col = first_existing(list(curated.columns), ["reference", "doi"])
        if ref_col is None:
            raise RuntimeError("Could not detect curated DOI column.")
        curated["doi_norm"] = curated[ref_col].map(norm_doi)
    else:
        curated["doi_norm"] = curated["doi_norm"].map(norm_doi)
    curated_ee_col = first_existing(list(curated.columns), ["EE", "encapsulation_efficiency_percent"])
    if curated_ee_col is None:
        raise RuntimeError("Could not detect curated EE column.")
    curated["ee_value"] = pd.to_numeric(curated[curated_ee_col], errors="coerce")
    curated_doi = (
        curated.groupby("doi_norm", dropna=False)["ee_value"]
        .agg(curated_ee_mean="mean", curated_ee_n="count")
        .reset_index()
    )

    # Build dedup keys
    drug_col = first_existing(list(extracted.columns), ["drug_name"])
    polymer_col = first_existing(list(extracted.columns), ["polymer_MW", "plga_mw_kDa"])
    solvent_col = first_existing(list(extracted.columns), ["solvent", "organic_solvent"])
    surf_col = first_existing(list(extracted.columns), ["surfactant", "surfactant_name"])

    ee_rows = extracted[extracted["ee_value"].notna()].copy()
    n_rows_before = int(len(ee_rows))

    dedup_key_cols = ["doi_norm", "ee_value"]
    for c in [drug_col, polymer_col, solvent_col, surf_col]:
        if c is not None:
            ee_rows[f"_k_{c}"] = ee_rows[c].map(clean_key_val)
            dedup_key_cols.append(f"_k_{c}")

    dedup_rows = ee_rows.drop_duplicates(subset=dedup_key_cols, keep="first").copy()
    n_rows_after = int(len(dedup_rows))

    dedup_rows_out = out_dir / "extracted_ee_dedup_rows.tsv"
    dedup_rows.to_csv(dedup_rows_out, sep="\t", index=False)

    # DOI-level extracted stats (original + dedup)
    ext_orig = (
        ee_rows.groupby("doi_norm", dropna=False)["ee_value"]
        .agg(extracted_ee_mean="mean", extracted_ee_n="count", extracted_rows="size")
        .reset_index()
    )
    ext_dedup = (
        dedup_rows.groupby("doi_norm", dropna=False)["ee_value"]
        .agg(extracted_ee_mean_dedup="mean", extracted_ee_n_dedup="count", extracted_rows_dedup="size")
        .reset_index()
    )

    scaffold_num = scaffold.copy()
    for c in [
        "curated_rows",
        "extracted_rows",
        "curated_ee_n",
        "extracted_ee_n",
        "curated_ee_mean",
        "extracted_ee_mean",
        "ee_mean_abs_diff",
    ]:
        if c in scaffold_num.columns:
            scaffold_num[c] = pd.to_numeric(scaffold_num[c], errors="coerce")

    doi_level = scaffold_num.merge(curated_doi, on="doi_norm", how="left", suffixes=("", "_curated_recalc"))
    doi_level = doi_level.merge(ext_orig, on="doi_norm", how="left", suffixes=("", "_orig_recalc"))
    doi_level = doi_level.merge(ext_dedup, on="doi_norm", how="left")

    # prefer scaffold extracted_rows if present; fall back to recalculated counts
    if "extracted_rows" in doi_level.columns:
        doi_level["extracted_rows"] = doi_level["extracted_rows"].fillna(doi_level["extracted_rows_orig_recalc"])
    else:
        doi_level["extracted_rows"] = doi_level["extracted_rows_orig_recalc"]

    if "extracted_ee_n" in doi_level.columns:
        doi_level["extracted_ee_n"] = doi_level["extracted_ee_n"].fillna(doi_level["extracted_ee_n_orig_recalc"])
    else:
        doi_level["extracted_ee_n"] = doi_level["extracted_ee_n_orig_recalc"]

    if "extracted_ee_mean" in doi_level.columns:
        doi_level["extracted_ee_mean"] = doi_level["extracted_ee_mean"].fillna(doi_level["extracted_ee_mean_orig_recalc"])
    else:
        doi_level["extracted_ee_mean"] = doi_level["extracted_ee_mean_orig_recalc"]

    doi_level["ee_mean_abs_diff"] = (doi_level["curated_ee_mean"] - doi_level["extracted_ee_mean"]).abs()
    doi_level["ee_mean_abs_diff_dedup"] = (doi_level["curated_ee_mean"] - doi_level["extracted_ee_mean_dedup"]).abs()
    doi_level["delta_abs_diff"] = doi_level["ee_mean_abs_diff"] - doi_level["ee_mean_abs_diff_dedup"]

    scaffold_dedup_out = out_dir / "doi_level_ee_scaffold__dedup.tsv"
    doi_level.to_csv(scaffold_dedup_out, sep="\t", index=False)

    impact_cols = [
        "doi_norm",
        "extracted_rows",
        "extracted_rows_dedup",
        "extracted_ee_mean",
        "extracted_ee_mean_dedup",
        "ee_mean_abs_diff",
        "ee_mean_abs_diff_dedup",
        "delta_abs_diff",
    ]
    impact = doi_level[impact_cols].copy()
    impact = impact.sort_values("ee_mean_abs_diff", ascending=False)
    impact_out = out_dir / "dedup_impact_summary.tsv"
    impact.to_csv(impact_out, sep="\t", index=False)

    # top-3 before/after (from top_doi_for_audit if present, otherwise by before diff)
    if not top_audit.empty and "reason" in top_audit.columns:
        top3_dois = top_audit[top_audit["reason"] == "A_top10_ee_mean_abs_diff"]["doi_norm"].head(3).map(norm_doi).tolist()
    else:
        top3_dois = impact["doi_norm"].head(3).tolist()

    top3_cmp = (
        impact[impact["doi_norm"].isin(top3_dois)]
        .set_index("doi_norm")
        .reindex(top3_dois)
        .reset_index()
    )

    n_improved = int((impact["delta_abs_diff"] > 0).sum())
    print(f"[OK] n_rows_before={n_rows_before} n_rows_after_dedup={n_rows_after} n_doi_improved={n_improved}")

    # Print before/after values for top3 DOIs
    for _, r in top3_cmp.iterrows():
        doi = r["doi_norm"]
        before = r["ee_mean_abs_diff"]
        after = r["ee_mean_abs_diff_dedup"]
        dec = bool(pd.notna(r["delta_abs_diff"]) and r["delta_abs_diff"] > 0)
        print(f"[TOP3] doi={doi} before={before} after={after} decreased={dec}")


if __name__ == "__main__":
    main()
