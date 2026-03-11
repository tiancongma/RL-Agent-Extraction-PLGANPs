#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def series_stats(s: pd.Series) -> Dict[str, Optional[float]]:
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if s_num.empty:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    return {
        "min": float(s_num.min()),
        "q25": float(s_num.quantile(0.25)),
        "median": float(s_num.median()),
        "q75": float(s_num.quantile(0.75)),
        "max": float(s_num.max()),
        "mean": float(s_num.mean()),
    }


def norm_doi(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    if not s:
        return ""
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def detect_col(cols: Iterable[str], candidates: List[str], contains: Optional[List[str]] = None) -> Optional[str]:
    cols_list = list(cols)
    lower_to_col = {c.lower(): c for c in cols_list}
    for c in candidates:
        if c.lower() in lower_to_col:
            return lower_to_col[c.lower()]
    if contains:
        for c in cols_list:
            cl = c.lower()
            if all(token in cl for token in contains):
                return c
    return None


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def resolve_path(cwd: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (cwd / p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated overlap subset + DOI-level EE scaffold.")
    parser.add_argument("--sample-jsonl", default="data/cleaned/samples/sample_goren18.jsonl")
    parser.add_argument("--key2txt", default="data/cleaned/index/key2txt_goren_2025.tsv")
    parser.add_argument("--goren-csv", default="data/benchmark/goren_2025/NP_dataset_formulations.csv")
    parser.add_argument("--weak-tsv", default="data/results/run_20260219_1643_780eb83_goren18_weaklabels_v1/weak_labels__gemini.tsv")
    parser.add_argument("--out-dir", default="data/benchmark/goren_2025/overlap_goren18_v1")
    args = parser.parse_args()

    cwd = Path.cwd()

    sample_jsonl = resolve_path(cwd, args.sample_jsonl)
    key2txt_tsv = resolve_path(cwd, args.key2txt)
    goren_csv = resolve_path(cwd, args.goren_csv)
    weak_tsv = resolve_path(cwd, args.weak_tsv)
    out_dir = resolve_path(cwd, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sample_jsonl.exists():
        raise FileNotFoundError(f"sample jsonl not found: {sample_jsonl}")
    if not key2txt_tsv.exists():
        raise FileNotFoundError(f"key2txt not found: {key2txt_tsv}")
    if not goren_csv.exists():
        raise FileNotFoundError(f"goren csv not found: {goren_csv}")
    if not weak_tsv.exists():
        raise FileNotFoundError(f"weak labels tsv not found: {weak_tsv}")

    sample_rows = read_jsonl(sample_jsonl)
    if not sample_rows:
        raise RuntimeError("sample jsonl is empty")

    sample_keys = list(sample_rows[0].keys())
    sample_key_col = detect_col(sample_keys, ["key", "zotero_key", "paper_id", "id"])
    sample_doi_col = detect_col(sample_keys, ["doi", "DOI", "reference"], contains=["doi"])
    if not sample_key_col:
        raise RuntimeError("could not detect key field in sample jsonl")
    if not sample_doi_col:
        raise RuntimeError("could not detect DOI field in sample jsonl")

    key_to_doi: Dict[str, str] = {}
    doi_set = set()
    for r in sample_rows:
        k = str(r.get(sample_key_col, "") or "").strip()
        d = norm_doi(r.get(sample_doi_col, ""))
        if d:
            doi_set.add(d)
        if k:
            key_to_doi[k] = d

    if not doi_set:
        raise RuntimeError("no DOI values found in sample jsonl after normalization")

    key2txt_df = pd.read_csv(key2txt_tsv, sep="\t", dtype=str, header=None)
    key2txt_n = int(len(key2txt_df))

    curated_df = pd.read_csv(goren_csv, dtype=str, encoding="utf-8-sig")
    curated_doi_col = detect_col(list(curated_df.columns), ["doi", "reference"], contains=["doi"])
    curated_ee_col = detect_col(
        list(curated_df.columns),
        ["EE", "encapsulation_efficiency_percent"],
        contains=["encapsulation", "eff"],
    )
    if not curated_doi_col:
        raise RuntimeError("could not detect DOI column in curated CSV")

    curated_df["doi_norm"] = curated_df[curated_doi_col].map(norm_doi)
    curated_overlap = curated_df[curated_df["doi_norm"].isin(doi_set)].copy()

    curated_subset_out = out_dir / "goren18_curated_overlap_subset.tsv"
    curated_overlap.to_csv(curated_subset_out, sep="\t", index=False)

    rows_per_doi = (
        curated_overlap.groupby("doi_norm", dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "doi_norm"], ascending=[False, True])
    )
    rows_per_doi_out = out_dir / "rows_per_doi.tsv"
    rows_per_doi.to_csv(rows_per_doi_out, sep="\t", index=False)

    coverage_report = {
        "n_unique_doi_in_overlap": int(curated_overlap["doi_norm"].nunique(dropna=True)),
        "n_rows": int(len(curated_overlap)),
    }
    coverage_report_out = out_dir / "coverage_report.json"
    coverage_report_out.write_text(json.dumps(coverage_report, indent=2), encoding="utf-8")

    extracted_df = pd.read_csv(weak_tsv, sep="\t", dtype=str)
    extracted_key_col = detect_col(list(extracted_df.columns), ["key", "zotero_key", "paper_id", "id"])
    extracted_doi_col = detect_col(list(extracted_df.columns), ["doi", "reference"], contains=["doi"])
    extracted_ee_col = detect_col(
        list(extracted_df.columns),
        ["encapsulation_efficiency_percent", "EE"],
        contains=["encapsulation", "eff"],
    )

    if extracted_doi_col:
        extracted_df["doi_norm"] = extracted_df[extracted_doi_col].map(norm_doi)
    elif extracted_key_col:
        extracted_df["doi_norm"] = extracted_df[extracted_key_col].map(lambda k: key_to_doi.get(str(k), ""))
    else:
        extracted_df["doi_norm"] = ""

    extracted_overlap = extracted_df[extracted_df["doi_norm"].isin(doi_set)].copy()

    if curated_ee_col:
        curated_overlap["ee_num"] = pd.to_numeric(curated_overlap[curated_ee_col], errors="coerce")
    else:
        curated_overlap["ee_num"] = pd.NA

    if extracted_ee_col:
        extracted_overlap["ee_num"] = pd.to_numeric(extracted_overlap[extracted_ee_col], errors="coerce")
    else:
        extracted_overlap["ee_num"] = pd.NA

    curated_summary = (
        curated_overlap.groupby("doi_norm", dropna=False)
        .agg(
            curated_rows=("doi_norm", "size"),
            curated_ee_n=("ee_num", "count"),
            curated_ee_mean=("ee_num", "mean"),
            curated_ee_median=("ee_num", "median"),
            curated_ee_min=("ee_num", "min"),
            curated_ee_max=("ee_num", "max"),
        )
        .reset_index()
    )
    extracted_summary = (
        extracted_overlap.groupby("doi_norm", dropna=False)
        .agg(
            extracted_rows=("doi_norm", "size"),
            extracted_ee_n=("ee_num", "count"),
            extracted_ee_mean=("ee_num", "mean"),
            extracted_ee_median=("ee_num", "median"),
            extracted_ee_min=("ee_num", "min"),
            extracted_ee_max=("ee_num", "max"),
        )
        .reset_index()
    )

    scaffold = pd.DataFrame({"doi_norm": sorted(doi_set)})
    scaffold = scaffold.merge(curated_summary, on="doi_norm", how="left")
    scaffold = scaffold.merge(extracted_summary, on="doi_norm", how="left")
    scaffold["curated_rows"] = scaffold["curated_rows"].fillna(0).astype(int)
    scaffold["extracted_rows"] = scaffold["extracted_rows"].fillna(0).astype(int)
    scaffold["curated_ee_n"] = scaffold["curated_ee_n"].fillna(0).astype(int)
    scaffold["extracted_ee_n"] = scaffold["extracted_ee_n"].fillna(0).astype(int)
    scaffold["has_curated_rows"] = scaffold["curated_rows"] > 0
    scaffold["has_extracted_rows"] = scaffold["extracted_rows"] > 0
    scaffold["has_curated_ee_values"] = scaffold["curated_ee_n"] > 0
    scaffold["has_extracted_ee_values"] = scaffold["extracted_ee_n"] > 0
    scaffold["ee_mean_abs_diff"] = (scaffold["curated_ee_mean"] - scaffold["extracted_ee_mean"]).abs()
    scaffold["ee_median_abs_diff"] = (scaffold["curated_ee_median"] - scaffold["extracted_ee_median"]).abs()
    scaffold["ee_mean_abs_diff_le_5"] = scaffold["ee_mean_abs_diff"].notna() & (scaffold["ee_mean_abs_diff"] <= 5.0)
    scaffold["ee_mean_abs_diff_le_10"] = scaffold["ee_mean_abs_diff"].notna() & (scaffold["ee_mean_abs_diff"] <= 10.0)

    scaffold_out = out_dir / "doi_level_ee_scaffold.tsv"
    scaffold.to_csv(scaffold_out, sep="\t", index=False)

    audit_cols = [
        "doi_norm",
        "reason",
        "curated_rows",
        "extracted_rows",
        "curated_ee_n",
        "extracted_ee_n",
        "ee_mean_abs_diff",
    ]
    block_a = (
        scaffold[
            scaffold["has_curated_ee_values"] & scaffold["has_extracted_ee_values"] & scaffold["ee_mean_abs_diff"].notna()
        ]
        .sort_values("ee_mean_abs_diff", ascending=False)
        .head(10)
        .copy()
    )
    block_a["reason"] = "A_top10_ee_mean_abs_diff"

    block_b = scaffold[(scaffold["extracted_ee_n"] > 0) & (scaffold["curated_ee_n"] == 0)].copy()
    block_b["reason"] = "B_extracted_ee_only"

    block_c = scaffold[(scaffold["curated_ee_n"] > 0) & (scaffold["extracted_ee_n"] == 0)].copy()
    block_c["reason"] = "C_curated_ee_only"

    block_d = scaffold.copy()
    block_d["rows_abs_diff"] = (block_d["extracted_rows"] - block_d["curated_rows"]).abs()
    block_d = block_d.sort_values("rows_abs_diff", ascending=False).head(10)
    block_d["reason"] = "D_top10_rows_abs_diff"

    audit_df = pd.concat([block_a, block_b, block_c, block_d], ignore_index=True)[audit_cols]
    audit_out = out_dir / "audit_priority__doi.tsv"
    audit_df.to_csv(audit_out, sep="\t", index=False)

    both_ee = scaffold[scaffold["has_curated_ee_values"] & scaffold["has_extracted_ee_values"]].copy()
    agreement_summary = {
        "n_doi_total": int(len(scaffold)),
        "n_doi_with_curated_ee": int(scaffold["has_curated_ee_values"].sum()),
        "n_doi_with_extracted_ee": int(scaffold["has_extracted_ee_values"].sum()),
        "n_doi_with_both_ee": int(len(both_ee)),
        "ee_mean_abs_diff_stats": series_stats(both_ee["ee_mean_abs_diff"] if not both_ee.empty else pd.Series(dtype=float)),
        "curated_rows_stats": series_stats(scaffold["curated_rows"]),
        "extracted_rows_stats": series_stats(scaffold["extracted_rows"]),
    }
    agreement_summary_out = out_dir / "agreement_summary.json"
    agreement_summary_out.write_text(json.dumps(agreement_summary, indent=2), encoding="utf-8")

    build_log = {
        "cwd": str(cwd),
        "inputs": {
            "sample_jsonl": str(sample_jsonl),
            "key2txt_tsv": str(key2txt_tsv),
            "goren_csv": str(goren_csv),
            "weak_tsv": str(weak_tsv),
        },
        "detected_columns": {
            "sample": {"key_col": sample_key_col, "doi_col": sample_doi_col},
            "curated": {"doi_col": curated_doi_col, "ee_col": curated_ee_col},
            "extracted": {
                "key_col": extracted_key_col,
                "doi_col": extracted_doi_col,
                "ee_col": extracted_ee_col,
            },
        },
        "counts": {
            "sample_rows": int(len(sample_rows)),
            "sample_unique_doi": int(len(doi_set)),
            "key2txt_rows": key2txt_n,
            "curated_overlap_rows": int(len(curated_overlap)),
            "curated_overlap_unique_doi": int(curated_overlap["doi_norm"].nunique(dropna=True)),
            "extracted_overlap_rows": int(len(extracted_overlap)),
            "scaffold_rows": int(len(scaffold)),
            "scaffold_has_curated_rows": int(scaffold["has_curated_rows"].sum()),
            "scaffold_has_extracted_rows": int(scaffold["has_extracted_rows"].sum()),
            "scaffold_has_curated_ee_values": int(scaffold["has_curated_ee_values"].sum()),
            "scaffold_has_extracted_ee_values": int(scaffold["has_extracted_ee_values"].sum()),
            "doi_with_computable_ee_mean_abs_diff": int(scaffold["ee_mean_abs_diff"].notna().sum()),
            "doi_with_computable_ee_median_abs_diff": int(scaffold["ee_median_abs_diff"].notna().sum()),
            "doi_ee_mean_abs_diff_le_5": int(scaffold["ee_mean_abs_diff_le_5"].sum()),
            "doi_ee_mean_abs_diff_le_10": int(scaffold["ee_mean_abs_diff_le_10"].sum()),
        },
        "outputs": {
            "curated_subset_tsv": str(curated_subset_out),
            "rows_per_doi_tsv": str(rows_per_doi_out),
            "coverage_report_json": str(coverage_report_out),
            "doi_level_ee_scaffold_tsv": str(scaffold_out),
            "agreement_summary_json": str(agreement_summary_out),
            "audit_priority_doi_tsv": str(audit_out),
        },
    }

    build_log_out = out_dir / "build_log.json"
    build_log_out.write_text(json.dumps(build_log, indent=2), encoding="utf-8")

    print(
        "[OK] "
        f"sample_unique_doi={len(doi_set)} "
        f"curated_overlap_rows={len(curated_overlap)} "
        f"curated_overlap_unique_doi={curated_overlap['doi_norm'].nunique(dropna=True)} "
        f"extracted_overlap_rows={len(extracted_overlap)} "
        f"scaffold_rows={len(scaffold)}"
    )


if __name__ == "__main__":
    main()
