#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def q(s: pd.Series, v: float) -> float:
    return float(s.quantile(v)) if len(s) else float("nan")


def add_row(rows: List[Dict[str, object]], metric: str, value: object, notes: str = "") -> None:
    rows.append({"metric": metric, "value": value, "notes": notes})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Goren overlap metrics tables (v1).")
    parser.add_argument(
        "--input-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1",
        help="Directory containing scaffold outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/metrics_tables_v1",
        help="Directory for metrics tables.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    in_dir = (cwd / args.input_dir).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scaffold_p = in_dir / "doi_level_ee_scaffold.tsv"
    rows_per_doi_p = in_dir / "rows_per_doi.tsv"
    agreement_summary_p = in_dir / "agreement_summary.json"
    audit_p = in_dir / "audit_priority__doi.tsv"
    build_log_p = in_dir / "build_log.json"

    scaffold = pd.read_csv(scaffold_p, sep="\t", dtype=str)
    rows_per_doi = pd.read_csv(rows_per_doi_p, sep="\t", dtype=str)
    audit = pd.read_csv(audit_p, sep="\t", dtype=str)
    agreement_summary = json.loads(agreement_summary_p.read_text(encoding="utf-8"))
    build_log = json.loads(build_log_p.read_text(encoding="utf-8"))

    # Numeric conversions
    for c in [
        "curated_rows",
        "extracted_rows",
        "curated_ee_n",
        "extracted_ee_n",
        "ee_mean_abs_diff",
        "ee_median_abs_diff",
    ]:
        scaffold[c] = pd.to_numeric(scaffold[c], errors="coerce")

    scaffold["has_curated_rows"] = to_bool_series(scaffold["has_curated_rows"])
    scaffold["has_extracted_rows"] = to_bool_series(scaffold["has_extracted_rows"])
    scaffold["has_curated_ee_values"] = to_bool_series(scaffold["has_curated_ee_values"])
    scaffold["has_extracted_ee_values"] = to_bool_series(scaffold["has_extracted_ee_values"])
    scaffold["ee_mean_abs_diff_le_5"] = to_bool_series(scaffold["ee_mean_abs_diff_le_5"])
    scaffold["ee_mean_abs_diff_le_10"] = to_bool_series(scaffold["ee_mean_abs_diff_le_10"])

    total_doi = len(scaffold)
    curated_rows_zero_n = int((scaffold["curated_rows"].fillna(0) == 0).sum())
    row_note = "No DOI has curated_rows==0." if curated_rows_zero_n == 0 else f"{curated_rows_zero_n} DOI(s) have curated_rows==0."

    # Table 1
    t1_rows: List[Dict[str, object]] = []
    add_row(t1_rows, "overlap_doi_total", int(total_doi), "")
    add_row(t1_rows, "doi_with_curated_rows", int(scaffold["has_curated_rows"].sum()), "")
    add_row(t1_rows, "doi_with_extracted_rows", int(scaffold["has_extracted_rows"].sum()), "")
    add_row(t1_rows, "curated_rows_total", float(scaffold["curated_rows"].sum()), "")
    add_row(t1_rows, "extracted_rows_total", float(scaffold["extracted_rows"].sum()), "")

    ratios = (scaffold[scaffold["curated_rows"] > 0]["extracted_rows"] / scaffold[scaffold["curated_rows"] > 0]["curated_rows"]).dropna()
    row_gap_abs = (scaffold["extracted_rows"] - scaffold["curated_rows"]).abs().dropna()

    add_row(t1_rows, "row_ratio_mean", float(ratios.mean()) if len(ratios) else float("nan"), "exclude curated_rows==0")
    add_row(t1_rows, "row_ratio_median", float(ratios.median()) if len(ratios) else float("nan"), "exclude curated_rows==0")
    add_row(t1_rows, "row_ratio_min", float(ratios.min()) if len(ratios) else float("nan"), "exclude curated_rows==0")
    add_row(t1_rows, "row_ratio_max", float(ratios.max()) if len(ratios) else float("nan"), "exclude curated_rows==0")
    add_row(t1_rows, "row_gap_abs_mean", float(row_gap_abs.mean()) if len(row_gap_abs) else float("nan"), "")
    add_row(t1_rows, "row_gap_abs_median", float(row_gap_abs.median()) if len(row_gap_abs) else float("nan"), "")
    add_row(t1_rows, "row_gap_abs_max", float(row_gap_abs.max()) if len(row_gap_abs) else float("nan"), row_note)
    t1 = pd.DataFrame(t1_rows)
    t1_out = out_dir / "table1_structure_coverage.tsv"
    t1.to_csv(t1_out, sep="\t", index=False)

    # Table 2
    t2_rows: List[Dict[str, object]] = []
    comparable_mask = scaffold["has_curated_ee_values"] & scaffold["has_extracted_ee_values"]
    cmp_df = scaffold[comparable_mask].copy()
    ee_mean = cmp_df["ee_mean_abs_diff"].dropna()
    ee_med = cmp_df["ee_median_abs_diff"].dropna()

    n_le5 = int(scaffold["ee_mean_abs_diff_le_5"].sum())
    n_le10 = int(scaffold["ee_mean_abs_diff_le_10"].sum())
    add_row(t2_rows, "doi_with_comparable_ee", int(comparable_mask.sum()), "")
    add_row(t2_rows, "ee_mean_abs_diff_mean", float(ee_mean.mean()) if len(ee_mean) else float("nan"), "comparable EE DOI only")
    add_row(t2_rows, "ee_mean_abs_diff_median", float(ee_mean.median()) if len(ee_mean) else float("nan"), "comparable EE DOI only")
    add_row(t2_rows, "ee_mean_abs_diff_q25", q(ee_mean, 0.25), "comparable EE DOI only")
    add_row(t2_rows, "ee_mean_abs_diff_q75", q(ee_mean, 0.75), "comparable EE DOI only")
    add_row(t2_rows, "ee_mean_abs_diff_max", float(ee_mean.max()) if len(ee_mean) else float("nan"), "comparable EE DOI only")
    add_row(t2_rows, "ee_median_abs_diff_mean", float(ee_med.mean()) if len(ee_med) else float("nan"), "comparable EE DOI only")
    add_row(t2_rows, "ee_median_abs_diff_median", float(ee_med.median()) if len(ee_med) else float("nan"), "comparable EE DOI only")
    add_row(t2_rows, "ee_median_abs_diff_q25", q(ee_med, 0.25), "comparable EE DOI only")
    add_row(t2_rows, "ee_median_abs_diff_q75", q(ee_med, 0.75), "comparable EE DOI only")
    add_row(t2_rows, "ee_median_abs_diff_max", float(ee_med.max()) if len(ee_med) else float("nan"), "comparable EE DOI only")
    add_row(t2_rows, "frac_mean_abs_diff_le_5", (n_le5 / total_doi) if total_doi else float("nan"), "denominator=total DOI")
    add_row(t2_rows, "frac_mean_abs_diff_le_10", (n_le10 / total_doi) if total_doi else float("nan"), "denominator=total DOI")
    add_row(t2_rows, "n_mean_abs_diff_le_5", n_le5, "")
    add_row(t2_rows, "n_mean_abs_diff_le_10", n_le10, "")
    t2 = pd.DataFrame(t2_rows)
    t2_out = out_dir / "table2_ee_agreement.tsv"
    t2.to_csv(t2_out, sep="\t", index=False)

    # Table 3
    t3_rows: List[Dict[str, object]] = []
    ee_sorted = scaffold["ee_mean_abs_diff"].dropna().sort_values(ascending=False).reset_index(drop=True)
    top1 = float(ee_sorted.iloc[0]) if len(ee_sorted) >= 1 else float("nan")
    top3_sum = float(ee_sorted.head(3).sum()) if len(ee_sorted) else float("nan")
    top5_sum = float(ee_sorted.head(5).sum()) if len(ee_sorted) else float("nan")
    total_sum = float(ee_sorted.sum()) if len(ee_sorted) else float("nan")
    top3_share = (top3_sum / total_sum) if total_sum and pd.notna(total_sum) else float("nan")

    row_gap = (scaffold["extracted_rows"] - scaffold["curated_rows"]).abs()
    add_row(t3_rows, "top1_mean_abs_diff", top1, "")
    add_row(t3_rows, "top3_mean_abs_diff_sum", top3_sum, "")
    add_row(t3_rows, "top5_mean_abs_diff_sum", top5_sum, "")
    add_row(t3_rows, "total_mean_abs_diff_sum", total_sum, "")
    add_row(t3_rows, "top3_error_share", top3_share, "top3_sum / total_sum")
    add_row(t3_rows, "n_doi_mean_abs_diff_gt_10", int((scaffold["ee_mean_abs_diff"] > 10).sum()), "")
    add_row(t3_rows, "n_doi_row_gap_abs_gt_5", int((row_gap > 5).sum()), "abs(extracted_rows-curated_rows) > 5")
    t3 = pd.DataFrame(t3_rows)
    t3_out = out_dir / "table3_error_distribution.tsv"
    t3.to_csv(t3_out, sep="\t", index=False)

    # Top DOI list from existing audit table (block A), merged with scaffold counts
    top_a = audit[audit["reason"] == "A_top10_ee_mean_abs_diff"][["doi_norm", "reason"]].copy()
    if len(top_a) == 0:
        top_a = audit[["doi_norm", "reason"]].copy().head(10)
    top_doi = top_a.merge(
        scaffold[["doi_norm", "ee_mean_abs_diff", "curated_rows", "extracted_rows"]],
        on="doi_norm",
        how="left",
    )[["doi_norm", "ee_mean_abs_diff", "curated_rows", "extracted_rows", "reason"]]
    top_doi_out = out_dir / "top_doi_for_audit.tsv"
    top_doi.to_csv(top_doi_out, sep="\t", index=False)

    metrics_log = {
        "cwd": str(cwd),
        "input_dir": str(in_dir),
        "inputs": {
            "doi_level_ee_scaffold_tsv": str(scaffold_p),
            "rows_per_doi_tsv": str(rows_per_doi_p),
            "agreement_summary_json": str(agreement_summary_p),
            "audit_priority_doi_tsv": str(audit_p),
            "build_log_json": str(build_log_p),
        },
        "input_row_counts": {
            "doi_level_ee_scaffold_rows": int(len(scaffold)),
            "rows_per_doi_rows": int(len(rows_per_doi)),
            "audit_priority_rows": int(len(audit)),
            "agreement_summary_exists": bool(isinstance(agreement_summary, dict)),
            "build_log_exists": bool(isinstance(build_log, dict)),
        },
        "exclusions": {
            "row_ratio_excluded_curated_rows_eq_0": curated_rows_zero_n,
            "ee_comparable_required_for_ee_stats": int(total_doi - comparable_mask.sum()),
        },
        "outputs": {
            "table1_structure_coverage_tsv": str(t1_out),
            "table2_ee_agreement_tsv": str(t2_out),
            "table3_error_distribution_tsv": str(t3_out),
            "top_doi_for_audit_tsv": str(top_doi_out),
        },
    }
    metrics_log_out = out_dir / "metrics_build_log.json"
    metrics_log_out.write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")

    print(f"[OK] metrics tables written to: {out_dir}")


if __name__ == "__main__":
    main()
