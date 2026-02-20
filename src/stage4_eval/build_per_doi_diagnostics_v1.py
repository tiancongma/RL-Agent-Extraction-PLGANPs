#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def recall_bucket(v: float) -> str:
    if pd.isna(v):
        return "low"
    if v >= 0.7:
        return "high"
    if v >= 0.3:
        return "medium"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-DOI diagnostics v1.")
    parser.add_argument(
        "--per-doi-recall",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/per_doi_recall.tsv",
    )
    parser.add_argument(
        "--doi-scaffold",
        default="data/benchmark/goren_2025/overlap_goren18_v1/doi_level_ee_scaffold.tsv",
    )
    parser.add_argument(
        "--table1-structure",
        default="data/benchmark/goren_2025/overlap_goren18_v1/metrics_tables_v1/table1_structure_coverage.tsv",
    )
    parser.add_argument(
        "--alignment-summary",
        default="data/benchmark/goren_2025/overlap_goren18_v1/formulation_group_v1/formulation_alignment_summary.tsv",
    )
    parser.add_argument(
        "--curated-overlap",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1/diagnostics_v1",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    p_recall = (cwd / args.per_doi_recall).resolve()
    p_scaffold = (cwd / args.doi_scaffold).resolve()
    p_table1 = (cwd / args.table1_structure).resolve()
    p_align_summary = (cwd / args.alignment_summary).resolve()
    p_curated = (cwd / args.curated_overlap).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_recall = pd.read_csv(p_recall, sep="\t", dtype=str)
    df_scaffold = pd.read_csv(p_scaffold, sep="\t", dtype=str)
    # Loaded for provenance / sanity coverage as requested inputs
    df_table1 = pd.read_csv(p_table1, sep="\t", dtype=str)
    df_align_summary = pd.read_csv(p_align_summary, sep="\t", dtype=str)
    df_curated = pd.read_csv(p_curated, sep="\t", dtype=str)

    # Normalize types
    for c in ["curated_rows", "extracted_rows", "ee_mean_abs_diff"]:
        df_scaffold[c] = pd.to_numeric(df_scaffold[c], errors="coerce")
    df_scaffold["ee_mean_abs_diff_le_5"] = to_bool(df_scaffold["ee_mean_abs_diff_le_5"])
    df_scaffold["ee_mean_abs_diff_le_10"] = to_bool(df_scaffold["ee_mean_abs_diff_le_10"])

    # per_doi_recall schema compatibility
    if "per_DOI_recall" in df_recall.columns:
        df_recall = df_recall.rename(columns={"per_DOI_recall": "per_doi_recall"})
    if "matched_curated_formulations" in df_recall.columns:
        df_recall = df_recall.rename(columns={"matched_curated_formulations": "per_doi_matched_curated"})
    if "total_curated_formulations" in df_recall.columns:
        df_recall = df_recall.rename(columns={"total_curated_formulations": "per_doi_total_curated"})

    for c in ["per_doi_recall", "per_doi_matched_curated", "per_doi_total_curated"]:
        df_recall[c] = pd.to_numeric(df_recall[c], errors="coerce")

    merged = df_scaffold[
        [
            "doi_norm",
            "curated_rows",
            "extracted_rows",
            "ee_mean_abs_diff",
            "ee_mean_abs_diff_le_5",
            "ee_mean_abs_diff_le_10",
        ]
    ].merge(
        df_recall[["doi_norm", "per_doi_recall", "per_doi_matched_curated", "per_doi_total_curated"]],
        on="doi_norm",
        how="left",
    )
    merged["row_ratio"] = merged["extracted_rows"] / merged["curated_rows"]
    merged["doi_level_stable"] = merged["ee_mean_abs_diff_le_5"]
    merged["recall_bucket"] = merged["per_doi_recall"].apply(recall_bucket)
    merged["row_gap_abs"] = (merged["extracted_rows"] - merged["curated_rows"]).abs()

    diagnostic_cols = [
        "doi_norm",
        "curated_rows",
        "extracted_rows",
        "row_ratio",
        "ee_mean_abs_diff",
        "ee_mean_abs_diff_le_5",
        "ee_mean_abs_diff_le_10",
        "per_doi_recall",
        "per_doi_matched_curated",
        "per_doi_total_curated",
        "doi_level_stable",
        "recall_bucket",
    ]
    per_doi_out = out_dir / "per_doi_diagnostic_v1.tsv"
    merged[diagnostic_cols].to_csv(per_doi_out, sep="\t", index=False)

    # Rollup
    rollup_rows: List[Dict[str, object]] = []
    # counts by recall bucket
    for b in ["high", "medium", "low"]:
        n = int((merged["recall_bucket"] == b).sum())
        rollup_rows.append(
            {
                "record_type": "count_by_recall_bucket",
                "recall_bucket": b,
                "doi_level_stable": "",
                "metric": "n_doi",
                "value": n,
                "notes": "",
            }
        )
    # stable vs unstable
    for stable_val, label in [(True, "stable"), (False, "unstable")]:
        n = int((merged["doi_level_stable"] == stable_val).sum())
        rollup_rows.append(
            {
                "record_type": "count_by_stability",
                "recall_bucket": "",
                "doi_level_stable": label,
                "metric": "n_doi",
                "value": n,
                "notes": "",
            }
        )
    # contingency 3x2
    for b in ["high", "medium", "low"]:
        for stable_val, label in [(True, "stable"), (False, "unstable")]:
            n = int(((merged["recall_bucket"] == b) & (merged["doi_level_stable"] == stable_val)).sum())
            rollup_rows.append(
                {
                    "record_type": "contingency_recall_x_stability",
                    "recall_bucket": b,
                    "doi_level_stable": label,
                    "metric": "n_doi",
                    "value": n,
                    "notes": "",
                }
            )
    # mean/median recall stable vs unstable
    for stable_val, label in [(True, "stable"), (False, "unstable")]:
        sub = merged[merged["doi_level_stable"] == stable_val]
        rollup_rows.append(
            {
                "record_type": "recall_stats_by_stability",
                "recall_bucket": "",
                "doi_level_stable": label,
                "metric": "per_doi_recall_mean",
                "value": float(sub["per_doi_recall"].mean()) if len(sub) else float("nan"),
                "notes": "",
            }
        )
        rollup_rows.append(
            {
                "record_type": "recall_stats_by_stability",
                "recall_bucket": "",
                "doi_level_stable": label,
                "metric": "per_doi_recall_median",
                "value": float(sub["per_doi_recall"].median()) if len(sub) else float("nan"),
                "notes": "",
            }
        )

    rollup_df = pd.DataFrame(rollup_rows)
    rollup_out = out_dir / "diagnostic_rollup_v1.tsv"
    rollup_df.to_csv(rollup_out, sep="\t", index=False)

    # Audit queue top10
    queue = merged.sort_values(["per_doi_recall", "row_gap_abs"], ascending=[True, False]).head(10).copy()
    queue_cols = [
        "doi_norm",
        "per_doi_recall",
        "per_doi_matched_curated",
        "per_doi_total_curated",
        "curated_rows",
        "extracted_rows",
        "row_gap_abs",
        "ee_mean_abs_diff",
        "doi_level_stable",
        "recall_bucket",
    ]
    queue_out = out_dir / "audit_queue_by_recall_v1.tsv"
    queue[queue_cols].to_csv(queue_out, sep="\t", index=False)

    # Requested print summary lines
    stable_low = int(((merged["doi_level_stable"] == True) & (merged["recall_bucket"] == "low")).sum())
    stable_total = int((merged["doi_level_stable"] == True).sum())
    print(f"stable_low_recall_count={stable_low} out_of_stable_total={stable_total}")

    lowest5 = merged.sort_values(["per_doi_recall", "row_gap_abs"], ascending=[True, False]).head(5)
    for _, r in lowest5.iterrows():
        print(
            f"low_recall_doi={r['doi_norm']} "
            f"recall={r['per_doi_recall']} "
            f"curated_rows={r['curated_rows']} "
            f"extracted_rows={r['extracted_rows']} "
            f"ee_mean_abs_diff={r['ee_mean_abs_diff']}"
        )

    # Build log
    log = {
        "inputs": {
            "per_doi_recall_tsv": str(p_recall),
            "doi_level_ee_scaffold_tsv": str(p_scaffold),
            "table1_structure_coverage_tsv": str(p_table1),
            "formulation_alignment_summary_tsv": str(p_align_summary),
            "curated_overlap_tsv": str(p_curated),
        },
        "input_row_counts": {
            "per_doi_recall_rows": int(len(df_recall)),
            "doi_level_ee_scaffold_rows": int(len(df_scaffold)),
            "table1_structure_rows": int(len(df_table1)),
            "formulation_alignment_summary_rows": int(len(df_align_summary)),
            "curated_overlap_rows": int(len(df_curated)),
        },
        "output_row_counts": {
            "per_doi_diagnostic_rows": int(len(merged)),
            "diagnostic_rollup_rows": int(len(rollup_df)),
            "audit_queue_rows": int(len(queue)),
        },
        "outputs": {
            "per_doi_diagnostic_v1_tsv": str(per_doi_out),
            "diagnostic_rollup_v1_tsv": str(rollup_out),
            "audit_queue_by_recall_v1_tsv": str(queue_out),
        },
    }
    log_out = out_dir / "diagnostics_build_log.json"
    log_out.write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
