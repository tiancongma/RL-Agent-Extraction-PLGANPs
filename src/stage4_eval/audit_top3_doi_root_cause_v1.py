#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def sanitize_doi(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", doi).strip("_")


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def write_sectioned_tsv(path: Path, curated_df: pd.DataFrame, extracted_df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Section A: Curated rows for DOI")
    lines.append("\t".join(curated_df.columns))
    for row in curated_df.itertuples(index=False, name=None):
        lines.append("\t".join("" if pd.isna(v) else str(v) for v in row))
    lines.append("")
    lines.append("# Section B: Extracted rows for DOI")
    lines.append("\t".join(extracted_df.columns))
    for row in extracted_df.itertuples(index=False, name=None):
        lines.append("\t".join("" if pd.isna(v) else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_hypothesis(
    doi: str,
    curated_rows_n: int,
    extracted_rows_n: int,
    curated_ee: pd.Series,
    extracted_ee: pd.Series,
    curated_lc: pd.Series,
) -> Tuple[str, str]:
    c_ee_n = int(curated_ee.notna().sum())
    e_ee_n = int(extracted_ee.notna().sum())

    c_ee_mean = float(curated_ee.mean()) if c_ee_n > 0 else float("nan")
    e_ee_mean = float(extracted_ee.mean()) if e_ee_n > 0 else float("nan")
    lc_n = int(curated_lc.notna().sum())
    lc_mean = float(curated_lc.mean()) if lc_n > 0 else float("nan")

    if (
        lc_n > 0
        and c_ee_n > 0
        and e_ee_n > 0
        and pd.notna(lc_mean)
        and pd.notna(c_ee_mean)
        and pd.notna(e_ee_mean)
        and abs(e_ee_mean - lc_mean) < abs(e_ee_mean - c_ee_mean)
    ):
        return (
            "LC_vs_EE_confusion",
            f"extracted EE mean ({e_ee_mean:.3f}) is closer to curated LC mean ({lc_mean:.3f}) than curated EE mean ({c_ee_mean:.3f})",
        )

    if curated_rows_n > 0 and extracted_rows_n > curated_rows_n * 2:
        return (
            "duplicate_mentions_or_over_splitting",
            f"extracted_rows={extracted_rows_n} vs curated_rows={curated_rows_n}",
        )

    if c_ee_n > 1 and e_ee_n > 1:
        c_range = float(curated_ee.max() - curated_ee.min())
        e_range = float(extracted_ee.max() - extracted_ee.min())
        if c_range > 20 or e_range > 20:
            return (
                "condition_mixing_or_aggregation",
                f"wide EE spread observed (curated_range={c_range:.3f}, extracted_range={e_range:.3f})",
            )

    return (
        "table_row_mapping_issue",
        f"row cardinality differs (curated_rows={curated_rows_n}, extracted_rows={extracted_rows_n})",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted top-3 DOI root-cause audit (evaluation-only).")
    parser.add_argument(
        "--overlap-dir",
        default="data/benchmark/goren_2025/overlap_goren18_v1",
        help="Base overlap directory.",
    )
    parser.add_argument(
        "--weak-tsv",
        default="data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/weak_labels__gemini.tsv",
        help="Extracted weak label TSV.",
    )
    parser.add_argument(
        "--sample-jsonl",
        default="data/cleaned/samples/sample_goren18.jsonl",
        help="Sample JSONL for key->doi mapping.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    overlap_dir = (cwd / args.overlap_dir).resolve()
    out_dir = (overlap_dir / "audit_top3_v1").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curated_p = overlap_dir / "goren18_curated_overlap_subset.tsv"
    scaffold_p = overlap_dir / "doi_level_ee_scaffold.tsv"
    top_doi_p = overlap_dir / "metrics_tables_v1" / "top_doi_for_audit.tsv"
    weak_p = (cwd / args.weak_tsv).resolve()
    sample_p = (cwd / args.sample_jsonl).resolve()

    curated_df = pd.read_csv(curated_p, sep="\t", dtype=str)
    scaffold_df = pd.read_csv(scaffold_p, sep="\t", dtype=str)
    top_doi_df = pd.read_csv(top_doi_p, sep="\t", dtype=str)
    extracted_df = pd.read_csv(weak_p, sep="\t", dtype=str)

    # Build key -> DOI map from sample jsonl
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

    # Ensure normalized DOI in curated
    if "doi_norm" not in curated_df.columns:
        if "reference" in curated_df.columns:
            curated_df["doi_norm"] = curated_df["reference"].map(norm_doi)
        else:
            raise RuntimeError("Curated subset missing doi_norm/reference column.")
    else:
        curated_df["doi_norm"] = curated_df["doi_norm"].map(norm_doi)

    # Ensure normalized DOI in extracted
    if "doi_norm" in extracted_df.columns:
        extracted_df["doi_norm"] = extracted_df["doi_norm"].map(norm_doi)
    else:
        key_col = "key" if "key" in extracted_df.columns else None
        if key_col is None:
            raise RuntimeError("Extracted TSV lacks both doi_norm and key columns.")
        extracted_df["doi_norm"] = extracted_df[key_col].map(lambda k: key2doi.get(str(k), ""))

    # Top 3 target DOI
    top3_df = top_doi_df[top_doi_df["reason"] == "A_top10_ee_mean_abs_diff"].head(3).copy()
    top3_dois = [norm_doi(v) for v in top3_df["doi_norm"].tolist()]

    # Prepare summary table from scaffold
    scaffold_df["doi_norm"] = scaffold_df["doi_norm"].map(norm_doi)
    summary_cols = [
        "doi_norm",
        "ee_mean_abs_diff",
        "curated_rows",
        "extracted_rows",
        "curated_ee_mean",
        "extracted_ee_mean",
        "curated_ee_n",
        "extracted_ee_n",
    ]
    summary_df = scaffold_df[scaffold_df["doi_norm"].isin(top3_dois)][summary_cols].copy()
    summary_df = summary_df.set_index("doi_norm").loc[top3_dois].reset_index()
    summary_out = out_dir / "audit_top3_summary.tsv"
    summary_df.to_csv(summary_out, sep="\t", index=False)

    # Per-DOI audit files + hypotheses
    hypo_rows: List[Dict[str, str]] = []
    per_doi_files: List[str] = []
    for doi in top3_dois:
        cdf = curated_df[curated_df["doi_norm"] == doi].copy()
        edf = extracted_df[extracted_df["doi_norm"] == doi].copy()

        # Compose sectioned artifact
        name = f"audit_{sanitize_doi(doi)}.tsv"
        per_path = out_dir / name
        write_sectioned_tsv(per_path, cdf, edf)
        per_doi_files.append(name)

        c_ee = to_float_series(cdf["EE"]) if "EE" in cdf.columns else pd.Series(dtype=float)
        e_ee = (
            to_float_series(edf["encapsulation_efficiency_percent"])
            if "encapsulation_efficiency_percent" in edf.columns
            else pd.Series(dtype=float)
        )
        c_lc = to_float_series(cdf["LC"]) if "LC" in cdf.columns else pd.Series(dtype=float)

        cat, hint = pick_hypothesis(
            doi=doi,
            curated_rows_n=len(cdf),
            extracted_rows_n=len(edf),
            curated_ee=c_ee,
            extracted_ee=e_ee,
            curated_lc=c_lc,
        )
        hypo_rows.append(
            {
                "doi_norm": doi,
                "hypothesis_category": cat,
                "evidence_hint": hint,
            }
        )

    hypo_df = pd.DataFrame(hypo_rows)
    hypo_out = out_dir / "audit_top3_hypotheses.tsv"
    hypo_df.to_csv(hypo_out, sep="\t", index=False)

    print(f"[OK] out_dir={out_dir} dois={','.join(top3_dois)}")


if __name__ == "__main__":
    main()
