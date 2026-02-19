"""
Build a Goren-driven overlap manifest against the local project corpus.

Usage examples:
    python src/stage5_benchmark/build_goren_overlap_manifest.py \
        --goren-csv "data/benchmark/goren_2025/NP_dataset_formulations.csv" \
        --manifest-tsv "data/cleaned/index/manifest_current.tsv" \
        --out-dir "data/benchmark/goren_2025/overlap_eval/goren_overlap_manifest"
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create overlap manifest between Goren 2025 DOI set and local corpus manifest."
    )
    parser.add_argument("--goren-csv", required=True, help="Path to Goren CSV.")
    parser.add_argument("--manifest-tsv", required=True, help="Path to manifest_current.tsv.")
    parser.add_argument("--out-dir", required=True, help="Output directory path.")
    return parser.parse_args()


def normalize_doi(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    text = urllib.parse.unquote(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    search_text = re.sub(r"/\s+", "/", text)

    doi_match = DOI_REGEX.search(search_text)
    if doi_match:
        doi = doi_match.group(0)
        doi = re.sub(r"[)\],.;]+$", "", doi)
        return doi

    return search_text


def is_doi_like(value: str | None) -> bool:
    if not value:
        return False
    return bool(re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", str(value).strip().lower()))


def has_text(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def has_local_fulltext(row: pd.Series) -> bool:
    pdf_ok = has_text(row.get("pdf"))
    html_ok = has_text(row.get("html"))
    notes = str(row.get("notes") or "")
    no_local_flag = "no_local_fulltext" in notes.lower()
    return (pdf_ok or html_ok) and not no_local_flag


def build_overlap_dois_table(goren_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    overlap = sorted(
        set(goren_df["doi_norm"].dropna().tolist()).intersection(
            set(manifest_df["doi_norm"].dropna().tolist())
        )
    )
    rows: list[dict[str, Any]] = []
    for doi in overlap:
        g_raw = (
            goren_df.loc[goren_df["doi_norm"] == doi, "reference"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        m_raw = (
            manifest_df.loc[manifest_df["doi_norm"] == doi, "doi"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        rows.append(
            {
                "normalized_doi": doi,
                "n_goren_rows": int((goren_df["doi_norm"] == doi).sum()),
                "n_manifest_rows": int((manifest_df["doi_norm"] == doi).sum()),
                "goren_raw_examples": " | ".join(sorted(g_raw)[:3]),
                "manifest_raw_examples": " | ".join(sorted(m_raw)[:3]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    goren_df = pd.read_csv(args.goren_csv)
    manifest_df = pd.read_csv(args.manifest_tsv, sep="\t")

    goren_df["reference"] = goren_df.get("reference")
    goren_df["doi_norm"] = goren_df["reference"].map(normalize_doi)
    goren_df["doi_norm"] = goren_df["doi_norm"].where(goren_df["doi_norm"].map(is_doi_like), None)

    manifest_df["doi_norm"] = manifest_df.get("doi").map(normalize_doi)
    manifest_df["doi_norm"] = manifest_df["doi_norm"].where(
        manifest_df["doi_norm"].map(is_doi_like), None
    )
    manifest_df["local_fulltext_available"] = manifest_df.apply(has_local_fulltext, axis=1)

    goren_unique = sorted(set(goren_df["doi_norm"].dropna().tolist()))
    manifest_unique = sorted(set(manifest_df["doi_norm"].dropna().tolist()))
    overlap_unique = sorted(set(goren_unique).intersection(set(manifest_unique)))

    overlap_dois_df = build_overlap_dois_table(goren_df, manifest_df)
    overlap_dois_df.to_csv(out_dir / "overlap_dois.tsv", sep="\t", index=False)

    manifest_overlap_df = manifest_df[manifest_df["doi_norm"].isin(overlap_unique)].copy()
    manifest_overlap_df.sort_values(
        by=[c for c in ["doi_norm", "zotero_key", "key"] if c in manifest_overlap_df.columns],
        inplace=True,
        kind="mergesort",
    )
    manifest_overlap_df.to_csv(out_dir / "manifest_overlap.tsv", sep="\t", index=False)

    benchmark_subset = manifest_overlap_df[manifest_overlap_df["local_fulltext_available"]].copy()
    benchmark_subset.sort_values(
        by=[c for c in ["zotero_key", "key"] if c in benchmark_subset.columns],
        inplace=True,
        kind="mergesort",
    )
    with (out_dir / "benchmark_keys.jsonl").open("w", encoding="utf-8") as f:
        for _, row in benchmark_subset.iterrows():
            obj = {
                "zotero_key": str(row.get("zotero_key") or row.get("key") or ""),
                "doi": row.get("doi_norm"),
                "title": row.get("title"),
                "pdf": row.get("pdf"),
                "html": row.get("html"),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def exclusion_reason(row: pd.Series) -> str:
        if not has_text(row.get("doi")):
            return "missing_doi"
        if not is_doi_like(row.get("doi_norm")):
            return "doi_not_parseable"
        if row.get("doi_norm") not in overlap_unique:
            return "doi_not_in_goren"
        if not row.get("local_fulltext_available"):
            return "no_local_fulltext"
        return "included"

    manifest_df["coverage_reason"] = manifest_df.apply(exclusion_reason, axis=1)
    exclusion_counts = Counter(
        r for r in manifest_df["coverage_reason"].tolist() if r != "included"
    )

    overlap_with_fulltext = manifest_overlap_df[manifest_overlap_df["local_fulltext_available"]]

    sample_cols = [c for c in ["zotero_key", "key", "doi", "doi_norm", "title", "pdf", "html"] if c in manifest_overlap_df.columns]
    sample_df = manifest_overlap_df[sample_cols].copy()
    sample_df["pdf_available"] = sample_df["pdf"].map(has_text) if "pdf" in sample_df.columns else False
    sample_df["html_available"] = sample_df["html"].map(has_text) if "html" in sample_df.columns else False
    sample_df = sample_df.sort_values(
        by=[c for c in ["doi_norm", "zotero_key", "key"] if c in sample_df.columns],
        kind="mergesort",
    ).head(20)

    report_lines: list[str] = []
    report_lines.append("Goren Overlap Coverage Report")
    report_lines.append("")
    report_lines.append(f"n_goren_total\t{len(goren_df)}")
    report_lines.append(f"n_goren_unique_doi\t{len(goren_unique)}")
    report_lines.append(f"n_manifest_total\t{len(manifest_df)}")
    report_lines.append(f"n_manifest_with_doi\t{int(manifest_df['doi_norm'].notna().sum())}")
    report_lines.append(f"n_manifest_unique_doi\t{len(manifest_unique)}")
    report_lines.append(f"n_overlap_unique_doi\t{len(overlap_unique)}")
    report_lines.append(
        f"n_overlap_with_local_fulltext\t{len(overlap_with_fulltext)}"
    )
    report_lines.append("")
    report_lines.append("Top exclusion reasons")
    if exclusion_counts:
        for reason, count in exclusion_counts.most_common(10):
            report_lines.append(f"- {reason}: {count}")
    else:
        report_lines.append("- none")
    report_lines.append("")
    report_lines.append("Sample overlapping rows (up to 20)")
    if sample_df.empty:
        report_lines.append("- none")
    else:
        for _, row in sample_df.iterrows():
            report_lines.append(
                f"- zotero_key={row.get('zotero_key')} | doi={row.get('doi_norm')} | "
                f"pdf_available={bool(row.get('pdf_available'))} | html_available={bool(row.get('html_available'))} | "
                f"title={row.get('title')}"
            )
    report_lines.append("")

    (out_dir / "coverage_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Overlap unique DOIs: {len(overlap_unique)}")
    print(f"Manifest overlap rows: {len(manifest_overlap_df)}")
    print(f"Benchmark keys written: {len(benchmark_subset)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
