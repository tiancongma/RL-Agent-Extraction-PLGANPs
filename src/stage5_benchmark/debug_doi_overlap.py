"""
Debug DOI overlap between our extraction outputs and the Goren 2025 benchmark.

Usage examples:
    python src/stage5_benchmark/debug_doi_overlap.py \
        --our-tsv "data/results/run_20260218_1334_4e58c55_sample20_evidence_v2/weak_labels__gemini.tsv" \
        --our-jsonl "data/results/run_20260218_1334_4e58c55_sample20_evidence_v2/weak_labels__gemini.jsonl" \
        --goren-csv "data/benchmark/goren_2025/NP_dataset_formulations.csv" \
        --out-dir "data/benchmark/goren_2025/overlap_eval/doi_debug"

    python src/stage5_benchmark/debug_doi_overlap.py \
        --our-tsv "data/results/run_xxx/weak_labels__gemini.tsv" \
        --our-jsonl "data/results/run_xxx/weak_labels__gemini.jsonl" \
        --goren-csv "data/benchmark/goren_2025/NP_dataset_formulations.csv" \
        --out-dir "data/benchmark/goren_2025/overlap_eval/doi_debug" \
        --our-doi-field "doi_from_jsonl" \
        --goren-doi-field "reference"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


RANDOM_SEED = 42
DEFAULT_GOREN_DOI_FIELD = "reference"
DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
OVERLAP_SUMMARY_PATH = Path("data/benchmark/goren_2025/overlap_eval/overlap_summary.json")
DOI_HINTS = ("doi", "reference", "url")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug DOI overlap diagnostics.")
    parser.add_argument("--our-tsv", required=True, help="Path to our TSV file.")
    parser.add_argument("--our-jsonl", required=True, help="Path to our JSONL file.")
    parser.add_argument("--goren-csv", required=True, help="Path to Goren benchmark CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory path.")
    parser.add_argument(
        "--our-doi-field",
        default=None,
        help="Optional explicit DOI field in our TSV. If not set, inferred from overlap_summary.json and alternatives.",
    )
    parser.add_argument(
        "--goren-doi-field",
        default=DEFAULT_GOREN_DOI_FIELD,
        help="DOI field in Goren CSV (default: reference).",
    )
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

    # Support broken DOI strings where whitespace appears right after slash.
    search_text = re.sub(r"/\s+", "/", text)
    doi_match = DOI_REGEX.search(search_text)
    if doi_match:
        doi = doi_match.group(0)
        doi = re.sub(r"[)\],.;]+$", "", doi)
        return doi

    # If no explicit DOI pattern is found, keep cleaned text for diagnostics.
    return text


def is_doi_like(value: str | None) -> bool:
    if not value:
        return False
    text = str(value).strip().lower()
    return bool(re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text))


def load_overlap_summary_our_field(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    field = data.get("our_doi_field")
    return str(field) if field else None


def gather_tsv_doi_candidates(df: pd.DataFrame) -> list[str]:
    name_candidates = [c for c in df.columns if any(h in c.lower() for h in DOI_HINTS)]
    pattern_candidates: list[str] = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        if sample.str.contains(r"10\.\d{4,9}/", regex=True, case=False).any():
            pattern_candidates.append(col)
    ordered = []
    for col in name_candidates + pattern_candidates:
        if col not in ordered:
            ordered.append(col)
    return ordered


def extract_strings_recursively(obj: Any) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(extract_strings_recursively(v))
    elif isinstance(obj, list):
        for v in obj:
            values.extend(extract_strings_recursively(v))
    elif isinstance(obj, str):
        values.append(obj)
    return values


def extract_key_to_dois_from_jsonl(jsonl_path: Path) -> dict[str, list[str]]:
    key_to_dois: dict[str, set[str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            key = obj.get("key")
            if key is None:
                continue
            key_str = str(key)
            all_strings = extract_strings_recursively(obj)
            for s in all_strings:
                normalized = normalize_doi(s)
                if is_doi_like(normalized):
                    key_to_dois.setdefault(key_str, set()).add(normalized)
    return {k: sorted(v) for k, v in key_to_dois.items()}


def collect_our_doi_rows(
    our_df: pd.DataFrame,
    our_doi_field: str | None,
    key_to_dois: dict[str, list[str]],
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    row_records: list[dict[str, Any]] = []
    source_label = "unknown"
    alternative_stats: list[dict[str, Any]] = []

    if our_doi_field and our_doi_field in our_df.columns:
        source_label = f"tsv:{our_doi_field}"
        for idx, row in our_df.iterrows():
            raw_value = row.get(our_doi_field)
            key_value = row.get("key")
            normalized = normalize_doi(raw_value)
            row_records.append(
                {
                    "row_id": idx,
                    "key": key_value,
                    "raw_doi": raw_value,
                    "normalized_doi": normalized,
                    "source": source_label,
                }
            )
        return pd.DataFrame(row_records), source_label, alternative_stats

    # Primary fallback: recover DOI-like values from JSONL using key.
    source_label = "jsonl:key->doi"
    for idx, row in our_df.iterrows():
        key = str(row.get("key")) if row.get("key") is not None else ""
        dois = key_to_dois.get(key, [])
        if len(dois) == 1:
            raw_value = dois[0]
            normalized = dois[0]
        elif len(dois) > 1:
            raw_value = "|".join(dois)
            normalized = max(dois, key=len)
        else:
            raw_value = None
            normalized = None
        row_records.append(
            {
                "row_id": idx,
                "key": row.get("key"),
                "raw_doi": raw_value,
                "normalized_doi": normalized,
                "source": source_label,
            }
        )
    jsonl_df = pd.DataFrame(row_records)
    doi_like_from_jsonl = int(jsonl_df["normalized_doi"].map(is_doi_like).sum())
    if doi_like_from_jsonl > 0:
        # Discover alternatives for diagnostics only.
        candidates = gather_tsv_doi_candidates(our_df)
        for candidate in candidates:
            vals = our_df[candidate].map(normalize_doi)
            alternative_stats.append(
                {
                    "candidate": candidate,
                    "doi_like_count": int(vals.map(is_doi_like).sum()),
                }
            )
        return jsonl_df, source_label, alternative_stats

    # Secondary fallback: TSV alternative candidates.
    candidates = gather_tsv_doi_candidates(our_df)
    for candidate in candidates:
        candidate_rows: list[dict[str, Any]] = []
        for idx, row in our_df.iterrows():
            raw_value = row.get(candidate)
            normalized = normalize_doi(raw_value)
            candidate_rows.append(
                {
                    "row_id": idx,
                    "key": row.get("key"),
                    "raw_doi": raw_value,
                    "normalized_doi": normalized,
                    "source": f"tsv:{candidate}",
                }
            )
        tmp = pd.DataFrame(candidate_rows)
        doi_like_count = int(tmp["normalized_doi"].map(is_doi_like).sum())
        alternative_stats.append({"candidate": candidate, "doi_like_count": doi_like_count})
        if doi_like_count > 0:
            return tmp, f"tsv:{candidate}", alternative_stats

    return jsonl_df, source_label, alternative_stats


def collect_goren_doi_rows(goren_df: pd.DataFrame, goren_doi_field: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for idx, row in goren_df.iterrows():
        raw_value = row.get(goren_doi_field)
        records.append(
            {
                "row_id": idx,
                "raw_doi": raw_value,
                "normalized_doi": normalize_doi(raw_value),
                "source": f"goren:{goren_doi_field}",
            }
        )
    return pd.DataFrame(records)


def deterministic_samples(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rng = random.Random(RANDOM_SEED)
    idxs = list(df.index)
    sample_size = min(n, len(idxs))
    chosen = rng.sample(idxs, sample_size)
    return df.loc[chosen].reset_index(drop=True)


def top_counts(values: pd.Series, n: int = 20) -> list[tuple[str, int]]:
    cleaned = [str(v) for v in values.dropna().tolist() if str(v).strip()]
    return Counter(cleaned).most_common(n)


def summarize_side(name: str, df: pd.DataFrame) -> dict[str, Any]:
    non_empty = df["normalized_doi"].dropna().astype(str)
    non_empty = non_empty[non_empty.str.strip() != ""]
    doi_like_mask = non_empty.map(lambda x: x.startswith("10."))
    doi_only = non_empty[doi_like_mask]
    return {
        "name": name,
        "total_rows": int(len(df)),
        "non_empty_normalized": int(len(non_empty)),
        "doi_like_count": int(doi_like_mask.sum()),
        "top20": top_counts(doi_only, n=20),
    }


def contains_non_doi_ids(df: pd.DataFrame) -> bool:
    vals = df["normalized_doi"].dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    if vals.empty:
        return False
    non_doi = vals[~vals.str.startswith("10.")]
    if non_doi.empty:
        return False
    short_code = non_doi.str.fullmatch(r"[A-Za-z0-9]{6,12}", na=False).mean()
    return bool(short_code > 0.3)


def contains_pmid_pmcid_or_urls(df: pd.DataFrame) -> bool:
    vals = df["raw_doi"].dropna().astype(str).str.lower()
    if vals.empty:
        return False
    patterns = [
        r"\bpmid\b",
        r"\bpmcid\b",
        r"\bpubmed\b",
        r"https?://",
        r"ncbi\.nlm\.nih\.gov",
    ]
    return any(vals.str.contains(p, regex=True).any() for p in patterns)


def side_by_side_first50(our_set: list[str], goren_set: list[str]) -> list[tuple[str, str]]:
    n = max(min(50, len(our_set)), min(50, len(goren_set)))
    rows: list[tuple[str, str]] = []
    for i in range(n):
        left = our_set[i] if i < len(our_set) else ""
        right = goren_set[i] if i < len(goren_set) else ""
        rows.append((left, right))
    return rows


def build_report(
    goren_summary: dict[str, Any],
    our_summary: dict[str, Any],
    overlap_set: set[str],
    our_df: pd.DataFrame,
    goren_df: pd.DataFrame,
    our_doi_source_used: str,
    our_alternative_stats: list[dict[str, Any]],
    goren_sample: pd.DataFrame,
    our_sample: pd.DataFrame,
) -> tuple[str, str, str]:
    our_norm = sorted(
        set(
            v
            for v in our_df["normalized_doi"].dropna().astype(str).tolist()
            if v.strip() and v.startswith("10.")
        )
    )
    goren_norm = sorted(
        set(
            v
            for v in goren_df["normalized_doi"].dropna().astype(str).tolist()
            if v.strip() and v.startswith("10.")
        )
    )

    lines: list[str] = []
    lines.append("DOI Overlap Debug Report")
    lines.append("")
    lines.append("[Alternative DOI Candidates in Our TSV]")
    if our_alternative_stats:
        for item in sorted(our_alternative_stats, key=lambda x: (-x["doi_like_count"], x["candidate"]))[:20]:
            lines.append(f"- {item['candidate']}: doi_like_count={item['doi_like_count']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append(f"Our DOI source used: {our_doi_source_used}")
    lines.append("")

    lines.append("[Goren Summary]")
    lines.append(f"- total_rows: {goren_summary['total_rows']}")
    lines.append(f"- non_empty_doi_strings: {goren_summary['non_empty_normalized']}")
    lines.append(f"- doi_like_strings_starting_10: {goren_summary['doi_like_count']}")
    lines.append("- top_20_normalized_dois:")
    for doi, count in goren_summary["top20"]:
        lines.append(f"  {doi}\t{count}")
    lines.append("")

    lines.append("[Our Summary]")
    lines.append(f"- total_rows: {our_summary['total_rows']}")
    lines.append(f"- non_empty_doi_strings: {our_summary['non_empty_normalized']}")
    lines.append(f"- doi_like_strings_starting_10: {our_summary['doi_like_count']}")
    lines.append("- top_20_normalized_dois:")
    for doi, count in our_summary["top20"]:
        lines.append(f"  {doi}\t{count}")
    lines.append("")

    lines.append("[Overlap]")
    lines.append(f"- normalized_overlap_count: {len(overlap_set)}")
    if overlap_set:
        lines.append("- overlap_examples:")
        for doi in sorted(overlap_set)[:50]:
            lines.append(f"  {doi}")
    lines.append("")
    lines.append("[Random Samples: Goren raw -> normalized]")
    for _, row in goren_sample.iterrows():
        lines.append(f"- {row.get('raw_doi')}\t=>\t{row.get('normalized_doi')}")
    lines.append("")
    lines.append("[Random Samples: Our raw -> normalized]")
    for _, row in our_sample.iterrows():
        lines.append(f"- {row.get('raw_doi')}\t=>\t{row.get('normalized_doi')}")
    lines.append("")

    likely_reason = "unknown"
    next_action = "inspect exported sample TSVs and adjust DOI source mapping"

    if our_summary["non_empty_normalized"] == 0:
        likely_reason = "our_jsonl_doi_field_empty_or_unrecoverable"
        next_action = "merge DOI into TSV from an external manifest keyed by paper identifier"
    elif our_summary["doi_like_count"] == 0:
        likely_reason = "our_identifiers_not_doi"
        next_action = "map internal paper keys to DOI before overlap evaluation"
    elif len(overlap_set) == 0:
        our_has_non_doi = contains_non_doi_ids(our_df)
        our_has_alt_id = contains_pmid_pmcid_or_urls(our_df)
        lines.append("[Zero-Overlap Diagnostics]")
        lines.append("- first_50_normalized_side_by_side:")
        for left, right in side_by_side_first50(our_norm, goren_norm):
            lines.append(f"  our={left}\tgoren={right}")
        lines.append(f"- our_contains_non_doi_identifier_pattern: {our_has_non_doi}")
        lines.append(f"- our_contains_pmid_pmcid_or_url_patterns: {our_has_alt_id}")
        lines.append("")

        if our_has_non_doi:
            likely_reason = "our_contains_internal_ids_instead_of_doi"
            next_action = "replace internal identifiers with DOI by joining a paper metadata table"
        elif our_has_alt_id:
            likely_reason = "our_uses_pmid_pmcid_or_publisher_urls_instead_of_clean_doi"
            next_action = "normalize and convert PMID/PMCID/URL references to DOI before matching"
        else:
            likely_reason = "no_shared_doi_between_datasets_after_normalization"
            next_action = "validate paper coverage overlap and DOI provenance between both datasets"
    else:
        likely_reason = "overlap_found"
        next_action = "proceed with matched DOI evaluation"

    lines.append("[Diagnosis]")
    lines.append(f"- likely_reason: {likely_reason}")
    lines.append(f"- next_action: {next_action}")
    lines.append("")

    return "\n".join(lines) + "\n", likely_reason, next_action


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    our_df = pd.read_csv(args.our_tsv, sep="\t")
    goren_df = pd.read_csv(args.goren_csv)

    summary_field = load_overlap_summary_our_field(OVERLAP_SUMMARY_PATH)
    our_doi_field = args.our_doi_field or summary_field

    key_to_dois = extract_key_to_dois_from_jsonl(Path(args.our_jsonl))
    our_doi_rows, our_doi_source_used, our_alternative_stats = collect_our_doi_rows(
        our_df=our_df,
        our_doi_field=our_doi_field,
        key_to_dois=key_to_dois,
    )
    goren_doi_rows = collect_goren_doi_rows(goren_df, args.goren_doi_field)

    # Build deterministic sample exports (before/after normalization visible).
    our_sample = deterministic_samples(our_doi_rows, n=20)
    goren_sample = deterministic_samples(goren_doi_rows, n=20)
    our_sample.to_csv(out_dir / "our_doi_samples.tsv", sep="\t", index=False)
    goren_sample.to_csv(out_dir / "goren_doi_samples.tsv", sep="\t", index=False)

    our_norm_set = set(
        v
        for v in our_doi_rows["normalized_doi"].dropna().astype(str)
        if v.strip() and v.startswith("10.")
    )
    goren_norm_set = set(
        v
        for v in goren_doi_rows["normalized_doi"].dropna().astype(str)
        if v.strip() and v.startswith("10.")
    )
    overlap_set = our_norm_set.intersection(goren_norm_set)

    if overlap_set:
        pd.DataFrame({"normalized_doi": sorted(overlap_set)}).to_csv(
            out_dir / "normalized_overlap_list.tsv",
            sep="\t",
            index=False,
        )

    our_summary = summarize_side("our", our_doi_rows)
    goren_summary = summarize_side("goren", goren_doi_rows)

    report_text, likely_reason, next_action = build_report(
        goren_summary=goren_summary,
        our_summary=our_summary,
        overlap_set=overlap_set,
        our_df=our_doi_rows,
        goren_df=goren_doi_rows,
        our_doi_source_used=our_doi_source_used,
        our_alternative_stats=our_alternative_stats,
        goren_sample=goren_sample,
        our_sample=our_sample,
    )
    (out_dir / "doi_overlap_report.txt").write_text(report_text, encoding="utf-8")

    print(f"Most likely reason overlap is zero: {likely_reason}")
    print(f"Next action: {next_action}")


if __name__ == "__main__":
    main()
