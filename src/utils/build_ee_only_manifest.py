# src/utils/build_ee_only_manifest.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def _find_hit_columns(df: pd.DataFrame, prefix: str = "hit__") -> List[str]:
    """Return all columns that start with the given prefix."""
    return [c for c in df.columns if c.startswith(prefix)]


def _to_int(series: pd.Series) -> pd.Series:
    """Robust integer coercion with NaN and non-numeric handled as 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _is_nonempty_str(x: object) -> bool:
    """True if x is a non-empty string after stripping."""
    if x is None:
        return False
    s = str(x).strip()
    return s != ""


def _pick_best_local_source(row: pd.Series, prefer_html: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick the best local fulltext source and return (source_type, source_path).
    This uses manifest fields. By default: html > pdf.

    Expected manifest columns:
        - 'html' : path or identifier for local html-derived text
        - 'pdf'  : path or identifier for local pdf-derived text

    If your manifest stores paths differently, adjust here.
    """
    html_val = row.get("html", "")
    pdf_val = row.get("pdf", "")

    has_html = _is_nonempty_str(html_val) and "NO_LOCAL_FULLTEXT" not in str(html_val)
    has_pdf = _is_nonempty_str(pdf_val) and "NO_LOCAL_FULLTEXT" not in str(pdf_val)

    if prefer_html:
        if has_html:
            return ("html", str(html_val).strip())
        if has_pdf:
            return ("pdf", str(pdf_val).strip())
        return (None, None)
    else:
        if has_pdf:
            return ("pdf", str(pdf_val).strip())
        if has_html:
            return ("html", str(html_val).strip())
        return (None, None)


def build_ee_only_manifest(
    manifest_df: pd.DataFrame,
    ee_hits_df: pd.DataFrame,
    min_total_hits: int = 1,
    require_local_fulltext: bool = True,
    prefer_html: bool = True,
    append_hit_columns: bool = True,
) -> pd.DataFrame:
    """
    Filter the authoritative manifest to EE-only rows while preserving all original columns.

    Inputs:
        manifest_df: authoritative manifest_current.tsv loaded as DataFrame
        ee_hits_df:  ee_hits_per_doc.tsv loaded as DataFrame

    Output:
        Filtered manifest DataFrame, optionally augmented with ee_hit_total and hit__* columns.
    """
    if "zotero_key" not in manifest_df.columns:
        raise ValueError("manifest_current.tsv must contain column: zotero_key")
    if "zotero_key" not in ee_hits_df.columns:
        raise ValueError("ee_hits_per_doc.tsv must contain column: zotero_key")
    if "has_ee" not in ee_hits_df.columns:
        raise ValueError("ee_hits_per_doc.tsv must contain column: has_ee")

    # Compute ee_hit_total from hit__* columns (if present)
    ee = ee_hits_df.copy()
    ee["has_ee"] = _to_int(ee["has_ee"])

    hit_cols = _find_hit_columns(ee, prefix="hit__")
    if hit_cols:
        for c in hit_cols:
            ee[c] = _to_int(ee[c])
        ee["ee_hit_total"] = ee[hit_cols].sum(axis=1).astype(int)
    else:
        ee["ee_hit_total"] = 0

    # Collapse ee_hits to one row per zotero_key by taking max over duplicates (html/pdf duplicate keys)
    # This is conservative: if either version has hits, the doc is treated as EE-positive.
    agg_cols = ["has_ee", "ee_hit_total"] + hit_cols
    ee_doc = (
        ee.groupby("zotero_key", as_index=False)[agg_cols]
        .max()
    )

    # Join hits onto manifest (left join so we preserve manifest rows before filtering)
    merged = manifest_df.merge(ee_doc, on="zotero_key", how="left")

    # Fill missing hit info as zeros (docs not in ee_hits table are treated as EE-negative)
    merged["has_ee"] = _to_int(merged["has_ee"].fillna(0))
    merged["ee_hit_total"] = _to_int(merged["ee_hit_total"].fillna(0))
    for c in hit_cols:
        merged[c] = _to_int(merged[c].fillna(0))

    # EE filters
    merged = merged[(merged["has_ee"] == 1) & (merged["ee_hit_total"] >= int(min_total_hits))].copy()

    if merged.empty:
        return merged

    # Local fulltext filter based on manifest columns html/pdf
    if require_local_fulltext:
        if ("html" not in merged.columns) and ("pdf" not in merged.columns):
            raise ValueError("manifest_current.tsv must contain 'html' and/or 'pdf' columns for local fulltext filtering.")
        sources = merged.apply(lambda r: _pick_best_local_source(r, prefer_html=prefer_html), axis=1)
        merged["best_source_type"] = [t for (t, _) in sources]
        merged["best_source"] = [p for (_, p) in sources]
        merged = merged[merged["best_source_type"].notna()].copy()
    else:
        # Still populate best_source fields if columns exist, but do not filter
        if ("html" in merged.columns) or ("pdf" in merged.columns):
            sources = merged.apply(lambda r: _pick_best_local_source(r, prefer_html=prefer_html), axis=1)
            merged["best_source_type"] = [t for (t, _) in sources]
            merged["best_source"] = [p for (_, p) in sources]

    # Optionally drop hit columns if you truly want "only original fields"
    if not append_hit_columns:
        drop_cols = ["has_ee", "ee_hit_total"] + hit_cols
        for c in drop_cols:
            if c in merged.columns:
                merged = merged.drop(columns=[c])

    return merged


def write_tsv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def write_jsonl_from_manifest(
    df_manifest: pd.DataFrame,
    out_path: Path,
    include_fields: Optional[List[str]] = None,
) -> None:
    """
    Write JSONL where each line corresponds to a single document (zotero_key),
    using selected fields from the filtered manifest.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if include_fields is None:
        # Default minimal fields for downstream sampling/extraction
        include_fields = ["zotero_key", "best_source_type", "best_source"]

    for f in include_fields:
        if f not in df_manifest.columns:
            # Skip missing fields silently to keep this utility robust across manifest schema changes
            pass

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df_manifest.iterrows():
            rec = {}
            for k in include_fields:
                if k in df_manifest.columns:
                    v = row.get(k)
                    if pd.isna(v):
                        continue
                    rec[k] = v
            # Always include zotero_key
            if "zotero_key" in df_manifest.columns:
                rec["zotero_key"] = row["zotero_key"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter manifest_current.tsv to EE-only docs while preserving original manifest fields."
    )
    p.add_argument("--manifest-tsv", required=True, help="Path to data/cleaned/index/manifest_current.tsv")
    p.add_argument("--ee-hits-tsv", required=True, help="Path to data/cleaned/analysis/ee_hits_per_doc.tsv")

    p.add_argument("--out-manifest-tsv", required=True, help="Output filtered manifest TSV path")
    p.add_argument("--out-jsonl", default="", help="Optional output JSONL path (derived from filtered manifest)")

    p.add_argument("--min-total-hits", type=int, default=1, help="Minimum ee_hit_total required (default: 1)")
    p.add_argument("--allow-no-local-fulltext", action="store_true", help="Do not require local html/pdf availability")
    p.add_argument("--prefer-pdf", action="store_true", help="Prefer PDF over HTML when selecting best source")
    p.add_argument("--no-append-hit-columns", action="store_true", help="Do not append has_ee/ee_hit_total/hit__* columns to output TSV")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_tsv)
    ee_hits_path = Path(args.ee_hits_tsv)
    out_manifest_tsv = Path(args.out_manifest_tsv)

    manifest_df = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    ee_hits_df = pd.read_csv(ee_hits_path, sep="\t", dtype=str).fillna("")

    filtered = build_ee_only_manifest(
        manifest_df=manifest_df,
        ee_hits_df=ee_hits_df,
        min_total_hits=args.min_total_hits,
        require_local_fulltext=not args.allow_no_local_fulltext,
        prefer_html=not args.prefer_pdf,
        append_hit_columns=not args.no_append_hit_columns,
    )

    write_tsv(filtered, out_manifest_tsv)

    if args.out_jsonl:
        write_jsonl_from_manifest(filtered, Path(args.out_jsonl))

    print(
        "[OK] "
        f"input_manifest_rows={len(manifest_df)} | "
        f"output_rows={len(filtered)} | "
        f"min_total_hits={args.min_total_hits} | "
        f"require_local_fulltext={not args.allow_no_local_fulltext} | "
        f"out={out_manifest_tsv}"
    )


if __name__ == "__main__":
    main()
