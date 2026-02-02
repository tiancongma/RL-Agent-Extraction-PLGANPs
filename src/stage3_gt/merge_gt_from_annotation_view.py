#!/usr/bin/env python3
"""
merge_gt_from_annotation_view.py

One-way merge:
    annotation Excel (human-edited) + authoritative TSV (read-only)
        -> authoritative GT TSV (machine-written)

Design goals
- Preserve row order and row count of the source TSV exactly.
- Only update (or create) three GT fields:
    gt_decision, gt_value_text, gt_notes
- Join uses stable keys:
    Prefer (key, formulation_id, field_name) if formulation_id exists in both;
    Else fall back to (key, field_name).

Typical usage (from repo root):
    python src/stage3_gt/merge_gt_from_annotation_view.py \
        --input-tsv data/cleaned/labels/manual/gt_field_decisions__run_XXXX.tsv \
        --annotation-xlsx data/cleaned/labels/manual/gt_annotation_view__run_XXXX.xlsx \
        --out-tsv data/cleaned/labels/manual/gt_field_decisions__run_XXXX__GT.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# Prefer centralized paths API when running inside the repo
try:
    from src.utils import paths as project_paths  # type: ignore
except Exception:
    project_paths = None


ALLOWED_DEFAULT = ["accept_model1", "accept_model2", "override", "unclear"]
GT_COLS = ["gt_decision", "gt_value_text", "gt_notes"]


def _read_tsv_robust(tsv_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with tsv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True,
            escapechar=None,
        )
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {tsv_path}")
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    df = pd.DataFrame(rows)
    df = df[[c for c in reader.fieldnames]]  # type: ignore[arg-type]
    return df


def _write_tsv_robust(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            delimiter="\t",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
            doublequote=True,
        )
        writer.writerow(list(df.columns))
        for _, row in df.iterrows():
            writer.writerow([("" if pd.isna(v) else str(v)) for v in row.tolist()])


def _infer_join_cols(src_cols: List[str], ann_cols: List[str]) -> List[str]:
    preferred = ["key", "formulation_id", "field_name"]
    fallback = ["key", "field_name"]

    if all(c in src_cols for c in preferred) and all(c in ann_cols for c in preferred):
        return preferred
    if all(c in src_cols for c in fallback) and all(c in ann_cols for c in fallback):
        return fallback

    raise ValueError(
        "Cannot infer join columns. Need either (key, field_name) or (key, formulation_id, field_name) "
        f"present in both source TSV and annotation XLSX.\n"
        f"source cols: {src_cols}\nannotation cols: {ann_cols}"
    )


def _validate_annotation(df_ann: pd.DataFrame, join_cols: List[str], allowed: List[str]) -> None:
    # Normalize NaN -> ""
    for c in GT_COLS:
        if c not in df_ann.columns:
            df_ann[c] = ""
        df_ann[c] = df_ann[c].fillna("").astype(str)

    if "gt_decision" in df_ann.columns:
        bad = df_ann.loc[
            (df_ann["gt_decision"].str.strip() != "") & (~df_ann["gt_decision"].isin(allowed)),
            join_cols + ["gt_decision"],
        ]
        if len(bad) > 0:
            example = bad.head(10).to_string(index=False)
            raise ValueError(
                "Found invalid gt_decision values in annotation file. "
                f"Allowed: {allowed}\nExamples:\n{example}"
            )

    # override requires gt_value_text
    if "gt_decision" in df_ann.columns and "gt_value_text" in df_ann.columns:
        needs = df_ann["gt_decision"].eq("override") & (df_ann["gt_value_text"].str.strip() == "")
        if needs.any():
            ex = df_ann.loc[needs, join_cols + ["gt_decision", "gt_value_text"]].head(10).to_string(index=False)
            raise ValueError(
                "Some rows have gt_decision='override' but empty gt_value_text. "
                "Either provide gt_value_text or change decision.\nExamples:\n" + ex
            )

    # Join key uniqueness
    dup = df_ann.duplicated(subset=join_cols, keep=False)
    if dup.any():
        ex = df_ann.loc[dup, join_cols + GT_COLS].head(10).to_string(index=False)
        raise ValueError(
            "Annotation file has duplicate join keys; merge would be ambiguous.\n"
            f"Join columns: {join_cols}\nExamples:\n{ex}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tsv", type=str, required=True, help="Authoritative TSV (read-only).")
    ap.add_argument("--annotation-xlsx", type=str, required=True, help="Human-edited annotation view XLSX.")
    ap.add_argument("--sheet", type=str, default="annotation", help="Worksheet name.")
    ap.add_argument("--out-tsv", type=str, required=True, help="Output TSV with GT merged in.")
    ap.add_argument(
        "--allowed-decisions",
        type=str,
        default=",".join(ALLOWED_DEFAULT),
        help="Comma-separated allowed gt_decision values.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting out-tsv.")
    args = ap.parse_args()

    in_tsv = Path(args.input_tsv)
    ann_xlsx = Path(args.annotation_xlsx)
    out_tsv = Path(args.out_tsv)

    if not in_tsv.exists():
        raise FileNotFoundError(f"input TSV not found: {in_tsv}")
    if not ann_xlsx.exists():
        raise FileNotFoundError(f"annotation XLSX not found: {ann_xlsx}")
    if out_tsv.exists() and not args.overwrite:
        raise FileExistsError(f"out-tsv exists (use --overwrite): {out_tsv}")

    allowed = [s.strip() for s in args.allowed_decisions.split(",") if s.strip()]
    if not allowed:
        raise ValueError("No allowed decisions provided.")

    df_src = _read_tsv_robust(in_tsv)
    df_ann = pd.read_excel(ann_xlsx, sheet_name=args.sheet, dtype=str, engine="openpyxl").fillna("")

    # Ensure GT columns exist in src
    for c in GT_COLS:
        if c not in df_src.columns:
            df_src[c] = ""

    join_cols = _infer_join_cols(list(df_src.columns), list(df_ann.columns))
    _validate_annotation(df_ann, join_cols=join_cols, allowed=allowed)

    # Keep only keys + GT columns from annotation
    df_ann_small = df_ann[join_cols + GT_COLS].copy()

    # Merge: left join to preserve source order/count
    df_merged = df_src.merge(df_ann_small, on=join_cols, how="left", suffixes=("", "__ann"))

    # Fill with annotation values when provided (non-empty), otherwise keep existing
    for c in GT_COLS:
        ann_c = c + "__ann"
        if ann_c not in df_merged.columns:
            continue
        src_vals = df_merged[c].fillna("").astype(str)
        ann_vals = df_merged[ann_c].fillna("").astype(str)
        df_merged[c] = src_vals.where(ann_vals.str.strip() == "", ann_vals)
        df_merged.drop(columns=[ann_c], inplace=True)

    # Final sanity: row count preserved
    if len(df_merged) != len(df_src):
        raise RuntimeError(
            f"Row count changed after merge (should never happen): src={len(df_src)} merged={len(df_merged)}"
        )

    _write_tsv_robust(df_merged, out_tsv)
    print(f"[OK] wrote merged GT TSV: {out_tsv}")
    print(f"[info] join columns: {join_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
