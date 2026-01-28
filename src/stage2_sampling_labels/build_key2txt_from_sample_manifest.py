#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_key2txt_from_sample_manifest.py

from small samples manifest (manifest_html10.tsv) generate key2txt.tsv,
For futher LLM extration

Usage (from project root):
    .\.venv\Scripts\python.exe .\scripts\build_key2txt_from_sample_manifest.py ^
        --sample-manifest data\cleaned\samples\manifest_html10.tsv ^
        --output data\cleaned\samples\key2txt_html10.tsv
"""

import argparse
from pathlib import Path
import pandas as pd


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Build key2txt TSV from sample manifest (HTML10)."
    )
    p.add_argument(
        "--sample-manifest",
        type=Path,
        required=True,
        help="e.g. data/cleaned/samples/manifest_html10.tsv",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output key2txt TSV, e.g. data/cleaned/samples/key2txt_html10.tsv",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    mf = args.sample_manifest

    if not mf.exists():
        raise SystemExit(f"[ERROR] Sample manifest not found: {mf}")

    df = pd.read_csv(mf, sep="\t")
    if "zotero_key" not in df.columns or "cleaned_text_sample" not in df.columns:
        raise SystemExit(
            "[ERROR] sample manifest must contain 'zotero_key' and 'cleaned_text_sample'."
        )

    out_df = pd.DataFrame(
        {
            "key": df["zotero_key"],
            "text_path": df["cleaned_text_sample"],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, sep="\t", index=False)
    print(f"[OK] key2txt -> {args.output}")
    print(f"[INFO] rows={len(out_df)}")

if __name__ == "__main__":
    main()
