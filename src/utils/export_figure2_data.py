"""
Export raw data tables for Figure 2 (Sample and heterogeneity overview).

This script prepares TSV files for plotting in OriginLab:
- Panel A: source modality / availability
- Panel B: document text length distribution
- Panel C: parse quality distribution

Design principles:
- Deterministic
- No model calls
- No additional parsing
- Operates only on frozen TSV/JSONL artifacts
"""

import os
import argparse
import pandas as pd


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    manifest = read_tsv(args.manifest_tsv)
    strata = read_tsv(args.strata_tsv)

    # ----------------------------
    # load sample keys (optional)
    # ----------------------------
    if args.sample_tsv:
        sample = read_tsv(args.sample_tsv)
        if "key" not in sample.columns:
            raise ValueError(f"sample TSV missing 'key' column: {args.sample_tsv}")
        sample_keys = set(sample["key"].astype(str))
    else:
        # fallback: use all strata entries
        sample_keys = set(strata["key"].astype(str))

    strata_s = strata[strata["key"].astype(str).isin(sample_keys)].copy()

    # normalize numeric columns
    if "text_length" in strata_s.columns:
        strata_s["text_length"] = pd.to_numeric(
            strata_s["text_length"], errors="coerce"
        )

    # ============================
    # Panel A: source modality
    # ============================
    if "source_type" not in strata_s.columns:
        raise ValueError("strata_tags.tsv missing column: source_type")

    panelA_sample = (
        strata_s.groupby("source_type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # overall retrieval pool (include NO_LOCAL_FULLTEXT)
    if "notes" in manifest.columns:
        missing_keys = set(
            manifest[manifest["notes"].str.contains("NO_LOCAL_FULLTEXT", na=False)]["key"]
        )
    else:
        missing_keys = set()

    auditable_keys = set(strata["key"].astype(str))

    panelA_pool = pd.DataFrame(
        [
            {
                "source_category": "PDF",
                "count": int((strata.get("source_type", "") == "PDF").sum()),
            },
            {
                "source_category": "HTML",
                "count": int((strata.get("source_type", "") == "HTML").sum()),
            },
            {
                "source_category": "NO_LOCAL_FULLTEXT",
                "count": int(len(missing_keys - auditable_keys)),
            },
        ]
    )

    # ============================
    # Panel B: text length
    # ============================
    if "text_length" not in strata_s.columns:
        raise ValueError("strata_tags.tsv missing column: text_length")

    panelB = strata_s[["key", "source_type", "text_length"]].copy()

    # ============================
    # Panel C: parse quality
    # ============================
    if "parse_quality" not in strata_s.columns:
        raise ValueError("strata_tags.tsv missing column: parse_quality")

    panelC = (
        strata_s.groupby("parse_quality", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # ============================
    # Optional: per-paper table
    # ============================
    keep_cols = [
        c
        for c in [
            "key",
            "source_type",
            "text_length",
            "table_detected",
            "parse_quality",
            "particle_scale_tag",
            "emulsion_route_tag",
            "reporting_style_tag",
            "notes",
            "text_path",
        ]
        if c in strata_s.columns
    ]

    panel_all = strata_s[keep_cols].copy()

    # ============================
    # write outputs
    # ============================
    panelA_sample.to_csv(
        os.path.join(args.outdir, "panelA_source_type__sample.tsv"),
        sep="\t",
        index=False,
    )
    panelA_pool.to_csv(
        os.path.join(args.outdir, "panelA_source_category__retrieval_pool.tsv"),
        sep="\t",
        index=False,
    )
    panelB.to_csv(
        os.path.join(args.outdir, "panelB_text_length.tsv"),
        sep="\t",
        index=False,
    )
    panelC.to_csv(
        os.path.join(args.outdir, "panelC_parse_quality.tsv"),
        sep="\t",
        index=False,
    )
    panel_all.to_csv(
        os.path.join(args.outdir, "sample_heterogeneity__per_paper.tsv"),
        sep="\t",
        index=False,
    )

    print("[OK] Figure 2 data exported to:", args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export raw TSVs for Figure 2 plotting (OriginLab)."
    )
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--strata-tsv", required=True)
    parser.add_argument("--sample-tsv", default=None)
    parser.add_argument("--outdir", required=True)

    main(parser.parse_args())
