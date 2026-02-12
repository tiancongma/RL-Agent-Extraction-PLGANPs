from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from src.utils.paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply demo-only canonicalization rules to formulations_consensus_weak.tsv "
                    "and write a new TSV with *_canon_demo columns + unknown value reports."
    )
    p.add_argument("--in-tsv", required=True, help="Input TSV (e.g., formulations_consensus_weak.tsv)")
    p.add_argument("--rules-json", required=True, help="Rules JSON path (method_canon_rules_v0.json)")
    p.add_argument("--out-tsv", required=True, help="Output TSV with added canonical columns")
    p.add_argument("--unknown-out", required=True, help="Output TSV listing unmatched original values and counts")
    p.add_argument("--verbose", action="store_true", help="Print basic diagnostics")
    return p.parse_args()


def _resolve(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (PROJECT_ROOT / pp).resolve()


def _normalize_text(s: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    out = s.astype(str)

    if cfg.get("strip", True):
        out = out.str.strip()

    if cfg.get("lowercase", True):
        out = out.str.lower()

    if cfg.get("collapse_whitespace", True):
        out = out.str.replace(r"\s+", " ", regex=True)

    out = out.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    return out


def _apply_rules_to_series(
    raw: pd.Series,
    norm: pd.Series,
    rules: List[Dict[str, str]],
    default: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Return:
      canon: canonical label per row
      matched_rule: canon label of the rule that matched, or NA if default
    """
    canon = pd.Series([default] * len(norm), index=norm.index, dtype="object")
    matched_rule = pd.Series([pd.NA] * len(norm), index=norm.index, dtype="object")

    # Only attempt matches on non-null normalized strings
    active = norm.notna()

    for r in rules:
        c = r["canon"]
        pat = r["pattern"]

        # match only where still default and active
        mask = active & (canon == default) & norm.str.contains(pat, regex=True, na=False)
        if mask.any():
            canon.loc[mask] = c
            matched_rule.loc[mask] = c

    # Keep NA rows as NA to separate "missing" from "other"
    canon = canon.where(norm.notna(), pd.NA)
    matched_rule = matched_rule.where(norm.notna(), pd.NA)
    return canon, matched_rule


def _unknown_table(
    raw: pd.Series,
    canon: pd.Series,
    default: str,
    colname: str,
) -> pd.DataFrame:
    """
    Unknown means:
      raw is not empty
      canon == default
    Missing raw values are excluded.
    """
    s_raw = raw.astype(str).str.strip().replace("", pd.NA)
    mask = s_raw.notna() & (canon == default)

    vc = s_raw.loc[mask].value_counts(dropna=True)
    out = pd.DataFrame(
        {
            "field": colname,
            "count": vc.values,
            "raw_value": vc.index,
        }
    )
    return out


def main() -> int:
    args = parse_args()

    in_path = _resolve(args.in_tsv)
    rules_path = _resolve(args.rules_json)
    out_path = _resolve(args.out_tsv)
    unknown_path = _resolve(args.unknown_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    unknown_path.parent.mkdir(parents=True, exist_ok=True)

    rules_obj = json.loads(rules_path.read_text(encoding="utf-8"))
    norm_cfg = rules_obj.get("normalization", {})
    fields: Dict[str, Any] = rules_obj.get("fields", {})

    df = pd.read_csv(in_path, sep="\t", dtype=str).fillna("")

    unknown_frames: List[pd.DataFrame] = []

    if args.verbose:
        print(f"[info] read rows={len(df)} from {in_path}")
        print(f"[info] rules={rules_obj.get('version')} scope={rules_obj.get('scope')}")

    for src_col, spec in fields.items():
        if src_col not in df.columns:
            raise KeyError(f"Missing required column in input TSV: {src_col}")

        out_col = spec["output_col"]
        default = spec.get("default", "other")
        rules = spec.get("rules", [])

        raw = df[src_col]
        norm = _normalize_text(raw, norm_cfg)

        canon, matched_rule = _apply_rules_to_series(raw=raw, norm=norm, rules=rules, default=default)

        df[out_col] = canon
        df[out_col + "__matched_rule"] = matched_rule

        unknown_frames.append(_unknown_table(raw=raw, canon=canon, default=default, colname=src_col))

        if args.verbose:
            vc = df[out_col].value_counts(dropna=False)
            print(f"[info] {src_col} -> {out_col} categories={len(vc)}")
            print(vc.head(12).to_string())

    df.to_csv(out_path, sep="\t", index=False)

    unknown_df = pd.concat(unknown_frames, ignore_index=True)
    unknown_df = unknown_df.sort_values(["field", "count"], ascending=[True, False])
    unknown_df.to_csv(unknown_path, sep="\t", index=False)

    if args.verbose:
        print(f"[OK] wrote: {out_path}")
        print(f"[OK] wrote: {unknown_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
