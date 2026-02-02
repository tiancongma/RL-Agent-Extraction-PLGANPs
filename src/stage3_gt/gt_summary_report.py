#!/usr/bin/env python3
"""
gt_summary_report.py

Summarize GT decisions from a merged GT TSV (e.g., *__GT.tsv).

Outputs
- Overall counts by gt_decision
- Counts by field_name (wide table)
- Basic quality checks:
  - invalid gt_decision values
  - override rows missing gt_value_text
  - empty evidence (optional, if evidence column exists)

Design notes
- Robust TSV parsing with Python's csv module to handle multiline quoted fields.
- Treats all columns as strings; never reorders or edits the source file.

Example (repo root, PowerShell):
  python src/stage3_gt/gt_summary_report.py ^
    --gt-tsv data/cleaned/labels/manual/gt_field_decisions__run_20260201_0927_bb13267_sample20__GT.tsv ^
    --exclude-fields note ^
    --out-tsv data/results/gt_summary__run_20260201_0927_bb13267_sample20.tsv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ALLOWED_DECISIONS = ["accept_model1", "accept_model2", "override", "unclear"]
GT_COLS = ["gt_decision", "gt_value_text", "gt_notes"]


def read_tsv_robust(tsv_path: Path) -> List[Dict[str, str]]:
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
    return rows


def normalize_field_name(x: str) -> str:
    return (x or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-tsv", type=str, required=True, help="Merged GT TSV (e.g., *__GT.tsv).")
    ap.add_argument(
        "--allowed-decisions",
        type=str,
        default=",".join(ALLOWED_DECISIONS),
        help="Comma-separated allowed gt_decision values.",
    )
    ap.add_argument(
        "--exclude-fields",
        type=str,
        default="",
        help="Comma-separated field_name values to exclude from stats (e.g., note).",
    )
    ap.add_argument(
        "--out-tsv",
        type=str,
        default="",
        help="Optional: write a tidy summary TSV (long format) to this path.",
    )
    ap.add_argument(
        "--out-by-field-tsv",
        type=str,
        default="",
        help="Optional: write decision counts by field_name (wide table) to this path.",
    )
    args = ap.parse_args()

    gt_path = Path(args.gt_tsv)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT TSV not found: {gt_path}")

    allowed = [s.strip() for s in args.allowed_decisions.split(",") if s.strip()]
    excluded = {s.strip() for s in args.exclude_fields.split(",") if s.strip()}

    rows = read_tsv_robust(gt_path)
    if not rows:
        print("[warn] no rows found")
        return 0

    # Required columns
    cols = set(rows[0].keys())
    if "gt_decision" not in cols:
        raise ValueError("Missing required column: gt_decision")
    if "field_name" not in cols:
        raise ValueError("Missing required column: field_name")

    # Optional evidence column (best-effort)
    evidence_col = None
    for c in ["evidence_span_text_main", "evidence_text", "evidence", "evidence_span"]:
        if c in cols:
            evidence_col = c
            break

    # Overall and by-field counters
    overall = Counter()
    by_field = defaultdict(Counter)

    invalid_decisions: List[Tuple[str, str, str]] = []  # (key, formulation_id, gt_decision)
    override_missing_value: List[Tuple[str, str, str]] = []  # (key, formulation_id, field_name)
    empty_evidence: List[Tuple[str, str, str]] = []  # (key, formulation_id, field_name)

    for r in rows:
        field = normalize_field_name(r.get("field_name", ""))
        if field in excluded:
            continue

        d = (r.get("gt_decision", "") or "").strip()
        overall[d] += 1
        by_field[field][d] += 1

        key = (r.get("key", "") or "").strip()
        fid = (r.get("formulation_id", "") or "").strip()

        if d and d not in allowed:
            invalid_decisions.append((key, fid, d))

        if d == "override":
            v = (r.get("gt_value_text", "") or "").strip()
            if not v:
                override_missing_value.append((key, fid, field))

        if evidence_col is not None:
            ev = (r.get(evidence_col, "") or "").strip()
            if not ev:
                empty_evidence.append((key, fid, field))

    # Print overall summary
    total = sum(overall.values())
    print(f"[GT] file: {gt_path}")
    print(f"[GT] total rows counted (after excludes): {total}")
    if excluded:
        print(f"[GT] excluded field_name: {sorted(excluded)}")

    print("\nOverall gt_decision counts:")
    for k in allowed + [d for d in sorted(overall.keys()) if d not in allowed]:
        if k in overall:
            n = overall[k]
            pct = (n / total * 100.0) if total else 0.0
            print(f"  {k:>14}: {n:>6}  ({pct:>6.2f}%)")

    # Quality checks
    print("\nQuality checks:")
    print(f"  invalid gt_decision rows: {len(invalid_decisions)}")
    print(f"  override missing gt_value_text: {len(override_missing_value)}")
    if evidence_col is not None:
        print(f"  empty evidence ({evidence_col}): {len(empty_evidence)}")
    else:
        print("  empty evidence: (evidence column not found)")

    # By-field wide table
    fields_sorted = sorted(by_field.keys(), key=lambda s: (s == "", s))
    wide_rows = []
    for f in fields_sorted:
        row = {"field_name": f, "n_total": sum(by_field[f].values())}
        for d in allowed:
            row[d] = by_field[f].get(d, 0)
        # include any unexpected decisions if present
        extras = [d for d in by_field[f].keys() if d not in allowed]
        for d in extras:
            row[f"unexpected::{d}"] = by_field[f][d]
        wide_rows.append(row)

    df_wide = pd.DataFrame(wide_rows)
    # Sort by total descending
    if not df_wide.empty:
        df_wide = df_wide.sort_values(["n_total", "field_name"], ascending=[False, True])

    print("\nTop fields by volume (first 20):")
    if df_wide.empty:
        print("  (none)")
    else:
        cols_show = ["field_name", "n_total"] + allowed
        print(df_wide[cols_show].head(20).to_string(index=False))

    # Optional outputs
    if args.out_by_field_tsv:
        outp = Path(args.out_by_field_tsv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df_wide.to_csv(outp, sep="\t", index=False)
        print(f"\n[OK] wrote by-field table: {outp}")

    if args.out_tsv:
        # tidy long format: field_name, gt_decision, count
        tidy = []
        for f, c in by_field.items():
            for d, n in c.items():
                tidy.append({"field_name": f, "gt_decision": d, "count": n})
        df_tidy = pd.DataFrame(tidy)
        outp = Path(args.out_tsv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df_tidy.to_csv(outp, sep="\t", index=False)
        print(f"[OK] wrote tidy summary: {outp}")

    # If there are critical issues, exit non-zero to fail CI-style checks
    if invalid_decisions or override_missing_value:
        print("\n[FAIL] critical GT issues detected.")
        if invalid_decisions:
            print("  Example invalid decisions (first 10):")
            for x in invalid_decisions[:10]:
                print(f"    key={x[0]} formulation_id={x[1]} gt_decision={x[2]}")
        if override_missing_value:
            print("  Example override missing value (first 10):")
            for x in override_missing_value[:10]:
                print(f"    key={x[0]} formulation_id={x[1]} field_name={x[2]}")
        return 2

    print("\n[OK] GT summary complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
