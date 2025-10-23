#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gt_tool.py

Create ground-truth (GT) annotation templates from weak labels produced by
auto_extract_weak_labels_v2.py. The template pairs each predicted field with
an empty GT column so annotators can correct values instead of starting from scratch.

Main command
------------
  make-template
    Reads weak_labels_v2.jsonl and writes a CSV/TSV with columns:
      key, title,
      pred.<field_1>, gt.<field_1>, pred.<field_2>, gt.<field_2>, ...

    Optionally include Tier-1 bin columns (pred/gt) to support bin-only labeling.

Example
-------
  python gt_tool.py make-template \
    --weak-jsonl data/cleaned/samples/weak_labels_v2.jsonl \
    --out data/cleaned/samples/manual_labels_v2.csv \
    --include-bins \
    --tiers 1,2

File format notes
-----------------
- Input JSONL format (one JSON per line), produced by v2 extractor:
    {
      "key": "...",
      "title": "...",
      "year": "...",
      "doi": "...",
      "fields": { ... raw numeric/text fields ... },
      "bins":   { ... categorical bins for Tier-1 ... },
      "meta":   { "model": "...", "chars_used": ... }
    }

- Output CSV/TSV: Encoded in UTF-8 with BOM if CSV, UTF-8 if TSV.
  You can open it directly in Excel/LibreOffice/VS Code.

"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ---- Default field order (matching v2 script) ----

T1_FIELDS = [
    "plga_mw_kDa",
    "la_ga_ratio",
    "emul_method",
    "emul_time_s",
    "emul_intensity",
    "pva_conc_text",
    "pva_conc_percent",
    "organic_solvent",
    "size_nm",
    "pdi",
    "zeta_mV",
]

T2_FIELDS = [
    "w1_vol_mL", "o_vol_mL", "w2_vol_mL",
    "plga_mass_g", "drug_feed_amount_g",
    "drug_polymer_ratio",
    "encapsulation_efficiency_percent",
    "drug_loading_percent",
]

T3_FIELDS = [
    "aux_materials",
    "organic_solvent_vol_mL",
    "release_profile_type",
    "drug_name",
]

T1_BINS = [
    "plga_mw_kDa_bin",
    "la_ga_ratio_bin",
    "pva_conc_bin",
    "emul_method_bin",
    "emul_time_s_bin",
    "size_nm_bin",
    "pdi_bin",
    "zeta_mV_bin",
]


def parse_tiers(tiers_arg: str) -> List[int]:
    if not tiers_arg:
        return [1, 2, 3]
    out = []
    for t in tiers_arg.split(","):
        t = t.strip()
        if t in {"1","2","3"}:
            out.append(int(t))
    return sorted(set(out)) or [1, 2, 3]


def pick_fields_by_tiers(tiers: List[int]) -> List[str]:
    fields: List[str] = []
    if 1 in tiers:
        fields += T1_FIELDS
    if 2 in tiers:
        fields += T2_FIELDS
    if 3 in tiers:
        fields += T3_FIELDS
    return fields


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def infer_separator(out_path: Path) -> Tuple[str, str]:
    """
    Decide delimiter and encoding by file extension.
    Returns (delimiter, encoding)
    """
    ext = out_path.suffix.lower()
    if ext == ".tsv":
        return ("\t", "utf-8")
    # default CSV
    return (",", "utf-8-sig")  # BOM for Excel friendliness


def make_template(weak_jsonl: Path, out_path: Path, include_bins: bool, tiers: List[int]) -> None:
    data = read_jsonl(weak_jsonl)
    if not data:
        raise SystemExit(f"No data found in: {weak_jsonl}")

    # Assemble column order
    field_list = pick_fields_by_tiers(tiers)
    cols: List[str] = ["key", "title"]

    for f in field_list:
        cols += [f"pred.{f}", f"gt.{f}"]

    bin_cols: List[str] = []
    if include_bins:
        for b in T1_BINS:
            bin_cols += [f"pred.{b}", f"gt.{b}"]
        cols += bin_cols

    # Write table
    delim, enc = infer_separator(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding=enc) as wf:
        import csv as _csv
        w = _csv.writer(wf, delimiter=delim)
        w.writerow(cols)

        for rec in data:
            key = rec.get("key", "")
            title = rec.get("title", "")
            fields = rec.get("fields", {}) or {}
            bins = rec.get("bins", {}) or {}

            row = [key, title]
            # predicted + empty GT
            for f in field_list:
                row.append(fields.get(f, ""))
                row.append("")  # gt empty for annotator

            if include_bins:
                for b in T1_BINS:
                    row.append(bins.get(b, ""))
                    row.append("")

            w.writerow(row)

    print(f"[OK] Wrote template -> {out_path}  (columns: {len(cols)})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make-template", help="Create a GT CSV/TSV from weak_labels_v2.jsonl.")
    mk.add_argument("--weak-jsonl", required=True, help="Path to weak_labels_v2.jsonl")
    mk.add_argument("--out", required=True, help="Output CSV/TSV path")
    mk.add_argument("--include-bins", action="store_true", help="Include Tier-1 bin columns (pred/gt)")
    mk.add_argument("--tiers", default="1,2,3", help="Which tiers to include, e.g., '1' or '1,2' or '1,2,3'")

    args = ap.parse_args()

    if args.cmd == "make-template":
        make_template(Path(args.weak_jsonl), Path(args.out), args.include_bins, parse_tiers(args.tiers))


if __name__ == "__main__":
    main()
