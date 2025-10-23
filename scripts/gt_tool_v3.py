#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gt_tool_v3.py

Create a per-formulation Ground-Truth (GT) annotation template from
weak_labels_v3.jsonl produced by auto_extract_weak_labels_v3.py.

Design goals:
- Annotators correct ONLY the RAW predictions (do not hand-calculate).
- DERIVED values are computed later by code; GT should not contain manual math.
- Optional Tier-1 bin columns can be included for bin-only labeling.
- A header note row reminds annotators not to compute by hand.
- Default output is TSV (recommended for Excel to avoid time-format issues with "50:50").
- Optional --excel-safe-ratios wraps pred.* ratio fields as ="50:50" to prevent Excel auto-formatting.

Usage examples:
  python gt_tool_v3.py make-template ^
    --weak-jsonl data/cleaned/samples/weak_labels_v3.jsonl ^
    --out data/cleaned/samples/manual_labels_v3.tsv ^
    --include-bins ^
    --tiers 1,2 ^
    --excel-safe-ratios
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ----- Fields to annotate (RAW) -----
T1_FIELDS = [
    "emul_type",
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
    # RAW volumes & ratios
    "w1_vol", "o_vol", "w2_vol",
    "w1_vol_mL", "o_vol_mL", "w2_vol_mL",
    "w1_o_ratio_text", "w1_o_ratio_norm",
    "o_w2_ratio_text", "o_w2_ratio_norm",
    "w1_o_w2_ratio_text", "w1_o_w2_ratio_norm",
    "total_phase_vol_text", "total_phase_vol_mL",
    # Mass/efficiency/context
    "plga_mass_g", "drug_feed_amount_g",
    "drug_polymer_ratio",
    "encapsulation_efficiency_percent",
    "drug_loading_percent",
    "aux_materials",
    "organic_solvent_vol_mL",
    "release_profile_type",
    "drug_name",
    # Applicability mask (useful to lock semantics in GT)
    "w1_vol_applicable","o_vol_applicable","w2_vol_applicable",
]

# Optional bins (Tier-1 only) for partial credit workflows
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

HEADER_NOTE = (
    "NOTE: Fill ONLY gt.* columns. Do NOT compute volumes by hand. "
    "Leave blank for missing; use ND/NA/UNK where appropriate. "
    "If ratios are given (e.g., '50:50'), record them in *_ratio_*; "
    "the code will compute *_derived volumes deterministically."
)

# --------- Helpers ---------
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
    # T3 (context) generally not required for GT; omit unless you want to add later
    return fields

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # ignore malformed lines silently
                continue
    return rows

def infer_separator_and_encoding(out_path: Path) -> Tuple[str, str]:
    """
    If .tsv -> (tab, utf-8) ; if .csv -> (comma, utf-8-sig for Excel)
    """
    ext = out_path.suffix.lower()
    if ext == ".tsv":
        return ("\t", "utf-8")
    return (",", "utf-8-sig")

def as_excel_text(val: str) -> str:
    """
    Force Excel to treat a value as text when opening CSV/TSV:
    ="value"
    Useful to prevent '50:50' -> time conversion.
    """
    if val is None or val == "":
        return ""
    return f'="{val}"'

def is_ratio_field(name: str) -> bool:
    return name.endswith("_ratio_text") or name.endswith("_ratio_norm")

# --------- Core: write template ---------
def make_template(
    weak_jsonl: Path,
    out_path: Path,
    include_bins: bool,
    tiers: List[int],
    excel_safe_ratios: bool,
) -> None:
    data = read_jsonl(weak_jsonl)
    if not data:
        raise SystemExit(f"No data found in: {weak_jsonl}")

    field_list = pick_fields_by_tiers(tiers)

    # Column order
    cols: List[str] = ["key", "title", "formulation_id", "note"]
    for f in field_list:
        cols += [f"pred.{f}", f"gt.{f}"]
    if include_bins:
        for b in T1_BINS:
            cols += [f"pred.{b}", f"gt.{b}"]

    delim, enc = infer_separator_and_encoding(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding=enc) as wf:
        w = csv.writer(wf, delimiter=delim)
        # header row
        w.writerow(cols)

        for paper in data:
            key = paper.get("key","")
            title = paper.get("title","")
            # New extractor format: each line is a per-paper object with 'formulations'
            formulations = paper.get("formulations", []) or []

            for f in formulations:
                fid = f.get("id","")
                fields = (f.get("fields") or {})
                bins = (f.get("bins") or {})

                row: List[str] = [str(key), str(title), str(fid)]
                # Put the note only for the first formulation of each paper; else empty
                note_text = HEADER_NOTE if str(fid) in ("1","1.0") else ""
                row.append(note_text)

                # pred / gt field pairs
                for field in field_list:
                    pred_val = fields.get(field, "")
                    # Excel-safe option for ratios (pred side)
                    if excel_safe_ratios and is_ratio_field(field):
                        pred_val = as_excel_text(str(pred_val))
                    row.append(pred_val)  # pred.field
                    row.append("")        # gt.field (left blank for annotators)

                if include_bins:
                    for b in T1_BINS:
                        pred_bin = bins.get(b, "")
                        if excel_safe_ratios and is_ratio_field(b):
                            # bins are not ratios, but keep symmetry in case of future fields
                            pred_bin = as_excel_text(str(pred_bin))
                        row.append(pred_bin)  # pred.bin
                        row.append("")        # gt.bin (optional annotation)

                w.writerow(row)

    print(f"[OK] Wrote template -> {out_path} (columns: {len(cols)})")
    print("Tips:")
    print(" - Prefer TSV and import with Excel's 'Data -> From Text/CSV', set ratio columns to 'Text'.")
    if excel_safe_ratios:
        print(" - Pred ratio columns were written as =\"A:B\" to avoid Excel time conversion.")

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make-template", help="Create a per-formulation GT TSV/CSV from weak_labels_v3.jsonl.")
    mk.add_argument("--weak-jsonl", required=True, help="Path to weak_labels_v3.jsonl")
    mk.add_argument("--out", required=True, help="Output path (.tsv recommended)")
    mk.add_argument("--include-bins", action="store_true", help="Include Tier-1 bin columns (pred/gt)")
    mk.add_argument("--tiers", default="1,2", help="Which tiers to include in GT: '1', '1,2' (default), or '1,2,3'")
    mk.add_argument("--excel-safe-ratios", action="store_true",
                    help="Wrap pred.* ratio fields as =\"A:B\" to prevent Excel auto-formatting (use with CSV/TSV).")

    args = ap.parse_args()

    if args.cmd == "make-template":
        make_template(
            Path(args.weak_jsonl),
            Path(args.out),
            args.include_bins,
            parse_tiers(args.tiers),
            args.excel_safe_ratios,
        )

if __name__ == "__main__":
    main()
