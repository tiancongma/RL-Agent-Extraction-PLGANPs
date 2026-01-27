#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
multi_model_merge_qc.py

Purpose
-------
Given a multi-model extraction TSV (one row per key/formulation_id/model),
this script:

1. Computes per-field agreement / conflict statistics between two models.
2. Produces a "merged" weak-label TSV with:
   - One row per (key, formulation_id)
   - For each field:
       <field>_main      : merged value (voting / conservative)
       <field>_model1    : value from model1 (can be empty)
       <field>_model2    : value from model2 (can be empty)
       <field>_conflict  : 0/1 (1 means both non-empty and disagree)
   - Additional derived fields such as particle_scale and has_any_conflict.

Usage (example, Windows)
------------------------
# Minimal example (adjust paths as needed):
# .\\.venv\\Scripts\\python.exe .\\scripts\\multi_model_merge_qc.py ^
#   --input .\\data\\cleaned\\weak_labels_multi_model.tsv ^
#   --out-merged .\\data\\cleaned\\weak_labels_merged_v2.tsv ^
#   --out-qc .\\data\\cleaned\\weak_labels_merged_v2_qc.tsv ^
#   --preferred-model "gemini-2.5-flash"

Assumptions
-----------
- Input TSV has at least these columns:
    key, model, formulation_id, emul_type, emul_method, plga_mw_kDa,
    la_ga_ratio, plga_mass_mg, drug_name, drug_feed_amount_text,
    pva_conc_percent, organic_solvent, size_nm, pdi, zeta_mV,
    encapsulation_efficiency_percent, loading_content_percent, notes

- There is at most one row per (key, formulation_id, model).

Design choices
--------------
- For numeric fields:
    * If both models have values and they are "close" (relative difference < 20%),
      we treat them as agree; main value prefers the preferred model.
    * If both models have values and they differ a lot, we mark conflict=1 and
      leave the main field empty (conservative), but keep model1/model2 values.
    * If only one model has a value, we use that as main and conflict=0.

- For text fields:
    * We normalize and compare. If normalized equal → agree; main is normalized.
    * If only one non-empty → main is that; conflict=0.
    * If both non-empty and different → conflict=1, main left empty.

- particle_scale is derived from main size_nm:
    * size_nm < 1000         → "nano"
    * 1000 ≤ size_nm < 10000 → "submicro/micro"
    * size_nm ≥ 10000        → "micro"
    * missing / invalid      → ""

- QC summary TSV:
    For each field, reports counts of:
      total_with_any_value, both_nonempty, agree, conflict,
      only_model1, only_model2.

"""

import argparse
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------
# Utility functions
# ---------------------------

def is_nan_or_empty(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def to_float_or_nan(x: Any) -> float:
    if is_nan_or_empty(x):
        return float("nan")
    try:
        return float(str(x).strip())
    except Exception:
        return float("nan")


def numeric_close(v1: float, v2: float, rel_tol: float = 0.2) -> bool:
    """Return True if v1 and v2 are "close enough" numerically."""
    if math.isnan(v1) or math.isnan(v2):
        return False
    if v1 == v2:
        return True
    denom = max(abs(v1), abs(v2), 1e-12)
    rel_diff = abs(v1 - v2) / denom
    return rel_diff <= rel_tol


def normalize_emul_type(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # Unify some common variants
    mapping = {
        "o/w": "O/W",
        "oil-in-water (o/w)": "O/W",
        "oil-in-water": "O/W",
        "w/o": "W/O",
        "w/o/w": "W/O/W",
        "w1/o/w2": "W1/O/W2",
        "w1/o/w2 double emulsion": "W1/O/W2",
        "w1/o/w2 double-emulsion": "W1/O/W2",
        "w1/o/w2 double emulsion solvent evaporation": "W1/O/W2",
        "o/w emulsion": "O/W",
        "o/w double emulsion": "O/W",
    }
    if s in mapping:
        return mapping[s]
    # Fallback: uppercase raw
    return s.upper()


def normalize_emul_method(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # very light normalization; you can extend later if needed
    s = s.replace("  ", " ")
    return s


def normalize_solvent(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    mapping = {
        "dcm": "DCM",
        "dichloromethane": "DCM",
        "methylene chloride": "Methylene chloride",
        "ac": "Acetone",
        "acetone": "Acetone",
    }
    if s in mapping:
        return mapping[s]
    return s


def normalize_generic_text(s: str) -> str:
    if s is None:
        return ""
    return s.strip()


def compare_text(v1: Any, v2: Any, field: str) -> Tuple[bool, str, str, str]:
    """
    Compare two text-like values.
    Returns (agree, main_value, norm1, norm2).
    """
    if is_nan_or_empty(v1) and is_nan_or_empty(v2):
        return True, "", "", ""

    # Normalization per field
    if field == "emul_type":
        n1 = normalize_emul_type(str(v1)) if not is_nan_or_empty(v1) else ""
        n2 = normalize_emul_type(str(v2)) if not is_nan_or_empty(v2) else ""
    elif field == "emul_method":
        n1 = normalize_emul_method(str(v1)) if not is_nan_or_empty(v1) else ""
        n2 = normalize_emul_method(str(v2)) if not is_nan_or_empty(v2) else ""
    elif field == "organic_solvent":
        n1 = normalize_solvent(str(v1)) if not is_nan_or_empty(v1) else ""
        n2 = normalize_solvent(str(v2)) if not is_nan_or_empty(v2) else ""
    else:
        n1 = normalize_generic_text(str(v1)) if not is_nan_or_empty(v1) else ""
        n2 = normalize_generic_text(str(v2)) if not is_nan_or_empty(v2) else ""

    # Only one side has value → no conflict
    if n1 and not n2:
        return True, n1, n1, n2
    if n2 and not n1:
        return True, n2, n1, n2

    # Both non-empty
    if n1 == n2:
        return True, n1, n1, n2
    else:
        # conflict, main value left empty (conservative)
        return False, "", n1, n2


def derive_particle_scale(size_val: Any) -> str:
    v = to_float_or_nan(size_val)
    if math.isnan(v) or v <= 0:
        return ""
    if v < 1000:
        return "nano"
    elif v < 10000:
        return "submicro/micro"
    else:
        return "micro"


# ---------------------------
# Core processing
# ---------------------------

def process(df: pd.DataFrame,
            model1: str,
            model2: str,
            preferred_model: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (merged_df, qc_df)
    """

    base_fields_text = [
        "emul_type",
        "emul_method",
        "la_ga_ratio",
        "drug_name",
        "drug_feed_amount_text",
        "organic_solvent",
        "notes",
    ]

    base_fields_numeric = [
        "plga_mw_kDa",
        "plga_mass_mg",
        "pva_conc_percent",
        "size_nm",
        "pdi",
        "zeta_mV",
        "encapsulation_efficiency_percent",
        "loading_content_percent",
    ]

    all_fields = base_fields_text + base_fields_numeric

    # Safety: ensure missing columns exist
    for col in ["key", "model", "formulation_id"]:
        if col not in df.columns:
            raise ValueError(f"Input TSV must contain column '{col}'.")

    # Ensure the base fields exist (if missing, create empty columns)
    for col in all_fields:
        if col not in df.columns:
            df[col] = ""

    # We'll accumulate merged rows here
    merged_rows: List[Dict[str, Any]] = []

    # QC stats: field -> counters
    qc_stats: Dict[str, Dict[str, int]] = {
        f: {
            "total_with_any_value": 0,
            "both_nonempty": 0,
            "agree": 0,
            "conflict": 0,
            "only_model1": 0,
            "only_model2": 0,
        }
        for f in all_fields
    }

    # Group by key + formulation_id
    group_cols = ["key", "formulation_id"]
    df_grouped = df.groupby(group_cols, dropna=False)

    for (key_val, form_id), g in df_grouped:
        # Pick row for each model if exists
        row_m1 = g.loc[g["model"] == model1].head(1)
        row_m2 = g.loc[g["model"] == model2].head(1)

        # If a model's row does not exist, we'll treat its values as empty
        has_m1 = len(row_m1) == 1
        has_m2 = len(row_m2) == 1

        row_dict: Dict[str, Any] = {
            "key": key_val,
            "formulation_id": form_id,
            "model1": model1,
            "model2": model2,
        }

        any_conflict = False

        for field in all_fields:
            col_main = f"{field}_main"
            col_m1 = f"{field}_model1"
            col_m2 = f"{field}_model2"
            col_conflict = f"{field}_conflict"

            v1 = row_m1[field].iloc[0] if has_m1 else ""
            v2 = row_m2[field].iloc[0] if has_m2 else ""

            # Update QC total_with_any_value
            if not is_nan_or_empty(v1) or not is_nan_or_empty(v2):
                qc_stats[field]["total_with_any_value"] += 1

            # Model availability stats
            if not is_nan_or_empty(v1) and is_nan_or_empty(v2):
                qc_stats[field]["only_model1"] += 1
            elif is_nan_or_empty(v1) and not is_nan_or_empty(v2):
                qc_stats[field]["only_model2"] += 1
            elif not is_nan_or_empty(v1) and not is_nan_or_empty(v2):
                qc_stats[field]["both_nonempty"] += 1

            # Store raw per-model values
            row_dict[col_m1] = v1
            row_dict[col_m2] = v2

            # Decide main and conflict
            if field in base_fields_numeric:
                f1 = to_float_or_nan(v1)
                f2 = to_float_or_nan(v2)

                if math.isnan(f1) and math.isnan(f2):
                    # Both empty
                    row_dict[col_main] = ""
                    row_dict[col_conflict] = 0
                    qc_stats[field]["agree"] += 1  # trivial agree
                elif math.isnan(f2) and not math.isnan(f1):
                    # Only model1
                    row_dict[col_main] = f1
                    row_dict[col_conflict] = 0
                    qc_stats[field]["agree"] += 1
                elif math.isnan(f1) and not math.isnan(f2):
                    # Only model2
                    row_dict[col_main] = f2
                    row_dict[col_conflict] = 0
                    qc_stats[field]["agree"] += 1
                else:
                    # Both have numeric values
                    if numeric_close(f1, f2, rel_tol=0.2):
                        # Near enough; prefer preferred_model for main
                        preferred_val = f1 if preferred_model == model1 else f2
                        row_dict[col_main] = preferred_val
                        row_dict[col_conflict] = 0
                        qc_stats[field]["agree"] += 1
                    else:
                        # Conflict: keep both as-is, main left empty
                        row_dict[col_main] = ""
                        row_dict[col_conflict] = 1
                        qc_stats[field]["conflict"] += 1
                        any_conflict = True

            else:
                # Text-like field
                agree, main_val, norm1, norm2 = compare_text(v1, v2, field)
                row_dict[col_main] = main_val
                # conflict flag: both non-empty and disagree
                if agree:
                    row_dict[col_conflict] = 0
                    qc_stats[field]["agree"] += 1
                else:
                    row_dict[col_conflict] = 1
                    qc_stats[field]["conflict"] += 1
                    any_conflict = True

        # Derived fields (e.g., particle_scale, has_any_conflict)
        size_main = row_dict.get("size_nm_main", "")
        row_dict["particle_scale"] = derive_particle_scale(size_main)
        row_dict["has_any_conflict"] = 1 if any_conflict else 0

        merged_rows.append(row_dict)

    merged_df = pd.DataFrame(merged_rows)

    # Build QC summary DF
    qc_records = []
    for field, stats in qc_stats.items():
        record = {"field": field}
        record.update(stats)
        qc_records.append(record)
    qc_df = pd.DataFrame(qc_records)

    return merged_df, qc_df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multi-model weak-label TSV and compute per-field QC."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input TSV with columns [key, model, formulation_id, ...].",
    )
    parser.add_argument(
        "--out-merged",
        required=True,
        help="Output TSV for merged 'voting' weak labels.",
    )
    parser.add_argument(
        "--out-qc",
        required=True,
        help="Output TSV for per-field agreement/conflict summary.",
    )
    parser.add_argument(
        "--model1",
        default=None,
        help="Name of first model (defaults to first distinct model in file).",
    )
    parser.add_argument(
        "--model2",
        default=None,
        help="Name of second model (defaults to second distinct model in file).",
    )
    parser.add_argument(
        "--preferred-model",
        default=None,
        help="Model name preferred when two numeric values are close. "
             "If not provided, defaults to model1.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    print(f"[INFO] Loading TSV: {args.input}")
    df = pd.read_csv(args.input, sep="\t", dtype=str)

    # Cast numeric-looking columns later; for now keep as string
    models = list(df["model"].dropna().unique())
    if len(models) < 1:
        raise ValueError("No 'model' values found in input TSV.")

    model1 = args.model1 or (models[0] if len(models) >= 1 else None)
    model2 = args.model2 or (models[1] if len(models) >= 2 else None)

    if model2 is None:
        raise ValueError(
            "Input TSV contains fewer than 2 distinct models. "
            "Need at least two for consistency comparison."
        )

    preferred_model = args.preferred_model or model1

    print(f"[INFO] Using model1      = {model1}")
    print(f"[INFO] Using model2      = {model2}")
    print(f"[INFO] Preferred model   = {preferred_model}")

    merged_df, qc_df = process(df, model1, model2, preferred_model)

    print(f"[INFO] Writing merged TSV to: {args.out_merged}")
    merged_df.to_csv(args.out_merged, sep="\t", index=False)

    print(f"[INFO] Writing QC summary TSV to: {args.out_qc}")
    qc_df.to_csv(args.out_qc, sep="\t", index=False)

    print("[INFO] Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
