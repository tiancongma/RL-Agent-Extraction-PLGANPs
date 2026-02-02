#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multi_model_consensus_vote.py

Purpose
-------
Given a *multi-model* extraction TSV (one row per key/formulation_id/model),
produce three artifacts for a publishable, low-manual-intervention workflow:

1) consensus weak-label TSV (one row per key/formulation_id)
   - Conservative field-level "consensus filtering" (not majority voting):
     * agree -> take preferred model value (or either if only one present)
     * conflict -> leave consensus empty, keep both model values, mark conflict
   - Adds risk / triage columns to support selective human GT (conflict-only).

2) conflict queue TSV (one row per key/formulation_id)
   - Only rows with has_any_conflict=1
   - Includes both models' extracted values and both models' evidence columns
     so you can do quick adjudication without re-opening the paper.

3) field-level QC summary TSV
   - Counts of agree/conflict/only_model1/only_model2/both_empty per field

Design constraints
------------------
- No API calls. No retries. Deterministic given input TSV.
- Works for 2 models only (intentionally). For >2, run pairwise or extend later.
- Keeps all data as strings to avoid lossy casting. Numeric comparisons use float
  parsing with a relative tolerance for "close".

Evidence handling
-----------------
Input contains evidence fields per model row (evidence_section, evidence_span_text,
evidence_span_start, evidence_span_end, evidence_method, evidence_quality).

Consensus output carries:
- evidence_*_model1 and evidence_*_model2 (for audit)
- evidence_*_main chosen from the row that contributed the consensus for *key fields*,
  following the same preferred-model policy used for numeric "agree" merges.
If a formulation has no consensus for any key field, evidence_*_main is left empty.

Usage (example)
---------------
python src/stage4_eval/multi_model_consensus_vote.py ^
  --input data/results/run_*/weak_labels__gemini_gemma.tsv ^
  --out-consensus data/results/run_*/formulations_consensus_weak.tsv ^
  --out-conflicts data/results/run_*/formulations_conflict_queue.tsv ^
  --out-qc data/results/run_*/field_level_qc_summary.tsv ^
  --preferred-model "gemini-2.5-flash"

Notes
-----
- If --model1/--model2 are not provided, they are inferred from the first two
  distinct values in the 'model' column (stable order in file).
"""

import argparse
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------
# Utility
# ---------------------------

EVIDENCE_COLS = [
    "evidence_section",
    "evidence_span_text",
    "evidence_span_start",
    "evidence_span_end",
    "evidence_method",
    "evidence_quality",
]

BASE_FIELDS_TEXT = [
    "emul_type",
    "emul_method",
    "la_ga_ratio",
    "drug_name",
    "drug_feed_amount_text",
    "organic_solvent",
    "notes",
]

BASE_FIELDS_NUMERIC = [
    "plga_mw_kDa",
    "plga_mass_mg",
    "pva_conc_percent",
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
]

ALL_FIELDS = BASE_FIELDS_TEXT + BASE_FIELDS_NUMERIC

# "Key fields" drive evidence_main selection and GT prioritization.
KEY_FIELDS = [
    "emul_type",
    "emul_method",
    "plga_mass_mg",
    "organic_solvent",
    "drug_name",
    "size_nm",
]


def is_nan_or_empty(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    if isinstance(x, str) and x.strip().lower() == "nan":
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
    return mapping.get(s, s.upper())


def normalize_emul_method(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = " ".join(s.split())
    return s


def normalize_solvent(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    mapping = {
        "dcm": "DCM",
        "dichloromethane": "DCM",
        "dichloromethane (dcm)": "DCM",
        "methylene chloride": "Methylene chloride",
        "acetone": "Acetone",
    }
    return mapping.get(s, s)


def normalize_generic_text(s: str) -> str:
    if s is None:
        return ""
    return s.strip()


def compare_text(v1: Any, v2: Any, field: str) -> Tuple[str, str, str]:
    """
    Returns (status, norm1, norm2), where status in:
      - both_empty
      - only_model1
      - only_model2
      - agree
      - conflict
    """
    if is_nan_or_empty(v1) and is_nan_or_empty(v2):
        return "both_empty", "", ""

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

    if n1 and not n2:
        return "only_model1", n1, ""
    if n2 and not n1:
        return "only_model2", "", n2
    if n1 == n2:
        return "agree", n1, n2
    return "conflict", n1, n2


def derive_particle_scale(size_val: Any) -> str:
    v = to_float_or_nan(size_val)
    if math.isnan(v) or v <= 0:
        return ""
    if v < 1000:
        return "nano"
    if v < 10000:
        return "submicro/micro"
    return "micro"


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["key", "model", "formulation_id"]:
        if col not in df.columns:
            raise ValueError(f"Input TSV must contain required column '{col}'.")

    for col in ALL_FIELDS:
        if col not in df.columns:
            df[col] = ""

    # Evidence columns are optional but strongly expected. If missing, create empties.
    for col in EVIDENCE_COLS:
        if col not in df.columns:
            df[col] = ""

    # Keep strings
    df = df.fillna("")
    return df


def pick_evidence_main(
    preferred_model: str,
    model1: str,
    model2: str,
    status_by_field: Dict[str, str],
    value_source_by_field: Dict[str, str],
    ev1: Dict[str, Any],
    ev2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Choose a single evidence_main block for the formulation to show in GT view.
    Policy: if any KEY_FIELDS has consensus (agree or only_*), choose evidence from
    the model that contributed the consensus for the first such key field in order.
    If that source is ambiguous (agree), use preferred_model.
    """
    for f in KEY_FIELDS:
        st = status_by_field.get(f, "both_empty")
        if st in ("agree", "only_model1", "only_model2"):
            src = value_source_by_field.get(f, "")
            if st == "agree":
                src = preferred_model
            if src == model1:
                return ev1
            if src == model2:
                return ev2
            # fallback: preferred_model
            return ev1 if preferred_model == model1 else ev2
    return {c: "" for c in EVIDENCE_COLS}


def process(
    df: pd.DataFrame,
    model1: str,
    model2: str,
    preferred_model: str,
    numeric_rel_tol: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = ensure_columns(df)

    # Uniqueness check (strong contract)
    dup = df.duplicated(subset=["key", "formulation_id", "model"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["key", "formulation_id", "model"]].sort_values(["key", "formulation_id", "model"])
        raise ValueError(
            "Duplicate rows detected for (key, formulation_id, model). "
            "Fix upstream before consensus.\n"
            f"Examples (first 20):\n{bad.head(20).to_string(index=False)}"
        )

    qc_stats: Dict[str, Dict[str, int]] = {
        f: {"both_empty": 0, "only_model1": 0, "only_model2": 0, "agree": 0, "conflict": 0}
        for f in ALL_FIELDS
    }

    consensus_rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []

    grouped = df.groupby(["key", "formulation_id"], dropna=False)

    for (key_val, form_id), g in grouped:
        r1 = g.loc[g["model"] == model1].head(1)
        r2 = g.loc[g["model"] == model2].head(1)

        has_m1 = len(r1) == 1
        has_m2 = len(r2) == 1

        # Pull evidence blocks per model (even if missing model row -> empty)
        ev1 = {c: (r1[c].iloc[0] if has_m1 else "") for c in EVIDENCE_COLS}
        ev2 = {c: (r2[c].iloc[0] if has_m2 else "") for c in EVIDENCE_COLS}

        row: Dict[str, Any] = {
            "key": key_val,
            "formulation_id": form_id,
            "model1": model1,
            "model2": model2,
            "preferred_model": preferred_model,
        }

        # Attach per-model evidence for audit
        for c in EVIDENCE_COLS:
            row[f"{c}_model1"] = ev1[c]
            row[f"{c}_model2"] = ev2[c]

        any_conflict = False
        conflict_fields: List[str] = []
        status_by_field: Dict[str, str] = {}
        value_source_by_field: Dict[str, str] = {}  # model1/model2/""

        for field in ALL_FIELDS:
            v1 = r1[field].iloc[0] if has_m1 else ""
            v2 = r2[field].iloc[0] if has_m2 else ""

            row[f"{field}_model1"] = v1
            row[f"{field}_model2"] = v2

            if field in BASE_FIELDS_NUMERIC:
                f1 = to_float_or_nan(v1)
                f2 = to_float_or_nan(v2)

                if math.isnan(f1) and math.isnan(f2):
                    st = "both_empty"
                    main = ""
                    conflict = 0
                    src = ""
                elif math.isnan(f2) and not math.isnan(f1):
                    st = "only_model1"
                    main = str(f1).rstrip("0").rstrip(".") if "." in str(f1) else str(f1)
                    conflict = 0
                    src = model1
                elif math.isnan(f1) and not math.isnan(f2):
                    st = "only_model2"
                    main = str(f2).rstrip("0").rstrip(".") if "." in str(f2) else str(f2)
                    conflict = 0
                    src = model2
                else:
                    if numeric_close(f1, f2, rel_tol=numeric_rel_tol):
                        st = "agree"
                        # choose preferred model value for main
                        preferred_val = f1 if preferred_model == model1 else f2
                        main = str(preferred_val).rstrip("0").rstrip(".") if "." in str(preferred_val) else str(preferred_val)
                        conflict = 0
                        src = preferred_model
                    else:
                        st = "conflict"
                        main = ""
                        conflict = 1
                        src = ""

                row[f"{field}_main"] = main
                row[f"{field}_conflict"] = conflict
                row[f"{field}_status"] = st

            else:
                st, n1, n2 = compare_text(v1, v2, field)
                if st == "only_model1":
                    main = n1
                    conflict = 0
                    src = model1
                elif st == "only_model2":
                    main = n2
                    conflict = 0
                    src = model2
                elif st == "agree":
                    # use normalized agree value; provenance is preferred_model
                    main = n1
                    conflict = 0
                    src = preferred_model
                elif st == "both_empty":
                    main = ""
                    conflict = 0
                    src = ""
                else:
                    main = ""
                    conflict = 1
                    src = ""

                row[f"{field}_main"] = main
                row[f"{field}_conflict"] = conflict
                row[f"{field}_status"] = st

            qc_stats[field][row[f"{field}_status"]] += 1
            status_by_field[field] = row[f"{field}_status"]
            value_source_by_field[field] = src

            if row[f"{field}_conflict"] == 1:
                any_conflict = True
                conflict_fields.append(field)

        row["has_any_conflict"] = 1 if any_conflict else 0
        row["conflict_fields"] = ";".join(conflict_fields)
        row["n_conflict_fields"] = str(len(conflict_fields))

        # Derived from size_nm_main
        row["particle_scale"] = derive_particle_scale(row.get("size_nm_main", ""))

        # Evidence main chosen by first resolvable KEY_FIELD
        ev_main = pick_evidence_main(preferred_model, model1, model2, status_by_field, value_source_by_field, ev1, ev2)
        for c in EVIDENCE_COLS:
            row[f"{c}_main"] = ev_main[c]

        # Minimal triage label for GT
        # - HIGH: any_conflict OR any key field missing main
        # - LOW: no conflict and all key fields present in main (can still be incomplete elsewhere)
        key_missing = [f for f in KEY_FIELDS if is_nan_or_empty(row.get(f"{f}_main", ""))]
        row["key_fields_missing"] = ";".join(key_missing)
        row["risk_level"] = "HIGH" if (any_conflict or len(key_missing) > 0) else "LOW"

        consensus_rows.append(row)
        if any_conflict:
            conflict_rows.append(row)

    consensus_df = pd.DataFrame(consensus_rows)
    conflicts_df = pd.DataFrame(conflict_rows)

    qc_records: List[Dict[str, Any]] = []
    for f, stats in qc_stats.items():
        rec = {"field": f}
        rec.update(stats)
        rec["total_formulations"] = sum(stats.values())
        # helpful rates
        total = rec["total_formulations"] or 1
        rec["conflict_rate"] = rec["conflict"] / total
        rec["agree_rate"] = rec["agree"] / total
        qc_records.append(rec)
    qc_df = pd.DataFrame(qc_records).sort_values(["conflict_rate", "field"], ascending=[False, True])

    # Stable column order
    leading = ["key", "formulation_id", "has_any_conflict", "risk_level", "n_conflict_fields", "conflict_fields",
               "key_fields_missing", "particle_scale", "model1", "model2", "preferred_model"]
    # Evidence blocks
    evidence_block = []
    for c in EVIDENCE_COLS:
        evidence_block.extend([f"{c}_main", f"{c}_model1", f"{c}_model2"])
    # Field blocks
    field_block = []
    for f in ALL_FIELDS:
        field_block.extend([f"{f}_main", f"{f}_status", f"{f}_conflict", f"{f}_model1", f"{f}_model2"])

    cols = [c for c in leading + evidence_block + field_block if c in consensus_df.columns]
    rest = [c for c in consensus_df.columns if c not in cols]
    consensus_df = consensus_df[cols + rest]

    conflicts_df = conflicts_df[cols + rest]

    return consensus_df, conflicts_df, qc_df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build conservative consensus weak labels + conflict queue + QC summary.")
    p.add_argument("--input", required=True, help="Input multi-model TSV (one row per key/formulation_id/model).")
    p.add_argument("--out-consensus", required=True, help="Output consensus weak-label TSV.")
    p.add_argument("--out-conflicts", required=True, help="Output conflict queue TSV (conflict-only).")
    p.add_argument("--out-qc", required=True, help="Output field-level QC summary TSV.")
    p.add_argument("--model1", default=None, help="First model name (optional).")
    p.add_argument("--model2", default=None, help="Second model name (optional).")
    p.add_argument("--preferred-model", default=None, help="Preferred model when values agree (optional).")
    p.add_argument("--numeric-rel-tol", type=float, default=0.2, help="Relative tolerance for numeric agreement (default 0.2).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    print(f"[INFO] Loading: {args.input}")
    df = pd.read_csv(args.input, sep="\t", dtype=str)

    models = [m for m in df["model"].dropna().unique().tolist() if str(m).strip() != ""]
    if len(models) < 2 and (args.model1 is None or args.model2 is None):
        raise ValueError("Need at least two distinct 'model' values in input TSV, or provide --model1 and --model2.")

    model1 = args.model1 or models[0]
    model2 = args.model2 or models[1]
    preferred = args.preferred_model or model1

    print(f"[INFO] model1           = {model1}")
    print(f"[INFO] model2           = {model2}")
    print(f"[INFO] preferred_model  = {preferred}")
    print(f"[INFO] numeric_rel_tol  = {args.numeric_rel_tol}")

    consensus_df, conflicts_df, qc_df = process(
        df=df,
        model1=model1,
        model2=model2,
        preferred_model=preferred,
        numeric_rel_tol=args.numeric_rel_tol,
    )

    print(f"[INFO] Writing consensus: {args.out_consensus}")
    consensus_df.to_csv(args.out_consensus, sep="\t", index=False)

    print(f"[INFO] Writing conflicts: {args.out_conflicts}")
    conflicts_df.to_csv(args.out_conflicts, sep="\t", index=False)

    print(f"[INFO] Writing QC: {args.out_qc}")
    qc_df.to_csv(args.out_qc, sep="\t", index=False)

    print(f"[OK] consensus rows: {len(consensus_df)}")
    print(f"[OK] conflict rows : {len(conflicts_df)}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
