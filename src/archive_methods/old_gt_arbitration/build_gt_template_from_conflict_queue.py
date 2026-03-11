#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_gt_template_from_conflict_queue.py

Purpose
-------
Convert `formulations_conflict_queue.tsv` (one row per key/formulation_id) into a
FIELD-LEVEL GT template where each row is a single (key, formulation_id, field)
decision task.

This implements the policy you chose:
- GT is a *field-level decision*, not a forced "true value" fill.
- Only conflicting fields are sent to human GT.
- If evidence is insufficient, label as `unclear` instead of guessing.

Inputs
------
A conflict-queue TSV produced by `multi_model_consensus_vote.py`, containing at least:
- key, formulation_id, conflict_fields
- per-field columns: <field>_model1 and <field>_model2
- evidence columns: evidence_*_main (recommended; model1/model2 evidence also optional)

Outputs
-------
A TSV where each row is a single GT decision task:
- key, formulation_id, field_name
- candidate values from model1/model2
- evidence span (main + optionally per-model)
- gt_decision (empty for you to fill): accept_model1 / accept_model2 / reject_both / unclear
- gt_value_text (optional, for when you want to type a corrected value)
- gt_notes (optional)

Design constraints
------------------
- No API calls. No retries. Deterministic.
- Additive: does not modify any existing files.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

EVIDENCE_COLS = [
    "evidence_section",
    "evidence_span_text",
    "evidence_span_start",
    "evidence_span_end",
    "evidence_method",
    "evidence_quality",
]

DECISIONS = ["accept_model1", "accept_model2", "reject_both", "unclear"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build field-level GT template from conflict queue TSV.")
    p.add_argument("--conflicts", required=True, help="Path to formulations_conflict_queue.tsv")
    p.add_argument("--out", required=True, help="Output TSV path for field-level GT template")
    p.add_argument("--keep-model-evidence", action="store_true",
                   help="If set, include evidence_*_model1 and evidence_*_model2 columns when present.")
    return p.parse_args(argv)


def split_fields(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def main(argv=None) -> None:
    args = parse_args(argv)
    in_path = Path(args.conflicts)
    out_path = Path(args.out)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, sep="\t", dtype=str).fillna("")

    required = ["key", "formulation_id", "conflict_fields", "model1", "model2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {in_path}")

    # Determine which evidence columns exist
    ev_main_cols = [f"{c}_main" for c in EVIDENCE_COLS if f"{c}_main" in df.columns]

    ev_m1_cols = [f"{c}_model1" for c in EVIDENCE_COLS if f"{c}_model1" in df.columns]
    ev_m2_cols = [f"{c}_model2" for c in EVIDENCE_COLS if f"{c}_model2" in df.columns]

    rows = []
    for _, r in df.iterrows():
        key = r["key"]
        fid = r["formulation_id"]
        m1 = r["model1"]
        m2 = r["model2"]
        preferred = r.get("preferred_model", "")

        fields = split_fields(r.get("conflict_fields", ""))
        if not fields:
            # Defensive: conflict_queue should have conflicts; but skip if empty
            continue

        for f in fields:
            rec = {
                "key": key,
                "formulation_id": fid,
                "field_name": f,
                "model1": m1,
                "model2": m2,
                "preferred_model": preferred,
                "value_model1": r.get(f"{f}_model1", ""),
                "value_model2": r.get(f"{f}_model2", ""),
            }

            # Evidence: main block is the default view for GT
            for c in EVIDENCE_COLS:
                k = f"{c}_main"
                rec[k] = r.get(k, "") if k in df.columns else ""

            # Optional: keep per-model evidence blocks (for deeper audit)
            if args.keep_model_evidence:
                for c in EVIDENCE_COLS:
                    k1 = f"{c}_model1"
                    k2 = f"{c}_model2"
                    if k1 in df.columns:
                        rec[k1] = r.get(k1, "")
                    if k2 in df.columns:
                        rec[k2] = r.get(k2, "")

            # Human fill columns
            rec["gt_decision"] = ""   # must be one of DECISIONS
            rec["gt_value_text"] = "" # optional: corrected value if reject_both
            rec["gt_notes"] = ""      # optional

            rows.append(rec)

    out_df = pd.DataFrame(rows)

    # Stable column order
    lead = [
        "key", "formulation_id", "field_name",
        "model1", "model2", "preferred_model",
        "value_model1", "value_model2",
    ]
    ev_block = [f"{c}_main" for c in EVIDENCE_COLS]
    ev_block += ([f"{c}_model1" for c in EVIDENCE_COLS if f"{c}_model1" in out_df.columns] +
                 [f"{c}_model2" for c in EVIDENCE_COLS if f"{c}_model2" in out_df.columns])
    tail = ["gt_decision", "gt_value_text", "gt_notes"]

    cols = [c for c in lead + ev_block + tail if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in cols]
    out_df = out_df[cols + rest]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

    print(f"[OK] Wrote field-level GT template: {out_path}")
    print(f"[INFO] Rows (field decisions): {len(out_df)}")
    print(f"[INFO] Allowed gt_decision values: {DECISIONS}")


if __name__ == "__main__":
    main(sys.argv[1:])
