#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"


FIELD_SPECS: dict[str, dict[str, Any]] = {
    "drug_feed_amount_text": {
        "unit_tokens": ["mg", "g", "ug", "mcg", "ng", "mg/ml", "g/ml", "%w/w"],
        "entity_type": "drug",
    },
    "plga_mass_mg": {
        "unit_tokens": ["mg", "g", "ug", "mcg", "ng", "mg/ml", "g/ml"],
        "entity_type": "polymer",
    },
    "pva_conc_percent": {
        "unit_tokens": ["%", "percent", "wt%", "w/v", "% w/v", "%w/v"],
        "entity_type": "surfactant",
    },
    "encapsulation_efficiency_percent": {
        "unit_tokens": ["%", "percent"],
        "entity_type": "",
    },
    "loading_content_percent": {
        "unit_tokens": ["%", "percent"],
        "entity_type": "",
    },
    "size_nm": {
        "unit_tokens": ["nm", "nanometer", "nanometers"],
        "entity_type": "",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run evidence-token gating QC on extracted numeric fields and flag unsupported values."
    )
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--input-tsv", default="", help="Default: data/results/<run_id>/weak_labels__gemini.tsv")
    p.add_argument(
        "--out-dir",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/derivation_v1",
    )
    p.add_argument("--unit-min-matches", type=int, default=1)
    p.add_argument("--entity-min-matches", type=int, default=1)
    p.add_argument("--require-unit-token", action="store_true", default=True)
    p.add_argument("--no-require-unit-token", dest="require_unit_token", action="store_false")
    p.add_argument("--require-entity-for-mass", action="store_true", default=True)
    p.add_argument("--no-require-entity-for-mass", dest="require_entity_for_mass", action="store_false")
    return p.parse_args()


def normalize_text(v: Any) -> str:
    s = str(v or "").lower()
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    return s


def tokenize_name(v: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z]{3,}", str(v or "").lower())
    stop = {"the", "and", "for", "with", "from", "into", "drug"}
    out = [t for t in toks if t not in stop]
    return sorted(set(out))


def find_numeric_tokens(v: str) -> list[str]:
    raw = re.findall(r"[+-]?\d+(?:\.\d+)?", str(v or ""))
    cleaned = []
    for x in raw:
        try:
            num = float(x)
        except ValueError:
            continue
        if abs(num) < 1e-12:
            cleaned.append("0")
        else:
            cleaned.append(f"{num:.12f}".rstrip("0").rstrip("."))
    return cleaned


def build_required_unit_tokens(field_name: str, value_text: str) -> list[str]:
    spec_units = FIELD_SPECS.get(field_name, {}).get("unit_tokens", [])
    v = normalize_text(value_text)
    detected = []
    for u in spec_units:
        if u in v:
            detected.append(u)
    if detected:
        return sorted(set(detected))
    return sorted(set(spec_units))


def build_required_entity_tokens(field_name: str, row: pd.Series) -> list[str]:
    et = FIELD_SPECS.get(field_name, {}).get("entity_type", "")
    if et == "polymer":
        return ["plga", "poly(lactic-co-glycolic acid)", "poly(lactide-co-glycolide)"]
    if et == "surfactant":
        return ["pva", "polyvinyl alcohol", "poloxamer", "surfactant"]
    if et == "drug":
        return tokenize_name(str(row.get("drug_name", "")).strip())
    return []


def contains_any(text: str, tokens: list[str]) -> int:
    if not tokens:
        return 0
    c = 0
    for t in tokens:
        tt = normalize_text(t).strip()
        if not tt:
            continue
        if tt in text:
            c += 1
    return c


def short_text(v: str, n: int = 220) -> str:
    s = str(v or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def run_qc(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for idx, r in df.iterrows():
        evidence = normalize_text(r.get("evidence_span_text", ""))
        if not evidence.strip():
            continue
        key = str(r.get("key", "")).strip()
        fid = str(r.get("formulation_id", "")).strip()
        group_key = f"{key}::{fid}"
        for field_name, spec in FIELD_SPECS.items():
            if field_name not in df.columns:
                continue
            value_text = str(r.get(field_name, "")).strip()
            if not value_text:
                continue
            numeric_tokens = find_numeric_tokens(value_text)
            if not numeric_tokens:
                continue
            main_numeric = numeric_tokens[0]

            unit_tokens = build_required_unit_tokens(field_name, value_text)
            entity_tokens = build_required_entity_tokens(field_name, r)
            has_numeric = main_numeric in evidence
            unit_matches = contains_any(evidence, unit_tokens)
            entity_matches = contains_any(evidence, entity_tokens)

            fail_numeric = not has_numeric
            fail_unit = bool(args.require_unit_token and unit_matches < int(args.unit_min_matches))
            require_entity = (
                args.require_entity_for_mass and field_name in {"drug_feed_amount_text", "plga_mass_mg"} and len(entity_tokens) > 0
            )
            fail_entity = bool(require_entity and entity_matches < int(args.entity_min_matches))
            mismatch = bool(fail_numeric or fail_unit or fail_entity)

            rows.append(
                {
                    "row_index": int(idx),
                    "key": key,
                    "formulation_id": fid,
                    "group_key": group_key,
                    "field_name": field_name,
                    "field_value": value_text,
                    "main_numeric_token": main_numeric,
                    "required_unit_tokens": "|".join(unit_tokens),
                    "required_entity_tokens": "|".join(entity_tokens),
                    "has_main_numeric_token": int(has_numeric),
                    "unit_match_count": int(unit_matches),
                    "entity_match_count": int(entity_matches),
                    "fail_numeric_token": int(fail_numeric),
                    "fail_unit_token": int(fail_unit),
                    "fail_entity_token": int(fail_entity),
                    "evidence_mismatch": int(mismatch),
                    "evidence_span_text": short_text(r.get("evidence_span_text", ""), 300),
                    "drug_name": str(r.get("drug_name", "")).strip(),
                }
            )

    checks = pd.DataFrame(rows)
    if checks.empty:
        checks = pd.DataFrame(
            columns=[
                "row_index",
                "key",
                "formulation_id",
                "group_key",
                "field_name",
                "field_value",
                "main_numeric_token",
                "required_unit_tokens",
                "required_entity_tokens",
                "has_main_numeric_token",
                "unit_match_count",
                "entity_match_count",
                "fail_numeric_token",
                "fail_unit_token",
                "fail_entity_token",
                "evidence_mismatch",
                "evidence_span_text",
                "drug_name",
            ]
        )
    flagged = checks[checks["evidence_mismatch"] == 1].copy()

    report_rows: list[dict[str, Any]] = []
    for fn, g in checks.groupby("field_name", sort=True):
        n = int(len(g))
        m = int((g["evidence_mismatch"] == 1).sum())
        report_rows.append(
            {
                "field_name": fn,
                "evaluated_rows": n,
                "mismatch_rows": m,
                "mismatch_rate": (m / n) if n else 0.0,
                "fail_numeric_token_rows": int((g["fail_numeric_token"] == 1).sum()),
                "fail_unit_token_rows": int((g["fail_unit_token"] == 1).sum()),
                "fail_entity_token_rows": int((g["fail_entity_token"] == 1).sum()),
            }
        )
    report = pd.DataFrame(report_rows).sort_values("mismatch_rate", ascending=False).reset_index(drop=True)

    bad_row_indices = set(flagged["row_index"].astype(int).tolist())
    high_conf = df.copy()
    high_conf["row_index"] = range(len(high_conf))
    high_conf["evidence_mismatch_any"] = high_conf["row_index"].map(lambda i: 1 if int(i) in bad_row_indices else 0)
    high_conf = high_conf[high_conf["evidence_mismatch_any"] == 0].reset_index(drop=True)
    return checks, report, flagged, high_conf


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    input_tsv = Path(args.input_tsv) if args.input_tsv else Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/{run_id}/benchmark_goren_2025/derivation_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_tsv.exists():
        raise FileNotFoundError(f"Missing input extraction TSV: {input_tsv}")

    df = pd.read_csv(input_tsv, sep="\t", dtype=str).fillna("")
    if "evidence_span_text" not in df.columns:
        raise RuntimeError("Input TSV missing required column: evidence_span_text")

    checks, report, flagged, high_conf = run_qc(df, args)

    p_report = out_dir / "qc_report.tsv"
    p_flagged = out_dir / "flagged_rows_for_review.tsv"
    p_high = out_dir / "weak_labels__gemini_high_confidence.tsv"
    p_checks = out_dir / "evidence_token_qc_checks.tsv"

    report.to_csv(p_report, sep="\t", index=False)
    flagged.to_csv(p_flagged, sep="\t", index=False)
    high_conf.to_csv(p_high, sep="\t", index=False)

    checks.to_csv(p_checks, sep="\t", index=False)

    summary = {
        "input_rows": int(len(df)),
        "qc_evaluated_field_rows": int(sum(report["evaluated_rows"])) if not report.empty else 0,
        "qc_mismatch_field_rows": int(sum(report["mismatch_rows"])) if not report.empty else 0,
        "flagged_rows_for_review": int(len(flagged)),
        "high_confidence_rows": int(len(high_conf)),
        "outputs": {
            "qc_report_tsv": str(p_report),
            "flagged_rows_for_review_tsv": str(p_flagged),
            "high_confidence_tsv": str(p_high),
            "checks_tsv": str(p_checks),
        },
        "params": {
            "unit_min_matches": int(args.unit_min_matches),
            "entity_min_matches": int(args.entity_min_matches),
            "require_unit_token": bool(args.require_unit_token),
            "require_entity_for_mass": bool(args.require_entity_for_mass),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
