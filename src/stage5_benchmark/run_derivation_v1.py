#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from derive_doe_coded_factors_v1 import derive_doe_coded_factors


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
DEFAULT_RULE_REGISTRY = "data/benchmark/goren_2025/rules/derivation_rule_registry.v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive normalized formulation fields from extraction output without changing extraction logic."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--input-tsv",
        default="",
        help="Defaults to data/results/<run_id>/weak_labels__gemini.tsv",
    )
    parser.add_argument(
        "--rule-registry",
        default=DEFAULT_RULE_REGISTRY,
        help="Immutable derivation rule registry JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Defaults to data/results/<run_id>/benchmark_goren_2025",
    )
    return parser.parse_args()


def parse_float(raw: Any) -> float | None:
    if raw is None or pd.isna(raw):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_mass_to_mg(raw: Any) -> float | None:
    if raw is None or pd.isna(raw):
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    text = text.replace("μ", "u")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(mg|g|ug|mcg|ng)\b", text)
    if not match:
        # Fallback for numeric text with unknown unit; treated as mg to avoid silent drop.
        return parse_float(text)
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "g":
        return value * 1000.0
    if unit in {"ug", "mcg"}:
        return value / 1000.0
    if unit == "ng":
        return value / 1_000_000.0
    return value


def parse_la_ga_ratio(raw: Any) -> tuple[float | None, float | None, float | None]:
    if raw is None or pd.isna(raw):
        return (None, None, None)
    text = str(raw).strip()
    if not text:
        return (None, None, None)
    text = text.replace(" ", "")
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            try:
                la = float(parts[0])
                ga = float(parts[1])
            except ValueError:
                return (None, None, None)
            total = la + ga
            if total == 0:
                return (None, None, None)
            la_fraction = la / total
            ga_fraction = ga / total
            la_over_ga = (la / ga) if ga != 0 else None
            return (la_fraction, ga_fraction, la_over_ga)
    # If only one numeric value exists, preserve it as LA/GA ratio.
    scalar = parse_float(text)
    return (None, None, scalar)


def parse_mw_range(raw: Any) -> tuple[float | None, float | None]:
    if raw is None or pd.isna(raw):
        return (None, None)
    text = str(raw).strip().replace(",", "")
    if not text:
        return (None, None)
    vals = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not vals:
        return (None, None)
    if len(vals) == 1:
        v = float(vals[0])
        return (v, v)
    lo = float(vals[0])
    hi = float(vals[1])
    if lo <= hi:
        return (lo, hi)
    return (hi, lo)


def parse_aqueous_organic_from_span(raw: Any) -> tuple[float | None, float | None]:
    if raw is None or pd.isna(raw):
        return (None, None)
    text = str(raw)
    if not text.strip():
        return (None, None)
    compact = re.sub(r"\s+", "", text.lower())

    # W1/O patterns.
    m_w1o = re.search(r"w1/o[:=]?(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", compact)
    w1_o = None
    if m_w1o:
        a = float(m_w1o.group(1))
        b = float(m_w1o.group(2))
        if b != 0:
            w1_o = a / b

    # (W1+W2)/O patterns.
    m_wtot_o = re.search(r"\(w1\+w2\)/o[:=]?(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", compact)
    wtot_o = None
    if m_wtot_o:
        a = float(m_wtot_o.group(1))
        b = float(m_wtot_o.group(2))
        if b != 0:
            wtot_o = a / b

    return (w1_o, wtot_o)


def make_trace_pointer(row: pd.Series, row_index: int) -> str:
    payload = {
        "row_index": int(row_index),
        "key": str(row.get("key", "") or ""),
        "formulation_id": str(row.get("formulation_id", "") or ""),
        "evidence_section": str(row.get("evidence_section", "") or ""),
        "evidence_span_start": str(row.get("evidence_span_start", "") or ""),
        "evidence_span_end": str(row.get("evidence_span_end", "") or ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def add_value(
    out_rows: list[dict[str, Any]],
    *,
    run_id: str,
    group_key: str,
    key: str,
    formulation_id: str,
    field_name: str,
    value: Any,
    rule_id: str,
    derived_from: str,
    value_source: str,
    trace_pointer: str,
) -> None:
    if value is None:
        return
    txt = str(value).strip()
    if txt == "":
        return
    out_rows.append(
        {
            "run_id": run_id,
            "group_key": group_key,
            "key": key,
            "formulation_id": formulation_id,
            "field_name": field_name,
            "value": txt,
            "rule_id": rule_id,
            "derived_from": derived_from,
            "value_source": value_source,
            "trace_pointer": trace_pointer,
        }
    )


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    input_tsv = Path(args.input_tsv) if args.input_tsv else Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/{run_id}/benchmark_goren_2025")
    rule_registry = Path(args.rule_registry)

    if not input_tsv.exists():
        raise FileNotFoundError(f"Input extraction TSV not found: {input_tsv}")
    if not rule_registry.exists():
        raise FileNotFoundError(f"Derivation rule registry not found: {rule_registry}")

    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = pd.read_csv(input_tsv, sep="\t", dtype=str).fillna("")
    required_cols = ["key", "formulation_id"]
    missing = [c for c in required_cols if c not in extracted.columns]
    if missing:
        raise RuntimeError(f"Extraction TSV missing required columns: {missing}")

    derived_rows: list[dict[str, Any]] = []
    anchors_for_ao = {"plga_mass_mg", "drug_feed_amount_text", "la_ga_ratio", "plga_mw_kDa"}

    for row_index, row in extracted.iterrows():
        key = str(row.get("key", "")).strip()
        formulation_id = str(row.get("formulation_id", "")).strip()
        group_key = f"{key}::{formulation_id}"
        trace_ptr = make_trace_pointer(row, row_index)

        # Direct parsed anchors.
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="small_molecule_name",
            value=row.get("drug_name", ""),
            rule_id="R_DIRECT_DRUG_NAME",
            derived_from="drug_name",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="solvent",
            value=row.get("organic_solvent", ""),
            rule_id="R_DIRECT_ORGANIC_SOLVENT",
            derived_from="organic_solvent",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="surfactant_concentration",
            value=row.get("pva_conc_percent", ""),
            rule_id="R_DIRECT_SURFACTANT_CONC",
            derived_from="pva_conc_percent",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="particle_size",
            value=row.get("size_nm", ""),
            rule_id="R_DIRECT_PARTICLE_SIZE",
            derived_from="size_nm",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="EE",
            value=row.get("encapsulation_efficiency_percent", ""),
            rule_id="R_DIRECT_EE",
            derived_from="encapsulation_efficiency_percent",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="LC",
            value=row.get("loading_content_percent", ""),
            rule_id="R_DIRECT_LC",
            derived_from="loading_content_percent",
            value_source="extracted_anchor",
            trace_pointer=trace_ptr,
        )

        la_fraction, ga_fraction, la_over_ga = parse_la_ga_ratio(row.get("la_ga_ratio", ""))
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="la_fraction",
            value=la_fraction,
            rule_id="R_LAGA_PARSE_FRACTIONS",
            derived_from="la_ga_ratio",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="ga_fraction",
            value=ga_fraction,
            rule_id="R_LAGA_PARSE_FRACTIONS",
            derived_from="la_ga_ratio",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="LA/GA",
            value=la_over_ga,
            rule_id="R_LAGA_PARSE_OVER_GA",
            derived_from="la_ga_ratio",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )

        mw_low, mw_high = parse_mw_range(row.get("plga_mw_kDa", ""))
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="polymer_mw_lower_kDa",
            value=mw_low,
            rule_id="R_POLYMER_MW_RANGE_PARSE",
            derived_from="plga_mw_kDa",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="polymer_mw_upper_kDa",
            value=mw_high,
            rule_id="R_POLYMER_MW_RANGE_PARSE",
            derived_from="plga_mw_kDa",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )

        polymer_mass = parse_mass_to_mg(row.get("plga_mass_mg", ""))
        drug_mass = parse_mass_to_mg(row.get("drug_feed_amount_text", ""))
        drug_polymer_ratio = None
        if polymer_mass is not None and polymer_mass != 0 and drug_mass is not None:
            drug_polymer_ratio = drug_mass / polymer_mass

        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="polymer_mass_mg",
            value=polymer_mass,
            rule_id="R_POLYMER_MASS_PARSE",
            derived_from="plga_mass_mg",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="drug_mass_mg",
            value=drug_mass,
            rule_id="R_DRUG_MASS_PARSE",
            derived_from="drug_feed_amount_text",
            value_source="parsed_from_extracted",
            trace_pointer=trace_ptr,
        )
        add_value(
            derived_rows,
            run_id=run_id,
            group_key=group_key,
            key=key,
            formulation_id=formulation_id,
            field_name="drug/polymer",
            value=drug_polymer_ratio,
            rule_id="R_DRUG_POLYMER_RATIO_COMPLETE",
            derived_from="drug_mass_mg,polymer_mass_mg",
            value_source="derived_math",
            trace_pointer=trace_ptr,
        )

        # v1 policy: derive aqueous/organic only from explicit evidence span with extracted anchors.
        span = row.get("evidence_span_text", "")
        has_anchor = any(str(row.get(col, "")).strip() != "" for col in anchors_for_ao)
        if has_anchor and str(span).strip():
            w1_o, wtot_o = parse_aqueous_organic_from_span(span)
            add_value(
                derived_rows,
                run_id=run_id,
                group_key=group_key,
                key=key,
                formulation_id=formulation_id,
                field_name="w1_over_o_ratio",
                value=w1_o,
                rule_id="R_AQUEOUS_ORGANIC_PARSE_W1_OVER_O",
                derived_from="evidence_span_text",
                value_source="parsed_evidence_span",
                trace_pointer=trace_ptr,
            )
            add_value(
                derived_rows,
                run_id=run_id,
                group_key=group_key,
                key=key,
                formulation_id=formulation_id,
                field_name="w1w2_over_o_ratio",
                value=wtot_o,
                rule_id="R_AQUEOUS_ORGANIC_PARSE_W1W2_OVER_O",
                derived_from="evidence_span_text",
                value_source="parsed_evidence_span",
                trace_pointer=trace_ptr,
            )

    derived_df = pd.DataFrame(derived_rows)
    if derived_df.empty:
        derived_df = pd.DataFrame(
            columns=[
                "run_id",
                "group_key",
                "key",
                "formulation_id",
                "field_name",
                "value",
                "rule_id",
                "derived_from",
                "value_source",
                "trace_pointer",
            ]
        )

    doe_out_dir = out_dir / "derivation_v1"
    doe_result = derive_doe_coded_factors(
        run_id=run_id,
        extracted=extracted,
        derived=derived_df,
        sample_manifest_path=Path("data/cleaned/samples/sample_goren18.tsv"),
        key2txt_path=Path("data/cleaned/content_goren_2025/key2txt.tsv"),
        out_dir=doe_out_dir,
    )
    derived_df = doe_result["derived_df"]

    out_tsv = out_dir / "derived_values.tsv"
    derived_df.to_csv(out_tsv, sep="\t", index=False)

    summary = {
        "run_id": run_id,
        "input_tsv": str(input_tsv),
        "rule_registry": str(rule_registry),
        "derived_values_rows": int(len(derived_df)),
        "distinct_group_keys": int(derived_df["group_key"].nunique()) if len(derived_df) > 0 else 0,
        "output": str(out_tsv),
        "doe_decode": doe_result["summary"],
    }
    (out_dir / "derivation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
