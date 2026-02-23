#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
DEFAULT_SEED = 20260222


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic parsing/derivation audit pack from benchmark outputs."
    )
    parser.add_argument("--run-id", required=True, help="Run identifier under data/results/<run_id>/")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic sampling seed.")
    return parser.parse_args()


def short_text(value: Any, limit: int = 200) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def unit_for_field(field_name: str) -> str:
    units = {
        "LA/GA": "ratio",
        "la_fraction": "ratio",
        "ga_fraction": "ratio",
        "drug/polymer": "ratio",
        "drug_mass_mg": "mg",
        "polymer_mw_lower_kDa": "kDa",
        "polymer_mw_upper_kDa": "kDa",
        "polymer_mw_range": "kDa",
    }
    return units.get(field_name, "")


def source_fields_for(field_name: str) -> list[str]:
    if field_name in {"LA/GA", "la_fraction", "ga_fraction"}:
        return ["la_ga_ratio"]
    if field_name in {"polymer_mw_lower_kDa", "polymer_mw_upper_kDa", "polymer_mw_range"}:
        return ["plga_mw_kDa"]
    if field_name == "drug_mass_mg":
        return ["drug_feed_amount_text"]
    if field_name == "drug/polymer":
        return ["drug_feed_amount_text", "plga_mass_mg"]
    return []


def sample_main_category(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    unique_by_group = df.drop_duplicates(subset=["group_key"])
    first_take = unique_by_group.sample(n=min(len(unique_by_group), n), random_state=seed)
    if len(first_take) >= n:
        return first_take
    remainder = df.loc[~df.index.isin(first_take.index)]
    if remainder.empty:
        return first_take
    second_take = remainder.sample(n=min(n - len(first_take), len(remainder)), random_state=seed)
    return pd.concat([first_take, second_take], ignore_index=False)


def build_output_rows(
    df: pd.DataFrame,
    category: str,
    extract_index: pd.DataFrame,
    notes_extra: str = "",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, r in df.iterrows():
        group_key = str(r.get("group_key", ""))
        ex = extract_index.loc[group_key] if group_key in extract_index.index else None
        if isinstance(ex, pd.DataFrame):
            ex = ex.iloc[0]

        field_name = str(r.get("field_name", ""))
        source_fields = source_fields_for(field_name)
        source_value_bits: list[str] = []
        evidence_excerpt = ""
        notes: list[str] = []

        if ex is not None:
            for sf in source_fields:
                if sf in ex.index:
                    sval = str(ex.get(sf, ""))
                    if sval.strip():
                        source_value_bits.append(f"{sf}={short_text(sval, 80)}")
            evidence_excerpt = short_text(ex.get("evidence_span_text", ""), 200)
            if str(ex.get("plga_mw_kDa", "")).strip() and re.search(
                r"\d+\s*[-–]\s*\d+", str(ex.get("plga_mw_kDa", ""))
            ):
                notes.append("range detected")

        if notes_extra:
            notes.append(notes_extra)

        rows.append(
            {
                "sample_category": category,
                "group_key": group_key,
                "key": str(r.get("key", "")),
                "formulation_id": str(r.get("formulation_id", "")),
                "field_name": field_name,
                "source_field_names": ",".join(source_fields),
                "source_value_texts": " | ".join(source_value_bits),
                "derived_value": str(r.get("value", "")),
                "derived_unit": unit_for_field(field_name),
                "rule_id": str(r.get("rule_id", "")),
                "derived_from": str(r.get("derived_from", "")),
                "value_source": str(r.get("value_source", "")),
                "trace_pointer": str(r.get("trace_pointer", "")),
                "evidence_excerpt": evidence_excerpt,
                "notes_for_reviewer": "; ".join([x for x in notes if x]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    run_id = args.run_id.strip() or DEFAULT_RUN_ID
    seed = int(args.seed)

    out_dir = Path(f"data/results/{run_id}/benchmark_goren_2025")
    derived_path = out_dir / "derived_values.tsv"
    extracted_path = Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    projected_path = out_dir / "projected_to_curated.tsv"
    curated_template_path = Path("data/benchmark/goren_2025/NP_dataset_formulations.csv")
    alignment_path = out_dir / "alignment_rows.tsv"

    required_paths = [derived_path, extracted_path, projected_path, curated_template_path]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    derived = pd.read_csv(derived_path, sep="\t", dtype=str).fillna("")
    required_derived_cols = {"rule_id", "derived_from", "value_source", "trace_pointer"}
    if not required_derived_cols.issubset(set(derived.columns)):
        missing_cols = sorted(required_derived_cols - set(derived.columns))
        raise RuntimeError(
            f"STOP: derived_values.tsv missing required columns: {missing_cols}. "
            "Please confirm derivation output schema."
        )

    extracted = pd.read_csv(extracted_path, sep="\t", dtype=str).fillna("")
    if "evidence_span_text" not in extracted.columns:
        raise RuntimeError("Cannot locate evidence_span_text in extraction baseline.")
    extracted["group_key"] = (
        extracted["key"].astype(str).str.strip() + "::" + extracted["formulation_id"].astype(str).str.strip()
    )
    extract_index = extracted.set_index("group_key", drop=False)

    rows: list[dict[str, str]] = []

    # Main categories.
    la_df = derived[derived["field_name"].isin(["LA/GA", "la_fraction", "ga_fraction"])].copy()
    mw_df = derived[derived["field_name"].isin(["polymer_mw_lower_kDa", "polymer_mw_upper_kDa"])].copy()
    drug_mass_df = derived[derived["field_name"].eq("drug_mass_mg")].copy()
    dpr_df = derived[derived["field_name"].eq("drug/polymer")].copy()

    rows += build_output_rows(sample_main_category(la_df, 20, seed), "la_ga", extract_index)
    rows += build_output_rows(sample_main_category(mw_df, 20, seed), "mw", extract_index)
    rows += build_output_rows(sample_main_category(drug_mass_df, 20, seed), "drug_mass", extract_index)
    rows += build_output_rows(sample_main_category(dpr_df, 20, seed), "dpr_ratio", extract_index)

    # Failures: source non-empty but no derived output.
    have_laga = set(derived.loc[derived["field_name"].eq("LA/GA"), "group_key"])
    have_mw = set(derived.loc[derived["field_name"].eq("polymer_mw_lower_kDa"), "group_key"])
    have_drug_mass = set(derived.loc[derived["field_name"].eq("drug_mass_mg"), "group_key"])
    have_dpr = set(derived.loc[derived["field_name"].eq("drug/polymer"), "group_key"])
    failure_records: list[dict[str, str]] = []

    for _, ex in extracted.iterrows():
        gk = ex["group_key"]
        if str(ex.get("la_ga_ratio", "")).strip() and gk not in have_laga:
            failure_records.append(
                {
                    "group_key": gk,
                    "key": str(ex.get("key", "")),
                    "formulation_id": str(ex.get("formulation_id", "")),
                    "field_name": "LA/GA",
                    "source_field_names": "la_ga_ratio",
                    "source_value_texts": f"la_ga_ratio={short_text(ex.get('la_ga_ratio', ''), 80)}",
                    "derived_unit": "ratio",
                }
            )
        if str(ex.get("plga_mw_kDa", "")).strip() and gk not in have_mw:
            failure_records.append(
                {
                    "group_key": gk,
                    "key": str(ex.get("key", "")),
                    "formulation_id": str(ex.get("formulation_id", "")),
                    "field_name": "polymer_mw_lower_kDa",
                    "source_field_names": "plga_mw_kDa",
                    "source_value_texts": f"plga_mw_kDa={short_text(ex.get('plga_mw_kDa', ''), 80)}",
                    "derived_unit": "kDa",
                }
            )
        if str(ex.get("drug_feed_amount_text", "")).strip() and gk not in have_drug_mass:
            failure_records.append(
                {
                    "group_key": gk,
                    "key": str(ex.get("key", "")),
                    "formulation_id": str(ex.get("formulation_id", "")),
                    "field_name": "drug_mass_mg",
                    "source_field_names": "drug_feed_amount_text",
                    "source_value_texts": f"drug_feed_amount_text={short_text(ex.get('drug_feed_amount_text', ''), 80)}",
                    "derived_unit": "mg",
                }
            )
        if (str(ex.get("drug_feed_amount_text", "")).strip() or str(ex.get("plga_mass_mg", "")).strip()) and gk not in have_dpr:
            failure_records.append(
                {
                    "group_key": gk,
                    "key": str(ex.get("key", "")),
                    "formulation_id": str(ex.get("formulation_id", "")),
                    "field_name": "drug/polymer",
                    "source_field_names": "drug_feed_amount_text,plga_mass_mg",
                    "source_value_texts": (
                        f"drug_feed_amount_text={short_text(ex.get('drug_feed_amount_text', ''), 80)} | "
                        f"plga_mass_mg={short_text(ex.get('plga_mass_mg', ''), 80)}"
                    ),
                    "derived_unit": "ratio",
                }
            )

    if failure_records:
        failure_df = pd.DataFrame(failure_records).drop_duplicates(subset=["group_key", "field_name"])
        failure_df = failure_df.sample(n=min(10, len(failure_df)), random_state=seed)
        for _, fr in failure_df.iterrows():
            ex = extract_index.loc[fr["group_key"]]
            if isinstance(ex, pd.DataFrame):
                ex = ex.iloc[0]
            rows.append(
                {
                    "sample_category": "failures",
                    "group_key": str(fr["group_key"]),
                    "key": str(fr["key"]),
                    "formulation_id": str(fr["formulation_id"]),
                    "field_name": str(fr["field_name"]),
                    "source_field_names": str(fr["source_field_names"]),
                    "source_value_texts": str(fr["source_value_texts"]),
                    "derived_value": "",
                    "derived_unit": str(fr["derived_unit"]),
                    "rule_id": "",
                    "derived_from": "",
                    "value_source": "",
                    "trace_pointer": "",
                    "evidence_excerpt": short_text(ex.get("evidence_span_text", ""), 200),
                    "notes_for_reviewer": "parse failed or insufficient inputs with non-empty source",
                }
            )

    # Ranges: rows with MW lower/upper present.
    mw_lo = (
        derived[derived["field_name"].eq("polymer_mw_lower_kDa")][
            ["group_key", "key", "formulation_id", "value", "rule_id", "derived_from", "value_source", "trace_pointer"]
        ]
        .rename(columns={"value": "lo"})
        .copy()
    )
    mw_hi = (
        derived[derived["field_name"].eq("polymer_mw_upper_kDa")][["group_key", "value"]]
        .rename(columns={"value": "hi"})
        .copy()
    )
    ranges_df = mw_lo.merge(mw_hi, on="group_key", how="inner")
    ranges_df = ranges_df[(ranges_df["lo"] != "") & (ranges_df["hi"] != "")]
    if not ranges_df.empty:
        ranges_df = ranges_df.sample(n=min(10, len(ranges_df)), random_state=seed)
        for _, rr in ranges_df.iterrows():
            ex = extract_index.loc[rr["group_key"]]
            if isinstance(ex, pd.DataFrame):
                ex = ex.iloc[0]
            rows.append(
                {
                    "sample_category": "ranges",
                    "group_key": str(rr["group_key"]),
                    "key": str(rr["key"]),
                    "formulation_id": str(rr["formulation_id"]),
                    "field_name": "polymer_mw_range",
                    "source_field_names": "plga_mw_kDa",
                    "source_value_texts": f"plga_mw_kDa={short_text(ex.get('plga_mw_kDa', ''), 80)}",
                    "derived_value": f"{rr['lo']}-{rr['hi']}",
                    "derived_unit": "kDa",
                    "rule_id": str(rr["rule_id"]),
                    "derived_from": str(rr["derived_from"]),
                    "value_source": str(rr["value_source"]),
                    "trace_pointer": str(rr["trace_pointer"]),
                    "evidence_excerpt": short_text(ex.get("evidence_span_text", ""), 200),
                    "notes_for_reviewer": "range/bounds audit",
                }
            )

    # Extremes: top/bottom 1% per numeric field, capped at 20 total.
    extreme_fields = ["LA/GA", "polymer_mw_lower_kDa", "drug_mass_mg", "drug/polymer"]
    extreme_rows: list[dict[str, str]] = []
    per_field_cap = 5
    for field in extreme_fields:
        sub = derived[derived["field_name"].eq(field)].copy()
        if len(sub) < 30:
            continue
        sub["num"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub[sub["num"].notna()].copy()
        if len(sub) < 30:
            continue
        q_low = sub["num"].quantile(0.01)
        q_high = sub["num"].quantile(0.99)
        ex_sub = sub[(sub["num"] <= q_low) | (sub["num"] >= q_high)]
        if ex_sub.empty:
            continue
        ex_sub = ex_sub.sample(n=min(per_field_cap, len(ex_sub)), random_state=seed)
        extreme_rows.extend(
            build_output_rows(
                ex_sub,
                "extremes",
                extract_index,
                notes_extra="extreme value (top/bottom 1%)",
            )
        )
    rows.extend(extreme_rows[:20])

    out_df = pd.DataFrame(rows)
    expected_cols = [
        "sample_category",
        "group_key",
        "key",
        "formulation_id",
        "field_name",
        "source_field_names",
        "source_value_texts",
        "derived_value",
        "derived_unit",
        "rule_id",
        "derived_from",
        "value_source",
        "trace_pointer",
        "evidence_excerpt",
        "notes_for_reviewer",
    ]
    if out_df.empty:
        out_df = pd.DataFrame(columns=expected_cols)
    else:
        out_df = out_df.drop_duplicates(subset=["sample_category", "group_key", "field_name", "derived_value"])
        out_df = out_df[expected_cols].sort_values(["sample_category", "group_key", "field_name"]).reset_index(drop=True)

    out_tsv = out_dir / "audit_parsing_derivation_samples.tsv"
    out_md = out_dir / "audit_parsing_derivation_checklist.md"
    out_json = out_dir / "audit_parsing_derivation_sanity.json"

    out_df.to_csv(out_tsv, sep="\t", index=False)

    checklist = """# Parsing + Derivation Audit Checklist

## Review focus by sample_category
- `la_ga`: verify `la_ga_ratio` parsing and LA/GA/fraction consistency.
- `mw`: verify `plga_mw_kDa` parsing into MW bounds.
- `drug_mass`: verify mass extraction and mg normalization from `drug_feed_amount_text`.
- `dpr_ratio`: verify `drug/polymer = drug_mass_mg / polymer_mass_mg`.
- `failures`: source is non-empty but derived output is missing; confirm expected vs parser limits.
- `ranges`: verify lower/upper bound handling with no midpoint inference.
- `extremes`: check outliers for unit, magnitude, and decimal-shift errors.

## Pass / fail criteria
- Pass: derived value matches source text and rule semantics, with coherent provenance.
- Fail: derived value contradicts source text, conversion is wrong, or provenance is inconsistent/missing.

## Provenance columns to trust
- Primary: `rule_id`, `derived_from`, `value_source`, `trace_pointer`
- Context: `source_field_names`, `source_value_texts`, `evidence_excerpt`
"""
    out_md.write_text(checklist, encoding="utf-8")

    projected_header = pd.read_csv(projected_path, sep="\t", nrows=0).columns.tolist()
    curated_header = pd.read_csv(curated_template_path, nrows=0).columns.tolist()

    sanity = {
        "run_id": run_id,
        "seed_used": seed,
        "derived_values_row_count": int(len(derived)),
        "unique_group_keys_in_derived": int(derived["group_key"].nunique()),
        "non_null_counts_per_audited_field": {
            "LA/GA": int((derived["field_name"] == "LA/GA").sum()),
            "polymer_mw_lower_kDa": int((derived["field_name"] == "polymer_mw_lower_kDa").sum()),
            "polymer_mw_upper_kDa": int((derived["field_name"] == "polymer_mw_upper_kDa").sum()),
            "drug_mass_mg": int((derived["field_name"] == "drug_mass_mg").sum()),
            "drug/polymer": int((derived["field_name"] == "drug/polymer").sum()),
        },
        "projection_header_equals_curated_template_header": bool(projected_header == curated_header),
        "alignment_rows_present": bool(alignment_path.exists()),
        "audit_rows_total": int(len(out_df)),
        "audit_unique_group_keys": int(out_df["group_key"].nunique()) if len(out_df) > 0 else 0,
        "audit_rows_by_category": {k: int(v) for k, v in out_df["sample_category"].value_counts().to_dict().items()},
        "outputs": {
            "audit_samples_tsv": str(out_tsv),
            "checklist_md": str(out_md),
            "sanity_json": str(out_json),
        },
    }
    out_json.write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"seed={seed}")
    print(f"audit_rows={len(out_df)}")
    print(f"audit_unique_group_keys={sanity['audit_unique_group_keys']}")
    print(f"projection_header_match={sanity['projection_header_equals_curated_template_header']}")
    print(f"output_tsv={out_tsv}")
    print(f"output_md={out_md}")
    print(f"output_json={out_json}")


if __name__ == "__main__":
    main()
