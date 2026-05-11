#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
SCHEMA_VERSION = "schema_v1"

POSTPROCESS_PATTERNS = [
    r"lyophil\w*",
    r"freeze[-\s]?dry\w*",
    r"cryoprotect\w*",
    r"trehalose",
    r"sucrose",
    r"mannitol",
    r"storage",
    r"reconstitution",
    r"stability",
    r"after\s+\d+\s*(day|days|week|weeks)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build two-table schema (formulation_core + measurements) for Goren benchmark outputs."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/schema_v1",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_doi(value: Any) -> str:
    s = normalize_text(value)
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[)\],.;]+$", "", s)
    return s


def norm_num_token(value: str) -> str:
    s = str(value).strip()
    if not s:
        return "missing"
    try:
        f = float(s)
    except ValueError:
        return s.lower()
    if f.is_integer():
        return str(int(f))
    return f"{f:.6g}"


def pick_best_field(group_df: pd.DataFrame, field_name: str) -> dict[str, str]:
    sub = group_df[group_df["field_name"] == field_name].copy()
    if sub.empty:
        return {"value": "", "rule_id": "", "value_source": "", "trace_pointer": ""}
    priority = {
        "extracted_anchor": 0,
        "parsed_from_extracted": 1,
        "parsed_evidence_span": 2,
        "derived_math": 3,
        "projection_compose": 4,
    }
    sub["prio"] = sub["value_source"].map(lambda x: priority.get(str(x), 9))
    sub = sub.sort_values(["prio"]).reset_index(drop=True)
    row = sub.iloc[0]
    return {
        "value": str(row.get("value", "")),
        "rule_id": str(row.get("rule_id", "")),
        "value_source": str(row.get("value_source", "")),
        "trace_pointer": str(row.get("trace_pointer", "")),
    }


def detect_condition(text: str) -> tuple[str, str]:
    raw = normalize_text(text)
    matched = []
    for p in POSTPROCESS_PATTERNS:
        if re.search(p, raw, flags=re.IGNORECASE):
            matched.append(p)
    if matched:
        return "postprocess", "|".join(sorted(set(matched)))
    return "fresh", ""


def stable_signature(parts: list[str]) -> str:
    joined = "||".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"{joined}||sig={digest}"


def build_dataframes(
    derived: pd.DataFrame,
    extracted: pd.DataFrame,
    projection_trace: pd.DataFrame,
    key_to_doi_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], pd.DataFrame]:
    # DOI mapping from projection trace (reference rows).
    ref_map = (
        projection_trace[projection_trace["curated_column"] == "reference"][
            ["group_key", "projected_value"]
        ]
        .drop_duplicates(subset=["group_key"], keep="first")
        .copy()
    )
    ref_map["reference_normalized_doi"] = ref_map["projected_value"].map(normalize_doi)
    group_to_doi = dict(zip(ref_map["group_key"], ref_map["reference_normalized_doi"]))

    # Core field selection per group_key.
    core_rows = []
    for gk, gdf in derived.groupby("group_key", dropna=False):
        key = str(gdf.iloc[0].get("key", ""))
        formulation_id = str(gdf.iloc[0].get("formulation_id", ""))
        doi = group_to_doi.get(str(gk), "")
        if not doi:
            doi = key_to_doi_map.get(key, "")

        drug = pick_best_field(gdf, "small_molecule_name")
        la_f = pick_best_field(gdf, "la_fraction")
        ga_f = pick_best_field(gdf, "ga_fraction")
        la_ga = pick_best_field(gdf, "LA/GA")
        mw_lo = pick_best_field(gdf, "polymer_mw_lower_kDa")
        mw_hi = pick_best_field(gdf, "polymer_mw_upper_kDa")
        solvent = pick_best_field(gdf, "solvent")
        ratio = pick_best_field(gdf, "drug/polymer")

        la_tok = norm_num_token(la_f["value"]) if la_f["value"] else "missing"
        ga_tok = norm_num_token(ga_f["value"]) if ga_f["value"] else "missing"
        laga_tok = norm_num_token(la_ga["value"]) if la_ga["value"] else "missing"
        mw_lo_tok = norm_num_token(mw_lo["value"]) if mw_lo["value"] else "missing"
        mw_hi_tok = norm_num_token(mw_hi["value"]) if mw_hi["value"] else "missing"
        ratio_tok = norm_num_token(ratio["value"]) if ratio["value"] else "missing"
        solvent_tok = normalize_text(solvent["value"]) if solvent["value"] else "missing"
        drug_tok = normalize_text(drug["value"]) if drug["value"] else "missing"
        doi_tok = doi if doi else "missing"

        signature = stable_signature(
            [
                f"doi={doi_tok}",
                f"drug={drug_tok}",
                f"la_fraction={la_tok}",
                f"ga_fraction={ga_tok}",
                f"la_ga_ratio={laga_tok}",
                f"mw_lo={mw_lo_tok}",
                f"mw_hi={mw_hi_tok}",
                f"solvent={solvent_tok}",
                f"dpr={ratio_tok}",
            ]
        )

        core_rows.append(
            {
                "group_key": str(gk),
                "key": key,
                "formulation_id": formulation_id,
                "reference_normalized_doi": doi,
                "drug_name": drug["value"],
                "la_fraction": la_f["value"],
                "ga_fraction": ga_f["value"],
                "la_ga_ratio": la_ga["value"],
                "polymer_mw_lower_kDa": mw_lo["value"],
                "polymer_mw_upper_kDa": mw_hi["value"],
                "organic_solvent": solvent["value"],
                "drug_to_polymer_mass_ratio": ratio["value"],
                "core_signature": signature,
            }
        )

    core_base_df = pd.DataFrame(core_rows)
    if core_base_df.empty:
        empty_cols_core = [
            "formulation_core_id",
            "reference_normalized_doi",
            "core_signature",
            "drug_name",
            "la_fraction",
            "ga_fraction",
            "la_ga_ratio",
            "polymer_mw_lower_kDa",
            "polymer_mw_upper_kDa",
            "organic_solvent",
            "drug_to_polymer_mass_ratio",
            "example_group_keys",
            "created_at",
            "schema_version",
        ]
        return (
            pd.DataFrame(columns=empty_cols_core),
            pd.DataFrame(),
            pd.DataFrame(),
            {"fresh": 0, "postprocess": 0},
            pd.DataFrame(),
        )

    # One formulation_core row per DOI + core_signature.
    core_uni = (
        core_base_df.groupby(["reference_normalized_doi", "core_signature"], dropna=False)
        .agg(
            drug_name=("drug_name", "first"),
            la_fraction=("la_fraction", "first"),
            ga_fraction=("ga_fraction", "first"),
            la_ga_ratio=("la_ga_ratio", "first"),
            polymer_mw_lower_kDa=("polymer_mw_lower_kDa", "first"),
            polymer_mw_upper_kDa=("polymer_mw_upper_kDa", "first"),
            organic_solvent=("organic_solvent", "first"),
            drug_to_polymer_mass_ratio=("drug_to_polymer_mass_ratio", "first"),
            example_group_keys=("group_key", lambda s: "|".join(sorted(set(s.tolist()))[:5])),
        )
        .reset_index()
    )

    # Stable formulation_core_id assignment.
    core_uni = core_uni.sort_values(["reference_normalized_doi", "core_signature"]).reset_index(drop=True)
    core_uni["formulation_core_id"] = [
        f"FC_{i+1:05d}" for i in range(len(core_uni))
    ]
    core_uni["created_at"] = datetime.now(timezone.utc).isoformat()
    core_uni["schema_version"] = SCHEMA_VERSION

    core_df = core_uni[
        [
            "formulation_core_id",
            "reference_normalized_doi",
            "core_signature",
            "drug_name",
            "la_fraction",
            "ga_fraction",
            "la_ga_ratio",
            "polymer_mw_lower_kDa",
            "polymer_mw_upper_kDa",
            "organic_solvent",
            "drug_to_polymer_mass_ratio",
            "example_group_keys",
            "created_at",
            "schema_version",
        ]
    ].copy()

    core_key = core_uni[["reference_normalized_doi", "core_signature", "formulation_core_id"]].copy()
    assign_df = core_base_df.merge(
        core_key,
        on=["reference_normalized_doi", "core_signature"],
        how="left",
    )

    # Measurement rows: include derived measurement fields + selected extracted raw fields.
    measurement_rows = []
    measurement_field_map = {
        "particle_size": ("size_nm", "nm"),
        "EE": ("ee_percent", "percent"),
        "LC": ("lc_percent", "percent"),
    }

    derived_meas = derived[derived["field_name"].isin(measurement_field_map.keys())].copy()
    extracted_idx = extracted.set_index("group_key", drop=False)

    for _, row in derived_meas.iterrows():
        gk = str(row.get("group_key", ""))
        ex = extracted_idx.loc[gk] if gk in extracted_idx.index else None
        if isinstance(ex, pd.DataFrame):
            ex = ex.iloc[0]
        ev = short_text(ex.get("evidence_span_text", ""), 200) if ex is not None else ""
        cond_tag, cond_kw = detect_condition(ev)
        mtype, unit = measurement_field_map[str(row.get("field_name", ""))]
        fcid = assign_df.loc[assign_df["group_key"] == gk, "formulation_core_id"]
        formulation_core_id = str(fcid.iloc[0]) if len(fcid) else ""
        measurement_rows.append(
            {
                "formulation_core_id": formulation_core_id,
                "group_key": gk,
                "measurement_type": mtype,
                "measurement_value": str(row.get("value", "")),
                "unit": unit,
                "condition_tag": cond_tag,
                "condition_keywords": cond_kw,
                "evidence_excerpt": ev,
                "rule_id": str(row.get("rule_id", "")),
                "value_source": str(row.get("value_source", "")),
                "trace_pointer": str(row.get("trace_pointer", "")),
            }
        )

    # Add selected raw extracted measurement fields if present.
    raw_fields = [("pdi", "pdi", ""), ("zeta_mV", "zeta_mv", "mV")]
    for _, ex in extracted.iterrows():
        gk = str(ex.get("group_key", ""))
        fcid = assign_df.loc[assign_df["group_key"] == gk, "formulation_core_id"]
        formulation_core_id = str(fcid.iloc[0]) if len(fcid) else ""
        ev = short_text(ex.get("evidence_span_text", ""), 200)
        cond_tag, cond_kw = detect_condition(ev)
        for raw_col, mtype, unit in raw_fields:
            val = str(ex.get(raw_col, "")).strip()
            if val:
                measurement_rows.append(
                    {
                        "formulation_core_id": formulation_core_id,
                        "group_key": gk,
                        "measurement_type": mtype,
                        "measurement_value": val,
                        "unit": unit,
                        "condition_tag": cond_tag,
                        "condition_keywords": cond_kw,
                        "evidence_excerpt": ev,
                        "rule_id": "",
                        "value_source": "extracted_raw",
                        "trace_pointer": "",
                    }
                )

    measurements_df = pd.DataFrame(measurement_rows)
    if measurements_df.empty:
        measurements_df = pd.DataFrame(
            columns=[
                "formulation_core_id",
                "group_key",
                "measurement_type",
                "measurement_value",
                "unit",
                "condition_tag",
                "condition_keywords",
                "evidence_excerpt",
                "rule_id",
                "value_source",
                "trace_pointer",
            ]
        )
    else:
        measurements_df = measurements_df[
            [
                "formulation_core_id",
                "group_key",
                "measurement_type",
                "measurement_value",
                "unit",
                "condition_tag",
                "condition_keywords",
                "evidence_excerpt",
                "rule_id",
                "value_source",
                "trace_pointer",
            ]
        ].copy()

    # Assignment trace.
    trace_rows = []
    for _, a in assign_df.iterrows():
        ex = extracted_idx.loc[a["group_key"]] if a["group_key"] in extracted_idx.index else None
        if isinstance(ex, pd.DataFrame):
            ex = ex.iloc[0]
        evidence_text = short_text(ex.get("evidence_span_text", ""), 200) if ex is not None else ""
        cond_tag, cond_kw = detect_condition(evidence_text)
        trace_rows.append(
            {
                "group_key": str(a.get("group_key", "")),
                "formulation_core_id": str(a.get("formulation_core_id", "")),
                "core_signature": str(a.get("core_signature", "")),
                "assignment_reason": "assigned_by_doi_plus_core_signature",
                "condition_tag": cond_tag,
                "matched_keywords": cond_kw,
            }
        )
    assignment_trace_df = pd.DataFrame(trace_rows)

    condition_dist = (
        measurements_df["condition_tag"].value_counts().to_dict()
        if not measurements_df.empty
        else {"fresh": 0, "postprocess": 0}
    )

    doi_core_measure = pd.DataFrame()
    if not measurements_df.empty and not core_df.empty:
        m = measurements_df.merge(
            assign_df[["group_key", "reference_normalized_doi"]].drop_duplicates("group_key"),
            on="group_key",
            how="left",
        )
        doi_core_measure = (
            m.groupby(["reference_normalized_doi", "formulation_core_id"], dropna=False)
            .size()
            .reset_index(name="measurement_rows_per_core")
            .sort_values("measurement_rows_per_core", ascending=False)
        )

    return core_df, measurements_df, assignment_trace_df, condition_dist, doi_core_measure


def short_text(value: Any, limit: int = 200) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def main() -> None:
    args = parse_args()
    run_id = args.run_id

    base_dir = Path(f"data/results/{run_id}/benchmark_goren_2025")
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "schema_v1"

    extracted_path = Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    derived_path = base_dir / "derived_values.tsv"
    projection_trace_path = base_dir / "projection_trace.tsv"
    sample_manifest_path = Path("data/cleaned/samples/sample_goren18.tsv")
    deriv_rules_path = Path("data/benchmark/goren_2025/rules/derivation_rule_registry.v1.json")
    proj_rules_path = Path("data/benchmark/goren_2025/rules/projection_ruleset.v1.json")

    required = [extracted_path, derived_path, projection_trace_path, deriv_rules_path, proj_rules_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    derived = pd.read_csv(derived_path, sep="\t", dtype=str).fillna("")
    extracted = pd.read_csv(extracted_path, sep="\t", dtype=str).fillna("")
    projection_trace = pd.read_csv(projection_trace_path, sep="\t", dtype=str).fillna("")
    extracted["group_key"] = (
        extracted["key"].astype(str).str.strip() + "::" + extracted["formulation_id"].astype(str).str.strip()
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build once.
    key_to_doi_map: dict[str, str] = {}
    if sample_manifest_path.exists():
        sm = pd.read_csv(sample_manifest_path, sep="\t", dtype=str).fillna("")
        if "key" in sm.columns and "doi" in sm.columns:
            sm["doi_norm"] = sm["doi"].map(normalize_doi)
            key_to_doi_map = dict(zip(sm["key"], sm["doi_norm"]))

    core_df, measurements_df, assignment_trace_df, condition_dist, doi_core_measure = build_dataframes(
        derived=derived,
        extracted=extracted,
        projection_trace=projection_trace,
        key_to_doi_map=key_to_doi_map,
    )

    # Determinism check: recompute and compare core signature counts.
    core_df_2, _, _, _, _ = build_dataframes(
        derived=derived,
        extracted=extracted,
        projection_trace=projection_trace,
        key_to_doi_map=key_to_doi_map,
    )
    c1 = core_df["core_signature"].value_counts().sort_index()
    c2 = core_df_2["core_signature"].value_counts().sort_index()
    deterministic = c1.equals(c2)

    core_out = out_dir / "formulation_core.tsv"
    meas_out = out_dir / "measurements.tsv"
    trace_out = out_dir / "core_assignment_trace.tsv"

    core_df.to_csv(core_out, sep="\t", index=False)
    measurements_df.to_csv(meas_out, sep="\t", index=False)
    assignment_trace_df.to_csv(trace_out, sep="\t", index=False)

    unique_dois = int(core_df["reference_normalized_doi"].replace("", pd.NA).dropna().nunique()) if not core_df.empty else 0
    top10 = doi_core_measure.head(10).to_dict(orient="records") if not doi_core_measure.empty else []

    print(f"run_id={run_id}")
    print(f"unique_dois={unique_dois}")
    print(f"formulation_core_rows={len(core_df)}")
    print(f"measurement_rows={len(measurements_df)}")
    print(f"condition_tag_distribution={json.dumps(condition_dist, ensure_ascii=False)}")
    print(f"top10_doi_measurement_rows_per_core={json.dumps(top10, ensure_ascii=False)}")
    print(f"determinism_core_signature_counts={deterministic}")
    print(f"output_formulation_core={core_out}")
    print(f"output_measurements={meas_out}")
    print(f"output_assignment_trace={trace_out}")


if __name__ == "__main__":
    main()
