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
SCHEMA_VERSION = "schema_v2"
RULESET_PATH = Path("data/benchmark/goren_2025/rules/formulation_core_signature_ruleset.v2.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build two-table schema v2 for Goren benchmark outputs."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/schema_v2",
    )
    parser.add_argument("--ruleset", default=str(RULESET_PATH))
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
        return ""
    try:
        f = float(s)
    except ValueError:
        return normalize_text(s)
    if f.is_integer():
        return str(int(f))
    return f"{f:.6g}"


def normalize_numeric_text(value: str) -> str:
    return norm_num_token(value)


def short_text(value: Any, limit: int = 200) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


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


def parse_percent_like(value: str) -> str:
    s = normalize_text(value)
    if not s:
        return ""
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", s)
    if not m:
        return ""
    return norm_num_token(m.group(1))


def normalize_formulation_id_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip().upper()
    if not s:
        return ""
    s = re.sub(r"[^A-Z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("_-")


def has_explicit_formulation_id(value: Any) -> bool:
    tok = normalize_formulation_id_token(value)
    if not tok:
        return False
    # Require mixed alpha+numeric token to avoid treating plain numeric row indices as explicit IDs.
    return bool(re.search(r"[A-Z]", tok) and re.search(r"\d", tok))


def detect_surfactant_type(notes: str, evidence_excerpt: str) -> str:
    text = normalize_text(f"{notes} {evidence_excerpt}")
    patterns = [
        (r"\bpva\b|polyvinyl alcohol", "pva"),
        (r"pluronic\s*f[-\s]?\d+|f68|f127", "pluronic"),
        (r"tween\s*[-\s]?\d+|polysorbate", "tween"),
        (r"\bspan\s*[-\s]?\d+", "span"),
        (r"\bpoloxamer\b", "poloxamer"),
        (r"\bsds\b|sodium dodecyl sulfate", "sds"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return ""


def detect_condition(text: str, keyword_patterns: list[str]) -> tuple[str, str]:
    raw = normalize_text(text)
    matched = []
    for p in keyword_patterns:
        if re.search(p, raw, flags=re.IGNORECASE):
            matched.append(p)
    if matched:
        return "postprocess", "|".join(sorted(set(matched)))
    return "fresh", ""


def missing_token(field_name: str) -> str:
    return f"MISSING_{field_name}"


def stable_signature(parts: list[str]) -> str:
    joined = "||".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"{joined}||sig={digest}"


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def build_collision_groups(core_base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if core_base_df.empty:
        return core_base_df.copy(), pd.DataFrame(
            columns=[
                "reference_normalized_doi",
                "core_signature",
                "collision_group_size",
                "group_key",
                "formulation_id",
                "normalized_formulation_id",
                "representative_source_formulation_id",
                "instance_assignment_row_id",
                "decision",
            ]
        )

    grouped_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, str]] = []

    for (_, _), group in core_base_df.groupby(["reference_normalized_doi", "core_signature"], dropna=False, sort=False):
        group = group.copy().reset_index(drop=True)
        group_size = len(group)
        decision = "collapsed"

        if group_size > 1:
            nonempty_ids = sorted(
                {
                    str(v).strip()
                    for v in group["normalized_formulation_id"].tolist()
                    if str(v).strip()
                }
            )
            distinct_rep_ids = sorted(
                {
                    str(v).strip()
                    for v in group["representative_source_formulation_id"].tolist()
                    if str(v).strip()
                }
            )
            distinct_assignment_ids = sorted(
                {
                    str(v).strip()
                    for v in group["instance_assignment_row_id"].tolist()
                    if str(v).strip()
                }
            )
            has_distinct_upstream_identity = len(distinct_rep_ids) >= 2 or len(distinct_assignment_ids) >= 2
            if len(nonempty_ids) >= 2 and has_distinct_upstream_identity:
                decision = "split_by_formulation_id"

        for _, row in group.iterrows():
            debug_rows.append(
                {
                    "reference_normalized_doi": str(row.get("reference_normalized_doi", "")),
                    "core_signature": str(row.get("core_signature", "")),
                    "collision_group_size": str(group_size),
                    "group_key": str(row.get("group_key", "")),
                    "formulation_id": str(row.get("formulation_id", "")),
                    "normalized_formulation_id": str(row.get("normalized_formulation_id", "")),
                    "representative_source_formulation_id": str(row.get("representative_source_formulation_id", "")),
                    "instance_assignment_row_id": str(row.get("instance_assignment_row_id", "")),
                    "decision": decision,
                }
            )

        if decision == "split_by_formulation_id":
            for formulation_id, sub in group.groupby("normalized_formulation_id", dropna=False, sort=False):
                subgroup = sub.copy()
                subgroup["collapse_group_key"] = subgroup["core_signature"] + "||split_formulation_id=" + str(formulation_id)
                grouped_rows.extend(subgroup.to_dict(orient="records"))
        else:
            group["collapse_group_key"] = group["core_signature"]
            grouped_rows.extend(group.to_dict(orient="records"))

    return pd.DataFrame(grouped_rows), pd.DataFrame(debug_rows)


def build_group_doi_map(
    projection_trace: pd.DataFrame,
    key_to_doi_map: dict[str, str],
    group_keys: pd.Series,
) -> dict[str, str]:
    ref_map = (
        projection_trace[projection_trace["curated_column"] == "reference"][
            ["group_key", "projected_value"]
        ]
        .drop_duplicates(subset=["group_key"], keep="first")
        .copy()
    )
    ref_map["reference_normalized_doi"] = ref_map["projected_value"].map(normalize_doi)
    group_to_doi = dict(zip(ref_map["group_key"], ref_map["reference_normalized_doi"]))
    for gk in group_keys.tolist():
        if gk in group_to_doi and group_to_doi[gk]:
            continue
        key = str(gk).split("::", 1)[0]
        group_to_doi[gk] = key_to_doi_map.get(key, "")
    return group_to_doi


def build_tables(
    derived: pd.DataFrame,
    extracted: pd.DataFrame,
    projection_trace: pd.DataFrame,
    key_to_doi_map: dict[str, str],
    ruleset: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], pd.DataFrame, pd.DataFrame]:
    included_fields: list[str] = ruleset["signature_composition"]["ordered_fields"]
    excluded_keywords: list[str] = ruleset["excluded_keywords_postprocess"]

    extracted_idx = extracted.set_index("group_key", drop=False)
    group_to_doi = build_group_doi_map(projection_trace, key_to_doi_map, derived["group_key"])

    core_rows = []
    for gk, gdf in derived.groupby("group_key", dropna=False):
        gk_str = str(gk)
        key = str(gdf.iloc[0].get("key", ""))
        formulation_id = str(gdf.iloc[0].get("formulation_id", ""))
        formulation_id_token = normalize_formulation_id_token(formulation_id)
        explicit_formulation_id = has_explicit_formulation_id(formulation_id)
        ex = extracted_idx.loc[gk_str] if gk_str in extracted_idx.index else None
        if isinstance(ex, pd.DataFrame):
            ex = ex.iloc[0]

        evidence_excerpt = short_text(ex.get("evidence_span_text", ""), 200) if ex is not None else ""
        notes_text = str(ex.get("notes", "")) if ex is not None else ""

        doi = group_to_doi.get(gk_str, "")
        drug = pick_best_field(gdf, "small_molecule_name")["value"]
        la_f = pick_best_field(gdf, "la_fraction")["value"]
        ga_f = pick_best_field(gdf, "ga_fraction")["value"]
        la_ga_ratio = pick_best_field(gdf, "LA/GA")["value"]
        mw_lo = pick_best_field(gdf, "polymer_mw_lower_kDa")["value"]
        mw_hi = pick_best_field(gdf, "polymer_mw_upper_kDa")["value"]
        solvent = pick_best_field(gdf, "solvent")["value"]
        dpr = pick_best_field(gdf, "drug/polymer")["value"]
        plga_mass_mg = pick_best_field(gdf, "polymer_mass_mg")["value"]
        surfactant_conc = pick_best_field(gdf, "surfactant_concentration")["value"]

        pva_from_extracted = str(ex.get("pva_conc_percent", "")) if ex is not None else ""
        pva_conc_percent = normalize_numeric_text(pva_from_extracted)
        if not pva_conc_percent:
            pva_conc_percent = parse_percent_like(surfactant_conc)
        surfactant_conc_percent = parse_percent_like(surfactant_conc)
        surfactant_type = detect_surfactant_type(notes_text, evidence_excerpt)

        polymer_mw_kda = ""
        mw_lo_norm = normalize_numeric_text(mw_lo)
        mw_hi_norm = normalize_numeric_text(mw_hi)
        if mw_lo_norm and mw_hi_norm and mw_lo_norm == mw_hi_norm:
            polymer_mw_kda = mw_lo_norm

        core_values_raw = {
            "reference_normalized_doi": normalize_doi(doi),
            "drug_name_normalized": normalize_text(drug),
            "la_fraction": normalize_numeric_text(la_f),
            "ga_fraction": normalize_numeric_text(ga_f),
            "la_ga_ratio": normalize_numeric_text(la_ga_ratio),
            "polymer_mw_lower_kDa": mw_lo_norm,
            "polymer_mw_upper_kDa": mw_hi_norm,
            "polymer_mw_kDa": polymer_mw_kda,
            "organic_solvent_normalized": normalize_text(solvent),
            "surfactant_type_normalized": surfactant_type,
            "pva_conc_percent": pva_conc_percent,
            "surfactant_conc_percent": surfactant_conc_percent,
            "plga_conc": "",
            "plga_mass_mg": normalize_numeric_text(plga_mass_mg),
            "aqueous_phase_pH": "",
            "drug_to_polymer_mass_ratio": normalize_numeric_text(dpr),
            "phase_ratio_w1_o": "",
            "phase_ratio_total_water_o": "",
        }

        missing_fields = [field for field in included_fields if not core_values_raw.get(field, "")]
        signature_parts = [
            f"{field}={core_values_raw[field] if core_values_raw[field] else missing_token(field)}"
            for field in included_fields
        ]
        if explicit_formulation_id:
            signature_parts.append(f"explicit_formulation_id={formulation_id_token}")
        core_signature = stable_signature(signature_parts)

        condition_tag, matched_keywords = detect_condition(evidence_excerpt, excluded_keywords)
        representative_source_formulation_id = first_nonempty(
            gdf.iloc[0].get("representative_source_formulation_id", ""),
            ex.get("representative_source_formulation_id", "") if ex is not None else "",
            ex.get("source_formulation_id", "") if ex is not None else "",
            ex.get("article_formulation_label", "") if ex is not None else "",
        )
        instance_assignment_row_id = first_nonempty(
            gdf.iloc[0].get("instance_assignment_row_id", ""),
            gdf.iloc[0].get("instance_id", ""),
            ex.get("instance_assignment_row_id", "") if ex is not None else "",
            ex.get("instance_id", "") if ex is not None else "",
            gk_str,
        )

        core_rows.append(
            {
                "group_key": gk_str,
                "key": key,
                "formulation_id": formulation_id,
                "normalized_formulation_id": formulation_id_token,
                "representative_source_formulation_id": representative_source_formulation_id,
                "instance_assignment_row_id": instance_assignment_row_id,
                "core_signature": core_signature,
                "condition_tag": condition_tag,
                "matched_keywords": matched_keywords,
                "explicit_formulation_id": "1" if explicit_formulation_id else "0",
                "explicit_formulation_id_token": formulation_id_token if explicit_formulation_id else "",
                "included_field_values": "|".join(
                    [
                        f"{field}:{core_values_raw[field] if core_values_raw[field] else missing_token(field)}"
                        for field in included_fields
                    ]
                    + (
                        [f"explicit_formulation_id:{formulation_id_token}"]
                        if explicit_formulation_id
                        else []
                    )
                ),
                "missing_fields": "|".join(missing_fields),
                **core_values_raw,
            }
        )

    core_base_df = pd.DataFrame(core_rows)
    if core_base_df.empty:
        empty_core_cols = [
            "formulation_core_id",
            "reference_normalized_doi",
            "core_signature",
            "drug_name_normalized",
            "la_fraction",
            "ga_fraction",
            "la_ga_ratio",
            "polymer_mw_lower_kDa",
            "polymer_mw_upper_kDa",
            "polymer_mw_kDa",
            "organic_solvent_normalized",
            "surfactant_type_normalized",
            "pva_conc_percent",
            "surfactant_conc_percent",
            "plga_conc",
            "plga_mass_mg",
            "aqueous_phase_pH",
            "drug_to_polymer_mass_ratio",
            "phase_ratio_w1_o",
            "phase_ratio_total_water_o",
            "example_group_keys",
            "created_at",
            "schema_version",
        ]
        return (
            pd.DataFrame(columns=empty_core_cols),
            pd.DataFrame(),
            pd.DataFrame(),
            {"fresh": 0, "postprocess": 0},
            pd.DataFrame(),
            pd.DataFrame(),
        )

    collapse_base_df, collapse_debug_df = build_collision_groups(core_base_df)

    core_uni = (
        collapse_base_df.groupby(["reference_normalized_doi", "collapse_group_key"], dropna=False)
        .agg(
            core_signature=("core_signature", "first"),
            drug_name_normalized=("drug_name_normalized", "first"),
            la_fraction=("la_fraction", "first"),
            ga_fraction=("ga_fraction", "first"),
            la_ga_ratio=("la_ga_ratio", "first"),
            polymer_mw_lower_kDa=("polymer_mw_lower_kDa", "first"),
            polymer_mw_upper_kDa=("polymer_mw_upper_kDa", "first"),
            polymer_mw_kDa=("polymer_mw_kDa", "first"),
            organic_solvent_normalized=("organic_solvent_normalized", "first"),
            surfactant_type_normalized=("surfactant_type_normalized", "first"),
            pva_conc_percent=("pva_conc_percent", "first"),
            surfactant_conc_percent=("surfactant_conc_percent", "first"),
            plga_conc=("plga_conc", "first"),
            plga_mass_mg=("plga_mass_mg", "first"),
            aqueous_phase_pH=("aqueous_phase_pH", "first"),
            drug_to_polymer_mass_ratio=("drug_to_polymer_mass_ratio", "first"),
            phase_ratio_w1_o=("phase_ratio_w1_o", "first"),
            phase_ratio_total_water_o=("phase_ratio_total_water_o", "first"),
            example_group_keys=("group_key", lambda s: "|".join(sorted(set(s.tolist()))[:8])),
        )
        .reset_index()
    )
    core_uni = core_uni.sort_values(["reference_normalized_doi", "core_signature"]).reset_index(drop=True)
    core_uni["formulation_core_id"] = [f"FC2_{i + 1:05d}" for i in range(len(core_uni))]
    core_uni["created_at"] = datetime.now(timezone.utc).isoformat()
    core_uni["schema_version"] = SCHEMA_VERSION

    core_df = core_uni[
        [
            "formulation_core_id",
            "reference_normalized_doi",
            "core_signature",
            "drug_name_normalized",
            "la_fraction",
            "ga_fraction",
            "la_ga_ratio",
            "polymer_mw_lower_kDa",
            "polymer_mw_upper_kDa",
            "polymer_mw_kDa",
            "organic_solvent_normalized",
            "surfactant_type_normalized",
            "pva_conc_percent",
            "surfactant_conc_percent",
            "plga_conc",
            "plga_mass_mg",
            "aqueous_phase_pH",
            "drug_to_polymer_mass_ratio",
            "phase_ratio_w1_o",
            "phase_ratio_total_water_o",
            "example_group_keys",
            "created_at",
            "schema_version",
        ]
    ].copy()

    core_key = core_uni[["reference_normalized_doi", "collapse_group_key", "formulation_core_id"]].copy()
    assign_df = collapse_base_df.merge(
        core_key,
        on=["reference_normalized_doi", "collapse_group_key"],
        how="left",
    )

    measurement_field_map = {
        "particle_size": ("size_nm", "nm"),
        "EE": ("ee_percent", "percent"),
        "LC": ("lc_percent", "percent"),
    }
    measurement_rows: list[dict[str, str]] = []
    derived_meas = derived[derived["field_name"].isin(measurement_field_map.keys())].copy()
    for _, row in derived_meas.iterrows():
        gk = str(row.get("group_key", ""))
        ex = extracted_idx.loc[gk] if gk in extracted_idx.index else None
        if isinstance(ex, pd.DataFrame):
            ex = ex.iloc[0]
        ev = short_text(ex.get("evidence_span_text", ""), 200) if ex is not None else ""
        cond_tag, cond_kw = detect_condition(ev, excluded_keywords)
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

    raw_fields = [("pdi", "pdi", ""), ("zeta_mV", "zeta_mv", "mV")]
    for _, ex in extracted.iterrows():
        gk = str(ex.get("group_key", ""))
        fcid = assign_df.loc[assign_df["group_key"] == gk, "formulation_core_id"]
        formulation_core_id = str(fcid.iloc[0]) if len(fcid) else ""
        ev = short_text(ex.get("evidence_span_text", ""), 200)
        cond_tag, cond_kw = detect_condition(ev, excluded_keywords)
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

    trace_df = assign_df[
        [
            "group_key",
            "formulation_core_id",
            "core_signature",
            "included_field_values",
            "missing_fields",
            "condition_tag",
            "matched_keywords",
        ]
    ].copy()
    trace_df["assignment_reason"] = "assigned_by_doi_plus_v2_core_signature"
    trace_df = trace_df[
        [
            "group_key",
            "formulation_core_id",
            "core_signature",
            "included_field_values",
            "missing_fields",
            "condition_tag",
            "matched_keywords",
            "assignment_reason",
        ]
    ]

    condition_dist = (
        measurements_df["condition_tag"].value_counts().to_dict()
        if not measurements_df.empty
        else {"fresh": 0, "postprocess": 0}
    )

    doi_core_measure = pd.DataFrame()
    if not measurements_df.empty and not core_df.empty:
        gk_to_doi = assign_df[["group_key", "reference_normalized_doi"]].drop_duplicates("group_key")
        merged = measurements_df.merge(gk_to_doi, on="group_key", how="left")
        doi_core_measure = (
            merged.groupby(["reference_normalized_doi", "formulation_core_id"], dropna=False)
            .size()
            .reset_index(name="measurement_rows_per_core")
            .sort_values("measurement_rows_per_core", ascending=False)
        )

    return core_df, measurements_df, trace_df, condition_dist, doi_core_measure, collapse_debug_df


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    ruleset_path = Path(args.ruleset)

    base_dir = Path(f"data/results/{run_id}/benchmark_goren_2025")
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "schema_v2"

    extracted_path = Path(f"data/results/{run_id}/weak_labels__gemini.tsv")
    derived_path = base_dir / "derived_values.tsv"
    projection_trace_path = base_dir / "projection_trace.tsv"
    sample_manifest_path = Path("data/cleaned/samples/sample_goren18.tsv")
    v1_core_path = base_dir / "schema_v1/formulation_core.tsv"
    deriv_rules_path = Path("data/benchmark/goren_2025/rules/derivation_rule_registry.v1.json")
    proj_rules_path = Path("data/benchmark/goren_2025/rules/projection_ruleset.v1.json")

    required = [
        extracted_path,
        derived_path,
        projection_trace_path,
        deriv_rules_path,
        proj_rules_path,
        ruleset_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    with ruleset_path.open("r", encoding="utf-8") as f:
        ruleset = json.load(f)

    derived = pd.read_csv(derived_path, sep="\t", dtype=str).fillna("")
    extracted = pd.read_csv(extracted_path, sep="\t", dtype=str).fillna("")
    projection_trace = pd.read_csv(projection_trace_path, sep="\t", dtype=str).fillna("")
    extracted["group_key"] = (
        extracted["key"].astype(str).str.strip() + "::" + extracted["formulation_id"].astype(str).str.strip()
    )

    key_to_doi_map: dict[str, str] = {}
    if sample_manifest_path.exists():
        sm = pd.read_csv(sample_manifest_path, sep="\t", dtype=str).fillna("")
        if "key" in sm.columns and "doi" in sm.columns:
            sm["doi_norm"] = sm["doi"].map(normalize_doi)
            key_to_doi_map = dict(zip(sm["key"], sm["doi_norm"]))

    out_dir.mkdir(parents=True, exist_ok=True)

    core_df, measurements_df, trace_df, condition_dist, doi_core_measure, collapse_debug_df = build_tables(
        derived=derived,
        extracted=extracted,
        projection_trace=projection_trace,
        key_to_doi_map=key_to_doi_map,
        ruleset=ruleset,
    )

    # Determinism check: re-run in-memory and compare signature counts.
    core_df_2, _, _, _, _, _ = build_tables(
        derived=derived,
        extracted=extracted,
        projection_trace=projection_trace,
        key_to_doi_map=key_to_doi_map,
        ruleset=ruleset,
    )
    c1 = core_df["core_signature"].value_counts().sort_index()
    c2 = core_df_2["core_signature"].value_counts().sort_index()
    deterministic = c1.equals(c2)

    core_out = out_dir / "formulation_core.tsv"
    meas_out = out_dir / "measurements.tsv"
    trace_out = out_dir / "core_assignment_trace.tsv"
    collapse_debug_out = out_dir / "schema_v2_collapse_debug.tsv"
    core_df.to_csv(core_out, sep="\t", index=False)
    measurements_df.to_csv(meas_out, sep="\t", index=False)
    trace_df.to_csv(trace_out, sep="\t", index=False)
    collapse_debug_df.to_csv(collapse_debug_out, sep="\t", index=False)

    unique_dois_v2 = int(core_df["reference_normalized_doi"].replace("", pd.NA).dropna().nunique()) if not core_df.empty else 0

    v1_per_doi = pd.DataFrame(columns=["reference_normalized_doi", "v1_core_count"])
    if v1_core_path.exists():
        v1_core = pd.read_csv(v1_core_path, sep="\t", dtype=str).fillna("")
        if "reference_normalized_doi" in v1_core.columns:
            v1_per_doi = (
                v1_core.groupby("reference_normalized_doi", dropna=False)
                .size()
                .reset_index(name="v1_core_count")
            )
    v2_per_doi = (
        core_df.groupby("reference_normalized_doi", dropna=False)
        .size()
        .reset_index(name="v2_core_count")
    )
    compare = v2_per_doi.merge(v1_per_doi, on="reference_normalized_doi", how="outer").fillna(0)
    compare["v1_core_count"] = compare["v1_core_count"].astype(int)
    compare["v2_core_count"] = compare["v2_core_count"].astype(int)
    compare["increase"] = compare["v2_core_count"] - compare["v1_core_count"]
    top10_increase = (
        compare.sort_values(["increase", "v2_core_count"], ascending=[False, False]).head(10).to_dict(orient="records")
    )

    doi_focus = "10.1002/jps.24101"
    focus_row = compare[compare["reference_normalized_doi"] == doi_focus]
    focus_payload = (
        focus_row.iloc[0][["reference_normalized_doi", "v1_core_count", "v2_core_count", "increase"]].to_dict()
        if not focus_row.empty
        else {
            "reference_normalized_doi": doi_focus,
            "v1_core_count": 0,
            "v2_core_count": 0,
            "increase": 0,
        }
    )

    print(f"run_id={run_id}")
    print(f"unique_dois_v2={unique_dois_v2}")
    print(f"formulation_core_rows_v2={len(core_df)}")
    print(f"measurement_rows_v2={len(measurements_df)}")
    print(f"condition_tag_distribution_v2={json.dumps(condition_dist, ensure_ascii=False)}")
    print(f"top10_doi_core_count_increase_v2_vs_v1={json.dumps(top10_increase, ensure_ascii=False)}")
    print(f"doi_10.1002/jps.24101_v2_vs_v1={json.dumps(focus_payload, ensure_ascii=False)}")
    print(f"determinism_core_signature_counts={deterministic}")
    print(f"output_formulation_core={core_out}")
    print(f"output_measurements={meas_out}")
    print(f"output_assignment_trace={trace_out}")
    print(f"output_collapse_debug={collapse_debug_out}")


if __name__ == "__main__":
    main()
