#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
SCHEMA_VERSION = "schema_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build schema_v3 by expanding schema_v2 core splitting with DOE factor signatures."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/schema_v3",
    )
    parser.add_argument(
        "--partial-varying-threshold",
        type=int,
        default=2,
        help="Minimum varying DOE keys required to treat partial DOI as DOE-enabled.",
    )
    parser.add_argument(
        "--max-doe-keys",
        type=int,
        default=8,
        help="Max DOE keys used in DOE signature per DOI.",
    )
    parser.add_argument(
        "--min-decoded-rate",
        type=float,
        default=0.8,
        help="Hard gate: minimum decoded rate required for DOE enablement.",
    )
    return parser.parse_args()


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def load_required(run_id: str) -> dict[str, pd.DataFrame]:
    base = Path(f"data/results/{run_id}/benchmark_goren_2025")
    p_core_v2 = base / "schema_v2/formulation_core.tsv"
    p_meas_v2 = base / "schema_v2/measurements.tsv"
    p_trace_v2 = base / "schema_v2/core_assignment_trace.tsv"
    p_doe_diag = base / "derivation_v1/doe_decode_diagnostics.tsv"
    p_doe_rows = base / "derivation_v1/doe_factor_rows.tsv"

    required = [p_core_v2, p_meas_v2, p_trace_v2, p_doe_diag, p_doe_rows]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    return {
        "core_v2": pd.read_csv(p_core_v2, sep="\t", dtype=str).fillna(""),
        "meas_v2": pd.read_csv(p_meas_v2, sep="\t", dtype=str).fillna(""),
        "trace_v2": pd.read_csv(p_trace_v2, sep="\t", dtype=str).fillna(""),
        "doe_diag": pd.read_csv(p_doe_diag, sep="\t", dtype=str).fillna(""),
        "doe_rows": pd.read_csv(p_doe_rows, sep="\t", dtype=str).fillna(""),
    }


def choose_doe_key_values(doi_rows: pd.DataFrame) -> tuple[dict[str, dict[str, str]], list[str]]:
    # For each factor key, prefer decoded values if available; fallback to coded.
    # Returns group_key -> factor -> value, and selected varying keys.
    gk_factor_value: dict[str, dict[str, str]] = {}
    varying_keys: list[str] = []
    keys = sorted(set(doi_rows["factor_name_normalized"].astype(str).tolist()))
    for fk in keys:
        sub = doi_rows[doi_rows["factor_name_normalized"] == fk]
        sub_dec = sub[sub["factor_kind"] == "decoded"]
        sub_cod = sub[sub["factor_kind"] == "coded"]
        use = sub_dec if not sub_dec.empty else sub_cod
        if use.empty:
            continue

        vals_per_gk: dict[str, str] = {}
        for gk, gdf in use.groupby("group_key", sort=True):
            if not sub_dec.empty:
                v = gdf["factor_value_num"].where(
                    gdf["factor_value_num"].astype(str).str.strip() != "",
                    gdf["factor_value_text"],
                ).iloc[0]
            else:
                v = gdf["factor_value_code"].iloc[0]
            vals_per_gk[str(gk)] = str(v).strip()

        uniq = sorted({v for v in vals_per_gk.values() if v != ""})
        if len(uniq) > 1:
            varying_keys.append(fk)
        for gk, v in vals_per_gk.items():
            gk_factor_value.setdefault(gk, {})
            gk_factor_value[gk][fk] = v
    return gk_factor_value, sorted(varying_keys)


def build_once(
    *,
    core_v2: pd.DataFrame,
    meas_v2: pd.DataFrame,
    trace_v2: pd.DataFrame,
    doe_diag: pd.DataFrame,
    doe_rows: pd.DataFrame,
    partial_varying_threshold: int,
    max_doe_keys: int,
    min_decoded_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    core_v2 = core_v2.copy()
    trace_v2 = trace_v2.copy()
    meas_v2 = meas_v2.copy()
    doe_diag = doe_diag.copy()
    doe_rows = doe_rows.copy()

    core_v2["reference_normalized_doi"] = core_v2["reference_normalized_doi"].astype(str).str.strip().str.lower()
    doe_diag["doi"] = doe_diag["doi"].astype(str).str.strip().str.lower()
    doe_diag["status"] = doe_diag["status"].astype(str).str.strip().str.lower()

    # Map group_key -> v2 core + doi
    gk_map = trace_v2.merge(
        core_v2[["formulation_core_id", "reference_normalized_doi", "core_signature"]],
        on="formulation_core_id",
        how="left",
        suffixes=("", "_core"),
    )
    gk_map = gk_map.rename(columns={"formulation_core_id": "v2_formulation_core_id"})
    gk_map = gk_map.rename(columns={"reference_normalized_doi": "doi"})
    gk_map["doi"] = gk_map["doi"].astype(str).str.strip().str.lower()

    doe_rows = doe_rows.copy()
    doe_rows["factor_kind"] = doe_rows["factor_kind"].astype(str).str.strip().str.lower()
    doe_rows["factor_name_normalized"] = doe_rows["factor_name_normalized"].astype(str).str.strip()
    doe_rows["group_key"] = doe_rows["group_key"].astype(str).str.strip()
    doe_rows = doe_rows.merge(gk_map[["group_key", "doi"]].drop_duplicates("group_key"), on="group_key", how="left")
    doe_rows["doi"] = doe_rows["doi"].astype(str).str.strip().str.lower()

    doi_plan: dict[str, dict[str, Any]] = {}
    for doi, dg in doe_diag.groupby("doi", sort=True):
        def num_col_max(frame: pd.DataFrame, col: str, default: float = 0.0) -> float:
            if col not in frame.columns:
                return float(default)
            return float(pd.to_numeric(frame[col], errors="coerce").fillna(default).max())

        def any_col_true(frame: pd.DataFrame, col: str) -> bool:
            if col not in frame.columns:
                return False
            vals = pd.to_numeric(frame[col], errors="coerce").fillna(0)
            return bool((vals >= 1).any())

        status_values = sorted(set(dg["status"].tolist()))
        status = "ok" if "ok" in status_values else ("partial" if "partial" in status_values else "failed")
        doi_doe_rows = doe_rows[doe_rows["doi"] == doi].copy()
        gk_vals, varying_keys = choose_doe_key_values(doi_doe_rows)
        has_codebook_table = any_col_true(dg, "has_codebook_table")
        has_runs_table = any_col_true(dg, "has_runs_table")
        coded_columns_count = int(num_col_max(dg, "coded_columns_count", 0.0))
        decoded_rate = float(num_col_max(dg, "decoded_rate", 0.0))
        source_used = "|".join(sorted(set(dg["source_used"].astype(str).tolist()))) if "source_used" in dg.columns else ""
        gate_hard_pass = (
            has_codebook_table
            and has_runs_table
            and coded_columns_count >= 2
            and decoded_rate >= float(min_decoded_rate)
        )
        enabled_soft = status == "ok" or (status == "partial" and len(varying_keys) >= partial_varying_threshold)
        enabled = gate_hard_pass and enabled_soft
        selected_keys = varying_keys[:max_doe_keys] if enabled else []
        doi_plan[doi] = {
            "status": status,
            "enabled": enabled and len(selected_keys) > 0,
            "selected_keys": selected_keys,
            "gk_vals": gk_vals,
            "has_codebook_table": has_codebook_table,
            "has_runs_table": has_runs_table,
            "coded_columns_count": coded_columns_count,
            "decoded_rate": decoded_rate,
            "source_used": source_used,
            "hard_gate_pass": gate_hard_pass,
        }

    split_rows: list[dict[str, str]] = []
    trace_rows: list[dict[str, str]] = []

    # Expand each group_key assignment into v3 signature.
    for _, row in gk_map.sort_values(["doi", "group_key"]).iterrows():
        doi = str(row.get("doi", ""))
        gk = str(row.get("group_key", ""))
        base_sig = str(row.get("core_signature_core", row.get("core_signature", "")))
        plan = doi_plan.get(doi, {"enabled": False, "selected_keys": [], "gk_vals": {}})
        selected_keys = list(plan.get("selected_keys", []))
        gk_vals = plan.get("gk_vals", {})
        selected_vals = gk_vals.get(gk, {})

        if plan.get("enabled", False):
            parts = [f"{k}={selected_vals.get(k, f'MISSING_{k}')}" for k in selected_keys]
            raw = "|".join(parts)
            dh = stable_hash(raw, 12)
            v3_sig = f"{base_sig}||doe_sig={dh}"
            assignment_reason = "split_by_v2_core_plus_doe_signature"
            anchors = doe_rows[
                (doe_rows["group_key"] == gk) & (doe_rows["factor_name_normalized"].isin(selected_keys))
            ]["provenance_anchor"].astype(str).tolist()
        else:
            raw = ""
            dh = ""
            v3_sig = base_sig
            assignment_reason = "carry_v2_core_signature"
            anchors = []

        split_rows.append(
            {
                "group_key": gk,
                "doi": doi,
                "v2_formulation_core_id": str(row.get("v2_formulation_core_id", "")),
                "v2_core_signature": base_sig,
                "v3_core_signature": v3_sig,
                "doe_keys_used": "|".join(selected_keys),
                "doe_signature_hash": dh,
                "doe_signature_raw": raw,
                "doe_source_anchors": "|".join(sorted(set([a for a in anchors if a]))),
                "assignment_reason": assignment_reason,
            }
        )

        trace_rows.append(
            {
                "group_key": gk,
                "formulation_core_id": "",
                "core_signature": v3_sig,
                "included_field_values": str(row.get("included_field_values", "")),
                "missing_fields": str(row.get("missing_fields", "")),
                "condition_tag": str(row.get("condition_tag", "")),
                "matched_keywords": str(row.get("matched_keywords", "")),
                "assignment_reason": assignment_reason,
                "doe_keys_used": "|".join(selected_keys),
                "doe_signature_hash": dh,
                "doe_source_anchors": "|".join(sorted(set([a for a in anchors if a]))),
            }
        )

    split_df = pd.DataFrame(split_rows).sort_values(["doi", "group_key"]).reset_index(drop=True)

    # Build v3 cores from v2 core rows keyed by v3 signature.
    core_join = split_df.merge(
        gk_map[["group_key", "v2_formulation_core_id"]].drop_duplicates("group_key"),
        on=["group_key", "v2_formulation_core_id"],
        how="left",
    )
    core_meta = core_v2.rename(columns={"formulation_core_id": "v2_formulation_core_id"})
    core_join = core_join.merge(core_meta, on="v2_formulation_core_id", how="left")

    group_cols = [
        "doi",
        "v3_core_signature",
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
    ]
    agg_df = (
        core_join.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            example_group_keys=("group_key", lambda s: "|".join(sorted(set(s.tolist()))[:8])),
            doe_keys_used=("doe_keys_used", lambda s: "|".join(sorted(set([x for x in s if x])))),
            doe_signature_hash=("doe_signature_hash", "first"),
        )
        .sort_values(["doi", "v3_core_signature"])
        .reset_index(drop=True)
    )
    agg_df["formulation_core_id"] = [f"FC3_{i+1:05d}" for i in range(len(agg_df))]
    agg_df["created_at"] = core_v2["created_at"].iloc[0] if len(core_v2) > 0 and "created_at" in core_v2.columns else ""
    agg_df["schema_version"] = SCHEMA_VERSION

    core_v3 = agg_df[
        [
            "formulation_core_id",
            "doi",
            "v3_core_signature",
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
    ].rename(
        columns={
            "doi": "reference_normalized_doi",
            "v3_core_signature": "core_signature",
        }
    )

    # Map group_key -> v3 core id.
    core_key = core_v3[["formulation_core_id", "reference_normalized_doi", "core_signature"]].drop_duplicates()
    split_df = split_df.merge(
        core_key,
        left_on=["doi", "v3_core_signature"],
        right_on=["reference_normalized_doi", "core_signature"],
        how="left",
    )

    # Trace output.
    trace_v3 = pd.DataFrame(trace_rows).merge(
        split_df[["group_key", "formulation_core_id"]].drop_duplicates("group_key"),
        on="group_key",
        how="left",
        suffixes=("", "_resolved"),
    )
    trace_v3["formulation_core_id"] = trace_v3["formulation_core_id_resolved"].fillna(trace_v3["formulation_core_id"])
    trace_v3 = trace_v3.drop(columns=["formulation_core_id_resolved"])
    trace_v3 = trace_v3[
        [
            "group_key",
            "formulation_core_id",
            "core_signature",
            "included_field_values",
            "missing_fields",
            "condition_tag",
            "matched_keywords",
            "assignment_reason",
            "doe_keys_used",
            "doe_signature_hash",
            "doe_source_anchors",
        ]
    ].sort_values(["group_key"]).reset_index(drop=True)

    # Measurements re-assigned by group_key.
    meas_v3 = meas_v2.merge(
        split_df[["group_key", "formulation_core_id"]].drop_duplicates("group_key"),
        on="group_key",
        how="left",
        suffixes=("", "_v3"),
    )
    meas_v3["formulation_core_id"] = meas_v3["formulation_core_id_v3"].fillna(meas_v3["formulation_core_id"])
    meas_v3 = meas_v3.drop(columns=["formulation_core_id_v3"]).sort_values(
        ["formulation_core_id", "group_key", "measurement_type"]
    ).reset_index(drop=True)

    # Diff report.
    v2_counts = (
        core_v2.groupby("reference_normalized_doi", dropna=False).size().reset_index(name="v2_core_rows")
    )
    v3_counts = (
        core_v3.groupby("reference_normalized_doi", dropna=False).size().reset_index(name="v3_core_rows")
    )
    diff = v2_counts.merge(v3_counts, on="reference_normalized_doi", how="outer").fillna(0)
    diff["v2_core_rows"] = diff["v2_core_rows"].astype(int)
    diff["v3_core_rows"] = diff["v3_core_rows"].astype(int)
    diff["delta_v3_minus_v2"] = diff["v3_core_rows"] - diff["v2_core_rows"]
    diff = diff.sort_values(["delta_v3_minus_v2", "reference_normalized_doi"], ascending=[False, True]).reset_index(drop=True)

    summary = {
        "v2_core_rows": int(len(core_v2)),
        "v3_core_rows": int(len(core_v3)),
        "v2_measurement_rows": int(len(meas_v2)),
        "v3_measurement_rows": int(len(meas_v3)),
        "doe_enabled_dois": int(
            sum(1 for _, p in doi_plan.items() if bool(p.get("enabled", False)))
        ),
        "focus_10.1002_jps_24101_v2": int(
            diff.loc[diff["reference_normalized_doi"] == "10.1002/jps.24101", "v2_core_rows"].iloc[0]
        ) if (diff["reference_normalized_doi"] == "10.1002/jps.24101").any() else 0,
        "focus_10.1002_jps_24101_v3": int(
            diff.loc[diff["reference_normalized_doi"] == "10.1002/jps.24101", "v3_core_rows"].iloc[0]
        ) if (diff["reference_normalized_doi"] == "10.1002/jps.24101").any() else 0,
        "doe_enabled_dois_list": sorted(
            [doi for doi, p in doi_plan.items() if bool(p.get("enabled", False))]
        ),
        "doe_gate_stats_by_doi": {
            doi: {
                "hard_gate_pass": bool(p.get("hard_gate_pass", False)),
                "enabled": bool(p.get("enabled", False)),
                "status": str(p.get("status", "")),
                "coded_columns_count": int(p.get("coded_columns_count", 0)),
                "decoded_rate": float(p.get("decoded_rate", 0.0)),
                "source_used": str(p.get("source_used", "")),
                "selected_keys": list(p.get("selected_keys", [])),
            }
            for doi, p in sorted(doi_plan.items())
        },
    }
    return core_v3, meas_v3, trace_v3, diff, summary


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    base = Path(f"data/results/{run_id}/benchmark_goren_2025")
    out_dir = Path(args.out_dir) if args.out_dir else base / "schema_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_required(run_id)

    core_v3, meas_v3, trace_v3, diff, summary = build_once(
        core_v2=loaded["core_v2"],
        meas_v2=loaded["meas_v2"],
        trace_v2=loaded["trace_v2"],
        doe_diag=loaded["doe_diag"],
        doe_rows=loaded["doe_rows"],
        partial_varying_threshold=args.partial_varying_threshold,
        max_doe_keys=args.max_doe_keys,
        min_decoded_rate=args.min_decoded_rate,
    )
    core_v3_2, _, _, _, _ = build_once(
        core_v2=loaded["core_v2"],
        meas_v2=loaded["meas_v2"],
        trace_v2=loaded["trace_v2"],
        doe_diag=loaded["doe_diag"],
        doe_rows=loaded["doe_rows"],
        partial_varying_threshold=args.partial_varying_threshold,
        max_doe_keys=args.max_doe_keys,
        min_decoded_rate=args.min_decoded_rate,
    )
    deterministic = (
        core_v3["core_signature"].value_counts().sort_index().equals(
            core_v3_2["core_signature"].value_counts().sort_index()
        )
    )

    p_core = out_dir / "formulation_core.tsv"
    p_meas = out_dir / "measurements.tsv"
    p_trace = out_dir / "core_assignment_trace.tsv"
    p_diff = out_dir / "schema_v2_vs_v3_core_counts.tsv"
    p_summary = out_dir / "schema_v3_summary.json"

    core_v3.to_csv(p_core, sep="\t", index=False)
    meas_v3.to_csv(p_meas, sep="\t", index=False)
    trace_v3.to_csv(p_trace, sep="\t", index=False)
    diff.to_csv(p_diff, sep="\t", index=False)
    p_summary.write_text(
        json.dumps({**summary, "determinism_core_signature_counts": deterministic}, indent=2),
        encoding="utf-8",
    )

    print(f"output_formulation_core={p_core}")
    print(f"output_measurements={p_meas}")
    print(f"output_assignment_trace={p_trace}")
    print(f"output_v2_vs_v3_diff={p_diff}")
    print(f"output_summary={p_summary}")
    print(json.dumps({**summary, "determinism_core_signature_counts": deterministic}, ensure_ascii=False))


if __name__ == "__main__":
    main()
