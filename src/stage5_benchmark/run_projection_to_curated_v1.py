#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
DEFAULT_PROJECTION_RULESET = "data/benchmark/goren_2025/rules/projection_ruleset.v1.json"
DEFAULT_CURATED_TEMPLATE = "data/benchmark/goren_2025/NP_dataset_formulations.csv"
DEFAULT_SAMPLE_MANIFEST = "data/cleaned/samples/sample_goren18.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy or branch-only modeling helper: project derived formulation values into curated schema without modifying extraction outputs."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--derived-values-tsv",
        default="",
        help="Defaults to data/results/<run_id>/benchmark_goren_2025/derived_values.tsv",
    )
    parser.add_argument("--curated-template", default=DEFAULT_CURATED_TEMPLATE)
    parser.add_argument("--sample-manifest", default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--projection-ruleset", default=DEFAULT_PROJECTION_RULESET)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Defaults to data/results/<run_id>/benchmark_goren_2025",
    )
    return parser.parse_args()


def normalize_doi(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"^doi\s*:\s*", "", text)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\.org/", "", text)
    text = text.strip()
    return text


def pick_first(values: list[str]) -> str:
    for v in values:
        if str(v).strip() != "":
            return str(v).strip()
    return ""


def select_value_with_trace(field_rows: pd.DataFrame) -> dict[str, str]:
    if field_rows.empty:
        return {
            "value": "",
            "rule_id": "",
            "derived_from": "",
            "value_source": "",
            "trace_pointer": "",
        }
    # Prefer direct anchors, then parsed values, then derived math.
    priority = {
        "extracted_anchor": 0,
        "parsed_from_extracted": 1,
        "parsed_evidence_span": 2,
        "derived_math": 3,
    }
    rows = field_rows.copy()
    rows["priority"] = rows["value_source"].map(lambda x: priority.get(str(x), 9))
    rows = rows.sort_values(["priority"]).reset_index(drop=True)
    chosen = rows.iloc[0]
    return {
        "value": str(chosen.get("value", "") or ""),
        "rule_id": str(chosen.get("rule_id", "") or ""),
        "derived_from": str(chosen.get("derived_from", "") or ""),
        "value_source": str(chosen.get("value_source", "") or ""),
        "trace_pointer": str(chosen.get("trace_pointer", "") or ""),
    }


def choose_polymer_mw(group_rows: pd.DataFrame) -> dict[str, str]:
    low = select_value_with_trace(group_rows[group_rows["field_name"] == "polymer_mw_lower_kDa"])
    high = select_value_with_trace(group_rows[group_rows["field_name"] == "polymer_mw_upper_kDa"])
    if not low["value"] and not high["value"]:
        return {
            "value": "",
            "rule_id": "",
            "derived_from": "",
            "value_source": "",
            "trace_pointer": "",
        }
    if low["value"] and high["value"]:
        if low["value"] == high["value"]:
            return {
                "value": low["value"],
                "rule_id": high["rule_id"] or low["rule_id"],
                "derived_from": "polymer_mw_lower_kDa,polymer_mw_upper_kDa",
                "value_source": "projection_compose",
                "trace_pointer": high["trace_pointer"] or low["trace_pointer"],
            }
        return {
            "value": f"{low['value']}-{high['value']}",
            "rule_id": "P_POLYMER_MW_RANGE_TO_TEXT",
            "derived_from": "polymer_mw_lower_kDa,polymer_mw_upper_kDa",
            "value_source": "projection_compose",
            "trace_pointer": pick_first([high["trace_pointer"], low["trace_pointer"]]),
        }
    only = low if low["value"] else high
    return only


def choose_aqueous_organic(group_rows: pd.DataFrame) -> dict[str, str]:
    w1o = select_value_with_trace(group_rows[group_rows["field_name"] == "w1_over_o_ratio"])
    wtot = select_value_with_trace(group_rows[group_rows["field_name"] == "w1w2_over_o_ratio"])
    # Never mix definitions; only project when exactly one definition exists.
    has_w1o = w1o["value"] != ""
    has_wtot = wtot["value"] != ""
    if has_w1o and has_wtot:
        return {
            "value": "",
            "rule_id": "P_AQORG_AMBIGUOUS_BOTH_PRESENT",
            "derived_from": "w1_over_o_ratio,w1w2_over_o_ratio",
            "value_source": "projection_null_on_ambiguity",
            "trace_pointer": pick_first([wtot["trace_pointer"], w1o["trace_pointer"]]),
        }
    if has_wtot:
        return {
            "value": wtot["value"],
            "rule_id": "P_AQORG_USE_W1W2_OVER_O",
            "derived_from": "w1w2_over_o_ratio",
            "value_source": wtot["value_source"],
            "trace_pointer": wtot["trace_pointer"],
        }
    if has_w1o:
        return {
            "value": w1o["value"],
            "rule_id": "P_AQORG_USE_W1_OVER_O",
            "derived_from": "w1_over_o_ratio",
            "value_source": w1o["value_source"],
            "trace_pointer": w1o["trace_pointer"],
        }
    return {
        "value": "",
        "rule_id": "",
        "derived_from": "",
        "value_source": "",
        "trace_pointer": "",
    }


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/{run_id}/benchmark_goren_2025")
    derived_values_tsv = (
        Path(args.derived_values_tsv)
        if args.derived_values_tsv
        else out_dir / "derived_values.tsv"
    )
    curated_template_path = Path(args.curated_template)
    sample_manifest_path = Path(args.sample_manifest)
    projection_ruleset = Path(args.projection_ruleset)

    if not derived_values_tsv.exists():
        raise FileNotFoundError(f"Derived values TSV not found: {derived_values_tsv}")
    if not curated_template_path.exists():
        raise FileNotFoundError(f"Curated template not found: {curated_template_path}")
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"Sample manifest not found: {sample_manifest_path}")
    if not projection_ruleset.exists():
        raise FileNotFoundError(f"Projection ruleset not found: {projection_ruleset}")

    out_dir.mkdir(parents=True, exist_ok=True)

    derived = pd.read_csv(derived_values_tsv, sep="\t", dtype=str).fillna("")
    curated_template = pd.read_csv(curated_template_path, dtype=str).fillna("")
    manifest = pd.read_csv(sample_manifest_path, sep="\t", dtype=str).fillna("")
    key_to_doi = dict(zip(manifest["key"], manifest["doi"].map(normalize_doi)))

    curated_columns = curated_template.columns.tolist()
    projection_rows: list[dict[str, str]] = []
    trace_rows: list[dict[str, str]] = []

    grouped = derived.groupby(["group_key", "key", "formulation_id"], dropna=False)
    for (group_key, key, formulation_id), group_rows in grouped:
        projected = {c: "" for c in curated_columns}

        doi_value = key_to_doi.get(str(key), "")
        if "reference" in projected:
            projected["reference"] = doi_value
        if "doi_norm" in projected:
            projected["doi_norm"] = doi_value

        # Field mapping to curated columns.
        mapped = {
            "small_molecule_name": select_value_with_trace(group_rows[group_rows["field_name"] == "small_molecule_name"]),
            "solvent": select_value_with_trace(group_rows[group_rows["field_name"] == "solvent"]),
            "surfactant_concentration": select_value_with_trace(group_rows[group_rows["field_name"] == "surfactant_concentration"]),
            "particle_size": select_value_with_trace(group_rows[group_rows["field_name"] == "particle_size"]),
            "EE": select_value_with_trace(group_rows[group_rows["field_name"] == "EE"]),
            "LC": select_value_with_trace(group_rows[group_rows["field_name"] == "LC"]),
            "LA/GA": select_value_with_trace(group_rows[group_rows["field_name"] == "LA/GA"]),
            "drug/polymer": select_value_with_trace(group_rows[group_rows["field_name"] == "drug/polymer"]),
            "polymer_MW": choose_polymer_mw(group_rows),
            "aqueous/organic": choose_aqueous_organic(group_rows),
            "surfactant_name": {"value": "", "rule_id": "", "derived_from": "", "value_source": "", "trace_pointer": ""},
            "pH": {"value": "", "rule_id": "", "derived_from": "", "value_source": "", "trace_pointer": ""},
        }

        for curated_col, info in mapped.items():
            if curated_col not in projected:
                continue
            projected[curated_col] = str(info["value"])
            trace_rows.append(
                {
                    "run_id": run_id,
                    "group_key": str(group_key),
                    "key": str(key),
                    "formulation_id": str(formulation_id),
                    "curated_column": curated_col,
                    "projected_value": str(info["value"]),
                    "rule_id": str(info["rule_id"]),
                    "derived_from": str(info["derived_from"]),
                    "value_source": str(info["value_source"]),
                    "trace_pointer": str(info["trace_pointer"]),
                }
            )

        # Keep non-mapped curated columns as null unless explicitly handled.
        projection_rows.append(projected)

    projected_df = pd.DataFrame(projection_rows)
    if projected_df.empty:
        projected_df = pd.DataFrame(columns=curated_columns)
    else:
        projected_df = projected_df[curated_columns]

    projected_out = out_dir / "projected_to_curated.tsv"
    projected_df.to_csv(projected_out, sep="\t", index=False)

    trace_df = pd.DataFrame(trace_rows)
    if trace_df.empty:
        trace_df = pd.DataFrame(
            columns=[
                "run_id",
                "group_key",
                "key",
                "formulation_id",
                "curated_column",
                "projected_value",
                "rule_id",
                "derived_from",
                "value_source",
                "trace_pointer",
            ]
        )
    trace_out = out_dir / "projection_trace.tsv"
    trace_df.to_csv(trace_out, sep="\t", index=False)

    summary = {
        "run_id": run_id,
        "derived_values_tsv": str(derived_values_tsv),
        "curated_template": str(curated_template_path),
        "sample_manifest": str(sample_manifest_path),
        "projection_ruleset": str(projection_ruleset),
        "projected_rows": int(len(projected_df)),
        "projected_columns": curated_columns,
        "output_projected_tsv": str(projected_out),
        "output_trace_tsv": str(trace_out),
    }
    (out_dir / "projection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
