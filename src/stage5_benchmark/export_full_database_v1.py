#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
DEFAULT_SOURCE_SCHEMA = "schema_v2"
DEFAULT_DB_VERSION = "db_v1"
DEFAULT_OUT_DIR = Path("data/db/db_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stable full database snapshot from schema outputs.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-schema", default=DEFAULT_SOURCE_SCHEMA)
    parser.add_argument("--db-version", default=DEFAULT_DB_VERSION)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_git_short_hash_safe() -> str:
    src_dir = Path("src").resolve()
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))
    try:
        from utils.run_id import get_git_short_hash  # type: ignore

        return str(get_git_short_hash(Path(".")))
    except Exception:
        return "nogit"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def parse_numeric(value: str) -> str:
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return f"{f:.12g}"
    except ValueError:
        return ""


def truncate_text(value: Any, limit: int = 200) -> str:
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:limit]


def build_factor_candidates(core_df: pd.DataFrame) -> tuple[list[tuple[str, str]], list[str]]:
    allowlist_aliases: dict[str, list[str]] = {
        "pva_conc_percent": ["pva_conc_percent"],
        "aqueous_phase_pH": ["aqueous_phase_pH"],
        "plga_mass_mg": ["plga_mass_mg"],
        "drug_to_polymer_mass_ratio": ["drug_to_polymer_mass_ratio"],
        "w1_volume_uL": ["w1_volume_uL"],
        "w2_volume_mL": ["w2_volume_mL"],
        "o_volume_mL": ["o_volume_mL"],
        "aqueous_organic_ratio": ["aqueous_organic_ratio"],
        "emul_method": ["emul_method"],
        "emul_type": ["emul_type"],
        "surfactant_type": ["surfactant_type", "surfactant_type_normalized"],
        "surfactant_conc": ["surfactant_conc", "surfactant_conc_percent"],
        "organic_solvent": ["organic_solvent", "organic_solvent_normalized"],
    }

    candidates: list[tuple[str, str]] = []
    used_columns: set[str] = set()
    for factor_name, aliases in allowlist_aliases.items():
        for col in aliases:
            if col in core_df.columns:
                candidates.append((factor_name, col))
                used_columns.add(col)
                break

    core_identity_columns = {
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
    }
    audit_columns = {
        "example_group_keys",
        "created_at",
        "schema_version",
    }

    heuristic_cols = []
    for col in core_df.columns:
        if col in core_identity_columns or col in audit_columns or col in used_columns:
            continue
        if col in {"reference", "doi", "doi_norm"}:
            continue
        heuristic_cols.append(col)

    if "reference_normalized_doi" in core_df.columns:
        for col in heuristic_cols:
            sub = core_df[["reference_normalized_doi", col]].copy()
            sub = sub[sub[col].astype(str).str.strip() != ""]
            if sub.empty:
                continue
            var_by_doi = sub.groupby("reference_normalized_doi")[col].nunique(dropna=True)
            if (var_by_doi > 1).any():
                candidates.append((col, col))

    return candidates, sorted(set([c[0] for c in candidates]))


def build_factors(
    core_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
    core_trace_df: pd.DataFrame,
    run_id: str,
) -> tuple[pd.DataFrame, list[str]]:
    required_cols = [
        "formulation_core_id",
        "factor_name",
        "factor_value_text",
        "factor_value_num",
        "factor_unit",
        "value_source",
        "evidence_excerpt",
        "trace_pointer",
        "factor_kind",
    ]
    if core_df.empty:
        return pd.DataFrame(columns=required_cols), []

    candidates, factor_names = build_factor_candidates(core_df)
    unit_map = {
        "pva_conc_percent": "percent",
        "surfactant_conc": "percent",
        "plga_mass_mg": "mg",
        "aqueous_phase_pH": "pH",
        "drug_to_polymer_mass_ratio": "ratio",
        "w1_volume_uL": "uL",
        "w2_volume_mL": "mL",
        "o_volume_mL": "mL",
        "aqueous_organic_ratio": "ratio",
    }

    evidence_map = (
        measurements_df.groupby("formulation_core_id", as_index=False)
        .first()[["formulation_core_id", "evidence_excerpt", "trace_pointer"]]
        if not measurements_df.empty and "formulation_core_id" in measurements_df.columns
        else pd.DataFrame(columns=["formulation_core_id", "evidence_excerpt", "trace_pointer"])
    )
    evidence_by_core = {
        str(r["formulation_core_id"]): (str(r.get("evidence_excerpt", "")), str(r.get("trace_pointer", "")))
        for _, r in evidence_map.iterrows()
    }

    rows: list[dict[str, str]] = []
    for _, row in core_df.iterrows():
        core_id = str(row.get("formulation_core_id", ""))
        ev, tp = evidence_by_core.get(core_id, ("", ""))
        for factor_name, col in candidates:
            raw = str(row.get(col, "")).strip()
            if not raw:
                continue
            if raw.startswith("MISSING_"):
                continue
            rows.append(
                {
                    "formulation_core_id": core_id,
                    "factor_name": factor_name,
                    "factor_value_text": raw,
                    "factor_value_num": parse_numeric(raw),
                    "factor_unit": unit_map.get(factor_name, ""),
                    "value_source": "schema_v2_allowlist_column" if factor_name != col else "schema_v2_heuristic_column",
                    "evidence_excerpt": truncate_text(ev, 200),
                    "trace_pointer": tp,
                    "factor_kind": "other",
                }
            )

    factors_df = pd.DataFrame(rows, columns=required_cols)

    # Append DOE coded/decoded factors derived in derivation_v1, if present.
    doe_path = Path(f"data/results/{run_id}/benchmark_goren_2025/derivation_v1/doe_factor_rows.tsv")
    if doe_path.exists() and not core_trace_df.empty:
        doe_df = read_tsv(doe_path)
        trace_map = core_trace_df[["group_key", "formulation_core_id"]].drop_duplicates("group_key")
        doe_df = doe_df.merge(trace_map, on="group_key", how="left")
        doe_df = doe_df[doe_df["formulation_core_id"].astype(str).str.strip() != ""].copy()
        if not doe_df.empty:
            doe_append = pd.DataFrame(
                {
                    "formulation_core_id": doe_df["formulation_core_id"].astype(str),
                    "factor_name": doe_df["factor_name_original"].astype(str),
                    "factor_value_text": doe_df["factor_value_text"].astype(str),
                    "factor_value_num": doe_df["factor_value_num"].astype(str),
                    "factor_unit": doe_df["factor_unit"].astype(str),
                    "value_source": doe_df["value_source"].astype(str),
                    "evidence_excerpt": doe_df["provenance_anchor"].astype(str),
                    "trace_pointer": doe_df["trace_pointer"].astype(str),
                    "factor_kind": doe_df["factor_kind"].astype(str).replace("", "other"),
                }
            )
            factors_df = pd.concat([factors_df, doe_append], ignore_index=True)
            factor_names = sorted(set(factor_names) | set(doe_append["factor_name"].tolist()))

    return factors_df, factor_names


def main() -> None:
    args = parse_args()

    run_id = args.run_id
    source_schema = args.source_schema
    db_version = args.db_version
    out_dir = Path(args.out_dir)

    source_dir = Path(f"data/results/{run_id}/benchmark_goren_2025/{source_schema}")
    core_path = source_dir / "formulation_core.tsv"
    measurements_path = source_dir / "measurements.tsv"
    trace_path = source_dir / "core_assignment_trace.tsv"
    doe_diag_path = Path(f"data/results/{run_id}/benchmark_goren_2025/derivation_v1/doe_decode_diagnostics.tsv")

    required_inputs = [core_path, measurements_path, trace_path]
    missing = [str(p) for p in required_inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    previous_counts: dict[str, int] = {}
    existing_outputs = {
        "formulation_core": out_dir / "formulation_core.tsv",
        "measurements": out_dir / "measurements.tsv",
        "factors": out_dir / "factors.tsv",
    }
    if out_dir.exists():
        if not args.overwrite and any(p.exists() for p in existing_outputs.values()):
            raise FileExistsError(f"Output exists under {out_dir}; rerun with --overwrite.")
        if args.overwrite and all(p.exists() for p in existing_outputs.values()):
            previous_counts = {
                "formulation_core": len(read_tsv(existing_outputs["formulation_core"])),
                "measurements": len(read_tsv(existing_outputs["measurements"])),
                "factors": len(read_tsv(existing_outputs["factors"])),
            }

    core_df = read_tsv(core_path)
    measurements_df = read_tsv(measurements_path)
    trace_df = read_tsv(trace_path)
    factors_df, factor_names = build_factors(core_df, measurements_df, trace_df, run_id)

    out_dir.mkdir(parents=True, exist_ok=True)

    core_out = out_dir / "formulation_core.tsv"
    meas_out = out_dir / "measurements.tsv"
    factors_out = out_dir / "factors.tsv"
    manifest_out = out_dir / "schema_manifest.json"

    core_df.to_csv(core_out, sep="\t", index=False)
    measurements_df.to_csv(meas_out, sep="\t", index=False)
    factors_df.to_csv(factors_out, sep="\t", index=False)

    warnings: list[str] = []
    if factors_df.empty:
        warnings.append("factors table is empty")
    if "reference_normalized_doi" in core_df.columns:
        missing_doi_count = int((core_df["reference_normalized_doi"].astype(str).str.strip() == "").sum())
        if missing_doi_count > 0:
            warnings.append(f"missing reference_normalized_doi rows: {missing_doi_count}")
    else:
        warnings.append("reference_normalized_doi column missing in formulation_core")
    if "drug_name_normalized" in core_df.columns:
        missing_drug_count = int((core_df["drug_name_normalized"].astype(str).str.strip() == "").sum())
        if missing_drug_count > 0:
            warnings.append(f"missing drug_name_normalized rows: {missing_drug_count}")
    else:
        warnings.append("drug_name_normalized column missing in formulation_core")

    condition_dist = (
        measurements_df["condition_tag"].value_counts().to_dict()
        if "condition_tag" in measurements_df.columns
        else {}
    )
    unique_dois_count = (
        int(core_df["reference_normalized_doi"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
        if "reference_normalized_doi" in core_df.columns
        else 0
    )

    previous_created_at = ""
    if manifest_out.exists():
        try:
            old_manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
            previous_created_at = str(old_manifest.get("created_at", ""))
        except Exception:
            previous_created_at = ""

    factor_kind_counts = (
        factors_df["factor_kind"].astype(str).str.strip().replace("", "other").value_counts().to_dict()
        if "factor_kind" in factors_df.columns and not factors_df.empty
        else {"other": int(len(factors_df))}
    )

    manifest = {
        "db_version": db_version,
        "source_run_id": run_id,
        "source_schema": source_schema,
        "created_at": previous_created_at or datetime.now(timezone.utc).isoformat(),
        "git_short_hash": get_git_short_hash_safe(),
        "row_counts": {
            "formulation_core": int(len(core_df)),
            "measurements": int(len(measurements_df)),
            "factors": int(len(factors_df)),
        },
        "unique_dois_count": unique_dois_count,
        "condition_tag_distribution": condition_dist,
        "core_columns_exported": core_df.columns.tolist(),
        "factor_names_emitted": sorted(set(factor_names)),
        "doe_decode_enabled": True,
        "doe_decode_diagnostics_path": str(doe_diag_path).replace("\\", "/"),
        "factor_kind_counts": factor_kind_counts,
        "note": "DOE factors derived via source re-read (html/txt) in derivation_v1, not directly from Stage2 spans",
        "warnings": warnings,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    current_counts = manifest["row_counts"]
    determinism_same_counts = (
        previous_counts == current_counts if previous_counts else None
    )

    top_factors = []
    if not factors_df.empty:
        factor_counter = Counter(factors_df["factor_name"].tolist())
        top_factors = factor_counter.most_common(15)

    print(f"output_formulation_core={core_out}")
    print(f"output_measurements={meas_out}")
    print(f"output_factors={factors_out}")
    print(f"output_manifest={manifest_out}")
    print(f"row_counts={json.dumps(current_counts, ensure_ascii=False)}")
    print(f"unique_dois_count={unique_dois_count}")
    print(f"condition_tag_distribution={json.dumps(condition_dist, ensure_ascii=False)}")
    print(f"top15_factor_names={json.dumps(top_factors, ensure_ascii=False)}")
    if determinism_same_counts is None:
        print("determinism_row_count_check_overwrite=not_applicable_no_previous_snapshot")
    else:
        print(f"determinism_row_count_check_overwrite={determinism_same_counts}")


if __name__ == "__main__":
    main()
