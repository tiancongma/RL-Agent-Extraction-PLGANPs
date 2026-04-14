#!/usr/bin/env python3
"""
Build the first row-wise Stage5 modeling-ready table.

Contract:
- consumes only the frozen `final_formulation_table_v1.tsv` plus
  `modeling_ready_values_v1.tsv`
- emits one row per frozen `final_formulation_id`
- preserves selected raw benchmark-final fields unchanged
- pivots selected transformed modeling values into explicit columns
- preserves row-level provenance summaries
- does not change formulation membership or benchmark-final semantics
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

try:
    from src.utils.active_data_source import (
        build_artifact_metadata,
        resolve_artifact_path,
        resolve_run_context,
        write_artifact_metadata_json,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.active_data_source import (
        build_artifact_metadata,
        resolve_artifact_path,
        resolve_run_context,
        write_artifact_metadata_json,
    )


OUTPUT_NAME = "modeling_ready_table_v1.tsv"
SUMMARY_NAME = "modeling_ready_table_summary_v1.json"
SIDECAR_RELATIVE = "modeling_ready_v1/modeling_ready_values_v1.tsv"

RAW_FIELD_SPECS = [
    ("drug_name_raw", ["drug_name_value_text", "drug_name_value"]),
    ("organic_solvent_raw", ["organic_solvent_value_text", "organic_solvent_value"]),
    ("surfactant_name_raw", ["surfactant_name_value_text", "surfactant_name_value"]),
    (
        "surfactant_concentration_text_raw",
        ["surfactant_concentration_text_value_text", "surfactant_concentration_text_value"],
    ),
    ("pva_conc_percent_raw", ["pva_conc_percent_value_text", "pva_conc_percent_value"]),
    ("la_ga_ratio_raw", ["la_ga_ratio_value_text", "la_ga_ratio_value"]),
    ("polymer_mw_kDa_raw", ["polymer_mw_kDa_value_text", "polymer_mw_kDa_value"]),
    ("plga_mass_mg_raw", ["plga_mass_mg_value_text", "plga_mass_mg_value"]),
    ("drug_feed_amount_text_raw", ["drug_feed_amount_text_value_text", "drug_feed_amount_text_value"]),
    ("size_nm_raw", ["size_nm_value_text", "size_nm_value"]),
    ("pdi_raw", ["pdi_value_text", "pdi_value"]),
    ("zeta_mV_raw", ["zeta_mV_value_text", "zeta_mV_value"]),
    (
        "encapsulation_efficiency_percent_raw",
        ["encapsulation_efficiency_percent_value_text", "encapsulation_efficiency_percent_value"],
    ),
    ("loading_content_percent_raw", ["loading_content_percent_value_text", "loading_content_percent_value"]),
    ("emul_method_raw", ["emul_method_value_text", "emul_method_value"]),
]

MODELING_FIELD_ORDER = [
    "la_fraction",
    "ga_fraction",
    "la_over_ga_ratio",
    "polymer_mw_lower_kDa",
    "polymer_mw_upper_kDa",
    "polymer_mass_mg",
    "drug_mass_mg",
    "drug_to_polymer_mass_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-table-tsv", default="", help="Path to frozen final_formulation_table_v1.tsv.")
    parser.add_argument("--modeling-values-tsv", default="", help="Path to modeling_ready_values_v1.tsv.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional authoritative run directory.")
    parser.add_argument("--run-id", default="", help="Compatibility alias for selecting an explicit run root by run_id.")
    parser.add_argument("--out-dir", default="", help="Optional output directory. Defaults to <run_dir>/modeling_ready_v1.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def resolve_modeling_values_path(
    *,
    explicit_path: str,
    run_context: dict[str, Any],
) -> Path:
    if explicit_path.strip():
        resolved = Path(explicit_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Explicit modeling values TSV not found: {resolved}")
        return resolved
    return resolve_artifact_path(
        explicit_path=None,
        run_context=run_context,
        pointer_key="stage5_modeling_ready_values_tsv",
        canonical_relative=SIDECAR_RELATIVE,
    )


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: normalize_text(v) for k, v in row.items()})


def first_nonblank(row: dict[str, str], candidates: list[str]) -> str:
    for name in candidates:
        value = normalize_text(row.get(name, ""))
        if value:
            return value
    return ""


def load_sidecar_by_final_id(sidecar_rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in sidecar_rows:
        final_id = normalize_text(row.get("final_formulation_id"))
        field_name = normalize_text(row.get("modeling_field_name"))
        if not final_id or not field_name:
            continue
        if field_name in grouped[final_id]:
            raise ValueError(
                f"Duplicate modeling sidecar entry for final_formulation_id={final_id!r} field={field_name!r}."
            )
        grouped[final_id][field_name] = row
    return grouped


def build_row(
    *,
    run_id: str,
    final_row: dict[str, str],
    sidecar_fields: dict[str, dict[str, str]],
) -> dict[str, str]:
    row = {
        "run_id": run_id,
        "key": normalize_text(final_row.get("key")),
        "doi": normalize_text(final_row.get("doi")),
        "final_formulation_id": normalize_text(final_row.get("final_formulation_id")),
        "representative_source_formulation_id": normalize_text(final_row.get("representative_source_formulation_id")),
        "representative_source_raw_formulation_label": normalize_text(final_row.get("representative_source_raw_formulation_label")),
        "family_id": normalize_text(final_row.get("family_id")),
        "parent_core_row_id": normalize_text(final_row.get("parent_core_row_id")),
        "variant_role": normalize_text(final_row.get("variant_role")),
        "payload_state": normalize_text(final_row.get("payload_state")),
        "benchmark_default_include": normalize_text(final_row.get("benchmark_default_include")),
        "benchmark_field_source_type": normalize_text(final_row.get("field_source_type")),
        "benchmark_final_output_rule": normalize_text(final_row.get("final_output_rule")),
        "polymer_identity_final_raw": normalize_text(final_row.get("polymer_identity_final")),
        "loaded_state_final_raw": normalize_text(final_row.get("loaded_state_final")),
        "source_candidate_count": normalize_text(final_row.get("source_candidate_count")),
        "collapsed_variant_count": normalize_text(final_row.get("collapsed_variant_count")),
    }

    for output_name, candidates in RAW_FIELD_SPECS:
        row[output_name] = first_nonblank(final_row, candidates)

    available_fields: list[str] = []
    missing_fields: list[str] = []
    rule_ids: list[str] = []
    rule_families: list[str] = []
    for field_name in MODELING_FIELD_ORDER:
        payload = sidecar_fields.get(field_name)
        if payload is None:
            row[field_name] = ""
            missing_fields.append(field_name)
            continue
        row[field_name] = normalize_text(payload.get("modeling_value"))
        available_fields.append(field_name)
        rule_id = normalize_text(payload.get("transform_rule_id"))
        rule_family = normalize_text(payload.get("transform_rule_family"))
        if rule_id:
            rule_ids.append(rule_id)
        if rule_family:
            rule_families.append(rule_family)

    row["modeling_transform_count"] = str(len(available_fields))
    row["modeling_available_fields_csv"] = ",".join(available_fields)
    row["modeling_missing_fields_csv"] = ",".join(missing_fields)
    row["modeling_rule_ids_csv"] = ",".join(sorted(set(rule_ids)))
    row["modeling_rule_families_csv"] = ",".join(sorted(set(rule_families)))
    return row


def build_table_rows(
    *,
    run_id: str,
    final_rows: list[dict[str, str]],
    sidecar_by_final_id: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, str]]:
    out_rows: list[dict[str, str]] = []
    seen_final_ids: set[str] = set()
    for final_row in final_rows:
        final_id = normalize_text(final_row.get("final_formulation_id"))
        if not final_id:
            raise ValueError("Frozen final table row is missing final_formulation_id.")
        if final_id in seen_final_ids:
            raise ValueError(f"Duplicate final_formulation_id in frozen final table: {final_id}")
        seen_final_ids.add(final_id)
        sidecar_fields = sidecar_by_final_id.get(final_id, {})
        out_rows.append(build_row(run_id=run_id, final_row=final_row, sidecar_fields=sidecar_fields))
    return out_rows


def main() -> None:
    args = parse_args()
    run_context = resolve_run_context(
        explicit_run_dir=args.run_dir,
        explicit_run_id=args.run_id,
    )
    final_table_path = resolve_artifact_path(
        explicit_path=Path(args.final_table_tsv) if args.final_table_tsv else None,
        run_context=run_context,
        pointer_key="stage5_final_table_tsv",
        canonical_relative="final_formulation_table_v1.tsv",
    )
    modeling_values_path = resolve_modeling_values_path(
        explicit_path=args.modeling_values_tsv,
        run_context=run_context,
    )
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (Path(run_context["run_dir"]).resolve() / "modeling_ready_v1")

    print(
        json.dumps(
            {
                "resolved_source_run_dir": str(run_context["run_dir"]),
                "resolved_source_run_id": str(run_context["run_id"]),
                "source_resolution": str(run_context["resolution_source"]),
                "active_run_pointer_path": str(run_context.get("pointer_path") or ""),
                "resolved_input_files": {
                    "final_table_tsv": str(final_table_path),
                    "modeling_values_tsv": str(modeling_values_path),
                },
                "resolved_out_dir": str(out_dir),
            },
            ensure_ascii=True,
            indent=2,
        )
    )

    final_rows = read_tsv(final_table_path)
    sidecar_rows = read_tsv(modeling_values_path)
    sidecar_by_final_id = load_sidecar_by_final_id(sidecar_rows)
    table_rows = build_table_rows(
        run_id=str(run_context["run_id"]),
        final_rows=final_rows,
        sidecar_by_final_id=sidecar_by_final_id,
    )

    fieldnames = [
        "run_id",
        "key",
        "doi",
        "final_formulation_id",
        "representative_source_formulation_id",
        "representative_source_raw_formulation_label",
        "family_id",
        "parent_core_row_id",
        "variant_role",
        "payload_state",
        "benchmark_default_include",
        "benchmark_field_source_type",
        "benchmark_final_output_rule",
        "polymer_identity_final_raw",
        "loaded_state_final_raw",
        "source_candidate_count",
        "collapsed_variant_count",
        *[name for name, _ in RAW_FIELD_SPECS],
        *MODELING_FIELD_ORDER,
        "modeling_transform_count",
        "modeling_available_fields_csv",
        "modeling_missing_fields_csv",
        "modeling_rule_ids_csv",
        "modeling_rule_families_csv",
    ]

    out_path = out_dir / OUTPUT_NAME
    write_tsv(out_path, fieldnames, table_rows)

    transform_counts = Counter(row["modeling_transform_count"] for row in table_rows)
    summary = {
        "run_id": str(run_context["run_id"]),
        "final_table_tsv": str(final_table_path),
        "modeling_values_tsv": str(modeling_values_path),
        "row_count": int(len(table_rows)),
        "distinct_final_formulation_count": int(len({row["final_formulation_id"] for row in table_rows})),
        "modeled_column_count": int(len(MODELING_FIELD_ORDER)),
        "raw_carrythrough_column_count": int(len(RAW_FIELD_SPECS)),
        "transform_count_distribution": dict(sorted(transform_counts.items())),
        "modeled_fields": MODELING_FIELD_ORDER,
        "output_tsv": str(out_path),
    }
    summary_path = out_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    metadata_path = write_artifact_metadata_json(
        out_path,
        build_artifact_metadata(
            source_run_context=run_context,
            source_files={
                "final_table_tsv": str(final_table_path),
                "modeling_values_tsv": str(modeling_values_path),
            },
            generated_by="src/stage5_benchmark/build_modeling_ready_table_v1.py",
            note="First row-wise modeling-ready table built from frozen final rows plus modeling_ready_values_v1 sidecar values.",
            extra={
                "summary_json": str(summary_path),
                "output_family": "stage5_modeling_ready_rowwise_table",
            },
        ),
    )

    print(
        json.dumps(
            {
                "output_tsv": str(out_path),
                "summary_json": str(summary_path),
                "metadata_json": str(metadata_path),
                "row_count": int(len(table_rows)),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
