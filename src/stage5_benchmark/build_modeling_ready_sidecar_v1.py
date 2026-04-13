#!/usr/bin/env python3
"""
Build the first true Stage5 modeling-ready sidecar from a frozen benchmark-final table.

Contract:
- downstream of `final_formulation_table_v1.tsv`
- preserves frozen benchmark-final row identity and raw source-faithful values
- emits only explicit deterministic parse/math transforms
- does not change formulation membership
- does not replace or redefine the benchmark-final table

This is a downstream Stage5 helper, not a new pipeline stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


OUTPUT_NAME = "modeling_ready_values_v1.tsv"
SUMMARY_NAME = "modeling_ready_summary_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-table-tsv",
        default="",
        help="Path to the frozen Stage5 final_formulation_table_v1.tsv.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional authoritative run directory. Overrides ACTIVE_RUN.json.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Compatibility alias for selecting an explicit run root by run_id.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory. Defaults to <run_dir>/modeling_ready_v1.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def parse_float(raw: Any) -> float | None:
    text = normalize_text(raw).replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    import re

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_mass_to_mg(raw: Any) -> float | None:
    text = normalize_text(raw).lower()
    if not text:
        return None
    import re

    text = text.replace("渭", "u")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(mg|g|ug|mcg|ng)\b", text)
    if not match:
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
    text = normalize_text(raw).replace(" ", "")
    if not text:
        return (None, None, None)
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
            return (la / total, ga / total, (la / ga) if ga != 0 else None)
    scalar = parse_float(text)
    return (None, None, scalar)


def parse_mw_range(raw: Any) -> tuple[float | None, float | None]:
    text = normalize_text(raw).replace(",", "")
    if not text:
        return (None, None)
    import re

    vals = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not vals:
        return (None, None)
    if len(vals) == 1:
        value = float(vals[0])
        return (value, value)
    lo = float(vals[0])
    hi = float(vals[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def get_final_field_text(row: dict[str, str], field_name: str) -> str:
    candidates = [
        f"{field_name}_value_text",
        f"{field_name}_value",
        field_name,
    ]
    if field_name == "polymer_mw_kDa":
        candidates.extend(["plga_mw_kDa_value_text", "plga_mw_kDa_value", "plga_mw_kDa"])
    for candidate in candidates:
        value = normalize_text(row.get(candidate, ""))
        if value:
            return value
    return ""


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return normalize_text(value)


def append_modeling_row(
    out_rows: list[dict[str, str]],
    *,
    run_id: str,
    row: dict[str, str],
    modeling_field_name: str,
    modeling_value: Any,
    transform_rule_id: str,
    transform_rule_family: str,
    source_field_names: list[str],
) -> None:
    value_text = format_value(modeling_value)
    if not value_text:
        return
    source_values = {name: normalize_text(row.get(name, "")) for name in source_field_names}
    out_rows.append(
        {
            "run_id": run_id,
            "key": normalize_text(row.get("key")),
            "doi": normalize_text(row.get("doi")),
            "final_formulation_id": normalize_text(row.get("final_formulation_id")),
            "representative_source_formulation_id": normalize_text(row.get("representative_source_formulation_id")),
            "benchmark_field_source_type": normalize_text(row.get("field_source_type")),
            "benchmark_final_output_rule": normalize_text(row.get("final_output_rule")),
            "modeling_field_name": modeling_field_name,
            "modeling_value": value_text,
            "transform_rule_id": transform_rule_id,
            "transform_rule_family": transform_rule_family,
            "source_field_names": ",".join(source_field_names),
            "source_field_values_json": json.dumps(source_values, ensure_ascii=False, sort_keys=True),
        }
    )


def build_modeling_ready_rows(final_rows: list[dict[str, str]], run_id: str) -> list[dict[str, str]]:
    modeling_rows: list[dict[str, str]] = []
    for row in final_rows:
        la_ga_ratio_raw = get_final_field_text(row, "la_ga_ratio")
        polymer_mw_raw = get_final_field_text(row, "polymer_mw_kDa")
        polymer_mass_raw = get_final_field_text(row, "plga_mass_mg")
        drug_mass_raw = get_final_field_text(row, "drug_feed_amount_text")

        la_fraction, ga_fraction, la_over_ga = parse_la_ga_ratio(la_ga_ratio_raw)
        append_modeling_row(
            modeling_rows,
            run_id=run_id,
            row=row,
            modeling_field_name="la_fraction",
            modeling_value=la_fraction,
            transform_rule_id="MR_PARSE_LAGA_FRACTIONS_V1",
            transform_rule_family="parse_text",
            source_field_names=["la_ga_ratio_value_text", "la_ga_ratio_value"],
        )
        append_modeling_row(
            modeling_rows,
            run_id=run_id,
            row=row,
            modeling_field_name="ga_fraction",
            modeling_value=ga_fraction,
            transform_rule_id="MR_PARSE_LAGA_FRACTIONS_V1",
            transform_rule_family="parse_text",
            source_field_names=["la_ga_ratio_value_text", "la_ga_ratio_value"],
        )
        append_modeling_row(
            modeling_rows,
            run_id=run_id,
            row=row,
            modeling_field_name="la_over_ga_ratio",
            modeling_value=la_over_ga,
            transform_rule_id="MR_PARSE_LAGA_RATIO_SCALAR_V1",
            transform_rule_family="parse_text",
            source_field_names=["la_ga_ratio_value_text", "la_ga_ratio_value"],
        )

        mw_low, mw_high = parse_mw_range(polymer_mw_raw)
        append_modeling_row(
            modeling_rows,
            run_id=run_id,
            row=row,
            modeling_field_name="polymer_mw_lower_kDa",
            modeling_value=mw_low,
            transform_rule_id="MR_PARSE_POLYMER_MW_RANGE_V1",
            transform_rule_family="parse_text",
            source_field_names=["polymer_mw_kDa_value_text", "polymer_mw_kDa_value", "plga_mw_kDa_value_text", "plga_mw_kDa_value"],
        )
        append_modeling_row(
            modeling_rows,
            run_id=run_id,
            row=row,
            modeling_field_name="polymer_mw_upper_kDa",
            modeling_value=mw_high,
            transform_rule_id="MR_PARSE_POLYMER_MW_RANGE_V1",
            transform_rule_family="parse_text",
            source_field_names=["polymer_mw_kDa_value_text", "polymer_mw_kDa_value", "plga_mw_kDa_value_text", "plga_mw_kDa_value"],
        )

        polymer_mass_mg = parse_mass_to_mg(polymer_mass_raw)
        drug_mass_mg = parse_mass_to_mg(drug_mass_raw)
        drug_to_polymer_mass_ratio = None
        if polymer_mass_mg is not None and polymer_mass_mg != 0 and drug_mass_mg is not None:
            drug_to_polymer_mass_ratio = drug_mass_mg / polymer_mass_mg

        append_modeling_row(
            modeling_rows,
            run_id=run_id,
            row=row,
            modeling_field_name="polymer_mass_mg",
            modeling_value=polymer_mass_mg,
            transform_rule_id="MR_PARSE_POLYMER_MASS_MG_V1",
            transform_rule_family="parse_text",
            source_field_names=["plga_mass_mg_value_text", "plga_mass_mg_value"],
        )
        append_modeling_row(
            modeling_rows,
            run_id=run_id,
            row=row,
            modeling_field_name="drug_mass_mg",
            modeling_value=drug_mass_mg,
            transform_rule_id="MR_PARSE_DRUG_MASS_MG_V1",
            transform_rule_family="parse_text",
            source_field_names=["drug_feed_amount_text_value_text", "drug_feed_amount_text_value"],
        )
        append_modeling_row(
            modeling_rows,
            run_id=run_id,
            row=row,
            modeling_field_name="drug_to_polymer_mass_ratio",
            modeling_value=drug_to_polymer_mass_ratio,
            transform_rule_id="MR_DERIVE_DRUG_TO_POLYMER_RATIO_V1",
            transform_rule_family="derived_math",
            source_field_names=["drug_feed_amount_text_value_text", "drug_feed_amount_text_value", "plga_mass_mg_value_text", "plga_mass_mg_value"],
        )
    return modeling_rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (Path(run_context["run_dir"]).resolve() / "modeling_ready_v1")
    )

    print(
        json.dumps(
            {
                "resolved_source_run_dir": str(run_context["run_dir"]),
                "resolved_source_run_id": str(run_context["run_id"]),
                "source_resolution": str(run_context["resolution_source"]),
                "active_run_pointer_path": str(run_context.get("pointer_path") or ""),
                "resolved_input_files": {
                    "final_table_tsv": str(final_table_path),
                },
                "resolved_out_dir": str(out_dir),
            },
            ensure_ascii=True,
            indent=2,
        )
    )

    final_rows = read_tsv(final_table_path)
    required_columns = {"key", "final_formulation_id", "representative_source_formulation_id"}
    if final_rows:
        missing_columns = required_columns.difference(final_rows[0].keys())
        if missing_columns:
            raise ValueError(f"Frozen final table missing required columns: {sorted(missing_columns)}")

    modeling_rows = build_modeling_ready_rows(
        final_rows=final_rows,
        run_id=str(run_context["run_id"]),
    )

    fieldnames = [
        "run_id",
        "key",
        "doi",
        "final_formulation_id",
        "representative_source_formulation_id",
        "benchmark_field_source_type",
        "benchmark_final_output_rule",
        "modeling_field_name",
        "modeling_value",
        "transform_rule_id",
        "transform_rule_family",
        "source_field_names",
        "source_field_values_json",
    ]
    out_path = out_dir / OUTPUT_NAME
    write_tsv(out_path, fieldnames, modeling_rows)

    field_counter = Counter(row["modeling_field_name"] for row in modeling_rows)
    rule_counter = Counter(row["transform_rule_id"] for row in modeling_rows)
    summary = {
        "run_id": str(run_context["run_id"]),
        "final_table_tsv": str(final_table_path),
        "input_final_row_count": int(len(final_rows)),
        "distinct_final_formulation_count": int(
            len(
                {
                    normalize_text(row.get("final_formulation_id"))
                    for row in final_rows
                    if normalize_text(row.get("final_formulation_id"))
                }
            )
        ),
        "modeling_value_rows": int(len(modeling_rows)),
        "field_counts": dict(sorted(field_counter.items())),
        "rule_counts": dict(sorted(rule_counter.items())),
        "allowed_operations": [
            "deterministic_parse",
            "safe_unit_harmonization",
            "deterministic_math_derivation",
        ],
        "forbidden_operations": [
            "membership_change",
            "donor_fill",
            "assumption_based_inference",
            "benchmark_final_redefinition",
        ],
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
            },
            generated_by="src/stage5_benchmark/build_modeling_ready_sidecar_v1.py",
            note="First true downstream Stage5 modeling-ready sidecar built only from the frozen benchmark-final table.",
            extra={
                "summary_json": str(summary_path),
                "output_family": "stage5_modeling_ready_sidecar",
            },
        ),
    )

    print(
        json.dumps(
            {
                "output_tsv": str(out_path),
                "summary_json": str(summary_path),
                "metadata_json": str(metadata_path),
                "modeling_value_rows": int(len(modeling_rows)),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
