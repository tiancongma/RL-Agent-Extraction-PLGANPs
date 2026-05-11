#!/usr/bin/env python3
from __future__ import annotations

"""
S5-5 derived-value sidecar skeleton.

This module builds a derived-value sidecar from S5-4 accepted direct values.  It is
intentionally separate from direct compare/final formulation outputs: derived rows
are never eligible for direct compare and are written only to S5-5 sidecar files.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ENTRYPOINT = "src/stage5_benchmark/build_s5_5_derived_values_v1.py"
BOUNDARY_CLASS = "Stage5 derived-value sidecar boundary"
BENCHMARK_VALID_STATUS = "no"

DERIVED_TSV_NAME = "s5_5_derived_values_v1.tsv"
PROVENANCE_TSV_NAME = "s5_5_derived_provenance_v1.tsv"
REVIEW_TSV_NAME = "s5_5_derived_review_queue_v1.tsv"
SUMMARY_JSON_NAME = "s5_5_derived_summary_v1.json"
RUN_CONTEXT_NAME = "RUN_CONTEXT.md"

DERIVED_COLUMNS = [
    "paper_key",
    "formulation_id",
    "target_field_name",
    "derived_value",
    "derived_unit",
    "formula_id",
    "formula_expression",
    "input_field_names",
    "input_values",
    "input_source_provenance",
    "eligible_for_direct_compare",
    "eligible_for_derived_compare",
    "needs_review",
]

PROVENANCE_COLUMNS = [
    "paper_key",
    "formulation_id",
    "target_field_name",
    "formula_id",
    "formula_expression",
    "input_field_names",
    "input_values",
    "input_source_provenance",
    "source_layer",
]

REVIEW_COLUMNS = [
    "paper_key",
    "formulation_id",
    "formula_id",
    "formula_expression",
    "target_field_name",
    "review_reason",
    "missing_input_field_names",
    "available_input_field_names",
    "eligible_for_direct_compare",
    "eligible_for_derived_compare",
    "needs_review",
]

PERCENT_WV_FIELD_ALIASES = {
    "concentration_percent_wv",
    "percent_wv",
    "drug_concentration_percent_wv",
    "drug_concentration",
    "concentration",
}
VOLUME_ML_FIELD_ALIASES = {
    "volume_ml",
    "final_volume_ml",
    "formulation_volume_ml",
    "volume",
}


@dataclass(frozen=True)
class FormulaFamily:
    formula_id: str
    formula_expression: str
    target_field_name: str
    derived_unit: str
    required_input_names: tuple[str, ...]
    derive: Callable[[list[dict[str, str]]], tuple[dict[str, Any] | None, dict[str, Any] | None]]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _clean(value).lower().replace("-", "_").replace(" ", "_")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def resolve_existing_path(value: Path, role: str) -> Path:
    path = value.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required explicit {role} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required explicit {role} is not a file: {path}")
    return path


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return [], []
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _clean(row.get(column)) for column in columns})


def _parse_float(value_text: str) -> float | None:
    text = _clean(value_text).replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _format_number(value: float) -> str:
    return f"{value:.12g}"


def _is_accepted(row: dict[str, str]) -> bool:
    decision = _norm(row.get("decision") or row.get("s5_4_decision"))
    return decision in {"", "accepted", "accept"}


def _field_matches(row: dict[str, str], aliases: set[str]) -> bool:
    return _norm(row.get("field_name")) in aliases


def _unit_contains_wv_percent(row: dict[str, str]) -> bool:
    unit = _clean(row.get("unit_text")).lower().replace(" ", "")
    return "%w/v" in unit or "w/v%" in unit or ("%" in unit and "w/v" in unit)


def _unit_is_ml(row: dict[str, str]) -> bool:
    unit = _clean(row.get("unit_text")).lower().replace(" ", "")
    return unit in {"ml", "milliliter", "milliliters", "millilitre", "millilitres"}


def _source_provenance(row: dict[str, str]) -> dict[str, str]:
    return {
        "field_name": _clean(row.get("field_name")),
        "value_text": _clean(row.get("value_text")),
        "unit_text": _clean(row.get("unit_text")),
        "source_quote": _clean(row.get("source_quote")),
        "evidence_scope": _clean(row.get("evidence_scope")),
        "decision": _clean(row.get("decision") or row.get("s5_4_decision")),
    }


def derive_percent_wv_x_ml_to_mg(rows: list[dict[str, str]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    formula_id = "percent_wv_x_ml_to_mg_v1"
    formula_expression = "%w/v × mL -> mg; mg = percent_value * 10 * volume_mL"
    target_field_name = "derived_mass_mg"

    percent_rows = [row for row in rows if _field_matches(row, PERCENT_WV_FIELD_ALIASES) and _unit_contains_wv_percent(row)]
    volume_rows = [row for row in rows if _field_matches(row, VOLUME_ML_FIELD_ALIASES) and _unit_is_ml(row)]

    available = []
    if percent_rows:
        available.append("percent_wv")
    if volume_rows:
        available.append("volume_ml")

    if len(percent_rows) != 1 or len(volume_rows) != 1:
        # Review only when at least one relevant input was present; do not guess missing values.
        if not available:
            return None, None
        missing = []
        if not percent_rows:
            missing.append("percent_wv")
        if not volume_rows:
            missing.append("volume_ml")
        reason = "insufficient_inputs_for_formula" if missing else "ambiguous_multiple_inputs_for_formula"
        return None, {
            "formula_id": formula_id,
            "formula_expression": formula_expression,
            "target_field_name": target_field_name,
            "review_reason": reason,
            "missing_input_field_names": ";".join(missing),
            "available_input_field_names": ";".join(available),
            "eligible_for_direct_compare": "no",
            "eligible_for_derived_compare": "no",
            "needs_review": "yes",
        }

    percent_value = _parse_float(percent_rows[0].get("value_text", ""))
    volume_ml = _parse_float(volume_rows[0].get("value_text", ""))
    if percent_value is None or volume_ml is None:
        missing = []
        if percent_value is None:
            missing.append("percent_wv_numeric_value")
        if volume_ml is None:
            missing.append("volume_ml_numeric_value")
        return None, {
            "formula_id": formula_id,
            "formula_expression": formula_expression,
            "target_field_name": target_field_name,
            "review_reason": "unparseable_numeric_input_for_formula",
            "missing_input_field_names": ";".join(missing),
            "available_input_field_names": ";".join(available),
            "eligible_for_direct_compare": "no",
            "eligible_for_derived_compare": "no",
            "needs_review": "yes",
        }

    mass_mg = percent_value * 10.0 * volume_ml
    input_rows = [percent_rows[0], volume_rows[0]]
    input_field_names = [_clean(row.get("field_name")) for row in input_rows]
    input_values = {
        _clean(percent_rows[0].get("field_name")): {
            "value_text": _clean(percent_rows[0].get("value_text")),
            "numeric_value": percent_value,
            "unit_text": _clean(percent_rows[0].get("unit_text")),
        },
        _clean(volume_rows[0].get("field_name")): {
            "value_text": _clean(volume_rows[0].get("value_text")),
            "numeric_value": volume_ml,
            "unit_text": _clean(volume_rows[0].get("unit_text")),
        },
    }
    provenance = [_source_provenance(row) for row in input_rows]
    return {
        "target_field_name": target_field_name,
        "derived_value": _format_number(mass_mg),
        "derived_unit": "mg",
        "formula_id": formula_id,
        "formula_expression": formula_expression,
        "input_field_names": ";".join(input_field_names),
        "input_values": _json(input_values),
        "input_source_provenance": _json(provenance),
        "eligible_for_direct_compare": "no",
        "eligible_for_derived_compare": "yes",
        "needs_review": "no",
    }, None


FORMULA_FAMILIES = [
    FormulaFamily(
        formula_id="percent_wv_x_ml_to_mg_v1",
        formula_expression="%w/v × mL -> mg; mg = percent_value * 10 * volume_mL",
        target_field_name="derived_mass_mg",
        derived_unit="mg",
        required_input_names=("percent_wv", "volume_ml"),
        derive=derive_percent_wv_x_ml_to_mg,
    )
]


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not _is_accepted(row):
            continue
        key = (_clean(row.get("paper_key")), _clean(row.get("formulation_id")))
        grouped[key].append(row)
    return dict(grouped)


def build_derived_sidecar(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    derived_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for (paper_key, formulation_id), group in sorted(group_rows(rows).items()):
        for family in FORMULA_FAMILIES:
            derived, review = family.derive(group)
            if derived is not None:
                row = {"paper_key": paper_key, "formulation_id": formulation_id, **derived}
                derived_rows.append(row)
                provenance_rows.append(
                    {
                        "paper_key": paper_key,
                        "formulation_id": formulation_id,
                        "target_field_name": row["target_field_name"],
                        "formula_id": row["formula_id"],
                        "formula_expression": row["formula_expression"],
                        "input_field_names": row["input_field_names"],
                        "input_values": row["input_values"],
                        "input_source_provenance": row["input_source_provenance"],
                        "source_layer": "s5_4_accepted_direct_values_v1",
                    }
                )
            if review is not None:
                review_rows.append({"paper_key": paper_key, "formulation_id": formulation_id, **review})
    return derived_rows, provenance_rows, review_rows


def build_summary(input_rows: list[dict[str, str]], derived_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "benchmark_valid": "no",
        "boundary_class": BOUNDARY_CLASS,
        "accepted_direct_input_rows": len(input_rows),
        "formula_families_registered": len(FORMULA_FAMILIES),
        "derived_rows": len(derived_rows),
        "review_rows": len(review_rows),
        "eligible_for_direct_compare": "no",
        "eligible_for_derived_compare_rows": sum(1 for row in derived_rows if row.get("eligible_for_derived_compare") == "yes"),
    }


def render_run_context(*, accepted_direct_values_tsv: Path, out_dir: Path, outputs: dict[str, Path], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# RUN_CONTEXT",
            "",
            "## 1. Entrypoint",
            "",
            f"- entrypoint: `{ENTRYPOINT}`",
            f"- boundary_class: `{BOUNDARY_CLASS}`",
            "",
            "## 2. Benchmark-valid status",
            "",
            f"- benchmark_valid_status: `{BENCHMARK_VALID_STATUS}`",
            "- benchmark_valid: `no`",
            "- reason: `S5-5 is a derived-value sidecar builder, not a benchmark scoring step`",
            "",
            "## 3. Exact inputs",
            "",
            f"- accepted_direct_values_tsv: `{accepted_direct_values_tsv}`",
            "",
            "## 4. Exact outputs",
            "",
            f"- out_dir: `{out_dir}`",
            *[f"- {name}: `{path}`" for name, path in sorted(outputs.items())],
            "",
            "## 5. Derived sidecar boundary",
            "",
            "- Derived sidecar boundary: reads S5-4 accepted direct values and writes only S5-5 sidecar files.",
            "- eligible_for_direct_compare: `no` for every derived/review row.",
            "- Does not change direct compare outputs.",
            "- Does not change final formulation table or formulation membership.",
            "- Does not consult GT values or benchmark answer keys.",
            "- First registered formula family: `%w/v × mL -> mg`, conventional interpretation mg = percent_value * 10 * volume_mL.",
            "- Missing inputs are routed to review; missing values are not guessed.",
            "",
            "## 6. Outcome summary",
            "",
            *[f"- {key}: `{value}`" for key, value in sorted(summary.items())],
            "",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build S5-5 derived-value sidecar from S5-4 accepted direct values.")
    parser.add_argument("--accepted-direct-values-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    accepted_direct_values_tsv = resolve_existing_path(args.accepted_direct_values_tsv, "accepted-direct-values-tsv")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _columns, rows = read_tsv(accepted_direct_values_tsv)
    accepted_rows = [row for row in rows if _is_accepted(row)]
    derived_rows, provenance_rows, review_rows = build_derived_sidecar(accepted_rows)

    derived_path = out_dir / DERIVED_TSV_NAME
    provenance_path = out_dir / PROVENANCE_TSV_NAME
    review_path = out_dir / REVIEW_TSV_NAME
    summary_path = out_dir / SUMMARY_JSON_NAME
    run_context_path = out_dir / RUN_CONTEXT_NAME

    write_tsv(derived_path, DERIVED_COLUMNS, derived_rows)
    write_tsv(provenance_path, PROVENANCE_COLUMNS, provenance_rows)
    write_tsv(review_path, REVIEW_COLUMNS, review_rows)
    summary = build_summary(accepted_rows, derived_rows, review_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = {
        "derived_values_tsv": derived_path,
        "derived_provenance_tsv": provenance_path,
        "derived_review_queue_tsv": review_path,
        "derived_summary_json": summary_path,
        "run_context_md": run_context_path,
    }
    run_context_path.write_text(
        render_run_context(
            accepted_direct_values_tsv=accepted_direct_values_tsv,
            out_dir=out_dir,
            outputs=outputs,
            summary=summary,
        ),
        encoding="utf-8",
    )

    return {"status": "ok", "out_dir": str(out_dir), "outputs": {key: str(value) for key, value in outputs.items()}, **summary}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
