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
UNIT_NORMALIZATION_TSV_NAME = "s5_5_unit_normalization_v1.tsv"
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
    "normalized_value",
    "normalized_unit",
    "value_kind",
    "formula_id",
    "formula_expression",
    "input_field_names",
    "input_values",
    "input_source_provenance",
    "eligible_for_direct_compare",
    "eligible_for_derived_compare",
    "eligible_for_modeling",
    "needs_review",
]

UNIT_NORMALIZATION_COLUMNS = [
    "paper_key",
    "formulation_id",
    "source_field_name",
    "source_value",
    "source_unit",
    "target_field_name",
    "normalized_value",
    "normalized_unit",
    "value_kind",
    "formula_id",
    "formula_expression",
    "input_field_names",
    "input_values",
    "input_source_provenance",
    "eligible_for_direct_compare",
    "eligible_for_derived_compare",
    "eligible_for_modeling",
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
MASS_FIELD_ALIASES = {
    "mass",
    "mass_mg",
    "drug_mass",
    "drug_mass_mg",
    "drug_feed_amount",
    "drug_feed_amount_text",
    "polymer_mass",
    "polymer_mass_mg",
    "plga_mass",
    "plga_mass_mg",
}
CONCENTRATION_FIELD_ALIASES = {
    "concentration",
    "drug_concentration",
    "drug_concentration_value",
    "polymer_concentration",
    "polymer_concentration_value",
    "surfactant_concentration",
    "surfactant_concentration_value",
}
TOTAL_MASS_FIELD_ALIASES = {
    "total_mass",
    "total_mass_mg",
    "total_solids_mass",
    "total_solids_mass_mg",
}
RATIO_FIELD_ALIASES = {
    "ratio",
    "mass_ratio",
    "drug_polymer_ratio",
    "drug_to_polymer_ratio",
    "polymer_to_drug_ratio",
    "drug_polymer_surfactant_ratio",
    "drug_to_polymer_to_surfactant_ratio",
    "drug_polymer_stabilizer_ratio",
    "drug_to_polymer_to_stabilizer_ratio",
}
ROLE_TOKENS = ("drug", "polymer", "surfactant", "stabilizer")


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


def _field_name(row: dict[str, str]) -> str:
    return _norm(row.get("field_name"))


def _unit_contains_wv_percent(row: dict[str, str]) -> bool:
    unit = _clean(row.get("unit_text")).lower().replace(" ", "")
    return "%w/v" in unit or "w/v%" in unit or ("%" in unit and "w/v" in unit)


def _unit_is_mass(row: dict[str, str]) -> bool:
    unit = _clean(row.get("unit_text")).lower().replace(" ", "").replace("μ", "u")
    return unit in {"mg", "milligram", "milligrams", "g", "gram", "grams", "ug", "µg", "microgram", "micrograms"}


def _unit_is_concentration_mg_per_ml(row: dict[str, str]) -> bool:
    unit = _clean(row.get("unit_text")).lower().replace(" ", "")
    return unit in {"mg/ml", "mg/mL".lower(), "milligram/milliliter", "milligrams/milliliter"}


def _unit_is_ml(row: dict[str, str]) -> bool:
    unit = _clean(row.get("unit_text")).lower().replace(" ", "")
    return unit in {"ml", "milliliter", "milliliters", "millilitre", "millilitres"}


def _unit_is_volume(row: dict[str, str]) -> bool:
    unit = _clean(row.get("unit_text")).lower().replace(" ", "").replace("μ", "u")
    return unit in {"ml", "milliliter", "milliliters", "millilitre", "millilitres", "l", "liter", "liters", "litre", "litres", "ul", "µl", "microliter", "microliters", "microlitre", "microlitres"}


def _has_mass_signal(row: dict[str, str]) -> bool:
    return _field_matches(row, MASS_FIELD_ALIASES) or _field_matches(row, TOTAL_MASS_FIELD_ALIASES) or _unit_is_mass(row)


def _source_provenance(row: dict[str, str]) -> dict[str, str]:
    return {
        "field_name": _clean(row.get("field_name")),
        "value_text": _clean(row.get("value_text")),
        "unit_text": _clean(row.get("unit_text")),
        "source_quote": _clean(row.get("source_quote")),
        "evidence_scope": _clean(row.get("evidence_scope")),
        "decision": _clean(row.get("decision") or row.get("s5_4_decision")),
    }


def _role_from_field_name(field_name: str) -> str:
    field = _norm(field_name)
    if "drug" in field:
        return "drug"
    if "polymer" in field or "plga" in field:
        return "polymer"
    if "surfactant" in field:
        return "surfactant"
    if "stabilizer" in field or "emulsifier" in field:
        return "stabilizer"
    return ""


def _parse_ratio_values(value_text: str) -> list[float]:
    text = _clean(value_text)
    if not text:
        return []
    if ":" in text:
        parts = text.split(":")
    elif "/" in text:
        parts = text.split("/")
    else:
        parts = re.split(r"\s*,\s*|\s+to\s+", text, flags=re.IGNORECASE)
    values: list[float] = []
    for part in parts:
        parsed = _parse_float(part)
        if parsed is None or parsed <= 0:
            return []
        values.append(parsed)
    return values if len(values) >= 2 else []


def _ratio_roles_from_field_name(field_name: str, value_count: int) -> list[str]:
    field = _norm(field_name)
    roles: list[str] = []
    for token in re.split(r"_+", field):
        role = "polymer" if token == "plga" else token
        if role in ROLE_TOKENS and role not in roles:
            roles.append(role)
    if len(roles) != value_count:
        return []
    return roles


def _derived_row(
    *,
    paper_key: str,
    formulation_id: str,
    target_field_name: str,
    value: float,
    unit: str,
    formula_id: str,
    formula_expression: str,
    input_rows: list[dict[str, str]],
    input_values: dict[str, Any],
) -> dict[str, Any]:
    return {
        "paper_key": paper_key,
        "formulation_id": formulation_id,
        "target_field_name": target_field_name,
        "derived_value": _format_number(value),
        "derived_unit": unit,
        "normalized_value": _format_number(value),
        "normalized_unit": unit,
        "value_kind": "derived",
        "formula_id": formula_id,
        "formula_expression": formula_expression,
        "input_field_names": ";".join(_clean(row.get("field_name")) for row in input_rows),
        "input_values": _json(input_values),
        "input_source_provenance": _json([_source_provenance(row) for row in input_rows]),
        "eligible_for_direct_compare": "no",
        "eligible_for_derived_compare": "yes",
        "eligible_for_modeling": "yes",
        "needs_review": "no",
    }


def _review_row(
    *,
    paper_key: str,
    formulation_id: str,
    formula_id: str,
    formula_expression: str,
    target_field_name: str,
    reason: str,
    missing: list[str] | None = None,
    available: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "paper_key": paper_key,
        "formulation_id": formulation_id,
        "formula_id": formula_id,
        "formula_expression": formula_expression,
        "target_field_name": target_field_name,
        "review_reason": reason,
        "missing_input_field_names": ";".join(missing or []),
        "available_input_field_names": ";".join(available or []),
        "eligible_for_direct_compare": "no",
        "eligible_for_derived_compare": "no",
        "needs_review": "yes",
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
        if not percent_rows:
            # This formula family is scoped to direct %w/v inputs.  A volume
            # alone, or a non-percent concentration plus volume, belongs to
            # another formula family and should not create review noise here.
            return None, None
        if not volume_rows and any(_has_mass_signal(row) for row in rows):
            return None, None
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
        "normalized_value": _format_number(mass_mg),
        "normalized_unit": "mg",
        "value_kind": "derived",
        "formula_id": formula_id,
        "formula_expression": formula_expression,
        "input_field_names": ";".join(input_field_names),
        "input_values": _json(input_values),
        "input_source_provenance": _json(provenance),
        "eligible_for_direct_compare": "no",
        "eligible_for_derived_compare": "yes",
        "eligible_for_modeling": "yes",
        "needs_review": "no",
    }, None


def derive_mg_per_ml_x_ml_to_mg(rows: list[dict[str, str]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    formula_id = "mg_per_ml_x_ml_to_mg_v1"
    formula_expression = "mg/mL × mL -> mg; mg = concentration_mg_per_mL * volume_mL"
    target_field_name = "derived_mass_mg"

    concentration_rows = [
        row for row in rows if _field_matches(row, CONCENTRATION_FIELD_ALIASES) and _unit_is_concentration_mg_per_ml(row)
    ]
    volume_rows = [row for row in rows if _field_matches(row, VOLUME_ML_FIELD_ALIASES) and _unit_is_ml(row)]

    available = []
    if concentration_rows:
        available.append("concentration_mg_per_ml")
    if volume_rows:
        available.append("volume_ml")

    if len(concentration_rows) != 1 or len(volume_rows) != 1:
        if not concentration_rows:
            # Avoid review noise for percent-only formulas; this family applies
            # only when a direct mg/mL concentration signal is present.
            return None, None
        if not volume_rows and any(_has_mass_signal(row) for row in rows):
            return None, None
        missing = []
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

    concentration = _parse_float(concentration_rows[0].get("value_text", ""))
    volume_ml = _parse_float(volume_rows[0].get("value_text", ""))
    if concentration is None or volume_ml is None:
        missing = []
        if concentration is None:
            missing.append("concentration_mg_per_ml_numeric_value")
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

    mass_mg = concentration * volume_ml
    input_rows = [concentration_rows[0], volume_rows[0]]
    input_field_names = [_clean(row.get("field_name")) for row in input_rows]
    input_values = {
        _clean(concentration_rows[0].get("field_name")): {
            "value_text": _clean(concentration_rows[0].get("value_text")),
            "numeric_value": concentration,
            "unit_text": _clean(concentration_rows[0].get("unit_text")),
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
        "normalized_value": _format_number(mass_mg),
        "normalized_unit": "mg",
        "value_kind": "derived",
        "formula_id": formula_id,
        "formula_expression": formula_expression,
        "input_field_names": ";".join(input_field_names),
        "input_values": _json(input_values),
        "input_source_provenance": _json(provenance),
        "eligible_for_direct_compare": "no",
        "eligible_for_derived_compare": "yes",
        "eligible_for_modeling": "yes",
        "needs_review": "no",
    }, None


def _normalize_unit_text(unit_text: str) -> str:
    return _clean(unit_text).lower().replace(" ", "").replace("μ", "u")


def normalize_direct_unit_value(row: dict[str, str]) -> dict[str, Any] | None:
    value = _parse_float(row.get("value_text", ""))
    if value is None:
        return None
    unit = _normalize_unit_text(row.get("unit_text", ""))
    if _field_matches(row, MASS_FIELD_ALIASES) or _unit_is_mass(row):
        if unit in {"mg", "milligram", "milligrams"}:
            normalized_value = value
            formula_id = "mass_identity_mg_v1"
            formula_expression = "mg -> mg; identity unit normalization"
        elif unit in {"g", "gram", "grams"}:
            normalized_value = value * 1000.0
            formula_id = "mass_g_to_mg_v1"
            formula_expression = "g -> mg; mg = g * 1000"
        elif unit in {"ug", "µg", "microgram", "micrograms"}:
            normalized_value = value / 1000.0
            formula_id = "mass_ug_to_mg_v1"
            formula_expression = "ug -> mg; mg = ug / 1000"
        else:
            return None
        target_unit = "mg"
    elif _field_matches(row, VOLUME_ML_FIELD_ALIASES) or _unit_is_volume(row):
        if unit in {"ml", "milliliter", "milliliters", "millilitre", "millilitres"}:
            normalized_value = value
            formula_id = "volume_identity_ml_v1"
            formula_expression = "mL -> mL; identity unit normalization"
        elif unit in {"l", "liter", "liters", "litre", "litres"}:
            normalized_value = value * 1000.0
            formula_id = "volume_l_to_ml_v1"
            formula_expression = "L -> mL; mL = L * 1000"
        elif unit in {"ul", "µl", "microliter", "microliters", "microlitre", "microlitres"}:
            normalized_value = value / 1000.0
            formula_id = "volume_ul_to_ml_v1"
            formula_expression = "uL -> mL; mL = uL / 1000"
        else:
            return None
        target_unit = "mL"
    elif _field_matches(row, CONCENTRATION_FIELD_ALIASES) and _unit_is_concentration_mg_per_ml(row):
        normalized_value = value
        formula_id = "concentration_identity_mg_per_ml_v1"
        formula_expression = "mg/mL -> mg/mL; identity unit normalization"
        target_unit = "mg/mL"
    else:
        return None

    field_name = _clean(row.get("field_name"))
    provenance = [_source_provenance(row)]
    return {
        "source_field_name": field_name,
        "source_value": _clean(row.get("value_text")),
        "source_unit": _clean(row.get("unit_text")),
        "target_field_name": f"{field_name}_normalized",
        "normalized_value": _format_number(normalized_value),
        "normalized_unit": target_unit,
        "value_kind": "direct_normalized",
        "formula_id": formula_id,
        "formula_expression": formula_expression,
        "input_field_names": field_name,
        "input_values": _json(
            {
                field_name: {
                    "value_text": _clean(row.get("value_text")),
                    "numeric_value": value,
                    "unit_text": _clean(row.get("unit_text")),
                }
            }
        ),
        "input_source_provenance": _json(provenance),
        "eligible_for_direct_compare": "no",
        "eligible_for_derived_compare": "no",
        "eligible_for_modeling": "yes",
        "needs_review": "no",
    }


FORMULA_FAMILIES = [
    FormulaFamily(
        formula_id="percent_wv_x_ml_to_mg_v1",
        formula_expression="%w/v × mL -> mg; mg = percent_value * 10 * volume_mL",
        target_field_name="derived_mass_mg",
        derived_unit="mg",
        required_input_names=("percent_wv", "volume_ml"),
        derive=derive_percent_wv_x_ml_to_mg,
    ),
    FormulaFamily(
        formula_id="mg_per_ml_x_ml_to_mg_v1",
        formula_expression="mg/mL × mL -> mg; mg = concentration_mg_per_mL * volume_mL",
        target_field_name="derived_mass_mg",
        derived_unit="mg",
        required_input_names=("concentration_mg_per_ml", "volume_ml"),
        derive=derive_mg_per_ml_x_ml_to_mg,
    ),
]


def _normalized_mass_mg(row: dict[str, str]) -> float | None:
    if not (_field_matches(row, MASS_FIELD_ALIASES) or _field_matches(row, TOTAL_MASS_FIELD_ALIASES) or _unit_is_mass(row)):
        return None
    normalized = normalize_direct_unit_value(row)
    if not normalized or normalized.get("normalized_unit") != "mg":
        return None
    return _parse_float(str(normalized.get("normalized_value", "")))


def _normalized_volume_ml(row: dict[str, str]) -> float | None:
    if not (_field_matches(row, VOLUME_ML_FIELD_ALIASES) or _unit_is_volume(row)):
        return None
    normalized = normalize_direct_unit_value(row)
    if not normalized or normalized.get("normalized_unit") != "mL":
        return None
    return _parse_float(str(normalized.get("normalized_value", "")))


def _normalized_concentration_mg_per_ml(row: dict[str, str]) -> float | None:
    value = _parse_float(row.get("value_text", ""))
    if value is None:
        return None
    if _unit_is_concentration_mg_per_ml(row):
        return value
    if _unit_contains_wv_percent(row):
        return value * 10.0
    return None


def solve_concentration_mass_volume(group: list[dict[str, str]], *, paper_key: str, formulation_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    derived_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    mass_rows = [row for row in group if _normalized_mass_mg(row) is not None and not _field_matches(row, TOTAL_MASS_FIELD_ALIASES)]
    concentration_rows = [row for row in group if _normalized_concentration_mg_per_ml(row) is not None]
    volume_rows = [row for row in group if _normalized_volume_ml(row) is not None]
    if len(mass_rows) == 1 and len(concentration_rows) == 1 and not volume_rows:
        mass_mg = _normalized_mass_mg(mass_rows[0])
        concentration = _normalized_concentration_mg_per_ml(concentration_rows[0])
        if mass_mg is not None and concentration and concentration > 0:
            role = _role_from_field_name(mass_rows[0].get("field_name", "")) or _role_from_field_name(concentration_rows[0].get("field_name", ""))
            target = f"{role}_derived_volume_mL" if role else "derived_volume_mL"
            derived_rows.append(
                _derived_row(
                    paper_key=paper_key,
                    formulation_id=formulation_id,
                    target_field_name=target,
                    value=mass_mg / concentration,
                    unit="mL",
                    formula_id="mass_mg_div_concentration_mg_per_ml_to_ml_v1",
                    formula_expression="mg / (mg/mL) -> mL; volume_mL = mass_mg / concentration_mg_per_mL",
                    input_rows=[mass_rows[0], concentration_rows[0]],
                    input_values={
                        _clean(mass_rows[0].get("field_name")): {"normalized_value": mass_mg, "normalized_unit": "mg"},
                        _clean(concentration_rows[0].get("field_name")): {
                            "normalized_value": concentration,
                            "normalized_unit": "mg/mL",
                        },
                    },
                )
            )
    if len(mass_rows) == 1 and len(volume_rows) == 1 and not concentration_rows:
        mass_mg = _normalized_mass_mg(mass_rows[0])
        volume_ml = _normalized_volume_ml(volume_rows[0])
        if mass_mg is not None and volume_ml and volume_ml > 0:
            role = _role_from_field_name(mass_rows[0].get("field_name", ""))
            target = f"{role}_derived_concentration_mg_per_mL" if role else "derived_concentration_mg_per_mL"
            derived_rows.append(
                _derived_row(
                    paper_key=paper_key,
                    formulation_id=formulation_id,
                    target_field_name=target,
                    value=mass_mg / volume_ml,
                    unit="mg/mL",
                    formula_id="mass_mg_div_volume_ml_to_mg_per_ml_v1",
                    formula_expression="mg / mL -> mg/mL; concentration_mg_per_mL = mass_mg / volume_mL",
                    input_rows=[mass_rows[0], volume_rows[0]],
                    input_values={
                        _clean(mass_rows[0].get("field_name")): {"normalized_value": mass_mg, "normalized_unit": "mg"},
                        _clean(volume_rows[0].get("field_name")): {"normalized_value": volume_ml, "normalized_unit": "mL"},
                    },
                )
            )
    if len(mass_rows) > 1 and len(concentration_rows) == 1 and not volume_rows:
        review_rows.append(
            _review_row(
                paper_key=paper_key,
                formulation_id=formulation_id,
                formula_id="mass_mg_div_concentration_mg_per_ml_to_ml_v1",
                formula_expression="mg / (mg/mL) -> mL; volume_mL = mass_mg / concentration_mg_per_mL",
                target_field_name="derived_volume_mL",
                reason="ambiguous_multiple_mass_inputs_for_formula",
                available=[_clean(row.get("field_name")) for row in mass_rows],
            )
        )
    return derived_rows, review_rows


def solve_ratio_mass_system(group: list[dict[str, str]], *, paper_key: str, formulation_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    derived_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    mass_by_role: dict[str, tuple[float, dict[str, str]]] = {}
    total_mass: tuple[float, dict[str, str]] | None = None
    for row in group:
        mass = _normalized_mass_mg(row)
        if mass is None:
            continue
        if _field_matches(row, TOTAL_MASS_FIELD_ALIASES):
            total_mass = (mass, row)
            continue
        role = _role_from_field_name(row.get("field_name", ""))
        if role and role not in mass_by_role:
            mass_by_role[role] = (mass, row)

    for ratio_row in [row for row in group if _field_matches(row, RATIO_FIELD_ALIASES) or "ratio" in _field_name(row)]:
        ratio_values = _parse_ratio_values(ratio_row.get("value_text", ""))
        roles = _ratio_roles_from_field_name(ratio_row.get("field_name", ""), len(ratio_values))
        formula_expression = "role ratio + known mass -> missing role masses; missing_mass = known_mass * target_ratio / known_ratio"
        if ratio_values and not roles:
            review_rows.append(
                _review_row(
                    paper_key=paper_key,
                    formulation_id=formulation_id,
                    formula_id="role_ratio_mass_solver_v1",
                    formula_expression=formula_expression,
                    target_field_name="derived_component_mass_mg",
                    reason="ambiguous_ratio_role_order",
                    available=[_clean(ratio_row.get("field_name"))],
                )
            )
            continue
        if len(roles) < 2:
            continue
        known_roles = [role for role in roles if role in mass_by_role]
        input_rows = [ratio_row]
        if len(known_roles) == 1:
            known_role = known_roles[0]
            known_mass, known_row = mass_by_role[known_role]
            known_ratio = ratio_values[roles.index(known_role)]
            input_rows.append(known_row)
            for role, ratio_value in zip(roles, ratio_values):
                if role == known_role or role in mass_by_role:
                    continue
                derived_rows.append(
                    _derived_row(
                        paper_key=paper_key,
                        formulation_id=formulation_id,
                        target_field_name=f"{role}_derived_mass_mg",
                        value=known_mass * ratio_value / known_ratio,
                        unit="mg",
                        formula_id="role_ratio_known_mass_to_missing_mass_v1",
                        formula_expression=formula_expression,
                        input_rows=input_rows,
                        input_values={
                            _clean(ratio_row.get("field_name")): {
                                "roles": roles,
                                "ratio_values": ratio_values,
                            },
                            _clean(known_row.get("field_name")): {
                                "role": known_role,
                                "normalized_value": known_mass,
                                "normalized_unit": "mg",
                            },
                        },
                    )
                )
        elif len(known_roles) > 1:
            review_rows.append(
                _review_row(
                    paper_key=paper_key,
                    formulation_id=formulation_id,
                    formula_id="role_ratio_mass_solver_v1",
                    formula_expression=formula_expression,
                    target_field_name="derived_component_mass_mg",
                    reason="multiple_known_masses_require_consistency_review",
                    available=known_roles,
                )
            )
        elif total_mass is not None:
            total_value, total_row = total_mass
            ratio_sum = sum(ratio_values)
            input_rows.append(total_row)
            for role, ratio_value in zip(roles, ratio_values):
                if role in mass_by_role:
                    continue
                derived_rows.append(
                    _derived_row(
                        paper_key=paper_key,
                        formulation_id=formulation_id,
                        target_field_name=f"{role}_derived_mass_mg",
                        value=total_value * ratio_value / ratio_sum,
                        unit="mg",
                        formula_id="role_ratio_total_mass_to_component_masses_v1",
                        formula_expression="role ratio + total mass -> component masses; component_mass = total_mass * role_ratio / sum_ratios",
                        input_rows=input_rows,
                        input_values={
                            _clean(ratio_row.get("field_name")): {
                                "roles": roles,
                                "ratio_values": ratio_values,
                            },
                            _clean(total_row.get("field_name")): {
                                "normalized_value": total_value,
                                "normalized_unit": "mg",
                            },
                        },
                    )
                )
    return derived_rows, review_rows


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not _is_accepted(row):
            continue
        key = (_clean(row.get("paper_key")), _clean(row.get("formulation_id")))
        grouped[key].append(row)
    return dict(grouped)


def build_unit_normalization_sidecar(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = normalize_direct_unit_value(row)
        if normalized is None:
            continue
        normalized_rows.append(
            {
                "paper_key": _clean(row.get("paper_key")),
                "formulation_id": _clean(row.get("formulation_id")),
                **normalized,
            }
        )
    return normalized_rows


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
        for solver in (solve_concentration_mass_volume, solve_ratio_mass_system):
            solver_derived_rows, solver_review_rows = solver(group, paper_key=paper_key, formulation_id=formulation_id)
            for row in solver_derived_rows:
                derived_rows.append(row)
                provenance_rows.append(
                    {
                        "paper_key": row["paper_key"],
                        "formulation_id": row["formulation_id"],
                        "target_field_name": row["target_field_name"],
                        "formula_id": row["formula_id"],
                        "formula_expression": row["formula_expression"],
                        "input_field_names": row["input_field_names"],
                        "input_values": row["input_values"],
                        "input_source_provenance": row["input_source_provenance"],
                        "source_layer": "s5_4_accepted_direct_values_v1",
                    }
                )
            review_rows.extend(solver_review_rows)
    return derived_rows, provenance_rows, review_rows


def build_summary(
    input_rows: list[dict[str, str]],
    derived_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    unit_normalization_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "benchmark_valid": "no",
        "boundary_class": BOUNDARY_CLASS,
        "accepted_direct_input_rows": len(input_rows),
        "formula_families_registered": len(FORMULA_FAMILIES),
        "derived_rows": len(derived_rows),
        "unit_normalization_rows": len(unit_normalization_rows),
        "review_rows": len(review_rows),
        "eligible_for_direct_compare": "no",
        "eligible_for_derived_compare_rows": sum(1 for row in derived_rows if row.get("eligible_for_derived_compare") == "yes"),
        "eligible_for_modeling_rows": sum(
            1 for row in [*derived_rows, *unit_normalization_rows] if row.get("eligible_for_modeling") == "yes"
        ),
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
            "- Additional formula family: `mg/mL × mL -> mg`, conventional interpretation mg = concentration_mg_per_mL * volume_mL.",
            "- Concentration-mass-volume solver can derive volume or concentration when exactly two dimensions are accepted and unit-normalizable.",
            "- Ratio-mass solver can derive missing component masses from explicit role-ordered binary or ternary ratios plus one known role mass or total mass.",
            "- Unit-normalization sidecar converts accepted direct mass and volume inputs into modeling units such as mg and mL without writing direct fields.",
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
    unit_normalization_rows = build_unit_normalization_sidecar(accepted_rows)
    derived_rows, provenance_rows, review_rows = build_derived_sidecar(accepted_rows)

    derived_path = out_dir / DERIVED_TSV_NAME
    unit_normalization_path = out_dir / UNIT_NORMALIZATION_TSV_NAME
    provenance_path = out_dir / PROVENANCE_TSV_NAME
    review_path = out_dir / REVIEW_TSV_NAME
    summary_path = out_dir / SUMMARY_JSON_NAME
    run_context_path = out_dir / RUN_CONTEXT_NAME

    write_tsv(derived_path, DERIVED_COLUMNS, derived_rows)
    write_tsv(unit_normalization_path, UNIT_NORMALIZATION_COLUMNS, unit_normalization_rows)
    write_tsv(provenance_path, PROVENANCE_COLUMNS, provenance_rows)
    write_tsv(review_path, REVIEW_COLUMNS, review_rows)
    summary = build_summary(accepted_rows, derived_rows, review_rows, unit_normalization_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = {
        "derived_values_tsv": derived_path,
        "unit_normalization_tsv": unit_normalization_path,
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
