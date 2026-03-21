#!/usr/bin/env python3
from __future__ import annotations

"""
Build a formulation-level value-focused GT annotation workbook from frozen
Layer 3 review artifacts.

Purpose:
- pivot field-level review rows into one row per formulation identity
- optimize manual numeric-value annotation for a small set of target fields
- preserve frozen benchmark semantics unchanged

This helper is annotation-surface only. It does not modify Stage 2, Stage 3,
or Stage 5 behavior, and it does not alter benchmark-valid outputs.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

try:
    from src.utils.paths import DATA_RESULTS_DIR
    from src.utils.run_id import is_valid_run_id, validate_artifact_subdir
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_RESULTS_DIR
    from src.utils.run_id import is_valid_run_id, validate_artifact_subdir


TARGET_FIELD_ORDER = [
    "drug_name",
    "drug_mass",
    "polymer_MW",
    "LA/GA",
    "surfactant_name",
    "surfactant_concentration",
    "drug_polymer_ratio",
    "particle_size",
    "EE",
    "LC",
]

FLAG_OPTIONS = ["correct", "incorrect", "missing", "unclear"]

BB3JUVW7_IDENTITY_OVERRIDE = {
    "BB3JUVW7_F04": "F2.2",
    "BB3JUVW7_F05": "F2.4",
    "BB3JUVW7_F06": "F2.5",
    "BB3JUVW7_F08": "F2.6",
    "BB3JUVW7_F10": "F2.7",
}

MAIN_COLUMNS = [
    "paper_key",
    "formulation_id",
    "formulation_label_stage5",
    "article_formulation_id",
    "article_formulation_label",
    "l2_gt_formulation_id",
    "l2_gt_formulation_label",
    "seed_pred_representative_source_formulation_id",
    "matched_system_formulation_id",
    "l2_gt_alignment_status",
    "paper_risk_level",
    "layer3_inclusion_flag",
    "sys_drug_name",
    "sys_drug_mass",
    "sys_polymer_MW",
    "sys_LA_GA",
    "sys_surfactant_name",
    "sys_surfactant_concentration",
    "sys_drug_polymer_ratio",
    "sys_particle_size",
    "sys_EE",
    "sys_LC",
    "gt_drug_name",
    "gt_drug_mass",
    "gt_polymer_MW",
    "gt_LA_GA",
    "gt_surfactant_name",
    "gt_surfactant_concentration",
    "gt_drug_polymer_ratio",
    "gt_particle_size",
    "gt_EE",
    "gt_LC",
    "gt_flag_drug_name",
    "gt_flag_drug_mass",
    "gt_flag_polymer_MW",
    "gt_flag_LA_GA",
    "gt_flag_surfactant_name",
    "gt_flag_surfactant_concentration",
    "gt_flag_drug_polymer_ratio",
    "gt_flag_particle_size",
    "gt_flag_EE",
    "gt_flag_LC",
    "source_locator_drug_name",
    "source_locator_drug_mass",
    "source_locator_polymer_MW",
    "source_locator_LA_GA",
    "source_locator_surfactant_name",
    "source_locator_surfactant_concentration",
    "source_locator_drug_polymer_ratio",
    "source_locator_particle_size",
    "source_locator_EE",
    "source_locator_LC",
]

REFERENCE_COLUMNS = [
    "paper_key",
    "formulation_id",
    "formulation_label_stage5",
    "article_formulation_id",
    "article_formulation_label",
    "l2_gt_formulation_id",
    "l2_gt_formulation_label",
    "seed_pred_representative_source_formulation_id",
    "matched_system_formulation_id",
    "l2_gt_alignment_status",
    "field_name",
    "extracted_value",
    "extracted_unit",
    "evidence_text",
    "evidence_anchor_text",
    "evidence_source_type",
    "evidence_status_detail",
    "relation_resolution_rule",
    "relation_resolution_confidence",
    "review_warning",
    "normalization_status",
    "paper_risk_level",
    "layer3_inclusion_flag",
]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanitize_out_subdir(value: str) -> str:
    return validate_artifact_subdir(value, param_name="--out-subdir")


def resolve_run_dir(run_id: str, explicit_run_dir: Path | None) -> Path:
    if explicit_run_dir is not None:
        return explicit_run_dir.resolve()
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run_id: {run_id}")
    return (DATA_RESULTS_DIR / run_id).resolve()


def workbook_name_for_version(version: int) -> str:
    return f"value_gt_annotation_workbook_v{int(version)}.xlsx"


def main_tsv_name_for_version(version: int) -> str:
    return f"value_gt_annotation_rows_v{int(version)}.tsv"


def reference_tsv_name_for_version(version: int) -> str:
    return f"value_gt_reference_rows_v{int(version)}.tsv"


def extra_tsv_name_for_version(version: int) -> str:
    return f"value_gt_extra_in_system_rows_v{int(version)}.tsv"


def resolve_workbook_name(version: int, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return workbook_name_for_version(version)


def format_system_value(extracted_value: str, extracted_unit: str) -> str:
    value = normalize_text(extracted_value)
    unit = normalize_text(extracted_unit)
    if not value:
        return ""
    if not unit or unit.lower() in {"ratio", "none"}:
        return value
    if unit.lower() in value.lower():
        return value
    return f"{value} {unit}"


def detect_source_locator(audit_row: dict[str, str]) -> str:
    table_id = normalize_text(audit_row.get("table_id"))
    if table_id:
        return f"Table {table_id}"

    locator = normalize_text(audit_row.get("evidence_locator"))
    figure_match = re.search(r"(?i)\bfig(?:ure)?\.?\s*(\d+)", locator)
    if figure_match:
        return f"Fig {figure_match.group(1)}"

    source_type = normalize_text(audit_row.get("evidence_source_type")).lower()
    if source_type.startswith("table"):
        return "table"
    if "figure" in source_type:
        return "figure"
    if source_type:
        return "text"

    candidate_source = normalize_text(audit_row.get("candidate_source")).lower()
    if "figure" in candidate_source:
        return "figure"
    if "table" in candidate_source:
        return "table"
    return "text"


def field_to_main_column(field_name: str) -> str:
    mapping = {
        "drug_name": "sys_drug_name",
        "drug_mass": "sys_drug_mass",
        "polymer_MW": "sys_polymer_MW",
        "LA/GA": "sys_LA_GA",
        "surfactant_name": "sys_surfactant_name",
        "surfactant_concentration": "sys_surfactant_concentration",
        "drug_polymer_ratio": "sys_drug_polymer_ratio",
        "particle_size": "sys_particle_size",
        "EE": "sys_EE",
        "LC": "sys_LC",
    }
    return mapping[field_name]


def field_to_locator_column(field_name: str) -> str:
    mapping = {
        "drug_name": "source_locator_drug_name",
        "drug_mass": "source_locator_drug_mass",
        "polymer_MW": "source_locator_polymer_MW",
        "LA/GA": "source_locator_LA_GA",
        "surfactant_name": "source_locator_surfactant_name",
        "surfactant_concentration": "source_locator_surfactant_concentration",
        "drug_polymer_ratio": "source_locator_drug_polymer_ratio",
        "particle_size": "source_locator_particle_size",
        "EE": "source_locator_EE",
        "LC": "source_locator_LC",
    }
    return mapping[field_name]


def build_seed_index(seed_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    index: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in seed_rows:
        key = (normalize_text(row.get("paper_key")), normalize_text(row.get("formulation_id")))
        field_name = normalize_text(row.get("field_name"))
        if field_name not in TARGET_FIELD_ORDER:
            continue
        index.setdefault(key, {})[field_name] = row
    return index


def build_final_row_index(final_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in final_rows:
        key = (
            normalize_text(row.get("key") or row.get("paper_key")),
            normalize_text(row.get("final_formulation_id")),
        )
        if key[0] and key[1]:
            index[key] = row
    return index


def format_final_table_value(row: dict[str, str], value_key: str, default_unit: str = "") -> str:
    value = normalize_text(row.get(value_key))
    if not value:
        return ""
    if not default_unit:
        return value
    if default_unit.lower() in value.lower():
        return value
    return f"{value} {default_unit}"


def build_final_table_field_values(row: dict[str, str]) -> dict[str, str]:
    return {
        "drug_name": normalize_text(row.get("drug_name_value")),
        "drug_mass": normalize_text(row.get("drug_feed_amount_text_value")),
        "surfactant_name": normalize_text(row.get("surfactant_name_value")),
        "particle_size": format_final_table_value(row, "size_nm_value", "nm"),
    }


def append_reference_row(
    reference_rows: list[dict[str, Any]],
    *,
    paper_key: str,
    formulation_id: str,
    formulation_label_stage5: str,
    article_formulation_id: str,
    article_formulation_label: str,
    l2_gt_formulation_id: str,
    l2_gt_formulation_label: str,
    seed_pred_representative_source_formulation_id: str,
    matched_system_formulation_id: str,
    field_name: str,
    extracted_value: str,
    extracted_unit: str,
    paper_risk_level: str,
    layer3_inclusion_flag: str,
    l2_gt_alignment_status: str = "",
) -> None:
    reference_rows.append(
        {
            "paper_key": paper_key,
            "formulation_id": formulation_id,
            "formulation_label_stage5": formulation_label_stage5,
            "article_formulation_id": article_formulation_id,
            "article_formulation_label": article_formulation_label,
            "l2_gt_formulation_id": l2_gt_formulation_id,
            "l2_gt_formulation_label": l2_gt_formulation_label,
            "seed_pred_representative_source_formulation_id": seed_pred_representative_source_formulation_id,
            "matched_system_formulation_id": matched_system_formulation_id,
            "l2_gt_alignment_status": normalize_text(l2_gt_alignment_status),
            "field_name": field_name,
            "extracted_value": normalize_text(extracted_value),
            "extracted_unit": normalize_text(extracted_unit),
            "evidence_text": "",
            "evidence_anchor_text": "",
            "evidence_source_type": "",
            "evidence_status_detail": "",
            "relation_resolution_rule": "",
            "relation_resolution_confidence": "",
            "review_warning": "",
            "normalization_status": "",
            "paper_risk_level": paper_risk_level,
            "layer3_inclusion_flag": layer3_inclusion_flag,
        }
    )


def load_workbook_rows(path: Path, sheet_name: str) -> list[dict[str, str]]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=False)
    worksheet = workbook[sheet_name]
    headers = [normalize_text(cell.value) for cell in worksheet[1]]
    rows: list[dict[str, str]] = []
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        rows.append({header: normalize_text(value) for header, value in zip(headers, row)})
    return rows


def build_prior_row_indexes(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    by_formulation_id: dict[str, dict[str, str]] = {}
    by_article_label: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        formulation_id = normalize_text(row.get("formulation_id"))
        if formulation_id:
            by_formulation_id[formulation_id] = row
        article_label = normalize_text(row.get("article_formulation_label"))
        paper_key = normalize_text(row.get("paper_key"))
        if paper_key and article_label and (paper_key, article_label) not in by_article_label:
            by_article_label[(paper_key, article_label)] = row
    return by_formulation_id, by_article_label


def build_trusted_alignment_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        gt_formulation_id = normalize_text(row.get("l2_gt_formulation_id") or row.get("formulation_id"))
        if not gt_formulation_id:
            continue
        index[gt_formulation_id] = row
    return index


def build_alignment_index(alignment_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    gt_index: dict[str, dict[str, str]] = {}
    extras: list[dict[str, str]] = []
    for row in alignment_rows:
        gt_formulation_id = normalize_text(row.get("gt_formulation_id"))
        pred_row_id = normalize_text(row.get("pred_row_id"))
        if gt_formulation_id:
            gt_index[gt_formulation_id] = row
        elif pred_row_id:
            extras.append(row)
    return gt_index, extras


def classify_gt_alignment_status(
    alignment_row: dict[str, str],
    *,
    pred_row_id: str,
    representative_source_formulation_id: str,
) -> str:
    alignment_decision = normalize_text(alignment_row.get("alignment_decision")).lower()
    alignment_confidence = normalize_text(alignment_row.get("alignment_confidence")).lower()

    trustworthy_aligned = (
        bool(pred_row_id)
        and bool(representative_source_formulation_id)
        and alignment_decision == "matched"
        and alignment_confidence in {"high", "medium"}
    )
    if trustworthy_aligned:
        return "aligned"
    if pred_row_id and alignment_decision == "matched":
        return "ambiguous_match"
    if pred_row_id:
        return "missing_in_system"
    return "missing_in_system"


def is_generated_gt_row_id(value: str, paper_key: str) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return False
    return bool(re.fullmatch(rf"{re.escape(paper_key)}_F\d+", normalized))


def has_valid_bridge_identity(
    *,
    paper_key: str,
    gt_formulation_id: str,
    gt_formulation_label: str,
    trusted_alignment_row: dict[str, str],
    article_formulation_id: str,
    article_formulation_label: str,
    seed_pred_representative_source_formulation_id: str,
) -> bool:
    seed_anchor = normalize_text(seed_pred_representative_source_formulation_id)
    article_id = normalize_text(article_formulation_id)
    article_label = normalize_text(article_formulation_label)
    gt_label = normalize_text(gt_formulation_label)
    trusted_l2_label = normalize_text(trusted_alignment_row.get("l2_gt_formulation_label"))

    if article_id and not is_generated_gt_row_id(article_id, paper_key):
        return True
    if seed_anchor and not is_generated_gt_row_id(seed_anchor, paper_key):
        return True
    if article_label and article_label not in {gt_label, gt_formulation_id}:
        return True
    if trusted_l2_label and trusted_l2_label not in {gt_label, gt_formulation_id}:
        return True
    return False


def bb3_identity_override(gt_formulation_id: str) -> tuple[str, str]:
    article_id = BB3JUVW7_IDENTITY_OVERRIDE.get(gt_formulation_id, "")
    if not article_id:
        return "", ""
    return article_id, article_id


def merge_preserved_cells(
    new_row: dict[str, Any],
    prior_row: dict[str, str] | None,
    preserve_columns: list[str],
) -> int:
    if prior_row is None:
        return 0
    preserved = 0
    for column in preserve_columns:
        if column not in new_row:
            continue
        old_value = normalize_text(prior_row.get(column))
        if old_value:
            new_row[column] = old_value
            preserved += 1
    return preserved


def match_prior_row(
    *,
    gt_formulation_id: str,
    paper_key: str,
    article_formulation_label: str,
    pred_row_id: str,
    prior_by_formulation_id: dict[str, dict[str, str]],
    prior_by_article_label: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str] | None:
    if gt_formulation_id in prior_by_formulation_id:
        return prior_by_formulation_id[gt_formulation_id]
    if pred_row_id and pred_row_id in prior_by_formulation_id:
        return prior_by_formulation_id[pred_row_id]
    if paper_key and article_formulation_label:
        return prior_by_article_label.get((paper_key, article_formulation_label))
    return None


def build_main_rows(
    audit_rows: list[dict[str, str]],
    seed_index: dict[tuple[str, str], dict[str, dict[str, str]]],
    final_row_index: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

    seen_keys: set[tuple[str, str]] = set()
    for audit_row in audit_rows:
        paper_key = normalize_text(audit_row.get("paper_id") or audit_row.get("paper_key"))
        formulation_id = normalize_text(audit_row.get("formulation_id"))
        key = (paper_key, formulation_id)
        if not paper_key or not formulation_id or key in seen_keys:
            continue
        seen_keys.add(key)

        field_rows = seed_index.get(key, {})
        identity_seed_row = next(iter(field_rows.values()), {})
        final_row = final_row_index.get(key, {})
        final_field_values = build_final_table_field_values(final_row)
        default_locator = detect_source_locator(audit_row)
        main_row: dict[str, Any] = {
            "paper_key": paper_key,
            "formulation_id": formulation_id,
            "formulation_label_stage5": normalize_text(audit_row.get("formulation_label_stage5"))
            or normalize_text(identity_seed_row.get("formulation_label_stage5")),
            "article_formulation_id": normalize_text(audit_row.get("article_formulation_id"))
            or normalize_text(identity_seed_row.get("article_formulation_id")),
            "article_formulation_label": normalize_text(audit_row.get("article_formulation_label"))
            or normalize_text(identity_seed_row.get("article_formulation_label")),
            "l2_gt_formulation_id": formulation_id,
            "l2_gt_formulation_label": normalize_text(audit_row.get("article_formulation_label"))
            or normalize_text(identity_seed_row.get("article_formulation_label")),
            "seed_pred_representative_source_formulation_id": normalize_text(
                audit_row.get("representative_source_formulation_id")
            ),
            "matched_system_formulation_id": formulation_id,
            "l2_gt_alignment_status": "aligned",
            "paper_risk_level": normalize_text(identity_seed_row.get("paper_risk_level")),
            "layer3_inclusion_flag": normalize_text(identity_seed_row.get("layer3_inclusion_flag")),
            "sys_drug_name": final_field_values.get("drug_name", ""),
            "sys_drug_mass": final_field_values.get("drug_mass", ""),
            "sys_polymer_MW": "",
            "sys_LA_GA": "",
            "sys_surfactant_name": final_field_values.get("surfactant_name", ""),
            "sys_surfactant_concentration": "",
            "sys_drug_polymer_ratio": "",
            "sys_particle_size": final_field_values.get("particle_size", ""),
            "sys_EE": "",
            "sys_LC": "",
            "gt_drug_name": "",
            "gt_drug_mass": "",
            "gt_polymer_MW": "",
            "gt_LA_GA": "",
            "gt_surfactant_name": "",
            "gt_surfactant_concentration": "",
            "gt_drug_polymer_ratio": "",
            "gt_particle_size": "",
            "gt_EE": "",
            "gt_LC": "",
            "gt_flag_drug_name": "",
            "gt_flag_drug_mass": "",
            "gt_flag_polymer_MW": "",
            "gt_flag_LA_GA": "",
            "gt_flag_surfactant_name": "",
            "gt_flag_surfactant_concentration": "",
            "gt_flag_drug_polymer_ratio": "",
            "gt_flag_particle_size": "",
            "gt_flag_EE": "",
            "gt_flag_LC": "",
            "source_locator_drug_name": default_locator if final_field_values.get("drug_name") else "",
            "source_locator_drug_mass": default_locator if final_field_values.get("drug_mass") else "",
            "source_locator_polymer_MW": "",
            "source_locator_LA_GA": "",
            "source_locator_surfactant_name": default_locator if final_field_values.get("surfactant_name") else "",
            "source_locator_surfactant_concentration": "",
            "source_locator_drug_polymer_ratio": "",
            "source_locator_particle_size": default_locator if final_field_values.get("particle_size") else "",
            "source_locator_EE": "",
            "source_locator_LC": "",
        }

        for field_name in ["drug_name", "drug_mass", "surfactant_name", "particle_size"]:
            existing_field_row = field_rows.get(field_name)
            if existing_field_row is not None and normalize_text(existing_field_row.get("extracted_value")):
                continue
            field_value = main_row[field_to_main_column(field_name)]
            if field_value:
                append_reference_row(
                    reference_rows,
                paper_key=paper_key,
                formulation_id=formulation_id,
                formulation_label_stage5=main_row["formulation_label_stage5"],
                article_formulation_id=main_row["article_formulation_id"],
                article_formulation_label=main_row["article_formulation_label"],
                l2_gt_formulation_id=main_row["formulation_id"],
                l2_gt_formulation_label=main_row["article_formulation_label"],
                seed_pred_representative_source_formulation_id="",
                matched_system_formulation_id=formulation_id,
                field_name=field_name,
                extracted_value=field_value,
                extracted_unit="",
                    paper_risk_level=main_row["paper_risk_level"],
                    layer3_inclusion_flag=main_row["layer3_inclusion_flag"],
                    l2_gt_alignment_status=main_row["l2_gt_alignment_status"],
                )

        for field_name in TARGET_FIELD_ORDER:
            field_row = field_rows.get(field_name)
            if field_row is None:
                continue
            main_row[field_to_main_column(field_name)] = format_system_value(
                extracted_value=field_row.get("extracted_value", ""),
                extracted_unit=field_row.get("extracted_unit", ""),
            )
            main_row[field_to_locator_column(field_name)] = default_locator
            append_reference_row(
                reference_rows,
                paper_key=paper_key,
                formulation_id=formulation_id,
                formulation_label_stage5=main_row["formulation_label_stage5"],
                article_formulation_id=main_row["article_formulation_id"],
                article_formulation_label=main_row["article_formulation_label"],
                l2_gt_formulation_id=main_row["formulation_id"],
                l2_gt_formulation_label=main_row["article_formulation_label"],
                seed_pred_representative_source_formulation_id="",
                matched_system_formulation_id=formulation_id,
                field_name=field_name,
                extracted_value=normalize_text(field_row.get("extracted_value")),
                extracted_unit=normalize_text(field_row.get("extracted_unit")),
                paper_risk_level=normalize_text(field_row.get("paper_risk_level")),
                layer3_inclusion_flag=normalize_text(field_row.get("layer3_inclusion_flag")),
                l2_gt_alignment_status=main_row["l2_gt_alignment_status"],
            )
            reference_rows[-1]["evidence_text"] = normalize_text(field_row.get("evidence_text"))
            reference_rows[-1]["evidence_anchor_text"] = normalize_text(field_row.get("evidence_anchor_text"))
            reference_rows[-1]["evidence_source_type"] = normalize_text(field_row.get("evidence_source_type"))
            reference_rows[-1]["evidence_status_detail"] = normalize_text(field_row.get("evidence_status_detail"))
            reference_rows[-1]["relation_resolution_rule"] = normalize_text(field_row.get("relation_resolution_rule"))
            reference_rows[-1]["relation_resolution_confidence"] = normalize_text(field_row.get("relation_resolution_confidence"))
            reference_rows[-1]["review_warning"] = normalize_text(field_row.get("review_warning"))
            reference_rows[-1]["normalization_status"] = normalize_text(field_row.get("normalization_status"))

        main_rows.append(main_row)

    main_rows.sort(key=lambda row: (row["paper_key"], row["formulation_id"]))
    reference_rows.sort(key=lambda row: (row["paper_key"], row["formulation_id"], row["field_name"]))
    return main_rows, reference_rows


def preserve_columns() -> list[str]:
    return [
        *[column for column in MAIN_COLUMNS if column.startswith("gt_")],
        *[column for column in MAIN_COLUMNS if column.startswith("source_locator_")],
    ]


def blank_main_row() -> dict[str, Any]:
    return {column: "" for column in MAIN_COLUMNS}


def build_gt_skeleton_rows(
    gt_rows: list[dict[str, str]],
    alignment_index: dict[str, dict[str, str]],
    extra_alignment_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    seed_index: dict[tuple[str, str], dict[str, dict[str, str]]],
    final_row_index: dict[tuple[str, str], dict[str, str]],
    prior_rows: list[dict[str, str]],
    trusted_alignment_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    audit_by_key = {
        (normalize_text(row.get("paper_id") or row.get("paper_key")), normalize_text(row.get("formulation_id"))): row
        for row in audit_rows
    }
    prior_by_formulation_id, prior_by_article_label = build_prior_row_indexes(prior_rows)
    trusted_alignment_index = build_trusted_alignment_index(trusted_alignment_rows)

    main_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    extra_rows: list[dict[str, Any]] = []
    preserved_nonempty_cells = 0

    for gt_row in gt_rows:
        paper_key = normalize_text(gt_row.get("paper_key"))
        gt_formulation_id = normalize_text(gt_row.get("formulation_id"))
        gt_formulation_label = normalize_text(gt_row.get("formulation_label_raw"))
        trusted_alignment_row = trusted_alignment_index.get(gt_formulation_id, {})
        alignment_row = alignment_index.get(gt_formulation_id, {})
        pred_row_id = normalize_text(alignment_row.get("pred_row_id"))
        trusted_status = normalize_text(trusted_alignment_row.get("l2_gt_alignment_status")).lower()
        audit_row = audit_by_key.get((paper_key, pred_row_id), {}) if pred_row_id else {}
        candidate_seed_source_id = normalize_text(audit_row.get("representative_source_formulation_id"))
        article_label = (
            normalize_text(trusted_alignment_row.get("article_formulation_label"))
            or normalize_text(audit_row.get("article_formulation_label"))
            or gt_formulation_label
        )
        article_formulation_id = (
            normalize_text(trusted_alignment_row.get("article_formulation_id"))
            or normalize_text(audit_row.get("article_formulation_id"))
            or normalize_text(trusted_alignment_row.get("seed_pred_representative_source_formulation_id"))
            or candidate_seed_source_id
        )
        seed_pred_representative_source_formulation_id = (
            normalize_text(trusted_alignment_row.get("seed_pred_representative_source_formulation_id"))
            or candidate_seed_source_id
        )
        bridge_identity_valid = has_valid_bridge_identity(
            paper_key=paper_key,
            gt_formulation_id=gt_formulation_id,
            gt_formulation_label=gt_formulation_label,
            trusted_alignment_row=trusted_alignment_row,
            article_formulation_id=article_formulation_id,
            article_formulation_label=article_label,
            seed_pred_representative_source_formulation_id=seed_pred_representative_source_formulation_id,
        )
        inherited_trusted_alignment = trusted_status == "aligned" and bool(pred_row_id) and bridge_identity_valid
        if inherited_trusted_alignment:
            l2_status = "aligned"
        else:
            l2_status = classify_gt_alignment_status(
                alignment_row,
                pred_row_id=pred_row_id,
                representative_source_formulation_id=candidate_seed_source_id,
            )
        if paper_key == "BB3JUVW7" and not bridge_identity_valid:
            override_article_id, override_article_label = bb3_identity_override(gt_formulation_id)
            if override_article_id:
                article_formulation_id = override_article_id
                article_label = override_article_label
                seed_pred_representative_source_formulation_id = override_article_id
                l2_status = "missing_in_system"
        aligned = l2_status == "aligned"
        field_rows = seed_index.get((paper_key, pred_row_id), {}) if aligned else {}
        final_row = final_row_index.get((paper_key, pred_row_id), {}) if aligned else {}
        final_field_values = build_final_table_field_values(final_row)
        default_locator = (
            detect_source_locator(audit_row)
            if audit_row
            else normalize_text(gt_row.get("source_locator"))
        )
        identity_seed_row = next(iter(field_rows.values()), {})

        row = blank_main_row()
        row.update(
            {
                "paper_key": paper_key,
                "formulation_id": gt_formulation_id,
                "formulation_label_stage5": normalize_text(trusted_alignment_row.get("formulation_label_stage5"))
                or normalize_text(identity_seed_row.get("formulation_label_stage5")),
                "article_formulation_id": article_formulation_id,
                "article_formulation_label": article_label,
                "l2_gt_formulation_id": gt_formulation_id,
                "l2_gt_formulation_label": gt_formulation_label,
                "seed_pred_representative_source_formulation_id": seed_pred_representative_source_formulation_id,
                "matched_system_formulation_id": normalize_text(trusted_alignment_row.get("matched_system_formulation_id"))
                or (pred_row_id if aligned else ""),
                "l2_gt_alignment_status": l2_status,
                "paper_risk_level": normalize_text(trusted_alignment_row.get("paper_risk_level"))
                or normalize_text(identity_seed_row.get("paper_risk_level")),
                "layer3_inclusion_flag": normalize_text(trusted_alignment_row.get("layer3_inclusion_flag"))
                or normalize_text(identity_seed_row.get("layer3_inclusion_flag")),
                "sys_drug_name": final_field_values.get("drug_name", ""),
                "sys_drug_mass": final_field_values.get("drug_mass", ""),
                "sys_surfactant_name": final_field_values.get("surfactant_name", ""),
                "sys_particle_size": final_field_values.get("particle_size", ""),
            }
        )

        for field_name in TARGET_FIELD_ORDER:
            field_row = field_rows.get(field_name)
            if field_row is None:
                continue
            row[field_to_main_column(field_name)] = format_system_value(
                extracted_value=field_row.get("extracted_value", ""),
                extracted_unit=field_row.get("extracted_unit", ""),
            )
            row[field_to_locator_column(field_name)] = default_locator
            append_reference_row(
                reference_rows,
                paper_key=paper_key,
                formulation_id=gt_formulation_id,
                formulation_label_stage5=row["formulation_label_stage5"],
                article_formulation_id=row["article_formulation_id"],
                article_formulation_label=row["article_formulation_label"],
                l2_gt_formulation_id=row["l2_gt_formulation_id"],
                l2_gt_formulation_label=row["l2_gt_formulation_label"],
                seed_pred_representative_source_formulation_id=row["seed_pred_representative_source_formulation_id"],
                matched_system_formulation_id=row["matched_system_formulation_id"],
                field_name=field_name,
                extracted_value=normalize_text(field_row.get("extracted_value")),
                extracted_unit=normalize_text(field_row.get("extracted_unit")),
                paper_risk_level=row["paper_risk_level"],
                layer3_inclusion_flag=row["layer3_inclusion_flag"],
                l2_gt_alignment_status=l2_status,
            )
            reference_rows[-1]["evidence_text"] = normalize_text(field_row.get("evidence_text"))
            reference_rows[-1]["evidence_anchor_text"] = normalize_text(field_row.get("evidence_anchor_text"))
            reference_rows[-1]["evidence_source_type"] = normalize_text(field_row.get("evidence_source_type"))
            reference_rows[-1]["evidence_status_detail"] = normalize_text(field_row.get("evidence_status_detail"))
            reference_rows[-1]["relation_resolution_rule"] = normalize_text(field_row.get("relation_resolution_rule"))
            reference_rows[-1]["relation_resolution_confidence"] = normalize_text(field_row.get("relation_resolution_confidence"))
            reference_rows[-1]["review_warning"] = normalize_text(field_row.get("review_warning"))
            reference_rows[-1]["normalization_status"] = normalize_text(field_row.get("normalization_status"))

        if not aligned and default_locator:
            for field_name in TARGET_FIELD_ORDER:
                row[field_to_locator_column(field_name)] = default_locator

        prior_row = match_prior_row(
            gt_formulation_id=gt_formulation_id,
            paper_key=paper_key,
            article_formulation_label=article_label,
            pred_row_id=pred_row_id,
            prior_by_formulation_id=prior_by_formulation_id,
            prior_by_article_label=prior_by_article_label,
        )
        preserved_nonempty_cells += merge_preserved_cells(row, prior_row, preserve_columns())
        main_rows.append(row)

    for alignment_row in extra_alignment_rows:
        paper_key = normalize_text(alignment_row.get("doi"))
        pred_row_id = normalize_text(alignment_row.get("pred_row_id"))
        if not pred_row_id:
            continue
        final_row = next((row for (pk, fid), row in final_row_index.items() if pk and row and fid == pred_row_id), {})
        resolved_paper_key = normalize_text(final_row.get("key") or paper_key)
        audit_row = audit_by_key.get((resolved_paper_key, pred_row_id), {})
        field_rows = seed_index.get((resolved_paper_key, pred_row_id), {})
        final_field_values = build_final_table_field_values(final_row)
        identity_seed_row = next(iter(field_rows.values()), {})
        default_locator = detect_source_locator(audit_row) if audit_row else normalize_text(alignment_row.get("pred_source_text"))
        row = blank_main_row()
        row.update(
            {
                "paper_key": resolved_paper_key,
                "formulation_id": pred_row_id,
                "formulation_label_stage5": normalize_text(identity_seed_row.get("formulation_label_stage5")),
                "article_formulation_id": normalize_text(audit_row.get("article_formulation_id")),
                "article_formulation_label": normalize_text(audit_row.get("article_formulation_label")),
                "l2_gt_formulation_id": "",
                "l2_gt_formulation_label": "",
                "seed_pred_representative_source_formulation_id": normalize_text(
                    audit_row.get("representative_source_formulation_id")
                ),
                "matched_system_formulation_id": pred_row_id,
                "l2_gt_alignment_status": "extra_in_system",
                "paper_risk_level": normalize_text(identity_seed_row.get("paper_risk_level")),
                "layer3_inclusion_flag": normalize_text(identity_seed_row.get("layer3_inclusion_flag")),
                "sys_drug_name": final_field_values.get("drug_name", ""),
                "sys_drug_mass": final_field_values.get("drug_mass", ""),
                "sys_surfactant_name": final_field_values.get("surfactant_name", ""),
                "sys_particle_size": final_field_values.get("particle_size", ""),
            }
        )
        for field_name in TARGET_FIELD_ORDER:
            field_row = field_rows.get(field_name)
            if field_row is None:
                continue
            row[field_to_main_column(field_name)] = format_system_value(
                extracted_value=field_row.get("extracted_value", ""),
                extracted_unit=field_row.get("extracted_unit", ""),
            )
            row[field_to_locator_column(field_name)] = default_locator

        prior_row = prior_by_formulation_id.get(pred_row_id)
        preserved_nonempty_cells += merge_preserved_cells(row, prior_row, preserve_columns())
        extra_rows.append(row)

    main_rows.sort(key=lambda row: (row["paper_key"], row["formulation_id"]))
    reference_rows.sort(key=lambda row: (row["paper_key"], row["formulation_id"], row["field_name"]))
    extra_rows.sort(key=lambda row: (row["paper_key"], row["formulation_id"]))
    return main_rows, reference_rows, extra_rows, preserved_nonempty_cells


def style_sheet(worksheet, freeze_panes: str = "E2") -> None:
    header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    worksheet.freeze_panes = freeze_panes
    worksheet.auto_filter.ref = worksheet.dimensions


def autosize_columns(worksheet) -> None:
    for column_cells in worksheet.columns:
        letter = get_column_letter(column_cells[0].column)
        max_len = max(len(normalize_text(cell.value)) for cell in column_cells)
        worksheet.column_dimensions[letter].width = min(max(max_len + 2, 12), 42)


def write_sheet(worksheet, columns: list[str], rows: list[dict[str, Any]], freeze_panes: str = "E2") -> None:
    worksheet.append(columns)
    for row in rows:
        worksheet.append([row.get(column, "") for column in columns])
    style_sheet(worksheet, freeze_panes=freeze_panes)
    autosize_columns(worksheet)


def validate_main_rows(main_rows: list[dict[str, Any]]) -> None:
    for row in main_rows:
        if normalize_text(row.get("l2_gt_alignment_status")) != "aligned":
            continue
        if not has_valid_bridge_identity(
            paper_key=normalize_text(row.get("paper_key")),
            gt_formulation_id=normalize_text(row.get("l2_gt_formulation_id") or row.get("formulation_id")),
            gt_formulation_label=normalize_text(row.get("l2_gt_formulation_label")),
            trusted_alignment_row={},
            article_formulation_id=normalize_text(row.get("article_formulation_id")),
            article_formulation_label=normalize_text(row.get("article_formulation_label")),
            seed_pred_representative_source_formulation_id=normalize_text(
                row.get("seed_pred_representative_source_formulation_id")
            ),
        ):
            raise ValueError(
                f"Aligned row lacks valid explicit identity anchor: {normalize_text(row.get('paper_key'))} "
                f"{normalize_text(row.get('formulation_id'))}"
            )

    bb3_expected = {
        "F1.1",
        "F1.2",
        "F1.3",
        "F1.4",
        "F1.5",
        "F2.1",
        "F2.2",
        "F2.3",
        "F2.4",
        "F2.5",
        "F2.6",
        "F2.7",
    }
    bb3_found = {
        normalize_text(row.get("article_formulation_id"))
        for row in main_rows
        if normalize_text(row.get("paper_key")) == "BB3JUVW7" and normalize_text(row.get("article_formulation_id"))
    }
    missing_bb3 = sorted(bb3_expected - bb3_found)
    if missing_bb3:
        raise ValueError(f"BB3JUVW7 missing explicit article identities: {', '.join(missing_bb3)}")


def add_flag_validation(worksheet, header_name: str) -> None:
    headers = [cell.value for cell in worksheet[1]]
    if header_name not in headers:
        return
    column_index = headers.index(header_name) + 1
    letter = get_column_letter(column_index)
    validation = DataValidation(
        type="list",
        formula1='"' + ",".join(FLAG_OPTIONS) + '"',
        allow_blank=True,
    )
    validation.prompt = "Select a structured GT flag."
    validation.error = "Use one of: correct, incorrect, missing, unclear."
    worksheet.add_data_validation(validation)
    validation.add(f"{letter}2:{letter}{max(worksheet.max_row, 2)}")


def build_workbook(
    path: Path,
    main_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
) -> None:
    workbook = Workbook()
    main_sheet = workbook.active
    main_sheet.title = "value_gt_annotation"
    write_sheet(main_sheet, MAIN_COLUMNS, main_rows, freeze_panes="H2")
    for header_name in [
        "gt_flag_drug_name",
        "gt_flag_drug_mass",
        "gt_flag_polymer_MW",
        "gt_flag_LA_GA",
        "gt_flag_surfactant_name",
        "gt_flag_surfactant_concentration",
        "gt_flag_drug_polymer_ratio",
        "gt_flag_particle_size",
        "gt_flag_EE",
        "gt_flag_LC",
    ]:
        add_flag_validation(main_sheet, header_name)

    reference_sheet = workbook.create_sheet("field_reference")
    write_sheet(reference_sheet, REFERENCE_COLUMNS, reference_rows, freeze_panes="F2")

    if extra_rows:
        extra_sheet = workbook.create_sheet("extra_in_system")
        write_sheet(extra_sheet, MAIN_COLUMNS, extra_rows, freeze_panes="H2")
        for header_name in [
            "gt_flag_drug_name",
            "gt_flag_drug_mass",
            "gt_flag_polymer_MW",
            "gt_flag_LA_GA",
            "gt_flag_surfactant_name",
            "gt_flag_surfactant_concentration",
            "gt_flag_drug_polymer_ratio",
            "gt_flag_particle_size",
            "gt_flag_EE",
            "gt_flag_LC",
        ]:
            add_flag_validation(extra_sheet, header_name)

    instructions_sheet = workbook.create_sheet("instructions")
    instructions_rows = [
        {"instruction": "Purpose", "details": "This workbook is for fast value-only GT construction using the Layer 2 GT skeleton as the authoritative main-sheet row set."},
        {"instruction": "Scope", "details": "Evidence adjudication remains a separate later phase; this sheet is optimized for rapid numeric/value entry."},
        {"instruction": "Identity", "details": "Main-sheet formulation_id comes from Layer 2 GT. Preserve l2_gt_formulation_* and seed_pred_representative_source_formulation_id as the traceable skeleton identity. Article-native labels are additive only."},
        {"instruction": "Drug mass", "details": "Drug mass is included to support later drug/polymer ratio checking or derivation."},
        {"instruction": "Alignment", "details": "Use aligned only for clear formulation-level matches with a preserved representative-source anchor. Unsafe candidates are downgraded to missing_in_system. System-only extras stay on the extra_in_system sheet."},
        {"instruction": "Flags", "details": "Use gt_flag_* dropdowns with: correct, incorrect, missing, unclear."},
    ]
    write_sheet(instructions_sheet, ["instruction", "details"], instructions_rows, freeze_panes="A2")

    dropdown_sheet = workbook.create_sheet("dropdown_options")
    dropdown_sheet.append(["flag_options"])
    for option in FLAG_OPTIONS:
        dropdown_sheet.append([option])
    style_sheet(dropdown_sheet, freeze_panes="A2")
    autosize_columns(dropdown_sheet)

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a formulation-level value GT annotation workbook.")
    parser.add_argument("--run-id", required=True, help="Run identifier for this annotation export.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory. Defaults to data/results/<run_id>.")
    parser.add_argument(
        "--out-subdir",
        default="value_gt_v1",
        type=sanitize_out_subdir,
        help="Functional artifact subdirectory under the run root.",
    )
    parser.add_argument("--audit-ready-tsv", type=Path, required=True, help="Path to final_formulation_table_audit_ready_v1.tsv")
    parser.add_argument("--seed-rows-tsv", type=Path, required=True, help="Path to field_gt_review_seed_rows_v*.tsv")
    parser.add_argument("--final-table-tsv", type=Path, required=True, help="Path to frozen final_formulation_table_v1.tsv")
    parser.add_argument("--gt-skeleton-tsv", type=Path, default=None, help="Optional Layer 2 GT skeleton TSV. When provided, main-sheet rows come from GT.")
    parser.add_argument("--alignment-scaffold-tsv", type=Path, default=None, help="Optional GT-to-system alignment scaffold TSV for GT-skeleton mode.")
    parser.add_argument("--prior-workbook-xlsx", type=Path, default=None, help="Optional prior annotation workbook to merge forward.")
    parser.add_argument("--trusted-alignment-tsv", type=Path, default=None, help="Optional prior GT-skeleton workbook/TSV rows used only as a trusted alignment bridge.")
    parser.add_argument("--workbook-name", default=None, help="Optional explicit workbook filename.")
    parser.add_argument("--artifact-version", type=int, default=1, help="Artifact version suffix.")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_id, args.run_dir)
    out_dir = run_dir / args.out_subdir

    audit_rows = read_tsv_rows(args.audit_ready_tsv.resolve())
    seed_rows = read_tsv_rows(args.seed_rows_tsv.resolve())
    final_rows = read_tsv_rows(args.final_table_tsv.resolve())
    seed_index = build_seed_index(seed_rows)
    final_row_index = build_final_row_index(final_rows)
    extra_rows: list[dict[str, Any]] = []
    preserved_nonempty_cells = 0

    if args.gt_skeleton_tsv is not None:
        if args.alignment_scaffold_tsv is None:
            raise ValueError("--alignment-scaffold-tsv is required when --gt-skeleton-tsv is provided.")
        gt_rows = read_tsv_rows(args.gt_skeleton_tsv.resolve())
        alignment_rows = read_tsv_rows(args.alignment_scaffold_tsv.resolve())
        alignment_index, extra_alignment_rows = build_alignment_index(alignment_rows)
        prior_rows = (
            load_workbook_rows(args.prior_workbook_xlsx.resolve(), "value_gt_annotation")
            if args.prior_workbook_xlsx is not None
            else []
        )
        trusted_alignment_rows = (
            read_tsv_rows(args.trusted_alignment_tsv.resolve())
            if args.trusted_alignment_tsv is not None
            else []
        )
        main_rows, reference_rows, extra_rows, preserved_nonempty_cells = build_gt_skeleton_rows(
            gt_rows=gt_rows,
            alignment_index=alignment_index,
            extra_alignment_rows=extra_alignment_rows,
            audit_rows=audit_rows,
            seed_index=seed_index,
            final_row_index=final_row_index,
            prior_rows=prior_rows,
            trusted_alignment_rows=trusted_alignment_rows,
        )
    else:
        main_rows, reference_rows = build_main_rows(audit_rows, seed_index, final_row_index)

    validate_main_rows(main_rows)

    main_tsv = out_dir / main_tsv_name_for_version(args.artifact_version)
    reference_tsv = out_dir / reference_tsv_name_for_version(args.artifact_version)
    extra_tsv = out_dir / extra_tsv_name_for_version(args.artifact_version)
    workbook_path = out_dir / resolve_workbook_name(args.artifact_version, args.workbook_name)

    write_tsv(main_tsv, MAIN_COLUMNS, main_rows)
    write_tsv(reference_tsv, REFERENCE_COLUMNS, reference_rows)
    if extra_rows:
        write_tsv(extra_tsv, MAIN_COLUMNS, extra_rows)
    build_workbook(workbook_path, main_rows, reference_rows, extra_rows)

    print(
        {
            "run_id": args.run_id,
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "final_table_tsv": str(args.final_table_tsv.resolve()),
            "main_tsv": str(main_tsv),
            "reference_tsv": str(reference_tsv),
            "extra_tsv": str(extra_tsv) if extra_rows else "",
            "workbook_path": str(workbook_path),
            "formulation_rows": len(main_rows),
            "reference_rows": len(reference_rows),
            "extra_in_system_rows": len(extra_rows),
            "preserved_nonempty_cells": preserved_nonempty_cells,
            "trusted_alignment_tsv": str(args.trusted_alignment_tsv.resolve()) if args.trusted_alignment_tsv else "",
            "target_fields": TARGET_FIELD_ORDER,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
