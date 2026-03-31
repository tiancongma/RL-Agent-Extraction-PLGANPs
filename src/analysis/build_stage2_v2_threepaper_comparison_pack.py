#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from src.utils.active_data_source import (
        build_artifact_metadata,
        resolve_artifact_path,
        resolve_run_context,
        write_artifact_metadata_json,
    )
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.active_data_source import (
        build_artifact_metadata,
        resolve_artifact_path,
        resolve_run_context,
        write_artifact_metadata_json,
    )
    from src.utils.paths import PROJECT_ROOT


PAPER_LEVEL_COUNTS = "paper_level_counts.tsv"
BOUNDARY_REVIEW = "boundary_review.tsv"
COMPONENT_COMPLETENESS_REVIEW = "component_completeness_review.tsv"
EXPRESSION_RICHNESS_REVIEW = "expression_richness_review.tsv"
VARIABLE_DETECTION_REVIEW = "variable_detection_review.tsv"
AMBIGUITY_HANDLING_REVIEW = "ambiguity_handling_review.tsv"
STRUCTURAL_SUMMARY = "structural_comparison_summary.tsv"
STRUCTURAL_REPORT = "structural_comparison_report.md"

BOUNDARY_SURFACE = "boundary_surface.tsv"
VARIABLE_RETENTION = "variable_retention.tsv"
MEASUREMENT_RETENTION = "measurement_retention.tsv"
COMPARISON_SUMMARY = "comparison_summary.tsv"
COMPARISON_REPORT = "comparison_report.md"

DEFAULT_HISTORICAL_TSV = (
    "data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/"
    "weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv"
)
DEFAULT_REPLAY_V2_JSONL = (
    "data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/"
    "semantic_stage2_v2/semantic_stage2_v2_objects.jsonl"
)
MEASUREMENT_TOKENS = {
    "particle_size",
    "size",
    "size_nm",
    "pdi",
    "zeta",
    "zeta_potential",
    "zeta_mv",
    "encapsulation_efficiency",
    "encapsulation_efficiency_percent",
    "loading_content",
    "loading_content_percent",
}
DOE_TOKENS = {
    "aqueous_organic_phase_ratio",
    "drug_concentration",
    "polymer_concentration",
    "surfactant_concentration",
    "cpf",
    "cplga",
    "cpva",
    "ph",
}
PH_TOKENS = {"ph", "aqueous_ph", "aqueous_phase_ph"}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_token(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_jsonish_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def with_default_path(value: str) -> Path | None:
    text = normalize_text(value)
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def has_unit_text(text: Any) -> bool:
    value = normalize_text(text)
    if not value:
        return False
    return bool(re.search(r"(mg/mL|mg|g|kg|mL|uL|L|%|nm|mV|kDa|ratio)", value, flags=re.IGNORECASE))


def split_notes(parts: list[str]) -> str:
    return "; ".join(part for part in parts if normalize_text(part))


def status_from_bool(flag: bool) -> str:
    return "yes" if flag else "no"


def overall_status(boundary_status: str, component_status: str, expression_status: str, variable_status: str, ambiguity_status: str) -> str:
    statuses = [boundary_status, component_status, expression_status, variable_status, ambiguity_status]
    if any(item == "no" for item in statuses):
        return "caution"
    if any(item == "partial" for item in statuses):
        return "caution"
    return "go"


def count_gt_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        key = normalize_text(row.get("paper_key") or row.get("key"))
        if key:
            counts[key] += 1
    return dict(counts)


def inspect_llm_doc(document: dict[str, Any], surface_name: str) -> dict[str, Any]:
    formulation_candidates = [item for item in document.get("formulation_candidates") or [] if isinstance(item, dict)]
    component_candidates = [item for item in document.get("component_candidates") or [] if isinstance(item, dict)]
    variable_candidates = [item for item in document.get("variable_candidates") or [] if isinstance(item, dict)]
    measurement_candidates = [item for item in document.get("measurement_candidates") or [] if isinstance(item, dict)]
    unassigned = [item for item in document.get("unassigned_observations") or [] if isinstance(item, dict)]

    variable_names = {
        normalize_token(item.get("variable_name") or item.get("variable_name_raw"))
        for item in variable_candidates
        if normalize_text(item.get("variable_name") or item.get("variable_name_raw"))
    }
    measurement_names = {
        normalize_token(item.get("measurement_name") or item.get("measurement_name_raw"))
        for item in measurement_candidates
        if normalize_text(item.get("measurement_name") or item.get("measurement_name_raw"))
    }
    component_names = {
        normalize_text(item.get("component_name") or item.get("name_raw"))
        for item in component_candidates
        if normalize_text(item.get("component_name") or item.get("name_raw"))
    }
    component_name_tokens = {normalize_token(name) for name in component_names if name}
    components_by_formulation: dict[str, set[str]] = defaultdict(set)
    expression_values_by_component: dict[tuple[str, str], set[str]] = defaultdict(set)
    expression_kind_counts: Counter[str] = Counter()
    unit_bearing_component_expressions = 0

    for item in component_candidates:
        fid = normalize_text(item.get("formulation_candidate_id"))
        component_name = normalize_text(item.get("component_name") or item.get("name_raw"))
        amount_text = normalize_text(item.get("amount_text"))
        amount_kind = normalize_token(item.get("amount_kind")) or "unknown"
        if fid and component_name:
            components_by_formulation[fid].add(component_name)
            if amount_text:
                expression_values_by_component[(fid, component_name)].add(amount_text)
        expression_kind_counts[amount_kind] += 1
        if has_unit_text(amount_text):
            unit_bearing_component_expressions += 1

    ambiguity_signal_count = len(unassigned)
    ambiguity_signal_count += sum(
        1
        for collection in [formulation_candidates, component_candidates, variable_candidates, measurement_candidates]
        for item in collection
        if normalize_text(item.get("ambiguity_note")) or normalize_text(item.get("status")) == "ambiguous"
    )

    multi_component_formulations = sum(1 for names in components_by_formulation.values() if len(names) >= 2)
    multiple_expression_preserved = any(len(values) >= 2 for values in expression_values_by_component.values())
    measurement_unit_count = sum(1 for item in measurement_candidates if has_unit_text(item.get("value_text")) or has_unit_text(item.get("unit_text")))

    return {
        "paper_key": normalize_text(document.get("document_key")),
        "surface_name": surface_name,
        "formulation_count": len(formulation_candidates),
        "component_count": len(component_candidates),
        "variable_count": len(variable_candidates),
        "measurement_count": len(measurement_candidates),
        "component_names": component_names,
        "component_name_tokens": component_name_tokens,
        "variable_names": variable_names,
        "measurement_names": measurement_names,
        "multi_component_formulation_count": multi_component_formulations,
        "expression_kind_counts": dict(expression_kind_counts),
        "unit_bearing_expression_count": unit_bearing_component_expressions + measurement_unit_count,
        "raw_units_preserved": unit_bearing_component_expressions + measurement_unit_count > 0,
        "multiple_expressions_preserved": multiple_expression_preserved,
        "ambiguity_signal_count": ambiguity_signal_count,
        "incorrect_variable_promotions": sorted(variable_names & MEASUREMENT_TOKENS),
        "doe_factor_retained": any(name in DOE_TOKENS for name in variable_names),
        "ph_retained": any(name in PH_TOKENS for name in variable_names),
        "pdi_retained": "pdi" in measurement_names,
        "zeta_retained": "zeta_mv" in measurement_names or "zeta_potential" in measurement_names or "zeta" in measurement_names,
        "notes": [],
    }


def inspect_current_semantic_doc(document: dict[str, Any]) -> dict[str, Any]:
    formulation_candidates = [
        item
        for item in (document.get("formulation_identity_candidates") or document.get("formulation_candidates") or [])
        if isinstance(item, dict)
    ]
    component_candidates = [item for item in document.get("component_candidates") or [] if isinstance(item, dict)]
    variable_candidates = [
        item
        for item in (document.get("variable_or_factor_candidates") or document.get("variable_candidates") or [])
        if isinstance(item, dict)
    ]
    measurement_candidates = [item for item in document.get("measurement_candidates") or [] if isinstance(item, dict)]

    component_names = {
        normalize_text(item.get("component_name_raw") or item.get("component_name") or item.get("component_name_normalized"))
        for item in component_candidates
        if normalize_text(item.get("component_name_raw") or item.get("component_name") or item.get("component_name_normalized"))
    }
    component_name_tokens = {normalize_token(name) for name in component_names if name}
    variable_names = {
        normalize_token(item.get("factor_name_raw") or item.get("variable_name"))
        for item in variable_candidates
        if normalize_text(item.get("factor_name_raw") or item.get("variable_name"))
    }
    measurement_names = {
        normalize_token(item.get("measurement_name_raw") or item.get("measurement_name"))
        for item in measurement_candidates
        if normalize_text(item.get("measurement_name_raw") or item.get("measurement_name"))
    }

    components_by_formulation: dict[str, set[str]] = defaultdict(set)
    expression_kind_counts: Counter[str] = Counter()
    unit_bearing_expression_count = 0
    for item in component_candidates:
        fid = normalize_text(item.get("formulation_candidate_id"))
        component_name = normalize_text(item.get("component_name_raw") or item.get("component_name") or item.get("component_name_normalized"))
        expression_text = normalize_text(item.get("component_expression_raw") or item.get("component_amount_raw"))
        if fid and component_name:
            components_by_formulation[fid].add(component_name)
        if expression_text:
            unit_bearing_expression_count += int(has_unit_text(expression_text))
            token = "ratio" if "ratio" in expression_text.lower() else "unknown"
            if "mg/ml" in expression_text.lower() or "%" in expression_text.lower():
                token = "concentration"
            elif re.search(r"\b(?:mg|g|kg)\b", expression_text.lower()):
                token = "mass"
            elif re.search(r"\b(?:ml|ul|l)\b", expression_text.lower()):
                token = "volume"
            expression_kind_counts[token] += 1
    for item in variable_candidates:
        expression_text = normalize_text(item.get("factor_expression_raw"))
        if expression_text:
            unit_bearing_expression_count += int(has_unit_text(expression_text))
            token = "ratio" if "ratio" in normalize_text(item.get("factor_name_raw")).lower() else "unknown"
            if "mg/ml" in expression_text.lower() or "%" in expression_text.lower():
                token = "concentration"
            elif re.search(r"\b(?:mg|g|kg)\b", expression_text.lower()):
                token = "mass"
            expression_kind_counts[token] += 1

    return {
        "paper_key": normalize_text(document.get("document_key")),
        "surface_name": "current_deterministic_semantic_active_run",
        "formulation_count": len(formulation_candidates),
        "component_count": len(component_candidates),
        "variable_count": len(variable_candidates),
        "measurement_count": len(measurement_candidates),
        "component_names": component_names,
        "component_name_tokens": component_name_tokens,
        "variable_names": variable_names,
        "measurement_names": measurement_names,
        "multi_component_formulation_count": sum(1 for names in components_by_formulation.values() if len(names) >= 2),
        "expression_kind_counts": dict(expression_kind_counts),
        "unit_bearing_expression_count": unit_bearing_expression_count,
        "raw_units_preserved": unit_bearing_expression_count > 0,
        "multiple_expressions_preserved": False,
        "ambiguity_signal_count": 0,
        "incorrect_variable_promotions": sorted(variable_names & MEASUREMENT_TOKENS),
        "doe_factor_retained": any(name in DOE_TOKENS for name in variable_names),
        "ph_retained": any(name in PH_TOKENS for name in variable_names),
        "pdi_retained": "pdi" in measurement_names,
        "zeta_retained": "zeta_mv" in measurement_names or "zeta_potential" in measurement_names or "zeta" in measurement_names,
        "notes": [],
    }


def inspect_widerow_rows(rows: list[dict[str, str]], surface_name: str) -> dict[str, Any]:
    formulation_ids = {
        normalize_text(row.get("formulation_id") or row.get("raw_formulation_label"))
        for row in rows
        if normalize_text(row.get("formulation_id") or row.get("raw_formulation_label"))
    }
    component_names: set[str] = set()
    component_name_tokens: set[str] = set()
    variable_names: set[str] = set()
    measurement_names: set[str] = set()
    expression_kind_counts: Counter[str] = Counter()
    unit_bearing_expression_count = 0
    multi_component_formulation_count = 0

    for row in rows:
        components_in_row = []
        for field in [
            "polymer_identity",
            "polymer_name_raw",
            "drug_name_value_text",
            "surfactant_name_value_text",
            "organic_solvent_value_text",
        ]:
            value = normalize_text(row.get(field))
            if value:
                component_names.add(value)
                component_name_tokens.add(normalize_token(value))
                components_in_row.append(value)
        if len(set(components_in_row)) >= 2:
            multi_component_formulation_count += 1
        for field, token in [
            ("plga_mass_mg_value_text", "mass"),
            ("drug_feed_amount_text_value_text", "mass"),
            ("surfactant_concentration_text_value_text", "concentration"),
            ("pva_conc_percent_value_text", "concentration"),
            ("size_nm_value_text", "size_nm"),
            ("pdi_value_text", "pdi"),
            ("zeta_mV_value_text", "zeta_mv"),
        ]:
            value = normalize_text(row.get(field))
            if value:
                if token in {"size_nm", "pdi", "zeta_mv"}:
                    measurement_names.add(token)
                else:
                    expression_kind_counts[token] += 1
                unit_bearing_expression_count += int(has_unit_text(value))
        for item in parse_jsonish_list(row.get("identity_variables_json")):
            if isinstance(item, dict):
                name = normalize_token(item.get("name") or item.get("name_raw"))
                if name:
                    variable_names.add(name)
        tag_blob = " ".join(
            [
                normalize_text(row.get("instance_context_tags")),
                normalize_text(row.get("change_context_tags")),
                normalize_text(row.get("raw_formulation_label")),
            ]
        ).lower()
        if "ph" in tag_blob:
            variable_names.add("ph")

    return {
        "paper_key": normalize_text(rows[0].get("key")) if rows else "",
        "surface_name": surface_name,
        "formulation_count": len(formulation_ids),
        "component_count": "",
        "variable_count": len(variable_names),
        "measurement_count": len(measurement_names),
        "component_names": component_names,
        "component_name_tokens": component_name_tokens,
        "variable_names": variable_names,
        "measurement_names": measurement_names,
        "multi_component_formulation_count": multi_component_formulation_count,
        "expression_kind_counts": dict(expression_kind_counts),
        "unit_bearing_expression_count": unit_bearing_expression_count,
        "raw_units_preserved": unit_bearing_expression_count > 0,
        "multiple_expressions_preserved": False,
        "ambiguity_signal_count": 0,
        "incorrect_variable_promotions": sorted(variable_names & MEASUREMENT_TOKENS),
        "doe_factor_retained": any(name in DOE_TOKENS for name in variable_names),
        "ph_retained": any(name in PH_TOKENS for name in variable_names),
        "pdi_retained": "pdi" in measurement_names,
        "zeta_retained": "zeta_mv" in measurement_names or "zeta_potential" in measurement_names or "zeta" in measurement_names,
        "notes": [],
    }


def compare_count_status(live_count: int, gt_count: int, deterministic_count: int, replay_count: str) -> str:
    status = []
    if live_count == gt_count:
        status.append("live_exact_gt")
    elif live_count < gt_count:
        status.append("live_under_gt")
    else:
        status.append("live_over_gt")
    if deterministic_count == gt_count:
        status.append("deterministic_exact_gt")
    elif deterministic_count < gt_count:
        status.append("deterministic_under_gt")
    else:
        status.append("deterministic_over_gt")
    if normalize_text(replay_count):
        status.append(f"replay={replay_count}")
    return ";".join(status)


def build_boundary_review(paper_key: str, live: dict[str, Any], gt_count: int, deterministic: dict[str, Any] | None, replay: dict[str, Any] | None, historical: dict[str, Any] | None) -> dict[str, Any]:
    live_count = int(live["formulation_count"])
    under = live_count < gt_count
    over = live_count > gt_count
    if live_count == gt_count:
        status = "yes"
    elif abs(live_count - gt_count) <= 2:
        status = "partial"
    else:
        status = "no"
    notes = [f"live={live_count}", f"gt={gt_count}"]
    if deterministic:
        notes.append(f"deterministic={deterministic['formulation_count']}")
    if replay:
        notes.append(f"replay={replay['formulation_count']}")
    if historical:
        notes.append(f"legacy={historical['formulation_count']}")
    if under:
        notes.append("live under-splits paper-reported boundary inventory")
    if over:
        notes.append("live over-splits beyond paper-reported boundary inventory")
    return {
        "paper_key": paper_key,
        "aligns_with_reported_boundaries": status,
        "obvious_under_splitting": status_from_bool(under),
        "obvious_over_splitting": status_from_bool(over),
        "note": split_notes(notes),
    }


def build_component_review(paper_key: str, live: dict[str, Any], deterministic: dict[str, Any] | None, replay: dict[str, Any] | None, historical: dict[str, Any] | None) -> dict[str, Any]:
    comparator_tokens = set()
    for surface in [deterministic, replay, historical]:
        if surface:
            comparator_tokens |= set(surface["component_name_tokens"])
    live_tokens = set(live["component_name_tokens"])
    missing = sorted(token for token in comparator_tokens - live_tokens if token)
    major_preserved = "yes" if not missing else ("partial" if len(missing) <= 2 else "no")
    multi_component_preserved = "yes" if int(live["multi_component_formulation_count"]) > 0 else "no"
    obvious_component_loss = "yes" if len(missing) >= 2 else "no"
    note_parts = [
        f"live_component_names={len(live_tokens)}",
        f"live_multi_component_formulations={live['multi_component_formulation_count']}",
    ]
    if missing:
        note_parts.append(f"missing_vs_comparators={','.join(missing[:6])}")
    return {
        "paper_key": paper_key,
        "major_reported_components_preserved": major_preserved,
        "multi_component_phases_preserved": multi_component_preserved,
        "obvious_component_loss": obvious_component_loss,
        "note": split_notes(note_parts),
    }


def build_expression_review(paper_key: str, live: dict[str, Any], deterministic: dict[str, Any] | None, replay: dict[str, Any] | None) -> dict[str, Any]:
    live_expr_types = {kind for kind, count in live["expression_kind_counts"].items() if count}
    comparator_expr_types = set()
    for surface in [deterministic, replay]:
        if surface:
            comparator_expr_types |= {kind for kind, count in surface["expression_kind_counts"].items() if count}
    missing_expr_types = sorted(kind for kind in comparator_expr_types - live_expr_types if kind in {"mass", "volume", "concentration", "ratio"})
    raw_expression_status = "yes" if not missing_expr_types else ("partial" if len(missing_expr_types) <= 1 else "no")
    raw_units_status = "yes" if live["raw_units_preserved"] else "no"
    multiple_expr_status = "yes" if live["multiple_expressions_preserved"] else "not_observed"
    note_parts = [
        f"live_expression_types={','.join(sorted(live_expr_types)) or 'none'}",
        f"live_unit_bearing_expressions={live['unit_bearing_expression_count']}",
    ]
    if missing_expr_types:
        note_parts.append(f"missing_expression_types_vs_comparators={','.join(missing_expr_types)}")
    return {
        "paper_key": paper_key,
        "raw_mass_volume_concentration_preserved": raw_expression_status,
        "raw_units_preserved": raw_units_status,
        "multiple_expressions_preserved_when_reported": multiple_expr_status,
        "note": split_notes(note_parts),
    }


def build_variable_review(paper_key: str, live: dict[str, Any], deterministic: dict[str, Any] | None) -> dict[str, Any]:
    live_vars = set(live["variable_names"])
    comparator_vars = set(deterministic["variable_names"]) if deterministic else set()
    live_explicit = bool(live_vars & (DOE_TOKENS | comparator_vars))
    incorrect_promotions = sorted(live["incorrect_variable_promotions"])
    note_parts = [
        f"live_variables={','.join(sorted(live_vars)[:8]) or 'none'}",
    ]
    if deterministic:
        note_parts.append(f"deterministic_variables={','.join(sorted(comparator_vars)[:8]) or 'none'}")
    if incorrect_promotions:
        note_parts.append(f"variable_measurement_overlap={','.join(incorrect_promotions)}")
    return {
        "paper_key": paper_key,
        "article_explicit_variables_captured": "yes" if live_explicit else "no",
        "non_factor_conditions_incorrectly_promoted": "yes" if incorrect_promotions else "no",
        "note": split_notes(note_parts),
    }


def build_ambiguity_review(paper_key: str, live: dict[str, Any], boundary_row: dict[str, Any]) -> dict[str, Any]:
    ambiguity_preserved = live["ambiguity_signal_count"] > 0
    forced_interpretation = boundary_row["obvious_over_splitting"] == "yes" and live["ambiguity_signal_count"] == 0
    note_parts = [
        f"live_ambiguity_signals={live['ambiguity_signal_count']}",
        f"live_formulations={live['formulation_count']}",
    ]
    if boundary_row["obvious_under_splitting"] == "yes" and not ambiguity_preserved:
        note_parts.append("boundary compression occurred without explicit ambiguity handoff")
    if forced_interpretation:
        note_parts.append("surface expands structure without visible ambiguity markers")
    return {
        "paper_key": paper_key,
        "unclear_information_preserved_as_ambiguous": "yes" if ambiguity_preserved else "no",
        "evidence_of_forced_interpretation": "yes" if forced_interpretation else "no",
        "note": split_notes(note_parts),
    }


def grouped_rows_by_key(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(row.get("key"))].append(row)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a governed Stage2 v2 three-paper structural comparison pack.")
    parser.add_argument("--llm-v2-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--paper-key", action="append", dest="paper_keys", default=[])
    parser.add_argument("--current-semantic-jsonl", default="")
    parser.add_argument("--current-compat-tsv", default="")
    parser.add_argument("--historical-legacy-tsv", default=DEFAULT_HISTORICAL_TSV)
    parser.add_argument("--replay-v2-jsonl", default=DEFAULT_REPLAY_V2_JSONL)
    parser.add_argument("--gt-skeleton-tsv", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = with_default_path(args.out_dir)
    if out_dir is None:
        raise ValueError("--out-dir is required.")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_v2_jsonl = with_default_path(args.llm_v2_jsonl)
    if llm_v2_jsonl is None:
        raise ValueError("--llm-v2-jsonl is required.")

    source_run_context = resolve_run_context(
        explicit_run_dir=with_default_path(args.run_dir),
        explicit_run_id=normalize_text(args.run_id),
    )
    current_semantic_jsonl = resolve_artifact_path(
        explicit_path=with_default_path(args.current_semantic_jsonl),
        run_context=source_run_context,
        pointer_key="stage2_semantic_objects_jsonl",
        required=True,
    )
    current_compat_tsv = resolve_artifact_path(
        explicit_path=with_default_path(args.current_compat_tsv),
        run_context=source_run_context,
        pointer_key="stage2_compatibility_tsv",
        required=True,
    )
    gt_skeleton_tsv = resolve_artifact_path(
        explicit_path=with_default_path(args.gt_skeleton_tsv),
        run_context=source_run_context,
        pointer_key="gt_skeleton_tsv",
        required=True,
    )
    historical_legacy_tsv = with_default_path(args.historical_legacy_tsv)
    replay_v2_jsonl = with_default_path(args.replay_v2_jsonl)

    assert current_semantic_jsonl is not None
    assert current_compat_tsv is not None
    assert gt_skeleton_tsv is not None
    assert historical_legacy_tsv is not None

    print(f"resolved_source_run_dir={source_run_context['run_dir']}")
    print(f"resolved_llm_v2_jsonl={llm_v2_jsonl}")
    print(f"resolved_current_semantic_jsonl={current_semantic_jsonl}")
    print(f"resolved_current_compat_tsv={current_compat_tsv}")
    print(f"resolved_gt_skeleton_tsv={gt_skeleton_tsv}")
    print(f"resolved_historical_legacy_tsv={historical_legacy_tsv}")
    print(f"resolved_replay_v2_jsonl={replay_v2_jsonl if replay_v2_jsonl and replay_v2_jsonl.exists() else ''}")

    selected_keys = [normalize_text(key) for key in args.paper_keys if normalize_text(key)]

    llm_docs = [row for row in read_jsonl(llm_v2_jsonl) if not selected_keys or normalize_text(row.get("document_key")) in selected_keys]
    semantic_docs = [row for row in read_jsonl(current_semantic_jsonl) if not selected_keys or normalize_text(row.get("document_key")) in selected_keys]
    current_compat_rows = [
        row for row in read_tsv(current_compat_tsv) if not selected_keys or normalize_text(row.get("key")) in selected_keys
    ]
    historical_rows = [
        row for row in read_tsv(historical_legacy_tsv) if not selected_keys or normalize_text(row.get("key")) in selected_keys
    ]
    gt_rows = [
        row for row in read_tsv(gt_skeleton_tsv) if not selected_keys or normalize_text(row.get("paper_key") or row.get("key")) in selected_keys
    ]
    replay_docs: list[dict[str, Any]] = []
    if replay_v2_jsonl and replay_v2_jsonl.exists():
        replay_docs = [row for row in read_jsonl(replay_v2_jsonl) if not selected_keys or normalize_text(row.get("document_key")) in selected_keys]

    live_by_key = {item["paper_key"]: item for item in (inspect_llm_doc(doc, "stage2_v2_live_gemini") for doc in llm_docs)}
    deterministic_by_key = {item["paper_key"]: item for item in (inspect_current_semantic_doc(doc) for doc in semantic_docs)}
    replay_by_key = {item["paper_key"]: item for item in (inspect_llm_doc(doc, "stage2_v2_replay") for doc in replay_docs)}
    compat_by_key = {paper_key: inspect_widerow_rows(rows, "current_deterministic_compat_active_run") for paper_key, rows in grouped_rows_by_key(current_compat_rows).items()}
    historical_by_key = {paper_key: inspect_widerow_rows(rows, "historical_legacy_stage2_20260314") for paper_key, rows in grouped_rows_by_key(historical_rows).items()}
    gt_counts = count_gt_rows(gt_rows)

    ordered_keys = selected_keys or sorted(live_by_key)
    paper_level_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    expression_rows: list[dict[str, Any]] = []
    variable_rows: list[dict[str, Any]] = []
    ambiguity_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    legacy_boundary_rows: list[dict[str, Any]] = []
    legacy_variable_rows: list[dict[str, Any]] = []
    legacy_measurement_rows: list[dict[str, Any]] = []
    legacy_comparison_rows: list[dict[str, Any]] = []

    for paper_key in ordered_keys:
        live = live_by_key.get(paper_key)
        if not live:
            continue
        deterministic = deterministic_by_key.get(paper_key)
        replay = replay_by_key.get(paper_key)
        compat = compat_by_key.get(paper_key)
        historical = historical_by_key.get(paper_key)
        gt_count = int(gt_counts.get(paper_key, 0))

        paper_level_rows.append(
            {
                "paper_key": paper_key,
                "gt_formulation_count": gt_count,
                "stage2_v2_live_count": live["formulation_count"],
                "deterministic_count": deterministic["formulation_count"] if deterministic else "",
                "replay_count_if_available": replay["formulation_count"] if replay else "",
                "count_status": compare_count_status(
                    int(live["formulation_count"]),
                    gt_count,
                    int(deterministic["formulation_count"]) if deterministic else 0,
                    str(replay["formulation_count"]) if replay else "",
                ),
            }
        )

        boundary_row = build_boundary_review(paper_key, live, gt_count, deterministic, replay, historical)
        component_row = build_component_review(paper_key, live, deterministic, replay, historical)
        expression_row = build_expression_review(paper_key, live, deterministic, replay)
        variable_row = build_variable_review(paper_key, live, deterministic)
        ambiguity_row = build_ambiguity_review(paper_key, live, boundary_row)

        boundary_rows.append(boundary_row)
        component_rows.append(component_row)
        expression_rows.append(expression_row)
        variable_rows.append(variable_row)
        ambiguity_rows.append(ambiguity_row)

        boundary_status = boundary_row["aligns_with_reported_boundaries"]
        component_status = component_row["major_reported_components_preserved"]
        expression_status = expression_row["raw_mass_volume_concentration_preserved"]
        variable_status = variable_row["article_explicit_variables_captured"]
        ambiguity_status = ambiguity_row["unclear_information_preserved_as_ambiguous"]
        summary_rows.append(
            {
                "paper_key": paper_key,
                "boundary_status": boundary_status,
                "component_status": component_status,
                "expression_status": expression_status,
                "variable_status": variable_status,
                "ambiguity_status": ambiguity_status,
                "overall_structural_judgment": overall_status(
                    boundary_status,
                    component_status,
                    expression_status,
                    variable_status,
                    ambiguity_status,
                ),
            }
        )

        all_surfaces = [live, deterministic, replay, compat, historical]
        llm_count = int(live["formulation_count"])
        llm_multi = int(live["multi_component_formulation_count"])
        for surface in [item for item in all_surfaces if item]:
            surface_name = str(surface["surface_name"])
            if surface_name != "stage2_v2_live_gemini":
                delta = int(surface["formulation_count"] or 0) - llm_count
                legacy_boundary_rows.append(
                    {
                        "paper_key": paper_key,
                        "surface_name": surface_name,
                        "llm_v2_formulation_count": llm_count,
                        "surface_formulation_count": surface["formulation_count"],
                        "boundary_delta_vs_llm_v2": delta,
                        "boundary_agreement": "exact_count_match" if delta == 0 else ("surface_over_llm_v2" if delta > 0 else "surface_under_llm_v2"),
                    }
                )
            legacy_variable_rows.append(
                {
                    "paper_key": paper_key,
                    "surface_name": surface_name,
                    "ph_retained": "yes" if surface["ph_retained"] else "no",
                    "doe_factor_retained": "yes" if surface["doe_factor_retained"] else "no",
                    "doe_signal_count": sum(1 for name in surface["variable_names"] if name in DOE_TOKENS),
                }
            )
            legacy_measurement_rows.append(
                {
                    "paper_key": paper_key,
                    "surface_name": surface_name,
                    "pdi_retained": "yes" if surface["pdi_retained"] else "no",
                    "zeta_retained": "yes" if surface["zeta_retained"] else "no",
                    "multi_component_formulation_count": surface["multi_component_formulation_count"],
                }
            )
            legacy_comparison_rows.append(
                {
                    "paper_key": paper_key,
                    "surface_name": surface_name,
                    "formulation_count": surface["formulation_count"],
                    "boundary_status_vs_llm_v2": "reference_surface"
                    if surface_name == "stage2_v2_live_gemini"
                    else ("exact_count_match" if int(surface["formulation_count"] or 0) == llm_count else ("surface_over_llm_v2" if int(surface["formulation_count"] or 0) > llm_count else "surface_under_llm_v2")),
                    "variable_retention_signature": f"ph={'yes' if surface['ph_retained'] else 'no'}; doe={'yes' if surface['doe_factor_retained'] else 'no'}",
                    "measurement_retention_signature": f"pdi={'yes' if surface['pdi_retained'] else 'no'}; zeta={'yes' if surface['zeta_retained'] else 'no'}",
                    "multi_component_delta_vs_llm_v2": ""
                    if surface_name == "stage2_v2_live_gemini"
                    else int(surface["multi_component_formulation_count"] or 0) - llm_multi,
                    "obvious_drift_note": split_notes(surface["notes"]) or "comparison_surface",
                }
            )

    write_tsv(
        out_dir / PAPER_LEVEL_COUNTS,
        ["paper_key", "gt_formulation_count", "stage2_v2_live_count", "deterministic_count", "replay_count_if_available", "count_status"],
        paper_level_rows,
    )
    write_tsv(
        out_dir / BOUNDARY_REVIEW,
        ["paper_key", "aligns_with_reported_boundaries", "obvious_under_splitting", "obvious_over_splitting", "note"],
        boundary_rows,
    )
    write_tsv(
        out_dir / COMPONENT_COMPLETENESS_REVIEW,
        ["paper_key", "major_reported_components_preserved", "multi_component_phases_preserved", "obvious_component_loss", "note"],
        component_rows,
    )
    write_tsv(
        out_dir / EXPRESSION_RICHNESS_REVIEW,
        ["paper_key", "raw_mass_volume_concentration_preserved", "raw_units_preserved", "multiple_expressions_preserved_when_reported", "note"],
        expression_rows,
    )
    write_tsv(
        out_dir / VARIABLE_DETECTION_REVIEW,
        ["paper_key", "article_explicit_variables_captured", "non_factor_conditions_incorrectly_promoted", "note"],
        variable_rows,
    )
    write_tsv(
        out_dir / AMBIGUITY_HANDLING_REVIEW,
        ["paper_key", "unclear_information_preserved_as_ambiguous", "evidence_of_forced_interpretation", "note"],
        ambiguity_rows,
    )
    write_tsv(
        out_dir / STRUCTURAL_SUMMARY,
        ["paper_key", "boundary_status", "component_status", "expression_status", "variable_status", "ambiguity_status", "overall_structural_judgment"],
        summary_rows,
    )

    write_tsv(
        out_dir / BOUNDARY_SURFACE,
        ["paper_key", "surface_name", "llm_v2_formulation_count", "surface_formulation_count", "boundary_delta_vs_llm_v2", "boundary_agreement"],
        legacy_boundary_rows,
    )
    write_tsv(
        out_dir / VARIABLE_RETENTION,
        ["paper_key", "surface_name", "ph_retained", "doe_factor_retained", "doe_signal_count"],
        legacy_variable_rows,
    )
    write_tsv(
        out_dir / MEASUREMENT_RETENTION,
        ["paper_key", "surface_name", "pdi_retained", "zeta_retained", "multi_component_formulation_count"],
        legacy_measurement_rows,
    )
    write_tsv(
        out_dir / COMPARISON_SUMMARY,
        [
            "paper_key",
            "surface_name",
            "formulation_count",
            "boundary_status_vs_llm_v2",
            "variable_retention_signature",
            "measurement_retention_signature",
            "multi_component_delta_vs_llm_v2",
            "obvious_drift_note",
        ],
        legacy_comparison_rows,
    )

    report_lines = [
        "# Stage2 v2 Three-Paper Structural Comparison Report",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- report_scope: `diagnostic-only Stage2 semantic-intermediate architecture-validation slice; not benchmark-valid final-table evaluation`",
        f"- resolved_source_run_id: `{source_run_context['run_id']}`",
        f"- resolved_source_run_dir: `{source_run_context['run_dir']}`",
        f"- llm_v2_live_source: `{llm_v2_jsonl}`",
        f"- current_semantic_source: `{current_semantic_jsonl}`",
        f"- current_compat_source: `{current_compat_tsv}`",
        f"- replay_source: `{replay_v2_jsonl}`" if replay_v2_jsonl and replay_v2_jsonl.exists() else "- replay_source: `not_available`",
        f"- historical_legacy_source: `{historical_legacy_tsv}`",
        f"- gt_boundary_source: `{gt_skeleton_tsv}`",
        "",
        "## Executive Conclusion",
        "",
    ]

    total_go = sum(1 for row in summary_rows if row["overall_structural_judgment"] == "go")
    total_caution = sum(1 for row in summary_rows if row["overall_structural_judgment"] == "caution")
    recommendation = "continue Stage2 v2" if total_go == len(summary_rows) and summary_rows else "continue with prompt revision"
    if any(row["boundary_status"] == "no" for row in summary_rows):
        recommendation = "continue with prompt revision"
    if sum(1 for row in summary_rows if row["boundary_status"] == "no") >= 2:
        recommendation = "stop and redesign"
    report_lines.extend(
        [
            "### Facts",
            "",
            f"- Papers reviewed: `{', '.join(ordered_keys)}`",
            f"- Per-paper overall judgments: `go={total_go}` `caution={total_caution}`",
            f"- Recommendation candidate: `{recommendation}`",
            "",
            "### Inference",
            "",
            "- This report evaluates the Stage2 semantic-discovery intermediate only, not the completed Stage2 output consumed by Stage3.",
            "- Boundary stability is weighted more heavily than count parity alone.",
            "- Expression preservation and ambiguity honesty are treated as stronger indicators of contract alignment than wide-row convenience.",
            "",
            "### Uncertainty",
            "",
            "- This report is limited to three papers and does not establish promotion readiness.",
            "- The current Stage2 v2 object schema still constrains some richer multi-expression representation, so favorable results here do not imply the schema is fully mature.",
            "",
            "## Per-Paper Review",
            "",
        ]
    )

    boundary_by_key = {row["paper_key"]: row for row in boundary_rows}
    component_by_key = {row["paper_key"]: row for row in component_rows}
    expression_by_key = {row["paper_key"]: row for row in expression_rows}
    variable_by_key = {row["paper_key"]: row for row in variable_rows}
    ambiguity_by_key = {row["paper_key"]: row for row in ambiguity_rows}
    summary_by_key = {row["paper_key"]: row for row in summary_rows}

    for paper_key in ordered_keys:
        if paper_key not in summary_by_key:
            continue
        report_lines.extend(
            [
                f"### {paper_key}",
                "",
                "#### Facts",
                "",
                f"- Count summary: `{next((row['count_status'] for row in paper_level_rows if row['paper_key'] == paper_key), '')}`",
                f"- Boundary review: `{boundary_by_key[paper_key]['note']}`",
                f"- Component review: `{component_by_key[paper_key]['note']}`",
                f"- Expression review: `{expression_by_key[paper_key]['note']}`",
                f"- Variable review: `{variable_by_key[paper_key]['note']}`",
                f"- Ambiguity review: `{ambiguity_by_key[paper_key]['note']}`",
                "",
                "#### Inference",
                "",
                f"- Overall structural judgment: `{summary_by_key[paper_key]['overall_structural_judgment']}`",
                "",
                "#### Uncertainty",
                "",
            "- This paper-level judgment is based on semantic-intermediate surfaces and comparator behavior, not a full evidence-audited benchmark pass or completed Stage2 evaluation.",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Final Recommendation",
            "",
            f"- {recommendation}",
            "",
            "## Guardrail",
            "",
            "- This pack compares raw Stage2 semantic-intermediate behavior only.",
            "- It must not be reported as final system benchmark evidence.",
            "- It must not be reported as the authoritative completed-Stage2 evaluation object.",
            "- Deterministic semantic surfaces in this pack remain comparator surfaces and do not supersede the frozen Stage2 authority split.",
            "",
        ]
    )

    structural_report_path = out_dir / STRUCTURAL_REPORT
    structural_report_path.write_text("\n".join(report_lines), encoding="utf-8")
    (out_dir / COMPARISON_REPORT).write_text("\n".join(report_lines), encoding="utf-8")

    metadata = build_artifact_metadata(
        source_run_context=source_run_context,
        source_files={
            "llm_v2_jsonl": str(llm_v2_jsonl),
            "current_semantic_jsonl": str(current_semantic_jsonl),
            "current_compat_tsv": str(current_compat_tsv),
            "historical_legacy_tsv": str(historical_legacy_tsv),
            "replay_v2_jsonl": str(replay_v2_jsonl) if replay_v2_jsonl and replay_v2_jsonl.exists() else "",
            "gt_skeleton_tsv": str(gt_skeleton_tsv),
        },
        generated_by="src/analysis/build_stage2_v2_threepaper_comparison_pack.py",
        note="Diagnostic-only three-paper structural comparison pack for Stage2 v2 live-vs-comparator evaluation. Not benchmark-valid final output.",
        extra={"paper_keys": ordered_keys},
    )
    for target in [
        out_dir / PAPER_LEVEL_COUNTS,
        out_dir / BOUNDARY_REVIEW,
        out_dir / COMPONENT_COMPLETENESS_REVIEW,
        out_dir / EXPRESSION_RICHNESS_REVIEW,
        out_dir / VARIABLE_DETECTION_REVIEW,
        out_dir / AMBIGUITY_HANDLING_REVIEW,
        out_dir / STRUCTURAL_SUMMARY,
        structural_report_path,
        out_dir / COMPARISON_SUMMARY,
        out_dir / COMPARISON_REPORT,
    ]:
        write_artifact_metadata_json(target, metadata)

    print(f"wrote_out_dir={out_dir}")
    print(f"wrote_report={structural_report_path}")


if __name__ == "__main__":
    main()
