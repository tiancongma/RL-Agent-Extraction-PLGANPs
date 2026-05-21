#!/usr/bin/env python3
"""Evaluate the governed S2-4a Layer A Hard Gate from frozen artifacts only.

This helper is read-only. It does not modify pipeline outputs or selector
behavior. It exists to apply the governed Hard Gate contract against explicit
artifact surfaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        build_prompt_render_bundle,
    )
except ModuleNotFoundError:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        build_prompt_render_bundle,
    )


ALLOWED_SUMMARY_TABLE_SOURCES = {"table_summary", "table_excerpt"}
TABLE_SOURCE_PREFIX = "table"
NEGATIVE_TABLE_SOURCE_ROLES = {
    "characterization_result_table",
    "release_profile_table",
    "pharmacokinetic_table",
    "tissue_distribution_table",
    "reference_spillover_table",
    "noise_or_nonformulation_table",
}
POSITIVE_TABLE_SOURCE_ROLES = {
    "formulation_composition_table",
    "preparation_parameter_table",
    "optimization_table",
    "parameter_sweep_table",
}
POSITIVE_TABLE_TYPES = {
    "formulation_table",
    "doe_table",
    "mixed_table",
    "optimization_table",
    "parameter_sweep_table",
}
TEXTUAL_FORMULATION_PHRASES = (
    "different formulations",
    "formulations developed",
    "formulation number",
    "characterization of the different formulations",
)
TEXTUAL_COMPONENT_TERMS = (
    "formulation",
    "polymer",
    "surfactant",
    "peg",
    "plga",
    "microsphere",
    "microspheres",
    "scaffold",
    "scaffolds",
    "emulsion",
    "emulsions",
    "nanofiber",
    "nanofibers",
    "nanoparticle",
    "nanoparticles",
    "np",
    "nps",
    "microparticle",
    "microparticles",
    "drug",
    "dna",
)
PREPARATION_CORE_POLYMER_PATTERN = re.compile(
    r"\b(?:s?plga|peg-plga|pla|pcl|pdlla|polymer|poly\s*\(?lactide-co-glycolide\)?|poly\s*\(?lactic-co-glycolic acid\)?|poly\(lactide-co-glycolide\)|poly\(lactic-co-glycolic acid\))\b",
    re.I,
)
PREPARATION_CORE_TARGET_PATTERN = re.compile(
    r"\b(?:nanoparticles?|nps?|microparticles?|microspheres?|nanospheres?|nanocapsules?|particles?|formulations?|nanocarriers?)\b",
    re.I,
)
PREPARATION_CORE_ACTION_PATTERN = re.compile(
    r"\b(?:prepared|fabricated|formulat(?:e|ed|ion)|develop(?:ed)?|synthesi[sz]ed|encapsulated|co[- ]?loaded|co[- ]?laded|loaded|dissolved|emulsified|sonicated|evaporat(?:ed|ion)|centrifuged|freeze[- ]?dried|stirred|added dropwise|nanoprecipitation|emulsion)\b",
    re.I,
)
PREPARATION_CORE_VALUE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|ug|µg|ml|mL|µL|ul|%|w/v|v/v|rpm|min|h|hr|hours?)\b",
    re.I,
)
PREPARATION_CORE_PHASE_PATTERN = re.compile(
    r"\b(?:oil phase|organic phase|aqueous phase|water phase|internal water phase|external phase|o/w|w/o/w|water-in-oil|oil-in-water|pva|polyvinyl alcohol|dcm|dichloromethane)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the governed S2-4a Layer A Hard Gate from explicit "
            "evidence_blocks, normalized_table_payloads, and prompt-audit "
            "artifacts."
        )
    )
    parser.add_argument(
        "--evidence-blocks-root",
        required=True,
        help="Root directory containing <paper_key>/evidence_blocks_v1.json.",
    )
    parser.add_argument(
        "--normalized-table-payloads-root",
        required=True,
        help=(
            "Root directory containing <paper_key>/normalized_table_payloads_v1.json."
        ),
    )
    parser.add_argument(
        "--prompt-audit-tsv",
        required=True,
        help="Path to analysis/s2_4a_prompt_audit_v1.tsv.",
    )
    parser.add_argument(
        "--paper-key",
        action="append",
        default=[],
        help="Optional paper key filter. Repeat to audit multiple papers.",
    )
    parser.add_argument(
        "--out-tsv",
        required=True,
        help="Output TSV path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_prompt_audit(path: Path) -> Dict[str, dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["paper_key"]: row for row in rows}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def matches_path2_text(text: str) -> bool:
    lowered = normalize_text(text)
    if any(phrase in lowered for phrase in TEXTUAL_FORMULATION_PHRASES):
        return True
    return bool(
        re.search(
            r"\bformulation(?:s)?\s+of\b.{0,120}\b(?:table|component|ratio|emulsion|scaffold|microsphere)",
            lowered,
        )
    )


def matches_path3_text(text: str) -> bool:
    lowered = normalize_text(text)
    prepared_targets = (
        "nanoparticle",
        "nanoparticles",
        "formulation",
        "formulations",
        "microsphere",
        "microspheres",
        "scaffold",
        "scaffolds",
        "emulsion",
        "emulsions",
        "nanofiber",
        "nanofibers",
        "nanoparticle",
        "nanoparticles",
        "np",
        "nps",
        "microparticle",
        "microparticles",
    )
    if any(action in lowered for action in ["prepared", "formulated", "formulate", "developed", "develop", "encapsulated", "loaded", "produced"]) and any(target in lowered for target in prepared_targets):
        matched_terms = sum(1 for term in TEXTUAL_COMPONENT_TERMS if term in lowered)
        if matched_terms >= 2:
            return True
    if "formulation" not in lowered:
        return False
    matched_terms = sum(1 for term in TEXTUAL_COMPONENT_TERMS if term in lowered)
    return matched_terms >= 3


def matches_path4_preparation_core_text(text: str) -> bool:
    """Detect source-backed PLGA-family preparation evidence without row creation."""
    if not (
        PREPARATION_CORE_POLYMER_PATTERN.search(text)
        and PREPARATION_CORE_ACTION_PATTERN.search(text)
        and PREPARATION_CORE_VALUE_PATTERN.search(text)
    ):
        return False
    if PREPARATION_CORE_TARGET_PATTERN.search(text):
        return True
    return bool(PREPARATION_CORE_PHASE_PATTERN.search(text))


def table_summary_text_has_formulation_or_design_surface(text: str) -> bool:
    """Detect structural formulation/design surfaces in neutral table summaries."""
    lowered = normalize_text(text)
    if not lowered:
        return False
    strong_structural_phrases = (
        "formulation a",
        "formulation b",
        "formulation c",
        "independent variables and their levels",
        "coded levels of factors",
        "factorial points",
        "box-behnken",
        "theoretical drug loading",
        "polymer ratio",
        "plga:e-rlpo",
        "external phase",
        "aqueous phase",
        "oil phase",
        "formulation parameters",
        "preparation formulation",
        "inline formulation table",
        "column_schema: formulation",
        "preparation of plgnps",
        "schematic illustration for the preparation",
        "nanoparticles coated with plga",
    )
    if any(phrase in lowered for phrase in strong_structural_phrases):
        return True
    has_formulation_axis = bool(re.search(r"\bformulation\s+[a-z0-9]+", lowered))
    has_composition_axis = sum(
        1
        for token in ("plga", "polymer", "pva", "drug", "loading", "phase", "ratio")
        if token in lowered
    ) >= 2
    has_metric_table_axis = (
        "encapsulation efficiency" in lowered
        and "particle size" in lowered
        and ("sample" in lowered or "formulation" in lowered)
    )
    return (has_formulation_axis and has_composition_axis) or has_metric_table_axis


def table_summary_satisfies_pre_live_evidence(block: dict, payload: dict) -> bool:
    """Return whether a rendered table summary is enough for pre-live sufficiency.

    This gate is deliberately not a formulation-authority classifier. Stage2
    semantic authority remains with the LLM; S2-4a may only check that the
    prompt has a compact, source-backed table evidence surface. Optional
    row-local/characterization payloads can satisfy pre-live evidence readiness
    when the full table authority is visible and non-hard-dropped.
    """
    if block.get("source_type") not in ALLOWED_SUMMARY_TABLE_SOURCES:
        return False
    block_text = str(block.get("text_content") or block.get("text") or "")
    if not payload:
        return table_summary_text_has_formulation_or_design_surface(block_text)
    if payload.get("table_inclusion_class") == "hard_drop":
        return False
    if normalize_text(str(payload.get("payload_authority_status", ""))) == "unusable_broken_payload":
        return False
    data_row_count = int(payload.get("data_row_count") or 0)
    representation_status = normalize_text(str(payload.get("representation_status", "")))
    source_kind = normalize_text(str(payload.get("source_table_asset_origin", "")))
    table_type = normalize_text(str(payload.get("table_type", "")))
    inclusion_class = normalize_text(str(payload.get("table_inclusion_class", "")))
    source_role = normalize_text(str(payload.get("table_source_role", "")))
    prompt_role = normalize_text(str(payload.get("table_role_hint", "")))
    if data_row_count < 2 and not (
        data_row_count >= 1
        and (
            representation_status == "text_metric_table_recovered"
            or source_kind == "clean_text_metric_sentence"
            or (
                table_type in {"formulation_table", "doe_table", "mixed_table"}
                and inclusion_class == "must_include"
            )
        )
    ):
        return False
    if representation_status == "text_metric_table_recovered" or source_kind == "clean_text_metric_sentence":
        return True
    if table_summary_text_has_formulation_or_design_surface(block_text):
        return True
    if source_role in NEGATIVE_TABLE_SOURCE_ROLES:
        return False
    if source_role in POSITIVE_TABLE_SOURCE_ROLES:
        return True
    if table_type in POSITIVE_TABLE_TYPES and inclusion_class == "must_include":
        return True
    if prompt_role in {"formulation", "design matrix", "optimization"}:
        return True
    return False


def iter_target_papers(
    evidence_blocks_root: Path,
    normalized_root: Path,
    prompt_audit: Dict[str, dict],
    selected_papers: Sequence[str],
) -> List[str]:
    discovered = {
        path.parent.name
        for path in evidence_blocks_root.glob("*/evidence_blocks_v1.json")
    }
    discovered.update(
        {
            path.parent.name
            for path in normalized_root.glob("*/normalized_table_payloads_v1.json")
        }
    )
    discovered.update(prompt_audit.keys())
    if selected_papers:
        return [paper for paper in selected_papers if paper in discovered]
    return sorted(discovered)


def evaluate_paper(
    paper_key: str,
    evidence_root: Path,
    normalized_root: Path,
    prompt_audit_row: dict | None,
) -> dict:
    evidence_path = evidence_root / paper_key / "evidence_blocks_v1.json"
    normalized_path = (
        normalized_root / paper_key / "normalized_table_payloads_v1.json"
    )
    evidence = load_json(evidence_path) if evidence_path.exists() else {}
    normalized = load_json(normalized_path) if normalized_path.exists() else {}

    evidence_blocks = evidence.get("evidence_blocks", [])
    payloads = normalized.get("normalized_table_payloads", [])
    payload_by_table_id = {row.get("table_id"): row for row in payloads}

    method_blocks = [
        block
        for block in evidence_blocks
        if block.get("source_type") in {"clean_text_paragraph", "method_text"}
        and "__method__" in block.get("block_id", "")
    ]
    text_blocks = [
        block
        for block in evidence_blocks
        if block.get("source_type")
        in {"clean_text_paragraph", "method_text", "materials_text", "supporting_text"}
    ]
    table_blocks = [
        block
        for block in evidence_blocks
        if block.get("is_table_derived")
        or str(block.get("source_type", "")).startswith(TABLE_SOURCE_PREFIX)
    ]
    table_summary_blocks = [
        block
        for block in table_blocks
        if block.get("source_type") in ALLOWED_SUMMARY_TABLE_SOURCES
    ]
    non_summary_table_blocks = [
        block
        for block in table_blocks
        if block.get("source_type") not in ALLOWED_SUMMARY_TABLE_SOURCES
    ]

    must_include_payloads = [
        row for row in payloads if row.get("table_inclusion_class") == "must_include"
    ]
    must_include_table_ids = {row.get("table_id") for row in must_include_payloads}
    preserved_table_ids = {block.get("table_id") for block in table_blocks}
    missing_must_include = sorted(
        table_id
        for table_id in must_include_table_ids
        if table_id and table_id not in preserved_table_ids
    )

    path1_matches: List[str] = []
    for block in table_summary_blocks:
        table_id = block.get("table_id")
        payload = payload_by_table_id.get(table_id, {})
        if table_summary_satisfies_pre_live_evidence(block, payload):
            path1_matches.append(table_id)
    path1 = bool(path1_matches)

    text_candidates = [
        normalize_text(block.get("text_content", "")) for block in text_blocks
    ]
    method_text_candidates = [
        normalize_text(block.get("text_content", "")) for block in method_blocks
    ]
    method_like_text_candidates = method_text_candidates or text_candidates
    has_method_evidence = bool(method_blocks) or any(
        matches_path3_text(text) or matches_path4_preparation_core_text(text)
        for text in text_candidates
    )
    has_materials_evidence = any(
        "__materials__" in block.get("block_id", "") for block in evidence_blocks
    )
    has_supporting_evidence = any(
        "__supporting__" in block.get("block_id", "") for block in evidence_blocks
    )
    path2 = (not path1) and has_method_evidence and any(
        matches_path2_text(text) for text in text_candidates
    )
    path3 = (not path1) and (not path2) and has_method_evidence and any(
        matches_path3_text(text) for text in text_candidates
    )
    if len(method_text_candidates) > 1:
        method_text_candidates.append(" ".join(method_text_candidates))
    path4 = (not path1) and (not path2) and (not path3) and has_method_evidence and any(
        matches_path4_preparation_core_text(text) for text in method_like_text_candidates
    )
    minimum_sufficiency_pass = path1 or path2 or path3 or path4

    prompt_render_bundle = build_prompt_render_bundle(evidence)
    rendered_block_ids = {
        normalize_text(block.get("block_id"))
        for block in prompt_render_bundle.get("rendered_blocks", [])
        if normalize_text(block.get("block_id"))
    }
    non_summary_prompt_table_blocks = [
        block
        for block in non_summary_table_blocks
        if normalize_text(block.get("block_id")) in rendered_block_ids
    ]
    summary_only_pass = not non_summary_prompt_table_blocks

    selector_boundary_pass = not missing_must_include
    if prompt_audit_row and prompt_audit_row.get("all_selected_blocks_included") != "yes":
        selector_boundary_pass = False

    prompt_legality_failures: List[str] = []
    if prompt_audit_row:
        if prompt_audit_row.get("uses_evidence_pack_only") != "yes":
            prompt_legality_failures.append("uses_evidence_pack_only")
        if prompt_audit_row.get("live_prompt_contains_runtime_metadata") != "no":
            prompt_legality_failures.append("live_prompt_contains_runtime_metadata")
        if prompt_audit_row.get("truncation_detected") != "no":
            prompt_legality_failures.append("truncation_detected")
        if int(prompt_audit_row.get("exact_duplicate_block_count") or 0) > 0:
            prompt_legality_failures.append("exact_duplicate_block_count")
        if prompt_audit_row.get("prompt_size_policy_status") != "healthy":
            prompt_legality_failures.append("prompt_size_policy_status")
    prompt_legality_pass = not prompt_legality_failures

    failure_labels: List[str] = []
    if not summary_only_pass:
        failure_labels.append("summary_contract_violation")
    if not selector_boundary_pass:
        failure_labels.append("selector_boundary_violation")
    if not prompt_legality_pass:
        failure_labels.append("prompt_inflation")
    if not minimum_sufficiency_pass and not non_summary_prompt_table_blocks:
        if must_include_payloads and not table_summary_blocks:
            failure_labels.append("table_missing")
        else:
            failure_labels.append("evidence_underselected")

    hard_gate_pass = (
        minimum_sufficiency_pass
        and summary_only_pass
        and selector_boundary_pass
        and prompt_legality_pass
    )

    satisfied_paths = []
    if path1:
        satisfied_paths.append("path1")
    if path2:
        satisfied_paths.append("path2")
    if path3:
        satisfied_paths.append("path3")
    if path4:
        satisfied_paths.append("path4")

    return {
        "paper_key": paper_key,
        "hard_gate_pass": "yes" if hard_gate_pass else "no",
        "failure_labels": "|".join(failure_labels),
        "satisfied_paths": "|".join(satisfied_paths),
        "has_method_evidence": "yes" if has_method_evidence else "no",
        "has_materials_evidence": "yes" if has_materials_evidence else "no",
        "has_supporting_evidence": "yes" if has_supporting_evidence else "no",
        "minimum_sufficiency_pass": "yes" if minimum_sufficiency_pass else "no",
        "path1_table_evidence_summary": "yes" if path1 else "no",
        "path1_table_ids": "|".join(path1_matches),
        "path2_method_plus_table_adjacent_text": "yes" if path2 else "no",
        "path3_method_plus_explicit_formulation_text": "yes" if path3 else "no",
        "path4_preparation_core_text": "yes" if path4 else "no",
        "summary_only_pass": "yes" if summary_only_pass else "no",
        "selector_boundary_pass": "yes" if selector_boundary_pass else "no",
        "prompt_legality_pass": "yes" if prompt_legality_pass else "no",
        "table_summary_count": str(len(table_summary_blocks)),
        "must_include_payload_count": str(len(must_include_payloads)),
        "must_include_preserved_count": str(len(must_include_table_ids - set(missing_must_include))),
        "missing_must_include_table_ids": "|".join(missing_must_include),
        "non_summary_table_block_ids": "|".join(
            block.get("block_id", "") for block in non_summary_prompt_table_blocks
        ),
        "prompt_legality_failures": "|".join(prompt_legality_failures),
        "prompt_length": (prompt_audit_row or {}).get("prompt_length", ""),
        "ordered_block_order": (prompt_audit_row or {}).get("ordered_block_order", ""),
    }


def write_tsv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_key",
        "hard_gate_pass",
        "failure_labels",
        "satisfied_paths",
        "has_method_evidence",
        "has_materials_evidence",
        "has_supporting_evidence",
        "minimum_sufficiency_pass",
        "path1_table_evidence_summary",
        "path1_table_ids",
        "path2_method_plus_table_adjacent_text",
        "path3_method_plus_explicit_formulation_text",
        "path4_preparation_core_text",
        "summary_only_pass",
        "selector_boundary_pass",
        "prompt_legality_pass",
        "table_summary_count",
        "must_include_payload_count",
        "must_include_preserved_count",
        "missing_must_include_table_ids",
        "non_summary_table_block_ids",
        "prompt_legality_failures",
        "prompt_length",
        "ordered_block_order",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    evidence_root = Path(args.evidence_blocks_root)
    normalized_root = Path(args.normalized_table_payloads_root)
    prompt_audit = load_prompt_audit(Path(args.prompt_audit_tsv))
    paper_keys = iter_target_papers(
        evidence_root,
        normalized_root,
        prompt_audit,
        args.paper_key,
    )
    rows = [
        evaluate_paper(
            paper_key=paper_key,
            evidence_root=evidence_root,
            normalized_root=normalized_root,
            prompt_audit_row=prompt_audit.get(paper_key),
        )
        for paper_key in paper_keys
    ]
    write_tsv(Path(args.out_tsv), rows)


if __name__ == "__main__":
    main()
