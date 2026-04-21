#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import ast
import csv
from functools import lru_cache
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from threading import Event, Thread
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

try:
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise
    from src.utils.paths import PROJECT_ROOT
    from src.stage2_sampling_labels.table_row_expansion_v1 import (
        EXECUTION_READY_MARKER,
        PARTIAL_SEMANTIC_MARKER,
        SCOPE_KIND as TABLE_FORMULATION_SCOPE_KIND,
        augment_document_with_table_markers,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise
    from src.utils.paths import PROJECT_ROOT
    from src.stage2_sampling_labels.table_row_expansion_v1 import (
        EXECUTION_READY_MARKER,
        PARTIAL_SEMANTIC_MARKER,
        SCOPE_KIND as TABLE_FORMULATION_SCOPE_KIND,
        augment_document_with_table_markers,
    )


OUTPUT_JSONL_NAME = "semantic_stage2_v2_objects.jsonl"
OUTPUT_SUMMARY_NAME = "semantic_stage2_v2_summary.tsv"
EVIDENCE_BLOCKS_FILENAME = "evidence_blocks_v1.json"
EVIDENCE_BLOCKS_SUBDIR = "evidence_blocks"
CANDIDATE_BLOCKS_FILENAME = "candidate_blocks_v1.json"
CANDIDATE_BLOCKS_SUBDIR = "candidate_blocks"
NORMALIZED_TABLE_PAYLOADS_FILENAME = "normalized_table_payloads_v1.json"
NORMALIZED_TABLE_PAYLOADS_SUBDIR = "normalized_table_payloads"
TABLE_AUTHORITY_VALIDATION_NAME = "table_authority_validation_v1.tsv"
PROMPT_PREVIEW_NAME = "stage2_prompt_preview_v1.tsv"
S2_2_BOUNDARY_VALIDATION_NAME = "s2_2_boundary_validation.tsv"
S2_3_BOUNDARY_VALIDATION_NAME = "s2_3_boundary_validation.tsv"
TABLE_SELECTION_DEBUG_NAME = "table_selection_debug_v1.json"
CANDIDATE_SEGMENTATION_DEBUG_NAME = "candidate_segmentation_debug_v1.tsv"
REQUEST_SUMMARY_NAME = "request_summary.tsv"
REQUEST_METADATA_SUBDIR = "request_metadata"
REQUEST_METADATA_FILENAME_TEMPLATE = "{paper_key}__stage2_v2_request_metadata.json"
MARKER_CLEANUP_AUDIT_SUFFIX = "__marker_cleanup_audit.json"
NVIDIA_HOSTED_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LEGACY_FIELD_ALIASES = {"plga_mw_kDa": "polymer_mw_kDa"}
SUMMARY_FIRST_COLUMN_ENHANCEMENT_ENV = "STAGE2_TABLE_SUMMARY_FIRST_COLUMN_ENHANCEMENT"
INPUT_PACKING_MODE_ENV = "STAGE2_INPUT_EVIDENCE_PACKING_MODE"
ORDERED_INPUT_PACKING_MODE = "ordered_blocks"
SEGMENTATION_PROFILE = "section_aware_candidate_segmentation_v1"
COMPONENT_FIELD_SPECS = [
    ("polymer", "polymer_identity", "plga_mass_mg"),
    ("polymer", "polymer_name_raw", "plga_mass_mg"),
    ("drug", "drug_name", "drug_feed_amount_text"),
    ("surfactant", "surfactant_name", "surfactant_concentration_text"),
    ("surfactant", "surfactant_name", "pva_conc_percent"),
    ("organic_solvent", "organic_solvent", ""),
]
VARIABLE_FIELDS = [
    "emul_type",
    "emul_method",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "surfactant_concentration_text",
    "pva_conc_percent",
    "drug_feed_amount_text",
]
MEASUREMENT_FIELDS = [
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
]
PH_TOKENS = {"ph", "aqueous_phase_ph", "aqueous_ph"}
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
STAGE2_SEMANTIC_SOURCE_MODE = "llm_first_composite"
LLM_SEMANTIC_DISCOVERY = "llm_semantic_discovery"
LLM_DECLARED_SCOPE = "llm_declared_scope"
LLM_EXPLICIT = "llm_explicit"
LLM_PARSED = "llm_parsed"
DOE_SCOPE_KIND = "doe_table_row_enumeration_scope"
DOCUMENT_SCOPE_KIND = "document_semantic_scope"
EVIDENCE_SELECTION_MODE = "evidence_priority_v1"
PROMPT_HEALTHY_CHAR_LIMIT = 30000
EVIDENCE_KIND_ORDER = {
    "method": 0,
    "materials": 1,
    "table": 2,
    "supporting": 3,
}
PROCUREMENT_CUES = [
    "purchased from",
    "obtained from",
    "supplied by",
    "procured",
    "analytical grade",
    "hplc grade",
]
PREPARATION_PROCEDURE_CUES = [
    "nanoprecipitation",
    "dissolved",
    "acetone",
    "organic phase",
    "aqueous phase",
    "added",
    "added dropwise",
    "stirring",
    "evaporation",
    "under vacuum",
    "filtered",
    "centrifuged",
    "washed",
]
ASSAY_COMPARATOR_CUES = [
    "lc-ms",
    "lc-ms/ms",
    "hplc",
    "pharmacokinetic",
    "pharmacokinetics",
    "rat plasma",
    "bioequivalent",
    "sporanox",
    "mean residence time",
    "clearance",
    "auc",
    "jugular",
]
OPTIMIZATION_DECISION_CUES = [
    "chosen for the preparation",
    "chosen as the optimal formulation",
    "chosen as optimal",
    "during the whole study",
    "utilized for the formulation of all the following studies",
    "selected as optimal",
]
PREPARATION_NEGATIVE_CUES = [
    "cell viability",
    "biodistribution",
    "radiolabeling",
    "imaging",
    "pharmacoscintigraphy",
    "ex vivo release",
    "stability study",
]
MATERIALS_NEGATIVE_CUES = [
    "prepared by",
    "stirred",
    "centrifuged",
    "dropwise",
]
TABLE_ROW_ID_PATTERNS = [
    r"\bf[-\s]?\d{1,3}\b",
    r"\brun\s*\d{1,3}\b",
    r"\bnp[a-z]{1,3}\d{1,3}\b",
]
EVIDENCE_PRIORITY_THRESHOLDS = {
    "method": 5.0,
    "materials": 4.0,
    "table": 6.0,
    "supporting": 5.5,
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_token(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_marker_list(value: Any, *, default_provenance: str = LLM_EXPLICIT) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for item in ensure_list(value):
        if not isinstance(item, dict):
            continue
        marker = dict(item)
        provenance = normalize_text(marker.get("marker_provenance")) or default_provenance
        if provenance not in {LLM_EXPLICIT, LLM_PARSED}:
            provenance = default_provenance
        marker["marker_provenance"] = provenance
        markers.append(marker)
    return markers


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_text(value).lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return default


def normalize_confidence(value: Any) -> str:
    text = normalize_text(value).lower()
    return text if text in {"high", "medium", "low"} else "low"


def is_shrunken_live_contract(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    keys = set(parsed.keys())
    required = {"paper_key", "table_scopes", "semantic_signals", "formulation_candidates"}
    return required.issubset(keys)


def normalize_scope_kind(value: Any) -> str:
    text = normalize_text(value).lower()
    allowed = {
        "doe_table",
        "formulation_table",
        "optimization_table",
        "sequential_child",
        "downstream_variant_table",
        "non_formulation",
        "unclear",
    }
    return text if text in allowed else "unclear"


def normalize_candidate_kind(value: Any) -> str:
    text = normalize_text(value).lower()
    allowed = {
        "single_formulation",
        "formulation_family",
        "variant_formulation",
        "unclear",
    }
    return text if text in allowed else "unclear"


def normalize_instance_role(value: Any) -> str:
    text = normalize_text(value).lower()
    allowed = {
        "synthesis_core",
        "downstream_variant",
        "control",
        "comparative",
        "characterization_only",
        "unclear",
    }
    return text if text in allowed else "unclear"


def normalize_candidate_status(value: Any) -> str:
    text = normalize_text(value).lower()
    return text if text in {"reported", "partial", "ambiguous"} else "ambiguous"


def normalize_shrunken_table_scopes(value: Any) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for item in ensure_list(value):
        if not isinstance(item, dict):
            continue
        table_id = normalize_text(item.get("table_id"))
        if not table_id:
            continue
        scopes.append(
            {
                "table_id": table_id,
                "scope_kind": normalize_scope_kind(item.get("scope_kind")),
                "is_formulation_bearing": normalize_bool(item.get("is_formulation_bearing"), True),
                "is_doe": normalize_bool(item.get("is_doe"), False),
                "parent_table_hint": normalize_text(item.get("parent_table_hint")),
                "confidence": normalize_confidence(item.get("confidence")),
            }
        )
    return scopes


def normalize_shrunken_semantic_signals(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    return {
        "has_variable_sweep": normalize_bool(payload.get("has_variable_sweep"), False),
        "has_sequential_optimization": normalize_bool(payload.get("has_sequential_optimization"), False),
        "has_parent_child_table_relation": normalize_bool(payload.get("has_parent_child_table_relation"), False),
        "has_downstream_non_synthesis_variants": normalize_bool(payload.get("has_downstream_non_synthesis_variants"), False),
        "has_measurement_only_variants": normalize_bool(payload.get("has_measurement_only_variants"), False),
        "primary_preparation_method_hint": normalize_text(payload.get("primary_preparation_method_hint")),
        "primary_variable_names": [
            normalize_text(item)
            for item in ensure_list(payload.get("primary_variable_names"))
            if normalize_text(item)
        ],
        "selected_condition_hints": [
            normalize_text(item)
            for item in ensure_list(payload.get("selected_condition_hints"))
            if normalize_text(item)
        ],
    }


def normalize_shrunken_formulation_candidates(value: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in ensure_list(value):
        if not isinstance(item, dict):
            continue
        candidate_id = normalize_text(item.get("candidate_id"))
        if not candidate_id:
            continue
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": normalize_candidate_kind(item.get("candidate_kind")),
                "source_table_id": normalize_text(item.get("source_table_id")),
                "label_hint": normalize_text(item.get("label_hint")),
                "instance_role": normalize_instance_role(item.get("instance_role")),
                "parent_candidate_hint": normalize_text(item.get("parent_candidate_hint")),
                "core_change_hint": normalize_text(item.get("core_change_hint")),
                "shared_context_hint": normalize_text(item.get("shared_context_hint")),
                "status": normalize_candidate_status(item.get("status")),
                "confidence": normalize_confidence(item.get("confidence")),
            }
        )
    return candidates


def inheritance_marker_contract_issues(marker: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    readiness = normalize_text(marker.get("marker_readiness")) or EXECUTION_READY_MARKER
    if readiness not in {EXECUTION_READY_MARKER, PARTIAL_SEMANTIC_MARKER}:
        issues.append("invalid_marker_readiness")
        return issues
    for field in ["inherit_type", "variable", "value"]:
        if not normalize_text(marker.get(field)):
            issues.append(f"missing_{field}")
    if readiness == EXECUTION_READY_MARKER:
        if normalize_text(marker.get("risk_label")) or normalize_text(marker.get("risk_reason")):
            issues.append("execution_ready_carries_risk_fields")
    else:
        if normalize_text(marker.get("risk_label")) != "review":
            issues.append("partial_missing_review_risk_label")
        if normalize_text(marker.get("risk_reason")) not in {
            "missing_source_table",
            "missing_target_table",
            "cross_table_link_unresolved",
        }:
            issues.append("partial_invalid_risk_reason")
    return issues


def write_marker_cleanup_audit(raw_response_path: Path, audit: dict[str, Any]) -> None:
    audit_path = raw_response_path.with_name(raw_response_path.stem + MARKER_CLEANUP_AUDIT_SUFFIX)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prune_invalid_live_inheritance_markers(parsed: dict[str, Any], raw_response_path: Path) -> dict[str, Any]:
    cleaned = dict(parsed)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for idx, item in enumerate(ensure_list(parsed.get("inheritance_markers")), start=1):
        if not isinstance(item, dict):
            dropped.append({"index": idx, "reason": ["non_dict_marker"], "marker": item})
            continue
        issues = inheritance_marker_contract_issues(item)
        if issues:
            dropped.append({"index": idx, "reason": issues, "marker": item})
            continue
        kept.append(item)
    cleaned["inheritance_markers"] = kept
    if dropped:
        write_marker_cleanup_audit(
            raw_response_path,
            {
                "artifact_version": "v1",
                "source_raw_response_path": str(raw_response_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "cleanup_scope": "inheritance_markers",
                "dropped_marker_count": len(dropped),
                "dropped_markers": dropped,
            },
        )
    return cleaned


def canonical_field_name(field_name: str) -> str:
    return LEGACY_FIELD_ALIASES.get(field_name, field_name)


def _mechanically_sanitize_json_text(text: str) -> tuple[str, dict[str, Any]]:
    """
    Apply only low-risk mechanical cleanup before strict JSON parsing.

    The goal is to preserve the model's semantic content while removing
    formatting wrappers and, when needed, clipping a truncated tail at the last
    complete JSON boundary.
    """

    original_text = str(text or "")
    cleaned = normalize_text(original_text.replace("\r\n", "\n").replace("\r", "\n"))
    audit: dict[str, Any] = {
        "original_length": len(original_text),
        "cleaned_length": len(cleaned),
        "removed_leading_chars": 0,
        "removed_code_fence": False,
        "removed_trailing_code_fence": False,
        "balanced_close_applied": False,
        "balanced_close_cut": None,
        "balanced_close_remaining_stack": "",
    }

    leading_fence_match = re.match(r"^```(?:json)?\s*", cleaned, flags=re.IGNORECASE)
    if leading_fence_match:
        cleaned = cleaned[leading_fence_match.end() :]
        audit["removed_code_fence"] = True

    trailing_fence_match = re.search(r"\s*```$", cleaned)
    if trailing_fence_match:
        cleaned = cleaned[: trailing_fence_match.start()]
        audit["removed_trailing_code_fence"] = True

    start_candidates = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1]
    if start_candidates:
        start_index = min(start_candidates)
        if start_index > 0:
            audit["removed_leading_chars"] = start_index
            cleaned = cleaned[start_index:]

    audit["cleaned_length"] = len(cleaned)
    return cleaned, audit


def _repair_truncated_json_text(cleaned: str) -> tuple[str | None, dict[str, Any]]:
    """
    Attempt a narrow, auditable repair for a JSON document that was truncated
    after a complete nested object or array boundary.

    This is intentionally conservative:
    - only closing-delimiter balancing is attempted
    - no semantic fields are invented
    - no token-level reconstruction is performed
    """

    stack: list[str] = []
    in_string = False
    escaped = False
    candidates: list[tuple[int, list[str]]] = []

    for index, char in enumerate(cleaned):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                continue
            open_char = stack[-1]
            if (open_char == "{" and char == "}") or (open_char == "[" and char == "]"):
                stack.pop()
                candidates.append((index + 1, stack.copy()))

    for cut_index, remaining_stack in reversed(candidates):
        repaired = cleaned[:cut_index] + "".join("}" if char == "{" else "]" for char in reversed(remaining_stack))
        try:
            json.loads(repaired)
            return repaired, {
                "balanced_close_applied": True,
                "balanced_close_cut": cut_index,
                "balanced_close_remaining_stack": "".join(remaining_stack),
            }
        except Exception:
            continue

    return None, {
        "balanced_close_applied": False,
        "balanced_close_cut": None,
        "balanced_close_remaining_stack": "",
    }


def sanitize_stage2_json_text(text: str) -> tuple[str, dict[str, Any]]:
    cleaned, audit = _mechanically_sanitize_json_text(text)
    audit["parse_stage"] = "direct"
    try:
        json.loads(cleaned)
        return cleaned, audit
    except Exception as exc:
        audit["direct_parse_error"] = f"{type(exc).__name__}: {exc}"

    repaired, repair_audit = _repair_truncated_json_text(cleaned)
    audit.update(repair_audit)
    if repaired is not None:
        audit["parse_stage"] = "balanced_close"
        return repaired, audit

    audit["parse_stage"] = "failed"
    return cleaned, audit


def safe_json_load(text: str) -> dict[str, Any]:
    sanitized_text, _ = sanitize_stage2_json_text(text)
    return json.loads(sanitized_text)


def write_json_sanitization_audit(raw_response_path: Path, audit: dict[str, Any]) -> None:
    if not audit:
        return
    audit_path = raw_response_path.with_suffix(".sanitization.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def json_sanitization_applied(audit: dict[str, Any]) -> bool:
    return bool(
        audit.get("removed_leading_chars")
        or audit.get("removed_code_fence")
        or audit.get("removed_trailing_code_fence")
        or audit.get("balanced_close_applied")
        or audit.get("parse_stage") != "direct"
    )


def parse_json_list(value: Any) -> list[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def stage2_authority_run_dir_for_artifact(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in [resolved, *resolved.parents]:
        if candidate.name == "semantic_stage2_objects":
            return candidate.parent
    return resolved.parent


def build_stage2_authority_metadata(*, stage2_artifact_path: Path) -> dict[str, str]:
    authority_run_dir = stage2_authority_run_dir_for_artifact(stage2_artifact_path)
    authority_payload_root = authority_run_dir / "semantic_stage2_objects" / NORMALIZED_TABLE_PAYLOADS_SUBDIR
    return {
        "authority_run_dir": to_repo_rel(authority_run_dir),
        "authority_payload_root": to_repo_rel(authority_payload_root),
    }


def authority_payload_manifest_path(document: dict[str, Any]) -> Path | None:
    authority_payload_root = normalize_text(document.get("authority_payload_root"))
    document_key = normalize_text(document.get("document_key") or document.get("paper_key") or document.get("key"))
    if not authority_payload_root or not document_key:
        return None
    root = Path(authority_payload_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root / document_key / NORMALIZED_TABLE_PAYLOADS_FILENAME


def load_authority_normalized_payloads(document: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = authority_payload_manifest_path(document)
    if manifest_path is None or not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [item for item in ensure_list(payload.get("normalized_table_payloads")) if isinstance(item, dict)]


def build_table_scope_locator_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "table_id": normalize_text(payload.get("table_id") or payload.get("source_table_id")),
        "source_table_asset_id": normalize_text(payload.get("source_table_asset_id")),
        "source_table_reference": normalize_text(payload.get("source_table_reference") or payload.get("source_csv_path")),
    }


def resolve_payload_for_table_ref(table_ref: Any, normalized_payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    wanted = normalize_text(table_ref)
    wanted_norm = normalize_token(wanted)
    if not wanted_norm:
        return None
    matches: list[dict[str, Any]] = []
    for item in normalized_payloads:
        for candidate in [
            item.get("table_id"),
            item.get("source_table_id"),
            item.get("source_table_asset_id"),
            item.get("source_table_reference"),
            item.get("source_csv_path"),
        ]:
            candidate_norm = normalize_token(candidate)
            if candidate_norm and candidate_norm == wanted_norm:
                matches.append(item)
                break
    if len(matches) == 1:
        return matches[0]
    return None


def attach_table_scope_locators(document: dict[str, Any]) -> None:
    normalized_payloads = load_authority_normalized_payloads(document)
    if not normalized_payloads:
        return
    for scope in ensure_list(document.get("table_scopes")):
        if not isinstance(scope, dict):
            continue
        payload = resolve_payload_for_table_ref(scope.get("table_id"), normalized_payloads)
        if payload is None:
            continue
        scope["table_scope_locators"] = build_table_scope_locator_from_payload(payload)
        scope.setdefault("source_table_asset_id", normalize_text(payload.get("source_table_asset_id")))
        scope.setdefault(
            "source_table_reference",
            normalize_text(payload.get("source_table_reference") or payload.get("source_csv_path")),
        )
    for scope in ensure_list(document.get("table_formulation_scopes")):
        if not isinstance(scope, dict):
            continue
        payload = resolve_payload_for_table_ref(scope.get("table_id"), normalized_payloads)
        if payload is None:
            continue
        scope["table_scope_locators"] = build_table_scope_locator_from_payload(payload)
        scope.setdefault("table_asset_id", normalize_text(payload.get("source_table_asset_id")))
        scope.setdefault("source_table_asset_id", normalize_text(payload.get("source_table_asset_id")))
        scope.setdefault(
            "source_table_reference",
            normalize_text(payload.get("source_table_reference") or payload.get("source_csv_path")),
        )
    for declaration in ensure_list(document.get("semantic_scope_declarations")):
        if not isinstance(declaration, dict):
            continue
        refs = [normalize_text(item) for item in ensure_list(declaration.get("table_scope_refs")) if normalize_text(item)]
        locators = []
        for ref in refs:
            payload = resolve_payload_for_table_ref(ref, normalized_payloads)
            if payload is None:
                continue
            locators.append(build_table_scope_locator_from_payload(payload))
        if locators:
            declaration["table_scope_locators"] = locators


def evidence_blocks_path(out_dir: Path, key: str) -> Path:
    return out_dir / EVIDENCE_BLOCKS_SUBDIR / key / EVIDENCE_BLOCKS_FILENAME


def candidate_blocks_path(out_dir: Path, key: str) -> Path:
    return out_dir / CANDIDATE_BLOCKS_SUBDIR / key / CANDIDATE_BLOCKS_FILENAME


def normalized_table_payloads_path(out_dir: Path, key: str) -> Path:
    return out_dir / NORMALIZED_TABLE_PAYLOADS_SUBDIR / key / NORMALIZED_TABLE_PAYLOADS_FILENAME


def ensure_genai(model: str) -> None:
    if genai is None:
        raise RuntimeError("google.generativeai is not installed.")
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in environment.")
    genai.configure(api_key=api_key)
    if not str(model or "").strip():
        raise RuntimeError("Gemini model name is empty.")


def call_gemini(
    model: str,
    prompt: str,
    retries: int,
    sleep_sec: float,
    *,
    progress_label: str = "",
    timeout_seconds: int | None = None,
) -> str:
    ensure_genai(model)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if progress_label:
                print(
                    f"{progress_label} request_start attempt={attempt + 1}/{retries + 1} "
                    f"prompt_chars={len(prompt)} timeout_seconds={timeout_seconds if timeout_seconds is not None else 'sdk_default'}",
                    flush=True,
                )
            mdl = genai.GenerativeModel(model)
            request_kwargs: dict[str, Any] = {
                "generation_config": {
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            }
            if timeout_seconds is not None:
                request_kwargs["request_options"] = {"timeout": timeout_seconds}
            resp = mdl.generate_content(prompt, **request_kwargs)
            if getattr(resp, "text", ""):
                return str(resp.text)
            candidates = getattr(resp, "candidates", []) or []
            if candidates:
                parts = getattr(candidates[0].content, "parts", []) or []
                if parts and getattr(parts[0], "text", ""):
                    return str(parts[0].text)
            raise RuntimeError("Gemini returned empty content.")
        except Exception as exc:  # pragma: no cover
            last_err = exc
            if progress_label:
                print(
                    f"{progress_label} request_exception attempt={attempt + 1}/{retries + 1} "
                    f"error_type={type(exc).__name__} error={exc}",
                    flush=True,
                )
        if attempt < retries:
            if progress_label:
                print(
                    f"{progress_label} retrying attempt={attempt + 2}/{retries + 1}",
                    flush=True,
                )
            time.sleep(sleep_sec)
    raise last_err or RuntimeError("Gemini call failed.")


def extract_gemini_stream_chunk_text(chunk: Any) -> str:
    """Safely recover streamed text without relying solely on SDK quick accessors."""
    try:
        direct_text = getattr(chunk, "text")
    except Exception:
        direct_text = ""
    if direct_text:
        return str(direct_text)

    collected: list[str] = []
    candidates = getattr(chunk, "candidates", None)
    if candidates is None and isinstance(chunk, dict):
        candidates = chunk.get("candidates")
    for candidate in ensure_list(candidates):
        content = getattr(candidate, "content", None)
        if content is None and isinstance(candidate, dict):
            content = candidate.get("content")
        parts = getattr(content, "parts", None)
        if parts is None and isinstance(content, dict):
            parts = content.get("parts")
        for part in ensure_list(parts):
            part_text = getattr(part, "text", None)
            if part_text is None and isinstance(part, dict):
                part_text = part.get("text")
            if part_text:
                collected.append(str(part_text))
    return "".join(collected)


def call_gemini_stream_collect(
    model: str,
    prompt: str,
    retries: int,
    sleep_sec: float,
    *,
    progress_label: str = "",
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    ensure_genai(model)
    last_result: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        parts: list[str] = []
        chunk_count = 0
        started_at = time.time()
        first_chunk_elapsed: float | None = None
        try:
            if progress_label:
                print(
                    f"{progress_label} stream_request_start attempt={attempt + 1}/{retries + 1} "
                    f"prompt_chars={len(prompt)} timeout_seconds={timeout_seconds if timeout_seconds is not None else 'sdk_default'}",
                    flush=True,
                )
            mdl = genai.GenerativeModel(model)
            request_kwargs: dict[str, Any] = {
                "generation_config": {
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
                "stream": True,
            }
            if timeout_seconds is not None:
                request_kwargs["request_options"] = {"timeout": timeout_seconds}
            stream = mdl.generate_content(prompt, **request_kwargs)
            for chunk in stream:
                chunk_count += 1
                text = extract_gemini_stream_chunk_text(chunk)
                if text:
                    parts.append(str(text))
                now = time.time()
                if first_chunk_elapsed is None:
                    first_chunk_elapsed = now - started_at
                if progress_label and (chunk_count == 1 or chunk_count % 25 == 0):
                    print(
                        f"{progress_label} stream_progress chunk_count={chunk_count} "
                        f"collected_chars={sum(len(part) for part in parts)} elapsed_seconds={round(now - started_at, 3)}",
                        flush=True,
                    )
            collected_text = "".join(parts)
            if not collected_text:
                raise RuntimeError("Gemini stream returned empty content.")
            return {
                "status": "success",
                "text": collected_text,
                "chunk_count": chunk_count,
                "first_chunk_elapsed_seconds": round(first_chunk_elapsed or 0.0, 3),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "error_type": "",
                "error_message": "",
            }
        except Exception as exc:  # pragma: no cover
            collected_text = "".join(parts)
            last_result = {
                "status": "request_failure",
                "text": collected_text,
                "chunk_count": chunk_count,
                "first_chunk_elapsed_seconds": round(first_chunk_elapsed or 0.0, 3),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            if progress_label:
                print(
                    f"{progress_label} stream_request_exception attempt={attempt + 1}/{retries + 1} "
                    f"chunk_count={chunk_count} collected_chars={len(collected_text)} "
                    f"error_type={type(exc).__name__} error={exc}",
                    flush=True,
                )
        if attempt < retries:
            if progress_label:
                print(
                    f"{progress_label} retrying_stream attempt={attempt + 2}/{retries + 1}",
                    flush=True,
                )
            time.sleep(sleep_sec)
    return last_result or {
        "status": "request_failure",
        "text": "",
        "chunk_count": 0,
        "first_chunk_elapsed_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "error_type": "RuntimeError",
        "error_message": "Gemini stream call failed.",
    }


def call_nvidia_hosted(model: str, prompt: str, retries: int, sleep_sec: float, *, progress_label: str = "") -> str:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is missing in environment.")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON matching the requested object-first schema.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                NVIDIA_HOSTED_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") or []
            message = choices[0].get("message") if choices else {}
            content = message.get("content") if isinstance(message, dict) else ""
            if isinstance(content, str) and content.strip():
                return content
            raise RuntimeError("NVIDIA hosted API returned empty content.")
        except Exception as exc:  # pragma: no cover
            last_err = exc
        if attempt < retries:
            if progress_label:
                print(
                    f"{progress_label} retrying attempt={attempt + 2}/{retries + 1}",
                    flush=True,
                )
            time.sleep(sleep_sec * (attempt + 1))
    raise last_err or RuntimeError("NVIDIA hosted API call failed.")


class Stage2ProgressReporter:
    def __init__(self, total_tasks: int, heartbeat_sec: float = 30.0) -> None:
        self.total_tasks = max(0, total_tasks)
        self.completed = 0
        self.failed = 0
        self.current_index = 0
        self.current_key = ""
        self.current_started_at = 0.0
        self._heartbeat_sec = max(0.0, heartbeat_sec)
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._heartbeat_sec <= 0:
            return
        self._thread = Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def begin_task(self, index: int, key: str) -> str:
        self.current_index = index
        self.current_key = key
        self.current_started_at = time.monotonic()
        print(self._format_message("running"), flush=True)
        return self.task_prefix()

    def complete_task(self) -> None:
        self.completed += 1
        print(self._format_message("completed"), flush=True)

    def fail_task(self, error: Exception) -> None:
        self.failed += 1
        print(self._format_message(f"failed error={type(error).__name__}: {error}"), flush=True)

    def finish(self) -> None:
        print(
            f"stage2_progress finished success={self.completed} failed={self.failed} total={self.total_tasks}",
            flush=True,
        )

    def task_prefix(self) -> str:
        return f"stage2_progress [{self.current_index}/{self.total_tasks}] paper={self.current_key}"

    def _format_message(self, status: str) -> str:
        return (
            f"{self.task_prefix()} {status} "
            f"completed={self.completed} failed={self.failed} total={self.total_tasks}"
        ).strip()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_sec):
            if not self.current_index or not self.current_key:
                continue
            elapsed_sec = int(time.monotonic() - self.current_started_at)
            print(
                f"{self.task_prefix()} heartbeat completed={self.completed} failed={self.failed} "
                f"elapsed_sec={elapsed_sec}",
                flush=True,
            )


def resolve_tables_dir(text_path: Path, key: str) -> Path | None:
    candidates = [
        text_path.parent.parent / "tables" / key,
        PROJECT_ROOT / "data" / "cleaned" / "content_goren_2025" / "tables" / key,
        PROJECT_ROOT / "data" / "cleaned" / "goren_2025" / "tables" / key,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_tables_dir_for_record(record: dict[str, Any], text_path: Path, key: str) -> Path | None:
    explicit_table_dir = normalize_text(record.get("table_dir"))
    explicit_table_available = normalize_text(record.get("table_available"))
    if explicit_table_dir:
        candidate = Path(explicit_table_dir)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.exists():
            return candidate
        if explicit_table_available in {"1", "true", "yes", "y"}:
            raise FileNotFoundError(f"Manifest row for {key} declares table_available=yes but table_dir is missing: {candidate}")
    if explicit_table_available in {"0", "false", "no", "n"}:
        return None
    return resolve_tables_dir(text_path, key)


def load_table_manifest_payload(table_dir: Path | None) -> dict[str, Any]:
    if table_dir is None or not table_dir.exists():
        return {}
    manifest_path = table_dir / "tables_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


NOISE_LINE_PATTERNS = [
    r"\bdownloaded from\b",
    r"\bwiley online library\b",
    r"\bdovepress\b",
    r"\bsubmit your manuscript\b",
    r"\btcpdf\b",
    r"\binternational journal of nanomedicine\b",
    r"\bpage \d+\b",
    r"\bopen access\b",
    r"\bpublished online\b",
    r"\bunauthenticated\b",
]

SEGMENT_MATERIALS_CUES = [
    "purchased from",
    "obtained from",
    "supplied by",
    "sigma",
    "aldrich",
    "fisher",
    "molecular weight",
    "resomer",
    "plga",
    "poloxamer",
    "labrafil",
    "polysorbate",
]
SEGMENT_METHOD_CUES = [
    "prepared by",
    "prepared using",
    "nanoprecipitation",
    "emulsion solvent evaporation",
    "dissolved",
    "added dropwise",
    "aqueous phase",
    "organic phase",
    "stirring",
    "centrifuged",
    "washed",
    "freeze-dried",
]
SEGMENT_RESULT_CUES = [
    "particle size",
    "pdi",
    "zeta potential",
    "encapsulation efficiency",
    "entrapment efficiency",
    "incorporation efficiency",
    "loading capacity",
    "maximum loading capacity",
    "mean values",
    "lower particle sizes",
    "higher ee",
    "as shown in table",
]
SEGMENT_EXPERIMENTAL_DESIGN_CUES = [
    "fixed amounts of polymer",
    "fixed amounts of surfactant",
    "fixed amounts of polymer and surfactant",
    "fixed amounts of polymer and surfactants",
    "fixed amounts of polymer (plga)",
    "variable quantities",
    "increasing theoretical concentrations",
    "different volumes",
    "concentrations ranging",
    "prepared using fixed amounts",
]
SEGMENT_OPTIMIZATION_CUES = [
    "selected as optimal",
    "chosen as optimal",
    "optimized formulation",
    "desirability",
    "remaining studies",
    "best results",
    "highest",
    "maximum",
    "increased",
    "decreased",
    "efficiency",
    "loading",
    "saturation",
    "optimal surfactant concentration",
    "maximum loading capacity",
    "higher concentrations",
    "crystals could be observed",
    "increase the oil volume",
    "increasing the volume",
    "close to 90%",
    "close to 80%",
]
SEGMENT_DOWNSTREAM_CUES = [
    "cell viability",
    "biodistribution",
    "pharmacokinetic",
    "gamma scintigraphy",
    "radiolabeling",
    "sem",
    "scanning electron microscopy",
    "in vitro release",
    "release profiles",
    "microscopy",
    "lc-ms/ms",
]
SEGMENT_VARIANT_CUES = [
    "same procedure",
    "same protocol",
    "same method",
    "incorporating",
    "with polysorbate 80",
    "with labrafil",
    "blank nps were prepared",
]
SEGMENT_ABSTRACT_CUES = [
    "abstract:",
    "purpose:",
    "materials and methods:",
    "results:",
    "conclusion:",
    "keywords:",
]
SEGMENT_PROSE_CARRYOVER_CUES = [
    "the morphology of",
    "pharmacokinetics study",
    "intravenous formulation",
    "biodistribution of",
    "mean concentrations of",
]
TABLE_FORMULATION_PRIORITY_CUES = [
    "formulation number",
    "drug:polymer ratio",
    "polymer used",
    "surfactant",
    "theoretical concentration",
    "final concentration",
    "drug loading",
    "loading capacity",
    "encapsulation efficiency",
    "entrapment efficiency",
    "ee",
    "dl",
    "ratio",
]
TABLE_CHARACTERIZATION_DEMOTION_CUES = [
    "ftir",
    "spectrum",
    "spectra",
    "microscopy",
    "tem",
    "sem",
    "fesem",
    "afm",
    "dsc",
    "thermogram",
    "thermograms",
    "xrd",
    "micrograph",
    "micrographs",
    "particle size distribution pattern",
    "zeta potential distribution profile",
    "image",
    "images",
]
TABLE_AUTHORITY_FORMULATION_TOKENS = [
    "formulation",
    "drug:polymer ratio",
    "plga:itz",
    "polymer",
    "drug",
    "surfactant",
    "concentration",
    "ratio",
    "entrapment",
    "encapsulation",
    "loading",
    "particle size",
    "pdi",
    "zeta potential",
]
TABLE_AUTHORITY_DESIGN_TOKENS = [
    "independent variables",
    "dependent variables",
    "levels",
    "coded levels",
    "factorial points",
    "factors",
    "design matrix",
    "box-behnken",
    "response surface",
    "run order",
]
TABLE_AUTHORITY_DOWNSTREAM_DEMOTION_TOKENS = [
    "pharmacokinetic",
    "pharmacokinetics",
    "plasma concentration-time",
    "concentration-time profile",
    "concentration-time profiles",
    "distribution of",
    "organ/tissue",
    "brain compartments",
    "blood and brain compartments",
    "tissue distribution",
    "time points",
    "auc",
    "mrt",
    "cmax",
    "tmax",
    "rats",
    "mice",
    "brain",
    "limit of detection",
    "limit of quantification",
    "calibration curve",
    "noncompartmental",
    "systemic clearance",
    "volume of distribution",
]
TABLE_AUTHORITY_STABILITY_DEMOTION_TOKENS = [
    "storage time",
    "day 1",
    "day 7",
    "day 15",
    "day 75",
    "stability",
    "release profile",
    "release profiles",
    "in-vitro release",
    "in vitro release",
]
TABLE_AUTHORITY_FRONTMATTER_DEMOTION_TOKENS = [
    "dovepress",
    "publish your work in this journal",
    "journal citation reports",
    "journal name",
    "article designation",
    "original research",
    "running head",
    "purpose:",
    "author information",
    "reviewed journal focusing",
    "downloaded from",
    "wiley online library",
]
TABLE_AUTHORITY_TIER_PRIMARY = "primary"
TABLE_AUTHORITY_TIER_SECONDARY = "secondary"
TABLE_AUTHORITY_TIER_WEAK_SECONDARY = "weak_secondary"
REFERENCE_TAIL_CUES = [
    "crossref",
    "pubmed",
    "author information",
    "article recommendations",
    "corresponding author",
]
FRONT_MATTER_CUES = [
    "www.nature.com",
    "scientific reports",
    "home annals of biomedical engineering",
    "annals of biomedical engineering",
    "nanomaterials 2020",
    "int j ophthalmol",
    "doi:",
    "academic editor",
    "correspondence:",
]


def is_obvious_noise_line(text: str) -> bool:
    compact = normalize_text(text).lower()
    if not compact:
        return True
    if any(re.search(pattern, compact) for pattern in NOISE_LINE_PATTERNS):
        return True
    if compact in {"article", "original article", "research article"}:
        return True
    if re.fullmatch(r"\d+\s*(?:of\s*\d+)?", compact):
        return True
    return False


def clean_candidate_text(text: str) -> tuple[str, list[str]]:
    kept_lines: list[str] = []
    noise_flags: list[str] = []
    for raw_line in str(text or "").splitlines():
        compact = normalize_text(raw_line)
        if not compact:
            continue
        if is_obvious_noise_line(compact):
            if "noise_line_removed" not in noise_flags:
                noise_flags.append("noise_line_removed")
            continue
        kept_lines.append(compact)
    return "\n".join(kept_lines).strip(), noise_flags


def count_segmentation_cues(text: str, cues: list[str]) -> int:
    lower = normalize_text(text).lower()
    return sum(1 for cue in cues if cue in lower)


def is_reference_like_text(text: str) -> bool:
    normalized = normalize_text(text)
    lower = normalized.lower()
    reference_hits = len(re.findall(r"\[\d+\]|\(\d+\)", normalized))
    journal_citation_hits = len(
        re.findall(
            r"\b(?:j\.|int\.|eur\.|pharm\.|biopharm\.|drug dev\.|expert opin\.|acs omega|small|nanomed|polymer|ann\.|doi:)\b",
            lower,
        )
    )
    ref_tail_hits = count_segmentation_cues(lower, REFERENCE_TAIL_CUES)
    et_al_hits = lower.count(" et al")
    return (
        reference_hits >= 2
        or journal_citation_hits >= 4
        or ref_tail_hits >= 1
        or (journal_citation_hits >= 2 and et_al_hits >= 2)
    )


def is_front_matter_like_text(text: str) -> bool:
    lower = normalize_text(text).lower()
    if not lower:
        return False
    if any(token in lower for token in FRONT_MATTER_CUES):
        return True
    return bool(
        re.match(r"^(?:home\s+|www\.|https?://)", lower)
        or re.match(r"^(?:scientific reports|nanomaterials\s+\d{4}|int j ophthalmol|annals of biomedical engineering)\b", lower)
    )


def has_optimization_text_signal(text: str) -> bool:
    lower = normalize_text(text).lower()
    if not lower or is_reference_like_text(lower):
        return False
    comparator_hits = count_segmentation_cues(
        lower,
        [
            "best results",
            "highest",
            "maximum",
            "increased",
            "decreased",
            "optimal",
            "optimized",
            "saturation",
        ],
    )
    metric_hits = count_segmentation_cues(
        lower,
        [
            "efficiency",
            "loading",
            "encapsulation",
            "entrapment",
            "particle size",
            "zeta potential",
            "pdi",
            "drug loading",
        ],
    )
    return (
        count_segmentation_cues(lower, SEGMENT_OPTIMIZATION_CUES) > 0
        or (comparator_hits >= 1 and metric_hits >= 1)
    )


def has_experimental_design_text_signal(text: str) -> bool:
    lower = normalize_text(text).lower()
    if not lower or is_reference_like_text(lower):
        return False
    if has_variable_sweep_design_signal(lower):
        return True
    if re.search(r"\bcryoprotectants?.{0,120}\bconcentrations?\b", lower):
        return True
    if re.search(r"\bat\s+\w+\s+di\S*erent concentrations\b", lower):
        return True
    direct_phrases = [
        "varying concentration",
        "varying concentrations",
        "various optimized concentrations",
        "different amounts",
        "multiple batches",
        "cryoprotectant concentrations",
        "different concentrations",
        "different volumes",
        "different surfactants",
        "different peg",
        "polymer used",
        "formulation number",
    ]
    if any(token in lower for token in direct_phrases):
        return True
    if re.search(r"\branging from\s+\d", lower):
        return True
    return False


def is_characterization_only_table_signal(signal_text: str) -> bool:
    lower = normalize_text(signal_text).lower()
    if ("fig." in lower or "figure " in lower) and any(
        token in lower for token in ["dsc", "thermogram", "micrograph", "microscopy", "image"]
    ):
        return True
    return count_segmentation_cues(lower, TABLE_CHARACTERIZATION_DEMOTION_CUES) >= 2


def has_strong_formulation_table_signal(signal_text: str) -> bool:
    lower = normalize_text(signal_text).lower()
    if not lower or is_front_matter_like_text(lower) or is_reference_like_text(lower):
        return False
    strong_cues = [
        "formulation characters",
        "formulation number",
        "drug:polymer ratio",
        "polymer used",
        "theoretical concentration",
        "final concentration",
        "drug loading",
        "loading capacity",
        "encapsulation efficiency",
        "entrapment efficiency",
    ]
    if any(token in lower for token in strong_cues):
        return True
    return (
        "formulation" in lower
        and any(
            token in lower
            for token in [
                "drug loaded",
                "empty",
                "nanospheres",
                "nanocapsules",
                "surfactant",
                "ratio",
                "concentration",
            ]
        )
    )


def is_obvious_figure_or_front_matter_table(signal_text: str) -> bool:
    lower = normalize_text(signal_text).lower()
    if is_front_matter_like_text(lower) or is_reference_like_text(lower):
        return True
    return ("fig." in lower or "figure " in lower) and any(
        token in lower
        for token in ["dsc", "thermogram", "micrograph", "microscopy", "image", "profile", "concentration-time"]
    )


def looks_like_short_heading(line: str) -> bool:
    compact = normalize_text(line)
    if not compact or len(compact) > 100 or compact.endswith("."):
        return False
    words = compact.split()
    if not 2 <= len(words) <= 12:
        return False
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if not alpha_words:
        return False
    titleish = 0
    for word in alpha_words:
        if word[:1].isupper() or word.upper() == word or re.search(r"[A-Z].*[-/]", word):
            titleish += 1
        elif word.lower() in {"and", "of", "for", "in", "to", "with", "by", "on", "the"}:
            titleish += 1
    return titleish / max(len(alpha_words), 1) >= 0.8


def classify_heading_line(line: str) -> str | None:
    compact = normalize_text(line)
    lower = compact.lower()
    if not compact:
        return None
    if re.match(r"^table\s+\d+\b", lower):
        return "table_inline"
    if re.match(r"^(abstract|purpose|materials and methods|results|conclusion|keywords|introduction)\s*:", lower):
        return "abstract_label"
    if re.match(r"^\d+(?:\.\d+)+\.?\s+[A-Z]", compact):
        return "numbered_heading"
    if re.match(
        r"^(materials(?: and methods)?|preparation(?: and characterization of nps)?|experimental design|results(?: and discussion)?|discussion|conclusion|introduction|morphology and size of nps|process yield and encapsulation efficiency|gatifloxacin-loaded plga nps|rhodamine-loaded plga nps|lyophilization|analytical methods|assay|pharmacokinetic study|process variables|data analysis and optimization)\b",
        lower,
    ):
        return "heading"
    return None


def split_sentences_for_segmentation(text: str) -> list[str]:
    compact = normalize_text(text)
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", compact)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def sentence_profile(text: str) -> str:
    lower = normalize_text(text).lower()
    if not lower:
        return "context"
    if is_reference_like_text(lower):
        return "context"
    if count_segmentation_cues(lower, SEGMENT_ABSTRACT_CUES) > 0:
        return "abstract_summary"
    if count_segmentation_cues(lower, SEGMENT_VARIANT_CUES) > 0 and count_segmentation_cues(lower, SEGMENT_METHOD_CUES) > 0:
        return "variant"
    if has_experimental_design_text_signal(lower):
        return "experimental_design"
    if has_optimization_text_signal(lower):
        return "optimization"
    scores = {
        "materials": count_segmentation_cues(lower, SEGMENT_MATERIALS_CUES),
        "preparation": count_segmentation_cues(lower, SEGMENT_METHOD_CUES),
        "result": count_segmentation_cues(lower, SEGMENT_RESULT_CUES),
        "optimization": count_segmentation_cues(lower, SEGMENT_OPTIMIZATION_CUES),
        "downstream_assay": count_segmentation_cues(lower, SEGMENT_DOWNSTREAM_CUES),
    }
    best_kind, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return "context"
    if best_kind == "downstream_assay" and best_score >= scores["preparation"] and best_score >= scores["result"]:
        return "downstream_assay"
    return best_kind


def should_drop_segment(text: str, section_kind: str, section_label: str) -> bool:
    lower = normalize_text(text).lower()
    if is_reference_like_text(lower):
        return True
    if is_front_matter_like_text(lower):
        return True
    if section_kind == "front_matter":
        return True
    if section_kind == "abstract_summary":
        return True
    if section_kind == "downstream_assay" and section_label and any(
        token in section_label.lower()
        for token in ["cell viability", "biodistribution", "gamma scintigraphy", "radiolabeling", "pharmacokinetic"]
    ):
        return True
    if any(token in lower for token in ["creative commons attribution", "submit your manuscript", "powered by tcpdf"]):
        return True
    if not section_label and any(token in lower for token in ["correspondence:", "department of", "faculty of", "college of", "school of pharmacy", "academic editor"]):
        return True
    return False


def extract_section_label(text: str) -> str:
    compact = normalize_text(text)
    if not compact:
        return ""
    structured_abstract = re.match(r"^((?:abstract|purpose|materials and methods|results|conclusion|keywords)\s*:)", compact, flags=re.I)
    if structured_abstract:
        return normalize_text(structured_abstract.group(1))
    numbered = re.match(r"^(\d+(?:\.\d+)+\.?\s+[A-Z][^.]{0,120})", compact)
    if numbered:
        return normalize_text(numbered.group(1))
    heading_like = re.match(
        r"^((?:materials(?: and methods)?|preparation(?: of [a-z0-9\-]+)?|experimental design|optimization|results(?: and discussion)?|discussion|conclusion)[^.]*)",
        compact,
        flags=re.I,
    )
    if heading_like:
        return normalize_text(heading_like.group(1))
    return ""


def infer_section_kind(text: str, section_label: str = "") -> str:
    combined = f"{section_label} {normalize_text(text)}".lower()
    text_only = normalize_text(text).lower()
    section_label_lower = normalize_text(section_label).lower()
    if is_front_matter_like_text(combined):
        return "front_matter"
    if is_reference_like_text(combined):
        return "front_matter"
    if any(token in combined for token in ["creative commons attribution", "submit your manuscript", "powered by tcpdf", "dovepress", "wiley online library"]):
        return "front_matter"
    if any(token in combined for token in ["correspondence:", "department of", "faculty of", "college of", "school of pharmacy", "academic editor"]):
        if count_segmentation_cues(combined, SEGMENT_METHOD_CUES + SEGMENT_MATERIALS_CUES + SEGMENT_RESULT_CUES + SEGMENT_OPTIMIZATION_CUES) == 0:
            return "front_matter"
    if re.match(r"^(abstract|purpose|conclusion|keywords)\s*:", text_only):
        return "abstract_summary"
    if text_only.startswith("materials and methods:"):
        if any(token in text_only for token in ["experimental design", "box-behnken", "response surface", "ratio", "optimized by modifying"]):
            return "experimental_design"
        return "preparation"
    if text_only.startswith("results:"):
        if count_segmentation_cues(text_only, SEGMENT_OPTIMIZATION_CUES) > 0:
            return "optimization"
        if count_segmentation_cues(text_only, SEGMENT_RESULT_CUES) > 0:
            return "table_related"
        return "context"
    if count_segmentation_cues(combined, SEGMENT_ABSTRACT_CUES) > 0:
        return "context"
    if count_segmentation_cues(combined, SEGMENT_VARIANT_CUES) > 0 and count_segmentation_cues(combined, SEGMENT_METHOD_CUES) > 0:
        return "variant_preparation"
    if has_experimental_design_text_signal(combined):
        return "experimental_design"
    if has_optimization_text_signal(combined):
        return "optimization"
    if section_label_lower.startswith("3.1.") or section_label_lower.startswith("3.2."):
        if has_optimization_text_signal(combined):
            return "optimization"
        if count_segmentation_cues(combined, SEGMENT_RESULT_CUES) > 0:
            return "result"
    if section_label_lower.startswith("materials") or " materials" in section_label_lower:
        return "materials"
    if any(token in combined for token in ["purchased from", "obtained from", "supplied by", "gifted by", "analytical grade", "hplc grade", "molecular weight", "resomer", "purasorb"]):
        return "materials"
    if any(token in combined for token in ["preparation", "synthesis", "nanoprecipitation", "emulsion solvent evaporation", "dissolved", "added dropwise"]):
        return "preparation"
    if has_experimental_design_text_signal(combined):
        return "experimental_design"
    if has_optimization_text_signal(combined):
        return "optimization"
    if any(token in combined for token in ["incorporation efficiency", "loading capacity", "maximum loading capacity", "crystals could be observed"]):
        return "optimization"
    if any(token in combined for token in ["table ", "formulation", "table 1", "table 2"]) and any(token in combined for token in ["particle size", "entrapment", "pdi", "zeta potential", "levels", "independent variables"]):
        return "table_related"
    if count_segmentation_cues(combined, SEGMENT_DOWNSTREAM_CUES) > 0:
        return "downstream_assay"
    if any(token in combined for token in ["table ", "runs", "particle size", "entrapment", "pdi", "zeta potential"]):
        return "table_related"
    return "context"


def build_candidate_quality_flags(text: str, section_kind: str, split_trigger: str = "") -> list[str]:
    lower = normalize_text(text).lower()
    flags: list[str] = []
    if extract_section_label(text):
        flags.append("heading_scoped")
    if split_trigger:
        flags.append(f"split_trigger:{split_trigger}")
    if any(token in lower for token in ["cell viability", "biodistribution", "pharmacokinetic", "release study", "stability study", "sem "]):
        flags.append("downstream_assay_terms")
    if section_kind in {"preparation", "materials"} and any(token in lower for token in ["results and discussion", "discussion", "optimized formulation", "entrapment efficiency"]):
        flags.append("possible_mixed_role_content")
    if section_kind == "variant_preparation":
        flags.append("variant_supporting_candidate")
    if has_experimental_design_text_signal(lower):
        flags.append("variable_sweep_design")
    if section_kind == "optimization":
        flags.append("optimization_candidate")
    if section_kind == "experimental_design":
        flags.append("experimental_design_candidate")
    if section_kind in {"abstract_summary", "front_matter"}:
        flags.append("suppressible_noncanonical_context")
    if is_reference_like_text(lower):
        flags.append("reference_like_content")
    if any(token in lower for token in ["copyright", "journal", "downloaded from", "tcpdf"]):
        flags.append("residual_noise")
    return flags


def split_paragraph_entries(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, part in enumerate(re.split(r"\n\s*\n+", text or "")):
        cleaned, noise_flags = clean_candidate_text(part)
        if not normalize_text(cleaned):
            continue
        entries.append(
            {
                "paragraph_index": index,
                "text": cleaned,
                "noise_flags": noise_flags,
                "split_trigger": "paragraph_boundary",
            }
        )
    return entries


def split_section_scoped_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    section_pattern = re.compile(r"(?=(?:\d+(?:\.\d+)+\.?\s*[A-Z]))")
    expanded: list[dict[str, Any]] = []
    for entry in entries:
        raw_text = str(entry.get("text") or "")
        text = normalize_text(raw_text)
        if not text:
            continue
        parts = [normalize_text(part) for part in section_pattern.split(text) if normalize_text(part)]
        if len(parts) <= 1:
            expanded.append(dict(entry))
            continue
        for segment_index, part in enumerate(parts):
            expanded.append(
                {
                    "paragraph_index": entry["paragraph_index"],
                    "segment_index": segment_index,
                    "text": part,
                    "noise_flags": list(entry.get("noise_flags") or []),
                    "split_trigger": "numbered_heading_split",
                }
            )
    return expanded


def split_inline_heading_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for entry in entries:
        text = str(entry.get("text") or "")
        if not normalize_text(text):
            continue
        lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
        if len(lines) <= 1:
            expanded.append(dict(entry))
            continue
        current_lines: list[str] = []
        current_trigger = str(entry.get("split_trigger") or "paragraph_boundary")
        split_count = 0
        for line in lines:
            heading_type = classify_heading_line(line)
            if heading_type is not None and current_lines:
                expanded.append(
                    {
                        "paragraph_index": entry["paragraph_index"],
                        "segment_index": split_count,
                        "text": "\n".join(current_lines),
                        "noise_flags": list(entry.get("noise_flags") or []),
                        "split_trigger": current_trigger,
                    }
                )
                split_count += 1
                current_lines = [line]
                current_trigger = heading_type
            else:
                current_lines.append(line)
                if heading_type is not None:
                    current_trigger = heading_type
        if current_lines:
            expanded.append(
                {
                    "paragraph_index": entry["paragraph_index"],
                    "segment_index": split_count,
                    "text": "\n".join(current_lines),
                    "noise_flags": list(entry.get("noise_flags") or []),
                    "split_trigger": current_trigger,
                }
            )
    return expanded


def split_transition_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for entry in entries:
        text = normalize_text(entry.get("text"))
        if not text:
            continue
        sentences = split_sentences_for_segmentation(text)
        if len(sentences) <= 1:
            expanded.append(dict(entry))
            continue
        current_sentences: list[str] = [sentences[0]]
        current_profile = sentence_profile(sentences[0])
        current_trigger = str(entry.get("split_trigger") or "paragraph_boundary")
        split_count = 0
        for sentence in sentences[1:]:
            next_profile = sentence_profile(sentence)
            transition = False
            transition_trigger = ""
            if "table " in sentence.lower() and len(" ".join(current_sentences)) >= 120:
                transition = True
                transition_trigger = "table_inline_split"
            elif current_profile == "experimental_design" and next_profile in {"result", "optimization", "downstream_assay"} and len(" ".join(current_sentences)) >= 140:
                transition = True
                transition_trigger = f"cue_transition_experimental_design_to_{next_profile}"
            elif current_profile in {"result", "optimization"} and next_profile == "experimental_design" and len(" ".join(current_sentences)) >= 120:
                transition = True
                transition_trigger = f"cue_transition_{current_profile}_to_experimental_design"
            elif next_profile == "variant" and current_profile in {"preparation", "variant"} and len(" ".join(current_sentences)) >= 180:
                transition = True
                transition_trigger = "variant_split"
            elif current_profile in {"preparation", "variant"} and next_profile in {"result", "optimization", "downstream_assay"} and len(" ".join(current_sentences)) >= 180:
                transition = True
                transition_trigger = f"cue_transition_{current_profile}_to_{next_profile}"
            elif current_profile == "materials" and next_profile in {"context", "downstream_assay"} and len(" ".join(current_sentences)) >= 140:
                transition = True
                transition_trigger = f"cue_transition_materials_to_{next_profile}"
            elif current_profile in {"result", "optimization"} and next_profile == "downstream_assay" and len(" ".join(current_sentences)) >= 140:
                transition = True
                transition_trigger = f"cue_transition_{current_profile}_to_downstream"
            if transition:
                expanded.append(
                    {
                        "paragraph_index": entry["paragraph_index"],
                        "segment_index": split_count,
                        "text": " ".join(current_sentences),
                        "noise_flags": list(entry.get("noise_flags") or []),
                        "split_trigger": current_trigger,
                    }
                )
                split_count += 1
                current_sentences = [sentence]
                current_profile = next_profile
                current_trigger = transition_trigger
            else:
                current_sentences.append(sentence)
                if current_profile == "context" and next_profile != "context":
                    current_profile = next_profile
        if current_sentences:
            expanded.append(
                {
                    "paragraph_index": entry["paragraph_index"],
                    "segment_index": split_count,
                    "text": " ".join(current_sentences),
                    "noise_flags": list(entry.get("noise_flags") or []),
                    "split_trigger": current_trigger,
                }
            )
    return expanded


def split_embedded_design_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for entry in entries:
        text = normalize_text(entry.get("text"))
        if not text:
            continue
        lower = text.lower()
        if not lower.startswith("materials and methods:") or "optimized by modifying" not in lower:
            expanded.append(dict(entry))
            continue
        prefix = "Materials and methods:"
        body = normalize_text(text[len(prefix) :])
        first_sentence_match = re.match(r"^(.*?\.)\s*(.*)$", body)
        if not first_sentence_match:
            expanded.append(dict(entry))
            continue
        first_sentence = normalize_text(first_sentence_match.group(1))
        remainder = normalize_text(first_sentence_match.group(2))
        split_phrase = " and optimized by modifying "
        split_idx = first_sentence.lower().find(split_phrase)
        if split_idx < 0:
            expanded.append(dict(entry))
            continue
        prep_clause = normalize_text(first_sentence[:split_idx]).rstrip(". ")
        design_clause = normalize_text(first_sentence[split_idx + len(" and ") :]).rstrip(". ")
        prep_text = normalize_text(f"{prefix} {prep_clause}. {remainder}")
        design_text = normalize_text(f"Experimental design: {design_clause}.")
        if not prep_text or not design_text:
            expanded.append(dict(entry))
            continue
        base_entry = dict(entry)
        expanded.append(
            {
                **base_entry,
                "segment_index": int(base_entry.get("segment_index", 0)) * 10,
                "text": prep_text,
                "split_trigger": "embedded_design_split",
            }
        )
        expanded.append(
            {
                **base_entry,
                "segment_index": int(base_entry.get("segment_index", 0)) * 10 + 1,
                "text": design_text,
                "split_trigger": "embedded_design_split",
            }
        )
    return expanded


def split_inline_table_result_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for entry in entries:
        text = normalize_text(entry.get("text"))
        if not text:
            continue
        sentences = split_sentences_for_segmentation(text)
        table_sentence_indexes = [
            idx
            for idx, sentence in enumerate(sentences)
            if re.search(r"\btable\s+\d+\b", sentence, flags=re.I)
        ]
        if len(sentences) <= 1 or not table_sentence_indexes:
            expanded.append(dict(entry))
            continue
        chunks: list[tuple[str, str]] = []
        current: list[str] = []
        current_trigger = str(entry.get("split_trigger") or "paragraph_boundary")
        for idx, sentence in enumerate(sentences):
            if idx in table_sentence_indexes:
                if current:
                    chunks.append((" ".join(current), current_trigger))
                    current = []
                chunks.append((sentence, "table_inline_split"))
                current_trigger = "post_table_split"
                continue
            current.append(sentence)
        if current:
            chunks.append((" ".join(current), current_trigger))
        if len(chunks) <= 1:
            expanded.append(dict(entry))
            continue
        for segment_index, (chunk_text, chunk_trigger) in enumerate(chunks):
            cleaned_chunk = normalize_text(chunk_text)
            if not cleaned_chunk:
                continue
            expanded.append(
                {
                    **dict(entry),
                    "segment_index": int(entry.get("segment_index", 0)) * 10 + segment_index,
                    "text": cleaned_chunk,
                    "split_trigger": chunk_trigger,
                }
            )
    return expanded


def build_sparse_sentence_window_entries(text: str) -> list[dict[str, Any]]:
    sentences = split_sentences_for_segmentation(text)
    if not sentences:
        return []
    entries: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for idx, sentence in enumerate(sentences):
        if sentence_profile(sentence) not in {"preparation", "materials"}:
            continue
        start = max(0, idx - 1)
        end = min(len(sentences), idx + 3)
        chunk = normalize_text(" ".join(sentences[start:end]))
        if len(chunk) < 80 or chunk in seen_chunks:
            continue
        section_label = extract_section_label(chunk)
        section_kind = infer_section_kind(chunk, section_label)
        if should_drop_segment(chunk, section_kind, section_label):
            continue
        seen_chunks.add(chunk)
        entries.append(
            {
                "paragraph_index": 100000 + idx,
                "segment_index": 0,
                "text": chunk,
                "noise_flags": [],
                "split_trigger": "sparse_sentence_window",
            }
        )
    return entries


def build_segmented_paragraph_entries(text: str) -> list[dict[str, Any]]:
    entries = split_paragraph_entries(text)
    entries = split_section_scoped_entries(entries)
    entries = split_inline_heading_entries(entries)
    entries = split_transition_entries(entries)
    entries = split_embedded_design_entries(entries)
    entries = split_inline_table_result_entries(entries)
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        cleaned = normalize_text(entry.get("text"))
        if not cleaned:
            continue
        section_label = extract_section_label(cleaned)
        section_kind = infer_section_kind(cleaned, section_label)
        if should_drop_segment(cleaned, section_kind, section_label):
            continue
        filtered.append(
            {
                "paragraph_index": entry.get("paragraph_index", 0),
                "segment_index": entry.get("segment_index", 0),
                "text": cleaned,
                "noise_flags": list(entry.get("noise_flags") or []),
                "split_trigger": str(entry.get("split_trigger") or "paragraph_boundary"),
            }
        )
    return filtered


def has_variable_sweep_design_signal(text: str) -> bool:
    lower = normalize_text(text).lower()
    if any(token in lower for token in ["experimental design", "box-behnken", "response surface", "design expert", "coded levels"]):
        return True
    fixed_patterns = [
        r"\b\w*xed amounts of polymer\b",
        r"\b\w*xed amounts of surfactant[s]?\b",
        r"\b\w*xed amounts of polymer and surfactant[s]?\b",
        r"\bprepared using \w*xed amounts\b",
        r"\balways maintained at\b",
        r"\bmaintained at\b",
        r"\bkept constant\b",
        r"\bheld constant\b",
        r"\bwhile .*? was changed\b",
    ]
    varying_patterns = [
        r"\bvariable quantities\b",
        r"\bincreasing theoretical concentrations\b",
        r"\bdifferent volumes\b",
        r"\bdifferent amounts\b",
        r"\bmultiple batches\b",
        r"\bcryoprotectant concentrations\b",
        r"\bvarying concentrations?\b",
        r"\bconcentrations ranging\b",
        r"\bincreasing the volume\b",
        r"\branging from\s+\d",
        r"\bchanged from\s+\d",
        r"\bdifferent ratios?\b",
        r"\binitial ratios?\b",
        r"\bamount of [a-z0-9/\-() ]+ was changed from\b",
    ]
    has_fixed = any(re.search(pattern, lower) for pattern in fixed_patterns)
    has_varying = any(re.search(pattern, lower) for pattern in varying_patterns)
    if has_fixed and has_varying:
        return True
    explicit_series_patterns = [
        r"\(\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){2,}(?:\s*(?:and|or)\s*\d+(?:\.\d+)?)?\s*(?:mg|%|ml|mL|ratio)?\s*\)",
        r"\b\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*(?:and|or)\s*\d+(?:\.\d+)?\b",
    ]
    has_explicit_series = any(re.search(pattern, lower) for pattern in explicit_series_patterns)
    if has_varying and has_explicit_series:
        return True
    return False


def row_labels_show_variable_series(row_labels: list[str]) -> bool:
    groups: dict[str, set[str]] = {}
    for raw_label in row_labels:
        label = normalize_text(raw_label)
        if not label:
            continue
        match = re.match(r"^(.*?)(\d+(?:\.\d+)?)(?:[A-Za-z])?$", label)
        if not match:
            continue
        prefix = normalize_text(match.group(1)).lower()
        value = normalize_text(match.group(2))
        if len(prefix) < 4:
            continue
        groups.setdefault(prefix, set()).add(value)
    return any(len(values) >= 3 for values in groups.values())


def has_variable_sweep_structure(
    text: str,
    *,
    row_labels: list[str] | None = None,
) -> bool:
    if has_variable_sweep_design_signal(text):
        return True
    lower = normalize_text(text).lower()
    direct_patterns = [
        r"\bdifferent amounts\b",
        r"\bdifferent ratios?\b",
        r"\binitial ratios?\b",
        r"\bvar(?:y|ied|ying)\b.{0,40}\b(?:concentration|ratio|amount|level)s?\b",
        r"\bchanged from\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b",
        r"\bamount of [a-z0-9/\-() ]+ was changed from\b",
    ]
    if any(re.search(pattern, lower) for pattern in direct_patterns):
        return True
    if row_labels and row_labels_show_variable_series(row_labels):
        return True
    return False


def build_inline_formulation_table_item(
    text_content: str,
    *,
    text_path: Path,
    paragraph_index: int,
    segment_index: int,
) -> dict[str, Any] | None:
    lower = normalize_text(text_content).lower()
    if "table " not in lower:
        return None
    header_patterns = [
        (r"\bsample\b", "Sample"),
        (r"\btheoretical(?:\s+concen(?:tration)?)?\b", "Theoretical concentration"),
        (r"\bfinal\s+concen(?:tration)?\b", "Final concentration"),
        (r"\bformulation\b", "Formulation"),
        (r"\bformulation number\b", "Formulation Number"),
        (r"\bdrug:polymer ratio\b", "Drug:Polymer ratio"),
        (r"\bpolymer used\b", "Polymer Used"),
        (r"\bsurfactant\b", "Surfactant"),
        (r"\brhodamine\b", "Rhodamine"),
        (r"\bgatifloxacin\b", "Gatifloxacin"),
        (r"\bpolysorbate\s*80\b", "Polysorbate 80"),
        (r"\blabrafil\b", "Labrafil"),
        (r"\bparticle size\b", "Particle size"),
        (r"\bpdi\b", "PDI"),
        (r"\bzeta(?:\s|-)?potential\b", "Zeta potential"),
        (r"\bencapsulation efficiency\b", "Encapsulation efficiency"),
        (r"\bincorporation efficiency\b", "Incorporation efficiency"),
        (r"\bloading capacity\b", "Loading capacity"),
        (r"\bdrug loading\b", "Drug loading"),
        (r"\bdl\b", "DL"),
        (r"\bee\b", "EE"),
        (r"\bentrapment\b", "Entrapment"),
    ]
    header_labels: list[str] = []
    for pattern, label in header_patterns:
        if re.search(pattern, text_content, flags=re.I) and label.lower() not in {item.lower() for item in header_labels}:
            header_labels.append(label)
    row_labels = list(
        dict.fromkeys(
            match.group(0)
            for match in re.finditer(r"\b(?:NP[BRG]?\d{1,2}|NPR\d{1,2}|NPB\d{1,2}|NPG\d{1,2}|F\d{1,3})\b", text_content)
        )
    )
    numeric_row_labels = list(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(r"(?:^|\s)(\d{2,4}(?:\.\d+)?)\s+(?:\d{1,4}(?:\.\d+)?G\d{1,4}(?:\.\d+)?|\d{1,4}(?:\.\d+)?%)", text_content)
        )
    )
    numeric_token_count = len(re.findall(r"\b\d{1,4}(?:\.\d+)?(?:G\d{1,4}(?:\.\d+)?)?%?\b", text_content))
    if len(row_labels) < 2 and len(numeric_row_labels) >= 2:
        row_labels = [f"run_{label}" for label in numeric_row_labels[:12]]
    if len(row_labels) < 2:
        if not (len(header_labels) >= 3 and numeric_token_count >= 8):
            return None
        row_labels = [f"row_{idx:02d}" for idx in range(1, min(6, max(3, numeric_token_count // 4)) + 1)]
    if len(header_labels) < 3:
        return None
    caption_match = re.search(r"(Table\s+\d+[^.]{0,160})", text_content, flags=re.I)
    caption = normalize_text(caption_match.group(1)) if caption_match else "Inline formulation table"
    rows = [header_labels] + [[label] + [""] * (len(header_labels) - 1) for label in row_labels[:12]]
    return {
        "path": text_path,
        "rows": rows,
        "meta": {
            "caption_or_title": caption,
            "header_keywords_hit": ["table", "inline_recovered"] + header_labels,
            "n_rows": len(rows),
            "n_cols": len(header_labels),
            "page_number": "",
            "fraction_numeric_cells": 0.08,
        },
        "score": 90 + min(24, len(row_labels) * 3),
        "row_pattern": infer_row_pattern(row_labels),
        "quality_flags": ["inline_formulation_table_recovered"],
        "filtered_noise_rows": 0,
        "origin_locator": f"{to_repo_rel(text_path)}#paragraph:{paragraph_index}#segment:{segment_index}",
    }


def render_table_text(table_dir: Path | None, max_tables: int = 4, max_lines_per_table: int = 24) -> str:
    if table_dir is None or not table_dir.exists():
        return ""
    blocks: list[str] = []
    for path in sorted(table_dir.glob("*.csv"))[:max_tables]:
        lines: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for idx, row in enumerate(reader):
                if idx >= max_lines_per_table:
                    break
                lines.append(" | ".join(normalize_text(cell) for cell in row if normalize_text(cell)))
        if lines:
            rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
            blocks.append(f"[TABLE {rel}]\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def render_full_table_block(path: Path, *, max_lines_per_table: int | None = 24) -> str:
    raw_rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            raw_rows.append(list(row))
    cleaned_rows, _, _ = clean_table_rows(raw_rows)
    cleaned_rows, _ = compact_table_rows_for_evidence(cleaned_rows)
    lines: list[str] = []
    for idx, row in enumerate(cleaned_rows):
        if max_lines_per_table is not None and idx >= max_lines_per_table:
            break
        compact = " | ".join(normalize_text(cell) for cell in row if normalize_text(cell))
        if compact:
            lines.append(compact)
    if not lines:
        return ""
    return f"[TABLE {to_repo_rel(path)}]\n" + "\n".join(lines)


def table_mode() -> str:
    # Locked S2-4a contract: all LLM-facing table evidence remains summary-only.
    mode = normalize_text(os.getenv("STAGE2_TABLE_MODE", "summary")).lower()
    return "summary" if mode in {"", "summary", "full"} else "summary"


def summary_first_column_enhancement_enabled() -> bool:
    value = normalize_text(os.getenv(SUMMARY_FIRST_COLUMN_ENHANCEMENT_ENV, "")).lower()
    return value in {"1", "true", "yes", "on"}


def parse_numeric_row_label(value: Any) -> int | None:
    text = normalize_text(value)
    match = re.fullmatch(r"(\d{1,3})\s*\.?", text)
    if match:
        return int(match.group(1))
    return None


def summarize_row_anchor_preview(row_ids: list[str], limit: int = 12) -> str:
    cleaned_ids = [normalize_text(value) for value in row_ids if normalize_text(value)]
    if not cleaned_ids:
        return ""
    numeric_ids: list[int] = []
    for value in cleaned_ids:
        parsed = parse_numeric_row_label(value)
        if parsed is None:
            return ", ".join(cleaned_ids[:limit]) + (" ..." if len(cleaned_ids) > limit else "")
        numeric_ids.append(parsed)
    if numeric_ids == list(range(1, len(numeric_ids) + 1)):
        return f"1..{numeric_ids[-1]}"
    if len(numeric_ids) <= limit:
        return ", ".join(str(value) for value in numeric_ids)
    return ", ".join(str(value) for value in numeric_ids[:limit]) + f", ... ({len(numeric_ids)} rows)"


def build_summary_table_signal_text(
    rows: list[list[str]],
    meta: dict[str, Any],
    header_parts: list[str],
) -> str:
    parts: list[str] = []
    parts.extend(header_parts)
    parts.extend(normalize_text(item) for item in (meta.get("header_keywords_hit") or []))
    parts.append(normalize_text(meta.get("caption_or_title")))
    parts.append(normalize_text(meta.get("footnotes") or meta.get("notes")))
    for row in rows:
        parts.extend(normalize_text(cell) for cell in row if normalize_text(cell))
    return " ".join(part for part in parts if part).lower()


def extract_informative_header_parts(rows: list[list[str]]) -> list[str]:
    header_parts: list[str] = []
    for row in rows[:6]:
        row_text = " ".join(normalize_text(cell) for cell in row if normalize_text(cell))
        if not row_text:
            continue
        if is_reference_like_text(row_text) or is_front_matter_like_text(row_text):
            continue
        for cell in row:
            header_parts.extend(parse_header_cell(cell))
        if re.search(r"\b(?:\d+|f\d+|run\s+\d+)\b", row_text.lower()) and header_parts:
            break
    return [part for part in header_parts if part]


def score_summary_table(path: Path, rows: list[list[str]], meta: dict[str, Any]) -> tuple[int, str]:
    header_row = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    header_parts = extract_informative_header_parts(rows)
    if not header_parts and header_row:
        for cell in header_row:
            header_parts.extend(parse_header_cell(cell))
        header_parts = [part for part in header_parts if part]
    row_ids = [row[0] for row in data_rows if row and normalize_text(row[0])]
    row_pattern = infer_row_pattern(row_ids)
    role_hint = infer_table_role_hint(header_parts, meta)
    numeric_rows = sum(1 for value in row_ids if parse_numeric_row_label(value) is not None)
    numeric_ratio = numeric_rows / max(len(data_rows), 1)
    signal_text = build_summary_table_signal_text(rows, meta, header_parts)
    formulation_schema_hit = has_strong_formulation_table_signal(signal_text)
    figure_carryover_hit = is_obvious_figure_or_front_matter_table(signal_text)
    score = 0
    if row_pattern in {"numeric runs", "F-numbered rows"}:
        score += 100
    if role_hint == "design matrix":
        score += 80
    if role_hint == "formulation":
        score += 95
    if role_hint == "optimization":
        score += 50
    if role_hint == "characterization_only":
        score -= 90
    if numeric_ratio >= 0.5:
        score += 60
    if len(data_rows) >= 8:
        score += min(20, len(data_rows) // 2)
    if meta.get("header_keywords_hit"):
        score += 10
    if any(token in " ".join(header_parts).lower() for token in ["design", "factor", "doe", "optimization", "formulation"]):
        score += 15
    if "selected" in signal_text:
        score += 35
    if "optimal" in signal_text:
        score += 25
    if "note:" in signal_text:
        score += 30
    if "chosen" in signal_text:
        score += 20
    if "remaining studies" in signal_text:
        score += 35
    if "whole study" in signal_text:
        score += 25
    if "mg/ml" in signal_text:
        score += 25
    if "ratio" in signal_text:
        score += 15
    if "concentration" in signal_text:
        score += 12
    if "formulation" in signal_text:
        score += 10
    if formulation_schema_hit:
        score += 35
    if formulation_schema_hit and any(
        token in signal_text for token in ["theoretical concentration", "final concentration", "formulation characters"]
    ):
        score += 45
    if "poloxamer 188" in signal_text:
        score += 20
    if "plga:itz" in signal_text:
        score += 15
    if is_characterization_only_table_signal(signal_text):
        score -= 100
    if figure_carryover_hit:
        score -= 140
    if any(token in signal_text for token in ["et al", "dovepress", "references", "submit your manuscript", "international journal of nanomedicine"]):
        score -= 60
    if any(token in signal_text for token in ["purpose:", "abstract", "correspondence:", "college of pharmacy", "department of"]):
        score -= 80
    if any(token in signal_text for token in ["pharmacokinetic", "lc-ms/ms", "hplc", "chromatogram", "calibration curve", "assay"]):
        score -= 20
    prose_like_rows = sum(
        1
        for row in data_rows[:20]
        if sum(len(normalize_text(cell)) for cell in row if normalize_text(cell)) >= 60
    )
    if len(header_parts) <= 2 and numeric_ratio < 0.15 and prose_like_rows >= 5:
        score -= 45
    restore_profile = wfdtq4vx_restore_profile(path, meta)
    if restore_profile == "wfdtq4vx_doe_execution_table":
        score += 420
    elif restore_profile == "wfdtq4vx_doe_companion_table":
        score += 80
    elif restore_profile == "wfdtq4vx_optimization_support_table":
        score += 25
    score += min(10, len(header_parts))
    return score, row_pattern


def clean_table_rows(rows: list[list[str]], meta: dict[str, Any] | None = None) -> tuple[list[list[str]], list[str], int]:
    cleaned_rows: list[list[str]] = []
    quality_flags: list[str] = []
    filtered_noise_rows = 0
    meta = meta or {}
    caption = normalize_text(meta.get("caption_or_title")).lower()
    if caption.startswith("figure "):
        quality_flags.append("figure_like_caption")
    for row in rows:
        cleaned_row = [normalize_text(cell) for cell in row if normalize_text(cell)]
        if not cleaned_row:
            continue
        row_text = " ".join(cleaned_row)
        lower = row_text.lower()
        alpha_count = sum(ch.isalpha() for ch in row_text)
        cid_count = row_text.count("(cid:")
        prose_like = len(lower.split()) >= 18 and not lower.startswith("note:")
        if is_obvious_noise_line(row_text):
            filtered_noise_rows += 1
            continue
        if is_reference_like_text(row_text):
            filtered_noise_rows += 1
            if "reference_rows_filtered" not in quality_flags:
                quality_flags.append("reference_rows_filtered")
            continue
        if cid_count >= 2:
            filtered_noise_rows += 1
            if "encoding_artifact_rows_filtered" not in quality_flags:
                quality_flags.append("encoding_artifact_rows_filtered")
            continue
        if any(token in lower for token in ["figure ", "plasma concentration-time profiles", "publish your work", "submit your manuscript", "dovepress", "biomed research international", "powered by tcpdf"]):
            filtered_noise_rows += 1
            if "noise_rows_filtered" not in quality_flags:
                quality_flags.append("noise_rows_filtered")
            continue
        if prose_like and any(token in lower for token in SEGMENT_PROSE_CARRYOVER_CUES) and alpha_count >= 80:
            filtered_noise_rows += 1
            if "prose_carryover_filtered" not in quality_flags:
                quality_flags.append("prose_carryover_filtered")
            continue
        cleaned_rows.append(cleaned_row)
    if filtered_noise_rows:
        quality_flags.append("noise_rows_filtered")
    if len(cleaned_rows) < 2:
        quality_flags.append("corrupted_or_sparse_table")
    return cleaned_rows, quality_flags, filtered_noise_rows


def is_placeholder_header(cell: str) -> bool:
    lower = normalize_text(cell).lower()
    return not lower or lower.startswith("unnamed:")


def compact_table_rows_for_evidence(rows: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    if not rows:
        return rows, []
    actions: list[str] = []
    max_width = max((len(row) for row in rows if isinstance(row, list)), default=0)
    if max_width <= 0:
        return rows, actions

    normalized_rows = [list(row) + [""] * (max_width - len(row)) for row in rows]
    keep_indices: list[int] = []
    for col_index in range(max_width):
        column = [normalize_text(row[col_index]) for row in normalized_rows]
        header = column[0]
        body = column[1:]
        informative_body = any(value for value in body)
        if not is_placeholder_header(header):
            keep_indices.append(col_index)
            continue
        if informative_body:
            unique_body = {value for value in body if value}
            if len(unique_body) > 1:
                keep_indices.append(col_index)
    if keep_indices and len(keep_indices) < max_width:
        normalized_rows = [[row[index] for index in keep_indices] for row in normalized_rows]
        actions.append("drop_sparse_placeholder_columns")

    compacted_rows: list[list[str]] = []
    for row in normalized_rows:
        cleaned = [normalize_text(cell) for cell in row]
        non_empty = [cell for cell in cleaned if cell]
        if not non_empty:
            continue
        unique_non_empty = list(dict.fromkeys(non_empty))
        repeated_long_note = (
            len(non_empty) >= 3
            and len(unique_non_empty) == 1
            and len(unique_non_empty[0].split()) >= 12
        )
        if repeated_long_note:
            compacted_rows.append([unique_non_empty[0]])
            if "collapse_repeated_note_row" not in actions:
                actions.append("collapse_repeated_note_row")
            continue
        compacted_rows.append(non_empty)
    return compacted_rows, actions


def table_manifest_path(table_dir: Path | None) -> str:
    if table_dir is None:
        return ""
    manifest_path = table_dir / "tables_manifest.json"
    return to_repo_rel(manifest_path) if manifest_path.exists() else ""


def selected_table_files(table_dir: Path | None) -> set[str]:
    payload = load_table_manifest_payload(table_dir)
    selected = {
        normalize_text(value)
        for value in ensure_list(payload.get("selected_table_files"))
        if normalize_text(value)
    }
    return selected


def candidate_surface_preview(rows: list[list[str]], *, max_rows: int = 4, max_cells: int = 5) -> str:
    lines: list[str] = []
    for row in rows[:max_rows]:
        compact = [normalize_text(cell) for cell in row[:max_cells] if normalize_text(cell)]
        if compact:
            lines.append(" | ".join(compact))
    return normalize_text(" || ".join(lines))


def looks_like_header_row(row: list[str]) -> bool:
    compact = [normalize_text(cell) for cell in row if normalize_text(cell)]
    if not compact:
        return False
    lower = " ".join(compact).lower()
    if any(token in lower for token in ["formulation", "ratio", "polymer", "surfactant", "particle size", "pdi", "zeta", "entrapment", "loading", "run"]):
        return True
    alpha_cells = sum(1 for cell in compact if re.search(r"[A-Za-z]", cell))
    numeric_cells = sum(1 for cell in compact if re.search(r"\d", cell))
    return alpha_cells >= max(2, numeric_cells)


def recover_caption_from_rows(rows: list[list[str]]) -> tuple[str, int]:
    for idx, row in enumerate(rows[:3]):
        row_text = normalize_text(" ".join(cell for cell in row if normalize_text(cell)))
        if re.match(r"^table\s+\d+\b", row_text, flags=re.I):
            return row_text, idx
    return "", -1


def repair_table_representation(
    *,
    path: Path,
    rows: list[list[str]],
    meta: dict[str, Any],
    quality_flags: list[str],
    filtered_noise_rows: int,
    table_dir: Path | None,
) -> dict[str, Any]:
    repair_actions: list[str] = []
    repair_warnings: list[str] = []
    repaired_meta = dict(meta)
    repaired_rows = [list(row) for row in rows]
    raw_preview = candidate_surface_preview(rows)
    repair_primary_source = "candidate_surface_fallback"
    selected_files = selected_table_files(table_dir)
    manifest_rel = table_manifest_path(table_dir)
    if not selected_files or path.name in selected_files:
        repair_primary_source = "stage1_selected_table_asset"
    else:
        repair_warnings.append("not_in_selected_table_files")

    caption = normalize_text(repaired_meta.get("caption_or_title"))
    if not caption:
        recovered_caption, caption_row_index = recover_caption_from_rows(repaired_rows)
        if recovered_caption:
            repaired_meta["caption_or_title"] = recovered_caption
            repair_actions.append("caption_recovery")
            if caption_row_index == 0 and len(repaired_rows) > 1:
                repaired_rows = repaired_rows[1:]

    if repaired_rows and not looks_like_header_row(repaired_rows[0]) and len(repaired_rows) > 1 and looks_like_header_row(repaired_rows[1]):
        repaired_rows = [repaired_rows[1]] + repaired_rows[:1] + repaired_rows[2:]
        repair_actions.append("header_repair")

    if len(repaired_rows) >= 2:
        first_column_values = [normalize_text(row[0]) for row in repaired_rows[1:] if row and normalize_text(row[0])]
        if first_column_values:
            repair_actions.append("first_column_preservation")
    if filtered_noise_rows:
        repair_actions.append("spillover_trim")
    if "corrupted_or_sparse_table" in {normalize_text(flag) for flag in quality_flags} and len(repaired_rows) >= 2:
        repair_actions.append("sparse_table_rescue")

    repaired_preview = candidate_surface_preview(repaired_rows)
    material_difference = repaired_preview != raw_preview
    role_hint = infer_table_role_hint(extract_informative_header_parts(repaired_rows), {**repaired_meta, "_signal_text": build_summary_table_signal_text(repaired_rows[:6], repaired_meta, extract_informative_header_parts(repaired_rows))})
    unresolved_reason = ""
    if len(repaired_rows) < 2:
        unresolved_reason = "insufficient_rows_after_repair"
        repair_warnings.append(unresolved_reason)
    elif role_hint == "unknown" and not repaired_preview:
        unresolved_reason = "empty_repaired_preview"
        repair_warnings.append(unresolved_reason)
    elif role_hint == "unknown":
        unresolved_reason = "role_legibility_still_weak"

    representation_status = "repaired_summary" if repair_actions else "raw_summary"
    if unresolved_reason:
        representation_status = "unrepaired_corrupted" if not repair_actions else "repair_insufficient"
    selector_readiness = "ready" if repair_primary_source == "stage1_selected_table_asset" and repaired_preview and not unresolved_reason else "weak"
    if unresolved_reason and len(repaired_rows) < 2:
        selector_readiness = "unresolved"
    repair_confidence = 0.9 if selector_readiness == "ready" and repair_primary_source == "stage1_selected_table_asset" else 0.45

    return {
        "path": path,
        "rows": repaired_rows,
        "meta": repaired_meta,
        "quality_flags": quality_flags,
        "filtered_noise_rows": filtered_noise_rows,
        "raw_table_preview": raw_preview,
        "repaired_table_preview": repaired_preview or raw_preview,
        "representation_status": representation_status,
        "repair_actions": list(dict.fromkeys(repair_actions)),
        "repair_warnings": list(dict.fromkeys(repair_warnings)),
        "repair_confidence": repair_confidence,
        "material_difference_from_raw": material_difference,
        "repair_primary_source": repair_primary_source,
        "repair_source_csv_path": to_repo_rel(path),
        "repair_source_manifest_path": manifest_rel,
        "repair_source_candidate_id": "",
        "candidate_variant_role": "raw",
        "same_source_table_asset": True,
        "derived_from_candidate_id": "",
        "selector_readiness_label": selector_readiness,
        "unresolved_reason": unresolved_reason,
    }


def collect_summary_table_candidates(table_dir: Path) -> list[dict[str, Any]]:
    manifest = load_table_manifest(table_dir)
    selected_files = selected_table_files(table_dir)
    table_paths = sorted(
        path
        for path in table_dir.glob("*.csv")
        if not selected_files or path.name in selected_files
    )
    entries: list[dict[str, Any]] = []
    for path in table_paths:
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                cleaned_row = [normalize_text(cell) for cell in row]
                if any(cleaned_row):
                    rows.append(cleaned_row)
        if not rows:
            continue
        meta = manifest.get(path.name, {})
        rows, quality_flags, filtered_noise_rows = clean_table_rows(rows, meta)
        if not rows:
            continue
        repaired = repair_table_representation(
            path=path,
            rows=rows,
            meta=meta,
            quality_flags=quality_flags,
            filtered_noise_rows=filtered_noise_rows,
            table_dir=table_dir,
        )
        rows = repaired["rows"]
        meta = repaired["meta"]
        header_parts = extract_informative_header_parts(rows)
        signal_text = build_summary_table_signal_text(rows[:6], meta, header_parts)
        role_hint = infer_table_role_hint(header_parts, {**meta, "_signal_text": signal_text})
        if role_hint == "design matrix":
            meta = {**meta, "caption_or_title": f"Experimental design variable table. {normalize_text(meta.get('caption_or_title'))}".strip()}
        elif role_hint == "formulation":
            meta = {**meta, "caption_or_title": f"Formulation table with drug/polymer/loading variables. {normalize_text(meta.get('caption_or_title'))}".strip()}
        elif role_hint == "optimization":
            meta = {**meta, "caption_or_title": f"Optimization result table. {normalize_text(meta.get('caption_or_title'))}".strip()}
        elif role_hint == "characterization_only":
            meta = {**meta, "caption_or_title": f"Characterization-only table. {normalize_text(meta.get('caption_or_title'))}".strip()}
        score, row_pattern = score_summary_table(path, rows, meta)
        restoration_profile = wfdtq4vx_restore_profile(path, meta)
        entries.append(
            {
                "path": path,
                "rows": rows,
                "meta": meta,
                "score": score,
                "row_pattern": row_pattern,
                "quality_flags": quality_flags,
                "filtered_noise_rows": filtered_noise_rows,
                "representation_status": repaired["representation_status"],
                "repair_actions": repaired["repair_actions"],
                "repair_warnings": repaired["repair_warnings"],
                "repair_confidence": repaired["repair_confidence"],
                "raw_table_preview": repaired["raw_table_preview"],
                "repaired_table_preview": repaired["repaired_table_preview"],
                "material_difference_from_raw": repaired["material_difference_from_raw"],
                "repair_primary_source": repaired["repair_primary_source"],
                "repair_source_csv_path": repaired["repair_source_csv_path"],
                "repair_source_manifest_path": repaired["repair_source_manifest_path"],
                "repair_source_candidate_id": repaired["repair_source_candidate_id"],
                "candidate_variant_role": repaired["candidate_variant_role"],
                "same_source_table_asset": repaired["same_source_table_asset"],
                "derived_from_candidate_id": repaired["derived_from_candidate_id"],
                "selector_readiness_label": repaired["selector_readiness_label"],
                "unresolved_reason": repaired["unresolved_reason"],
                "restoration_profile": restoration_profile,
                "full_prompt_table_authority": restoration_profile == "wfdtq4vx_doe_execution_table",
            }
        )
    return rank_table_authority_payloads(entries)


def select_summary_tables(
    table_dir: Path,
    *,
    max_tables: int,
) -> list[dict[str, Any]]:
    return collect_summary_table_candidates(table_dir)[:max_tables]


def build_table_selection_debug_payload(
    *,
    document_key: str,
    table_dir: Path | None,
    max_tables: int,
    summary_enhanced: bool,
) -> dict[str, Any] | None:
    if table_dir is None or not table_dir.exists():
        return None
    candidates = collect_summary_table_candidates(table_dir)
    selected_names = {item["path"].name for item in candidates[:max_tables]}
    payload_candidates: list[dict[str, Any]] = []
    for item in candidates:
        rows = item["rows"]
        first_data_row = rows[1] if len(rows) > 1 else rows[0]
        payload_candidates.append(
            {
                "file": item["path"].name,
                "score": item["score"],
                "selected": item["path"].name in selected_names,
                "row_pattern": item["row_pattern"],
                "quality_flags": item.get("quality_flags") or [],
                "page_number": normalize_text(item["meta"].get("page_number")),
                "n_rows": item["meta"].get("n_rows", len(rows)),
                "n_cols": item["meta"].get("n_cols", max((len(r) for r in rows), default=0)),
                "caption_or_title": normalize_text(item["meta"].get("caption_or_title")),
                "header_keywords_hit": item["meta"].get("header_keywords_hit") or [],
                "first_data_row_preview": " | ".join(cell for cell in first_data_row if cell),
                "representation_status": normalize_text(item.get("representation_status")),
                "repair_actions": item.get("repair_actions") or [],
                "repair_primary_source": normalize_text(item.get("repair_primary_source")),
                "selector_readiness_label": normalize_text(item.get("selector_readiness_label")),
                "unresolved_reason": normalize_text(item.get("unresolved_reason")),
            }
        )
    return {
        "document_key": document_key,
        "table_mode": table_mode(),
        "selection_ranking_mode": "score_ranked_top_k",
        "summary_first_column_enhancement": "yes" if summary_enhanced else "no",
        "max_tables": max_tables,
        "selected_tables": [item["path"].name for item in candidates[:max_tables]],
        "candidates": payload_candidates,
    }


def load_table_manifest(table_dir: Path | None) -> dict[str, dict[str, Any]]:
    payload = load_table_manifest_payload(table_dir)
    if isinstance(payload, dict):
        tables = payload.get("tables") or []
    else:
        tables = []
    manifest: dict[str, dict[str, Any]] = {}
    for item in tables:
        if not isinstance(item, dict):
            continue
        csv_path = normalize_text(item.get("csv_path"))
        if not csv_path:
            continue
        manifest[Path(csv_path).name] = item
    return manifest


def parse_header_cell(cell: str) -> list[str]:
    text = normalize_text(cell)
    if not text:
        return []
    if text.startswith("(") and text.endswith(")"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, tuple):
                return [normalize_text(part) for part in parsed if normalize_text(part)]
        except Exception:
            return [text]
    return [text]


def derive_stable_table_id(source_csv_path: str, meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    explicit = normalize_text(meta.get("table_id"))
    if explicit:
        return explicit
    caption = normalize_text(meta.get("caption_or_title"))
    match = re.search(r"\btable\s+(\d+)\b", caption, flags=re.IGNORECASE)
    if match:
        return f"Table {int(match.group(1))}"
    path = Path(normalize_text(source_csv_path))
    stem_match = re.search(r"__table_(\d+)__", path.name, flags=re.IGNORECASE)
    if stem_match:
        return f"Table {int(stem_match.group(1))}"
    return path.stem or "unknown_table"


def wfdtq4vx_restore_profile(path: Path, meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    source_hint = " ".join(
        [
            path.name,
            normalize_text(meta.get("csv_path")),
            normalize_text(meta.get("caption_or_title")),
        ]
    ).upper()
    if "WFDTQ4VX" not in source_hint:
        return ""
    stem_match = re.search(r"__table_(\d+)__", path.name, flags=re.I)
    table_number = stem_match.group(1) if stem_match else ""
    if not table_number:
        table_id = derive_stable_table_id(str(path), meta).lower()
        table_match = re.search(r"\btable\s+(\d+)\b", table_id, flags=re.I)
        table_number = table_match.group(1) if table_match else ""
    if table_number == "12":
        return "wfdtq4vx_doe_execution_table"
    if table_number == "13":
        return "wfdtq4vx_doe_companion_table"
    if table_number == "15":
        return "wfdtq4vx_optimization_support_table"
    return ""


def table_restore_profile(candidate_or_item: dict[str, Any]) -> str:
    item = candidate_or_item.get("item") if isinstance(candidate_or_item.get("item"), dict) else candidate_or_item
    if not isinstance(item, dict):
        return ""
    explicit = normalize_text(item.get("restoration_profile") or candidate_or_item.get("restoration_profile"))
    if explicit:
        return explicit
    meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
    source_csv_path = normalize_text(item.get("repair_source_csv_path")) or normalize_text(candidate_or_item.get("origin_locator"))
    path = Path(source_csv_path or normalize_text(candidate_or_item.get("origin_locator")))
    return wfdtq4vx_restore_profile(path, meta)


def candidate_prefers_full_table_prompt(candidate: dict[str, Any]) -> bool:
    del candidate
    return False


def candidate_summary_is_lossy(candidate: dict[str, Any]) -> bool:
    return candidate.get("candidate_kind") == "table"


def render_selected_table_candidate(candidate: dict[str, Any]) -> str:
    # Locked S2-4a contract: selected table evidence stays summary-only.
    return normalize_text(candidate.get("text_content"))


def infer_units_from_headers(headers: list[str]) -> list[str]:
    units: list[str] = []
    seen: set[str] = set()
    for header in headers:
        for match in re.findall(r"\(([^()]+)\)", header):
            unit = normalize_text(match)
            if unit and unit.lower() not in seen:
                seen.add(unit.lower())
                units.append(unit)
    return units


def infer_row_pattern(row_ids: list[str]) -> str:
    cleaned = [normalize_text(value) for value in row_ids if normalize_text(value)]
    if not cleaned:
        return "unknown"
    if all(re.fullmatch(r"F\d+", value, flags=re.IGNORECASE) for value in cleaned):
        return "F-numbered rows"
    if all(re.fullmatch(r"\d+(?:\.\d+)?", value) for value in cleaned):
        return "numeric runs"
    if all(re.fullmatch(r"[A-Za-z]+\d+", value) for value in cleaned):
        return "letter-number identifiers"
    if len({value.lower() for value in cleaned}) <= 3:
        return "repeated categorical identifiers"
    return "mixed identifiers"


def infer_table_role_hint(headers: list[str], meta: dict[str, Any]) -> str:
    signals = " ".join(
        headers
        + [normalize_text(item) for item in (meta.get("header_keywords_hit") or [])]
        + [normalize_text(meta.get("caption_or_title")), normalize_text(meta.get("_signal_text"))]
    ).lower()
    if is_characterization_only_table_signal(signals):
        return "characterization_only"
    if any(token in signals for token in ["independent variables", "dependent variables", "factors", "levels"]):
        return "design matrix"
    if any(token in signals for token in ["cryoprotectant", "concentration (%", "concentration %", "different concentrations", "ranging from"]):
        return "design matrix"
    if any(token in signals for token in ["factorial", "coded", "design", "run", "doe", "cpf", "cpva", "cplga"]):
        return "design matrix"
    if has_strong_formulation_table_signal(signals):
        return "formulation"
    if any(token in signals for token in ["selected", "optimal", "optimized", "desirability"]):
        return "optimization"
    if any(token in signals for token in ["size", "pdi", "zeta", "entrapment", "efficiency", "loading"]):
        return "characterization"
    if any(token in signals for token in ["release", "stability", "yield", "response"]):
        return "results"
    return "unknown"


def table_authority_signal_text(item: dict[str, Any]) -> str:
    rows = item.get("rows") or []
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    header_parts = extract_informative_header_parts(rows)
    return build_summary_table_signal_text(rows[:10], meta, header_parts)


def table_authority_row_anchor_count(item: dict[str, Any]) -> int:
    rows = item.get("rows") or []
    if len(rows) <= 1:
        return 0
    return sum(
        1
        for row in rows[1:]
        if isinstance(row, list) and row and normalize_text(row[0])
    )


def table_prose_like_row_count(rows: list[list[str]]) -> int:
    prose_like = 0
    for row in rows[:8]:
        if not isinstance(row, list):
            continue
        compact_cells = [normalize_text(cell) for cell in row if normalize_text(cell)]
        if not compact_cells:
            continue
        char_count = sum(len(cell) for cell in compact_cells)
        numeric_like = sum(1 for cell in compact_cells if re.search(r"\d", cell))
        if char_count >= 80 and numeric_like <= 2:
            prose_like += 1
    return prose_like


def table_compact_anchor_count(item: dict[str, Any]) -> int:
    compact_count = 0
    for label in table_row_label_preview(item, limit=20):
        normalized = normalize_text(label)
        if not normalized:
            continue
        if len(normalized) <= 32 and (
            re.search(r"\d", normalized)
            or ":" in normalized
            or normalized.lower().startswith(("f", "run", "sample", "formulation"))
        ):
            compact_count += 1
    return compact_count


def clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def build_table_authority_score_breakdown(item: dict[str, Any]) -> dict[str, float]:
    rows = item.get("rows") or []
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    signal_text = table_authority_signal_text(item)
    role_hint = infer_table_role_hint(extract_informative_header_parts(rows), {**meta, "_signal_text": signal_text})
    row_pattern = infer_row_pattern(table_row_label_preview(item, limit=20))
    selector_readiness = normalize_text(item.get("selector_readiness_label")).lower()
    representation_status = normalize_text(item.get("representation_status")).lower()
    repair_primary_source = normalize_text(item.get("repair_primary_source")).lower()
    repair_confidence = float(item.get("repair_confidence") or 0.0)
    filtered_noise_rows = int(item.get("filtered_noise_rows") or 0)
    data_row_count = max(0, len(rows) - 1)
    row_anchor_count = table_authority_row_anchor_count(item)
    prose_like_rows = table_prose_like_row_count(rows)
    compact_anchor_count = table_compact_anchor_count(item)
    formulation_hits = count_cue_hits(signal_text, TABLE_AUTHORITY_FORMULATION_TOKENS)
    design_hits = count_cue_hits(signal_text, TABLE_AUTHORITY_DESIGN_TOKENS)
    downstream_hits = count_cue_hits(signal_text, TABLE_AUTHORITY_DOWNSTREAM_DEMOTION_TOKENS)
    stability_hits = count_cue_hits(signal_text, TABLE_AUTHORITY_STABILITY_DEMOTION_TOKENS)
    frontmatter_hits = count_cue_hits(signal_text, TABLE_AUTHORITY_FRONTMATTER_DEMOTION_TOKENS)
    figure_spillover = is_obvious_figure_or_front_matter_table(signal_text)
    front_matter_like = is_front_matter_like_text(signal_text)

    breakdown: dict[str, float] = {
        "legacy_recovery_score": round(clamp_float(float(item.get("score") or 0.0) / 40.0, -10.0, 12.0), 2),
        "selector_readiness": 12.0 if selector_readiness == "ready" else (-16.0 if selector_readiness == "unresolved" else -10.0),
        "representation_quality": (
            8.0
            if representation_status in {"repaired_summary", "raw_summary"}
            else (-18.0 if representation_status == "unrepaired_corrupted" else -12.0)
        ),
        "repair_confidence": round(repair_confidence * 10.0, 2),
        "authoritative_source": 5.0 if repair_primary_source == "stage1_selected_table_asset" else 0.0,
        "row_anchor_density": 10.0 if row_pattern in {"numeric runs", "F-numbered rows"} else (4.0 if row_anchor_count >= 4 else 0.0),
        "row_count_density": 6.0 if data_row_count >= 8 else (3.0 if data_row_count >= 3 else -2.0),
        "formulation_density": min(18.0, formulation_hits * 2.5),
        "design_density": min(16.0, design_hits * 3.0),
        "characterization_demotion": -20.0 if role_hint == "characterization_only" else (-4.0 if role_hint == "characterization" else 0.0),
        "downstream_demotion": max(-60.0, -10.0 * downstream_hits),
        "stability_demotion": max(-24.0, -8.0 * stability_hits),
        "frontmatter_demotion": max(-28.0, -7.0 * frontmatter_hits),
        "front_matter_like_demotion": -35.0 if front_matter_like else 0.0,
        "figure_spillover_demotion": -35.0 if figure_spillover else 0.0,
        "noise_row_demotion": max(-10.0, -1.5 * float(filtered_noise_rows)),
        "prose_row_demotion": max(-20.0, -5.0 * float(prose_like_rows)),
        "anchor_legibility_demotion": -18.0 if compact_anchor_count < 3 and prose_like_rows >= 3 else 0.0,
    }
    if role_hint == "design matrix":
        breakdown["role_hint_bonus"] = 10.0
    elif role_hint == "formulation":
        breakdown["role_hint_bonus"] = 12.0
    elif role_hint == "optimization":
        breakdown["role_hint_bonus"] = 2.0
    else:
        breakdown["role_hint_bonus"] = 0.0
    if any(token in signal_text for token in ["table 1", "table 2", "table 3"]):
        breakdown["early_table_bonus"] = 2.0
    else:
        breakdown["early_table_bonus"] = 0.0
    return breakdown


PRIMARY_EXCLUDED_ROLE_HINTS = {
    "characterization",
    "characterization_only",
    "results",
}


PRIMARY_EXCLUDED_PATTERN_TOKENS = {
    "release",
    "diffusion",
    "stability",
    "storage",
    "dialysis",
    "pharmacokinetic",
    "pharmacokinetics",
    "biodistribution",
    "cell viability",
    "assay",
    "figure",
}


PRIMARY_ELIGIBLE_HEADER_TOKENS = {
    "concentration",
    "ratio",
    "loading",
    "efficiency",
    "entrapment",
    "size",
    "zeta",
    "pdi",
    "diameter",
}


PRIMARY_LABEL_ONLY_REASONS = {
    "table_type_non_formulation_table",
    "table_role_hint_characterization",
    "table_role_hint_characterization_only",
    "table_role_hint_results",
}

TABLE_INCLUSION_MUST_INCLUDE = "must_include"
TABLE_INCLUSION_OPTIONAL_CONTEXT = "optional_context"
TABLE_INCLUSION_HARD_DROP = "hard_drop"

HARD_DROP_SIGNAL_TOKENS = {
    "for peer review",
    "received:",
    "accepted:",
    "published:",
    "correspondence:",
    "keywords:",
    "references",
}


def table_fraction_numeric_cells(item: dict[str, Any]) -> float:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    explicit = meta.get("fraction_numeric_cells")
    if explicit not in {None, ""}:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass
    rows = item.get("rows") or []
    nonempty_cells = 0
    numeric_cells = 0
    for row in rows[:12]:
        if not isinstance(row, list):
            continue
        for cell in row:
            normalized = normalize_text(cell)
            if not normalized:
                continue
            nonempty_cells += 1
            if re.search(r"\d", normalized):
                numeric_cells += 1
    if nonempty_cells <= 0:
        return 0.0
    return round(float(numeric_cells) / float(nonempty_cells), 4)


def infer_authority_table_type(item: dict[str, Any], *, role_hint: str, breakdown: dict[str, float]) -> str:
    if role_hint in {"design matrix", "formulation", "optimization"}:
        return "formulation_table"
    formulation_density = float(breakdown.get("formulation_density", 0.0) or 0.0)
    design_density = float(breakdown.get("design_density", 0.0) or 0.0)
    if max(formulation_density, design_density) >= 6.0:
        return "formulation_table"
    return "non_formulation_table"


def table_primary_guardrail_reasons(item: dict[str, Any], *, breakdown: dict[str, float]) -> list[str]:
    rows = item.get("rows") or []
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    signal_text = table_authority_signal_text(item)
    role_hint = infer_table_role_hint(extract_informative_header_parts(rows), {**meta, "_signal_text": signal_text})
    table_type = infer_authority_table_type(item, role_hint=role_hint, breakdown=breakdown)
    fraction_numeric_cells = table_fraction_numeric_cells(item)
    header_keywords_hit = [
        normalize_text(value)
        for value in ensure_list(meta.get("header_keywords_hit"))
        if normalize_text(value)
    ]
    prose_like_rows = table_prose_like_row_count(rows)
    row_anchor_count = table_authority_row_anchor_count(item)
    reasons: list[str] = []
    if table_type == "non_formulation_table":
        reasons.append("table_type_non_formulation_table")
    if role_hint in PRIMARY_EXCLUDED_ROLE_HINTS:
        reasons.append(f"table_role_hint_{normalize_token(role_hint)}")
    if fraction_numeric_cells < 0.08 and not header_keywords_hit:
        reasons.append("very_low_numeric_density_without_header_keywords")
    if prose_like_rows >= 3 and row_anchor_count < 3:
        reasons.append("narrative_or_figure_caption_dominated")
    lower_signal = signal_text.lower()
    if any(token in lower_signal for token in PRIMARY_EXCLUDED_PATTERN_TOKENS):
        reasons.append("non_formulation_pattern_surface")
    return reasons


def primary_structural_guardrail_reasons(reasons: list[str]) -> list[str]:
    return [reason for reason in reasons if reason not in PRIMARY_LABEL_ONLY_REASONS]


def table_hard_drop_reasons(item: dict[str, Any], *, breakdown: dict[str, float]) -> list[str]:
    rows = item.get("rows") or []
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    signal_text = table_authority_signal_text(item)
    fraction_numeric_cells = table_fraction_numeric_cells(item)
    first_labels = [normalize_text(label) for label in table_row_label_preview(item, limit=12) if normalize_text(label)]
    prose_label_count = sum(1 for label in first_labels if len(re.findall(r"[A-Za-z]+", label)) >= 8)
    header_keywords_hit = [
        normalize_text(value)
        for value in ensure_list(meta.get("header_keywords_hit"))
        if normalize_text(value)
    ]
    prose_like_rows = table_prose_like_row_count(rows)
    row_anchor_count = table_authority_row_anchor_count(item)
    representation_status = normalize_text(item.get("representation_status")).lower()
    lower_signal = signal_text.lower()
    reasons: list[str] = []
    if representation_status == "unrepaired_corrupted":
        reasons.append("unrepaired_corrupted_representation")
    if prose_like_rows >= 3 and row_anchor_count < 3:
        reasons.append("narrative_or_figure_caption_dominated")
    if prose_label_count >= 3 and fraction_numeric_cells < 0.2:
        reasons.append("prose_row_label_surface")
    if fraction_numeric_cells < 0.08 and not header_keywords_hit and row_anchor_count < 3:
        reasons.append("very_low_numeric_density_without_header_keywords")
    if count_cue_hits(lower_signal, TABLE_AUTHORITY_FRONTMATTER_DEMOTION_TOKENS) >= 2:
        reasons.append("front_matter_or_bibliographic_surface")
    if re.search(r"\b\d+\s+of\s+\d+\b", lower_signal):
        reasons.append("journal_page_or_review_surface")
    if any(token in lower_signal for token in HARD_DROP_SIGNAL_TOKENS):
        reasons.append("high_confidence_non_content_fragment")
    if (
        float(breakdown.get("formulation_density", 0.0) or 0.0) < 2.0
        and float(breakdown.get("design_density", 0.0) or 0.0) < 2.0
        and prose_like_rows >= 4
        and row_anchor_count < 2
    ):
        reasons.append("no_structured_formulation_surface")
    return list(dict.fromkeys(reasons))


def table_primary_eligibility_signals(item: dict[str, Any], *, breakdown: dict[str, float]) -> list[str]:
    rows = item.get("rows") or []
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    signal_text = table_authority_signal_text(item)
    role_hint = infer_table_role_hint(extract_informative_header_parts(rows), {**meta, "_signal_text": signal_text})
    representation_status = normalize_text(item.get("representation_status")).lower()
    quality_flags = {normalize_text(flag).lower() for flag in ensure_list(item.get("quality_flags"))}
    row_pattern = infer_row_pattern(table_row_label_preview(item, limit=20))
    row_anchor_count = table_authority_row_anchor_count(item)
    data_row_count = max(0, len(rows) - 1)
    header_signal_hits = sum(1 for token in PRIMARY_ELIGIBLE_HEADER_TOKENS if token in signal_text.lower())
    signals: list[str] = []
    if representation_status not in {"repair_insufficient", "unrepaired_corrupted"} and "weak_legibility" not in quality_flags:
        signals.append("representation_not_repair_insufficient")
    if row_pattern in {"numeric runs", "F-numbered rows"} or row_anchor_count >= 4:
        signals.append("stable_row_anchors")
    if data_row_count >= 3 and (float(breakdown.get("formulation_density", 0.0) or 0.0) >= 6.0 or float(breakdown.get("design_density", 0.0) or 0.0) >= 6.0):
        signals.append("multiple_condition_rows")
    if header_signal_hits >= 2 and data_row_count >= 3:
        signals.append("formulation_numeric_header_surface")
    if role_hint == "design matrix":
        signals.append("design_matrix_surface")
    return signals


def table_inclusion_class(item: dict[str, Any], *, breakdown: dict[str, float]) -> str:
    if table_hard_drop_reasons(item, breakdown=breakdown):
        return TABLE_INCLUSION_HARD_DROP
    signals = set(table_primary_eligibility_signals(item, breakdown=breakdown))
    if "formulation_numeric_header_surface" in signals:
        return TABLE_INCLUSION_MUST_INCLUDE
    if {"stable_row_anchors", "multiple_condition_rows"}.issubset(signals):
        return TABLE_INCLUSION_MUST_INCLUDE
    if {"stable_row_anchors", "design_matrix_surface"}.issubset(signals):
        return TABLE_INCLUSION_MUST_INCLUDE
    return TABLE_INCLUSION_OPTIONAL_CONTEXT


def table_authority_duplicate_like(item: dict[str, Any], prior_items: list[dict[str, Any]]) -> bool:
    current_signature = table_duplicate_signature(item)
    current_text = table_authority_signal_text(item)
    for prior in prior_items:
        if current_signature == table_duplicate_signature(prior):
            return True
        if is_semantic_near_duplicate(current_text, table_authority_signal_text(prior), threshold=0.68):
            return True
    return False


def rank_table_authority_payloads(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in entries:
        enriched = dict(item)
        breakdown = build_table_authority_score_breakdown(enriched)
        authority_score = round(sum(breakdown.values()), 2)
        enriched["authority_score_breakdown"] = breakdown
        enriched["authority_score"] = authority_score
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (
            -float(item.get("authority_score") or 0.0),
            -int(item.get("score") or 0),
            str(item.get("path", "")).lower(),
        )
    )
    preferred_items: list[dict[str, Any]] = []
    top_score = float(ranked[0].get("authority_score") or 0.0) if ranked else 0.0
    primary_assigned = False
    for index, item in enumerate(ranked, start=1):
        authority_score = float(item.get("authority_score") or 0.0)
        readiness = normalize_text(item.get("selector_readiness_label")).lower()
        representation_status = normalize_text(item.get("representation_status")).lower()
        signal_text = table_authority_signal_text(item)
        breakdown = item.get("authority_score_breakdown") or {}
        strong_structural_table = float(breakdown.get("design_density", 0.0) or 0.0) >= 6.0 or float(breakdown.get("formulation_density", 0.0) or 0.0) >= 10.0
        primary_guardrail_reason = table_primary_guardrail_reasons(item, breakdown=breakdown)
        structural_guardrail_reason = primary_structural_guardrail_reasons(primary_guardrail_reason)
        primary_eligibility_signals = table_primary_eligibility_signals(item, breakdown=breakdown)
        inclusion_class = table_inclusion_class(item, breakdown=breakdown)
        hard_drop_reasons = table_hard_drop_reasons(item, breakdown=breakdown)
        weak_table = (
            authority_score < 18.0
            or readiness == "unresolved"
            or representation_status == "unrepaired_corrupted"
            or count_cue_hits(signal_text, TABLE_AUTHORITY_DOWNSTREAM_DEMOTION_TOKENS) >= 2
            or count_cue_hits(signal_text, TABLE_AUTHORITY_STABILITY_DEMOTION_TOKENS) >= 2
            or count_cue_hits(signal_text, TABLE_AUTHORITY_FRONTMATTER_DEMOTION_TOKENS) >= 2
        )
        primary_guardrail_applied = False
        primary_eligible = (
            inclusion_class == TABLE_INCLUSION_MUST_INCLUDE
            and not weak_table
            and not structural_guardrail_reason
            and bool(primary_eligibility_signals)
        )
        if not primary_assigned and primary_eligible:
            authority_tier = TABLE_AUTHORITY_TIER_PRIMARY
            primary_assigned = True
            preferred_items.append(item)
        elif inclusion_class == TABLE_INCLUSION_HARD_DROP:
            primary_guardrail_applied = True
            authority_tier = TABLE_AUTHORITY_TIER_WEAK_SECONDARY
        elif not primary_assigned and (structural_guardrail_reason or not primary_eligibility_signals):
            primary_guardrail_applied = bool(structural_guardrail_reason or not primary_eligibility_signals)
            if weak_table or table_authority_duplicate_like(item, preferred_items):
                authority_tier = TABLE_AUTHORITY_TIER_WEAK_SECONDARY
            elif strong_structural_table and authority_score >= 35.0:
                authority_tier = TABLE_AUTHORITY_TIER_SECONDARY
                preferred_items.append(item)
            elif authority_score >= max(22.0, top_score - 18.0):
                authority_tier = TABLE_AUTHORITY_TIER_SECONDARY
                preferred_items.append(item)
            else:
                authority_tier = TABLE_AUTHORITY_TIER_WEAK_SECONDARY
        elif weak_table or table_authority_duplicate_like(item, preferred_items):
            authority_tier = TABLE_AUTHORITY_TIER_WEAK_SECONDARY
        elif strong_structural_table and authority_score >= 35.0:
            authority_tier = TABLE_AUTHORITY_TIER_SECONDARY
            preferred_items.append(item)
        elif authority_score >= max(22.0, top_score - 18.0):
            authority_tier = TABLE_AUTHORITY_TIER_SECONDARY
            preferred_items.append(item)
        else:
            authority_tier = TABLE_AUTHORITY_TIER_WEAK_SECONDARY
        item["authority_rank"] = index
        item["authority_tier"] = authority_tier
        item["table_inclusion_class"] = inclusion_class
        item["hard_drop_reason"] = list(hard_drop_reasons)
        item["preserved_by_authority_ranking"] = inclusion_class != TABLE_INCLUSION_HARD_DROP
        item["primary_guardrail_applied"] = "yes" if primary_guardrail_applied else "no"
        item["primary_guardrail_reason"] = list(primary_guardrail_reason)
        item["primary_eligibility_signals"] = list(primary_eligibility_signals)
    return ranked


def select_sample_rows(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []
    scored_picks: list[tuple[int, int]] = []
    for idx, row in enumerate(rows):
        row_text = " ".join(normalize_text(cell) for cell in row if normalize_text(cell)).lower()
        score = 0
        if "note:" in row_text:
            score += 60
        if "selected" in row_text:
            score += 50
        if "optimal" in row_text:
            score += 40
        if "chosen" in row_text:
            score += 30
        if "remaining studies" in row_text or "whole study" in row_text:
            score += 30
        if "mg/ml" in row_text:
            score += 25
        if "ratio" in row_text:
            score += 20
        if "concentration" in row_text:
            score += 15
        if "poloxamer 188" in row_text:
            score += 20
        if score > 0:
            scored_picks.append((score, idx))
    scored_picks.sort(key=lambda item: (-item[0], item[1]))
    picks = [idx for _, idx in scored_picks[:2]]
    if 0 not in picks:
        picks.append(0)
    if len(rows) >= 3:
        picks.append(len(rows) // 2)
    if len(rows) >= 2:
        picks.append(len(rows) - 1)
    selected: list[list[str]] = []
    seen: set[int] = set()
    for idx in picks:
        if idx in seen or idx < 0 or idx >= len(rows):
            continue
        seen.add(idx)
        selected.append(rows[idx])
        if len(selected) >= 3:
            break
    return selected


def render_table_summary_text(table_dir: Path | None, max_tables: int = 4) -> str:
    if table_dir is None or not table_dir.exists():
        return ""
    enhancement_enabled = summary_first_column_enhancement_enabled()
    selected_tables = select_summary_tables(
        table_dir,
        max_tables=max_tables,
    )
    blocks: list[str] = []
    for item in selected_tables:
        blocks.append(render_summary_table_block(item, enhancement_enabled=enhancement_enabled))
    return "\n\n".join(blocks)


def truncate_summary_cell(text: str, *, max_chars: int = 72) -> str:
    compact = normalize_text(text)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def flatten_table_headers_for_summary(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    width = max((len(row) for row in rows if isinstance(row, list)), default=0)
    return flatten_header_rows([rows[0]], width)


def summary_column_score(header: str, values: list[str], *, index: int) -> float:
    compact_header = normalize_text(header)
    lower_header = compact_header.lower()
    compact_values = [normalize_text(value) for value in values if normalize_text(value)]
    lower_values = " ".join(value.lower() for value in compact_values[:8])
    score = 0.0
    if index == 0:
        score += 100.0
    if not compact_header or lower_header.startswith("unnamed"):
        score -= 40.0
    if any(
        token in lower_header
        for token in [
            "formulation",
            "sample",
            "batch",
            "run",
            "drug",
            "polymer",
            "surfactant",
            "concentration",
            "ratio",
            "size",
            "pdi",
            "zeta",
            "encapsulation",
            "loading",
            "yield",
            "efficiency",
            "desirability",
        ]
    ):
        score += 15.0
    if any(token in lower_values for token in ["mg/ml", "%", "ratio", "particle size", "ee", "dl"]):
        score += 6.0
    unique_values = {value.lower() for value in compact_values}
    if compact_values and len(unique_values) <= 1:
        score -= 12.0
    if compact_values and sum(len(value) for value in compact_values[:4]) > 320:
        score -= 10.0
    return score


def select_summary_column_indices(rows: list[list[str]], *, max_columns: int = 6) -> list[int]:
    if not rows:
        return []
    width = max((len(row) for row in rows if isinstance(row, list)), default=0)
    headers = flatten_table_headers_for_summary(rows)
    data_rows = rows[1:] if len(rows) > 1 else []
    ranked: list[tuple[float, int]] = []
    for index in range(width):
        header = headers[index] if index < len(headers) else ""
        values = [row[index] for row in data_rows[:8] if index < len(row)]
        score = summary_column_score(header, values, index=index)
        if score > 0:
            ranked.append((-score, index))
    ranked.sort()
    chosen = [index for _, index in ranked[:max_columns]]
    if 0 not in chosen and width > 0:
        chosen.append(0)
    return sorted(dict.fromkeys(chosen))


def build_summary_sample_lines(rows: list[list[str]], column_indices: list[int], *, max_rows: int = 3) -> list[str]:
    if not rows:
        return []
    data_rows = rows[1:] if len(rows) > 1 else rows
    samples = select_sample_rows(data_rows)[:max_rows]
    sample_lines: list[str] = []
    for idx, row in enumerate(samples, start=1):
        selected_cells = [
            truncate_summary_cell(row[col_index], max_chars=72)
            for col_index in column_indices
            if col_index < len(row) and normalize_text(row[col_index])
        ]
        if selected_cells:
            sample_lines.append(f"- sample_row_{idx}: " + " | ".join(selected_cells))
    return sample_lines


def build_table_summary_lines(item: dict[str, Any], *, enhancement_enabled: bool) -> list[str]:
    path = item["path"]
    rows = item["rows"]
    meta = item["meta"]
    header_parts = [part for part in flatten_table_headers_for_summary(rows) if normalize_text(part)]
    data_rows = rows[1:] if len(rows) > 1 else []
    row_ids = [row[0] for row in data_rows if row and normalize_text(row[0])]
    column_indices = select_summary_column_indices(rows)
    column_headers = [
        truncate_summary_cell(header_parts[index] if index < len(header_parts) else f"column_{index + 1}", max_chars=52)
        for index in column_indices
    ]
    units = infer_units_from_headers(column_headers)
    sample_lines = build_summary_sample_lines(rows, column_indices)
    table_match = re.search(r"__table_(\d+)__", path.name)
    table_id = f"Table {int(table_match.group(1))}" if table_match else path.stem
    caption = normalize_text(meta.get("caption_or_title"))
    footnotes = truncate_summary_cell(normalize_text(meta.get("footnotes") or meta.get("notes")), max_chars=220)
    row_anchor_preview = summarize_row_anchor_preview(row_ids) if enhancement_enabled else ""
    block_lines = [
        f"[TABLE_SUMMARY {to_repo_rel(path)}]",
        f"- table_id: {table_id}",
        f"- title_or_caption: {caption or '(not available)'}",
        f"- key_columns: {' || '.join(column_headers) if column_headers else '(not available)'}",
        f"- header_units: {', '.join(units) if units else '(none inferred)'}",
        f"- row_identifier_pattern: {infer_row_pattern(row_ids)}",
        f"- table_role_hint: {infer_table_role_hint(header_parts, meta)}",
        f"- table_shape_hint: rows={meta.get('n_rows', len(rows))}, cols={meta.get('n_cols', max((len(r) for r in rows), default=0))}",
    ]
    if enhancement_enabled and row_anchor_preview:
        block_lines.append(f"- first_column_row_labels_preview: {row_anchor_preview}")
    representation_status = normalize_text(item.get("representation_status"))
    if representation_status:
        block_lines.append(f"- representation_status: {representation_status}")
    repair_primary_source = normalize_text(item.get("repair_primary_source"))
    if repair_primary_source:
        block_lines.append(f"- repair_primary_source: {repair_primary_source}")
    repair_actions = [normalize_text(value) for value in ensure_list(item.get("repair_actions")) if normalize_text(value)]
    if repair_actions:
        block_lines.append(f"- repair_actions: {', '.join(repair_actions)}")
    selector_readiness = normalize_text(item.get("selector_readiness_label"))
    if selector_readiness:
        block_lines.append(f"- selector_readiness: {selector_readiness}")
    if sample_lines:
        block_lines.append("- sample_rows:")
        block_lines.extend(sample_lines)
    block_lines.append(f"- footnotes_or_notes: {footnotes or '(not available)'}")
    return block_lines


def render_summary_table_block(item: dict[str, Any], *, enhancement_enabled: bool) -> str:
    return "\n".join(build_table_summary_lines(item, enhancement_enabled=enhancement_enabled))


@lru_cache(maxsize=256)
def cached_summary_table_candidates(table_dir_str: str) -> tuple[dict[str, Any], ...]:
    table_dir = Path(table_dir_str)
    if not table_dir.exists():
        return tuple()
    return tuple(collect_summary_table_candidates(table_dir))


def resolve_prompt_summary_table_item(origin_locator: str) -> dict[str, Any] | None:
    locator = normalize_text(origin_locator)
    if not locator:
        return None
    table_path = Path(locator)
    if not table_path.is_absolute():
        table_path = (PROJECT_ROOT / table_path).resolve()
    if not table_path.exists() or table_path.suffix.lower() != ".csv":
        return None
    for item in cached_summary_table_candidates(str(table_path.parent.resolve())):
        candidate_path = item.get("path")
        if isinstance(candidate_path, Path) and candidate_path.resolve() == table_path.resolve():
            return dict(item)
    return None


def summary_preserves_variable_structure(item: dict[str, Any]) -> bool:
    rows = item.get("rows") or []
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    header_parts = extract_informative_header_parts(rows)
    signal_text = build_summary_table_signal_text(rows[:8], meta, header_parts)
    role_hint = infer_table_role_hint(header_parts, {**meta, "_signal_text": signal_text})
    row_labels = table_row_label_preview(item, limit=12)
    if role_hint == "design matrix":
        return True
    if has_variable_sweep_structure(signal_text, row_labels=row_labels):
        return True
    if any(
        token in signal_text
        for token in [
            "independent variables",
            "dependent variables",
            "factors",
            "levels",
            "concentration",
            "ratio",
            "run order",
            "experimental design",
            "factorial design",
            "box-behnken",
        ]
    ):
        return True
    if infer_row_pattern(row_labels) in {"numeric runs", "F-numbered rows"} and len(row_labels) >= 4:
        return True
    return False


def should_force_full_table_prompt(
    block: dict[str, Any],
    *,
    summary_item: dict[str, Any] | None,
) -> tuple[bool, str]:
    if not bool(block.get("requires_variable_structure")):
        return False, ""
    if summary_item is None:
        return True, "summary_table_unavailable"
    if not summary_preserves_variable_structure(summary_item):
        return True, "summary_table_preservation_failed"
    return False, ""


def render_prompt_block(
    block: dict[str, Any],
    *,
    summary_enhanced: bool,
) -> dict[str, Any]:
    text_content = normalize_text(block.get("text_content"))
    block_type = normalize_text(block.get("block_type"))
    evidence_kind = normalize_text(block.get("evidence_kind")).lower()
    label = {
        "method": "[METHOD]\n",
        "materials": "[MATERIALS]\n",
        "supporting": "[SUPPORTING]\n",
    }.get(evidence_kind, "")
    if block_type != "table":
        return {
            "block_id": normalize_text(block.get("block_id")),
            "rendered_text": (label + text_content).strip(),
            "table_mode_used": "",
            "summary_applied": "no",
            "reason_for_full_table": "",
        }
    summary_item = resolve_prompt_summary_table_item(normalize_text(block.get("origin_locator")))
    if summary_item is not None:
        return {
            "block_id": normalize_text(block.get("block_id")),
            "rendered_text": "[TABLE]\n" + render_summary_table_block(summary_item, enhancement_enabled=summary_enhanced),
            "table_mode_used": "summary",
            "summary_applied": "yes",
            "reason_for_full_table": "",
        }
    if bool(block.get("is_table_derived")) and normalize_text(block.get("source_type")) == "inline_table_text":
        # Summary-only S2-4a contract: inline table text is allowed as an
        # intermediate diagnostic / selector surface, but it is never lawful
        # as LLM-facing table evidence unless it has been converted into a
        # governed summary-backed table surface first.
        return {
            "block_id": normalize_text(block.get("block_id")),
            "rendered_text": "",
            "table_mode_used": "",
            "summary_applied": "no",
            "reason_for_full_table": "inline_table_text_blocked_by_summary_only_contract",
        }
    return {
        "block_id": normalize_text(block.get("block_id")),
        "rendered_text": "[TABLE]\n" + text_content,
        "table_mode_used": "summary",
        "summary_applied": "yes",
        "reason_for_full_table": "",
    }


def build_prompt_render_bundle(evidence_artifact: dict[str, Any]) -> dict[str, Any]:
    input_contract = evidence_artifact.get("input_contract", {})
    summary_enhanced = bool(input_contract.get("summary_first_column_enhancement"))
    rendered_blocks: list[dict[str, Any]] = []
    rendered_payloads: list[str] = []
    normalized_rendered_payloads: list[str] = []
    for block in ensure_list(evidence_artifact.get("evidence_blocks")):
        if not isinstance(block, dict):
            continue
        rendered = render_prompt_block(block, summary_enhanced=summary_enhanced)
        rendered_text = str(rendered.get("rendered_text") or "")
        normalized_rendered_text = normalize_text(rendered_text)
        if not normalized_rendered_text:
            continue
        rendered["rendered_text"] = rendered_text
        if not rendered_text:
            continue
        rendered_blocks.append(rendered)
        rendered_payloads.append(rendered_text)
        normalized_rendered_payloads.append(normalized_rendered_text)
    table_modes = [rendered.get("table_mode_used") for rendered in rendered_blocks if normalize_text(rendered.get("table_mode_used"))]
    overall_table_mode_used = "summary" if "summary" in table_modes else ""
    return {
        "rendered_blocks": rendered_blocks,
        "rendered_payloads": rendered_payloads,
        "normalized_rendered_payloads": normalized_rendered_payloads,
        "evidence_text": "\n\n".join(rendered_payloads).strip(),
        "table_mode_used": overall_table_mode_used,
        "summary_applied": "yes" if "summary" in table_modes else "no",
        "reason_for_full_table": "",
    }


def resolved_selector_strategy(current_table_mode: str) -> str:
    del current_table_mode
    return "summary_only_coverage_first_v1"


def selector_strategy_uses_evidence_pack_layout(selector_strategy: str) -> bool:
    strategy = normalize_text(selector_strategy).lower()
    return bool(strategy)


def build_prompt_runtime_metadata(evidence_artifact: dict[str, Any]) -> dict[str, Any]:
    input_contract = evidence_artifact.get("input_contract", {})
    current_table_mode = normalize_text(input_contract.get("table_mode")).lower() or table_mode()
    summary_enhanced = bool(input_contract.get("summary_first_column_enhancement"))
    current_input_packing_mode = normalize_text(input_contract.get("input_packing_mode")).lower() or input_packing_mode()
    ordered_block_order = [str(value) for value in ensure_list(input_contract.get("ordered_block_order")) if str(value).strip()]
    prompt_render_bundle = build_prompt_render_bundle(evidence_artifact)
    runtime_notes: list[str] = [f"Table mode: {current_table_mode}"]
    if prompt_render_bundle["table_mode_used"] == "summary":
        runtime_notes.append("Table evidence is provided in summary-first mode by default.")
        runtime_notes.append("All LLM-facing table evidence remains summary-only under the governed S2-4a contract.")
        if summary_enhanced:
            runtime_notes.append("Summary selection prioritizes DOE-like tables with explicit numbered row anchors.")
            runtime_notes.append("First-column row labels / numbering previews are exposed for explicit row-anchor tables.")
    if current_input_packing_mode == ORDERED_INPUT_PACKING_MODE:
        runtime_notes.append("Controlled evidence packing is enabled.")
        runtime_notes.append(
            "Prompt order prioritizes synthesis/preparation blocks, then materials/procurement blocks, then table evidence, then narrative fallback."
        )
        if ordered_block_order:
            runtime_notes.append(f"Resolved evidence block order: {' > '.join(ordered_block_order)}.")
        runtime_notes.append("Treat the evidence pack as the governed live input ordering for this run.")
    return {
        "table_mode": current_table_mode,
        "summary_enhanced": summary_enhanced,
        "input_packing_mode": current_input_packing_mode,
        "ordered_block_order": ordered_block_order,
        "prompt_render_bundle": prompt_render_bundle,
        "runtime_notes": runtime_notes,
    }


def prompt_size_policy_status(prompt_length: int) -> str:
    return "healthy" if prompt_length <= PROMPT_HEALTHY_CHAR_LIMIT else "oversized"


def semantic_token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9%./:-]{2,}", normalize_text(text).lower())
        if not token.isdigit()
    }


def is_semantic_near_duplicate(text_a: str, text_b: str, *, threshold: float = 0.72) -> bool:
    tokens_a = semantic_token_set(text_a)
    tokens_b = semantic_token_set(text_b)
    if len(tokens_a) < 12 or len(tokens_b) < 12:
        return False
    overlap = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))
    if smaller <= 0:
        return False
    return (overlap / smaller) >= threshold


def evidence_text_has_noise(text: str) -> bool:
    lower = normalize_text(text).lower()
    if not lower:
        return False
    noisy_tokens = [
        "creative commons attribution",
        "submit your manuscript",
        "powered by tcpdf",
        "downloaded from",
        "for personal use only",
        "dovepress",
        "wiley online library",
        "references",
    ]
    return any(token in lower for token in noisy_tokens)


def select_bounded_role_aware_table_candidates(
    selected_candidates: list[dict[str, Any]],
    *,
    max_tables: int = 4,
) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    seen_origins: set[str] = set()
    for candidate in selected_candidates:
        if candidate.get("candidate_kind") != "table":
            continue
        origin_locator = normalize_text(candidate.get("origin_locator"))
        if not origin_locator or origin_locator in seen_origins:
            continue
        candidate_path = (PROJECT_ROOT / origin_locator).resolve()
        if not candidate_path.exists():
            continue
        seen_origins.add(origin_locator)
        bounded.append(candidate)
        if len(bounded) >= max_tables:
            break
    return bounded


def supplement_sequential_table_candidates(
    selected_candidates: list[dict[str, Any]],
    *,
    segmented_candidates: list[dict[str, Any]],
    signals: dict[str, bool],
    max_tables: int = 4,
    min_distinct_tables: int = 2,
) -> list[dict[str, Any]]:
    bounded = select_bounded_role_aware_table_candidates(
        selected_candidates,
        max_tables=max_tables,
    )
    if not signals.get("has_sequential_signal"):
        return bounded

    seen_origins = {
        normalize_text(candidate.get("origin_locator"))
        for candidate in bounded
        if normalize_text(candidate.get("origin_locator"))
    }
    if len(seen_origins) >= min_distinct_tables:
        return bounded

    ranked_candidates: list[tuple[float, float, str, dict[str, Any]]] = []
    sequential_support_tokens = [
        "different concentrations",
        "different ratios",
        "different ratio",
        "initial ratios",
        "remaining studies",
        "remaining experiments",
        "whole study",
        "chosen as optimal",
        "selected as optimal",
        "optimal surfactant concentration",
        "optimal formulation",
    ]
    for candidate in segmented_candidates:
        if candidate.get("candidate_kind") != "table":
            continue
        origin_locator = normalize_text(candidate.get("origin_locator"))
        if not origin_locator or origin_locator in seen_origins:
            continue
        candidate_path = (PROJECT_ROOT / origin_locator).resolve()
        if not candidate_path.exists():
            continue
        candidate_text = normalize_text(candidate.get("text_content")).lower()
        section_kind = normalize_text(candidate.get("section_kind")).lower()
        support_score = 0.0
        if section_kind == "optimization":
            support_score += 3.0
        elif section_kind == "experimental_design":
            support_score += 2.0
        support_score += float(
            sum(1 for token in sequential_support_tokens if token in candidate_text)
        )
        if support_score <= 0.0:
            continue
        ranked_candidates.append(
            (
                -support_score,
                -float(candidate.get("table_score", 0.0) or 0.0),
                origin_locator,
                candidate,
            )
        )

    ranked_candidates.sort()
    for _, _, origin_locator, candidate in ranked_candidates:
        bounded.append(candidate)
        seen_origins.add(origin_locator)
        if len(bounded) >= max_tables or len(seen_origins) >= min_distinct_tables:
            break
    return bounded


def select_scored_full_mode_fallback_tables(
    summary_candidates: list[dict[str, Any]],
    *,
    max_tables: int = 4,
) -> list[dict[str, Any]]:
    fallback: list[dict[str, Any]] = []
    seen_origins: set[str] = set()
    for item in summary_candidates:
        path = item.get("path")
        if not isinstance(path, Path):
            continue
        origin_locator = to_repo_rel(path)
        if not origin_locator or origin_locator in seen_origins:
            continue
        seen_origins.add(origin_locator)
        fallback.append(
            {
                "candidate_kind": "table",
                "candidate_id": normalize_text(item.get("repair_source_candidate_id")) or None,
                "origin_locator": origin_locator,
                "source_type": "table_excerpt",
                "evidence_kind": "table",
                "priority_score": float(item.get("authority_score", item.get("score")) or 0.0),
                "authority_rank": item.get("authority_rank"),
                "authority_score": item.get("authority_score"),
                "authority_tier": normalize_text(item.get("authority_tier")),
                "item": item,
            }
        )
        if len(fallback) >= max_tables:
            break
    return fallback


def detect_pre_llm_signals(raw_text: str, table_candidates: list[dict[str, Any]]) -> dict[str, bool]:
    combined = normalize_text(raw_text).lower()
    table_row_labels: list[str] = []
    if table_candidates:
        table_blob_parts: list[str] = []
        for item in table_candidates:
            path = item["path"]
            meta = item["meta"]
            table_blob_parts.append(path.name.lower())
            table_blob_parts.append(normalize_text(meta.get("caption_or_title")).lower())
            table_blob_parts.extend(normalize_text(value).lower() for value in (meta.get("header_keywords_hit") or []))
            table_blob_parts.extend(
                normalize_text(cell).lower()
                for row in item["rows"][:6]
                for cell in row
                if normalize_text(cell)
            )
            if item.get("rows"):
                table_row_labels.extend(
                    normalize_text(row[0])
                    for row in item["rows"][1:]
                    if isinstance(row, list) and row and normalize_text(row[0])
                )
        combined = f"{combined} {' '.join(part for part in table_blob_parts if part)}"
    has_doe_signal = any(
        token in combined
        for token in [
            "doe",
            "design matrix",
            "response surface",
            "box-behnken",
            "factorial",
            "experimental runs",
            "run order",
        ]
    )
    has_sequential_signal = any(
        token in combined
        for token in [
            "selected as optimal",
            "chosen as optimal",
            "remaining studies",
            "remaining experiments",
            "used that condition",
            "used for the remaining studies",
            "carried forward",
            "subsequent experiments",
            "remaining optimization",
        ]
    )
    has_optimization_signal = has_sequential_signal or any(
        token in combined
        for token in [
            "optimal",
            "optimized",
            "optimization",
            "selected condition",
            "chosen condition",
        ]
    )
    has_variable_sweep_signal = has_variable_sweep_structure(
        combined,
        row_labels=table_row_labels,
    )
    return {
        "has_doe_signal": has_doe_signal,
        "has_sequential_signal": has_sequential_signal,
        "has_optimization_signal": has_optimization_signal,
        "has_variable_sweep_signal": has_variable_sweep_signal,
    }


def count_cue_hits(text: str, cues: list[str]) -> int:
    lower = normalize_text(text).lower()
    return sum(1 for cue in cues if cue in lower)


def count_any_cues(text: str, cues: list[str]) -> int:
    lower = normalize_text(text).lower()
    return sum(1 for cue in cues if cue in lower)


def is_materials_inventory_candidate(text: str) -> bool:
    lower = normalize_text(text).lower()
    if not lower:
        return False
    procurement_hits = count_any_cues(lower, PROCUREMENT_CUES)
    chemical_hits = count_any_cues(
        lower,
        ["plga", "poloxamer", "itz", "acetone", "acetonitrile", "sucrose", "mannitol", "hp-", "phosphate buffered saline"],
    )
    return ("materials" in lower and procurement_hits >= 1) or (procurement_hits >= 1 and chemical_hits >= 2)


def procedure_signal_count(text: str) -> int:
    return count_any_cues(text, PREPARATION_PROCEDURE_CUES)


def assay_comparator_signal_count(text: str) -> int:
    return count_any_cues(text, ASSAY_COMPARATOR_CUES)


def is_assay_comparator_dominated(text: str) -> bool:
    lower = normalize_text(text).lower()
    return assay_comparator_signal_count(lower) >= 2 and procedure_signal_count(lower) <= 2


def has_preparation_procedure_signal(text: str) -> bool:
    lower = normalize_text(text).lower()
    return "nanoprecipitation" in lower or procedure_signal_count(lower) >= 4


def optimization_decision_signature(text: str) -> str:
    lower = normalize_text(text).lower()
    if "during the whole study" in lower or "chosen for the preparation" in lower:
        return "selected_condition"
    if "chosen as the optimal formulation" in lower or "utilized for the formulation of all the following studies" in lower:
        return "optimal_formulation"
    if "selected as optimal" in lower or "optimal lyoprotectant" in lower:
        return "selected_auxiliary_condition"
    return ""


def table_blob(item: dict[str, Any]) -> str:
    meta = item.get("meta", {})
    rows = item.get("rows", [])
    row_blob = " ".join(
        normalize_text(cell)
        for row in rows[:8]
        for cell in row[:8]
        if normalize_text(cell)
    )
    return " ".join(
        part
        for part in [
            normalize_text(meta.get("caption_or_title")),
            " ".join(normalize_text(value) for value in (meta.get("header_keywords_hit") or [])),
            row_blob,
        ]
        if part
    ).lower()


def table_row_label_preview(item: dict[str, Any], *, limit: int = 6) -> list[str]:
    rows = item.get("rows", [])
    labels: list[str] = []
    for row in rows[1:]:
        if not row:
            continue
        label = normalize_text(row[0])
        if label:
            labels.append(label.lower())
        if len(labels) >= limit:
            break
    return labels


def table_duplicate_signature(item: dict[str, Any]) -> str:
    rows = item.get("rows", [])
    shape = f"{len(rows)}x{max((len(row) for row in rows), default=0)}"
    labels = "|".join(table_row_label_preview(item))
    sampled = "|".join(
        normalize_text(cell).lower()
        for row in select_sample_rows(rows[1:] if len(rows) > 1 else rows)
        for cell in row[:4]
        if normalize_text(cell)
    )
    return f"{shape}::{labels}::{sampled}"


def table_preview_labels_from_text(text: str, *, limit: int = 12) -> list[str]:
    match = re.search(r"first_column_row_labels_preview:\s*(.+?)(?:\s+- sample_rows:|\s+- footnotes_or_notes:|$)", text, flags=re.I)
    if not match:
        return []
    values = [normalize_text(part) for part in re.split(r"\s*,\s*", match.group(1)) if normalize_text(part)]
    return values[:limit]


def selector_table_context(candidate_or_item: dict[str, Any]) -> tuple[dict[str, Any], list[list[str]], str, set[str], list[str], str, float]:
    item = candidate_or_item.get("item") if isinstance(candidate_or_item.get("item"), dict) else candidate_or_item
    meta = item.get("meta", {}) if isinstance(item, dict) else {}
    rows = item.get("rows", []) if isinstance(item, dict) else []
    blob = table_blob(item) if isinstance(item, dict) and item.get("rows") is not None else normalize_text(candidate_or_item.get("text_content"))
    quality_flags = {
        normalize_text(flag).lower()
        for flag in ensure_list(candidate_or_item.get("quality_flags") or item.get("quality_flags"))
    }
    row_labels = table_row_label_preview(item, limit=12) if rows else table_preview_labels_from_text(blob, limit=12)
    section_kind = normalize_text(candidate_or_item.get("section_kind"))
    raw_score = float(candidate_or_item.get("table_score", item.get("score", 0)) or 0.0)
    return meta, rows, blob, quality_flags, row_labels, section_kind, raw_score


def selector_candidates_from_candidate_artifact(candidate_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    selector_candidates: list[dict[str, Any]] = []
    for candidate in ensure_list(candidate_artifact.get("candidate_blocks")):
        candidate_type = normalize_text(candidate.get("candidate_type")).lower()
        candidate_kind = "table" if candidate_type == "table" else "paragraph"
        origin_locator = normalize_text(candidate.get("origin_locator"))
        selector_candidate = {
            "candidate_id": normalize_text(candidate.get("candidate_id")),
            "candidate_kind": candidate_kind,
            "source_type": normalize_text(candidate.get("source_type")),
            "origin_key": origin_locator or normalize_text(candidate.get("candidate_id")),
            "origin_locator": origin_locator,
            "text_content": normalize_text(candidate.get("text_content")),
            "paragraph_index": int(candidate.get("paragraph_index", 0) or 0),
            "segment_index": int(candidate.get("segment_index", 0) or 0),
            "section_label": normalize_text(candidate.get("section_label")),
            "section_kind": normalize_text(candidate.get("section_kind")),
            "split_trigger": "artifact_replay",
            "noise_flags": list(candidate.get("noise_flags") or []),
            "quality_flags": list(candidate.get("quality_flags") or []),
        }
        if candidate_kind == "table":
            selector_candidate["table_role_hint"] = normalize_text(candidate.get("table_role_hint"))
            selector_candidate["table_row_pattern"] = normalize_text(candidate.get("table_row_pattern"))
            selector_candidate["table_score"] = float(candidate.get("table_score", 0) or 0.0)
        selector_candidates.append(selector_candidate)
    return selector_candidates


def build_selector_surface_text(
    text_content: str,
    *,
    section_kind: str,
    split_trigger: str,
    table_role_hint: str | None = None,
) -> str:
    del section_kind, split_trigger, table_role_hint
    return normalize_text(text_content)


def build_candidate_segmentation_artifact(
    *,
    record: dict[str, str],
    manifest_path: Path,
    text_path: Path,
    table_dir: Path | None,
    producer_script: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_text = text_path.read_text(encoding="utf-8", errors="replace")
    paragraph_entries = build_segmented_paragraph_entries(raw_text)
    if not paragraph_entries:
        paragraph_entries = build_sparse_sentence_window_entries(raw_text)
    summary_candidates = collect_summary_table_candidates(table_dir) if table_dir is not None and table_dir.exists() else []
    signals = detect_pre_llm_signals(raw_text, summary_candidates)
    candidates: list[dict[str, Any]] = []
    selector_candidates: list[dict[str, Any]] = []

    for idx, entry in enumerate(paragraph_entries, start=1):
        text_content = normalize_text(entry.get("text"))
        if not text_content:
            continue
        section_label = extract_section_label(text_content)
        section_kind = infer_section_kind(text_content, section_label)
        split_trigger = str(entry.get("split_trigger") or "paragraph_boundary")
        quality_flags = build_candidate_quality_flags(text_content, section_kind, split_trigger)
        noise_flags = list(entry.get("noise_flags") or [])
        inline_table_item = build_inline_formulation_table_item(
            text_content,
            text_path=text_path,
            paragraph_index=int(entry.get("paragraph_index", 0)),
            segment_index=int(entry.get("segment_index", 0)),
        )
        is_inline_table_candidate = inline_table_item is not None and split_trigger == "table_inline_split"
        table_role_hint = "formulation" if is_inline_table_candidate else None
        block_type = "table" if is_inline_table_candidate else "paragraph"
        method_cue_count = count_segmentation_cues(text_content, SEGMENT_METHOD_CUES)
        explicit_preparation_label = "preparation" in normalize_text(section_label).lower()
        classified_block_type, _, _ = classify_ordered_paragraph_block(text_content)
        if is_inline_table_candidate:
            pass
        elif section_kind in {"preparation", "variant_preparation"} and (
            explicit_preparation_label
            or text_content.lower().startswith("empty nanospheres were prepared")
            or text_content.lower().startswith("empty nanocapsules were prepared")
            or (method_cue_count >= 2 and classified_block_type == "synthesis_method")
        ):
            block_type = "synthesis_method"
        elif section_kind == "materials":
            block_type = "materials_procurement"
        candidate_id = f"{record['key']}__candidate_paragraph__{idx:02d}"
        origin_locator = f"{to_repo_rel(text_path)}#paragraph:{entry['paragraph_index']}#segment:{entry.get('segment_index', 0)}"
        selector_text_content = build_selector_surface_text(
            text_content,
            section_kind="table_related" if is_inline_table_candidate else section_kind,
            split_trigger="inline_table_recovery" if is_inline_table_candidate else split_trigger,
            table_role_hint=table_role_hint,
        )
        candidate_payload = {
            "candidate_id": candidate_id,
            "candidate_type": "table" if is_inline_table_candidate else "prose",
            "block_type": block_type,
            "is_table_derived": is_inline_table_candidate,
            "source_type": "inline_table_text" if is_inline_table_candidate else "clean_text_paragraph",
            "origin_locator": origin_locator,
            "paragraph_index": int(entry.get("paragraph_index", 0)),
            "segment_index": int(entry.get("segment_index", 0)),
            "section_label": section_label,
            "section_kind": "table_related" if is_inline_table_candidate else section_kind,
            "segmentation_method": "inline_text_table_recovery" if is_inline_table_candidate else "double_newline_then_section_heading_split",
            "split_trigger": "inline_table_recovery" if is_inline_table_candidate else split_trigger,
            "noise_flags": noise_flags,
            "quality_flags": (
                quality_flags + list(inline_table_item.get("quality_flags") or [])
                if is_inline_table_candidate
                else quality_flags
            ),
            "text_content": text_content,
            "text_preview": normalize_text(text_content[:220]),
        }
        if is_inline_table_candidate:
            candidate_payload["table_role_hint"] = table_role_hint
            candidate_payload["table_row_pattern"] = inline_table_item.get("row_pattern", "")
            candidate_payload["table_score"] = inline_table_item.get("score")
        candidates.append(candidate_payload)
        selector_candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": "table" if is_inline_table_candidate else "paragraph",
                "source_type": "inline_table_text" if is_inline_table_candidate else "clean_text_paragraph",
                "origin_key": f"paragraph:{entry['paragraph_index']}#segment:{entry.get('segment_index', 0)}",
                "origin_locator": origin_locator,
                "text_content": selector_text_content,
                "paragraph_index": int(entry.get("paragraph_index", 0)),
                "segment_index": int(entry.get("segment_index", 0)),
                "section_label": section_label,
                "section_kind": "table_related" if is_inline_table_candidate else section_kind,
                "split_trigger": "inline_table_recovery" if is_inline_table_candidate else split_trigger,
                "noise_flags": noise_flags,
                "quality_flags": (
                    quality_flags + list(inline_table_item.get("quality_flags") or [])
                    if is_inline_table_candidate
                    else quality_flags
                ),
            }
        )
        if is_inline_table_candidate:
            selector_candidates[-1]["item"] = inline_table_item

    for idx, item in enumerate(summary_candidates, start=1):
        text_content = render_summary_table_block(item, enhancement_enabled=summary_first_column_enhancement_enabled())
        if not normalize_text(text_content):
            continue
        meta = item.get("meta", {})
        section_label = normalize_text(meta.get("caption_or_title"))
        section_kind = infer_section_kind(text_content, section_label)
        quality_flags = list(item.get("quality_flags") or [])
        if item.get("filtered_noise_rows"):
            quality_flags.append(f"filtered_noise_rows:{item['filtered_noise_rows']}")
        header_parts = extract_informative_header_parts(item.get("rows") or [])
        table_role_hint = infer_table_role_hint(header_parts, {**meta, "_signal_text": build_summary_table_signal_text((item.get("rows") or [])[:6], meta, header_parts)})
        candidate_id = f"{record['key']}__candidate_table__{idx:02d}"
        candidate_payload = {
            "candidate_id": candidate_id,
            "candidate_type": "table",
            "block_type": "table",
            "is_table_derived": True,
            "source_type": "table_summary",
            "origin_locator": to_repo_rel(item["path"]),
            "section_label": section_label,
            "section_kind": section_kind,
            "segmentation_method": "manifest_aware_table_summary_isolation",
            "split_trigger": "table_isolation",
            "noise_flags": [],
            "quality_flags": quality_flags,
            "table_role_hint": table_role_hint,
            "table_row_pattern": item.get("row_pattern", ""),
            "table_score": item.get("score"),
            "text_content": text_content,
            "text_preview": normalize_text(text_content[:220]),
            "representation_status": normalize_text(item.get("representation_status")) or "raw_summary",
            "repair_actions": item.get("repair_actions") or [],
            "repair_warnings": item.get("repair_warnings") or [],
            "repair_confidence": item.get("repair_confidence"),
            "raw_table_preview": normalize_text(item.get("raw_table_preview")),
            "repaired_table_preview": normalize_text(item.get("repaired_table_preview")),
            "material_difference_from_raw": bool(item.get("material_difference_from_raw")),
            "repair_primary_source": normalize_text(item.get("repair_primary_source")),
            "repair_source_csv_path": normalize_text(item.get("repair_source_csv_path")) or to_repo_rel(item["path"]),
            "repair_source_manifest_path": normalize_text(item.get("repair_source_manifest_path")),
            "repair_source_candidate_id": candidate_id,
            "candidate_variant_role": normalize_text(item.get("candidate_variant_role")) or "raw",
            "same_source_table_asset": bool(item.get("same_source_table_asset", True)),
            "derived_from_candidate_id": normalize_text(item.get("derived_from_candidate_id")),
            "selector_readiness_label": normalize_text(item.get("selector_readiness_label")),
            "unresolved_reason": normalize_text(item.get("unresolved_reason")),
            "restoration_profile": normalize_text(item.get("restoration_profile")),
            "full_prompt_table_authority": bool(item.get("full_prompt_table_authority")),
            "authority_rank": item.get("authority_rank"),
            "authority_score": item.get("authority_score"),
            "authority_tier": normalize_text(item.get("authority_tier")),
            "table_inclusion_class": normalize_text(item.get("table_inclusion_class")),
            "hard_drop_reason": item.get("hard_drop_reason") or [],
            "authority_score_breakdown": item.get("authority_score_breakdown") or {},
            "preserved_by_authority_ranking": bool(item.get("preserved_by_authority_ranking")),
            "primary_guardrail_applied": normalize_text(item.get("primary_guardrail_applied")),
            "primary_guardrail_reason": item.get("primary_guardrail_reason") or [],
        }
        candidates.append(candidate_payload)
        selector_candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": "table",
                "source_type": "table_summary",
                "origin_key": to_repo_rel(item["path"]),
                "origin_locator": to_repo_rel(item["path"]),
                "text_content": build_selector_surface_text(
                    text_content,
                    section_kind=section_kind,
                    split_trigger="table_isolation",
                    table_role_hint=table_role_hint,
                ),
                "item": item,
                "section_label": section_label,
                "section_kind": section_kind,
                "split_trigger": "table_isolation",
                "noise_flags": [],
                "quality_flags": quality_flags,
                "table_role_hint": table_role_hint,
                "table_row_pattern": item.get("row_pattern", ""),
                "table_score": item.get("score"),
                "representation_status": normalize_text(item.get("representation_status")) or "raw_summary",
                "repair_primary_source": normalize_text(item.get("repair_primary_source")),
                "repair_actions": item.get("repair_actions") or [],
                "selector_readiness_label": normalize_text(item.get("selector_readiness_label")),
                "restoration_profile": normalize_text(item.get("restoration_profile")),
                "full_prompt_table_authority": bool(item.get("full_prompt_table_authority")),
                "authority_rank": item.get("authority_rank"),
                "authority_score": item.get("authority_score"),
                "authority_tier": normalize_text(item.get("authority_tier")),
                "table_inclusion_class": normalize_text(item.get("table_inclusion_class")),
                "hard_drop_reason": item.get("hard_drop_reason") or [],
                "authority_score_breakdown": item.get("authority_score_breakdown") or {},
                "preserved_by_authority_ranking": bool(item.get("preserved_by_authority_ranking")),
                "primary_guardrail_applied": normalize_text(item.get("primary_guardrail_applied")),
                "primary_guardrail_reason": item.get("primary_guardrail_reason") or [],
            }
        )

    artifact = {
        "paper_key": record["key"],
        "source_clean_text_path": to_repo_rel(text_path),
        "source_manifest_path": to_repo_rel(manifest_path),
        "producer_script": producer_script,
        "contract_version": "s2_candidate_blocks_v1",
        "segmentation_profile": SEGMENTATION_PROFILE,
        "selector_boundary": "candidate_segmentation_only",
        "feature_activation_snapshot": {
            "section_aware_split": True,
            "table_isolation": table_dir is not None and table_dir.exists(),
            "table_representation_repair": table_dir is not None and table_dir.exists(),
            "table_authority_ranking": table_dir is not None and table_dir.exists(),
            "noise_filtering": True,
        },
        "coverage_summary": {
            "total_candidates": len(candidates),
            "prose_candidates": sum(1 for item in candidates if item["candidate_type"] == "prose"),
            "table_candidates": sum(1 for item in candidates if item["candidate_type"] == "table"),
            "repaired_table_candidates": sum(1 for item in candidates if normalize_text(item.get("representation_status")) in {"repaired_summary", "repair_insufficient"}),
            "authoritative_stage1_table_repairs": sum(1 for item in candidates if normalize_text(item.get("repair_primary_source")) == "stage1_selected_table_asset"),
            "primary_authority_tables": sum(1 for item in candidates if normalize_text(item.get("authority_tier")) == TABLE_AUTHORITY_TIER_PRIMARY),
            "secondary_authority_tables": sum(1 for item in candidates if normalize_text(item.get("authority_tier")) == TABLE_AUTHORITY_TIER_SECONDARY),
            "candidates_with_noise_flags": sum(1 for item in candidates if item.get("noise_flags")),
            "candidates_with_quality_flags": sum(1 for item in candidates if item.get("quality_flags")),
            "has_doe_signal": signals["has_doe_signal"],
            "has_sequential_signal": signals["has_sequential_signal"],
            "has_optimization_signal": signals["has_optimization_signal"],
        },
        "candidate_blocks": candidates,
    }
    return artifact, {"selector_candidates": selector_candidates, "signals": signals}

def selector_candidate_title(candidate: dict[str, Any]) -> str:
    if candidate.get("candidate_kind") == "table":
        item = candidate.get("item") if isinstance(candidate.get("item"), dict) else candidate
        meta = item.get("meta", {}) if isinstance(item, dict) else {}
        return normalize_text(meta.get("caption_or_title"))
    return normalize_text(candidate.get("section_label"))


def selector_candidate_table_id(candidate: dict[str, Any]) -> str:
    if candidate.get("candidate_kind") != "table":
        return ""
    item = candidate.get("item") if isinstance(candidate.get("item"), dict) else candidate
    meta = item.get("meta", {}) if isinstance(item, dict) else {}
    source_csv_path = normalize_text(item.get("repair_source_csv_path")) or normalize_text(candidate.get("origin_locator"))
    return derive_stable_table_id(source_csv_path, meta)


def selector_candidate_table_inclusion_class(candidate: dict[str, Any]) -> str:
    if normalize_text(candidate.get("candidate_kind")) != "table":
        return ""
    item = candidate.get("item") if isinstance(candidate.get("item"), dict) else candidate
    inclusion_class = normalize_text(item.get("table_inclusion_class") or candidate.get("table_inclusion_class"))
    return inclusion_class or TABLE_INCLUSION_OPTIONAL_CONTEXT


def selector_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": normalize_text(candidate.get("candidate_id")),
        "origin_locator": normalize_text(candidate.get("origin_locator")),
        "source_filename": Path(normalize_text(candidate.get("origin_locator"))).name if normalize_text(candidate.get("origin_locator")) else "",
        "table_id": selector_candidate_table_id(candidate),
        "title_or_caption": selector_candidate_title(candidate),
        "evidence_kind": normalize_text(candidate.get("evidence_kind")),
        "table_inclusion_class": selector_candidate_table_inclusion_class(candidate),
        "authority_rank": candidate.get("authority_rank", ""),
        "authority_score": candidate.get("authority_score", ""),
        "authority_tier": normalize_text(candidate.get("authority_tier")),
    }


def candidate_evidence_kind(candidate: dict[str, Any]) -> str:
    if normalize_text(candidate.get("candidate_kind")) == "table":
        return "table"
    section_kind = normalize_text(candidate.get("section_kind")).lower()
    block_type = normalize_text(candidate.get("block_type")).lower()
    text = normalize_text(candidate.get("text_content"))
    if (
        has_preparation_procedure_signal(text)
        or (
            section_kind in {"preparation", "variant_preparation"}
            and procedure_signal_count(text) >= 3
        )
        or (
            block_type == "synthesis_method"
            and len(text) >= 80
        )
    ):
        return "method"
    if block_type == "materials_procurement" or section_kind == "materials" or is_materials_inventory_candidate(text):
        return "materials"
    return "supporting"


def candidate_signal_strength(candidate: dict[str, Any]) -> float:
    text = normalize_text(candidate.get("text_content"))
    lower = text.lower()
    kind = candidate_evidence_kind(candidate)
    if kind == "table":
        authority_score = float(candidate.get("authority_score", candidate.get("table_score", 0.0)) or 0.0)
        selector_readiness = normalize_text(candidate.get("selector_readiness_label")).lower()
        representation_status = normalize_text(candidate.get("representation_status")).lower()
        inclusion_class = selector_candidate_table_inclusion_class(candidate)
        base = authority_score / 6.0
        if has_variable_sweep_structure(text):
            base += 2.0
        if has_strong_formulation_table_signal(text):
            base += 2.0
        if inclusion_class == TABLE_INCLUSION_MUST_INCLUDE:
            base += 4.0
        elif inclusion_class == TABLE_INCLUSION_HARD_DROP:
            base -= 10.0
        if selector_readiness == "ready":
            base += 1.0
        elif selector_readiness == "weak":
            base -= 1.5
        elif selector_readiness == "unresolved":
            base -= 3.0
        if representation_status in {"repair_insufficient", "unrepaired_corrupted"}:
            base -= 3.0
        return base + 4.0
    if kind == "method":
        return 2.0 + 1.5 * procedure_signal_count(lower) + (2.5 if has_preparation_procedure_signal(lower) else 0.0)
    if kind == "materials":
        return 1.5 + 1.5 * count_any_cues(lower, PROCUREMENT_CUES) + 2.5 * float(is_materials_inventory_candidate(lower))
    score = 0.5 * count_segmentation_cues(lower, SEGMENT_RESULT_CUES + SEGMENT_OPTIMIZATION_CUES)
    if has_experimental_design_text_signal(lower):
        score += 1.5
    if has_optimization_text_signal(lower):
        score += 1.5
    if "table " in lower:
        score += 1.0
    return score


def candidate_noise_penalty(candidate: dict[str, Any]) -> float:
    text = normalize_text(candidate.get("text_content"))
    lower = text.lower()
    quality_flags = {normalize_text(flag).lower() for flag in ensure_list(candidate.get("quality_flags"))}
    penalty = 0.0
    if candidate_evidence_kind(candidate) == "table" and selector_candidate_table_inclusion_class(candidate) == TABLE_INCLUSION_HARD_DROP:
        penalty += 12.0
    if should_drop_segment(text, normalize_text(candidate.get("section_kind")), normalize_text(candidate.get("section_label"))):
        penalty += 8.0
    if "residual_noise" in quality_flags or "reference_like_content" in quality_flags:
        penalty += 4.0
    if is_assay_comparator_dominated(lower):
        penalty += 5.0
    if normalize_text(candidate.get("section_kind")).lower() == "context":
        penalty += 2.0
    return penalty


def candidate_structure_quality(candidate: dict[str, Any]) -> float:
    kind = candidate_evidence_kind(candidate)
    quality_flags = {normalize_text(flag).lower() for flag in ensure_list(candidate.get("quality_flags"))}
    if kind == "table":
        quality = 6.0
        inclusion_class = selector_candidate_table_inclusion_class(candidate)
        if normalize_text(candidate.get("source_type")) == "inline_table_text":
            quality -= 3.0
        if "inline_formulation_table_recovered" in quality_flags:
            quality -= 1.0
        if inclusion_class == TABLE_INCLUSION_MUST_INCLUDE:
            quality += 2.5
        elif inclusion_class == TABLE_INCLUSION_HARD_DROP:
            quality -= 6.0
        return quality
    if kind == "method":
        return 3.5
    if kind == "materials":
        return 3.0
    return 1.0


def candidate_locality_score(candidate: dict[str, Any]) -> float:
    split_trigger = normalize_text(candidate.get("split_trigger")).lower()
    section_kind = normalize_text(candidate.get("section_kind")).lower()
    score = 0.0
    if split_trigger in {"table_isolation", "inline_table_recovery", "table_inline_split", "post_table_split"}:
        score += 1.5
    if section_kind in {"preparation", "variant_preparation", "materials", "optimization"}:
        score += 1.0
    if int(candidate.get("paragraph_index", 0) or 0) <= 25:
        score += 0.5
    return score


def candidate_priority_bonus(candidate: dict[str, Any]) -> float:
    kind = candidate_evidence_kind(candidate)
    if kind == "method":
        return 1.4
    if kind == "materials":
        return 1.1
    if kind == "table":
        return 1.8
    if normalize_text(candidate.get("source_type")) == "inline_table_text":
        return -2.5
    return -0.5


def candidate_priority_score(candidate: dict[str, Any]) -> float:
    return (
        candidate_signal_strength(candidate)
        + candidate_structure_quality(candidate)
        + candidate_locality_score(candidate)
        + candidate_priority_bonus(candidate)
        - candidate_noise_penalty(candidate)
    )


def candidate_table_reference_hints(candidate: dict[str, Any]) -> set[str]:
    hints: set[str] = set()
    table_id = selector_candidate_table_id(candidate)
    if table_id:
        hints.add(table_id.lower())
    for match in re.findall(r"\btable\s+\d+\b", normalize_text(candidate.get("text_content")), flags=re.I):
        hints.add(normalize_text(match).lower())
    origin_locator = normalize_text(candidate.get("origin_locator"))
    if origin_locator:
        hints.add(origin_locator.lower())
    return hints


def candidate_numeric_signature(candidate: dict[str, Any]) -> set[str]:
    return {
        token
        for token in re.findall(r"\b\d+(?:\.\d+)?%?\b", normalize_text(candidate.get("text_content")))
        if token
    }


def candidate_is_proxy_like(candidate: dict[str, Any]) -> bool:
    if candidate_evidence_kind(candidate) == "table":
        return normalize_text(candidate.get("source_type")) == "inline_table_text"
    return normalize_text(candidate.get("split_trigger")).lower() in {"table_inline_split", "post_table_split", "inline_table_recovery"} or "table " in normalize_text(candidate.get("text_content")).lower()


def method_family_signature(candidate: dict[str, Any]) -> str:
    lower = normalize_text(candidate.get("text_content")).lower()
    if "w/o/w" in lower or "emulsion/solvent evaporation" in lower or "double emulsion" in lower:
        return "w_o_w"
    if "nanoprecipitation" in lower or "solvent displacement" in lower:
        return "nanoprecipitation"
    if "emulsion solvent evaporation" in lower:
        return "emulsion_solvent_evaporation"
    return normalize_text(candidate.get("origin_locator")) or normalize_text(candidate.get("candidate_id"))


def is_semantic_duplicate(block_a: dict[str, Any], block_b: dict[str, Any]) -> bool:
    if normalize_text(block_a.get("origin_locator")) and normalize_text(block_a.get("origin_locator")) == normalize_text(block_b.get("origin_locator")):
        return True
    if candidate_evidence_kind(block_a) == "table" and candidate_evidence_kind(block_b) == "table":
        item_a = block_a.get("item") if isinstance(block_a.get("item"), dict) else block_a
        item_b = block_b.get("item") if isinstance(block_b.get("item"), dict) else block_b
        if table_duplicate_signature(item_a) == table_duplicate_signature(item_b):
            return True
    if candidate_table_reference_hints(block_a) & candidate_table_reference_hints(block_b):
        if is_semantic_near_duplicate(normalize_text(block_a.get("text_content")), normalize_text(block_b.get("text_content")), threshold=0.55):
            return True
    shared_numbers = candidate_numeric_signature(block_a) & candidate_numeric_signature(block_b)
    if len(shared_numbers) >= 4 and is_semantic_near_duplicate(normalize_text(block_a.get("text_content")), normalize_text(block_b.get("text_content")), threshold=0.45):
        return True
    return is_semantic_near_duplicate(normalize_text(block_a.get("text_content")), normalize_text(block_b.get("text_content")))


def is_exact_table_duplicate(block_a: dict[str, Any], block_b: dict[str, Any]) -> bool:
    if candidate_evidence_kind(block_a) != "table" or candidate_evidence_kind(block_b) != "table":
        return False
    if normalize_text(block_a.get("origin_locator")) and normalize_text(block_a.get("origin_locator")) == normalize_text(block_b.get("origin_locator")):
        return True
    item_a = block_a.get("item") if isinstance(block_a.get("item"), dict) else block_a
    item_b = block_b.get("item") if isinstance(block_b.get("item"), dict) else block_b
    return table_duplicate_signature(item_a) == table_duplicate_signature(item_b)


def candidate_table_neutral_order_key(candidate: dict[str, Any]) -> tuple[int, str, str]:
    table_id = selector_candidate_table_id(candidate)
    match = re.search(r"\btable\s+(\d+)\b", table_id, flags=re.I)
    if match:
        return (0, f"{int(match.group(1)):04d}", normalize_text(candidate.get("origin_locator")))
    return (1, normalize_text(candidate.get("origin_locator")), normalize_text(candidate.get("candidate_id")))


def selected_candidate_output_order(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """
    IMPORTANT:
    Selector authority ranking is preserved for audit/debug only.
    It MUST NOT influence:
    - table selection
    - table ordering
    - LLM prompt construction

    LLM is the sole authority for semantic table interpretation.
    """
    kind = normalize_text(candidate.get("evidence_kind") or candidate_evidence_kind(candidate))
    if kind == "table":
        inclusion_class = selector_candidate_table_inclusion_class(candidate)
        inclusion_rank = {
            TABLE_INCLUSION_MUST_INCLUDE: 0,
            TABLE_INCLUSION_OPTIONAL_CONTEXT: 1,
            TABLE_INCLUSION_HARD_DROP: 2,
        }.get(inclusion_class, 1)
        return (
            EVIDENCE_KIND_ORDER.get(kind, 9),
            inclusion_rank,
            *candidate_table_neutral_order_key(candidate),
        )
    return (
        EVIDENCE_KIND_ORDER.get(kind, 9),
        normalize_text(candidate.get("origin_locator")),
        normalize_text(candidate.get("candidate_id")),
    )


def supporting_context_is_distinct(candidate: dict[str, Any], selected_tables: list[dict[str, Any]]) -> bool:
    section_kind = normalize_text(candidate.get("section_kind")).lower()
    if section_kind == "context":
        return False
    if candidate_is_proxy_like(candidate) and selected_tables:
        return False
    if section_kind == "experimental_design" and selected_tables:
        candidate_tables = candidate_table_reference_hints(candidate)
        for table_candidate in selected_tables:
            if candidate_tables & candidate_table_reference_hints(table_candidate):
                return False
            if candidate_numeric_signature(candidate) & candidate_numeric_signature(table_candidate):
                return False
    return True


def method_candidate_is_floor_eligible(candidate: dict[str, Any]) -> bool:
    if candidate_evidence_kind(candidate) == "table":
        return False
    text = normalize_text(candidate.get("text_content"))
    if len(text) < 80:
        return False
    if normalize_text(candidate.get("section_kind")).lower() == "context":
        return False
    if should_drop_segment(text, normalize_text(candidate.get("section_kind")), normalize_text(candidate.get("section_label"))):
        return False
    return (
        has_preparation_procedure_signal(text)
        or procedure_signal_count(text) >= 4
        or (
            normalize_text(candidate.get("section_kind")).lower() in {"preparation", "variant_preparation"}
            and procedure_signal_count(text) >= 3
        )
    )


def materials_candidate_is_floor_eligible(candidate: dict[str, Any]) -> bool:
    if candidate_evidence_kind(candidate) == "table":
        return False
    text = normalize_text(candidate.get("text_content"))
    if len(text) < 80:
        return False
    if should_drop_segment(text, normalize_text(candidate.get("section_kind")), normalize_text(candidate.get("section_label"))):
        return False
    lower = text.lower()
    procurement_hits = count_any_cues(lower, PROCUREMENT_CUES)
    chemical_hits = count_any_cues(lower, SEGMENT_MATERIALS_CUES)
    return is_materials_inventory_candidate(text) or (procurement_hits >= 1 and chemical_hits >= 2)


def supporting_candidate_is_floor_eligible(candidate: dict[str, Any], *, selected_tables: list[dict[str, Any]], selected: list[dict[str, Any]]) -> bool:
    if candidate_evidence_kind(candidate) != "supporting":
        return False
    text = normalize_text(candidate.get("text_content"))
    lower = text.lower()
    if len(text) < 80 or len(text) > 700:
        return False
    if normalize_text(candidate.get("section_kind")).lower() == "context":
        return False
    if candidate_is_proxy_like(candidate):
        return False
    if not supporting_context_is_distinct(candidate, selected_tables):
        return False
    if should_drop_segment(text, normalize_text(candidate.get("section_kind")), normalize_text(candidate.get("section_label"))):
        return False
    if any(is_semantic_duplicate(candidate, prior) for prior in selected):
        return False
    if not selected_tables:
        return False
    lexical_support = any(
        token in lower
        for token in [
            "selected",
            "chosen",
            "optimal",
            "subsequent experiments",
            "remaining studies",
            "used for the",
            "used in the following",
            "higher than",
            "lower than",
            "compared with",
            "increased",
            "decreased",
            "plateau",
            "highest",
            "lowest",
        ]
    )
    if not lexical_support:
        return False
    return candidate_locality_score(candidate) >= 1.0


def best_floor_candidate(
    ranked_candidates: list[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    predicate,
    selected_filter=None,
) -> dict[str, Any] | None:
    already_selected = {
        normalize_text(item.get("candidate_id"))
        for item in selected
        if normalize_text(item.get("candidate_id"))
    }
    comparison_selected = selected_filter(selected) if selected_filter is not None else selected
    for candidate in ranked_candidates:
        candidate_id = normalize_text(candidate.get("candidate_id"))
        if candidate_id and candidate_id in already_selected:
            continue
        if not predicate(candidate):
            continue
        if any(is_semantic_duplicate(candidate, prior) for prior in comparison_selected):
            continue
        return candidate
    return None


def apply_minimal_evidence_floor(
    *,
    selected_candidates: list[dict[str, Any]],
    ranked_candidates: list[dict[str, Any]],
    suppression_events: list[dict[str, str]],
) -> dict[str, Any]:
    selected = list(selected_candidates)
    floor_rationale: list[str] = []
    floor_added_method = False
    floor_added_materials = False
    floor_added_supporting = False
    floor_added_formulation_surface = False

    def append_floor_candidate(candidate: dict[str, Any], *, reason: str) -> None:
        selected.append(candidate)
        suppression_events.append(
            {
                "candidate_id": normalize_text(candidate.get("candidate_id")),
                "reason": reason,
            }
        )

    selected_tables = [candidate for candidate in selected if candidate_evidence_kind(candidate) == "table"]
    if not selected_tables:
        best_table = best_floor_candidate(
            ranked_candidates,
            selected=selected,
            predicate=lambda c: candidate_evidence_kind(c) == "table" and selector_candidate_table_inclusion_class(c) != TABLE_INCLUSION_HARD_DROP,
        )
        if best_table is not None:
            append_floor_candidate(best_table, reason="minimal_evidence_floor_added_formulation_surface")
            floor_added_formulation_surface = True
            floor_rationale.append("added_authoritative_formulation_surface")
            selected_tables = [candidate for candidate in selected if candidate_evidence_kind(candidate) == "table"]

    selected_methods = [candidate for candidate in selected if method_candidate_is_floor_eligible(candidate)]
    if not selected_methods:
        best_method = best_floor_candidate(
            ranked_candidates,
            selected=selected,
            predicate=method_candidate_is_floor_eligible,
            selected_filter=lambda items: [item for item in items if candidate_evidence_kind(item) == "method"],
        )
        if best_method is not None:
            append_floor_candidate(best_method, reason="minimal_evidence_floor_added_method")
            floor_added_method = True
            floor_rationale.append("added_single_best_method")

    selected_materials = [candidate for candidate in selected if materials_candidate_is_floor_eligible(candidate)]
    if not selected_materials:
        best_materials = best_floor_candidate(
            ranked_candidates,
            selected=selected,
            predicate=materials_candidate_is_floor_eligible,
            selected_filter=lambda items: [item for item in items if candidate_evidence_kind(item) == "materials"],
        )
        if best_materials is not None:
            append_floor_candidate(best_materials, reason="minimal_evidence_floor_added_materials")
            floor_added_materials = True
            floor_rationale.append("added_single_best_materials")

    selected_supporting = [candidate for candidate in selected if candidate_evidence_kind(candidate) == "supporting"]
    evidence_body_count = sum(1 for candidate in selected if candidate_evidence_kind(candidate) != "metadata")
    support_floor_needed = (
        bool(selected_tables)
        and not selected_supporting
        and evidence_body_count <= 3
        and not (selected_methods and selected_materials)
    )
    if support_floor_needed:
        best_supporting = best_floor_candidate(
            ranked_candidates,
            selected=selected,
            predicate=lambda c: supporting_candidate_is_floor_eligible(c, selected_tables=selected_tables, selected=selected),
            selected_filter=lambda items: items,
        )
        if best_supporting is not None:
            append_floor_candidate(best_supporting, reason="minimal_evidence_floor_added_supporting")
            floor_added_supporting = True
            floor_rationale.append("added_single_distinct_supporting")

    selected.sort(
        key=selected_candidate_output_order
    )
    return {
        "selected_candidates": selected,
        "minimal_evidence_floor_applied": "yes" if any([floor_added_method, floor_added_materials, floor_added_supporting, floor_added_formulation_surface]) else "no",
        "floor_added_method": "yes" if floor_added_method else "no",
        "floor_added_materials": "yes" if floor_added_materials else "no",
        "floor_added_supporting": "yes" if floor_added_supporting else "no",
        "floor_added_formulation_surface": "yes" if floor_added_formulation_surface else "no",
        "floor_rationale": "|".join(floor_rationale),
    }


def build_evidence_priority_selection(
    *,
    segmented_candidates: list[dict[str, Any]],
    signals: dict[str, bool],
) -> dict[str, Any]:
    ranked_candidates: list[dict[str, Any]] = []
    for candidate in segmented_candidates:
        enriched = dict(candidate)
        kind = candidate_evidence_kind(candidate)
        enriched["evidence_kind"] = kind
        enriched["priority_score"] = candidate_priority_score(candidate)
        enriched["signal_strength"] = candidate_signal_strength(candidate)
        enriched["noise_penalty"] = candidate_noise_penalty(candidate)
        enriched["structure_quality"] = candidate_structure_quality(candidate)
        enriched["locality_score"] = candidate_locality_score(candidate)
        enriched["priority_rank"] = EVIDENCE_KIND_ORDER.get(kind, 9)
        ranked_candidates.append(enriched)
    ranked_candidates.sort(
        key=lambda item: (
            item["priority_rank"],
            -float(item["priority_score"]),
            normalize_text(item.get("origin_locator")),
        )
    )

    selected: list[dict[str, Any]] = []
    suppression_events: list[dict[str, str]] = []
    selected_tables: list[dict[str, Any]] = []
    selected_method_families: set[str] = set()
    selected_materials = 0
    selected_supporting = 0

    def add_candidate(candidate: dict[str, Any]) -> None:
        selected.append(candidate)
        if candidate["evidence_kind"] == "table":
            selected_tables.append(candidate)
        elif candidate["evidence_kind"] == "materials":
            nonlocal_selected_materials[0] += 1
        elif candidate["evidence_kind"] == "supporting":
            nonlocal_selected_supporting[0] += 1

    nonlocal_selected_materials = [selected_materials]
    nonlocal_selected_supporting = [selected_supporting]

    for candidate in ranked_candidates:
        if candidate["evidence_kind"] != "table":
            continue
        if selector_candidate_table_inclusion_class(candidate) != TABLE_INCLUSION_MUST_INCLUDE:
            continue
        if any(is_exact_table_duplicate(candidate, prior) for prior in selected_tables):
            suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "semantic_duplicate_must_include_table"})
            continue
        selected.append(candidate)
        selected_tables.append(candidate)

    for candidate in ranked_candidates:
        kind = candidate["evidence_kind"]
        if kind == "table" and selector_candidate_table_inclusion_class(candidate) == TABLE_INCLUSION_HARD_DROP:
            suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "hard_drop_table_noise"})
            continue
        if kind == "table" and selector_candidate_table_inclusion_class(candidate) == TABLE_INCLUSION_MUST_INCLUDE:
            continue
        score = float(candidate["priority_score"])
        if score < EVIDENCE_PRIORITY_THRESHOLDS.get(kind, 5.0):
            continue
        if kind == "table":
            if any(is_semantic_duplicate(candidate, prior) for prior in selected_tables):
                suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "semantic_duplicate_table"})
                continue
            selected.append(candidate)
            selected_tables.append(candidate)
            continue
        if kind == "materials":
            if nonlocal_selected_materials[0] >= 1:
                suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "materials_already_selected"})
                continue
            if any(is_semantic_duplicate(candidate, prior) for prior in selected):
                suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "semantic_duplicate_materials"})
                continue
            selected.append(candidate)
            nonlocal_selected_materials[0] += 1
            continue
        if kind == "method":
            family = method_family_signature(candidate)
            if len(normalize_text(candidate.get("text_content"))) < 80:
                suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "method_too_short"})
                continue
            if len(selected_method_families) >= 2:
                suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "method_budget_reached"})
                continue
            if family in selected_method_families:
                suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "duplicate_method_family"})
                continue
            if any(is_semantic_duplicate(candidate, prior) for prior in selected if prior["evidence_kind"] == "method"):
                suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "semantic_duplicate_method"})
                continue
            selected.append(candidate)
            selected_method_families.add(family)
            continue
        if not supporting_context_is_distinct(candidate, selected_tables):
            suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "supporting_not_distinct"})
            continue
        if any(is_semantic_duplicate(candidate, prior) for prior in selected):
            suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "semantic_duplicate_supporting"})
            continue
        if selected_tables and candidate_is_proxy_like(candidate):
            suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "proxy_suppressed_by_authoritative_table"})
            continue
        if normalize_text(candidate.get("section_kind")).lower() == "context":
            suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "low_value_context"})
            continue
        if nonlocal_selected_supporting[0] >= 1 and selected_tables:
            suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "supporting_budget_reached"})
            continue
        if not selected_tables and nonlocal_selected_supporting[0] >= 2:
            suppression_events.append({"candidate_id": normalize_text(candidate.get("candidate_id")), "reason": "supporting_budget_reached"})
            continue
        selected.append(candidate)
        nonlocal_selected_supporting[0] += 1

    floor_result = apply_minimal_evidence_floor(
        selected_candidates=selected,
        ranked_candidates=ranked_candidates,
        suppression_events=suppression_events,
    )
    return {
        "selection_mode": EVIDENCE_SELECTION_MODE,
        "selected_candidates": list(floor_result["selected_candidates"]),
        "suppression_events": suppression_events,
        "signals": dict(signals),
        "selected_candidate_summaries": [selector_candidate_summary(candidate) for candidate in floor_result["selected_candidates"]],
        "minimal_evidence_floor_applied": floor_result["minimal_evidence_floor_applied"],
        "floor_added_method": floor_result["floor_added_method"],
        "floor_added_materials": floor_result["floor_added_materials"],
        "floor_added_supporting": floor_result["floor_added_supporting"],
        "floor_added_formulation_surface": floor_result["floor_added_formulation_surface"],
        "floor_rationale": floor_result["floor_rationale"],
    }


def build_metadata_block(key: str, doi: str, title: str) -> str:
    parts = ["[METADATA]"]
    if key:
        parts.append(f"key: {key}")
    if doi:
        parts.append(f"doi: {doi}")
    if title:
        parts.append(f"title: {title}")
    return "\n".join(parts).strip()


def input_packing_mode() -> str:
    mode = normalize_text(os.getenv(INPUT_PACKING_MODE_ENV, "off")).lower()
    return mode if mode in {"off", ORDERED_INPUT_PACKING_MODE} else "off"


def ordered_input_packing_enabled() -> bool:
    return input_packing_mode() == ORDERED_INPUT_PACKING_MODE


def split_paragraph_blocks(text: str) -> list[str]:
    return [entry["text"] for entry in split_paragraph_entries(text)]


def classify_ordered_paragraph_block(text: str) -> tuple[str, int, int]:
    lower = text.lower()
    score = 0
    label_hits = len(re.findall(r"\b(?:f|run|sample|formulation)\s*[-:]?\s*\d{1,3}\b", lower))
    inheritance_hits = len(re.findall(r"\b(?:prepared similarly|except|all other variables unchanged|same protocol|compared with|relative to)\b", lower))
    prep_hits = len(
        re.findall(
            r"\b(?:prepared|preparation|formulations? were prepared|new formulations|strategy adopted|same method|same procedure|fixed amounts|varying|different concentrations|using fixed amounts of polymer|while maintaining|this led to|omitting the xanthones)\b",
            lower,
        )
    )
    entity_hits = len(re.findall(r"\b(?:nanospheres|nanocapsules|nanoparticles|polymer|surfactants|oil volume|myritol|pluronic|lecithin|drug concentration|polymer concentration|reagents)\b", lower))
    negative_hits = len(
        re.findall(
            r"\b(?:dsc|in vitro release|release profiles|physical stability|storage|thermal behaviour|particle size analysis|zeta potential analysis|characterization|biological activity|ocular tolerance|statistics)\b",
            lower,
        )
    )
    materials_heading = bool(
        re.match(r"^(?:materials and methods\s+)?materials\b", text.strip(), flags=re.I)
        or re.match(r"^\d+\.\d+\.?\s*materials\b", text.strip(), flags=re.I)
    )
    procurement_hits = len(
        re.findall(
            r"\b(?:purchased|procured|gifted|obtained from|supplied by|used as received|analytical grade)\b",
            lower,
        )
    )
    mw_hits = len(re.findall(r"\b(?:molecular weight|mw|kda|polymer grade|resomer|purasorb)\b", lower))
    polymer_identity_hits = len(
        re.findall(r"\b(?:plga|pcl|pla|peg-plga|polylactide|polycaprolactone|copolymer)\b", lower)
    )
    table_hits = len(re.findall(r"\btable\s+\d+\b", lower))
    sweep_hits = len(re.findall(r"\b(?:sweep|optimization|design matrix|doe|box-behnken|response surface|factorial|experimental runs|run order)\b", lower))
    reference_hits = len(re.findall(r"\[\d+\]", text))
    journal_citation_hits = len(re.findall(r"\b(?:j\.|int\.|eur\.|pharm\.|biopharm\.|drug dev\.|thesis|patent)\b", lower))
    is_reference_like = reference_hits >= 2 or journal_citation_hits >= 4
    is_conclusion_like = lower.startswith("4. conclusions") or lower.startswith("conclusions")

    if (
        ((prep_hits >= 2 and entity_hits >= 2 and negative_hits <= 2) or inheritance_hits >= 1 or label_hits >= 1)
        and not is_reference_like
        and not is_conclusion_like
    ):
        score = 20 + (6 * prep_hits) + (4 * entity_hits) + (3 * inheritance_hits) + (2 * label_hits)
        return "synthesis_method", 1, score
    if (
        (materials_heading and procurement_hits >= 1 and (mw_hits >= 1 or polymer_identity_hits >= 1))
        or (procurement_hits >= 1 and mw_hits >= 1 and polymer_identity_hits >= 1)
    ) and not is_reference_like and not is_conclusion_like:
        score = 16 + (5 * procurement_hits) + (4 * mw_hits) + (3 * polymer_identity_hits)
        return "materials_procurement", 2, score
    if table_hits or sweep_hits:
        score = 10 + (3 * table_hits) + (2 * sweep_hits)
        return "paragraph", 4, score
    score = max(1, prep_hits + entity_hits + procurement_hits + mw_hits + polymer_identity_hits + table_hits)
    return "paragraph", 5, score


def select_ordered_packing_paragraph_entries(raw_text: str, *, limit_per_kind: int = 1) -> dict[str, list[dict[str, Any]]]:
    # This reuses the historical block-ranking idea from the earlier v7pilot packer:
    # synthesis/preparation and materials/procurement are surfaced before residual narrative.
    synthesis_blocks: list[dict[str, Any]] = []
    materials_blocks: list[dict[str, Any]] = []
    fallback_blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in split_paragraph_entries(raw_text):
        paragraph = entry["text"]
        normalized = normalize_text(paragraph).lower()
        if not normalized or normalized in seen:
            continue
        block_type, _, score = classify_ordered_paragraph_block(paragraph)
        enriched = {
            "paragraph_index": entry["paragraph_index"],
            "text": paragraph,
            "block_type": block_type,
            "rank_score": score,
        }
        if block_type == "synthesis_method" and len(synthesis_blocks) < limit_per_kind and score > 0:
            synthesis_blocks.append(enriched)
            seen.add(normalized)
            continue
        if block_type == "materials_procurement" and len(materials_blocks) < limit_per_kind and score > 0:
            materials_blocks.append(enriched)
            seen.add(normalized)
            continue
        if len(fallback_blocks) < limit_per_kind and score > 0:
            fallback_blocks.append(enriched)
            seen.add(normalized)
    return {
        "synthesis_blocks": synthesis_blocks,
        "materials_blocks": materials_blocks,
        "fallback_blocks": fallback_blocks,
    }


def select_ordered_packing_paragraphs(raw_text: str, *, limit_per_kind: int = 1) -> dict[str, list[str]]:
    selected = select_ordered_packing_paragraph_entries(raw_text, limit_per_kind=limit_per_kind)
    return {
        "synthesis_blocks": [entry["text"] for entry in selected["synthesis_blocks"]],
        "materials_blocks": [entry["text"] for entry in selected["materials_blocks"]],
        "fallback_blocks": [entry["text"] for entry in selected["fallback_blocks"]],
    }


def build_trimmed_context_fallback(
    raw_text: str,
    *,
    max_chars: int,
    excluded_texts: list[str] | None = None,
) -> tuple[str, dict[str, int]]:
    excluded = {
        normalize_text(text).lower()
        for text in ensure_list(excluded_texts)
        if normalize_text(text)
    }
    segments = split_inline_heading_entries(split_section_scoped_entries(split_paragraph_entries(raw_text)))
    kept_segments: list[str] = []
    seen_segments: set[str] = set()
    stats = {
        "kept_segments": 0,
        "dropped_front_matter": 0,
        "dropped_assay_noise": 0,
        "dropped_duplicate_or_excluded": 0,
        "dropped_low_value_context": 0,
    }
    target_chars = max_chars if max_chars > 0 else len(raw_text)
    for entry in segments:
        text = normalize_text(entry.get("text"))
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen_segments or normalized in excluded:
            stats["dropped_duplicate_or_excluded"] += 1
            continue
        section_label = extract_section_label(text)
        section_kind = infer_section_kind(text, section_label)
        if should_drop_segment(text, section_kind, section_label):
            stats["dropped_front_matter"] += 1
            continue
        is_bridge_like = any(
            token in normalized
            for token in [
                "selected",
                "chosen",
                "optimal",
                "remaining studies",
                "all the following studies",
                "after the optimal",
                "had been determined",
            ]
        )
        is_local_formulation_context = (
            has_experimental_design_text_signal(text)
            or has_optimization_text_signal(text)
            or has_strong_formulation_table_signal(text)
            or ("table " in normalized and any(token in normalized for token in ["formulation", "ratio", "surfactant", "concentration"]))
        )
        if section_kind == "downstream_assay" and not is_bridge_like and not is_local_formulation_context:
            stats["dropped_assay_noise"] += 1
            continue
        if section_kind == "context" and not is_bridge_like and not is_local_formulation_context:
            stats["dropped_low_value_context"] += 1
            continue
        if any(
            token in normalized
            for token in [
                "pharmacokinetic",
                "lc-ms",
                "lc-ms/ms",
                "biodistribution",
                "cell viability",
                "animal study",
                "animals were",
                "mice were",
                "rats were",
            ]
        ) and not is_bridge_like and not is_local_formulation_context:
            stats["dropped_assay_noise"] += 1
            continue
        projected_len = len("\n\n".join(kept_segments + [text]))
        if kept_segments and projected_len > target_chars:
            break
        kept_segments.append(text)
        seen_segments.add(normalized)
    fallback_text = "\n\n".join(kept_segments)
    if target_chars > 0:
        fallback_text = fallback_text[:target_chars]
    stats["kept_segments"] = len(kept_segments)
    return fallback_text.strip(), stats


def build_controlled_evidence_pack(
    *,
    record: dict[str, str],
    raw_text: str,
    table_text: str,
    max_chars: int,
) -> tuple[str, list[str]]:
    selected = select_ordered_packing_paragraphs(raw_text)
    block_order = ["metadata"]
    blocks: list[str] = [build_metadata_block(record["key"], record["doi"], record["title"])]

    if selected["synthesis_blocks"]:
        blocks.append("[SYNTHESIS_METHOD_BLOCK]\n" + "\n\n".join(selected["synthesis_blocks"]))
        block_order.append("synthesis_method")
    if selected["materials_blocks"]:
        blocks.append("[MATERIALS_PROCUREMENT_BLOCK]\n" + "\n\n".join(selected["materials_blocks"]))
        block_order.append("materials_procurement")
    if table_text:
        blocks.append("[TABLE_BLOCK]\n" + table_text.strip())
        block_order.append("table")
    if selected["fallback_blocks"]:
        blocks.append("[PARAGRAPH_BLOCK]\n" + "\n\n".join(selected["fallback_blocks"]))
        block_order.append("paragraph")
    packed = "\n\n".join(blocks)
    if max_chars > 0:
        packed = packed[:max_chars]
    return packed, block_order


def build_evidence_blocks_artifact(
    *,
    record: dict[str, str],
    manifest_path: Path,
    text_path: Path,
    table_dir: Path | None,
    max_chars: int,
    producer_script: str,
    candidate_artifact_path: Path,
    segmentation_bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authority_metadata = build_stage2_authority_metadata(stage2_artifact_path=candidate_artifact_path)
    current_table_mode = table_mode()
    summary_enhanced = summary_first_column_enhancement_enabled()
    current_input_packing_mode = input_packing_mode()
    raw_text = text_path.read_text(encoding="utf-8", errors="replace")
    summary_candidates = collect_summary_table_candidates(table_dir) if table_dir is not None and table_dir.exists() else []
    signals = dict(segmentation_bundle.get("signals") or detect_pre_llm_signals(raw_text, summary_candidates if current_table_mode == "summary" else []))
    evidence_selection = build_evidence_priority_selection(
        segmented_candidates=list(segmentation_bundle.get("selector_candidates") or []),
        signals=signals,
    )
    selected_candidates = list(evidence_selection.get("selected_candidates") or [])
    active_table_candidates = [
        candidate
        for candidate in selected_candidates
        if candidate.get("candidate_kind") == "table"
    ]
    scored_full_mode_fallback_tables = select_scored_full_mode_fallback_tables(summary_candidates)
    explicit_table_fallback_used = False
    evidence_blocks: list[dict[str, Any]] = []
    order_tokens: list[str] = []
    selected_table_ids = {
        normalize_text(candidate.get("candidate_id"))
        for candidate in active_table_candidates
        if normalize_text(candidate.get("candidate_id"))
    }

    def append_block(
        *,
        block_id: str,
        block_type: str,
        evidence_kind: str,
        source_type: str,
        candidate_id: str | None,
        origin_locator: str,
        selection_reason: str,
        selection_feature: str,
        rank_score: int | None,
        text_content: str,
        is_table_derived: bool | None,
        table_id: str | None,
        summary_is_lossy: bool,
        requires_variable_structure: bool,
        authority_rank: Any = "",
        authority_score: Any = "",
        authority_tier: str = "",
        primary_guardrail_applied: str = "",
        primary_guardrail_reason: list[str] | None = None,
    ) -> None:
        if not normalize_text(text_content):
            return
        evidence_blocks.append(
            {
                "block_id": block_id,
                "block_type": block_type,
                "evidence_kind": evidence_kind,
                "source_type": source_type,
                "candidate_id": candidate_id,
                "origin_locator": origin_locator,
                "selection_reason": selection_reason,
                "selection_feature": selection_feature,
                "rank_score": rank_score,
                "order_index": len(evidence_blocks),
                "char_count": len(text_content),
                "text_content": text_content,
                "is_table_derived": is_table_derived,
                "table_id": normalize_text(table_id),
                "summary_is_lossy": bool(summary_is_lossy),
                "requires_variable_structure": bool(requires_variable_structure),
                "authority_rank": authority_rank,
                "authority_score": authority_score,
                "authority_tier": authority_tier,
                "primary_guardrail_applied": normalize_text(primary_guardrail_applied),
                "primary_guardrail_reason": list(primary_guardrail_reason or []),
            }
        )
        order_tokens.append(block_type)

    append_block(
        block_id=f"{record['key']}__metadata__01",
        block_type="metadata",
        source_type="document_metadata",
        candidate_id=None,
        origin_locator=to_repo_rel(manifest_path),
        selection_reason="document_identity_context",
        selection_feature="build_metadata_block",
        rank_score=None,
        text_content=build_metadata_block(record["key"], record["doi"], record["title"]),
        is_table_derived=False,
        table_id="",
        summary_is_lossy=False,
        evidence_kind="metadata",
        requires_variable_structure=False,
    )
    for candidate in selected_candidates:
        evidence_kind = normalize_text(candidate.get("evidence_kind")) or candidate_evidence_kind(candidate)
        block_type = {
            "method": "method",
            "materials": "materials",
            "table": "table",
            "supporting": "supporting",
        }.get(evidence_kind, "supporting")
        if candidate.get("candidate_kind") == "table":
            append_block(
                block_id=f"{record['key']}__table__{len([block for block in evidence_blocks if block.get('block_type') == 'table']) + 1:02d}",
                block_type="table",
                evidence_kind="table",
                source_type=str(candidate["source_type"]),
                candidate_id=normalize_text(candidate.get("candidate_id")) or None,
                origin_locator=normalize_text(candidate.get("origin_locator")),
                selection_reason="selected_high_signal_table",
                selection_feature=EVIDENCE_SELECTION_MODE,
                rank_score=int(round(float(candidate.get("priority_score", 0.0) or 0.0))),
                text_content=render_selected_table_candidate(candidate),
                is_table_derived=True,
                table_id=selector_candidate_table_id(candidate),
                summary_is_lossy=candidate_summary_is_lossy(candidate),
                requires_variable_structure=bool(
                    normalize_text(candidate.get("table_role_hint")) == "design matrix"
                    or has_variable_sweep_structure(normalize_text(candidate.get("text_content")))
                ),
                authority_rank=candidate.get("authority_rank", ""),
                authority_score=candidate.get("authority_score", ""),
                authority_tier=normalize_text(candidate.get("authority_tier")),
                primary_guardrail_applied=normalize_text(candidate.get("primary_guardrail_applied")),
                primary_guardrail_reason=candidate.get("primary_guardrail_reason") or [],
            )
            continue
        append_block(
            block_id=f"{record['key']}__{normalize_token(evidence_kind)}__{len([block for block in evidence_blocks if block.get('evidence_kind') == evidence_kind]) + 1:02d}",
            block_type=block_type,
            evidence_kind=evidence_kind,
            source_type=str(candidate["source_type"]),
            candidate_id=normalize_text(candidate.get("candidate_id")) or None,
            origin_locator=normalize_text(candidate.get("origin_locator")),
            selection_reason=f"selected_{normalize_token(evidence_kind)}_evidence",
            selection_feature=EVIDENCE_SELECTION_MODE,
            rank_score=int(round(float(candidate.get("priority_score", 0.0) or 0.0))),
            text_content=normalize_text(candidate.get("text_content")),
            is_table_derived=False,
            table_id="",
            summary_is_lossy=False,
            requires_variable_structure=False,
        )

    if not active_table_candidates and table_dir is not None and table_dir.exists():
        fallback_tables = scored_full_mode_fallback_tables or [
            {
                "candidate_kind": "table",
                "candidate_id": "",
                "origin_locator": to_repo_rel(path),
                "source_type": "table_excerpt",
                "evidence_kind": "table",
                "priority_score": 0.0,
            }
            for path in sorted(table_dir.glob("*.csv"))[:4]
        ]
        explicit_table_fallback_used = bool(fallback_tables)
        for idx, candidate in enumerate(fallback_tables, start=1):
            origin_locator = normalize_text(candidate.get("origin_locator"))
            summary_item = next(
                (
                    item
                    for item in summary_candidates
                    if to_repo_rel(item["path"]) == origin_locator
                ),
                None,
            )
            text_content = (
                render_summary_table_block(summary_item, enhancement_enabled=summary_first_column_enhancement_enabled())
                if summary_item is not None
                else normalize_text(candidate.get("text_content"))
            )
            append_block(
                block_id=f"{record['key']}__table__{idx:02d}",
                block_type="table",
                evidence_kind="table",
                source_type="table_excerpt",
                candidate_id=normalize_text(candidate.get("candidate_id")) or None,
                origin_locator=origin_locator,
                selection_reason="explicit_table_fallback",
                selection_feature=resolved_selector_strategy(current_table_mode),
                rank_score=int(round(float(candidate.get("priority_score", 0.0) or 0.0))) if candidate.get("priority_score") is not None else None,
                text_content=text_content,
                is_table_derived=True,
                table_id=normalize_text(candidate.get("table_id")) or selector_candidate_table_id(candidate),
                summary_is_lossy=True,
                requires_variable_structure=False,
                authority_rank=candidate.get("authority_rank", ""),
                authority_score=candidate.get("authority_score", ""),
                authority_tier=normalize_text(candidate.get("authority_tier")),
                primary_guardrail_applied=normalize_text(candidate.get("primary_guardrail_applied")),
                primary_guardrail_reason=candidate.get("primary_guardrail_reason") or [],
            )

    selector_strategy = EVIDENCE_SELECTION_MODE if selected_candidates else resolved_selector_strategy(current_table_mode)
    selector_debug = {
        "selection_mode": EVIDENCE_SELECTION_MODE,
        "selected_candidate_count": len(selected_candidates),
        "selected_candidate_ids": [normalize_text(candidate.get("candidate_id")) for candidate in selected_candidates if normalize_text(candidate.get("candidate_id"))],
        "selected_table_count": len(active_table_candidates),
        "selected_primary_table_count": sum(1 for candidate in active_table_candidates if normalize_text(candidate.get("authority_tier")) == TABLE_AUTHORITY_TIER_PRIMARY),
        "selected_secondary_table_count": sum(1 for candidate in active_table_candidates if normalize_text(candidate.get("authority_tier")) == TABLE_AUTHORITY_TIER_SECONDARY),
        "explicit_table_fallback_used": explicit_table_fallback_used,
        "suppression_events": list(evidence_selection.get("suppression_events") or []),
        "minimal_evidence_floor_applied": normalize_text(evidence_selection.get("minimal_evidence_floor_applied")),
        "floor_added_method": normalize_text(evidence_selection.get("floor_added_method")),
        "floor_added_materials": normalize_text(evidence_selection.get("floor_added_materials")),
        "floor_added_supporting": normalize_text(evidence_selection.get("floor_added_supporting")),
        "floor_added_formulation_surface": normalize_text(evidence_selection.get("floor_added_formulation_surface")),
        "floor_rationale": normalize_text(evidence_selection.get("floor_rationale")),
    }
    feature_activation_snapshot = {
        "ordered_evidence_packing": current_input_packing_mode == ORDERED_INPUT_PACKING_MODE,
        "summary_table_mode": current_table_mode == "summary",
        "table_selection_scoring": bool(summary_candidates),
        "selector_debug_available": bool(selected_candidates) or explicit_table_fallback_used,
        "candidate_segmentation_profile": SEGMENTATION_PROFILE,
        "section_aware_candidate_split": True,
        "candidate_table_isolation": table_dir is not None and table_dir.exists(),
        "candidate_noise_filtering": True,
        "table_authority_ranking": bool(summary_candidates),
        "doe_pre_llm_detection": signals["has_doe_signal"],
        "sequential_optimization_detection": signals["has_sequential_signal"],
        "variable_sweep_detection": signals.get("has_variable_sweep_signal", False),
        "evidence_priority_selection": bool(selected_candidates),
        "weak_importance_ordering": False,
        "semantic_overlap_suppression": True,
        "proxy_suppression_when_authoritative_table_exists": True,
        "minimal_evidence_floor": normalize_text(evidence_selection.get("minimal_evidence_floor_applied")) == "yes",
        "explicit_table_fallback": explicit_table_fallback_used,
        "archetype_detection_metadata_only": True,
    }
    coverage_summary = {
        "total_blocks": len(evidence_blocks),
        "method_blocks": sum(1 for block in evidence_blocks if block["block_type"] == "method"),
        "table_blocks": sum(1 for block in evidence_blocks if block["block_type"] == "table"),
        "primary_table_blocks": sum(1 for block in evidence_blocks if normalize_text(block.get("authority_tier")) == TABLE_AUTHORITY_TIER_PRIMARY),
        "secondary_table_blocks": sum(1 for block in evidence_blocks if normalize_text(block.get("authority_tier")) == TABLE_AUTHORITY_TIER_SECONDARY),
        "materials_blocks": sum(1 for block in evidence_blocks if block["block_type"] == "materials"),
        "supporting_blocks": sum(1 for block in evidence_blocks if block["block_type"] == "supporting"),
        "has_optimization_signal": signals["has_optimization_signal"],
        "has_doe_signal": signals["has_doe_signal"],
        "has_sequential_signal": signals["has_sequential_signal"],
        "has_variable_sweep_signal": signals.get("has_variable_sweep_signal", False),
    }
    required_top_level_fields_present = all(
        [
            record.get("key"),
            to_repo_rel(text_path),
            to_repo_rel(manifest_path),
            producer_script,
            evidence_blocks,
        ]
    )
    required_block_fields_present = all(
        all(
            field in block
            for field in [
                "block_id",
                "block_type",
                "source_type",
                "candidate_id",
                "origin_locator",
                "selection_reason",
                "selection_feature",
                "rank_score",
                "order_index",
                "char_count",
                "text_content",
                "evidence_kind",
                "table_id",
                "summary_is_lossy",
                "requires_variable_structure",
            ]
        )
        for block in evidence_blocks
    )
    technical_status = {
        "artifact_readable": True,
        "required_top_level_fields_present": required_top_level_fields_present,
        "required_block_fields_present": required_block_fields_present,
        "non_empty_evidence_blocks": bool(evidence_blocks),
        "overall": "pass"
        if required_top_level_fields_present and required_block_fields_present and bool(evidence_blocks)
        else "fail",
    }
    noise_nonconformance = any(
        evidence_text_has_noise(str(block.get("text_content", "")))
        for block in evidence_blocks
        if normalize_text(block.get("block_type")) != "metadata"
    )
    input_contract_satisfied = bool(evidence_blocks) and feature_activation_snapshot["selector_debug_available"]
    required_features_active = input_contract_satisfied and not noise_nonconformance
    design_status = {
        "input_contract_satisfied": input_contract_satisfied,
        "required_features_active": required_features_active,
        "overall": "pass" if input_contract_satisfied and required_features_active else "fail",
        "nonconformance_reasons": [
            reason
            for reason, active in [
                ("evidence_priority_selection_not_executed", feature_activation_snapshot["evidence_priority_selection"]),
                ("selector_debug_unavailable", feature_activation_snapshot["selector_debug_available"]),
            ]
            if not active
        ]
        + (["explicit_table_fallback_used"] if explicit_table_fallback_used else [])
        + (["canonical_evidence_noise_present"] if noise_nonconformance else [])
    }
    ordered_block_order = order_tokens if order_tokens else ["none"]
    artifact = {
        "paper_key": record["key"],
        "source_clean_text_path": to_repo_rel(text_path),
        "source_manifest_path": to_repo_rel(manifest_path),
        "source_scope_manifest_path": to_repo_rel(manifest_path),
        "source_candidate_artifact_path": to_repo_rel(candidate_artifact_path),
        "authority_run_dir": authority_metadata["authority_run_dir"],
        "authority_payload_root": authority_metadata["authority_payload_root"],
        "input_contract": {
            "input_packing_mode": current_input_packing_mode,
            "table_mode": current_table_mode,
            "summary_first_column_enhancement": summary_enhanced,
            "summary_view_is_lossy": True,
            "ordered_block_order": ordered_block_order,
            "selector_strategy": selector_strategy,
        },
        "producer_script": producer_script,
        "contract_version": "s2_2_evidence_blocks_v1",
        "segmentation_profile": SEGMENTATION_PROFILE,
        "selection_mode": EVIDENCE_SELECTION_MODE,
        "evidence_blocks": evidence_blocks,
        "coverage_summary": coverage_summary,
        "selector_debug": selector_debug,
        "floor_debug": {
            "minimal_evidence_floor_applied": normalize_text(evidence_selection.get("minimal_evidence_floor_applied")),
            "floor_added_method": normalize_text(evidence_selection.get("floor_added_method")),
            "floor_added_materials": normalize_text(evidence_selection.get("floor_added_materials")),
            "floor_added_supporting": normalize_text(evidence_selection.get("floor_added_supporting")),
            "floor_added_formulation_surface": normalize_text(evidence_selection.get("floor_added_formulation_surface")),
            "floor_rationale": normalize_text(evidence_selection.get("floor_rationale")),
        },
        "feature_activation_snapshot": feature_activation_snapshot,
        "technical_status": technical_status,
        "design_status": design_status,
    }
    debug_payload = None
    if table_dir is not None and table_dir.exists():
        selected_summary_names = {
            Path(candidate["origin_locator"]).name
            for candidate in selected_candidates
            if candidate["candidate_kind"] == "table"
        }
        payload_candidates: list[dict[str, Any]] = []
        if current_table_mode == "summary":
            for item in summary_candidates:
                rows = item["rows"]
                first_data_row = rows[1] if len(rows) > 1 else rows[0]
                selected_candidate = next(
                    (
                        candidate
                        for candidate in selected_candidates
                        if candidate["candidate_kind"] == "table" and candidate["origin_locator"] == to_repo_rel(item["path"])
                    ),
                    None,
                )
                selected_candidates_for_table = [
                    candidate
                    for candidate in selected_candidates
                    if candidate["candidate_kind"] == "table" and candidate["origin_locator"] == to_repo_rel(item["path"])
                ]
                payload_candidates.append(
                    {
                        "file": item["path"].name,
                        "score": item["score"],
                        "authority_score": item.get("authority_score"),
                        "authority_rank": item.get("authority_rank"),
                        "authority_tier": normalize_text(item.get("authority_tier")),
                        "selected": item["path"].name in selected_summary_names,
                        "selected_candidate_id": normalize_text(selected_candidate.get("candidate_id")) if selected_candidate is not None else "",
                        "selected_count": len(selected_candidates_for_table),
                        "row_pattern": item["row_pattern"],
                        "quality_flags": item.get("quality_flags") or [],
                        "page_number": normalize_text(item["meta"].get("page_number")),
                        "n_rows": item["meta"].get("n_rows", len(rows)),
                        "n_cols": item["meta"].get("n_cols", max((len(r) for r in rows), default=0)),
                        "caption_or_title": normalize_text(item["meta"].get("caption_or_title")),
                        "header_keywords_hit": item["meta"].get("header_keywords_hit") or [],
                        "first_data_row_preview": " | ".join(cell for cell in first_data_row if cell),
                    }
                )
        else:
            table_candidates = [
                candidate
                for candidate in ensure_list(segmentation_bundle.get("selector_candidates"))
                if candidate.get("candidate_kind") == "table"
            ]
            for candidate in table_candidates:
                origin_locator = normalize_text(candidate.get("origin_locator"))
                selected_candidates_for_table = [
                    selected_candidate
                    for selected_candidate in selected_candidates
                    if selected_candidate["candidate_kind"] == "table" and normalize_text(selected_candidate.get("origin_locator")) == origin_locator
                ]
                selected_candidate = selected_candidates_for_table[0] if selected_candidates_for_table else None
                payload_candidates.append(
                    {
                        "file": Path(origin_locator).name,
                        "score": candidate.get("table_score", 0),
                        "authority_score": candidate.get("authority_score", ""),
                        "authority_rank": candidate.get("authority_rank", ""),
                        "authority_tier": normalize_text(candidate.get("authority_tier")),
                        "selected": Path(origin_locator).name in selected_summary_names,
                        "selected_candidate_id": normalize_text(selected_candidate.get("candidate_id")) if selected_candidate is not None else "",
                        "selected_count": len(selected_candidates_for_table),
                        "row_pattern": normalize_text(candidate.get("table_row_pattern")),
                        "quality_flags": candidate.get("quality_flags") or [],
                        "page_number": "",
                        "n_rows": "",
                        "n_cols": "",
                        "caption_or_title": normalize_text(candidate.get("section_label")),
                        "header_keywords_hit": [],
                        "first_data_row_preview": normalize_text(candidate.get("text_content")).splitlines()[0][:200] if normalize_text(candidate.get("text_content")) else "",
                    }
                )
        debug_payload = {
            "document_key": record["key"],
            "table_mode": current_table_mode,
            "selection_ranking_mode": selector_strategy,
            "summary_first_column_enhancement": "yes" if summary_enhanced else "no",
            "candidate_artifact_path": to_repo_rel(candidate_artifact_path),
            "max_tables": 4,
            "selected_tables": sorted(selected_summary_names),
            "suppression_events": selector_debug.get("suppression_events") or [],
            "candidates": payload_candidates,
        }
    return artifact, debug_payload


def _normalized_table_csv_name(source_csv_path: str, candidate_id: str) -> str:
    path = Path(normalize_text(source_csv_path))
    stem = path.stem if path.stem else normalize_token(candidate_id) or "normalized_table"
    return f"{stem}__normalized.csv"


def normalize_selected_table_rows(
    rows: list[list[str]],
    *,
    table_role_hint: str,
) -> tuple[list[list[str]], list[str], dict[str, Any]]:
    normalized_rows = [list(row) for row in rows]
    actions: list[str] = []
    metadata: dict[str, Any] = {
        "numbered_row_column_index": "",
        "numbered_row_count": 0,
        "numbered_row_start_index": "",
    }
    if normalized_rows:
        first = [normalize_text(cell) for cell in normalized_rows[0]]
        if first and all(cell.isdigit() for cell in first):
            expected = [str(i) for i in range(len(first))]
            if first == expected:
                normalized_rows = normalized_rows[1:]
                actions.append("drop_enumerator_index_row")
    normalized_rows, compaction_actions = compact_table_rows_for_evidence(normalized_rows)
    actions.extend(compaction_actions)
    matrix_view = detect_shifted_numbered_matrix_view(normalized_rows)
    if matrix_view is not None:
        normalized_rows = matrix_view["normalized_rows"]
        actions.extend(matrix_view["actions"])
        metadata.update(
            {
                "numbered_row_column_index": str(int(matrix_view["anchor_col"])),
                "numbered_row_count": int(matrix_view["numbered_row_count"]),
                "numbered_row_start_index": str(int(matrix_view["first_numbered_row_index"])),
            }
        )
    return normalized_rows, actions, metadata


def detect_shifted_numbered_matrix_view(
    rows: list[list[str]],
    *,
    min_rows: int = 8,
) -> dict[str, Any] | None:
    if not rows:
        return None
    width = max((len(row) for row in rows if isinstance(row, list)), default=0)
    best_view: dict[str, Any] | None = None
    for anchor_col in range(width):
        numbered_hits: list[tuple[int, int]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, list) or anchor_col >= len(row):
                continue
            anchor = normalize_text(row[anchor_col])
            if not re.fullmatch(r"\d{1,3}\.?", anchor):
                continue
            anchor_number = int(re.sub(r"\D", "", anchor) or "0")
            trailing = [normalize_text(cell) for cell in row[anchor_col + 1 :] if normalize_text(cell)]
            numeric_like = sum(1 for cell in trailing if re.search(r"\d", cell))
            if numeric_like < 3:
                continue
            numbered_hits.append((row_index, anchor_number))
        if len(numbered_hits) < min_rows:
            continue
        longest_run: list[tuple[int, int]] = []
        current_run: list[tuple[int, int]] = []
        for hit in numbered_hits:
            row_index, anchor_number = hit
            if anchor_number == 1:
                current_run = [hit]
                if len(current_run) > len(longest_run):
                    longest_run = list(current_run)
                continue
            if current_run and anchor_number == current_run[-1][1] + 1:
                current_run.append(hit)
                if len(current_run) > len(longest_run):
                    longest_run = list(current_run)
            elif anchor_number == 1:
                current_run = [hit]
            else:
                current_run = []
        if len(longest_run) < min_rows:
            continue
        start_index = longest_run[0][0]
        view_rows: list[list[str]] = []
        for row in rows[max(0, start_index - 5) :]:
            if not isinstance(row, list):
                continue
            trimmed = [normalize_text(cell) for cell in row[anchor_col:]]
            while trimmed and not trimmed[-1]:
                trimmed.pop()
            if any(trimmed):
                view_rows.append(trimmed)
        if not view_rows:
            continue
        candidate = {
            "anchor_col": anchor_col,
            "numbered_row_count": len(longest_run),
            "first_numbered_row_index": start_index,
            "normalized_rows": view_rows,
            "actions": ["left_align_shifted_numbered_matrix"],
        }
        if best_view is None or candidate["numbered_row_count"] > best_view["numbered_row_count"]:
            best_view = candidate
    return best_view


def load_table_cells(source_csv_path: str, fallback_rows: list[list[str]] | None = None) -> list[list[str]]:
    path = Path(normalize_text(source_csv_path))
    if path and not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if path.exists():
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows.append([normalize_text(cell) for cell in row])
        if rows:
            return rows
    return [[normalize_text(cell) for cell in row] for row in ensure_list(fallback_rows)]


def flatten_header_rows(header_rows: list[list[str]], column_count: int) -> list[str]:
    flattened: list[str] = []
    for col_index in range(column_count):
        values: list[str] = []
        for row in header_rows:
            if col_index >= len(row):
                continue
            cell = normalize_text(row[col_index])
            if cell and cell.lower() not in {value.lower() for value in values}:
                values.append(cell)
        flattened.append(" / ".join(values))
    return flattened


def infer_header_structure(rows: list[list[str]]) -> dict[str, Any]:
    if not rows:
        return {
            "header_row_count": 0,
            "column_count": 0,
            "header_rows": [],
            "flattened_headers": [],
            "header_hierarchy_detected": False,
        }
    column_count = max((len(row) for row in rows if isinstance(row, list)), default=0)
    header_rows: list[list[str]] = []
    for idx, row in enumerate(rows[:3]):
        compact = [normalize_text(cell) for cell in row if normalize_text(cell)]
        if not compact:
            continue
        if idx == 0 or looks_like_header_row(row):
            header_rows.append([normalize_text(cell) for cell in row])
        else:
            break
    if not header_rows:
        header_rows = [[normalize_text(cell) for cell in rows[0]]]
    flattened_headers = flatten_header_rows(header_rows, column_count)
    return {
        "header_row_count": len(header_rows),
        "column_count": column_count,
        "header_rows": header_rows,
        "flattened_headers": flattened_headers,
        "header_hierarchy_detected": len(header_rows) > 1,
    }


def build_normalized_row_entries(
    normalized_matrix: list[list[str]],
    *,
    header_structure: dict[str, Any],
    numbered_row_column_index: str,
) -> list[dict[str, Any]]:
    header_row_count = int(header_structure.get("header_row_count") or 0)
    headers = [normalize_text(item) for item in ensure_list(header_structure.get("flattened_headers"))]
    numbering_index = int(numbered_row_column_index) if normalize_text(numbered_row_column_index).isdigit() else -1
    row_entries: list[dict[str, Any]] = []
    for row_index, row in enumerate(normalized_matrix[header_row_count:], start=header_row_count + 1):
        cells = [normalize_text(cell) for cell in row]
        cell_map: dict[str, str] = {}
        for col_index, value in enumerate(cells):
            header = headers[col_index] if col_index < len(headers) else f"column_{col_index + 1}"
            if header and value:
                cell_map[header] = value
        row_entries.append(
            {
                "row_index": row_index,
                "row_number": normalize_text(cells[numbering_index]) if 0 <= numbering_index < len(cells) else "",
                "cells": cells,
                "cell_map": cell_map,
                "row_text": " | ".join(value for value in cells if value),
            }
        )
    return row_entries


def build_row_identity_signals(
    normalized_matrix: list[list[str]],
    *,
    header_structure: dict[str, Any],
    numbered_row_column_index: str,
) -> dict[str, Any]:
    row_entries = build_normalized_row_entries(
        normalized_matrix,
        header_structure=header_structure,
        numbered_row_column_index=numbered_row_column_index,
    )
    row_numbers = [normalize_text(item.get("row_number")) for item in row_entries if normalize_text(item.get("row_number"))]
    first_column_labels = [normalize_text(item["cells"][0]) for item in row_entries if item.get("cells") and normalize_text(item["cells"][0])]
    return {
        "row_number_column_index": normalize_text(numbered_row_column_index),
        "row_number_values": row_numbers[:100],
        "first_column_labels": first_column_labels[:100],
        "row_pattern": infer_row_pattern(row_numbers or first_column_labels),
    }


def classify_execution_table_type(
    normalized_matrix: list[list[str]],
    *,
    meta: dict[str, Any],
    table_role_hint: str,
    normalization_metadata: dict[str, Any],
) -> str:
    header_parts = extract_informative_header_parts(normalized_matrix)
    signal_text = build_summary_table_signal_text(normalized_matrix[:6], meta, header_parts)
    base_hint = infer_table_role_hint(header_parts, {**meta, "_signal_text": signal_text})
    numbered_row_count = int(normalization_metadata.get("numbered_row_count") or 0)
    row_ids = [normalize_text(row[0]) for row in normalized_matrix[1:] if isinstance(row, list) and row and normalize_text(row[0])]
    row_pattern = infer_row_pattern(row_ids)
    has_formulation = has_strong_formulation_table_signal(signal_text) or any(
        token in signal_text for token in ["formulation", "drug", "polymer", "surfactant", "loading", "ratio"]
    )
    has_optimization = any(token in signal_text for token in ["selected", "optimal", "optimized", "desirability"])
    has_parameter = any(token in signal_text for token in ["factor", "level", "run", "concentration", "ratio", "parameter", "variable"])
    has_doe = (
        base_hint == "design matrix"
        or numbered_row_count >= 8
        or row_pattern in {"numeric runs", "F-numbered rows"}
        or any(token in signal_text for token in ["factorial", "design", "independent variables", "dependent variables", "coded levels"])
    )
    if has_doe and (has_formulation or has_optimization):
        return "mixed_table"
    if has_doe:
        return "DOE_table"
    if has_formulation and has_optimization:
        return "mixed_table"
    if base_hint == "optimization" or has_optimization:
        return "optimization_table"
    if base_hint == "formulation" or has_formulation:
        return "formulation_table"
    if has_parameter:
        return "parameter_sweep_table"
    return "non_formulation_table"


def compute_reconstruction_confidence(
    *,
    representation_status: str,
    selector_readiness_label: str,
    normalization_actions: list[str],
    normalized_row_count: int,
    raw_row_count: int,
) -> float:
    confidence = 0.55
    if selector_readiness_label == "ready":
        confidence += 0.2
    if representation_status in {"repaired_summary", "raw_summary"}:
        confidence += 0.1
    if normalized_row_count >= max(raw_row_count - 1, 1):
        confidence += 0.1
    if "reload_from_source_table_asset" in normalization_actions:
        confidence += 0.1
    if "derive_matrix_from_selected_companion_table" in normalization_actions:
        confidence += 0.05
    return max(0.0, min(1.0, round(confidence, 2)))


def payload_fraction_numeric_cells(payload: dict[str, Any]) -> float:
    raw_cells = payload.get("raw_cells") or []
    nonempty_cells = 0
    numeric_cells = 0
    for row in raw_cells[:12]:
        if not isinstance(row, list):
            continue
        for cell in row:
            normalized = normalize_text(cell)
            if not normalized:
                continue
            nonempty_cells += 1
            if re.search(r"\d", normalized):
                numeric_cells += 1
    if nonempty_cells <= 0:
        return 0.0
    return round(float(numeric_cells) / float(nonempty_cells), 4)


def payload_primary_guardrail_reasons(payload: dict[str, Any]) -> list[str]:
    table_type = normalize_text(payload.get("table_type")).lower()
    table_role_hint = normalize_text(payload.get("table_role_hint")).lower()
    raw_cells = payload.get("raw_cells") or []
    first_labels = ensure_list((payload.get("row_identity_signals") or {}).get("first_column_labels"))
    header_structure = payload.get("header_structure") if isinstance(payload.get("header_structure"), dict) else {}
    headers = " ".join(normalize_text(cell) for row in ensure_list(header_structure.get("header_rows")) for cell in ensure_list(row)).lower()
    source_caption = normalize_text(payload.get("source_caption_or_title")).lower()
    signal_text = " ".join([headers, source_caption]).strip()
    fraction_numeric_cells = payload_fraction_numeric_cells(payload)
    prose_like_rows = table_prose_like_row_count(raw_cells)
    reasons: list[str] = []
    if table_type == "non_formulation_table":
        reasons.append("table_type_non_formulation_table")
    if table_role_hint in PRIMARY_EXCLUDED_ROLE_HINTS:
        reasons.append(f"table_role_hint_{normalize_token(table_role_hint)}")
    if fraction_numeric_cells < 0.08 and not normalize_text(headers):
        reasons.append("very_low_numeric_density_without_header_keywords")
    if prose_like_rows >= 3 and len([label for label in first_labels if normalize_text(label)]) < 3:
        reasons.append("narrative_or_figure_caption_dominated")
    if any(token in signal_text for token in PRIMARY_EXCLUDED_PATTERN_TOKENS):
        reasons.append("non_formulation_pattern_surface")
    return reasons


def payload_primary_eligibility_signals(payload: dict[str, Any]) -> list[str]:
    table_type = normalize_text(payload.get("table_type")).lower()
    representation_status = normalize_text(payload.get("representation_status")).lower()
    row_identity_signals = payload.get("row_identity_signals") if isinstance(payload.get("row_identity_signals"), dict) else {}
    row_pattern = normalize_text(row_identity_signals.get("row_pattern")).lower()
    first_labels = [normalize_text(label) for label in ensure_list(row_identity_signals.get("first_column_labels")) if normalize_text(label)]
    header_structure = payload.get("header_structure") if isinstance(payload.get("header_structure"), dict) else {}
    headers = " ".join(normalize_text(cell) for row in ensure_list(header_structure.get("header_rows")) for cell in ensure_list(row)).lower()
    data_row_count = int(payload.get("data_row_count") or 0)
    header_signal_hits = sum(1 for token in PRIMARY_ELIGIBLE_HEADER_TOKENS if token in headers)
    signals: list[str] = []
    if representation_status not in {"repair_insufficient", "unrepaired_corrupted"}:
        signals.append("representation_not_repair_insufficient")
    if row_pattern in {"numeric runs", "f-numbered rows"} or len(first_labels) >= 3:
        signals.append("stable_row_anchors")
    if data_row_count >= 3:
        signals.append("multiple_condition_rows")
    if table_type in {"formulation_table", "doe_table", "optimization_table", "mixed_table"} and header_signal_hits >= 2:
        signals.append("formulation_numeric_header_surface")
    return signals


def payload_hard_drop_reasons(payload: dict[str, Any]) -> list[str]:
    raw_cells = payload.get("raw_cells") or []
    first_labels = ensure_list((payload.get("row_identity_signals") or {}).get("first_column_labels"))
    prose_label_count = sum(1 for label in first_labels if len(re.findall(r"[A-Za-z]+", normalize_text(label))) >= 8)
    header_structure = payload.get("header_structure") if isinstance(payload.get("header_structure"), dict) else {}
    headers = " ".join(normalize_text(cell) for row in ensure_list(header_structure.get("header_rows")) for cell in ensure_list(row)).lower()
    source_caption = normalize_text(payload.get("source_caption_or_title")).lower()
    signal_text = " ".join([headers, source_caption]).strip()
    fraction_numeric_cells = payload_fraction_numeric_cells(payload)
    prose_like_rows = table_prose_like_row_count(raw_cells)
    representation_status = normalize_text(payload.get("representation_status")).lower()
    reasons: list[str] = []
    if representation_status == "unrepaired_corrupted":
        reasons.append("unrepaired_corrupted_representation")
    if prose_like_rows >= 3 and len([label for label in first_labels if normalize_text(label)]) < 3:
        reasons.append("narrative_or_figure_caption_dominated")
    if prose_label_count >= 3 and fraction_numeric_cells < 0.2:
        reasons.append("prose_row_label_surface")
    if fraction_numeric_cells < 0.08 and not normalize_text(headers) and len([label for label in first_labels if normalize_text(label)]) < 3:
        reasons.append("very_low_numeric_density_without_header_keywords")
    if count_cue_hits(signal_text, TABLE_AUTHORITY_FRONTMATTER_DEMOTION_TOKENS) >= 2:
        reasons.append("front_matter_or_bibliographic_surface")
    if re.search(r"\b\d+\s+of\s+\d+\b", signal_text):
        reasons.append("journal_page_or_review_surface")
    if any(token in signal_text for token in HARD_DROP_SIGNAL_TOKENS):
        reasons.append("high_confidence_non_content_fragment")
    return list(dict.fromkeys(reasons))


def payload_inclusion_class(payload: dict[str, Any]) -> str:
    if payload_hard_drop_reasons(payload):
        return TABLE_INCLUSION_HARD_DROP
    signals = set(payload_primary_eligibility_signals(payload))
    if "formulation_numeric_header_surface" in signals:
        return TABLE_INCLUSION_MUST_INCLUDE
    if {"stable_row_anchors", "multiple_condition_rows"}.issubset(signals):
        return TABLE_INCLUSION_MUST_INCLUDE
    return TABLE_INCLUSION_OPTIONAL_CONTEXT


def payload_guardrail_priority(payload: dict[str, Any]) -> float:
    priority = float(payload.get("authority_score") or 0.0)
    table_type = normalize_text(payload.get("table_type")).lower()
    if table_type == "formulation_table":
        priority += 40.0
    elif table_type == "doe_table":
        priority += 36.0
    elif table_type == "mixed_table":
        priority += 32.0
    elif table_type == "optimization_table":
        priority += 28.0
    priority += min(10.0, float(int(payload.get("data_row_count") or 0)))
    priority += 3.0 * float(len(payload_primary_eligibility_signals(payload)))
    return priority


def apply_primary_table_guardrail_to_preserved_payloads(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        return entries
    for index, entry in enumerate(entries, start=1):
        reasons = payload_primary_guardrail_reasons(entry)
        structural_reasons = primary_structural_guardrail_reasons(reasons)
        signals = payload_primary_eligibility_signals(entry)
        inclusion_class = payload_inclusion_class(entry)
        hard_drop_reasons = payload_hard_drop_reasons(entry)
        primary_guardrail_applied = False
        if normalize_text(entry.get("authority_tier")) == TABLE_AUTHORITY_TIER_PRIMARY and (
            inclusion_class == TABLE_INCLUSION_HARD_DROP or structural_reasons or not signals
        ):
            primary_guardrail_applied = True
            entry["authority_tier"] = TABLE_AUTHORITY_TIER_SECONDARY if bool(entry.get("preserved_by_authority_ranking")) else TABLE_AUTHORITY_TIER_WEAK_SECONDARY
        entry["primary_guardrail_applied"] = "yes" if primary_guardrail_applied else "no"
        entry["primary_guardrail_reason"] = list(reasons)
        entry["primary_eligibility_signals"] = list(signals)
        entry["table_inclusion_class"] = inclusion_class
        entry["hard_drop_reason"] = list(hard_drop_reasons)
        if entry.get("authority_rank") in {None, ""}:
            entry["authority_rank"] = index
    return entries


def build_table_authority_validation_row(
    *,
    paper_key: str,
    table_id: str,
    source_table_reference: str,
    table_type: str,
    raw_cells: list[list[str]],
    normalized_matrix: list[list[str]],
    normalization_actions: list[str],
    selector_readiness_label: str,
    authority_rank: Any,
    authority_score: Any,
    authority_tier: str,
) -> dict[str, Any]:
    raw_row_count = len(raw_cells)
    normalized_row_count = len(normalized_matrix)
    raw_width = max((len(row) for row in raw_cells if isinstance(row, list)), default=0)
    normalized_width = max((len(row) for row in normalized_matrix if isinstance(row, list)), default=0)
    normalized_row_texts = [
        normalize_text(" | ".join(normalize_text(cell) for cell in row if normalize_text(cell)))
        for row in normalized_matrix
        if isinstance(row, list)
    ]
    non_empty_texts = [row for row in normalized_row_texts if row]
    duplicate_row_count = max(0, len(non_empty_texts) - len(set(non_empty_texts)))
    allowed_row_drop_actions = {"drop_enumerator_index_row", "caption_recovery"}
    missing_rows_detected = normalized_row_count < raw_row_count and not bool(
        set(normalization_actions).intersection(allowed_row_drop_actions)
    )
    return {
        "paper_key": paper_key,
        "table_id": table_id,
        "source_table_reference": source_table_reference,
        "table_type": table_type,
        "row_count_before": raw_row_count,
        "row_count_after": normalized_row_count,
        "missing_rows_detected": "yes" if missing_rows_detected else "no",
        "duplicate_row_count": duplicate_row_count,
        "column_count_before": raw_width,
        "column_count_after": normalized_width,
        "column_collapse_detected": "yes" if normalized_width < raw_width else "no",
        "selector_readiness_label": selector_readiness_label,
        "authority_rank": authority_rank,
        "authority_score": authority_score,
        "authority_tier": authority_tier,
        "normalization_actions": "|".join(normalization_actions),
    }


def build_normalized_table_payload_artifact(
    *,
    record: dict[str, str],
    out_dir: Path,
    producer_script: str,
    evidence_artifact_path: Path,
    evidence_artifact: dict[str, Any],
    segmentation_bundle: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_table_ids = {
        normalize_text(block.get("candidate_id"))
        for block in ensure_list(evidence_artifact.get("evidence_blocks"))
        if normalize_text(block.get("candidate_id")) and bool(block.get("is_table_derived"))
    }
    selected_entries: list[dict[str, Any]] = []
    candidate_rows = ensure_list(segmentation_bundle.get("selector_candidates"))
    candidate_lookup = {
        normalize_text(candidate.get("candidate_id")): candidate
        for candidate in candidate_rows
        if normalize_text(candidate.get("candidate_id"))
    }
    payload_dir = normalized_table_payloads_path(out_dir, record["key"]).parent / "payloads"
    validation_rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        if normalize_text(candidate.get("candidate_id")) not in selected_table_ids:
            continue
        if normalize_text(candidate.get("candidate_kind")) != "table":
            continue
        item = candidate.get("item")
        if not isinstance(item, dict):
            continue
        source_csv_path = normalize_text(item.get("repair_source_csv_path")) or normalize_text(candidate.get("origin_locator"))
        rows = item.get("rows") or []
        if not isinstance(rows, list) or not rows:
            continue
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        raw_cells = load_table_cells(source_csv_path, rows)
        normalized_rows, normalization_actions, normalization_metadata = normalize_selected_table_rows(
            rows,
            table_role_hint=normalize_text(candidate.get("table_role_hint")),
        )
        raw_source_path = Path(source_csv_path)
        if not raw_source_path.is_absolute():
            raw_source_path = (PROJECT_ROOT / raw_source_path).resolve()
        if int(normalization_metadata.get("numbered_row_count") or 0) < 8 and raw_source_path.exists():
            raw_source_rows: list[list[str]] = []
            with raw_source_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    raw_source_rows.append([normalize_text(cell) for cell in row])
            raw_normalized_rows, raw_actions, raw_metadata = normalize_selected_table_rows(
                raw_source_rows,
                table_role_hint=normalize_text(candidate.get("table_role_hint")),
            )
            if int(raw_metadata.get("numbered_row_count") or 0) > int(normalization_metadata.get("numbered_row_count") or 0):
                normalized_rows = raw_normalized_rows
                normalization_actions = list(dict.fromkeys(list(normalization_actions) + ["reload_from_source_table_asset"] + raw_actions))
                normalization_metadata = raw_metadata
        payload_basis_csv_path = source_csv_path
        payload_basis_candidate_id = normalize_text(candidate.get("candidate_id"))
        payload_basis_same_source = True
        if (
            normalize_text(candidate.get("table_role_hint")) == "design matrix"
            and int(normalization_metadata.get("numbered_row_count") or 0) < 8
        ):
            best_companion: dict[str, Any] | None = None
            for companion_id in selected_table_ids:
                if companion_id == normalize_text(candidate.get("candidate_id")):
                    continue
                companion = candidate_lookup.get(companion_id)
                if not isinstance(companion, dict):
                    continue
                companion_item = companion.get("item")
                if not isinstance(companion_item, dict):
                    continue
                companion_rows = companion_item.get("rows") or []
                if not isinstance(companion_rows, list) or not companion_rows:
                    continue
                companion_normalized_rows, companion_actions, companion_metadata = normalize_selected_table_rows(
                    companion_rows,
                    table_role_hint=normalize_text(companion.get("table_role_hint")),
                )
                companion_source_csv_path = normalize_text(companion_item.get("repair_source_csv_path")) or normalize_text(companion.get("origin_locator"))
                companion_source_path = Path(companion_source_csv_path)
                if not companion_source_path.is_absolute():
                    companion_source_path = (PROJECT_ROOT / companion_source_path).resolve()
                if int(companion_metadata.get("numbered_row_count") or 0) < 8 and companion_source_path.exists():
                    companion_source_rows: list[list[str]] = []
                    with companion_source_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                        reader = csv.reader(handle)
                        for row in reader:
                            companion_source_rows.append([normalize_text(cell) for cell in row])
                    raw_companion_rows, raw_companion_actions, raw_companion_metadata = normalize_selected_table_rows(
                        companion_source_rows,
                        table_role_hint=normalize_text(companion.get("table_role_hint")),
                    )
                    if int(raw_companion_metadata.get("numbered_row_count") or 0) > int(companion_metadata.get("numbered_row_count") or 0):
                        companion_normalized_rows = raw_companion_rows
                        companion_actions = list(dict.fromkeys(list(companion_actions) + ["reload_from_source_table_asset"] + raw_companion_actions))
                        companion_metadata = raw_companion_metadata
                companion_count = int(companion_metadata.get("numbered_row_count") or 0)
                if companion_count < 8:
                    continue
                companion_payload = {
                    "normalized_rows": companion_normalized_rows,
                    "actions": companion_actions,
                    "metadata": companion_metadata,
                    "candidate_id": companion_id,
                    "source_csv_path": companion_source_csv_path,
                }
                if best_companion is None or companion_count > int(best_companion["metadata"].get("numbered_row_count") or 0):
                    best_companion = companion_payload
            if best_companion is not None:
                normalized_rows = best_companion["normalized_rows"]
                normalization_actions = list(normalization_actions) + ["derive_matrix_from_selected_companion_table"]
                normalization_actions.extend(best_companion["actions"])
                normalization_actions = list(dict.fromkeys(normalization_actions))
                normalization_metadata = dict(best_companion["metadata"])
                payload_basis_csv_path = best_companion["source_csv_path"]
                payload_basis_candidate_id = best_companion["candidate_id"]
                payload_basis_same_source = False
        csv_name = _normalized_table_csv_name(source_csv_path, normalize_text(candidate.get("candidate_id")))
        normalized_csv_path = payload_dir / csv_name
        normalized_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with normalized_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(normalized_rows)
        table_id = derive_stable_table_id(source_csv_path, meta)
        header_structure = infer_header_structure(normalized_rows)
        normalized_row_entries = build_normalized_row_entries(
            normalized_rows,
            header_structure=header_structure,
            numbered_row_column_index=normalize_text(normalization_metadata.get("numbered_row_column_index")),
        )
        row_identity_signals = build_row_identity_signals(
            normalized_rows,
            header_structure=header_structure,
            numbered_row_column_index=normalize_text(normalization_metadata.get("numbered_row_column_index")),
        )
        table_type = classify_execution_table_type(
            normalized_rows,
            meta=meta,
            table_role_hint=normalize_text(candidate.get("table_role_hint")),
            normalization_metadata=normalization_metadata,
        )
        reconstruction_confidence = compute_reconstruction_confidence(
            representation_status=normalize_text(item.get("representation_status")),
            selector_readiness_label=normalize_text(item.get("selector_readiness_label")),
            normalization_actions=normalization_actions,
            normalized_row_count=len(normalized_rows),
            raw_row_count=len(raw_cells),
        )
        first_column_labels = [
            normalize_text(row[0])
            for row in normalized_rows[1:]
            if isinstance(row, list) and row and normalize_text(row[0])
        ]
        source_table_reference = source_csv_path
        source_table_asset_id = Path(source_csv_path).stem
        selected_entries.append(
            {
                "table_id": table_id,
                "candidate_id": normalize_text(candidate.get("candidate_id")),
                "source_csv_path": source_csv_path,
                "source_table_reference": source_table_reference,
                "source_table_asset_id": source_table_asset_id,
                "source_filename": Path(source_csv_path).name,
                "source_table_id": table_id,
                "source_caption_or_title": normalize_text(meta.get("caption_or_title")),
                "table_role_hint": normalize_text(candidate.get("table_role_hint")),
                "table_type": table_type,
                "selector_readiness_label": normalize_text(item.get("selector_readiness_label")),
                "representation_status": normalize_text(item.get("representation_status")),
                "authority_rank": item.get("authority_rank"),
                "authority_score": item.get("authority_score"),
                "authority_tier": normalize_text(item.get("authority_tier")),
                "table_inclusion_class": normalize_text(item.get("table_inclusion_class")),
                "hard_drop_reason": item.get("hard_drop_reason") or [],
                "authority_score_breakdown": item.get("authority_score_breakdown") or {},
                "preserved_by_authority_ranking": bool(item.get("preserved_by_authority_ranking")),
                "primary_guardrail_applied": normalize_text(item.get("primary_guardrail_applied")),
                "primary_guardrail_reason": item.get("primary_guardrail_reason") or [],
                "repair_actions": ensure_list(item.get("repair_actions")),
                "normalization_actions": normalization_actions,
                "same_source_table_asset": payload_basis_same_source and bool(item.get("same_source_table_asset", True)),
                "derived_from_candidate_id": "" if payload_basis_same_source else payload_basis_candidate_id,
                "repair_source_manifest_path": normalize_text(item.get("repair_source_manifest_path")),
                "payload_basis_source_csv_path": payload_basis_csv_path,
                "payload_basis_candidate_id": payload_basis_candidate_id,
                "normalized_csv_path": to_repo_rel(normalized_csv_path),
                "row_count": len(normalized_rows),
                "normalized_row_count": len(normalized_rows),
                "raw_row_count": len(raw_cells),
                "header_row_count": int(header_structure.get("header_row_count") or 0),
                "data_row_count": max(0, len(normalized_rows) - 1),
                "has_row_numbering": bool(int(normalization_metadata.get("numbered_row_count") or 0)),
                "header_structure": header_structure,
                "raw_cells": raw_cells,
                "normalized_rows": normalized_row_entries,
                "normalized_matrix": normalized_rows,
                "row_identity_signals": row_identity_signals,
                "reconstruction_confidence": reconstruction_confidence,
                "first_column_labels": first_column_labels[:25],
                "numbered_row_column_index": normalize_text(normalization_metadata.get("numbered_row_column_index")),
                "numbered_row_count": int(normalization_metadata.get("numbered_row_count") or 0),
                "numbered_row_start_index": normalize_text(normalization_metadata.get("numbered_row_start_index")),
            }
        )
        validation_rows.append(
            build_table_authority_validation_row(
                paper_key=record["key"],
                table_id=table_id,
                source_table_reference=source_table_reference,
                table_type=table_type,
                raw_cells=raw_cells,
                normalized_matrix=normalized_rows,
                normalization_actions=normalization_actions,
                selector_readiness_label=normalize_text(item.get("selector_readiness_label")),
                authority_rank=item.get("authority_rank"),
                authority_score=item.get("authority_score"),
                authority_tier=normalize_text(item.get("authority_tier")),
            )
        )
    selected_entries = apply_primary_table_guardrail_to_preserved_payloads(selected_entries)
    selected_entries.sort(
        key=lambda item: (
            *candidate_table_neutral_order_key(
                {
                    "source_table_id": normalize_text(item.get("source_table_id")),
                    "table_id": normalize_text(item.get("table_id")),
                    "origin_locator": normalize_text(item.get("source_table_reference")),
                    "candidate_id": normalize_text(item.get("candidate_id")),
                }
            ),
        )
    )
    return {
        "paper_key": record["key"],
        "producer_script": producer_script,
        "contract_version": "s2_2_normalized_table_payloads_v1",
        "source_evidence_artifact_path": to_repo_rel(evidence_artifact_path),
        "authority_run_dir": to_repo_rel(out_dir.parent),
        "authority_payload_root": to_repo_rel(out_dir / NORMALIZED_TABLE_PAYLOADS_SUBDIR),
        "full_table_authority_role": "execution_grade_table_authority",
        "normalized_table_payloads": selected_entries,
    }, validation_rows


def build_prompt_preview_row(
    *,
    document: dict[str, str],
    prompt_text: str,
    table_mode_value: str,
    summary_enhanced: bool,
    input_packing_mode_value: str,
    ordered_block_order: str,
    evidence_artifact_path: str,
    evidence_artifact: dict[str, Any],
    technical_status_overall: str,
    design_status_overall: str,
) -> dict[str, Any]:
    runtime_metadata = build_prompt_runtime_metadata(evidence_artifact)
    paper_text_index = prompt_text.find("Paper text:\n")
    evidence_pack_index = prompt_text.find("Evidence pack:\n")
    table_excerpts_index = prompt_text.find("Table excerpts:\n")
    evidence_blocks = [
        block
        for block in ensure_list(evidence_artifact.get("evidence_blocks"))
        if isinstance(block, dict) and normalize_text(block.get("text_content"))
    ]
    prompt_render_bundle = build_prompt_render_bundle(evidence_artifact)
    normalized_prompt_text = normalize_text(prompt_text)
    exact_payload_counts = Counter(prompt_render_bundle["normalized_rendered_payloads"])
    exact_duplicate_block_count = sum(count - 1 for count in exact_payload_counts.values() if count > 1)
    all_selected_blocks_included = all(
        payload in normalized_prompt_text for payload in prompt_render_bundle["normalized_rendered_payloads"]
    )
    uses_evidence_pack_only = (
        evidence_pack_index >= 0
        and paper_text_index < 0
        and table_excerpts_index < 0
    )
    truncation_detected = not all_selected_blocks_included
    readiness_reasons: list[str] = []
    if not normalize_text(prompt_text):
        readiness_reasons.append("empty_prompt")
    if not uses_evidence_pack_only:
        readiness_reasons.append("unexpected_prompt_layout")
    if truncation_detected:
        readiness_reasons.append("selected_block_missing_from_prompt")
    if exact_duplicate_block_count > 0:
        readiness_reasons.append("exact_duplicate_evidence_block_payloads")
    if normalize_text(design_status_overall) != "pass":
        readiness_reasons.append("upstream_evidence_nonconformant")
    prompt_size_status = prompt_size_policy_status(len(prompt_text))
    if prompt_size_status != "healthy":
        readiness_reasons.append("prompt_exceeds_healthy_size_policy")
    s2_3_ready_overall = "pass" if not readiness_reasons else "fail"
    layout_class = "unknown"
    if evidence_pack_index >= 0:
        layout_class = "ordered_controlled_evidence_pack"
    elif paper_text_index >= 0 and table_excerpts_index >= 0:
        if paper_text_index < table_excerpts_index:
            layout_class = "raw_prefix_then_table_excerpts"
        else:
            layout_class = "unexpected_table_before_text"
    prompt_head_preview = normalize_text(prompt_text[:260])
    prompt_tail_preview = normalize_text(prompt_text[-260:]) if len(prompt_text) > 260 else normalize_text(prompt_text)
    live_prompt_contains_runtime_metadata = any(note in prompt_text for note in runtime_metadata["runtime_notes"])
    return {
        "document_key": document["document_key"],
        "doi": document["doi"],
        "source_mode": document["source_mode"],
        "table_mode": table_mode_value,
        "summary_first_column_enhancement": "yes" if summary_enhanced else "no",
        "table_mode_used": runtime_metadata["prompt_render_bundle"]["table_mode_used"],
        "summary_applied": runtime_metadata["prompt_render_bundle"]["summary_applied"],
        "reason_for_full_table": runtime_metadata["prompt_render_bundle"]["reason_for_full_table"],
        "input_packing_mode": input_packing_mode_value,
        "prompt_layout_class": layout_class,
        "paper_text_marker_index": paper_text_index,
        "evidence_pack_marker_index": evidence_pack_index,
        "table_excerpts_marker_index": table_excerpts_index,
        "ordered_block_order": ordered_block_order,
        "evidence_artifact_path": evidence_artifact_path,
        "technical_status_overall": technical_status_overall,
        "design_status_overall": design_status_overall,
        "s2_3_ready_overall": s2_3_ready_overall,
        "s2_3_readiness_reasons": "|".join(readiness_reasons),
        "selected_evidence_block_count": len(evidence_blocks),
        "all_selected_blocks_included": "yes" if all_selected_blocks_included else "no",
        "uses_evidence_pack_only": "yes" if uses_evidence_pack_only else "no",
        "truncation_detected": "yes" if truncation_detected else "no",
        "exact_duplicate_block_count": exact_duplicate_block_count,
        "prompt_size_policy_status": prompt_size_status,
        "prompt_length": len(prompt_text),
        "live_prompt_header_mode": "semantic_only",
        "runtime_metadata_removed_from_live_prompt": "yes" if not live_prompt_contains_runtime_metadata else "no",
        "live_prompt_contains_runtime_metadata": "yes" if live_prompt_contains_runtime_metadata else "no",
        "prompt_head_preview": prompt_head_preview,
        "prompt_tail_preview": prompt_tail_preview,
    }


def build_s2_2_boundary_validation_row(
    *,
    candidate_artifact: dict[str, Any],
    evidence_artifact: dict[str, Any],
    normalized_payload_artifact: dict[str, Any],
) -> dict[str, Any]:
    evidence_blocks = [
        block for block in ensure_list(evidence_artifact.get("evidence_blocks")) if isinstance(block, dict)
    ]
    selector_debug = evidence_artifact.get("selector_debug") or {}
    design_status = evidence_artifact.get("design_status") or {}
    feature_snapshot = evidence_artifact.get("feature_activation_snapshot") or {}
    noise_detected = any(
        evidence_text_has_noise(str(block.get("text_content", "")))
        for block in evidence_blocks
        if normalize_text(block.get("block_type")) != "metadata"
    )
    selector_truth_mismatch = (
        bool(selector_debug.get("selected_candidate_count"))
        and not bool(feature_snapshot.get("evidence_priority_selection"))
    )
    status = "pass"
    if normalize_text(design_status.get("overall")) != "pass":
        status = "fail"
    if noise_detected or selector_truth_mismatch:
        status = "fail"
    return {
        "paper_key": normalize_text(evidence_artifact.get("paper_key")),
        "candidate_count": len(ensure_list(candidate_artifact.get("candidate_blocks"))),
        "evidence_block_count": len(evidence_blocks),
        "table_block_count": sum(1 for block in evidence_blocks if normalize_text(block.get("block_type")) == "table"),
        "selected_table_payload_count": len(ensure_list(normalized_payload_artifact.get("normalized_table_payloads"))),
        "selector_strategy": normalize_text(evidence_artifact.get("input_contract", {}).get("selector_strategy")),
        "evidence_priority_selection_executed": "yes" if selector_debug.get("selected_candidate_count") else "no",
        "explicit_table_fallback_used": "yes" if selector_debug.get("explicit_table_fallback_used") else "no",
        "selector_truth_mismatch": "yes" if selector_truth_mismatch else "no",
        "design_status_overall": normalize_text(design_status.get("overall")),
        "suppression_event_count": len(ensure_list(selector_debug.get("suppression_events"))),
        "noise_detected": "yes" if noise_detected else "no",
        "status": status,
    }


def build_s2_3_boundary_validation_row(
    *,
    prompt_preview_row: dict[str, Any],
) -> dict[str, Any]:
    status = "pass"
    if normalize_text(prompt_preview_row.get("s2_3_ready_overall")) != "pass":
        status = "fail"
    return {
        "paper_key": normalize_text(prompt_preview_row.get("document_key")),
        "s2_3_ready_overall": normalize_text(prompt_preview_row.get("s2_3_ready_overall")),
        "uses_evidence_pack_only": normalize_text(prompt_preview_row.get("uses_evidence_pack_only")),
        "all_selected_blocks_included": normalize_text(prompt_preview_row.get("all_selected_blocks_included")),
        "selected_evidence_block_count": prompt_preview_row.get("selected_evidence_block_count", ""),
        "truncation_detected": normalize_text(prompt_preview_row.get("truncation_detected")),
        "exact_duplicate_block_count": prompt_preview_row.get("exact_duplicate_block_count", ""),
        "prompt_size_policy_status": normalize_text(prompt_preview_row.get("prompt_size_policy_status")),
        "prompt_layout_class": normalize_text(prompt_preview_row.get("prompt_layout_class")),
        "status": status,
    }


def build_candidate_segmentation_debug_rows(candidate_artifact: dict[str, Any], candidate_artifact_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in ensure_list(candidate_artifact.get("candidate_blocks")):
        if not isinstance(candidate, dict):
            continue
        rows.append(
            {
                "document_key": normalize_text(candidate_artifact.get("paper_key")),
                "candidate_artifact_path": to_repo_rel(candidate_artifact_path),
                "segmentation_profile": normalize_text(candidate_artifact.get("segmentation_profile")),
                "candidate_id": normalize_text(candidate.get("candidate_id")),
                "candidate_type": normalize_text(candidate.get("candidate_type")),
                "source_type": normalize_text(candidate.get("source_type")),
                "origin_locator": normalize_text(candidate.get("origin_locator")),
                "section_label": normalize_text(candidate.get("section_label")),
                "section_kind": normalize_text(candidate.get("section_kind")),
                "segmentation_method": normalize_text(candidate.get("segmentation_method")),
                "split_trigger": normalize_text(candidate.get("split_trigger")),
                "noise_flags": "|".join(str(item) for item in ensure_list(candidate.get("noise_flags")) if str(item).strip()),
                "quality_flags": "|".join(str(item) for item in ensure_list(candidate.get("quality_flags")) if str(item).strip()),
                "representation_status": normalize_text(candidate.get("representation_status")),
                "repair_primary_source": normalize_text(candidate.get("repair_primary_source")),
                "repair_actions": "|".join(str(item) for item in ensure_list(candidate.get("repair_actions")) if str(item).strip()),
                "material_difference_from_raw": str(bool(candidate.get("material_difference_from_raw"))).lower() if candidate.get("candidate_type") == "table" else "",
                "selector_readiness_label": normalize_text(candidate.get("selector_readiness_label")),
                "authority_rank": candidate.get("authority_rank", ""),
                "authority_score": candidate.get("authority_score", ""),
                "authority_tier": normalize_text(candidate.get("authority_tier")),
                "preserved_by_authority_ranking": "yes" if bool(candidate.get("preserved_by_authority_ranking")) else "no" if candidate.get("candidate_type") == "table" else "",
                "primary_guardrail_applied": normalize_text(candidate.get("primary_guardrail_applied")),
                "primary_guardrail_reason": "|".join(str(item) for item in ensure_list(candidate.get("primary_guardrail_reason")) if str(item).strip()),
                "unresolved_reason": normalize_text(candidate.get("unresolved_reason")),
                "raw_table_preview": normalize_text(candidate.get("raw_table_preview"))[:260],
                "repaired_table_preview": normalize_text(candidate.get("repaired_table_preview"))[:260],
                "text_preview": normalize_text(candidate.get("text_preview") or candidate.get("text_content"))[:260],
            }
        )
    return rows


def build_live_prompt(record: dict[str, str], evidence_artifact: dict[str, Any]) -> str:
    runtime_metadata = build_prompt_runtime_metadata(evidence_artifact)
    ordered_blocks = list(runtime_metadata["prompt_render_bundle"]["rendered_payloads"])
    schema = {
        "paper_key": record["key"],
        "table_scopes": [
            {
                "table_id": "string",
                "scope_kind": "doe_table | formulation_table | optimization_table | sequential_child | downstream_variant_table | non_formulation | unclear",
                "is_formulation_bearing": True,
                "is_doe": False,
                "parent_table_hint": "string",
                "confidence": "high | medium | low",
            }
        ],
        "semantic_signals": {
            "has_variable_sweep": True,
            "has_sequential_optimization": False,
            "has_parent_child_table_relation": False,
            "has_downstream_non_synthesis_variants": False,
            "has_measurement_only_variants": False,
            "primary_preparation_method_hint": "string",
            "primary_variable_names": ["string"],
            "selected_condition_hints": ["string"],
        },
        "formulation_candidates": [
            {
                "candidate_id": "string",
                "candidate_kind": "single_formulation | formulation_family | variant_formulation | unclear",
                "source_table_id": "string",
                "label_hint": "string",
                "instance_role": "synthesis_core | downstream_variant | control | comparative | characterization_only | unclear",
                "parent_candidate_hint": "string",
                "core_change_hint": "string",
                "shared_context_hint": "string",
                "status": "reported | partial | ambiguous",
                "confidence": "high | medium | low",
            }
        ],
    }
    evidence_text = "\n\n".join(ordered_blocks).strip()
    input_block = "Evidence pack:\n" + f"{evidence_text}\n"
    return (
        "You are extracting Stage2 semantic understanding objects for a governed comparator slice.\n"
        "Rules:\n"
        "- Preserve ambiguity explicitly.\n"
        "- Emit understanding-level structure only.\n"
        "- Do not emit provenance payloads, quoted supporting-text objects, or any other evidence-reporting fields.\n"
        "- Do not perform relation resolution, inheritance closure, or final-row materialization.\n"
        "- Do not emit execution-ready markers, boundary markers, or Stage3-like relation structures.\n"
        "- Do not emit component families, process-context families, response-signal families, or near-final row payloads.\n"
        "- Every top-level key in the schema is required.\n"
        "- If a family is absent, return an empty list or an empty understanding object as appropriate.\n"
        "- Use literal paper table labels such as 'Table 1' or 'Table 2' when a table is identifiable.\n"
        "- Multiple candidate tables may be provided; some may be formulation-bearing and others may be downstream/result-only or non-formulation.\n"
        "- Deterministic pre-filtering did not decide the one true formulation table for you; make the semantic table judgment from the evidence pack.\n"
        "- `table_scopes` should answer what each included table means at a high level, including whether it is formulation-bearing, DOE-like, downstream-only, or non-formulation.\n"
        "- `semantic_signals` should capture paper-level semantic cues only.\n"
        "- `formulation_candidates` should describe likely formulation units only, using concise hints rather than materialized rows.\n"
        "- Mark downstream handling, storage, measurement-only, or characterization-only variants at the understanding level only.\n"
        "- Keep parent-child hints only when they are explicit enough to support a high-level hint.\n"
        "- If a table is clearly DOE-like, set `is_doe=true` and use `scope_kind='doe_table'`.\n"
        "- Use `scope_kind='formulation_table'` only when the table semantically represents formulation instances or formulation families rather than measurement-only outputs.\n"
        "- If an included table is supportive but not formulation-bearing, keep it in `table_scopes` with `is_formulation_bearing=false` instead of pretending it is irrelevant.\n"
        "- If a later table seems to continue a selected earlier condition, use `scope_kind='sequential_child'` and set `parent_table_hint` when possible.\n"
        "- If the paper reports downstream non-synthesis variants, keep them visible through `semantic_signals` and `formulation_candidates` but do not resolve them into execution logic.\n"
        "- Keep outputs compact and interpretation-focused.\n"
        "- Return valid JSON only.\n\n"
        "- Treat the evidence pack as the complete semantic input for this paper.\n"
        "- Use table summaries and method/materials blocks as semantic context, not as instructions to reproduce runtime settings.\n\n"
        f"Paper key: {record['key']}\n"
        f"Title: {record['title']}\n\n"
        "Return JSON with exactly these top-level keys:\n"
        f"{json.dumps(schema, ensure_ascii=True, indent=2)}\n\n"
        f"{input_block}"
    )


def find_legacy_raw_response(raw_dir: Path, key: str) -> Path:
    # Replay surfaces in this repository have been written with either legacy
    # text or JSON response suffixes depending on the producing step. Keep the
    # lookup narrow to the paper key, but accept the governed saved-response
    # formats that actually exist on disk.
    matches = sorted(raw_dir.glob(f"*{key}*.json")) + sorted(raw_dir.glob(f"*{key}*.txt"))
    if not matches:
        matches = sorted(raw_dir.glob(f"*{key}*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"Legacy raw response not found for {key} under {raw_dir}")
    return matches[0]


def is_live_v2_raw_response_shape(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    if is_shrunken_live_contract(parsed):
        return True
    return any(
        key in parsed
        for key in [
            "formulation_candidates",
            "component_candidates",
            "variable_candidates",
            "measurement_candidates",
            "relation_hints",
            "evidence_spans",
            "unassigned_observations",
        ]
    )


def field_object(formulation: dict[str, Any], field_name: str) -> dict[str, Any]:
    fields = formulation.get("fields") or {}
    value = fields.get(field_name) if isinstance(fields, dict) else None
    if isinstance(value, dict):
        return value
    return {}


def field_value_text(formulation: dict[str, Any], field_name: str) -> str:
    obj = field_object(formulation, field_name)
    value_text = normalize_text(obj.get("value_text"))
    if value_text:
        return value_text
    value = obj.get("value")
    if value is None:
        return ""
    return normalize_text(value)


def add_evidence_span(
    spans: list[dict[str, Any]],
    *,
    source_region_type: str,
    source_locator_text: str,
    supporting_text: str,
) -> str:
    span_id = f"span_{len(spans) + 1:03d}"
    spans.append(
        {
            "span_id": span_id,
            "source_region_type": source_region_type or "unknown",
            "source_locator_text": source_locator_text,
            "supporting_text": supporting_text,
        }
    )
    return span_id


def build_component_candidates(
    key: str,
    formulation_id: str,
    formulation: dict[str, Any],
    evidence_span_ids: list[str],
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for role, name_field, amount_field in COMPONENT_FIELD_SPECS:
        name_text = field_value_text(formulation, canonical_field_name(name_field))
        if not name_text:
            continue
        pair = (role, name_text.lower())
        if pair in seen:
            continue
        seen.add(pair)
        component = {
            "component_id": f"{key}__{formulation_id}__component_{len(components) + 1:02d}",
            "formulation_candidate_id": formulation_id,
            "component_name": name_text,
            "component_role": role,
            "amount_text": field_value_text(formulation, canonical_field_name(amount_field)) if amount_field else "",
            "amount_kind": (
                "concentration"
                if amount_field in {"plga_mass_mg", "surfactant_concentration_text", "pva_conc_percent", "drug_feed_amount_text"}
                else "unknown"
            ),
            "phase_hint": "",
            "ambiguity_note": "",
            "evidence_span_ids": evidence_span_ids,
        }
        components.append(component)
    return components


def build_variable_candidates(
    key: str,
    formulation_id: str,
    formulation: dict[str, Any],
    evidence_span_ids: list[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for field_name in VARIABLE_FIELDS:
        raw_value = field_value_text(formulation, field_name)
        if not raw_value:
            continue
        variable_name = field_name
        role = "identity_signal"
        if field_name in {"emul_method", "emul_type"}:
            role = "process_setting"
        candidates.append(
            {
                "variable_id": f"{key}__{formulation_id}__variable_{len(candidates) + 1:02d}",
                "formulation_candidate_id": formulation_id,
                "variable_name": variable_name,
                "value_text": raw_value,
                "variable_role": role,
                "ambiguity_note": "",
                "evidence_span_ids": evidence_span_ids,
            }
        )
    identity_variables = parse_json_list(formulation.get("identity_variables_json"))
    for item in identity_variables:
        if not isinstance(item, dict):
            continue
        name = normalize_text(item.get("name") or item.get("name_raw"))
        value_text = normalize_text(item.get("value") or item.get("value_raw"))
        if not name or not value_text:
            continue
        candidates.append(
            {
                "variable_id": f"{key}__{formulation_id}__variable_{len(candidates) + 1:02d}",
                "formulation_candidate_id": formulation_id,
                "variable_name": name,
                "value_text": value_text,
                "variable_role": "doe_factor" if normalize_token(name) in DOE_TOKENS else "identity_signal",
                "ambiguity_note": "",
                "evidence_span_ids": evidence_span_ids,
            }
        )
    return candidates


def build_measurement_candidates(
    key: str,
    formulation_id: str,
    formulation: dict[str, Any],
    evidence_span_ids: list[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for field_name in MEASUREMENT_FIELDS:
        value_text = field_value_text(formulation, field_name)
        if not value_text:
            continue
        measure_name = field_name
        if field_name == "zeta_mV":
            measure_name = "zeta_potential"
        elif field_name == "encapsulation_efficiency_percent":
            measure_name = "encapsulation_efficiency"
        elif field_name == "loading_content_percent":
            measure_name = "loading_content"
        unit_text = ""
        if field_name == "size_nm":
            unit_text = "nm"
        elif field_name == "zeta_mV":
            unit_text = "mV"
        elif field_name.endswith("_percent"):
            unit_text = "%"
        candidates.append(
            {
                "measurement_id": f"{key}__{formulation_id}__measurement_{len(candidates) + 1:02d}",
                "formulation_candidate_id": formulation_id,
                "measurement_name": measure_name,
                "value_text": value_text,
                "unit_text": unit_text,
                "ambiguity_note": "",
                "evidence_span_ids": evidence_span_ids,
            }
        )
    return candidates


def convert_legacy_raw_response_to_v2(
    *,
    record: dict[str, str],
    raw_response_path: Path,
    raw_response_text: str,
    authority_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sanitized_text, sanitization_audit = sanitize_stage2_json_text(raw_response_text)
    if json_sanitization_applied(sanitization_audit):
        write_json_sanitization_audit(raw_response_path, sanitization_audit)
    parsed = json.loads(sanitized_text)
    if is_live_v2_raw_response_shape(parsed):
        return normalize_replayed_live_document(
            record,
            parsed,
            raw_response_path,
            authority_metadata=authority_metadata,
        )
    formulations = parsed.get("formulations") or []
    text_path = Path(record["text_path"])
    if not text_path.is_absolute():
        text_path = (PROJECT_ROOT / text_path).resolve()
    table_dir = resolve_tables_dir_for_record(record, text_path, record["key"])
    evidence_spans: list[dict[str, Any]] = []
    formulation_candidates: list[dict[str, Any]] = []
    component_candidates: list[dict[str, Any]] = []
    variable_candidates: list[dict[str, Any]] = []
    measurement_candidates: list[dict[str, Any]] = []
    relation_hints: list[dict[str, Any]] = []
    unassigned_observations: list[dict[str, Any]] = []

    paper_notes = normalize_text(parsed.get("paper_notes"))
    paper_note_span_ids: list[str] = []
    if paper_notes:
        paper_note_span_ids.append(
            add_evidence_span(
                evidence_spans,
                source_region_type="paper_notes",
                source_locator_text=str(raw_response_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                supporting_text=paper_notes,
            )
        )
        unassigned_observations.append(
            {
                "observation_id": f"{record['key']}__unassigned_001",
                "category": "shared_context",
                "note": paper_notes,
                "evidence_span_ids": paper_note_span_ids,
            }
        )
        if "not presented in a systematic, enumerated way" in paper_notes.lower():
            unassigned_observations.append(
                {
                    "observation_id": f"{record['key']}__unassigned_002",
                    "category": "reported_but_unassigned",
                    "note": "Paper reports formulation-variable sweeps that are not enumerated as stable formulation instances in the saved LLM response.",
                    "evidence_span_ids": paper_note_span_ids,
                }
            )

    for idx, formulation in enumerate(formulations, start=1):
        if not isinstance(formulation, dict):
            continue
        raw_label = normalize_text(formulation.get("raw_formulation_label") or formulation.get("formulation_id") or f"candidate_{idx}")
        formulation_id = normalize_token(formulation.get("formulation_id") or raw_label or f"candidate_{idx}")
        span_ids = list(paper_note_span_ids)
        support_refs = formulation.get("supporting_evidence_refs")
        if isinstance(support_refs, list):
            support_note = "; ".join(normalize_text(item) for item in support_refs if normalize_text(item))
        else:
            support_note = normalize_text(support_refs)
        snippet = raw_label
        if support_note:
            snippet = f"{raw_label} | refs={support_note}"
        span_ids.append(
            add_evidence_span(
                evidence_spans,
                source_region_type="legacy_raw_response",
                source_locator_text=str(raw_response_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                supporting_text=snippet,
            )
        )
        ambiguity_note = ""
        if record["key"] == "5GIF3D8W":
            ambiguity_note = "Saved LLM response enumerates optimized formulations only; broader sweep structure remains unassigned."
        formulation_candidates.append(
            {
                "candidate_id": formulation_id,
                "raw_label": raw_label,
                "normalized_label": normalize_token(raw_label),
                "instance_kind": normalize_text(formulation.get("instance_kind")) or "unclear",
                "formulation_role": normalize_text(formulation.get("formulation_role")) or "unclear",
                "parent_candidate_id": normalize_token(formulation.get("parent_instance_id")),
                "ambiguity_note": ambiguity_note,
                "evidence_span_ids": span_ids,
                "status": "ambiguous" if ambiguity_note else "reported",
            }
        )
        component_candidates.extend(build_component_candidates(record["key"], formulation_id, formulation, span_ids))
        variable_candidates.extend(build_variable_candidates(record["key"], formulation_id, formulation, span_ids))
        measurement_candidates.extend(build_measurement_candidates(record["key"], formulation_id, formulation, span_ids))
        parent_id = normalize_token(formulation.get("parent_instance_id"))
        if parent_id:
            relation_hints.append(
                {
                    "relation_id": f"{record['key']}__relation_{len(relation_hints) + 1:02d}",
                    "source_candidate_id": formulation_id,
                    "target_candidate_id": parent_id,
                    "relation_type": "inherits_from",
                    "note": "Preserved from legacy raw response parent_instance_id.",
                    "evidence_span_ids": span_ids,
                }
            )

    source_table_files = []
    if table_dir and table_dir.exists():
        source_table_files = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in sorted(table_dir.glob("*.csv"))]
    return finalize_llm_first_document(
        {
        "document_key": record["key"],
        "doi": record["doi"],
        "title": record["title"],
        "source_mode": "legacy_llm_raw_response_replay_to_stage2_v2",
        "source_raw_response_path": str(raw_response_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_text_path": str(text_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_table_files": source_table_files,
        "formulation_candidates": formulation_candidates,
        "component_candidates": component_candidates,
        "variable_candidates": variable_candidates,
        "measurement_candidates": measurement_candidates,
        "relation_hints": relation_hints,
        "evidence_spans": evidence_spans,
        "unassigned_observations": unassigned_observations,
        },
        authority_metadata=authority_metadata,
    )


def normalize_live_document(
    record: dict[str, str],
    parsed: dict[str, Any],
    raw_response_path: Path,
    *,
    authority_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_live_v2_document(
        record=record,
        parsed=parsed,
        raw_response_path=raw_response_path,
        source_mode="live_llm_stage2_v2",
        replay_mode="none",
        authority_metadata=authority_metadata,
    )


def normalize_replayed_live_document(
    record: dict[str, str],
    parsed: dict[str, Any],
    raw_response_path: Path,
    *,
    authority_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_live_v2_document(
        record=record,
        parsed=parsed,
        raw_response_path=raw_response_path,
        source_mode="saved_raw_live_v2_replay_to_stage2_v2",
        replay_mode="saved_raw_response_replay",
        authority_metadata=authority_metadata,
    )


def build_live_v2_document(
    *,
    record: dict[str, str],
    parsed: dict[str, Any],
    raw_response_path: Path,
    source_mode: str,
    replay_mode: str,
    authority_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_path = Path(record["text_path"])
    if not text_path.is_absolute():
        text_path = (PROJECT_ROOT / text_path).resolve()
    table_dir = resolve_tables_dir_for_record(record, text_path, record["key"])
    source_table_files = []
    if table_dir and table_dir.exists():
        source_table_files = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in sorted(table_dir.glob("*.csv"))]
    if is_shrunken_live_contract(parsed):
        return finalize_llm_first_document(
            {
                "document_key": record["key"],
                "paper_key": record["key"],
                "doi": record["doi"],
                "title": record["title"],
                "source_mode": source_mode,
                "replay_mode": replay_mode,
                "source_raw_response_schema": "stage2_live_v2_raw_response_minimal_contract",
                "source_raw_response_path": str(raw_response_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_text_path": str(text_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_table_files": source_table_files,
                "table_scopes": normalize_shrunken_table_scopes(parsed.get("table_scopes")),
                "semantic_signals": normalize_shrunken_semantic_signals(parsed.get("semantic_signals")),
                "formulation_candidates": normalize_shrunken_formulation_candidates(parsed.get("formulation_candidates")),
            },
            authority_metadata=authority_metadata,
        )
    parsed = prune_invalid_live_inheritance_markers(parsed, raw_response_path)
    return finalize_llm_first_document(
        {
        "document_key": record["key"],
        "doi": record["doi"],
        "title": record["title"],
        "source_mode": source_mode,
        "replay_mode": replay_mode,
        "source_raw_response_schema": "stage2_live_v2_raw_response",
        "source_raw_response_path": str(raw_response_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_text_path": str(text_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_table_files": source_table_files,
        "formulation_candidates": parsed.get("formulation_candidates") or [],
        "component_candidates": parsed.get("component_candidates") or [],
        "variable_candidates": parsed.get("variable_candidates") or [],
        "measurement_candidates": parsed.get("measurement_candidates") or [],
        "relation_hints": parsed.get("relation_hints") or [],
        "evidence_spans": parsed.get("evidence_spans") or [],
        "unassigned_observations": parsed.get("unassigned_observations") or [],
        "table_formulation_scopes": normalize_marker_list(parsed.get("table_formulation_scopes")),
        "table_variable_roles": normalize_marker_list(parsed.get("table_variable_roles")),
        "selection_markers": normalize_marker_list(parsed.get("selection_markers")),
        "inheritance_markers": normalize_marker_list(parsed.get("inheritance_markers")),
        "preparation_inheritance_markers": normalize_marker_list(parsed.get("preparation_inheritance_markers")),
        "boundary_markers": normalize_marker_list(parsed.get("boundary_markers")),
        },
        authority_metadata=authority_metadata,
    )


def infer_semantic_scope_declarations(document: dict[str, Any]) -> list[dict[str, Any]]:
    if "table_scopes" in document and "semantic_signals" in document:
        document_key = normalize_text(document.get("document_key") or document.get("paper_key") or document.get("key"))
        signals = normalize_shrunken_semantic_signals(document.get("semantic_signals"))
        scopes = normalize_shrunken_table_scopes(document.get("table_scopes"))
        declarations: list[dict[str, Any]] = [
            {
                "scope_id": f"{document_key}__llm_document_scope__01",
                "scope_kind": DOCUMENT_SCOPE_KIND,
                "declared_by": LLM_PARSED,
                "authorizes_row_materialization_modes": [LLM_SEMANTIC_DISCOVERY],
                "row_enumeration_required": "no",
                "table_scope_refs": [],
                "declaration_basis": "default_llm_semantic_document_scope",
            }
        ]
        if signals["has_variable_sweep"] and any(scope.get("is_doe") for scope in scopes):
            declarations.append(
                {
                    "scope_id": f"{document_key}__llm_declared_doe_scope__01",
                    "scope_kind": DOE_SCOPE_KIND,
                    "declared_by": LLM_PARSED,
                    "authorizes_row_materialization_modes": [
                        LLM_SEMANTIC_DISCOVERY,
                        "deterministic_row_expansion_within_llm_scope",
                    ],
                    "row_enumeration_required": "yes",
                    "table_scope_refs": [
                        normalize_text(scope.get("table_id"))
                        for scope in scopes
                        if scope.get("is_doe") and normalize_text(scope.get("table_id"))
                    ],
                    "declared_doe_factors": list(signals.get("primary_variable_names") or []),
                    "declaration_basis": "llm_understanding_level_doe_scope",
                }
            )
        for index, scope in enumerate(scopes, start=1):
            if not scope.get("is_formulation_bearing") or scope.get("is_doe"):
                continue
            table_id = normalize_text(scope.get("table_id"))
            if not table_id:
                continue
            declarations.append(
                {
                    "scope_id": f"{document_key}__table_formulation_scope__{index:02d}",
                    "scope_kind": TABLE_FORMULATION_SCOPE_KIND,
                    "declared_by": LLM_PARSED,
                    "authorizes_row_materialization_modes": [
                        LLM_SEMANTIC_DISCOVERY,
                        "table_row_expansion_v1",
                    ],
                    "row_enumeration_required": "yes",
                    "table_scope_refs": [table_id],
                    "declaration_basis": f"llm_understanding_level_table_scope::{normalize_text(scope.get('scope_kind'))}",
                }
            )
        return declarations
    augment_document_with_table_markers(document)
    document_key = normalize_text(document.get("document_key") or document.get("key"))
    declarations: list[dict[str, Any]] = [
        {
            "scope_id": f"{document_key}__llm_document_scope__01",
            "scope_kind": DOCUMENT_SCOPE_KIND,
            "declared_by": LLM_PARSED,
            "authorizes_row_materialization_modes": [LLM_SEMANTIC_DISCOVERY],
            "row_enumeration_required": "no",
            "table_scope_refs": [],
            "declaration_basis": "default_llm_semantic_document_scope",
        }
    ]
    variable_candidates = [
        item for item in document.get("variable_candidates", []) if isinstance(item, dict)
    ]
    evidence_spans = [item for item in document.get("evidence_spans", []) if isinstance(item, dict)]
    unassigned_observations = [
        item for item in document.get("unassigned_observations", []) if isinstance(item, dict)
    ]
    doe_factor_names = {
        normalize_token(item.get("variable_name"))
        for item in variable_candidates
        if normalize_text(item.get("variable_role")) == "doe_factor"
    }
    evidence_by_id = {
        normalize_text(item.get("span_id") or item.get("evidence_span_id")): item
        for item in evidence_spans
        if normalize_text(item.get("span_id") or item.get("evidence_span_id"))
    }
    scope_text = " ".join(
        normalize_text(item.get("supporting_text") or item.get("note"))
        for item in [*evidence_spans, *unassigned_observations]
    ).lower()
    strong_doe_language = any(
        token in scope_text
        for token in [
            "box-behnken",
            "response surface",
            "factorial",
            "experimental design",
            "design matrix",
            "design expert",
            "run order",
            " doe ",
        ]
    )
    if "doe" in scope_text:
        strong_doe_language = True
    numbered_formulation_count = sum(
        1
        for item in ensure_list(document.get("formulation_candidates"))
        if isinstance(item, dict)
        and re.fullmatch(r"(?:F[- ]?\d{1,3}|\d{1,3}\.?)", normalize_text(item.get("raw_label") or item.get("raw_formulation_label")))
    )
    table_scope_refs: list[str] = []
    seen_table_refs: set[str] = set()
    for variable in variable_candidates:
        if normalize_text(variable.get("variable_role")) != "doe_factor":
            continue
        for span_id in ensure_list(variable.get("evidence_span_ids")):
            span = evidence_by_id.get(normalize_text(span_id))
            if not span:
                continue
            region = normalize_text(span.get("source_region_type")).lower()
            locator = normalize_text(span.get("source_locator_text"))
            if not (region.startswith("table") or "table" in locator.lower()):
                continue
            ref = locator or normalize_text(span.get("supporting_text"))
            if not ref or ref in seen_table_refs:
                continue
            seen_table_refs.add(ref)
            table_scope_refs.append(ref)
    if not table_scope_refs and strong_doe_language:
        for span in evidence_spans:
            region = normalize_text(span.get("source_region_type")).lower()
            locator = normalize_text(span.get("source_locator_text"))
            supporting_text = normalize_text(span.get("supporting_text")).lower()
            if not (region.startswith("table") or "table" in locator.lower()):
                continue
            if not any(
                token in supporting_text or token in locator.lower()
                for token in [
                    "box-behnken",
                    "experimental design",
                    "independent variable",
                    "dependent variable",
                    "effect of independent",
                    "design",
                    "level",
                ]
            ):
                continue
            ref = locator or normalize_text(span.get("supporting_text"))
            if not ref or ref in seen_table_refs:
                continue
            seen_table_refs.add(ref)
            table_scope_refs.append(ref)
    if doe_factor_names and table_scope_refs and (strong_doe_language or numbered_formulation_count >= 8):
        declarations.append(
            {
                "scope_id": f"{document_key}__llm_declared_doe_scope__01",
                "scope_kind": DOE_SCOPE_KIND,
                "declared_by": LLM_PARSED,
                "authorizes_row_materialization_modes": [
                    LLM_SEMANTIC_DISCOVERY,
                    "deterministic_row_expansion_within_llm_scope",
                ],
                "row_enumeration_required": "yes",
                "table_scope_refs": table_scope_refs,
                "declared_doe_factors": sorted(doe_factor_names),
                "declaration_basis": (
                    "llm_detected_strong_doe_language_plus_doe_factor_candidates_plus_table_scopes"
                    if strong_doe_language
                    else "llm_detected_numbered_formulation_sweep_plus_doe_factor_candidates_plus_table_scopes"
                ),
            }
        )
    for table_scope in [
        item
        for item in ensure_list(document.get("table_formulation_scopes"))
        if isinstance(item, dict) and bool(item.get("is_formulation_table")) and not bool(
            next(
                (
                    boundary.get("is_doe")
                    for boundary in ensure_list(document.get("boundary_markers"))
                    if isinstance(boundary, dict) and normalize_text(boundary.get("table_id")) == normalize_text(item.get("table_id"))
                ),
                False,
            )
        )
    ]:
        scope_id = normalize_text(table_scope.get("scope_id"))
        if not scope_id:
            continue
        declarations.append(
            {
                "scope_id": scope_id,
                "scope_kind": TABLE_FORMULATION_SCOPE_KIND,
                "declared_by": normalize_text(table_scope.get("marker_provenance")) or LLM_EXPLICIT,
                "authorizes_row_materialization_modes": [
                    LLM_SEMANTIC_DISCOVERY,
                    "table_row_expansion_v1",
                ],
                "row_enumeration_required": "yes",
                "table_scope_refs": [normalize_text(table_scope.get("table_id"))],
                "declaration_basis": f"llm_explicit_formulation_bearing_table::{normalize_text(table_scope.get('table_type'))}",
            }
        )
    return declarations


def default_llm_scope_ref(document: dict[str, Any], candidate_id: str) -> str:
    declarations = infer_semantic_scope_declarations(document)
    default_scope = next(
        (
            normalize_text(item.get("scope_id"))
            for item in declarations
            if normalize_text(item.get("scope_kind")) == DOCUMENT_SCOPE_KIND
        ),
        "",
    )
    return f"{default_scope}|candidate:{candidate_id}" if candidate_id else default_scope


def finalize_llm_first_document(
    document: dict[str, Any],
    *,
    authority_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if "table_scopes" in document and "semantic_signals" in document:
        document["paper_key"] = normalize_text(document.get("paper_key") or document.get("document_key") or document.get("key"))
        document["document_key"] = normalize_text(document.get("document_key") or document.get("paper_key"))
        document["table_scopes"] = normalize_shrunken_table_scopes(document.get("table_scopes"))
        document["semantic_signals"] = normalize_shrunken_semantic_signals(document.get("semantic_signals"))
        document["formulation_candidates"] = normalize_shrunken_formulation_candidates(document.get("formulation_candidates"))
    else:
        augment_document_with_table_markers(document)
    document["stage2_semantic_source_mode"] = STAGE2_SEMANTIC_SOURCE_MODE
    document["semantic_universe_authority"] = LLM_SEMANTIC_DISCOVERY
    authority_metadata = authority_metadata or {}
    document["authority_run_dir"] = normalize_text(authority_metadata.get("authority_run_dir"))
    document["authority_payload_root"] = normalize_text(authority_metadata.get("authority_payload_root"))
    declarations = infer_semantic_scope_declarations(document)
    document["semantic_scope_declarations"] = declarations
    attach_table_scope_locators(document)
    for item in document.get("formulation_candidates", []):
        if not isinstance(item, dict):
            continue
        candidate_id = normalize_text(item.get("candidate_id") or item.get("formulation_candidate_id"))
        item["stage2_semantic_source_mode"] = normalize_text(item.get("stage2_semantic_source_mode")) or STAGE2_SEMANTIC_SOURCE_MODE
        item["semantic_universe_authority"] = normalize_text(item.get("semantic_universe_authority")) or LLM_SEMANTIC_DISCOVERY
        item["row_materialization_mode"] = normalize_text(item.get("row_materialization_mode")) or LLM_SEMANTIC_DISCOVERY
        item["semantic_scope_authority"] = normalize_text(item.get("semantic_scope_authority")) or LLM_DECLARED_SCOPE
        item["semantic_scope_ref"] = normalize_text(item.get("semantic_scope_ref")) or default_llm_scope_ref(document, candidate_id)
    return document


def summary_row(document: dict[str, Any]) -> dict[str, Any]:
    if "table_scopes" in document and "semantic_signals" in document:
        signals = normalize_shrunken_semantic_signals(document.get("semantic_signals"))
        return {
            "document_key": document["document_key"],
            "doi": document.get("doi", ""),
            "source_mode": document["source_mode"],
            "stage2_semantic_source_mode": normalize_text(document.get("stage2_semantic_source_mode")) or STAGE2_SEMANTIC_SOURCE_MODE,
            "table_scope_count": len(normalize_shrunken_table_scopes(document.get("table_scopes"))),
            "formulation_candidate_count": len(
                [item for item in ensure_list(document.get("formulation_candidates")) if isinstance(item, dict)]
            ),
            "has_variable_sweep": "yes" if signals["has_variable_sweep"] else "no",
            "has_sequential_optimization": "yes" if signals["has_sequential_optimization"] else "no",
            "has_parent_child_table_relation": "yes" if signals["has_parent_child_table_relation"] else "no",
            "has_downstream_non_synthesis_variants": "yes" if signals["has_downstream_non_synthesis_variants"] else "no",
            "has_measurement_only_variants": "yes" if signals["has_measurement_only_variants"] else "no",
            "primary_preparation_method_hint": signals["primary_preparation_method_hint"],
            "primary_variable_count": len(signals["primary_variable_names"]),
            "selected_condition_hint_count": len(signals["selected_condition_hints"]),
        }
    variable_names = {
        normalize_token(item.get("variable_name"))
        for item in document.get("variable_candidates", [])
        if isinstance(item, dict)
    }
    measurement_names = {
        normalize_token(item.get("measurement_name"))
        for item in document.get("measurement_candidates", [])
        if isinstance(item, dict)
    }
    components_by_formulation: dict[str, set[str]] = {}
    for item in document.get("component_candidates", []):
        if not isinstance(item, dict):
            continue
        fid = normalize_text(item.get("formulation_candidate_id"))
        cid = normalize_text(item.get("component_id"))
        if not fid or not cid:
            continue
        components_by_formulation.setdefault(fid, set()).add(cid)
    multi_component_count = sum(1 for ids in components_by_formulation.values() if len(ids) >= 2)
    source_text_path = normalize_text(document.get("source_text_path"))
    table_dir: Path | None = None
    if source_text_path:
        text_path = Path(source_text_path)
        if not text_path.is_absolute():
            text_path = (PROJECT_ROOT / text_path).resolve()
        table_dir = resolve_tables_dir(text_path, str(document.get("document_key") or document.get("key") or ""))
    enhancement_enabled = summary_first_column_enhancement_enabled()
    row_anchor_preview = ""
    table_selection_strategy = "score_ranked_top_4"
    if enhancement_enabled and table_dir is not None and table_dir.exists():
        row_anchor_preview_parts: list[str] = []
        for item in select_summary_tables(table_dir, max_tables=4):
            row_ids = [row[0] for row in item["rows"][1:] if row and normalize_text(row[0])]
            preview = summarize_row_anchor_preview(row_ids)
            if preview:
                table_name = item["path"].name
                row_anchor_preview_parts.append(f"{table_name}:{preview}")
        row_anchor_preview = " | ".join(row_anchor_preview_parts)
    return {
        "document_key": document["document_key"],
        "doi": document["doi"],
        "source_mode": document["source_mode"],
        "stage2_semantic_source_mode": normalize_text(document.get("stage2_semantic_source_mode")) or STAGE2_SEMANTIC_SOURCE_MODE,
        "summary_table_mode": table_mode(),
        "summary_first_column_enhancement": "yes" if enhancement_enabled else "no",
        "summary_table_selection_strategy": table_selection_strategy,
        "summary_first_column_row_labels_preview": row_anchor_preview,
        "formulation_count": len(document.get("formulation_candidates", [])),
        "component_count": len(document.get("component_candidates", [])),
        "variable_count": len(document.get("variable_candidates", [])),
        "measurement_count": len(document.get("measurement_candidates", [])),
        "relation_hint_count": len(document.get("relation_hints", [])),
        "evidence_span_count": len(document.get("evidence_spans", [])),
        "unassigned_observation_count": len(document.get("unassigned_observations", [])),
        "ph_variable_count": sum(1 for name in variable_names if name in PH_TOKENS),
        "doe_factor_count": sum(1 for name in variable_names if name in DOE_TOKENS),
        "doe_scope_declared": "yes"
        if any(
            normalize_text(item.get("scope_kind")) == DOE_SCOPE_KIND
            for item in document.get("semantic_scope_declarations", [])
            if isinstance(item, dict)
        )
        else "no",
        "pdi_measurement_present": "yes" if "pdi" in measurement_names else "no",
        "zeta_measurement_present": "yes" if "zeta_potential" in measurement_names or "zeta_mv" in measurement_names else "no",
        "multi_component_formulation_count": multi_component_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract minimal object-first Stage2 v2 semantic artifacts for a declared manifest scope."
    )
    parser.add_argument("--manifest-tsv", required=True, help="TSV manifest containing key/doi/title/text_path columns.")
    parser.add_argument("--out-dir", required=True, help="Output directory for Stage2 v2 artifacts.")
    parser.add_argument("--paper-key", action="append", dest="paper_keys", default=[], help="Repeatable paper key filter.")
    parser.add_argument(
        "--source-mode",
        choices=["legacy_llm_replay", "live_llm"],
        default="legacy_llm_replay",
        help="Use saved raw responses for replay/rehydration or call a live model.",
    )
    parser.add_argument(
        "--legacy-raw-responses-dir",
        default="",
        help="Directory containing saved raw responses for replay/rehydration mode.",
    )
    parser.add_argument("--model", default=PRIMARY_DEFAULT)
    parser.add_argument("--llm-backend", choices=["gemini", "nvidia"], default="gemini")
    parser.add_argument("--max-text-chars", type=int, default=30000)
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    parser.add_argument(
        "--stop-before-live-call",
        action="store_true",
        help="Materialize pre-LLM S2-2/S2-3 artifacts only and stop before any live or replay raw-response handling.",
    )
    return parser.parse_args()


def build_request_metadata_payload(
    *,
    paper_key: str,
    doi: str,
    source_mode: str,
    llm_backend: str,
    model: str,
    request_timeout_seconds: int,
    request_retries: int,
    retry_sleep_sec: float,
    manifest_path: Path,
    evidence_artifact_path: Path,
    raw_response_path: Path,
    prompt_text: str,
    authority_run_dir: str = "",
    authority_payload_root: str = "",
) -> dict[str, Any]:
    return {
        "stage_boundary": "Stage2 composite live call",
        "paper_key": paper_key,
        "doi": doi,
        "source_mode": source_mode,
        "llm_backend": llm_backend,
        "model": model,
        "request_timeout_seconds": request_timeout_seconds,
        "request_retries": request_retries,
        "retry_sleep_sec": retry_sleep_sec,
        "source_manifest_path": to_repo_rel(manifest_path),
        "source_evidence_artifact_path": to_repo_rel(evidence_artifact_path),
        "authority_run_dir": normalize_text(authority_run_dir),
        "authority_payload_root": normalize_text(authority_payload_root),
        "request_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "",
        "raw_response_path": to_repo_rel(raw_response_path),
        "raw_payload_persisted": False,
        "raw_payload_character_count": 0,
        "raw_payload_sha256": "",
        "prompt_character_count": len(prompt_text),
        "api_failure": "",
    }


def main() -> None:
    args = parse_args()
    validate_models_or_raise([args.model], context="stage2_objects_v2 extractor model check")
    request_timeout_seconds = max(1, min(int(args.request_timeout_seconds), 180))
    request_retries = max(0, min(int(args.request_retries), 1))

    manifest_path = Path(args.manifest_tsv)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = out_dir / REQUEST_METADATA_SUBDIR
    metadata_dir.mkdir(parents=True, exist_ok=True)

    records = read_tsv(manifest_path)
    selected_keys = [normalize_text(key) for key in args.paper_keys if normalize_text(key)]
    if selected_keys:
        records = [
            record
            for record in records
            if normalize_text(record.get("key") or record.get("paper_key")) in selected_keys
        ]
    if not records:
        raise ValueError("No manifest records selected for extraction.")

    legacy_raw_dir: Path | None = None
    if args.source_mode == "legacy_llm_replay":
        if not str(args.legacy_raw_responses_dir).strip():
            raise ValueError("--legacy-raw-responses-dir is required for legacy_llm_replay mode.")
        legacy_raw_dir = Path(args.legacy_raw_responses_dir)
        if not legacy_raw_dir.is_absolute():
            legacy_raw_dir = (PROJECT_ROOT / legacy_raw_dir).resolve()
        if not legacy_raw_dir.exists():
            raise FileNotFoundError(f"Legacy raw responses directory not found: {legacy_raw_dir}")

    jsonl_path = out_dir / OUTPUT_JSONL_NAME
    summary_path = out_dir / OUTPUT_SUMMARY_NAME
    prompt_preview_rows: list[dict[str, Any]] = []
    s2_2_boundary_validation_rows: list[dict[str, Any]] = []
    s2_3_boundary_validation_rows: list[dict[str, Any]] = []
    table_selection_debug_rows: list[dict[str, Any]] = []
    candidate_segmentation_debug_rows: list[dict[str, Any]] = []
    table_authority_validation_rows: list[dict[str, Any]] = []
    request_summary_rows: list[dict[str, Any]] = []
    prompt_preview_path = out_dir.parent / "analysis" / PROMPT_PREVIEW_NAME
    s2_2_boundary_validation_path = out_dir.parent / "analysis" / S2_2_BOUNDARY_VALIDATION_NAME
    s2_3_boundary_validation_path = out_dir.parent / "analysis" / S2_3_BOUNDARY_VALIDATION_NAME
    table_selection_debug_path = out_dir.parent / "analysis" / TABLE_SELECTION_DEBUG_NAME
    candidate_segmentation_debug_path = out_dir.parent / "analysis" / CANDIDATE_SEGMENTATION_DEBUG_NAME
    table_authority_validation_path = out_dir.parent / "analysis" / TABLE_AUTHORITY_VALIDATION_NAME
    request_summary_path = out_dir.parent / "analysis" / REQUEST_SUMMARY_NAME
    current_table_mode = table_mode()
    summary_enhanced = summary_first_column_enhancement_enabled()
    current_input_packing_mode = input_packing_mode()
    ordered_block_order = "metadata > synthesis_method > materials_procurement > table > paragraph" if ordered_input_packing_enabled() else "raw_prefix"
    summary_rows: list[dict[str, Any]] = []
    progress = Stage2ProgressReporter(total_tasks=len(records))
    print(
        f"stage2_progress total={progress.total_tasks} source_mode={args.source_mode} llm_backend={args.llm_backend}",
        flush=True,
    )
    progress.start()

    try:
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(records, start=1):
                key = normalize_text(record.get("key") or record.get("paper_key"))
                if not key:
                    continue
                if "text_path" not in record or not normalize_text(record.get("text_path")):
                    raise ValueError(f"Manifest row for {key} is missing text_path.")

                progress_label = progress.begin_task(index, key)
                try:
                    raw_copy_path = raw_dir / f"{key}__stage2_v2_raw_response.json"
                    request_metadata_path = metadata_dir / REQUEST_METADATA_FILENAME_TEMPLATE.format(paper_key=key)
                    text_path = Path(record["text_path"])
                    if not text_path.is_absolute():
                        text_path = (PROJECT_ROOT / text_path).resolve()
                    if not text_path.exists():
                        raise FileNotFoundError(f"Missing paper text for {key}: {text_path}")
                    table_dir = resolve_tables_dir_for_record(record, text_path, key)
                    candidate_artifact_path = candidate_blocks_path(out_dir, key)
                    candidate_artifact, segmentation_bundle = build_candidate_segmentation_artifact(
                        record=record,
                        manifest_path=manifest_path,
                        text_path=text_path,
                        table_dir=table_dir,
                        producer_script="src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py",
                    )
                    write_json(candidate_artifact_path, candidate_artifact)
                    candidate_segmentation_debug_rows.extend(
                        build_candidate_segmentation_debug_rows(candidate_artifact, candidate_artifact_path)
                    )
                    artifact_path = evidence_blocks_path(out_dir, key)
                    evidence_artifact, debug_payload = build_evidence_blocks_artifact(
                        record=record,
                        manifest_path=manifest_path,
                        text_path=text_path,
                        table_dir=table_dir,
                        max_chars=args.max_text_chars,
                        producer_script="src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py",
                        candidate_artifact_path=candidate_artifact_path,
                        segmentation_bundle=segmentation_bundle,
                    )
                    write_json(artifact_path, evidence_artifact)
                    normalized_payload_path = normalized_table_payloads_path(out_dir, key)
                    normalized_payload_artifact, table_authority_rows = build_normalized_table_payload_artifact(
                        record=record,
                        out_dir=out_dir,
                        producer_script="src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py",
                        evidence_artifact_path=artifact_path,
                        evidence_artifact=evidence_artifact,
                        segmentation_bundle=segmentation_bundle,
                    )
                    write_json(normalized_payload_path, normalized_payload_artifact)
                    table_authority_validation_rows.extend(table_authority_rows)
                    prompt = build_live_prompt(record, evidence_artifact)
                    prompt_preview_row = build_prompt_preview_row(
                        document={
                            "document_key": key,
                            "doi": record["doi"],
                            "source_mode": "live_llm_stage2_v2" if args.source_mode == "live_llm" else "legacy_llm_replay",
                        },
                        prompt_text=prompt,
                        table_mode_value=current_table_mode,
                        summary_enhanced=summary_enhanced,
                        input_packing_mode_value=current_input_packing_mode,
                        ordered_block_order=" > ".join(
                            str(value)
                            for value in ensure_list(evidence_artifact.get("input_contract", {}).get("ordered_block_order"))
                            if str(value).strip()
                        ),
                        evidence_artifact_path=to_repo_rel(artifact_path),
                        evidence_artifact=evidence_artifact,
                        technical_status_overall=normalize_text(evidence_artifact.get("technical_status", {}).get("overall")),
                        design_status_overall=normalize_text(evidence_artifact.get("design_status", {}).get("overall")),
                    )
                    prompt_preview_rows.append(prompt_preview_row)
                    s2_2_boundary_validation_rows.append(
                        build_s2_2_boundary_validation_row(
                            candidate_artifact=candidate_artifact,
                            evidence_artifact=evidence_artifact,
                            normalized_payload_artifact=normalized_payload_artifact,
                        )
                    )
                    s2_3_boundary_validation_rows.append(
                        build_s2_3_boundary_validation_row(
                            prompt_preview_row=prompt_preview_row,
                        )
                    )
                    if debug_payload is not None:
                        debug_payload["evidence_artifact_path"] = to_repo_rel(artifact_path)
                        table_selection_debug_rows.append(debug_payload)
                    if args.stop_before_live_call:
                        request_summary_rows.append(
                            {
                                "paper_key": key,
                                "doi": normalize_text(record.get("doi")),
                                "status": "skipped_pre_llm_boundary",
                                "llm_backend": "",
                                "model": "",
                                "request_timeout_seconds": "",
                                "request_retries": "",
                                "raw_response_path": "",
                                "raw_payload_persisted": "no",
                                "request_metadata_path": "",
                                "semantic_object_written": "no",
                                "failure_type": "",
                                "failure_message": "",
                            }
                        )
                        progress.complete_task()
                        continue
                    if args.source_mode == "legacy_llm_replay":
                        assert legacy_raw_dir is not None
                        legacy_raw_path = find_legacy_raw_response(legacy_raw_dir, key)
                        raw_copy_path = raw_dir / legacy_raw_path.name
                        shutil.copy2(legacy_raw_path, raw_copy_path)
                        document = convert_legacy_raw_response_to_v2(
                            record=record,
                            raw_response_path=raw_copy_path,
                            raw_response_text=raw_copy_path.read_text(encoding="utf-8", errors="replace"),
                            authority_metadata={
                                "authority_run_dir": normalize_text(evidence_artifact.get("authority_run_dir")),
                                "authority_payload_root": normalize_text(evidence_artifact.get("authority_payload_root")),
                            },
                        )
                        request_summary_rows.append(
                            {
                                "paper_key": key,
                                "doi": normalize_text(record.get("doi")),
                                "status": "success",
                                "llm_backend": "replay",
                                "model": "",
                                "request_timeout_seconds": "",
                                "request_retries": "",
                                "raw_response_path": to_repo_rel(raw_copy_path),
                                "raw_payload_persisted": "yes",
                                "request_metadata_path": "",
                                "semantic_object_written": "yes",
                                "failure_type": "",
                                "failure_message": "",
                            }
                        )
                    else:
                        metadata_payload = build_request_metadata_payload(
                            paper_key=key,
                            doi=normalize_text(record.get("doi")),
                            source_mode=args.source_mode,
                            llm_backend=args.llm_backend,
                            model=args.model,
                            request_timeout_seconds=request_timeout_seconds,
                            request_retries=request_retries,
                            retry_sleep_sec=args.retry_sleep_sec,
                            manifest_path=manifest_path,
                            evidence_artifact_path=artifact_path,
                            raw_response_path=raw_copy_path,
                            prompt_text=prompt,
                            authority_run_dir=normalize_text(evidence_artifact.get("authority_run_dir")),
                            authority_payload_root=normalize_text(evidence_artifact.get("authority_payload_root")),
                        )
                        request_metadata_path.write_text(
                            json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        if args.llm_backend == "gemini":
                            raw_text = call_gemini(
                                args.model,
                                prompt,
                                request_retries,
                                args.retry_sleep_sec,
                                progress_label=progress_label,
                                timeout_seconds=request_timeout_seconds,
                            )
                        else:
                            raw_text = call_nvidia_hosted(
                                args.model,
                                prompt,
                                request_retries,
                                args.retry_sleep_sec,
                                progress_label=progress_label,
                            )
                        raw_copy_path.write_text(raw_text, encoding="utf-8")
                        metadata_payload["raw_payload_persisted"] = bool(raw_text and raw_copy_path.exists())
                        metadata_payload["raw_payload_character_count"] = len(raw_text)
                        metadata_payload["raw_payload_sha256"] = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else ""
                        sanitized_text, sanitization_audit = sanitize_stage2_json_text(raw_text)
                        if json_sanitization_applied(sanitization_audit):
                            write_json_sanitization_audit(raw_copy_path, sanitization_audit)
                            print(
                                f"{progress_label} sanitized_json parse_stage={sanitization_audit.get('parse_stage')} "
                                f"balanced_close={sanitization_audit.get('balanced_close_applied')} "
                                f"cut={sanitization_audit.get('balanced_close_cut')}",
                                flush=True,
                            )
                        parsed = json.loads(sanitized_text)
                        document = normalize_live_document(
                            record,
                            parsed,
                            raw_copy_path,
                            authority_metadata={
                                "authority_run_dir": normalize_text(evidence_artifact.get("authority_run_dir")),
                                "authority_payload_root": normalize_text(evidence_artifact.get("authority_payload_root")),
                            },
                        )
                        metadata_payload["status"] = "success"
                        metadata_payload["request_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                        request_metadata_path.write_text(
                            json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        request_summary_rows.append(
                            {
                                "paper_key": key,
                                "doi": normalize_text(record.get("doi")),
                                "status": "success",
                                "llm_backend": args.llm_backend,
                                "model": args.model,
                                "request_timeout_seconds": str(request_timeout_seconds),
                                "request_retries": str(request_retries),
                                "raw_response_path": to_repo_rel(raw_copy_path),
                                "raw_payload_persisted": "yes" if metadata_payload["raw_payload_persisted"] else "no",
                                "request_metadata_path": to_repo_rel(request_metadata_path),
                                "semantic_object_written": "yes",
                                "failure_type": "",
                                "failure_message": "",
                            }
                        )

                    handle.write(json.dumps(document, ensure_ascii=False) + "\n")
                    summary_rows.append(summary_row(document))
                    progress.complete_task()
                except Exception as exc:
                    progress.fail_task(exc)
                    if args.source_mode == "live_llm":
                        failure_type = type(exc).__name__
                        failure_message = str(exc)
                        raw_payload_text = raw_copy_path.read_text(encoding="utf-8", errors="replace") if raw_copy_path.exists() else ""
                        metadata_payload = {}
                        if request_metadata_path.exists():
                            try:
                                metadata_payload = json.loads(request_metadata_path.read_text(encoding="utf-8"))
                            except Exception:
                                metadata_payload = {}
                        metadata_payload.update(
                            {
                                "paper_key": key,
                                "doi": normalize_text(record.get("doi")),
                                "source_mode": args.source_mode,
                                "llm_backend": args.llm_backend,
                                "model": args.model,
                                "request_timeout_seconds": request_timeout_seconds,
                                "request_retries": request_retries,
                                "retry_sleep_sec": args.retry_sleep_sec,
                                "request_finished_at_utc": datetime.now(timezone.utc).isoformat(),
                                "status": "request_failure",
                                "raw_response_path": to_repo_rel(raw_copy_path),
                                "raw_payload_persisted": raw_copy_path.exists(),
                                "raw_payload_character_count": len(raw_payload_text),
                                "raw_payload_sha256": hashlib.sha256(raw_payload_text.encode("utf-8")).hexdigest() if raw_payload_text else "",
                                "api_failure": {
                                    "error_type": failure_type,
                                    "error_message": failure_message,
                                },
                            }
                        )
                        request_metadata_path.write_text(
                            json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        request_summary_rows.append(
                            {
                                "paper_key": key,
                                "doi": normalize_text(record.get("doi")),
                                "status": "request_failure",
                                "llm_backend": args.llm_backend,
                                "model": args.model,
                                "request_timeout_seconds": str(request_timeout_seconds),
                                "request_retries": str(request_retries),
                                "raw_response_path": to_repo_rel(raw_copy_path) if raw_copy_path.exists() else "",
                                "raw_payload_persisted": "yes" if raw_copy_path.exists() else "no",
                                "request_metadata_path": to_repo_rel(request_metadata_path),
                                "semantic_object_written": "no",
                                "failure_type": failure_type,
                                "failure_message": failure_message,
                            }
                        )
                    continue
    finally:
        progress.stop()
        progress.finish()

    write_tsv(
        summary_path,
        summary_rows,
        [
            "document_key",
            "doi",
            "source_mode",
            "stage2_semantic_source_mode",
            "formulation_count",
            "component_count",
            "variable_count",
            "measurement_count",
            "relation_hint_count",
            "evidence_span_count",
            "unassigned_observation_count",
            "ph_variable_count",
            "doe_factor_count",
            "doe_scope_declared",
            "pdi_measurement_present",
            "zeta_measurement_present",
            "multi_component_formulation_count",
        ],
    )
    if prompt_preview_rows:
        write_tsv(
            prompt_preview_path,
            prompt_preview_rows,
            [
                "document_key",
                "doi",
                "source_mode",
                "table_mode",
                "summary_first_column_enhancement",
                "table_mode_used",
                "summary_applied",
                "reason_for_full_table",
                "input_packing_mode",
                "prompt_layout_class",
                "paper_text_marker_index",
                "evidence_pack_marker_index",
                "table_excerpts_marker_index",
                "ordered_block_order",
                "evidence_artifact_path",
                "technical_status_overall",
                "design_status_overall",
                "s2_3_ready_overall",
                "s2_3_readiness_reasons",
                "selected_evidence_block_count",
                "all_selected_blocks_included",
                "uses_evidence_pack_only",
                "truncation_detected",
                "exact_duplicate_block_count",
                "prompt_size_policy_status",
                "prompt_length",
                "live_prompt_header_mode",
                "runtime_metadata_removed_from_live_prompt",
                "live_prompt_contains_runtime_metadata",
                "prompt_head_preview",
                "prompt_tail_preview",
            ],
        )
    if s2_2_boundary_validation_rows:
        write_tsv(
            s2_2_boundary_validation_path,
            s2_2_boundary_validation_rows,
            [
                "paper_key",
                "candidate_count",
                "evidence_block_count",
                "table_block_count",
                "selected_table_payload_count",
                "selector_strategy",
                "evidence_priority_selection_executed",
                "explicit_table_fallback_used",
                "selector_truth_mismatch",
                "design_status_overall",
                "suppression_event_count",
                "noise_detected",
                "status",
            ],
        )
    if s2_3_boundary_validation_rows:
        write_tsv(
            s2_3_boundary_validation_path,
            s2_3_boundary_validation_rows,
            [
                "paper_key",
                "s2_3_ready_overall",
                "uses_evidence_pack_only",
                "all_selected_blocks_included",
                "selected_evidence_block_count",
                "truncation_detected",
                "exact_duplicate_block_count",
                "prompt_size_policy_status",
                "prompt_layout_class",
                "status",
            ],
        )
    if table_selection_debug_rows:
        table_selection_debug_path.write_text(
            json.dumps(table_selection_debug_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if candidate_segmentation_debug_rows:
        write_tsv(
            candidate_segmentation_debug_path,
            candidate_segmentation_debug_rows,
            [
                "document_key",
                "candidate_artifact_path",
                "segmentation_profile",
                "candidate_id",
                "candidate_type",
                "source_type",
                "origin_locator",
                "section_label",
                "section_kind",
                "segmentation_method",
                "split_trigger",
                "noise_flags",
                "quality_flags",
                "representation_status",
                "repair_primary_source",
                "repair_actions",
                "material_difference_from_raw",
                "selector_readiness_label",
                "authority_rank",
                "authority_score",
                "authority_tier",
                "preserved_by_authority_ranking",
                "unresolved_reason",
                "raw_table_preview",
                "repaired_table_preview",
                "text_preview",
            ],
        )
    if table_authority_validation_rows:
        write_tsv(
            table_authority_validation_path,
            table_authority_validation_rows,
            [
                "paper_key",
                "table_id",
                "source_table_reference",
                "table_type",
                "row_count_before",
                "row_count_after",
                "missing_rows_detected",
                "duplicate_row_count",
                "column_count_before",
                "column_count_after",
                "column_collapse_detected",
                "selector_readiness_label",
                "authority_rank",
                "authority_score",
                "authority_tier",
                "normalization_actions",
            ],
        )
    if request_summary_rows:
        write_tsv(
            request_summary_path,
            request_summary_rows,
            [
                "paper_key",
                "doi",
                "status",
                "llm_backend",
                "model",
                "request_timeout_seconds",
                "request_retries",
                "raw_response_path",
                "raw_payload_persisted",
                "request_metadata_path",
                "semantic_object_written",
                "failure_type",
                "failure_message",
            ],
        )
    print(f"wrote_jsonl={jsonl_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_candidate_blocks_dir={out_dir / CANDIDATE_BLOCKS_SUBDIR}")
    print(f"wrote_evidence_blocks_dir={out_dir / EVIDENCE_BLOCKS_SUBDIR}")
    print(f"wrote_normalized_table_payloads_dir={out_dir / NORMALIZED_TABLE_PAYLOADS_SUBDIR}")
    if prompt_preview_rows:
        print(f"wrote_prompt_preview={prompt_preview_path}")
    if table_selection_debug_rows:
        print(f"wrote_table_selection_debug={table_selection_debug_path}")
    if candidate_segmentation_debug_rows:
        print(f"wrote_candidate_segmentation_debug={candidate_segmentation_debug_path}")
    print(f"wrote_raw_responses_dir={raw_dir}")
    print(f"wrote_request_metadata_dir={metadata_dir}")
    if request_summary_rows:
        success_count = sum(1 for row in request_summary_rows if row.get("status") == "success")
        failure_count = sum(1 for row in request_summary_rows if row.get("status") != "success")
        print(f"wrote_request_summary={request_summary_path}")
        print(f"success_count={success_count}")
        print(f"failure_count={failure_count}")


if __name__ == "__main__":
    main()
